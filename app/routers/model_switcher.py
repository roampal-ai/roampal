"""
Runtime Model Switching API
Allows hot-swapping of models without restart
"""
import os
import re
import subprocess
import logging
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model-switcher"])

# Global state for concurrency control
_download_lock = asyncio.Lock()
_downloading_models: set = set()
_env_file_lock = asyncio.Lock()

class ModelSwitchRequest(BaseModel):
    model_name: str

def _validate_model_name(model_name: str) -> bool:
    """
    Validate model name format to prevent command injection.
    Valid format: name:tag (e.g., 'qwen3:8b', 'llama2:13b-chat')
    """
    if not model_name or len(model_name) > 100:
        return False
    # Allow alphanumeric, underscore, hyphen, dot, colon
    pattern = r'^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$'
    return bool(re.match(pattern, model_name))

def _format_size(bytes_size: int) -> str:
    """Format byte size in human-readable format"""
    if bytes_size == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

@router.get("/available")
async def list_available_models():
    """List all locally available models using Ollama's JSON API"""
    try:
        # Use Ollama's JSON API for structured output (more reliable than parsing text)
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get('http://localhost:11434/api/tags')

            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('models', []):
                    models.append({
                        "name": model.get('name', ''),
                        "id": model.get('digest', '')[:12],  # Short hash like git
                        "size": _format_size(model.get('size', 0)),
                        "modified": model.get('modified_at', '')
                    })
                return {"models": models, "count": len(models)}
            else:
                logger.warning(f"Ollama API returned status {response.status_code}, falling back to subprocess")
                raise Exception(f"API returned {response.status_code}")

    except Exception as api_error:
        logger.warning(f"Failed to use Ollama JSON API: {api_error}, falling back to subprocess")

        # Fallback to subprocess parsing if API fails
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )

            # Parse the text output (legacy fallback)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        models.append({
                            "name": parts[0],
                            "id": parts[1],
                            "size": parts[2] + " " + parts[3],
                            "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
                        })

            return {"models": models, "count": len(models)}
        except subprocess.CalledProcessError as e:
            return {"models": [], "error": str(e)}
        except Exception as e:
            return {"models": [], "error": str(e)}

@router.get("/current")
async def get_current_model(request: Request):
    """Get the currently active model"""
    # ALWAYS prioritize runtime state over env vars (runtime is source of truth)
    current_model = None

    if hasattr(request.app.state, 'llm_client'):
        if hasattr(request.app.state.llm_client, 'model_name'):
            current_model = request.app.state.llm_client.model_name
            logger.info(f"Current model from runtime: {current_model}")

    # Fallback to env var only if no runtime client
    if not current_model:
        current_model = os.getenv('ROAMPAL_LLM_OLLAMA_MODEL') or os.getenv('OLLAMA_MODEL') or 'codellama'
        logger.info(f"Current model from env fallback: {current_model}")

    # CRITICAL: Verify model actually exists in Ollama before saying it's active
    # This prevents false positives when runtime is None but env var is set
    model_exists = False
    if current_model:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                available_models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                model_exists = current_model in available_models
                if not model_exists:
                    logger.warning(f"Model {current_model} not found in Ollama - treating as unavailable")
                    current_model = None
        except Exception as e:
            logger.error(f"Failed to verify model existence: {e}")

    # Check if current model is an embedding model or None
    embedding_models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']
    is_embedding = current_model and any(embed in current_model.lower() for embed in embedding_models)
    can_chat = current_model and not is_embedding and model_exists

    return {
        "current_model": current_model,
        "can_chat": can_chat,
        "is_embedding_model": is_embedding
    }

@router.post("/switch")
async def switch_model(request: Request, model_request: ModelSwitchRequest):
    """Switch to a different model at runtime"""
    try:
        model_name = model_request.model_name

        # First verify the model exists
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                available_models = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            available_models.append(parts[0])

                # Check if requested model exists
                if model_name not in available_models:
                    logger.warning(f"Model '{model_name}' not found in Ollama")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
                    )
        except subprocess.TimeoutExpired:
            logger.warning("Could not verify model availability - timeout")
        except FileNotFoundError:
            logger.warning("Ollama command not found")

        # Update the LLM client if it exists
        if hasattr(request.app.state, 'llm_client'):
            # Lazy initialization: create LLM client if None (happens after uninstalling all models)
            if request.app.state.llm_client is None:
                from modules.llm.ollama_client import OllamaClient
                request.app.state.llm_client = OllamaClient()
                await request.app.state.llm_client.initialize({"ollama_model": model_name})
                logger.info(f"Lazily initialized LLM client with model: {model_name}")
                previous_model = None
            else:
                # Store previous model for rollback if health check fails
                previous_model = request.app.state.llm_client.model_name
                request.app.state.llm_client.model_name = model_name

            # Update global agent_service LLM reference if it exists
            from app.routers.agent_chat import agent_service
            if agent_service and hasattr(agent_service, 'llm'):
                agent_service.llm = request.app.state.llm_client

            logger.info(f"Switched model to: {model_name}")

            # Verify model health with test inference (explicit 10s timeout)
            try:
                logger.info(f"Performing health check on model: {model_name}")

                # Wrap with explicit timeout as documented (10 seconds)
                test_response = await asyncio.wait_for(
                    request.app.state.llm_client.generate_response(prompt="Hi"),
                    timeout=10.0
                )

                logger.info(f"Model health check passed for: {model_name}")
            except asyncio.TimeoutError:
                logger.error(f"Model health check timed out for {model_name} after 10 seconds")
                # Rollback to previous model
                request.app.state.llm_client.model_name = previous_model
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_name} health check timed out after 10s. Rolled back to {previous_model}"
                )
            except Exception as health_error:
                logger.error(f"Model health check failed for {model_name}: {health_error}")
                # Rollback to previous model
                request.app.state.llm_client.model_name = previous_model
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_name} failed health check: {str(health_error)}. Rolled back to {previous_model}"
                )

            # Also update environment variable for new sessions
            os.environ['OLLAMA_MODEL'] = model_name
            os.environ['ROAMPAL_LLM_OLLAMA_MODEL'] = model_name  # Update both

            # Update .env file for persistence with file locking
            async with _env_file_lock:
                env_path = Path(__file__).parent.parent.parent / '.env'
                if env_path.exists():
                    lines = env_path.read_text().splitlines()
                    updated = False
                    for i, line in enumerate(lines):
                        if line.startswith('OLLAMA_MODEL='):
                            lines[i] = f'OLLAMA_MODEL={model_name}'
                            updated = True
                        elif line.startswith('ROAMPAL_LLM_OLLAMA_MODEL='):
                            lines[i] = f'ROAMPAL_LLM_OLLAMA_MODEL={model_name}'
                            updated = True

                    # If ROAMPAL_LLM_OLLAMA_MODEL doesn't exist, add it
                    if not any(line.startswith('ROAMPAL_LLM_OLLAMA_MODEL=') for line in lines):
                        # Find where to insert it (after OLLAMA_MODEL line)
                        for i, line in enumerate(lines):
                            if line.startswith('OLLAMA_MODEL='):
                                lines.insert(i + 1, f'ROAMPAL_LLM_OLLAMA_MODEL={model_name}')
                                break

                    env_path.write_text('\n'.join(lines) + '\n')

            return {
                "status": "success",
                "message": f"Switched to model: {model_name}",
                "current_model": model_name
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="LLM client not initialized"
            )
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper error messages)
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_type = type(e).__name__
        error_message = str(e) if str(e) else f"{error_type} (no message)"
        logger.error(f"Error switching model ({error_type}): {error_message}")
        logger.error(f"Traceback: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch model: {error_message}"
        )

@router.post("/pull")
async def pull_model(request_body: Dict[str, Any] = Body(...)):
    """Download a new model from Ollama registry"""
    try:
        # Extract model name from request body (UI sends 'model', accept both)
        model_name = request_body.get('model') or request_body.get('model_name', '')
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Model name is required"
            )

        # Validate model name format
        if not _validate_model_name(model_name):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name format. Expected format: name:tag (e.g., 'qwen3:8b')"
            )

        # Check if already downloading
        if model_name in _downloading_models:
            raise HTTPException(
                status_code=409,
                detail=f"Model {model_name} is already being downloaded"
            )

        # Acquire download lock
        async with _download_lock:
            _downloading_models.add(model_name)

        try:
            logger.info(f"Pulling model: {model_name}")
            # Start the pull process
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large models
            )

            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Successfully pulled {model_name}",
                    "output": result.stdout
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to pull model: {result.stderr}"
                )
        finally:
            # Always remove from downloading set
            _downloading_models.discard(model_name)

    except subprocess.TimeoutExpired:
        _downloading_models.discard(model_name)
        raise HTTPException(
            status_code=504,
            detail="Model pull timed out after 10 minutes"
        )
    except HTTPException:
        raise
    except Exception as e:
        _downloading_models.discard(model_name)
        raise HTTPException(
            status_code=500,
            detail=f"Error pulling model: {str(e)}"
        )

@router.post("/pull-stream")
async def pull_model_stream(request_body: Dict[str, Any] = Body(...)):
    """Download a model with real-time progress streaming"""
    model_name = request_body.get('model') or request_body.get('model_name', '')
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    # Validate model name format
    if not _validate_model_name(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name format. Expected format: name:tag (e.g., 'qwen3:8b')"
        )

    # Acquire download lock and check if already downloading (prevents race condition)
    async with _download_lock:
        if model_name in _downloading_models:
            raise HTTPException(
                status_code=409,
                detail=f"Model {model_name} is already being downloaded"
            )
        _downloading_models.add(model_name)

    async def generate_progress() -> AsyncGenerator[str, None]:
        """Generate SSE events with download progress using Ollama HTTP API"""
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': model_name})}\n\n"

            # Use Ollama's HTTP API endpoint for pulling models
            # Configure granular timeouts: 60s between data chunks prevents infinite hangs
            async with httpx.AsyncClient(timeout=httpx.Timeout(
                connect=5.0,   # 5s to establish connection
                read=60.0,     # 60s between receiving data chunks (prevents hang)
                write=5.0,     # 5s to send request
                pool=5.0       # 5s to get connection from pool
            )) as client:
                async with client.stream(
                    'POST',
                    'http://localhost:11434/api/pull',
                    json={'name': model_name, 'stream': True}
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"Ollama API error: {error_text.decode('utf-8', errors='replace')}")

                    # Process NDJSON stream (one JSON object per line)
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)
                            status = data.get('status', '')

                            logger.info(f"Ollama API: {status}")

                            # Check for errors first
                            if 'error' in data:
                                error_msg = data.get('error', 'Unknown error')
                                logger.error(f"Ollama API error: {error_msg}")
                                error_data = {'type': 'error', 'message': f'Download failed: {error_msg}'}
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return  # Exit generator, don't send false success

                            # Map Ollama API responses to SSE events
                            if 'total' in data and 'completed' in data:
                                # Download progress
                                total = data['total']
                                completed = data['completed']
                                percent = int((completed / total * 100)) if total > 0 else 0

                                # Format sizes
                                def format_bytes(b):
                                    for unit in ['B', 'KB', 'MB', 'GB']:
                                        if b < 1024:
                                            return f"{b:.1f} {unit}"
                                        b /= 1024
                                    return f"{b:.1f} TB"

                                downloaded_str = format_bytes(completed)
                                total_str = format_bytes(total)

                                progress_data = {
                                    'type': 'progress',
                                    'percent': percent,
                                    'downloaded': downloaded_str,
                                    'total': total_str,
                                    'speed': 'downloading',
                                    'message': f'{status}: {percent}%'
                                }
                                yield f"data: {json.dumps(progress_data)}\n\n"

                            elif status.startswith('pulling'):
                                # Layer pull in progress
                                yield f"data: {json.dumps({'type': 'progress', 'message': status})}\n\n"

                            elif status == 'verifying sha256 digest':
                                verify_data = {'type': 'progress', 'percent': 100, 'message': 'Verifying model...'}
                                yield f"data: {json.dumps(verify_data)}\n\n"

                            elif status == 'success':
                                # Verify model actually installed before claiming success
                                try:
                                    async with httpx.AsyncClient(timeout=5.0) as verify_client:
                                        verify_response = await verify_client.get('http://localhost:11434/api/tags')
                                        tags_data = verify_response.json()
                                        model_exists = any(m['name'] == model_name for m in tags_data.get('models', []))

                                        if model_exists:
                                            logger.info(f"Model {model_name} downloaded successfully and verified")
                                            complete_data = {'type': 'complete', 'success': True, 'message': f'Successfully installed {model_name}'}
                                            yield f"data: {json.dumps(complete_data)}\n\n"
                                        else:
                                            logger.error(f"Model {model_name} reported success but not found in Ollama")
                                            error_data = {'type': 'error', 'message': f'Download reported success but {model_name} not found in Ollama'}
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                except Exception as verify_error:
                                    logger.error(f"Verification failed: {verify_error}")
                                    error_data = {'type': 'error', 'message': f'Verification failed: {str(verify_error)}'}
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Ollama API response: {line[:100]}")
                            continue

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error streaming model pull: {e}\n{error_details}")
            exception_data = {'type': 'error', 'message': f'Error: {str(e)}'}
            yield f"data: {json.dumps(exception_data)}\n\n"
        finally:
            # Always remove from downloading set
            _downloading_models.discard(model_name)

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@router.delete("/uninstall/{model_name:path}")
async def uninstall_model(model_name: str, request: Request):
    """Uninstall/remove a model from the local Ollama installation"""
    try:
        # Decode the URL-encoded model name
        import urllib.parse
        model_name = urllib.parse.unquote(model_name)

        # Validate model name format to prevent command injection
        if not _validate_model_name(model_name):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name format. Expected format: name:tag (e.g., 'qwen3:8b')"
            )

        logger.info(f"Attempting to uninstall model: {model_name}")

        # Use ollama rm command to remove the model
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.info(f"Successfully uninstalled model: {model_name}")

            # Check if this was the current model and switch to another if needed
            if hasattr(request.app.state, 'llm_client'):
                current_model = getattr(request.app.state.llm_client, 'model_name', '')
                if current_model == model_name:
                    # Get list of available models
                    list_result = subprocess.run(
                        ["ollama", "list"],
                        capture_output=True,
                        text=True
                    )
                    if list_result.returncode == 0:
                        lines = list_result.stdout.strip().split('\n')
                        if len(lines) > 1:  # Skip header
                            # Filter for chat models only (exclude embedding models)
                            chat_models = []
                            embedding_models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']

                            for line in lines[1:]:  # Skip header
                                parts = line.split()
                                if parts:
                                    model_name_candidate = parts[0]
                                    # Exclude embedding models
                                    is_embedding = any(embed in model_name_candidate.lower() for embed in embedding_models)
                                    if not is_embedding:
                                        chat_models.append(model_name_candidate)

                            if chat_models:
                                # Switch to first available chat model
                                new_model = chat_models[0]
                                request.app.state.llm_client.model_name = new_model
                                os.environ['OLLAMA_MODEL'] = new_model
                                os.environ['ROAMPAL_LLM_OLLAMA_MODEL'] = new_model

                                # Update .env file with locking
                                async with _env_file_lock:
                                    env_path = Path(__file__).parent.parent.parent / '.env'
                                    if env_path.exists():
                                        lines = env_path.read_text().splitlines()
                                        for i, line in enumerate(lines):
                                            if line.startswith('OLLAMA_MODEL='):
                                                lines[i] = f'OLLAMA_MODEL={new_model}'
                                            elif line.startswith('ROAMPAL_LLM_OLLAMA_MODEL='):
                                                lines[i] = f'ROAMPAL_LLM_OLLAMA_MODEL={new_model}'
                                        env_path.write_text('\n'.join(lines) + '\n')

                                logger.info(f"Switched from uninstalled model to: {new_model}")
                            else:
                                # No chat models available - warn user
                                logger.warning("No chat models available after uninstall. User must install a chat model.")
                                # Set to None to prevent embedding model usage
                                request.app.state.llm_client.model_name = None

            return {
                "success": True,
                "message": f"Successfully uninstalled {model_name}",
                "output": result.stdout
            }
        else:
            logger.error(f"Failed to uninstall model: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr or "Unknown error occurred",
                "message": f"Failed to uninstall {model_name}"
            }
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while uninstalling model: {model_name}")
        return {
            "success": False,
            "error": "Operation timed out after 30 seconds",
            "message": f"Timeout uninstalling {model_name}"
        }
    except FileNotFoundError:
        logger.error("Ollama command not found")
        return {
            "success": False,
            "error": "Ollama is not installed or not in PATH",
            "message": "Ollama not found"
        }
    except Exception as e:
        logger.error(f"Error uninstalling model {model_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error uninstalling {model_name}"
        }
# WebSocket endpoint for model installation (Tauri production)

@router.websocket("/pull-ws")
async def pull_model_websocket(websocket: WebSocket):
    """WebSocket endpoint for model downloads in Tauri production"""
    # Accept connection without origin validation (localhost only)
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"WebSocket accept failed: {e}")
        return
    
    try:
        # Receive model name from client
        data = await websocket.receive_text()
        request_data = json.loads(data)
        model_name = request_data.get('model', '')
        
        if not model_name:
            await websocket.send_json({"type": "error", "message": "Model name is required"})
            await websocket.close()
            return
        
        # Validate model name
        if not _validate_model_name(model_name):
            await websocket.send_json({"type": "error", "message": "Invalid model name format"})
            await websocket.close()
            return
        
        # Check if already downloading
        async with _download_lock:
            if model_name in _downloading_models:
                await websocket.send_json({"type": "error", "message": f"Model {model_name} is already being downloaded"})
                await websocket.close()
                return
            _downloading_models.add(model_name)
        
        try:
            logger.info(f"WebSocket: Pulling model: {model_name}")
            await websocket.send_json({"type": "start", "model": model_name})
            
            # Use Ollama HTTP API for streaming
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    "http://localhost:11434/api/pull",
                    json={"name": model_name, "stream": True},
                    timeout=600.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                progress_data = json.loads(line)
                                status = progress_data.get("status", "")
                                
                                if "error" in progress_data:
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": progress_data["error"]
                                    })
                                    break
                                
                                # Parse progress info
                                completed = progress_data.get("completed", 0)
                                total = progress_data.get("total", 0)
                                
                                if total > 0:
                                    percent = int((completed / total) * 100)
                                    downloaded_mb = f"{completed / (1024*1024):.1f} MB"
                                    total_mb = f"{total / (1024*1024):.1f} MB"
                                    
                                    await websocket.send_json({
                                        "type": "progress",
                                        "percent": percent,
                                        "downloaded": downloaded_mb,
                                        "total": total_mb,
                                        "message": f"{status}: {percent}%"
                                    })
                                else:
                                    await websocket.send_json({
                                        "type": "progress",
                                        "message": status
                                    })
                                
                                # Check if done
                                if status == "success" or progress_data.get("status") == "success":
                                    await websocket.send_json({"type": "complete", "model": model_name})
                                    logger.info(f"WebSocket: Successfully pulled {model_name}")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
            
        finally:
            _downloading_models.discard(model_name)
        
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket: Client disconnected during download of {model_name}")
        _downloading_models.discard(model_name)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        _downloading_models.discard(model_name)
