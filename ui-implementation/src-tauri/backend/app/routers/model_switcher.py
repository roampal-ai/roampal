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
import time
import sys
from pathlib import Path

# Windows-specific: Hide terminal windows when spawning subprocesses
_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW
from typing import List, Dict, Any, AsyncGenerator, Optional
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Body,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from app.routers.model_registry import QUANTIZATION_OPTIONS, HUGGINGFACE_REPOS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model-switcher"])

# Global state for concurrency control
_download_lock = asyncio.Lock()
_switch_lock = asyncio.Lock()  # v0.3.0: Prevent concurrent model switches
_downloading_models: set = set()
_cancel_flags: Dict[str, bool] = {}  # Track which downloads should be cancelled
_env_file_lock = asyncio.Lock()

# Multi-provider configuration
PROVIDERS = {
    "ollama": {
        "port": 11434,
        "health": "/api/tags",
        "models": "/api/tags",
        "api_style": "ollama",
    },
    "lmstudio": {
        "port": 1234,
        "health": "/v1/models",
        "models": "/v1/models",
        "api_style": "openai",
    },
}


def resolve_model_for_lmstudio(model_name: str) -> Dict[str, Any]:
    """
    Resolve a model name (possibly with quantization suffix) to HuggingFace download info.

    Handles:
    - Base names: "qwen2.5:7b" -> default Q4_K_M
    - Ollama tags: "qwen2.5:7b-instruct-q8_0" -> Q8_0 quantization
    - Direct matches in MODEL_TO_HUGGINGFACE

    Returns: {"repo": "...", "file": "...", "size_gb": ...} or None
    """
    # First check direct match in legacy mapping
    if model_name in MODEL_TO_HUGGINGFACE:
        return MODEL_TO_HUGGINGFACE[model_name]

    # Try to parse quantization from model name
    # Patterns: "model:size-instruct-q4_K_M" or "model:size-q4_K_M"
    for base_model, quants in QUANTIZATION_OPTIONS.items():
        for quant_level, quant_info in quants.items():
            ollama_tag = quant_info.get("ollama_tag", "")
            if ollama_tag and model_name.lower() == ollama_tag.lower():
                # Found matching quantization
                repo = HUGGINGFACE_REPOS.get(base_model)
                if repo and "file" in quant_info:
                    return {
                        "repo": repo,
                        "file": quant_info["file"],
                        "size_gb": quant_info.get("size_gb", 0),
                    }

    # Try fuzzy match - extract base model and check
    # e.g., "qwen2.5:7b-instruct-q8_0" -> base="qwen2.5:7b"
    for base_model in QUANTIZATION_OPTIONS.keys():
        if model_name.startswith(base_model):
            # Found base model, now find the quant
            suffix = model_name[len(base_model) :].lower()
            for quant_level, quant_info in QUANTIZATION_OPTIONS[base_model].items():
                if quant_level.lower() in suffix:
                    repo = HUGGINGFACE_REPOS.get(base_model)
                    if repo and "file" in quant_info:
                        return {
                            "repo": repo,
                            "file": quant_info["file"],
                            "size_gb": quant_info.get("size_gb", 0),
                        }
            # No specific quant found, use default
            if base_model in MODEL_TO_HUGGINGFACE:
                return MODEL_TO_HUGGINGFACE[base_model]

    return None


# GGUF model mapping for LM Studio downloads (legacy - now uses QUANTIZATION_OPTIONS)
MODEL_TO_HUGGINGFACE = {
    "qwen2.5:7b": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.68,
    },
    "llama3.2:3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
    },
    "llama3.1:8b": {
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
    },
    "qwen2.5:14b": {
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 8.99,
    },
    "mixtral:8x7b": {
        "repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "size_gb": 26.44,
    },
    "llama3.3:70b": {
        "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "size_gb": 42.5,
    },
    "qwen2.5:72b": {
        "repo": "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "file": "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "size_gb": 47.4,
    },
    "qwen2.5:32b": {
        "repo": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "size_gb": 19.9,
    },
    "qwen2.5:3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.93,
    },
    "qwen3:32b": {
        "repo": "bartowski/Qwen_Qwen3-32B-GGUF",
        "file": "Qwen_Qwen3-32B-Q4_K_M.gguf",
        "size_gb": 19.8,
    },
    "qwen3:8b": {
        "repo": "bartowski/Qwen_Qwen3-8B-GGUF",
        "file": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "size_gb": 5.0,
    },
    "gemma4:26b": {
        "repo": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
        "file": "google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "size_gb": 17.04,
    },
    "gemma4": {
        "repo": "bartowski/google_gemma-4-E4B-it-GGUF",
        "file": "google_gemma-4-E4B-it-Q4_K_M.gguf",
        "size_gb": 5.41,
    },
    "gpt-oss:20b": {
        "repo": "bartowski/openai_gpt-oss-20b-GGUF",
        "file": "openai_gpt-oss-20b-Q4_K_M.gguf",
        "size_gb": 11.67,
    },
}


async def detect_provider(name: str, config: dict) -> Optional[dict]:
    """Detect if provider is running"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            url = f"http://localhost:{config['port']}{config['health']}"
            response = await client.get(url)
            if response.status_code == 200:
                return {
                    "name": name,
                    "port": config["port"],
                    "url": f"http://localhost:{config['port']}",
                    "api_style": config["api_style"],
                    "status": "running",
                }
    except Exception as e:
        logger.debug(f"Provider {name} not detected on port {config['port']}: {e}")
    return None


async def get_provider_models(name: str, config: dict) -> List[str]:
    """Get models from provider by querying their API"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"http://localhost:{config['port']}{config['models']}"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            if config["api_style"] == "ollama":
                # Ollama format: {"models": [{"name": "..."}, ...]}
                # Filter out embedding models
                embedding_models = [
                    "nomic-embed",
                    "mxbai-embed",
                    "all-minilm",
                    "bge-",
                    "text-embedding",
                ]
                return [
                    m["name"]
                    for m in data.get("models", [])
                    if not any(embed in m["name"].lower() for embed in embedding_models)
                ]
            else:
                # OpenAI format (LM Studio): {"data": [{"id": "..."}, ...]}
                return [
                    m["id"]
                    for m in data.get("data", [])
                    if not m["id"].startswith("text-embedding")
                ]
    except Exception as e:
        logger.error(f"Error fetching models from {name}: {e}")
        return []


def get_lmstudio_cache_path() -> Path:
    """Get LM Studio cache directory for GGUF downloads"""
    cache_dir = Path.home() / ".cache" / "lm-studio" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_lms_cli_path() -> Optional[Path]:
    """Find lms.exe CLI tool"""
    lms_path = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    if lms_path.exists():
        return lms_path
    # Try alternate Windows location
    appdata = os.getenv("LOCALAPPDATA")
    if appdata:
        alt_path = Path(appdata) / "LM Studio" / "bin" / "lms.exe"
        if alt_path.exists():
            return alt_path
    return None


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


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
    pattern = r"^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$"
    return bool(re.match(pattern, model_name))


def _format_size(bytes_size: int) -> str:
    """Format byte size in human-readable format"""
    if bytes_size == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


@router.get("/ollama/status")
async def check_ollama_status():
    """Check if Ollama is installed and running"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return {"available": True, "running": True}
            return {
                "available": False,
                "running": False,
                "reason": "Ollama not responding",
            }
    except (httpx.ConnectError, httpx.TimeoutException):
        return {
            "available": False,
            "running": False,
            "reason": "Ollama not running or not installed",
        }
    except Exception as e:
        logger.error(f"Error checking Ollama status: {e}")
        return {"available": False, "running": False, "reason": str(e)}


@router.get("/lmstudio/status")
async def check_lmstudio_status():
    """Check if LM Studio server is running"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                return {"available": True, "running": True}
            return {
                "available": False,
                "running": False,
                "reason": "LM Studio not responding",
            }
    except (httpx.ConnectError, httpx.TimeoutException):
        return {
            "available": False,
            "running": False,
            "reason": "LM Studio server not running",
        }
    except Exception as e:
        logger.error(f"Error checking LM Studio status: {e}")
        return {"available": False, "running": False, "reason": str(e)}


@router.get("/providers/detect")
async def detect_providers():
    """Detect all running LLM providers"""
    detected = []

    for name, config in PROVIDERS.items():
        provider_info = await detect_provider(name, config)
        if provider_info:
            # Get models for this provider
            models = await get_provider_models(name, config)
            provider_info["models"] = models
            provider_info["model_count"] = len(models)
            detected.append(provider_info)
            logger.info(
                f"✓ Detected {name} on port {config['port']} with {len(models)} models"
            )

    # Get current active provider from environment
    active_provider = os.getenv("ROAMPAL_LLM_PROVIDER", "ollama")

    return {"providers": detected, "active": active_provider, "count": len(detected)}


@router.get("/providers/all/models")
async def list_all_provider_models():
    """Get models from ALL detected providers"""
    all_models = {}

    for name, config in PROVIDERS.items():
        provider_info = await detect_provider(name, config)
        if provider_info:
            models = await get_provider_models(name, config)
            all_models[name] = models
        else:
            all_models[name] = []

    return {"providers": all_models}


@router.get("/providers/{provider_name}/models")
async def list_provider_models(provider_name: str):
    """Get models for a specific provider"""
    if provider_name not in PROVIDERS:
        raise HTTPException(
            400,
            f"Unknown provider: {provider_name}. Supported: {list(PROVIDERS.keys())}",
        )

    config = PROVIDERS[provider_name]

    # Check if provider is running
    provider_info = await detect_provider(provider_name, config)
    if not provider_info:
        raise HTTPException(
            503, f"Provider '{provider_name}' is not running on port {config['port']}"
        )

    models = await get_provider_models(provider_name, config)

    return {"provider": provider_name, "models": models, "count": len(models)}


@router.post("/download-gguf-stream")
async def download_gguf_stream(request_body: Dict[str, Any] = Body(...)):
    """Download GGUF model from HuggingFace and import to LM Studio"""

    model_name = request_body.get("model", "")

    # Resolve model name to HuggingFace info (supports quantization suffixes)
    resolved_model_info = resolve_model_for_lmstudio(model_name)
    if not resolved_model_info:
        raise HTTPException(
            404, f"Model {model_name} not available for LM Studio auto-download"
        )

    async def generate():
        # Use resolved model info (closure captures it)
        model_info = resolved_model_info

        # Acquire download lock INSIDE generator to ensure it's held during the actual download
        logger.info(
            f"[LOCK DEBUG] Generator started, attempting to acquire lock for {model_name}"
        )
        async with _download_lock:
            logger.info(
                f"[LOCK DEBUG] Lock acquired for {model_name}, current downloads: {_downloading_models}"
            )
            # Only allow one download at a time (any model)
            if _downloading_models:
                current = next(iter(_downloading_models))
                logger.info(
                    f"[LOCK DEBUG] Blocking {model_name} - {current} is already downloading"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': f'Cannot download {model_name}: {current} is already downloading'})}\n\n"
                return
            if model_name in _downloading_models:
                logger.info(
                    f"[LOCK DEBUG] Blocking {model_name} - already in download set"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': f'Model {model_name} is already being downloaded'})}\n\n"
                return
            _downloading_models.add(model_name)
            logger.info(
                f"[LOCK DEBUG] Added {model_name} to download set, releasing lock. Set: {_downloading_models}"
            )

        logger.info(
            f"[LOCK DEBUG] Lock released, starting download for {model_name}, set ID: {id(_downloading_models)}"
        )
        cache_path = None  # v0.3.0: Track for cleanup on error
        download_success = False  # v0.3.0: Track if download completed successfully
        try:
            # 1. Download GGUF file from HuggingFace
            cache_path = get_lmstudio_cache_path() / model_info["file"]
            hf_url = f"https://huggingface.co/{model_info['repo']}/resolve/main/{model_info['file']}"

            logger.info(f"Downloading {model_name} from {hf_url}")

            yield f"data: {json.dumps({'type': 'start', 'model': model_name})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'status': 'downloading', 'progress': 0, 'message': 'Starting download...'})}\n\n"

            async with httpx.AsyncClient(
                timeout=600.0, follow_redirects=True
            ) as client:
                async with client.stream("GET", hf_url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    last_progress = 0
                    download_start_time = time.time()
                    bytes_at_last_update = 0
                    last_update_time = download_start_time

                    with open(cache_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            # Check if download was cancelled
                            if _cancel_flags.get(model_name, False):
                                logger.info(f"Download cancelled for {model_name}")
                                raise Exception("Download cancelled by user")

                            f.write(chunk)
                            downloaded += len(chunk)

                            if total > 0:
                                progress = int((downloaded / total) * 100)
                                # Yield every 1% for smooth progress (same as Ollama)
                                if progress >= last_progress + 1 or progress == 100:
                                    downloaded_str = format_bytes(downloaded)
                                    total_str = format_bytes(total)

                                    # Calculate download speed
                                    current_time = time.time()
                                    time_elapsed = current_time - last_update_time
                                    if time_elapsed > 0:
                                        bytes_diff = downloaded - bytes_at_last_update
                                        speed_bps = bytes_diff / time_elapsed
                                        speed_str = f"{format_bytes(int(speed_bps))}/s"
                                    else:
                                        speed_str = "calculating..."

                                    bytes_at_last_update = downloaded
                                    last_update_time = current_time

                                    yield f"data: {json.dumps({'type': 'progress', 'status': 'downloading', 'percent': progress, 'downloaded': downloaded_str, 'total': total_str, 'speed': speed_str, 'message': f'Downloading {progress}%'})}\n\n"
                                    last_progress = progress

            logger.info(f"Downloaded {model_name} to {cache_path}")

            # 2. Import via lms.exe
            yield f"data: {json.dumps({'type': 'progress', 'status': 'importing', 'progress': 0, 'message': 'Importing to LM Studio...'})}\n\n"

            lms_cli = get_lms_cli_path()
            if not lms_cli:
                raise Exception("lms.exe not found. Is LM Studio installed?")

            logger.info(f"Importing with lms.exe: {lms_cli}")

            # Use Popen with stdin PIPE to answer first-run prompt
            process = subprocess.Popen(
                [
                    str(lms_cli),
                    "import",
                    "--yes",
                    "--copy",
                    "--user-repo",
                    model_info["repo"],
                    str(cache_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=_SUBPROCESS_FLAGS,
            )

            # Send Y to answer first-run prompt, then close stdin
            if process.stdin:
                process.stdin.write("Y\n")
                process.stdin.flush()
                process.stdin.close()

            # Read output in real-time with heartbeat updates
            stdout_lines = []
            stderr_lines = []
            last_heartbeat = 0

            while True:
                # Check if process is done
                retcode = process.poll()
                if retcode is not None:
                    # Process finished - read any remaining output
                    remaining_out = process.stdout.read() if process.stdout else ""
                    remaining_err = process.stderr.read() if process.stderr else ""
                    if remaining_out:
                        stdout_lines.append(remaining_out)
                    if remaining_err:
                        stderr_lines.append(remaining_err)
                    break

                # Send heartbeat every 5 seconds to show it's still working
                current_time = time.time()
                if current_time - last_heartbeat >= 5:
                    yield f"data: {json.dumps({'type': 'progress', 'status': 'importing', 'progress': 0, 'message': 'Importing to LM Studio (this may take a minute)...'})}\n\n"
                    last_heartbeat = current_time

                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.5)

            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)

            # Create result object to match expected interface
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            result = Result(retcode, stdout, stderr)

            # Check if file was actually imported (don't trust stderr)
            imported_file_path = (
                Path.home()
                / ".lmstudio"
                / "models"
                / model_info["repo"].split("/")[0]
                / model_info["repo"].split("/")[1]
                / model_info["file"]
            )

            if not imported_file_path.exists():
                logger.error(
                    f"lms import failed - file not found at: {imported_file_path}"
                )
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise Exception(
                    f"Import failed: Model file not found in LM Studio models directory"
                )

            logger.info(f"Successfully imported {model_name} to {imported_file_path}")

            # Delete cache file after successful import
            try:
                cache_path.unlink()
                logger.info(f"Deleted cache file: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not delete cache file {cache_path}: {e}")

            # Query LM Studio API to get the actual model ID after import
            actual_model_id = None
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:1234/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        lmstudio_models = [
                            m.get("id", "") for m in data.get("data", [])
                        ]
                        logger.info(f"LM Studio models after import: {lmstudio_models}")

                        # v0.3.0: Use unified matching function
                        for model_id in lmstudio_models:
                            if _model_names_match(model_name, model_id, model_info):
                                actual_model_id = model_id
                                logger.info(
                                    f"Found LM Studio model ID: {actual_model_id}"
                                )
                                break
            except Exception as e:
                logger.warning(f"Failed to query LM Studio for actual model ID: {e}")

            # Use actual model ID if found, otherwise fall back to original name
            model_to_return = actual_model_id if actual_model_id else model_name
            logger.info(
                f"Successfully imported {model_name} - ready for use as '{model_to_return}'"
            )
            download_success = True  # v0.3.0: Mark success to prevent cleanup
            yield f"data: {json.dumps({'type': 'loaded', 'model': model_to_return, 'message': 'Model ready!'})}\n\n"

        except httpx.ConnectError as e:
            logger.error(f"Connection failed for {model_name}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cannot connect to HuggingFace. Check your internet connection.'})}\n\n"
        except httpx.TimeoutException as e:
            logger.error(f"Download timeout for {model_name}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Download timed out for {model_name}. Try again or check your connection.'})}\n\n"
        except OSError as e:
            if "No space left" in str(e) or "not enough space" in str(e).lower():
                model_info = MODEL_TO_HUGGINGFACE.get(model_name, {})
                size_gb = model_info.get("size_gb", "unknown")
                logger.error(f"Disk space error for {model_name}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Not enough disk space to download {model_name} ({size_gb}GB required)'})}\n\n"
            else:
                logger.error(f"Disk error for {model_name}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Disk error: {str(e)}'})}\n\n"
        except PermissionError as e:
            logger.error(f"Permission error for {model_name}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Permission denied. Try running as administrator or check folder permissions.'})}\n\n"
        except Exception as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                logger.info(f"Download cancelled for {model_name}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Download cancelled for {model_name}'})}\n\n"
            else:
                logger.error(
                    f"GGUF download/import failed for {model_name}: {e}", exc_info=True
                )
                yield f"data: {json.dumps({'type': 'error', 'message': f'Unexpected error: {str(e)}'})}\n\n"
        finally:
            logger.info(
                f"[LOCK DEBUG] Cleaning up {model_name} from download set. Set before: {_downloading_models}"
            )
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)  # Clean up cancel flag

            # v0.3.0: Delete partial download on failure
            if not download_success and cache_path and cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(
                        f"Deleted partial download for {model_name}: {cache_path}"
                    )
                except Exception as cleanup_err:
                    logger.warning(
                        f"Could not delete partial download {cache_path}: {cleanup_err}"
                    )

            logger.info(
                f"[LOCK DEBUG] Cleaned up {model_name}. Set after: {_downloading_models}"
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/cancel-download")
async def cancel_download(request_body: Dict[str, Any] = Body(...)):
    """Cancel ongoing GGUF download and cleanup"""
    model_name = request_body.get("model", "")

    if model_name in _downloading_models:
        # Set cancel flag to stop the download loop
        _cancel_flags[model_name] = True
        logger.info(f"Cancel flag set for {model_name}")

        # Wait a moment for the download to stop
        await asyncio.sleep(0.5)

        # Remove from downloading set
        _downloading_models.discard(model_name)

        # Try to delete partial download if it exists
        if model_name in MODEL_TO_HUGGINGFACE:
            model_info = MODEL_TO_HUGGINGFACE[model_name]
            cache_path = get_lmstudio_cache_path() / model_info["file"]
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(f"Deleted partial download for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not delete partial file {cache_path}: {e}")

        return {"status": "cancelled", "model": model_name}

    return {"status": "not_found", "model": model_name}


def _model_names_match(
    registry_name: str, provider_id: str, model_info: dict = None
) -> bool:
    """
    v0.3.0: Robust model name matching for LM Studio and Ollama.
    Handles naming convention differences like:
    - llama3.3:70b vs llama-3.3-70b-instruct
    - qwen2.5:7b vs qwen2.5-7b-instruct

    Args:
        registry_name: Roampal registry name (e.g., "llama3.3:70b")
        provider_id: LM Studio/Ollama model ID (e.g., "llama-3.3-70b-instruct")
        model_info: Optional dict with 'file' key containing GGUF filename

    Returns:
        True if the names refer to the same model
    """
    provider_lower = provider_id.lower()

    # Strategy 1: Direct match after basic normalization
    normalized = registry_name.replace(":", "-").lower()
    if normalized in provider_lower or provider_lower.startswith(normalized):
        return True

    # Strategy 2: Match via GGUF filename stem (most reliable for LM Studio)
    if model_info and model_info.get("file"):
        gguf_file = model_info["file"]
        # Extract stem: "Llama-3.3-70B-Instruct-Q4_K_M.gguf" -> "llama-3.3-70b-instruct"
        stem = gguf_file.replace(".gguf", "").lower()
        # Remove quantization suffixes
        for q in ["-q4_k_m", "-q8_0", "-q5_k_m", "-q6_k", "-q4_0", ".q4_k_m", ".q8_0"]:
            stem = stem.replace(q, "")
        if stem in provider_lower or provider_lower in stem:
            return True

    # Strategy 3: Extract and compare key components (model family + size)
    # "llama3.3:70b" -> ["llama", "3", "3", "70b"]
    # "qwen2.5:7b" -> ["qwen", "2", "5", "7b"]
    import re

    registry_parts = re.split(r"[:.\\-_]", registry_name.lower())
    registry_parts = [p for p in registry_parts if p]  # Remove empty

    # All registry parts must appear in provider ID
    if all(part in provider_lower for part in registry_parts):
        # But also check size match to avoid qwen2.5:7b matching qwen2.5:70b
        # Extract size patterns (numbers followed by 'b')
        registry_sizes = re.findall(r"(\d+b)", registry_name.lower())
        provider_sizes = re.findall(r"(\d+b)", provider_lower)
        if registry_sizes and provider_sizes:
            # Must have at least one common size
            if any(s in provider_sizes for s in registry_sizes):
                return True
        elif not registry_sizes:
            # No size in registry name, parts match is enough
            return True

    return False


def _map_lmstudio_id_to_original_name(lmstudio_id: str) -> str:
    """Map LM Studio model ID back to original download name (e.g., qwen2.5-7b-instruct -> qwen2.5:7b)"""
    # v0.3.0: Use unified matching function
    for original_name, model_info in MODEL_TO_HUGGINGFACE.items():
        if _model_names_match(original_name, lmstudio_id, model_info):
            return original_name

    # If no match, return the LM Studio ID as-is
    return lmstudio_id


@router.get("/available")
async def list_available_models():
    """List all locally available models from both Ollama and LM Studio"""
    all_models = []

    # 1. Get Ollama models (exclude embedding models)
    embedding_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                ollama_count = 0
                for model in data.get("models", []):
                    model_name = model.get("name", "")
                    # Skip embedding models
                    is_embedding = any(
                        embed in model_name.lower() for embed in embedding_models
                    )
                    if not is_embedding:
                        all_models.append(
                            {
                                "name": model_name,
                                "id": model.get("digest", "")[:12],
                                "size": _format_size(model.get("size", 0)),
                                "modified": model.get("modified_at", ""),
                                "provider": "ollama",
                            }
                        )
                        ollama_count += 1
                logger.info(f"Found {ollama_count} Ollama chat models")
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")

    # 2. Get LM Studio models
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                data = response.json()
                lmstudio_count = 0
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if (
                        model_id and model_id != "text-embedding-nomic-embed-text-v1.5"
                    ):  # Skip embedding models
                        # Map LM Studio ID back to original name for UI matching
                        original_name = _map_lmstudio_id_to_original_name(model_id)
                        all_models.append(
                            {
                                "name": original_name,  # Use original name (qwen2.5:7b) for UI matching
                                "id": model_id[:12] if len(model_id) > 12 else model_id,
                                "size": "N/A",  # LM Studio API doesn't provide size easily
                                "modified": "",
                                "provider": "lmstudio",
                                "lmstudio_id": model_id,  # Store actual LM Studio ID for switching
                            }
                        )
                        lmstudio_count += 1
                logger.info(f"Found {lmstudio_count} LM Studio models")
    except Exception as e:
        logger.warning(f"Failed to fetch LM Studio models: {e}")

    return {"models": all_models, "count": len(all_models)}


@router.get("/gpu-info")
async def get_gpu_info(request: Request):
    """v0.3.2 (0g): Report detected GPU VRAM so the UI can size model warnings."""
    return {
        "gpu_vram_gb": getattr(request.app.state, "gpu_vram_gb", None),
        "gpu_count": getattr(request.app.state, "gpu_count", 0),
        "source": getattr(request.app.state, "gpu_source", "none"),
    }


class _DebugVramRequest(BaseModel):
    gpu_vram_gb: Optional[float] = None  # None = restore real detected value


@router.post("/_debug/set-vram")
async def _debug_set_vram(request: Request, body: _DebugVramRequest):
    """v0.3.2 (0g) dev-only: override app.state.gpu_vram_gb for manual QA.

    Gated on `ROAMPAL_DEV=1` — returns 403 in production builds. Lets us
    exercise the VRAM warning toast without downloading an oversized model.
    POST body: {"gpu_vram_gb": 2.0} to simulate a 2GB card, or
    {"gpu_vram_gb": null} to restore the real detected value.
    """
    if os.getenv("ROAMPAL_DEV", "").strip() not in ("1", "true", "yes"):
        raise HTTPException(status_code=403, detail="Dev-only endpoint")

    # Remember the original on first override so we can restore it.
    if not hasattr(request.app.state, "_gpu_vram_gb_original"):
        request.app.state._gpu_vram_gb_original = getattr(
            request.app.state, "gpu_vram_gb", None
        )

    if body.gpu_vram_gb is None:
        request.app.state.gpu_vram_gb = request.app.state._gpu_vram_gb_original
        logger.info(f"[DEV] Restored gpu_vram_gb to {request.app.state.gpu_vram_gb}")
    else:
        request.app.state.gpu_vram_gb = float(body.gpu_vram_gb)
        logger.info(f"[DEV] Overrode gpu_vram_gb to {request.app.state.gpu_vram_gb}")

    return {"gpu_vram_gb": request.app.state.gpu_vram_gb}


def _vram_warning_for(model_name: str, available_vram_gb: Optional[float]) -> Optional[Dict[str, Any]]:
    """v0.3.2 (0g): Return a warning dict if a model likely won't fit in VRAM.

    Strategy (universal, per Logan's rule — no per-model blocklists):
      1. Curated registry with a `vram_gb` field → exact comparison
      2. Tag parses as NB → estimate at ~params × 0.75 (Q4_K_M heuristic)
      3. Otherwise → return None (don't guess)
    """
    if available_vram_gb is None:
        return None  # detection failed — never guess

    # 1. Curated registry
    try:
        for base, quants in QUANTIZATION_OPTIONS.items():
            for level, meta in quants.items():
                if meta.get("ollama_tag") == model_name or base == model_name:
                    required = float(meta.get("vram_gb") or 0)
                    if required and required > available_vram_gb:
                        return {
                            "kind": "curated",
                            "required_gb": required,
                            "available_gb": available_vram_gb,
                            "message": (
                                f"`{model_name}` recommends {required} GB VRAM but "
                                f"only {available_vram_gb:.1f} GB detected. Expect "
                                f"slow inference or OOM."
                            ),
                        }
    except Exception:
        pass  # treat any registry lookup failure as "can't curate — fall through"

    # 2. Tag heuristic
    from utils.gpu_detection import estimate_vram_from_tag
    estimated = estimate_vram_from_tag(model_name)
    if estimated is not None and estimated > available_vram_gb:
        return {
            "kind": "estimated",
            "required_gb": estimated,
            "available_gb": available_vram_gb,
            "message": (
                f"`{model_name}` is estimated to need ~{estimated} GB VRAM but "
                f"only {available_vram_gb:.1f} GB detected. This is a rough "
                f"heuristic — confirm quantization before download."
            ),
        }

    # 3. No signal — silent pass
    return None


@router.get("/current")
async def get_current_model(request: Request):
    """Get the currently active model"""
    # ALWAYS prioritize runtime state over env vars (runtime is source of truth)
    current_model = None
    llm_client_available = False

    if (
        hasattr(request.app.state, "llm_client")
        and request.app.state.llm_client is not None
    ):
        if hasattr(request.app.state.llm_client, "model_name"):
            current_model = request.app.state.llm_client.model_name
            llm_client_available = True
            logger.info(f"Current model from runtime: {current_model}")

    # Fallback to env var only if no runtime client
    if not current_model:
        current_model = (
            os.getenv("ROAMPAL_LLM_OLLAMA_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "codellama"
        )
        logger.info(f"Current model from env fallback: {current_model}")

    # CRITICAL: Verify model actually exists in its provider before saying it's active
    # Use llm_client's api_style as source of truth - it knows which provider it's configured for
    model_exists = False
    if current_model:
        checking_provider = None
        if (
            hasattr(request.app.state, "llm_client")
            and request.app.state.llm_client is not None
        ):
            if hasattr(request.app.state.llm_client, "api_style"):
                api_style = request.app.state.llm_client.api_style
                checking_provider = "lmstudio" if api_style == "openai" else "ollama"
            elif hasattr(request.app.state.llm_client, "base_url"):
                base_url = request.app.state.llm_client.base_url
                checking_provider = "lmstudio" if "1234" in base_url else "ollama"

        # Check the appropriate provider
        if checking_provider == "ollama":
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=_SUBPROCESS_FLAGS,
                )
                if result.returncode == 0:
                    available_models = [
                        line.split()[0]
                        for line in result.stdout.strip().split("\n")[1:]
                        if line.strip()
                    ]
                    model_exists = current_model in available_models
                    if not model_exists:
                        logger.warning(f"Model {current_model} not found in Ollama")
            except Exception as e:
                logger.error(f"Failed to verify model in Ollama: {e}")

        elif checking_provider == "lmstudio":
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get("http://localhost:1234/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        available_models = [m["id"] for m in data.get("data", [])]
                        model_exists = current_model in available_models
                        if not model_exists:
                            logger.warning(
                                f"Model {current_model} not found in LM Studio"
                            )
            except Exception as e:
                logger.error(f"Failed to verify model in LM Studio: {e}")

        # If model doesn't exist in its provider, clear it
        if not model_exists:
            current_model = None

    # Check if current model is an embedding model or None
    embedding_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
    is_embedding = current_model and any(
        embed in current_model.lower() for embed in embedding_models
    )
    # Only can_chat if: llm_client exists AND model exists AND not embedding
    can_chat = (
        llm_client_available and current_model and not is_embedding and model_exists
    )

    # Get current provider from llm_client
    current_provider = None
    if (
        hasattr(request.app.state, "llm_client")
        and request.app.state.llm_client is not None
    ):
        if hasattr(request.app.state.llm_client, "api_style"):
            api_style = request.app.state.llm_client.api_style
            # Map api_style to provider name
            current_provider = "lmstudio" if api_style == "openai" else "ollama"
        elif hasattr(request.app.state.llm_client, "base_url"):
            # Fallback: detect from base_url
            base_url = request.app.state.llm_client.base_url
            current_provider = "lmstudio" if "1234" in base_url else "ollama"

    return {
        "current_model": current_model
        if (llm_client_available and model_exists)
        else None,
        "can_chat": can_chat,
        "is_embedding_model": is_embedding,
        "provider": current_provider,
    }


@router.post("/switch")
async def switch_model(request: Request, model_request: ModelSwitchRequest):
    """Switch to a different model at runtime - supports both Ollama and LM Studio"""
    # v0.3.0: Prevent concurrent switches from corrupting state
    async with _switch_lock:
        try:
            model_name = model_request.model_name

            # Step 1: Detect which provider owns this model
            provider = None
            base_url = None
            api_style = None

            # Check Ollama
            try:
                import shutil

                ollama_cmd = shutil.which("ollama")
                if not ollama_cmd:
                    common_paths = [
                        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(
                            os.environ.get("USERNAME", "")
                        ),
                        r"C:\Program Files\Ollama\ollama.exe",
                    ]
                    for path in common_paths:
                        if os.path.exists(path):
                            ollama_cmd = path
                            break

                if ollama_cmd:
                    result = subprocess.run(
                        [ollama_cmd, "list"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        creationflags=_SUBPROCESS_FLAGS,
                    )
                    if result.returncode == 0:
                        available_models = [
                            line.split()[0]
                            for line in result.stdout.strip().split("\n")[1:]
                            if line.strip()
                        ]
                        if model_name in available_models:
                            provider = "ollama"
                            base_url = "http://localhost:11434"
                            api_style = "ollama"
                            logger.info(f"Model '{model_name}' found in Ollama")
            except Exception as e:
                logger.warning(f"Failed to check Ollama: {e}")

            # Check LM Studio if not found in Ollama
            if not provider:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get("http://localhost:1234/v1/models")
                        if response.status_code == 200:
                            data = response.json()
                            lmstudio_models = [
                                model.get("id", "") for model in data.get("data", [])
                            ]
                            if model_name in lmstudio_models:
                                provider = "lmstudio"
                                base_url = "http://localhost:1234"
                                api_style = "openai"
                                logger.info(f"Model '{model_name}' found in LM Studio")
                except Exception as e:
                    logger.warning(f"Failed to check LM Studio: {e}")

            # If model not found in either provider, raise error
            if not provider:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found in Ollama or LM Studio",
                )

            # Step 2: Update the LLM client
            if hasattr(request.app.state, "llm_client"):
                # Lazy initialization: create LLM client if None
                if request.app.state.llm_client is None:
                    from modules.llm.ollama_client import OllamaClient

                    request.app.state.llm_client = OllamaClient()
                    await request.app.state.llm_client.initialize(
                        {"ollama_model": model_name}
                    )
                    logger.info(
                        f"Lazily initialized LLM client with model: {model_name}"
                    )
                    previous_model = None
                    previous_provider = None
                else:
                    # Store previous model and provider for rollback
                    previous_model = request.app.state.llm_client.model_name
                    previous_provider = getattr(
                        request.app.state.llm_client, "api_style", "ollama"
                    )

                    # Update llm_client with new provider configuration
                    request.app.state.llm_client.model_name = model_name
                    request.app.state.llm_client.base_url = base_url
                    request.app.state.llm_client.api_style = api_style
                    logger.info(
                        f"Updated llm_client: model={model_name}, provider={provider}, base_url={base_url}"
                    )

                # Update global agent_service LLM reference
                from app.routers.agent_chat import agent_service

                if agent_service and hasattr(agent_service, "llm"):
                    agent_service.llm = request.app.state.llm_client

                # Step 3: Run lightweight health check (don't load model into GPU!)
                try:
                    logger.info(
                        f"Performing lightweight health check on {provider} model: {model_name}"
                    )

                    # Recycle HTTP client if available
                    if hasattr(request.app.state.llm_client, "_recycle_client"):
                        await request.app.state.llm_client._recycle_client()

                    # Give provider time to start (but don't load model)
                    await asyncio.sleep(1.0)

                    # LIGHTWEIGHT check: verify model is available without loading it
                    health_timeout = 10.0  # Much shorter timeout for availability check
                    async with httpx.AsyncClient(
                        timeout=health_timeout
                    ) as health_client:
                        if provider == "ollama":
                            # Check if model is in Ollama's list (doesn't load it)
                            tags_response = await health_client.get(
                                "http://localhost:11434/api/tags"
                            )
                            tags_response.raise_for_status()
                            tags_data = tags_response.json()
                            available_models = [
                                model.get("name")
                                for model in tags_data.get("models", [])
                            ]

                            if model_name not in available_models:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Model '{model_name}' not found in Ollama. Available: {', '.join(available_models[:5])}...",
                                )

                            logger.info(
                                f"✓ Model '{model_name}' available in Ollama (not loaded into GPU)"
                            )

                        else:  # lmstudio
                            # Check if model is in LM Studio's list (doesn't load it)
                            models_response = await health_client.get(
                                "http://localhost:1234/v1/models"
                            )
                            models_response.raise_for_status()
                            models_data = models_response.json()
                            available_models = [
                                model.get("id", "")
                                for model in models_data.get("data", [])
                            ]

                            if model_name not in available_models:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Model '{model_name}' not found in LM Studio. Available: {', '.join(available_models[:5])}...",
                                )

                            logger.info(
                                f"✓ Model '{model_name}' available in LM Studio (not loaded into GPU)"
                            )

                except HTTPException:
                    raise  # Re-raise HTTP exceptions (like 404 for model not found)
                except Exception as health_error:
                    # Lightweight check failed - provider might not be running
                    error_detail = (
                        str(health_error).strip() or "timeout (no response within 10s)"
                    )
                    logger.error(
                        f"Lightweight health check failed for {model_name}: {error_detail}"
                    )

                    # Rollback
                    if previous_model:
                        request.app.state.llm_client.model_name = previous_model
                        if previous_provider == "lmstudio":
                            request.app.state.llm_client.base_url = (
                                "http://localhost:1234"
                            )
                            request.app.state.llm_client.api_style = "openai"
                        else:
                            request.app.state.llm_client.base_url = (
                                "http://localhost:11434"
                            )
                            request.app.state.llm_client.api_style = "ollama"

                    # Better error messages
                    error_str = str(health_error).lower()
                    if "timeout" in error_str or "timed out" in error_str:
                        error_msg = f"{provider.upper()} not responding. Make sure {provider.upper()} is running."
                    elif "connection" in error_str or "refused" in error_str:
                        error_msg = f"Cannot connect to {provider.upper()}. Make sure it's running on port {11434 if provider == 'ollama' else 1234}."
                    else:
                        error_msg = f"Provider check failed: {str(health_error)}"

                    raise HTTPException(
                        status_code=503,
                        detail=f"{error_msg} Rolled back to {previous_model}.",
                    )
                    logger.error(
                        f"Health check failed for {model_name}: {error_detail}"
                    )
                    # Rollback
                    if previous_model:
                        request.app.state.llm_client.model_name = previous_model
                        if previous_provider == "lmstudio":
                            request.app.state.llm_client.base_url = (
                                "http://localhost:1234"
                            )
                            request.app.state.llm_client.api_style = "openai"
                        else:
                            request.app.state.llm_client.base_url = (
                                "http://localhost:11434"
                            )
                            request.app.state.llm_client.api_style = "ollama"

                    # v0.3.0: Better error messages for common failures
                    error_str = str(health_error).lower()
                    if (
                        "timeout" in error_str
                        or "timed out" in error_str
                        or not str(health_error).strip()
                    ):
                        error_msg = f"Model load timed out. The model may be too large or still loading. Try a smaller model or pre-load it in {provider.upper()}."
                    elif "400" in error_str or "bad request" in error_str:
                        error_msg = f"Model {model_name} returned an error. It may not be fully loaded in {provider.upper()}."
                    else:
                        error_msg = f"Health check failed: {str(health_error)}"

                    raise HTTPException(
                        status_code=503,
                        detail=f"{error_msg} Rolled back to {previous_model}.",
                    )

                # v0.3.2 (0b): Persist the selection to DATA_PATH, not the
                # install dir. The old `.env` write silently failed on default
                # Windows installs because Program Files isn't user-writable,
                # so model selection never survived a restart.
                os.environ["ROAMPAL_LLM_PROVIDER"] = provider
                if provider == "ollama":
                    os.environ["OLLAMA_MODEL"] = model_name
                    os.environ["ROAMPAL_LLM_OLLAMA_MODEL"] = model_name
                else:
                    os.environ["ROAMPAL_LLM_LMSTUDIO_MODEL"] = model_name

                save_main_model_config(model=model_name, provider=provider)

                # v0.3.2 (0f): If sidecar is in mirror-chat mode, update it
                # atomically so the sidecar re-uses the current chat model
                # and avoids costly single-GPU model swaps on each message.
                try:
                    existing = _load_sidecar_config(request)
                    if existing.get("mirror_chat", True):
                        _save_sidecar_config(
                            request,
                            {
                                "enabled": True,
                                "model": model_name,
                                "provider": provider,
                                "mirror_chat": True,
                            },
                        )
                        request.app.state.sidecar_model = model_name
                        request.app.state.sidecar_provider = provider
                        # Share the freshly-switched main client.
                        request.app.state.sidecar_client = request.app.state.llm_client
                        logger.info(
                            f"[SIDECAR] Mirror updated to {provider}/{model_name}"
                        )
                except Exception as mirror_err:
                    logger.warning(f"[SIDECAR] Mirror update failed: {mirror_err}")

                # v0.3.0: Validate state after switch
                actual_model = request.app.state.llm_client.model_name
                if actual_model != model_name:
                    logger.warning(
                        f"State validation failed: expected {model_name}, got {actual_model}. Correcting."
                    )
                    request.app.state.llm_client.model_name = model_name

                # v0.3.2 (0g): surface a VRAM warning alongside the success
                # response so the UI can render a badge without a second RTT.
                warning = _vram_warning_for(
                    model_name,
                    getattr(request.app.state, "gpu_vram_gb", None),
                )
                return {
                    "status": "success",
                    "message": f"Switched to {provider} model: {model_name}",
                    "current_model": model_name,
                    "provider": provider,
                    "vram_warning": warning,
                }
            else:
                raise HTTPException(
                    status_code=503, detail="LLM client not initialized"
                )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            logger.error(f"Error switching model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail="Failed to switch model"
            )


@router.post("/pull")
async def pull_model(request_body: Dict[str, Any] = Body(...)):
    """Download a new model from Ollama registry"""
    try:
        # Extract model name from request body (UI sends 'model', accept both)
        model_name = request_body.get("model") or request_body.get("model_name", "")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")

        # Validate model name format
        if not _validate_model_name(model_name):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name format. Expected format: name:tag (e.g., 'qwen3:8b')",
            )

        # Check if already downloading
        if model_name in _downloading_models:
            raise HTTPException(
                status_code=409,
                detail=f"Model {model_name} is already being downloaded",
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
                timeout=600,  # 10 minute timeout for large models
                creationflags=_SUBPROCESS_FLAGS,
            )

            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Successfully pulled {model_name}",
                    "output": result.stdout,
                }
            else:
                raise HTTPException(
                    status_code=500, detail=f"Failed to pull model: {result.stderr}"
                )
        finally:
            # Always remove from downloading set and clean up cancel flag
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)

    except subprocess.TimeoutExpired:
        _downloading_models.discard(model_name)
        _cancel_flags.pop(model_name, None)
        raise HTTPException(
            status_code=504, detail="Model pull timed out after 10 minutes"
        )
    except HTTPException:
        raise
    except Exception as e:
        _downloading_models.discard(model_name)
        _cancel_flags.pop(model_name, None)
        logger.error(f"Error pulling model: {e}"); raise HTTPException(status_code=500, detail="Error pulling model")


@router.post("/pull-stream")
async def pull_model_stream(request_body: Dict[str, Any] = Body(...)):
    """Download a model with real-time progress streaming"""
    model_name = request_body.get("model") or request_body.get("model_name", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    # Validate model name format
    if not _validate_model_name(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name format. Expected format: name:tag (e.g., 'qwen3:8b')",
        )

    # Acquire download lock and check if already downloading (prevents race condition)
    async with _download_lock:
        # Only allow one download at a time (any model)
        if _downloading_models:
            current = next(iter(_downloading_models))
            raise HTTPException(
                status_code=409,
                detail=f"Cannot download {model_name}: {current} is already downloading. Only one download at a time.",
            )
        _downloading_models.add(model_name)

    async def generate_progress() -> AsyncGenerator[str, None]:
        """Generate SSE events with download progress using Ollama HTTP API"""
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': model_name})}\n\n"

            # Use Ollama's HTTP API endpoint for pulling models
            # Configure granular timeouts: 60s between data chunks prevents infinite hangs
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,  # 5s to establish connection
                    read=60.0,  # 60s between receiving data chunks (prevents hang)
                    write=5.0,  # 5s to send request
                    pool=5.0,  # 5s to get connection from pool
                )
            ) as client:
                async with client.stream(
                    "POST",
                    "http://localhost:11434/api/pull",
                    json={"name": model_name, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(
                            f"Ollama API error: {error_text.decode('utf-8', errors='replace')}"
                        )

                    # Process NDJSON stream (one JSON object per line)
                    async for line in response.aiter_lines():
                        # Check if download was cancelled
                        if _cancel_flags.get(model_name, False):
                            logger.info(f"Ollama download cancelled for {model_name}")
                            raise Exception("Download cancelled by user")

                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)
                            status = data.get("status", "")

                            logger.info(f"Ollama API: {status}")

                            # Check for errors first
                            if "error" in data:
                                error_msg = data.get("error", "Unknown error")
                                logger.error(f"Ollama API error: {error_msg}")
                                error_data = {
                                    "type": "error",
                                    "message": f"Download failed: {error_msg}",
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return  # Exit generator, don't send false success

                            # Map Ollama API responses to SSE events
                            if "total" in data and "completed" in data:
                                # Download progress
                                total = data["total"]
                                completed = data["completed"]
                                percent = (
                                    int((completed / total * 100)) if total > 0 else 0
                                )

                                # Format sizes
                                def format_bytes(b):
                                    for unit in ["B", "KB", "MB", "GB"]:
                                        if b < 1024:
                                            return f"{b:.1f} {unit}"
                                        b /= 1024
                                    return f"{b:.1f} TB"

                                downloaded_str = format_bytes(completed)
                                total_str = format_bytes(total)

                                progress_data = {
                                    "type": "progress",
                                    "percent": percent,
                                    "downloaded": downloaded_str,
                                    "total": total_str,
                                    "speed": "downloading",
                                    "message": f"{status}: {percent}%",
                                }
                                yield f"data: {json.dumps(progress_data)}\n\n"

                            elif status.startswith("pulling"):
                                # Layer pull in progress
                                yield f"data: {json.dumps({'type': 'progress', 'message': status})}\n\n"

                            elif status == "verifying sha256 digest":
                                verify_data = {
                                    "type": "progress",
                                    "percent": 100,
                                    "message": "Verifying model...",
                                }
                                yield f"data: {json.dumps(verify_data)}\n\n"

                            elif status == "success":
                                # Verify model actually installed before claiming success
                                try:
                                    async with httpx.AsyncClient(
                                        timeout=5.0
                                    ) as verify_client:
                                        verify_response = await verify_client.get(
                                            "http://localhost:11434/api/tags"
                                        )
                                        tags_data = verify_response.json()
                                        model_exists = any(
                                            m["name"] == model_name
                                            for m in tags_data.get("models", [])
                                        )

                                        if model_exists:
                                            logger.info(
                                                f"Model {model_name} downloaded successfully and verified"
                                            )
                                            complete_data = {
                                                "type": "complete",
                                                "success": True,
                                                "message": f"Successfully installed {model_name}",
                                            }
                                            yield f"data: {json.dumps(complete_data)}\n\n"
                                        else:
                                            logger.error(
                                                f"Model {model_name} reported success but not found in Ollama"
                                            )
                                            error_data = {
                                                "type": "error",
                                                "message": f"Download reported success but {model_name} not found in Ollama",
                                            }
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                except Exception as verify_error:
                                    logger.error(f"Verification failed: {verify_error}")
                                    error_data = {
                                        "type": "error",
                                        "message": f"Verification failed: {str(verify_error)}",
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse Ollama API response: {line[:100]}"
                            )
                            continue

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Error streaming model pull: {e}\n{error_details}")
            exception_data = {"type": "error", "message": f"Error: {str(e)}"}
            yield f"data: {json.dumps(exception_data)}\n\n"
        finally:
            # Always remove from downloading set and clean up cancel flag
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.delete("/uninstall/{model_name:path}")
async def uninstall_model(model_name: str, request: Request, provider_hint: str = None):
    """Uninstall/remove a model from Ollama or LM Studio

    Args:
        model_name: The model name from the UI (registry name like qwen2.5:7b)
        provider_hint: Optional hint from frontend about which provider tab user was viewing
    """
    try:
        # Decode the URL-encoded model name
        import urllib.parse

        model_name = urllib.parse.unquote(model_name)

        logger.info(
            f"Attempting to uninstall model: {model_name} (provider_hint: {provider_hint})"
        )

        # If provider_hint is given, use it to determine which provider to check first
        # For LM Studio, map registry name to actual model name
        provider = None
        actual_model_name = model_name

        if provider_hint == "lmstudio":
            # Map registry name to LM Studio model name
            model_info = MODEL_TO_HUGGINGFACE.get(model_name)
            # Check what's actually installed in LM Studio
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get("http://localhost:1234/v1/models")
                    if resp.status_code == 200:
                        lms_models = [m["id"] for m in resp.json().get("data", [])]

                        # v0.3.0: Use unified matching function
                        for lms_id in lms_models:
                            if _model_names_match(model_name, lms_id, model_info):
                                provider = "lmstudio"
                                actual_model_name = lms_id
                                logger.info(
                                    f"Mapped registry name {model_name} to LM Studio model {lms_id}"
                                )
                                break
            except Exception as e:
                logger.error(f"Failed to check LM Studio for {model_name}: {e}")

        elif provider_hint == "ollama":
            # For Ollama, registry name matches actual model name
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "http://localhost:11434/api/tags", timeout=2.0
                    )
                    if resp.status_code == 200:
                        ollama_models = [
                            m["name"] for m in resp.json().get("models", [])
                        ]
                        if model_name in ollama_models:
                            provider = "ollama"
                            actual_model_name = model_name
            except Exception as e:
                logger.error(f"Failed to check Ollama for {model_name}: {e}")

        # If provider_hint didn't find it, fall back to checking both providers
        if not provider:
            logger.info(f"Provider hint didn't find model, checking both providers...")
            # Reset actual_model_name to original since hint failed
            actual_model_name = model_name

            # Check Ollama models
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "http://localhost:11434/api/tags", timeout=2.0
                    )
                    if resp.status_code == 200:
                        ollama_models = [
                            m["name"] for m in resp.json().get("models", [])
                        ]
                        if model_name in ollama_models:
                            provider = "ollama"
                            actual_model_name = model_name
            except:
                pass

            # Check LM Studio models
            if not provider:
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            "http://localhost:1234/v1/models", timeout=2.0
                        )
                        if resp.status_code == 200:
                            lms_models = [m["id"] for m in resp.json().get("data", [])]

                            # Try direct match first
                            if model_name in lms_models:
                                provider = "lmstudio"
                                actual_model_name = model_name
                            else:
                                # v0.3.0: Use unified matching function
                                model_info = MODEL_TO_HUGGINGFACE.get(model_name)
                                for lms_id in lms_models:
                                    if _model_names_match(
                                        model_name, lms_id, model_info
                                    ):
                                        provider = "lmstudio"
                                        actual_model_name = lms_id
                                        logger.info(
                                            f"Mapped {model_name} to LM Studio ID: {lms_id}"
                                        )
                                        break
                except:
                    pass

        if not provider:
            logger.error(f"Model {model_name} not found in any provider")
            return {
                "success": False,
                "error": "Model not found",
                "message": f"Model {model_name} not found",
            }

        # Uninstall based on provider
        if provider == "ollama":
            # Validate model name format for Ollama
            if not _validate_model_name(actual_model_name):
                raise HTTPException(400, f"Invalid Ollama model name format")

            result = subprocess.run(
                ["ollama", "rm", actual_model_name],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=_SUBPROCESS_FLAGS,
            )
        elif provider == "lmstudio":
            # For LM Studio, find and delete the GGUF file
            # LM Studio model IDs look like: qwen2.5-7b-instruct, llama-3.1-8b-instruct, etc.
            # Files are stored in ~/.lmstudio/models/publisher/repo/file.gguf

            lms_models_dir = Path.home() / ".lmstudio" / "models"
            deleted = False

            # Use the actual LM Studio ID we mapped from registry name
            model_id_to_delete = actual_model_name
            logger.info(f"Searching for LM Studio model ID: {model_id_to_delete}")

            # Search for matching model directory based on the model we want to delete
            # For qwen2.5:7b -> qwen2.5-7b-instruct, we need to find the matching repo
            for publisher_dir in lms_models_dir.iterdir():
                if not publisher_dir.is_dir():
                    continue
                for repo_dir in publisher_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    # Check if any GGUF files exist in this repo
                    gguf_files = list(repo_dir.glob("*.gguf"))
                    if gguf_files:
                        # Match by checking if the model_id matches this repo's pattern
                        # For qwen2.5-7b-instruct -> Qwen2.5-7B-Instruct-GGUF repo
                        repo_name_lower = repo_dir.name.lower()
                        model_id_lower = model_id_to_delete.lower().replace(":", "-")

                        # Strategy 1: Direct substring match (qwen2.5-7b-instruct in qwen2.5-7b-instruct-gguf)
                        if (
                            model_id_lower in repo_name_lower
                            or repo_name_lower.replace("-gguf", "") == model_id_lower
                        ):
                            import shutil

                            shutil.rmtree(repo_dir)
                            logger.info(
                                f"Deleted LM Studio model directory: {repo_dir}"
                            )
                            deleted = True
                            break

                        # Strategy 2: Match by normalized parts (handle qwen2.5 correctly)
                        # Remove dots, then split and compare
                        model_normalized = model_id_lower.replace(
                            ".", ""
                        )  # qwen25-7b-instruct
                        repo_normalized = repo_name_lower.replace(
                            ".", ""
                        )  # qwen25-7b-instruct-gguf
                        if model_normalized in repo_normalized:
                            import shutil

                            shutil.rmtree(repo_dir)
                            logger.info(
                                f"Deleted LM Studio model directory: {repo_dir}"
                            )
                            deleted = True
                            break

                if deleted:
                    break

            if not deleted:
                logger.error(
                    f"LM Studio model {model_name} (ID: {model_id_to_delete}) not found on disk"
                )
                return {
                    "success": False,
                    "error": "Model files not found",
                    "message": f"Model {model_name} files not found",
                }

            # Unload model from LM Studio to refresh its cache
            try:
                lms_path = Path.home() / ".lmstudio" / "bin" / "lms.exe"
                if lms_path.exists():
                    unload_result = subprocess.run(
                        [str(lms_path), "unload", model_id_to_delete],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=_SUBPROCESS_FLAGS,
                    )
                    logger.info(
                        f"Unloaded {model_id_to_delete} from LM Studio: {unload_result.stdout}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to unload model from LM Studio (non-critical): {e}"
                )

            # Create a fake result object for unified handling below
            class Result:
                returncode = 0
                stdout = f"Deleted LM Studio model {model_name}"
                stderr = ""

            result = Result()

        if result.returncode == 0:
            logger.info(f"Successfully uninstalled model: {model_name}")

            # Check if this was the current model and switch to another if needed
            if hasattr(request.app.state, "llm_client"):
                current_model = getattr(request.app.state.llm_client, "model_name", "")
                if current_model == model_name:
                    # Get list of available models from BOTH providers
                    chat_models = []
                    embedding_models = [
                        "nomic-embed-text",
                        "mxbai-embed-large",
                        "all-minilm",
                    ]

                    # Try to find models from the SAME provider first
                    replacement_found = False

                    if provider == "ollama":
                        # Check Ollama for other models
                        try:
                            list_result = subprocess.run(
                                ["ollama", "list"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                                creationflags=_SUBPROCESS_FLAGS,
                            )
                            if list_result.returncode == 0:
                                lines = list_result.stdout.strip().split("\n")
                                for line in lines[1:]:  # Skip header
                                    parts = line.split()
                                    if parts:
                                        model_name_candidate = parts[0]
                                        is_embedding = any(
                                            embed in model_name_candidate.lower()
                                            for embed in embedding_models
                                        )
                                        if not is_embedding:
                                            chat_models.append(
                                                ("ollama", model_name_candidate)
                                            )
                        except Exception as e:
                            logger.error(f"Failed to check Ollama models: {e}")

                    elif provider == "lmstudio":
                        # Check LM Studio for other models
                        try:
                            async with httpx.AsyncClient(timeout=2.0) as client:
                                resp = await client.get(
                                    "http://localhost:1234/v1/models"
                                )
                                if resp.status_code == 200:
                                    lms_models = [
                                        m["id"] for m in resp.json().get("data", [])
                                    ]
                                    for lms_model in lms_models:
                                        is_embedding = any(
                                            embed in lms_model.lower()
                                            for embed in embedding_models
                                        )
                                        if not is_embedding:
                                            chat_models.append(("lmstudio", lms_model))
                        except Exception as e:
                            logger.error(f"Failed to check LM Studio models: {e}")

                    # If no models in same provider, check the other provider
                    if not chat_models:
                        if provider == "ollama":
                            # Try LM Studio
                            try:
                                async with httpx.AsyncClient(timeout=2.0) as client:
                                    resp = await client.get(
                                        "http://localhost:1234/v1/models"
                                    )
                                    if resp.status_code == 200:
                                        lms_models = [
                                            m["id"] for m in resp.json().get("data", [])
                                        ]
                                        for lms_model in lms_models:
                                            is_embedding = any(
                                                embed in lms_model.lower()
                                                for embed in embedding_models
                                            )
                                            if not is_embedding:
                                                chat_models.append(
                                                    ("lmstudio", lms_model)
                                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to check LM Studio as fallback: {e}"
                                )
                        else:
                            # Try Ollama
                            try:
                                list_result = subprocess.run(
                                    ["ollama", "list"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                    creationflags=_SUBPROCESS_FLAGS,
                                )
                                if list_result.returncode == 0:
                                    lines = list_result.stdout.strip().split("\n")
                                    for line in lines[1:]:
                                        parts = line.split()
                                        if parts:
                                            model_name_candidate = parts[0]
                                            is_embedding = any(
                                                embed in model_name_candidate.lower()
                                                for embed in embedding_models
                                            )
                                            if not is_embedding:
                                                chat_models.append(
                                                    ("ollama", model_name_candidate)
                                                )
                            except Exception as e:
                                logger.error(f"Failed to check Ollama as fallback: {e}")

                    if chat_models:
                        # Switch to first available chat model
                        new_provider, new_model = chat_models[0]

                        # Update llm_client with correct provider
                        if new_provider == "ollama":
                            request.app.state.llm_client.model_name = new_model
                            request.app.state.llm_client.base_url = (
                                "http://localhost:11434"
                            )
                            request.app.state.llm_client.api_style = "ollama"
                            os.environ["OLLAMA_MODEL"] = new_model
                            os.environ["ROAMPAL_LLM_OLLAMA_MODEL"] = new_model
                        else:
                            request.app.state.llm_client.model_name = new_model
                            request.app.state.llm_client.base_url = (
                                "http://localhost:1234"
                            )
                            request.app.state.llm_client.api_style = "openai"

                        # CRITICAL: Recycle HTTP client to use new base_url
                        if hasattr(request.app.state.llm_client, "_recycle_client"):
                            await request.app.state.llm_client._recycle_client()
                            logger.info(
                                f"Recycled HTTP client for new provider: {new_provider}"
                            )

                        # Update .env file with locking
                        async with _env_file_lock:
                            env_path = Path(__file__).parent.parent.parent / ".env"
                            if env_path.exists():
                                lines = env_path.read_text().splitlines()
                                for i, line in enumerate(lines):
                                    if line.startswith("OLLAMA_MODEL="):
                                        lines[i] = (
                                            f"OLLAMA_MODEL={new_model}"
                                            if new_provider == "ollama"
                                            else line
                                        )
                                    elif line.startswith("ROAMPAL_LLM_OLLAMA_MODEL="):
                                        lines[i] = (
                                            f"ROAMPAL_LLM_OLLAMA_MODEL={new_model}"
                                            if new_provider == "ollama"
                                            else line
                                        )
                                env_path.write_text("\n".join(lines) + "\n")

                        logger.info(
                            f"Switched from uninstalled {provider} model to {new_provider} model: {new_model}"
                        )
                    else:
                        # No chat models available - warn user
                        logger.warning(
                            "No chat models available after uninstall. User must install a chat model."
                        )
                        # Set to None to prevent embedding model usage
                        request.app.state.llm_client.model_name = None

            return {
                "success": True,
                "message": f"Successfully uninstalled {model_name}",
                "output": result.stdout,
            }
        else:
            logger.error(f"Failed to uninstall model: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr or "Unknown error occurred",
                "message": f"Failed to uninstall {model_name}",
            }
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while uninstalling model: {model_name}")
        return {
            "success": False,
            "error": "Operation timed out after 30 seconds",
            "message": f"Timeout uninstalling {model_name}",
        }
    except FileNotFoundError:
        logger.error("Ollama command not found")
        return {
            "success": False,
            "error": "Ollama is not installed or not in PATH",
            "message": "Ollama not found",
        }
    except Exception as e:
        logger.error(f"Error uninstalling model {model_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error uninstalling {model_name}",
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
        model_name = request_data.get("model", "")

        if not model_name:
            await websocket.send_json(
                {"type": "error", "message": "Model name is required"}
            )
            await websocket.close()
            return

        # Validate model name
        if not _validate_model_name(model_name):
            await websocket.send_json(
                {"type": "error", "message": "Invalid model name format"}
            )
            await websocket.close()
            return

        # Check if already downloading
        async with _download_lock:
            if model_name in _downloading_models:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Model {model_name} is already being downloaded",
                    }
                )
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
                    timeout=600.0,
                ) as response:
                    async for line in response.aiter_lines():
                        # Check if download was cancelled
                        if _cancel_flags.get(model_name, False):
                            logger.info(
                                f"WebSocket: Ollama download cancelled for {model_name}"
                            )
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "Download cancelled by user",
                                }
                            )
                            break

                        if line.strip():
                            try:
                                progress_data = json.loads(line)
                                status = progress_data.get("status", "")

                                if "error" in progress_data:
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "message": progress_data["error"],
                                        }
                                    )
                                    break

                                # Parse progress info
                                completed = progress_data.get("completed", 0)
                                total = progress_data.get("total", 0)

                                if total > 0:
                                    percent = int((completed / total) * 100)
                                    downloaded_mb = (
                                        f"{completed / (1024 * 1024):.1f} MB"
                                    )
                                    total_mb = f"{total / (1024 * 1024):.1f} MB"

                                    await websocket.send_json(
                                        {
                                            "type": "progress",
                                            "percent": percent,
                                            "downloaded": downloaded_mb,
                                            "total": total_mb,
                                            "message": f"{status}: {percent}%",
                                        }
                                    )
                                else:
                                    await websocket.send_json(
                                        {"type": "progress", "message": status}
                                    )

                                # Check if done
                                if (
                                    status == "success"
                                    or progress_data.get("status") == "success"
                                ):
                                    await websocket.send_json(
                                        {"type": "complete", "model": model_name}
                                    )
                                    logger.info(
                                        f"WebSocket: Successfully pulled {model_name}"
                                    )
                                    break

                            except json.JSONDecodeError:
                                continue

        finally:
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)

        await websocket.close()

    except WebSocketDisconnect:
        logger.info(f"WebSocket: Client disconnected during download of {model_name}")
        _downloading_models.discard(model_name)
        _cancel_flags.pop(model_name, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        _downloading_models.discard(model_name)
        _cancel_flags.pop(model_name, None)


@router.websocket("/download-gguf-ws")
async def download_gguf_websocket(websocket: WebSocket):
    """WebSocket endpoint for LM Studio GGUF downloads in Tauri production"""
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"WebSocket accept failed: {e}")
        return

    model_name = None
    try:
        # Receive model name from client
        data = await websocket.receive_text()
        request_data = json.loads(data)
        model_name = request_data.get("model", "")

        if not model_name:
            await websocket.send_json(
                {"type": "error", "message": "Model name is required"}
            )
            await websocket.close()
            return

        # Validate model exists in mapping
        if model_name not in MODEL_TO_HUGGINGFACE:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Model {model_name} not available for LM Studio auto-download",
                }
            )
            await websocket.close()
            return

        model_info = MODEL_TO_HUGGINGFACE[model_name]

        # Check concurrency - use lock to prevent race conditions
        async with _download_lock:
            if model_name in _downloading_models:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Model {model_name} is already being downloaded",
                    }
                )
                await websocket.close()
                return

            if _downloading_models:
                current = next(iter(_downloading_models))
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Cannot download {model_name}: {current} is downloading",
                    }
                )
                await websocket.close()
                return

            _downloading_models.add(model_name)

        try:
            # 1. Download GGUF file from HuggingFace
            cache_path = get_lmstudio_cache_path() / model_info["file"]
            hf_url = f"https://huggingface.co/{model_info['repo']}/resolve/main/{model_info['file']}"

            logger.info(f"WebSocket: Downloading {model_name} from {hf_url}")
            await websocket.send_json({"type": "start", "model": model_name})
            await websocket.send_json(
                {
                    "type": "progress",
                    "status": "downloading",
                    "percent": 0,
                    "message": "Starting download...",
                }
            )

            async with httpx.AsyncClient(
                timeout=600.0, follow_redirects=True
            ) as client:
                async with client.stream("GET", hf_url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    last_progress = 0

                    with open(cache_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            # Check if download was cancelled
                            if _cancel_flags.get(model_name, False):
                                logger.info(
                                    f"WebSocket: LM Studio download cancelled for {model_name}"
                                )
                                await websocket.send_json(
                                    {
                                        "type": "error",
                                        "message": "Download cancelled by user",
                                    }
                                )
                                break

                            f.write(chunk)
                            downloaded += len(chunk)

                            if total > 0:
                                percent = int((downloaded / total) * 100)
                                if percent >= last_progress + 1 or percent == 100:
                                    downloaded_str = format_bytes(downloaded)
                                    total_str = format_bytes(total)

                                    await websocket.send_json(
                                        {
                                            "type": "progress",
                                            "status": "downloading",
                                            "percent": percent,
                                            "downloaded": downloaded_str,
                                            "total": total_str,
                                            "message": f"Downloading {percent}%",
                                        }
                                    )
                                    last_progress = percent

            logger.info(f"WebSocket: Downloaded {model_name} to {cache_path}")

            # 2. Import via lms.exe
            await websocket.send_json(
                {
                    "type": "progress",
                    "status": "importing",
                    "percent": 100,
                    "message": "Importing to LM Studio...",
                }
            )

            lms_cli = get_lms_cli_path()
            if not lms_cli:
                raise Exception("lms.exe not found. Is LM Studio installed?")

            logger.info(f"WebSocket: Importing with lms.exe: {lms_cli}")

            # Use Popen with stdin PIPE to answer first-run prompt
            process = subprocess.Popen(
                [
                    str(lms_cli),
                    "import",
                    "--yes",
                    "--copy",
                    "--user-repo",
                    model_info["repo"],
                    str(cache_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=_SUBPROCESS_FLAGS,
            )

            stdout, stderr = process.communicate(input="Y\n", timeout=300)

            # Create result object to match expected interface
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            result = Result(process.returncode, stdout, stderr)

            # lms import sends warnings to stderr even on success, so check if file was actually imported
            # LM Studio copies files to ~/.lmstudio/models/user/repo/
            imported_file_path = (
                Path.home()
                / ".lmstudio"
                / "models"
                / model_info["repo"].split("/")[0]
                / model_info["repo"].split("/")[1]
                / model_info["file"]
            )

            if not imported_file_path.exists():
                logger.error(
                    f"lms import failed - file not found at expected location: {imported_file_path}"
                )
                logger.error(f"lms stdout: {result.stdout}")
                logger.error(f"lms stderr: {result.stderr}")
                raise Exception(
                    f"Import failed: Model file not found in LM Studio models directory"
                )

            logger.info(
                f"WebSocket: Successfully imported {model_name} to {imported_file_path}"
            )

            # Delete cache file after successful import to avoid duplicating disk usage
            try:
                cache_path.unlink()
                logger.info(f"Deleted cache file: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not delete cache file {cache_path}: {e}")

            # Model is now available - LM Studio will auto-load on first use
            logger.info(
                f"WebSocket: Successfully imported {model_name} - ready for use"
            )
            await websocket.send_json(
                {"type": "loaded", "model": model_name, "message": "Model ready!"}
            )

        finally:
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)

        await websocket.close()

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket: Client disconnected during LM Studio download of {model_name}"
        )
        if model_name:
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)
    except Exception as e:
        logger.error(f"WebSocket LM Studio error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        if model_name:
            _downloading_models.discard(model_name)
            _cancel_flags.pop(model_name, None)


# =============================================================================
# v0.3.0: Ghost Model Detection & Cleanup
# =============================================================================


@router.get("/ghost-models")
async def get_ghost_models():
    """
    Detect orphaned/ghost models - models that appear in LM Studio but have
    no backing file on disk, or partial downloads that weren't cleaned up.
    """
    ghost_models = []

    try:
        # Get models LM Studio thinks are installed
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:1234/v1/models")
            if response.status_code != 200:
                return {"ghost_models": [], "error": "LM Studio not running"}

            data = response.json()
            lmstudio_models = [model.get("id", "") for model in data.get("data", [])]

        # Check each model for actual file existence
        lmstudio_models_dir = Path.home() / ".lmstudio" / "models"
        cache_path = get_lmstudio_cache_path()

        for model_id in lmstudio_models:
            is_ghost = False
            reason = ""

            # Check if model directory exists in LM Studio models folder
            # Model IDs are like "user/repo/filename" or "filename"
            model_path = lmstudio_models_dir / model_id

            # Also check for GGUF file directly
            if not model_path.exists():
                # Try to find a matching .gguf file
                found = False
                for gguf_file in lmstudio_models_dir.rglob("*.gguf"):
                    if model_id.lower() in gguf_file.name.lower():
                        found = True
                        break

                if not found:
                    is_ghost = True
                    reason = "No backing GGUF file found"

            if is_ghost:
                ghost_models.append({"model_id": model_id, "reason": reason})

        # Also check for orphaned cache files (partial downloads)
        if cache_path.exists():
            for cache_file in cache_path.glob("*.gguf"):
                # If there's a .gguf in cache but not in models, it's orphaned
                ghost_models.append(
                    {
                        "model_id": f"[cache] {cache_file.name}",
                        "reason": "Partial download in cache",
                        "path": str(cache_file),
                        "size_mb": round(cache_file.stat().st_size / (1024 * 1024), 2),
                    }
                )

        return {"ghost_models": ghost_models, "count": len(ghost_models)}

    except Exception as e:
        logger.error(f"Error detecting ghost models: {e}")
        return {"ghost_models": [], "error": str(e)}


@router.delete("/ghost-models/{model_id:path}")
async def cleanup_ghost_model(model_id: str):
    """
    Remove a ghost/orphaned model. For cache files, deletes the file.
    For LM Studio registered models, removes from registry if possible.
    """
    try:
        # Handle cache files
        if model_id.startswith("[cache] "):
            filename = model_id.replace("[cache] ", "")
            cache_path = get_lmstudio_cache_path() / filename

            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Deleted ghost cache file: {cache_path}")
                return {"status": "deleted", "model_id": model_id}
            else:
                return {"status": "not_found", "model_id": model_id}

        # For LM Studio models, try to find and delete the file
        lmstudio_models_dir = Path.home() / ".lmstudio" / "models"

        # Try direct path
        model_path = lmstudio_models_dir / model_id
        if model_path.exists():
            if model_path.is_file():
                model_path.unlink()
            else:
                import shutil

                shutil.rmtree(model_path)
            logger.info(f"Deleted ghost model: {model_path}")
            return {"status": "deleted", "model_id": model_id}

        # Try to find matching .gguf file
        for gguf_file in lmstudio_models_dir.rglob("*.gguf"):
            if (
                model_id.lower() in gguf_file.name.lower()
                or model_id.lower() in str(gguf_file).lower()
            ):
                gguf_file.unlink()
                logger.info(f"Deleted ghost model file: {gguf_file}")
                return {
                    "status": "deleted",
                    "model_id": model_id,
                    "path": str(gguf_file),
                }

        return {"status": "not_found", "model_id": model_id}

    except Exception as e:
        logger.error(f"Error cleaning up ghost model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# Sidecar Model Management (v0.3.1)
# ===========================================================================


class SidecarSetRequest(BaseModel):
    model: str
    provider: str = "ollama"  # "ollama" or "lmstudio"
    # v0.3.2 (0f): Explicit override — when set, sidecar stops mirroring chat.
    mirror_chat: bool = False


class SidecarMirrorRequest(BaseModel):
    """v0.3.2 (0f): Toggle sidecar mirror-chat mode from the Advanced settings panel."""
    mirror_chat: bool


def _get_sidecar_config_path(request: Request) -> Path:
    """Get path to persisted sidecar config."""
    from config.settings import settings

    return Path(settings.paths.data_dir) / "sidecar_config.json"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """v0.3.2: Atomic JSON write — temp file + os.replace.

    A plain `write_text(json.dumps(data))` can leave a truncated/corrupt file
    if the process crashes mid-write. This helper writes to a sibling `.tmp`
    file and then renames it, which is atomic on both POSIX and NTFS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _get_main_model_config_path(data_dir: Optional[Path] = None) -> Path:
    """Return the path to main_model_config.json under DATA_PATH."""
    if data_dir is None:
        from config.settings import settings as _settings
        data_dir = Path(_settings.paths.data_dir)
    return Path(data_dir) / "main_model_config.json"


def load_main_model_config(data_dir: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """Read {"model": ..., "provider": ...} from main_model_config.json, or None."""
    path = _get_main_model_config_path(data_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("provider"):
            return {"model": data.get("model") or "", "provider": data["provider"]}
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
    return None


def save_main_model_config(model: str, provider: str, data_dir: Optional[Path] = None) -> None:
    """Persist the user's main-model selection atomically."""
    path = _get_main_model_config_path(data_dir)
    _atomic_write_json(path, {"model": model, "provider": provider})
    logger.info(f"[MAIN MODEL] Persisted selection to {path}: {provider}/{model}")


def _load_sidecar_config(request: Request) -> Dict[str, Any]:
    """Load persisted sidecar config.

    v0.3.2 migration: if the persisted config predates `mirror_chat`, infer
    it by comparing the stored sidecar model against the live chat model.
    Matching → True (safe to mirror). Differing → False (preserve the v0.3.1
    user's explicit intent to run two different models).
    """
    path = _get_sidecar_config_path(request)
    defaults = {"enabled": False, "model": "", "provider": "", "mirror_chat": True}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in defaults.items():
                if k == "mirror_chat" and k not in data:
                    main_client = getattr(getattr(request, "app", None), "state", None)
                    chat_model = getattr(getattr(main_client, "llm_client", None), "model_name", "") if main_client else ""
                    data["mirror_chat"] = bool(chat_model) and (data.get("model") == chat_model)
                else:
                    data.setdefault(k, v)
            return data
        except Exception:
            pass
    return defaults


def _save_sidecar_config(request: Request, config: Dict[str, Any]):
    """Persist sidecar config to disk (atomic write — crash-safe)."""
    path = _get_sidecar_config_path(request)
    _atomic_write_json(path, config)


@router.get("/sidecar/status")
async def sidecar_status(request: Request):
    """Get current sidecar model status."""
    model = getattr(request.app.state, "sidecar_model", "")
    provider = getattr(request.app.state, "sidecar_provider", "")
    last_error = getattr(request.app.state, "sidecar_last_error", "")
    # v0.3.2 (0f): surface mirror_chat so the chat-header badge can render
    # "Mirror: <model>" vs an explicit sidecar selection.
    try:
        persisted = _load_sidecar_config(request)
        mirror_chat = bool(persisted.get("mirror_chat", True))
    except Exception:
        mirror_chat = True
    return {
        "enabled": bool(model),
        "model": model,
        "provider": provider,
        "last_error": last_error,
        "mirror_chat": mirror_chat,
    }


@router.post("/sidecar/set")
async def sidecar_set(request: Request, body: SidecarSetRequest):
    """Set sidecar model. Creates a dedicated client for sidecar LLM calls."""
    from modules.llm.ollama_client import OllamaClient
    import httpx

    model = body.model
    provider = body.provider

    # Note: Same model allowed - will use shared client to avoid GPU duplication
    # Sidecar will queue behind main LLM if same model/provider

    # Determine base URL
    if provider == "lmstudio":
        base_url = "http://localhost:1234"
        api_style = "openai"
    else:
        base_url = "http://localhost:11434"
        api_style = "ollama"

    # v0.3.1.3: Validate model exists on provider, fuzzy-match if needed
    try:
        async with httpx.AsyncClient(timeout=5) as http:
            if provider == "lmstudio":
                resp = await http.get(f"{base_url}/v1/models")
                available = [m["id"] for m in resp.json().get("data", [])]
            else:
                resp = await http.get(f"{base_url}/api/tags")
                available = [m["name"] for m in resp.json().get("models", [])]

        if model not in available:
            # Fuzzy match: qwen2.5:7b → qwen2.5-7b-instruct, or vice versa
            base = model.split(":")[0].replace("-", "").replace(".", "").lower()
            matched = None
            for name in available:
                normalized = name.split(":")[0].replace("-", "").replace(".", "").lower()
                if base in normalized or normalized in base:
                    matched = name
                    break
            if matched:
                logger.info(f"[SIDECAR] Model '{model}' not found on {provider}, using '{matched}'")
                model = matched
            else:
                logger.warning(f"[SIDECAR] Model '{model}' not found on {provider}. Available: {available}")
    except Exception as e:
        logger.debug(f"[SIDECAR] Could not validate model on {provider}: {e}")

    # Check if we can share main LLM client (same provider and model)
    can_share_client = False
    if hasattr(request.app.state, "llm_client") and request.app.state.llm_client:
        main_client = request.app.state.llm_client
        # Check if same provider (base URL) and model
        main_base_url = getattr(main_client, "base_url", None)
        main_model = getattr(main_client, "model_name", None)

        # Extract base URL from main client for comparison
        if hasattr(main_client, "_client") and main_client._client:
            main_actual_base = str(main_client._client.base_url)
            if "localhost:11434" in main_actual_base and provider == "ollama":
                can_share_client = True
            elif "localhost:1234" in main_actual_base and provider == "lmstudio":
                can_share_client = True
            else:
                can_share_client = False

        if can_share_client and main_model == model:
            logger.info(f"[SIDECAR] Sharing main LLM client for {model} ({provider})")
            sidecar_client = main_client
        else:
            can_share_client = False

    if not can_share_client:
        # Create dedicated sidecar client
        sidecar_client = OllamaClient()
        await sidecar_client.initialize(
            {
                "ollama_base_url": base_url,
                "ollama_model": model,
            }
        )
        sidecar_client.api_style = api_style

    # Store in app state
    request.app.state.sidecar_model = model
    request.app.state.sidecar_provider = provider
    request.app.state.sidecar_client = sidecar_client

    # Persist — honor the caller's explicit mirror_chat flag (default False
    # since picking a specific sidecar is an override of the mirror default).
    _save_sidecar_config(
        request,
        {
            "enabled": True,
            "model": model,
            "provider": provider,
            "mirror_chat": bool(getattr(body, "mirror_chat", False)),
        },
    )

    request.app.state.sidecar_last_error = ""  # Clear any prior error
    logger.info(f"[SIDECAR] Set model: {model} ({provider})")
    return {"status": "ok", "model": model, "provider": provider}


@router.post("/sidecar/mirror")
async def sidecar_mirror(request: Request, body: SidecarMirrorRequest):
    """v0.3.2 (0f): Toggle sidecar mirror-chat mode.

    When enabled, the sidecar is kept in lockstep with the chat model so
    single-GPU users don't pay a 10-30 s model swap on every sidecar call.
    Turning it off leaves the current sidecar selection in place as an
    independent model — the Advanced settings picker can then change it.
    """
    config = _load_sidecar_config(request)
    config["mirror_chat"] = bool(body.mirror_chat)

    if body.mirror_chat:
        # Snap to chat model immediately.
        main_client = getattr(request.app.state, "llm_client", None)
        if main_client and getattr(main_client, "model_name", ""):
            config["enabled"] = True
            config["model"] = main_client.model_name
            config["provider"] = getattr(main_client, "api_style", "ollama")
            if config["provider"] == "openai":
                config["provider"] = "lmstudio"
            request.app.state.sidecar_model = config["model"]
            request.app.state.sidecar_provider = config["provider"]
            request.app.state.sidecar_client = main_client

    _save_sidecar_config(request, config)
    return {"status": "ok", "mirror_chat": config["mirror_chat"], "model": config.get("model", "")}


@router.post("/sidecar/disable")
async def sidecar_disable(request: Request):
    """Disable sidecar processing."""
    request.app.state.sidecar_model = ""
    request.app.state.sidecar_provider = ""
    request.app.state.sidecar_client = None

    _save_sidecar_config(request, {"enabled": False, "model": "", "provider": ""})

    logger.info("[SIDECAR] Disabled")
    return {"status": "ok", "enabled": False}
