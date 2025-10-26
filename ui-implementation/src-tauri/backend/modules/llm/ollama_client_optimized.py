# C:\RoampalAI\backend\modules\llm\ollama_client_optimized.py
"""
Optimized Ollama Client with enhanced connection pooling and performance features
"""
import httpx
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
import time

from core.interfaces.llm_client_interface import LLMClientInterface
from config.settings import settings

logger = logging.getLogger(__name__)

class OllamaException(Exception):
    pass

class OllamaClientOptimized(LLMClientInterface):
    """
    Optimized Ollama client with:
    - Enhanced connection pooling
    - Request batching support
    - Performance metrics tracking
    - Automatic retry with exponential backoff
    """
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url: str = ""
        self.model_name: str = ""
        self.request_timeout: int = settings.llm.ollama_request_timeout_seconds
        self.keep_alive_seconds: int = settings.llm.ollama_keep_alive_seconds
        
        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Connection pool configuration
        self.pool_config = {
            "max_keepalive_connections": 20,  # Increased from default
            "max_connections": 50,  # Increased pool size
            "keepalive_expiry": 600  # 10 minutes keepalive
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        self.base_url = config.get("ollama_base_url", settings.llm.ollama_base_url)
        self.model_name = config.get("ollama_model", settings.llm.ollama_model)
        self.request_timeout = config.get("ollama_request_timeout_seconds", self.request_timeout)
        self.keep_alive_seconds = config.get("ollama_keep_alive_seconds", self.keep_alive_seconds)

        if not self.model_name:
            raise ValueError("Ollama 'ollama_model' name must be provided in the configuration.")

        try:
            # Enhanced connection pooling configuration
            limits = httpx.Limits(
                max_keepalive_connections=self.pool_config["max_keepalive_connections"],
                max_connections=self.pool_config["max_connections"],
                keepalive_expiry=self.pool_config["keepalive_expiry"]
            )
            
            # Optimized timeout configuration (adjusted for various model sizes)
            timeout_config = httpx.Timeout(
                timeout=self.request_timeout,
                connect=30.0,  # Increased for initial model loading
                read=self.request_timeout,
                write=30.0,
                pool=10.0  # Increased pool acquisition timeout
            )
            
            # Create client with optimized settings
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout_config,
                limits=limits,
                http2=True,  # Enable HTTP/2 for multiplexing
                follow_redirects=True,
                headers={
                    "Connection": "keep-alive",
                    "Keep-Alive": f"timeout={self.pool_config['keepalive_expiry']}"
                }
            )
            
            # Warm up the connection pool
            await self._warmup_connection_pool()
            
            logger.info(f"OllamaClientOptimized initialized with enhanced pooling: {self.pool_config}")
            
        except Exception as e:
            raise OllamaException(f"Failed to initialize optimized Ollama client: {e}") from e

    async def _warmup_connection_pool(self):
        """Pre-establish connections to reduce first-request latency"""
        try:
            # Make a lightweight request to establish connections
            warmup_tasks = []
            for _ in range(min(3, self.pool_config["max_keepalive_connections"])):
                warmup_tasks.append(self._health_check())
            
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            logger.info("Connection pool warmed up successfully")
        except Exception as e:
            logger.warning(f"Connection pool warmup failed (non-critical): {e}")

    async def _health_check(self) -> bool:
        """Quick health check for the Ollama service"""
        try:
            response = await self.client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except:
            return False

    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        format: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """Generate response with automatic retry and performance tracking"""
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for item in history:
                messages.append(item)
        messages.append({"role": "user", "content": prompt})

        actual_model = model or self.model_name
        logger.critical(f"[MODEL DEBUG] LLM Client - Requested model: {model}, Default model: {self.model_name}, Using: {actual_model}")
        
        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": False,
            "keep_alive": f"{self.keep_alive_seconds}s",
            "options": {
                "num_ctx": 4096,  # Context window
                "num_predict": 2048,  # Max tokens to generate
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        if format:
            payload["format"] = format

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                api_response = await self.client.post("/api/chat", json=payload)
                api_response.raise_for_status()
                response_data = api_response.json()
                
                # Track performance metrics
                latency = time.time() - start_time
                self.request_count += 1
                self.total_latency += latency
                
                if self.request_count % 100 == 0:
                    avg_latency = self.total_latency / self.request_count
                    logger.info(f"Ollama performance: {self.request_count} requests, avg latency: {avg_latency:.2f}s, errors: {self.error_count}")

                if "message" in response_data and "content" in response_data["message"]:
                    content = response_data["message"]["content"]
                    # Strip <think> tags if present (some models like qwen3 wrap their thinking in these)
                    import re
                    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
                    # Also handle case where entire response is wrapped in tags
                    content = content.strip()
                    return content
                elif "error" in response_data:
                    raise OllamaException(f"Ollama API error: {response_data['error']}")
                else:
                    raise OllamaException(f"Unexpected response: {response_data}")
                    
            except httpx.TimeoutException as e:
                self.error_count += 1
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Ollama timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise OllamaException(f"Ollama request timed out after {max_retries} attempts") from e
                    
            except httpx.HTTPStatusError as e:
                self.error_count += 1
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Ollama service unavailable (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise OllamaException(f"Ollama HTTP error: {e}") from e
                    
            except Exception as e:
                self.error_count += 1
                raise OllamaException(f"Ollama generate_response failed: {e}") from e

    async def stream_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response with connection reuse"""
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        import re  # Import for think tag stripping

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for item in history:
                messages.append(item)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": f"{self.keep_alive_seconds}s"
        }

        try:
            start_time = time.time()
            accumulated_content = ""  # Accumulate content to strip think tags from complete segments
            in_think_tag = False

            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                if chunk:
                                    accumulated_content += chunk

                                    # Check for think tags and yield content outside them
                                    while True:
                                        if in_think_tag:
                                            # Look for closing tag
                                            end_idx = accumulated_content.find('</think>')
                                            if end_idx != -1:
                                                # Skip past the think content
                                                accumulated_content = accumulated_content[end_idx + 8:]
                                                in_think_tag = False
                                            else:
                                                break  # Wait for more content
                                        else:
                                            # Look for opening tag
                                            start_idx = accumulated_content.find('<think>')
                                            if start_idx != -1:
                                                # Yield content before the tag
                                                if start_idx > 0:
                                                    yield accumulated_content[:start_idx]
                                                accumulated_content = accumulated_content[start_idx + 7:]
                                                in_think_tag = True
                                            else:
                                                # No think tags found, yield all but keep some buffer for potential tag
                                                if len(accumulated_content) > 20:
                                                    to_yield = accumulated_content[:-20]
                                                    accumulated_content = accumulated_content[-20:]
                                                    if to_yield:
                                                        yield to_yield
                                                break

                            if data.get("done", False):
                                # Yield any remaining content
                                if accumulated_content and not in_think_tag:
                                    # Final cleanup - strip any remaining think tags
                                    accumulated_content = re.sub(r'<think>.*?</think>\s*', '', accumulated_content, flags=re.DOTALL)
                                    if accumulated_content.strip():
                                        yield accumulated_content
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming response: {line}")
                            continue
            
            # Track streaming performance
            latency = time.time() - start_time
            self.request_count += 1
            self.total_latency += latency
            
        except Exception as e:
            self.error_count += 1
            raise OllamaException(f"Ollama stream_response failed: {e}") from e

    async def close(self) -> None:
        """Close the client and cleanup resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info(f"OllamaClientOptimized closed. Total requests: {self.request_count}, Avg latency: {self.total_latency/max(1, self.request_count):.2f}s")

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "average_latency": self.total_latency / max(1, self.request_count),
            "error_rate": self.error_count / max(1, self.request_count) * 100,
            "pool_config": self.pool_config
        }