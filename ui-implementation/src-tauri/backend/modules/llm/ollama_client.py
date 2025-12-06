# C:\RoampalAI\backend\modules\llm\ollama_client.py
import httpx
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Optional

from core.interfaces.llm_client_interface import LLMClientInterface
from config.settings import settings

logger = logging.getLogger(__name__)
logger.info("[OLLAMA MODULE LOADED] This module has the temporary fix to use /api/generate for all models")

class OllamaException(Exception):
    pass

class OllamaClient(LLMClientInterface):
    # Class-level cache for model capabilities (shared across instances)
    _model_capabilities = {}

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url: str = settings.llm.ollama_base_url
        self.model_name: str = settings.llm.ollama_model  # Use default from settings
        self.api_style: str = "ollama"  # NEW: API style (ollama or openai)
        self.request_timeout: int = settings.llm.ollama_request_timeout_seconds
        self.keep_alive_seconds: int = settings.llm.ollama_keep_alive_seconds
        # Track request count for connection recycling
        self._request_count = 0
        self._max_requests_per_client = 10  # Recycle client after N requests

    async def initialize(self, config: Dict[str, Any]) -> None:
        self.base_url = config.get("ollama_base_url", settings.llm.ollama_base_url)
        self.model_name = config.get("ollama_model", settings.llm.ollama_model)
        self.request_timeout = config.get("ollama_request_timeout_seconds", self.request_timeout)
        self.keep_alive_seconds = config.get("ollama_keep_alive_seconds", self.keep_alive_seconds)

        if not self.model_name:
            raise ValueError("Ollama 'ollama_model' name must be provided in the configuration.")

        try:
            # Standard httpx client without special limits - deepseek-r1 doesn't work with conservative settings
            timeout_config = httpx.Timeout(
                self.request_timeout,
                connect=30,
                read=self.request_timeout,
                write=30
            )
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout_config,
                # Use default limits - no special configuration needed
            )
        except Exception as e:
            raise OllamaException(f"Failed to initialize httpx.AsyncClient for Ollama: {e}") from e

    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        format: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        # Recycle client connection periodically to avoid stale connections
        self._request_count += 1
        if self._request_count >= self._max_requests_per_client:
            logger.info(f"Recycling HTTP client after {self._request_count} requests")
            await self._recycle_client()
            self._request_count = 0

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for item in history:
                messages.append(item)
        messages.append({"role": "user", "content": prompt})

        actual_model = model or self.model_name
        logger.debug(f"Using model: {actual_model} (requested: {model}, default: {self.model_name})")

        # ==================== MULTI-PROVIDER API ROUTING ====================
        if self.api_style == "openai":
            # OpenAI-compatible providers (LM Studio, llama.cpp, etc.)
            try:
                logger.debug(f"Using OpenAI-style API at {self.base_url}/v1/chat/completions")

                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": actual_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4000,
                        "stream": False
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                data = response.json()

                return data['choices'][0]['message']['content']

            except Exception as e:
                logger.error(f"OpenAI-style API error: {e}")
                raise OllamaException(f"OpenAI API error: {str(e)}")
        # ==================== END MULTI-PROVIDER ROUTING ====================

        # DEBUG: Log the exact payload being sent
        logger.info(f"[DEBUG PAYLOAD] Messages count: {len(messages)}")
        for i, msg in enumerate(messages):
            logger.info(f"[DEBUG PAYLOAD] Message {i}: role={msg.get('role')}, content_length={len(str(msg.get('content', '')))}")
            if i == len(messages) - 1:  # Log last message content
                logger.info(f"[DEBUG PAYLOAD] Last message content: {str(msg.get('content', ''))[:200]}")

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": False,
            "think": False  # Disable thinking mode (qwen3, etc.) - much faster responses
        }

        # Initialize options to avoid UnboundLocalError in fallback path
        options = {}

        # Use centralized context configuration - no more duplicates!
        from config.model_contexts import get_context_size
        num_ctx = get_context_size(actual_model)
        options["num_ctx"] = num_ctx
        options["num_gpu"] = 99  # Force max GPU layers - Ollama is too conservative by default

        # Universal generation params - apply for all models from configuration
        from config.model_limits import get_generation_params

        gen_params = get_generation_params(actual_model)

        # Apply generation parameters if available
        if gen_params:
            if gen_params.get("num_predict"):
                options["num_predict"] = gen_params["num_predict"]
            if gen_params.get("temperature") is not None:
                options["temperature"] = gen_params["temperature"]
            if gen_params.get("repeat_penalty"):
                options["repeat_penalty"] = gen_params["repeat_penalty"]
            if gen_params.get("stop"):
                options["stop"] = gen_params["stop"]

        # Always apply options (at minimum we have num_ctx)
        payload["options"] = options
        logger.info(f"[GENERATION] Applied params for {actual_model}: {options}")

        if format:
            payload["format"] = format

        api_response = None
        use_generate = False

        try:
            # Model-specific endpoint selection (modular approach)
            force_generate_models = [
                # Models that work better with generate endpoint
                # Remove deepseek-r1 since it's unreliable with both endpoints
            ]

            # Check if model requires generate endpoint
            for model_pattern in force_generate_models:
                if model_pattern in actual_model.lower():
                    logger.debug(f"Model {actual_model} requires generate endpoint")
                    use_generate = True
                    break

            if not use_generate:
                # Try chat endpoint first (preferred for modern models)
                logger.debug(f"Trying /api/chat endpoint for model: {actual_model}")
                try:
                    # Add retry logic for transient failures
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            api_response = await self.client.post("/api/chat", json=payload)
                            # Check status before raising to handle 404 gracefully
                            if api_response.status_code == 404:
                                logger.warning(f"Chat endpoint returned 404 for model: {actual_model}, falling back to generate")
                                use_generate = True
                                break
                            else:
                                api_response.raise_for_status()
                                logger.debug(f"Chat endpoint successful for model: {actual_model}")
                                # Cache that this model works with chat
                                self._model_capabilities[actual_model] = 'chat'
                                break
                        except httpx.TimeoutException as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                                await asyncio.sleep(1)  # Brief pause before retry
                            else:
                                raise
                except Exception as chat_error:
                    logger.error(f"[CHAT ENDPOINT ERROR] Chat failed for {actual_model}: Type={type(chat_error).__name__}, Error={chat_error}")
                    logger.error(f"[CHAT ENDPOINT ERROR] Full traceback:", exc_info=True)
                    logger.warning(f"Chat endpoint failed for {actual_model}: {chat_error}, falling back to generate")
                    use_generate = True
                    # Cache that this model needs generate
                    self._model_capabilities[actual_model] = 'generate'
            
            # If chat failed or model needs generate endpoint, use generate
            if use_generate:
                logger.warning(f"Using generate endpoint for model {actual_model}")
                
                # Convert messages format to prompt for generate endpoint
                prompt_text = ""
                images = []
                for msg in messages:
                    if msg["role"] == "system":
                        prompt_text += f"{msg['content']}\n\n"
                    elif msg["role"] == "user":
                        # Check for images in user messages
                        if isinstance(msg.get("content"), list):
                            for item in msg["content"]:
                                if item.get("type") == "text":
                                    prompt_text += f"{item['text']}\n\n"
                                elif item.get("type") == "image_url":
                                    # Extract base64 image
                                    image_data = item["image_url"]["url"]
                                    if image_data.startswith("data:image"):
                                        # Remove data URL prefix
                                        base64_data = image_data.split(",")[1] if "," in image_data else image_data
                                        images.append(base64_data)
                        else:
                            prompt_text += f"{msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        prompt_text += f"{msg['content']}\n\n"
                
                # For deepseek-r1 and other models, don't add role labels
                # Just use the prompt as-is since it's already properly formatted
                if not prompt_text:
                    prompt_text = prompt  # Use original prompt if conversion failed

                # Ensure prompt_text is never empty
                if not prompt_text or not prompt_text.strip():
                    logger.warning(f"Empty prompt detected, using fallback: {prompt[:100]}")
                    prompt_text = prompt if prompt else "Hello"
                
                generate_payload = {
                    "model": actual_model,
                    "prompt": prompt_text,
                    "stream": False
                }

                # Apply model-specific options
                if options:
                    generate_payload["options"] = options

                if images:
                    generate_payload["images"] = images
                if format:
                    generate_payload["format"] = format

                # Log the actual prompt for debugging
                logger.critical(f"[GENERATE FALLBACK] Trying /api/generate with model={actual_model}, prompt length={len(prompt_text)}")
                logger.critical(f"[GENERATE PROMPT] First 200 chars: {prompt_text[:200]}")

                # WINDOWS FIX: Use synchronous requests to avoid async issues
                import requests
                # asyncio already imported at module level
                try:
                    # Run sync request in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.post(
                            f"{self.base_url}/api/generate",
                            json=generate_payload,
                            timeout=self.request_timeout
                        )
                    )
                    response.raise_for_status()

                    # DEBUG: Log the raw response to understand what's happening
                    response_json = response.json()
                    logger.critical(f"[DEBUG] Raw response keys: {response_json.keys() if response_json else 'None'}")
                    logger.critical(f"[DEBUG] Response 'response' field: {response_json.get('response', 'MISSING')[:200] if response_json else 'No JSON'}")

                    # Create a mock httpx response object for compatibility
                    class MockResponse:
                        def __init__(self, requests_response):
                            self.status_code = requests_response.status_code
                            self._json_data = requests_response.json()

                        def json(self):
                            return self._json_data

                        def raise_for_status(self):
                            pass

                    api_response = MockResponse(response)
                    logger.critical(f"[GENERATE SUCCESS - SYNC] Generate endpoint worked for model: {actual_model}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"[GENERATE FAIL - SYNC] Sync request failed: {e}")
                    # Fall back to original async method
                    api_response = await self.client.post("/api/generate", json=generate_payload)
                    api_response.raise_for_status()
                    logger.critical(f"[GENERATE SUCCESS - ASYNC] Generate endpoint worked for model: {actual_model}")
            
            if not api_response:
                raise OllamaException(f"No valid response from either /api/chat or /api/generate for model {actual_model}")
            
            response_data = api_response.json()

            # DEBUG: Log the full response structure
            logger.info(f"[DEBUG RESPONSE] Response keys: {response_data.keys()}")
            if "message" in response_data:
                logger.info(f"[DEBUG RESPONSE] Message keys: {response_data['message'].keys() if response_data['message'] else 'None'}")
                logger.info(f"[DEBUG RESPONSE] Message content type: {type(response_data['message'].get('content')) if response_data['message'] else 'None'}")
                logger.info(f"[DEBUG RESPONSE] Message content: {response_data['message'].get('content', '')[:100] if response_data['message'] else 'None'}")

            # Handle both chat and generate response formats
            # Extract response text based on endpoint format
            response_text = None
            if "message" in response_data and "content" in response_data["message"]:
                # Chat endpoint response format
                response_text = response_data["message"]["content"]
            elif "response" in response_data:
                # Generate endpoint response format
                response_text = response_data["response"]
            else:
                raise OllamaException(f"Unexpected response format: {response_data}")

            # DEBUG: Log raw response from Ollama
            logger.info(f"[DEBUG OLLAMA] Raw response from Ollama (length: {len(response_text) if response_text else 0}): {response_text[:500] if response_text else '(empty)'}")

            # Clean up model-specific artifacts (modular approach)
            # Strip thinking blocks for JSON responses (outcome detection, routing)
            strip_thinking = (format == "json")
            response_text = self._clean_model_artifacts(response_text, actual_model, strip_thinking=strip_thinking)
            logger.info(f"[DEBUG OLLAMA] After cleaning (length: {len(response_text) if response_text else 0}): {response_text[:200] if response_text else '(empty)'}")
            return response_text
        except OllamaException:
            # Re-raise OllamaException as-is
            raise
        except Exception as e:
            # Log the full error for debugging
            logger.error(f"[FINAL ERROR] Complete failure in generate_response: {e}")
            raise OllamaException(f"Ollama generate_response failed: {e}") from e

    async def generate_response_with_tools(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict]] = None,
        format: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response with tool support (function calling).

        Returns:
            Dict with keys:
                - "content": The text response (if no tool calls)
                - "tool_calls": List of tool calls (if LLM wants to use tools)
        """
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        # Recycle client connection periodically
        self._request_count += 1
        if self._request_count >= self._max_requests_per_client:
            logger.info(f"Recycling HTTP client after {self._request_count} requests")
            await self._recycle_client()
            self._request_count = 0

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for item in history:
                messages.append(item)
        messages.append({"role": "user", "content": prompt})

        actual_model = model or self.model_name
        logger.debug(f"Using model with tools: {actual_model}")

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": False,
            "think": False  # Disable thinking mode (qwen3, etc.) - much faster responses
        }

        # Tools are called via <tool_call> tags in the response, not Ollama's native API
        # This makes the system universal - works with all models regardless of Ollama support
        if tools:
            logger.debug(f"{len(tools)} tools available (via tag-based calling)")

        # Add model-specific options
        if "deepseek" not in actual_model.lower():
            from config.model_limits import get_generation_params
            gen_params = get_generation_params(actual_model)
            options = {"num_gpu": 99}  # Force max GPU layers

            if gen_params.get("num_predict"):
                options["num_predict"] = gen_params["num_predict"]
            if gen_params.get("temperature") is not None:
                options["temperature"] = gen_params["temperature"]
            if gen_params.get("repeat_penalty"):
                options["repeat_penalty"] = gen_params["repeat_penalty"]
            if gen_params.get("stop"):
                options["stop"] = gen_params["stop"]

            payload["options"] = options  # Always set options now (at minimum num_gpu)

            if format:
                payload["format"] = format

        try:
            # Use /api/chat endpoint for tool support
            logger.debug(f"Calling /api/chat with tools for model: {actual_model}")
            api_response = await self.client.post("/api/chat", json=payload)
            api_response.raise_for_status()

            response_data = api_response.json()
            logger.debug(f"Tool-enabled response keys: {response_data.keys()}")

            # Extract message
            if "message" not in response_data:
                raise OllamaException(f"No message in response: {response_data}")

            message = response_data["message"]

            # Check if LLM wants to use tools
            if "tool_calls" in message and message["tool_calls"]:
                logger.info(f"LLM requested {len(message['tool_calls'])} tool call(s)")
                return {
                    "content": None,
                    "tool_calls": message["tool_calls"]
                }

            # No tool calls, return content
            content = message.get("content", "")
            strip_thinking = (format == "json")
            content = self._clean_model_artifacts(content, actual_model, strip_thinking=strip_thinking)
            return {
                "content": content,
                "tool_calls": None
            }

        except Exception as e:
            logger.error(f"Tool-enabled response failed: {e}")
            raise OllamaException(f"generate_response_with_tools failed: {e}") from e

    def _clean_model_artifacts(self, text: str, model: str, strip_thinking: bool = False) -> str:
        """Format model-specific artifacts for better display.

        This method handles different model outputs in a modular way.

        Args:
            text: Raw model output
            model: Model name
            strip_thinking: If True, remove <think> blocks (for JSON responses)
        """
        if not text:
            return text

        import re

        # Strip thinking blocks ONLY for JSON responses (outcome detection, routing, etc.)
        # For chat responses, agent_chat.py extracts thinking for UI display
        if strip_thinking:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Remove common artifacts from any model
        # Remove excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Remove trailing/leading whitespace
        text = text.strip()

        # Remove common prefixes some models add
        prefixes_to_remove = [
            "Assistant: ",  # Some models prefix their responses
            "Response: ",
            "Answer: ",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        return text

    async def stream_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        # Build messages array for /api/chat
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        actual_model = model or self.model_name
        logger.info(f"[STREAM] Using model: {actual_model}")

        # ==================== MULTI-PROVIDER STREAMING ====================
        if self.api_style == "openai":
            # OpenAI-compatible streaming (Server-Sent Events format)
            try:
                logger.debug(f"Streaming from OpenAI-style API at {self.base_url}/v1/chat/completions")

                async with self.client.stream(
                    'POST',
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": actual_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "stream": True
                    },
                    timeout=self.request_timeout
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if content := delta.get('content'):
                                        yield content
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                logger.error(f"OpenAI-style streaming error: {e}")
                raise OllamaException(f"OpenAI streaming error: {str(e)}")

            return
        # ==================== END MULTI-PROVIDER STREAMING ====================

        # Use /api/chat for streaming (works better with modern models like Qwen2.5)
        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": True
        }

        try:
            async with self.client.stream("POST", "/api/chat", json=payload, timeout=self.request_timeout) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line)
                        if "error" in chunk_data:
                            raise OllamaException(f"Ollama API stream error: {chunk_data['error']}")
                        # For /api/chat, the response is in message.content field
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            yield chunk_data["message"]["content"]
                        if chunk_data.get("done"):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed JSON line from stream: {line}")
        except Exception as e:
            raise OllamaException(f"Ollama stream_response failed: {e}") from e

    async def stream_response_with_tools(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response with tool support for function calling.

        Yields dictionaries with either:
        - {"type": "text", "content": str} for text chunks
        - {"type": "tool_call", "tool_calls": list} when LLM wants to use tools
        - {"type": "done"} when complete
        """
        if not self.client:
            raise OllamaException("OllamaClient is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        # Only add user message if prompt is not empty
        if prompt:
            messages.append({"role": "user", "content": prompt})

        actual_model = model or self.model_name
        logger.info(f"[STREAM WITH TOOLS] Using model: {actual_model}")
        logger.info(f"[DEBUG MESSAGES] Message count: {len(messages)}, Roles: {[m['role'] for m in messages]}")
        if messages:
            logger.info(f"[DEBUG MESSAGES] Last 200 chars of last message: {messages[-1]['content'][-200:]}")

        # Model capability detection for native tool support (October 2025)
        NATIVE_TOOL_MODELS = [
            "gpt-oss",  # OpenAI's open source models (20b, 120b)
            "llama3.1", "llama3.2", "llama-3.1", "llama-3.2",  # Meta Llama variants
            "qwen", "qwen2", "qwen2.5",  # Alibaba Qwen family - excellent tool support
            "mistral", "mixtral",  # Mistral AI family
            "command-r", "command-r-plus",  # Cohere models
            "phi-4", "phi4",  # Microsoft Phi-4
            # "dolphin", "dolphin3",  # REMOVED 2025-10-09: Causes 400 errors with tools
            "firefunction"  # FireFunction optimized for tools
        ]

        # Check if model supports native tools
        model_lower = actual_model.lower()
        supports_native_tools = any(
            native_model in model_lower
            for native_model in NATIVE_TOOL_MODELS
        )

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": True,
            "think": False  # Disable thinking mode (qwen3, etc.) - much faster responses
        }

        # Use centralized context configuration
        from config.model_contexts import get_context_size
        num_ctx = get_context_size(actual_model)
        payload["options"] = {
            "num_ctx": num_ctx,
            "num_gpu": 99  # Force full GPU offload - fixes Ollama's conservative default
        }
        logger.info(f"[STREAM WITH TOOLS] Set context window to {num_ctx} for {actual_model} (num_gpu=99)")

        # Only pass tools to models that support native API
        if tools and supports_native_tools:
            payload["tools"] = tools
            logger.info(f"[STREAM WITH TOOLS] {len(tools)} tools passed to {actual_model} (native support)")
            logger.debug(f"[STREAM WITH TOOLS] Tools: {json.dumps(tools, indent=2)}")
        elif tools:
            logger.warning(f"[STREAM WITH TOOLS] Model {actual_model} doesn't support native tools - tools disabled for this request")

        try:
            accumulated_content = ""
            tool_calls_buffer = ""
            line_count = 0
            yield_count = 0

            logger.info(f"[STREAM DEBUG] Starting stream from /api/chat")

            # ==================== MULTI-PROVIDER API ROUTING ====================
            if self.api_style == "openai":
                # OpenAI-compatible providers (LM Studio) use /v1/chat/completions with SSE
                logger.debug(f"Using OpenAI-style streaming API at {self.base_url}/v1/chat/completions")

                openai_payload = {
                    "model": actual_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "stream": True
                }

                # Add tools if provided
                if tools:
                    openai_payload["tools"] = tools
                    logger.info(f"[OPENAI PAYLOAD] Added {len(tools)} tools to OpenAI payload")
                    # Log system message length for debugging
                    sys_msg_len = len(messages[0]['content']) if messages and messages[0]['role'] == 'system' else 0
                    logger.info(f"[OPENAI PAYLOAD] System message length: {sys_msg_len} chars, Total messages: {len(messages)}")
                    logger.info(f"[OPENAI PAYLOAD] Message roles: {[m['role'] for m in messages]}")
                    for idx, msg in enumerate(messages):
                        content_preview = msg['content'][:100] if len(msg['content']) > 100 else msg['content']
                        logger.info(f"[OPENAI PAYLOAD] Message {idx} ({msg['role']}): {content_preview}...")
                    # Dump full payload for debugging (excluding large tool definitions)
                    debug_payload = {**openai_payload}
                    debug_payload['tools'] = f"[{len(tools)} tools]"  # Replace with count
                    logger.info(f"[OPENAI PAYLOAD] Full request: {json.dumps(debug_payload, indent=2)}")
                else:
                    logger.warning(f"[OPENAI PAYLOAD] NO TOOLS in payload! tools parameter was: {tools}")

                try:
                    response_stream = self.client.stream("POST", "/v1/chat/completions", json=openai_payload, timeout=self.request_timeout)
                    response = await response_stream.__aenter__()
                    response.raise_for_status()
                    logger.info(f"[STREAM DEBUG] OpenAI-style Response status: {response.status_code}")
                except httpx.HTTPStatusError as e:
                    logger.error(f"OpenAI-style API streaming error: {e}")
                    raise OllamaException(f"OpenAI API streaming error: {str(e)}")

                # Stream OpenAI-style SSE responses
                # Accumulator for tool calls (OpenAI streams them in chunks)
                tool_call_accumulator = {}  # {index: {"id": "", "name": "", "arguments": ""}}
                # Track reasoning mode for qwen3 - emit <think> tag once at start, </think> once at end
                in_reasoning_mode = False

                async for line in response.aiter_lines():
                    line_count += 1
                    if not line.strip() or line.startswith(":"):
                        continue

                    # OpenAI SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            logger.info(f"[STREAM DEBUG] OpenAI stream complete")
                            
                            # Close <think> tag if still open at end of stream
                            if in_reasoning_mode:
                                yield {"type": "text", "content": "</think>"}
                                in_reasoning_mode = False

                            # Yield accumulated tool calls before finishing
                            if tool_call_accumulator:
                                logger.info(f"[OPENAI STREAM] Yielding {len(tool_call_accumulator)} accumulated tool calls")
                                tool_calls = []
                                for idx, tc in tool_call_accumulator.items():
                                    try:
                                        args_dict = json.loads(tc["arguments"]) if tc["arguments"] else {}
                                        tool_calls.append({
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["name"],
                                                "arguments": args_dict
                                            }
                                        })
                                    except json.JSONDecodeError as e:
                                        logger.error(f"[OPENAI STREAM] Failed to parse tool arguments: {tc['arguments'][:100]}... Error: {e}")

                                if tool_calls:
                                    yield {"type": "tool_call", "tool_calls": tool_calls}
                                    yield_count += 1

                            break

                        try:
                            chunk = json.loads(data_str)

                            # Check for error responses from LM Studio / OpenAI-compatible servers
                            if "error" in chunk:
                                error_msg = chunk.get("error", {}).get("message", str(chunk.get("error")))
                                logger.error(f"[OPENAI STREAM] Server returned error: {error_msg}")

                                # Provide helpful context-specific error messages
                                if "context" in error_msg.lower() and ("overflow" in error_msg.lower() or "length" in error_msg.lower()):
                                    user_msg = (
                                        f"**Context Length Error:** LM Studio loaded this model with only 4096 context, but Roampal needs ~6000 tokens.\n\n"
                                        f"**Fix in LM Studio (must be done there, not in Roampal):**\n"
                                        f"1. In LM Studio, **unload** the model first\n"
                                        f"2. Change the Context Length slider to at least **8192** (16384+ recommended)\n"
                                        f"3. **Load** the model again\n\n"
                                        f"*LM Studio ignores context settings unless you reload the model. Roampal's context settings only work with Ollama.*\n\n"
                                        f"*Or use Ollama instead:* `ollama pull {actual_model}` - it handles context automatically."
                                    )
                                else:
                                    user_msg = f"**Model Error:** {error_msg}"

                                yield {"type": "text", "content": user_msg}
                                yield_count += 1
                                break

                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                            # Log first few chunks for debugging
                            if line_count <= 3:
                                logger.info(f"[OPENAI STREAM] Chunk {line_count}: delta keys = {list(delta.keys())}, finish_reason = {finish_reason}")
                                # Log raw chunk for debugging empty responses
                                logger.info(f"[OPENAI STREAM] Raw chunk {line_count}: {data_str[:500]}")

                            # Handle content (standard OpenAI format)
                            if "content" in delta and delta["content"]:
                                # Close <think> tag if we were in reasoning mode
                                if in_reasoning_mode:
                                    yield {"type": "text", "content": "</think>"}
                                    in_reasoning_mode = False
                                yield {"type": "text", "content": delta["content"]}
                                yield_count += 1

                            # Handle reasoning_content (qwen3 thinking mode)
                            # v0.2.5: Discard reasoning content entirely - users don't want to see it
                            # LM Studio doesn't support enable_thinking=False, so we filter here
                            elif "reasoning_content" in delta and delta["reasoning_content"]:
                                # Just track that we're in reasoning mode (for proper </think> handling)
                                # but don't emit the content
                                if not in_reasoning_mode:
                                    in_reasoning_mode = True
                                    logger.debug(f"[OPENAI STREAM] Discarding reasoning_content (thinking mode)")
                                # Don't yield reasoning content - skip it entirely

                            # Universal fallback: check for any text-like field we haven't handled
                            # Some models use non-standard field names
                            elif not delta.get("tool_calls"):
                                for key in ["text", "message", "response", "output", "generated_text"]:
                                    if key in delta and delta[key]:
                                        logger.warning(f"[OPENAI STREAM] Found content in non-standard field '{key}'")
                                        yield {"type": "text", "content": delta[key]}
                                        yield_count += 1
                                        break

                            # Handle tool calls (accumulate chunks)
                            if "tool_calls" in delta:
                                for tool_call_chunk in delta["tool_calls"]:
                                    idx = tool_call_chunk.get("index", 0)

                                    # Initialize accumulator for this tool call index
                                    if idx not in tool_call_accumulator:
                                        tool_call_accumulator[idx] = {
                                            "id": tool_call_chunk.get("id", ""),
                                            "name": "",
                                            "arguments": ""
                                        }

                                    # Accumulate function name
                                    if "function" in tool_call_chunk:
                                        func = tool_call_chunk["function"]
                                        if "name" in func:
                                            tool_call_accumulator[idx]["name"] = func["name"]
                                        if "arguments" in func:
                                            tool_call_accumulator[idx]["arguments"] += func["arguments"]

                            # If finish_reason is "tool_calls", yield accumulated tools immediately
                            if finish_reason == "tool_calls" and tool_call_accumulator:
                                logger.info(f"[OPENAI STREAM] finish_reason=tool_calls, yielding {len(tool_call_accumulator)} tool calls")
                                tool_calls = []
                                for idx, tc in tool_call_accumulator.items():
                                    try:
                                        args_dict = json.loads(tc["arguments"]) if tc["arguments"] else {}
                                        tool_calls.append({
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["name"],
                                                "arguments": args_dict
                                            }
                                        })
                                        logger.info(f"[OPENAI STREAM] Tool call: {tc['name']}({tc['arguments']})")
                                    except json.JSONDecodeError as e:
                                        logger.error(f"[OPENAI STREAM] Failed to parse tool arguments: {tc['arguments'][:100]}... Error: {e}")

                                if tool_calls:
                                    yield {"type": "tool_call", "tool_calls": tool_calls}
                                    yield_count += 1
                                    break  # Exit streaming to let agent handle tool execution

                        except json.JSONDecodeError as e:
                            logger.warning(f"[STREAM DEBUG] Failed to parse OpenAI chunk: {data_str[:100]}... Error: {e}")
                            continue

                logger.info(f"[STREAM DEBUG] OpenAI stream finished: {line_count} lines, {yield_count} yields")
                return
            # ==================== END MULTI-PROVIDER ROUTING ====================

            # OOM Protection: Try with full context, retry with reduced if crashes
            oom_retry = False
            original_ctx = payload.get("options", {}).get("num_ctx", "unknown")

            try:
                response_stream = self.client.stream("POST", "/api/chat", json=payload, timeout=self.request_timeout)
                response = await response_stream.__aenter__()
                response.raise_for_status()
                logger.info(f"[STREAM DEBUG] Response status: {response.status_code}")
            except httpx.HTTPStatusError as e:
                # Detect OOM: Ollama returns 500 with "terminated" when it runs out of memory
                if e.response.status_code == 500:
                    error_text = await e.response.aread() if hasattr(e.response, 'aread') else str(e)
                    error_str = error_text.decode() if isinstance(error_text, bytes) else str(error_text)

                    if "terminated" in error_str.lower():
                        logger.warning(f"⚠️ OOM detected (context: {original_ctx}), retrying with 2048")
                        payload["options"]["num_ctx"] = 2048
                        oom_retry = True

                        # Retry with reduced context
                        response_stream = self.client.stream("POST", "/api/chat", json=payload, timeout=self.request_timeout)
                        response = await response_stream.__aenter__()
                        response.raise_for_status()
                        logger.info(f"[STREAM DEBUG] OOM recovery successful, Response status: {response.status_code}")

                        # If OOM recovery happened, prepend warning message
                        warning_msg = (
                            f"⚠️ **Memory Limit Reached**\n\n"
                            f"This model ran out of memory with {original_ctx} context window. "
                            f"Reduced to 2048 tokens for this response.\n\n"
                            f"**To fix permanently:** Open Settings → Context Window Settings → "
                            f"Lower context for `{actual_model}` to 8192 or less.\n\n"
                            f"---\n\n"
                        )
                        yield {"type": "text", "content": warning_msg}
                        yield_count += 1
                    else:
                        raise  # Re-raise if not OOM
                else:
                    raise  # Re-raise if not 500

            # Stream the response
            async for line in response.aiter_lines():
                line_count += 1
                if not line.strip():
                    logger.debug(f"[STREAM DEBUG] Line {line_count}: empty, skipping")
                    continue

                logger.debug(f"[STREAM DEBUG] Line {line_count}: {line[:100]}")
                try:
                    chunk_data = json.loads(line)
                    logger.debug(f"[STREAM DEBUG] Parsed chunk keys: {chunk_data.keys()}")

                    if "error" in chunk_data:
                        raise OllamaException(f"Ollama API stream error: {chunk_data['error']}")

                    if "message" in chunk_data:
                        message = chunk_data["message"]
                        logger.debug(f"[STREAM DEBUG] Message keys: {message.keys()}")

                        # Check for tool calls
                        if "tool_calls" in message and message["tool_calls"]:
                            logger.info(f"[STREAM WITH TOOLS] Tool call detected")
                            yield {"type": "tool_call", "tool_calls": message["tool_calls"]}
                            yield_count += 1
                            # IMPORTANT: We need to break here to allow agent_chat.py to handle the tool call
                            # The native tool handler in agent_chat.py will execute the tool and continue
                            # the conversation with the tool results
                            break  # Exit the streaming loop to let agent handle tool execution

                        # Regular content streaming
                        if "content" in message:
                            content = message["content"]
                            accumulated_content += content

                            # DEBUG: Log chunks containing think tags to diagnose splitting
                            if '<think' in content or '</think' in content or 'think>' in content:
                                logger.info(f"[OLLAMA THINK DEBUG] Raw chunk from Ollama: '{content}'")

                            logger.debug(f"[STREAM DEBUG] Yielding text chunk (length: {len(content)})")
                            yield {"type": "text", "content": content}
                            yield_count += 1
                        else:
                            logger.debug(f"[STREAM DEBUG] Message has no 'content' field")

                    if chunk_data.get("done"):
                        logger.info(f"[STREAM DEBUG] Done signal received")
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"[STREAM DEBUG] Malformed JSON at line {line_count}: {line[:200]}")
                    logger.warning(f"[STREAM DEBUG] JSON error: {e}")

            logger.info(f"[STREAM DEBUG] Stream complete - processed {line_count} lines, yielded {yield_count} chunks")
            yield {"type": "done"}

        except Exception as e:
            raise OllamaException(f"Ollama stream_response_with_tools failed: {e}") from e

    async def _recycle_client(self) -> None:
        """Close and recreate the HTTP client to avoid connection issues"""
        if self.client:
            await self.client.aclose()

        # Recreate with same settings
        timeout_config = httpx.Timeout(
            self.request_timeout,
            connect=30,
            read=self.request_timeout,
            write=30
        )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_config,
            # Use default limits - no special configuration needed
        )
        logger.info("HTTP client recycled successfully")

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None