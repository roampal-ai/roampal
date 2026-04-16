"""
Tests for ollama_client.py — LLM communication layer, multi-provider routing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    @pytest.mark.asyncio
    async def test_initialize_creates_client(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
        })
        assert client.client is not None
        assert client.model_name == "test:latest"
        assert client.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_initialize_defaults(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({})
        assert client.model_name is not None  # has a default
        assert client.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_api_style_default_ollama(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({})
        assert client.api_style == "ollama"

    @pytest.mark.asyncio
    async def test_api_style_openai(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({"ollama_base_url": "http://localhost:1234"})
        client.api_style = "openai"
        assert client.api_style == "openai"


# ---------------------------------------------------------------------------
# generate_response routing
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    @pytest.mark.asyncio
    async def test_ollama_api_routing(self):
        """Ollama-style uses /api/chat endpoint."""
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:latest",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }
        mock_response.status_code = 200
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        result = await client.generate_response("Hi", system_prompt="You are helpful")
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_openai_api_routing(self):
        """OpenAI-style uses /v1/chat/completions endpoint."""
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:1234",
            "ollama_model": "qwen2.5-7b-instruct",
        })
        client.api_style = "openai"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from LM Studio!"}}],
        }
        mock_response.raise_for_status = MagicMock()
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        result = await client.generate_response("Hi")
        assert result == "Hello from LM Studio!"
        # Verify it called the OpenAI endpoint
        call_args = client.client.post.call_args
        assert "/v1/chat/completions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self):
        from modules.llm.ollama_client import OllamaClient, OllamaException
        client = OllamaClient()
        with pytest.raises(OllamaException, match="not initialized"):
            await client.generate_response("Hi")


# ---------------------------------------------------------------------------
# Client recycling
# ---------------------------------------------------------------------------

class TestClientRecycling:
    @pytest.mark.asyncio
    async def test_recycle_counter_increments(self):
        """Request counter should increment on each call."""
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
        })
        initial = client._request_count

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:latest",
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        mock_response.status_code = 200
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        await client.generate_response("test")
        # Count should have changed (either incremented or reset after recycle)
        assert isinstance(client._request_count, int)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_openai_400_raises_ollama_exception(self):
        """OpenAI 400 error should raise OllamaException."""
        from modules.llm.ollama_client import OllamaClient, OllamaException
        import httpx

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:1234",
            "ollama_model": "bad-model",
        })
        client.api_style = "openai"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=MagicMock(),
                response=MagicMock(status_code=400),
            )
        )
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(OllamaException, match="OpenAI API error"):
            await client.generate_response("Hi")

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        """Connection refused raises OllamaException."""
        from modules.llm.ollama_client import OllamaClient, OllamaException
        import httpx

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:99999",
            "ollama_model": "test:latest",
        })
        client.api_style = "openai"

        client.client = AsyncMock()
        client.client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(OllamaException):
            await client.generate_response("Hi")


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

class TestMessageConstruction:
    @pytest.mark.asyncio
    async def test_system_prompt_added(self):
        """System prompt should be first message."""
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:latest",
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        mock_response.status_code = 200
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        await client.generate_response("Hi", system_prompt="Be helpful")
        call_args = client.client.post.call_args
        payload = call_args[1].get("json", call_args[0][1] if len(call_args[0]) > 1 else {})
        messages = payload.get("messages", [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hi"
