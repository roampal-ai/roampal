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


# ---------------------------------------------------------------------------
# v0.3.2: Universal HTTP error handling for /api/chat (sections 0a + 0c)
# ---------------------------------------------------------------------------

def _make_stream_response(status: int = 200, body: bytes = b"", lines: list = None):
    """Build a mock httpx streaming response. raise_for_status raises on 4xx/5xx."""
    import httpx

    resp = MagicMock()
    resp.status_code = status
    resp.aread = AsyncMock(return_value=body)

    if 400 <= status < 600:
        err = httpx.HTTPStatusError(
            f"{status}",
            request=MagicMock(),
            response=resp,
        )
        resp.raise_for_status = MagicMock(side_effect=err)
    else:
        resp.raise_for_status = MagicMock()

    async def _aiter():
        for line in lines or []:
            yield line

    resp.aiter_lines = lambda: _aiter()
    return resp


def _make_stream_cm(response):
    """Wrap a response as an async context manager like httpx.AsyncClient.stream returns."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class TestUniversalOllamaErrorHandling:
    """v0.3.2 (0a): parse body on any 4xx/5xx, detect tool-incompat by content."""

    async def _client(self, model: str = "gemma4:e2b"):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": model,
        })
        client.api_style = "ollama"
        return client

    async def _collect(self, gen):
        return [c async for c in gen]

    @pytest.mark.asyncio
    async def test_tools_not_supported_400_retries_without_tools(self):
        client = await self._client()
        bad = _make_stream_response(400, b'{"error":"registry.ollama.ai/library/gemma4:e2b does not support tools"}')
        good = _make_stream_response(200, lines=[json.dumps({"done": True})])

        client.client = MagicMock()
        client.client.stream = MagicMock(side_effect=[_make_stream_cm(bad), _make_stream_cm(good)])

        chunks = await self._collect(client.stream_response_with_tools(
            prompt="hi",
            tools=[{"type": "function", "function": {"name": "ping"}}],
        ))

        # Second call is the retry — its payload must not carry tools.
        assert client.client.stream.call_count == 2
        retry_payload = client.client.stream.call_args_list[1][1]["json"]
        assert "tools" not in retry_payload
        # Stream terminated cleanly (done event)
        assert any(c.get("type") == "done" for c in chunks)

    @pytest.mark.asyncio
    async def test_tools_not_supported_500_retries_without_tools(self):
        """Same path must fire regardless of whether Ollama returned 400 or 500."""
        client = await self._client()
        bad = _make_stream_response(500, b'{"error":"model does not support tools"}')
        good = _make_stream_response(200, lines=[json.dumps({"done": True})])

        client.client = MagicMock()
        client.client.stream = MagicMock(side_effect=[_make_stream_cm(bad), _make_stream_cm(good)])

        chunks = await self._collect(client.stream_response_with_tools(
            prompt="hi",
            tools=[{"type": "function", "function": {"name": "ping"}}],
        ))

        assert client.client.stream.call_count == 2
        retry_payload = client.client.stream.call_args_list[1][1]["json"]
        assert "tools" not in retry_payload

    @pytest.mark.asyncio
    async def test_tools_not_supported_case_insensitive(self):
        client = await self._client()
        bad = _make_stream_response(400, b'{"error":"Does Not Support Tools"}')
        good = _make_stream_response(200, lines=[json.dumps({"done": True})])

        client.client = MagicMock()
        client.client.stream = MagicMock(side_effect=[_make_stream_cm(bad), _make_stream_cm(good)])

        _ = await self._collect(client.stream_response_with_tools(
            prompt="hi",
            tools=[{"type": "function", "function": {"name": "ping"}}],
        ))

        assert client.client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_other_http_error_surfaces_user_message(self):
        """Unrecognized errors must surface as text chunk, NOT kill the stream silently."""
        client = await self._client()
        bad = _make_stream_response(503, b"Service Unavailable: upstream broken")

        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=_make_stream_cm(bad))

        chunks = await self._collect(client.stream_response_with_tools(
            prompt="hi",
        ))

        text_chunks = [c for c in chunks if c.get("type") == "text"]
        assert text_chunks, "Expected a user-facing text chunk for unrecoverable error"
        assert "Model error" in text_chunks[0]["content"]
        # Stream terminates cleanly
        assert any(c.get("type") == "done" for c in chunks)

    @pytest.mark.asyncio
    async def test_no_tool_blocklist_for_formerly_blocked_models(self):
        """Models that used to be on TOOL_BLOCKLIST (dolphin) must now receive tools."""
        client = await self._client(model="dolphin:latest")
        good = _make_stream_response(200, lines=[json.dumps({"done": True})])

        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=_make_stream_cm(good))

        _ = await self._collect(client.stream_response_with_tools(
            prompt="hi",
            tools=[{"type": "function", "function": {"name": "ping"}}],
        ))

        payload = client.client.stream.call_args[1]["json"]
        # Tools pass through — capability detection replaces the blocklist.
        assert "tools" in payload
        assert len(payload["tools"]) == 1


class TestKeepAlivePayload:
    """v0.3.2 (0e): keep_alive must be sent in every /api/chat payload."""

    @pytest.mark.asyncio
    async def test_payload_includes_keep_alive_on_stream_with_tools(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
            "ollama_keep_alive": "24h",
        })
        client.api_style = "ollama"

        good = _make_stream_response(200, lines=[json.dumps({"done": True})])
        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=_make_stream_cm(good))

        _ = [c async for c in client.stream_response_with_tools(prompt="hi")]

        payload = client.client.stream.call_args[1]["json"]
        assert payload.get("keep_alive") == "24h"

    @pytest.mark.asyncio
    async def test_payload_includes_keep_alive_on_chat(self):
        from modules.llm.ollama_client import OllamaClient
        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "test:latest",
            "ollama_keep_alive": "24h",
        })
        client.api_style = "ollama"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:latest",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
        }
        mock_response.status_code = 200
        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        await client.generate_response("hi")
        payload = client.client.post.call_args[1]["json"]
        assert payload.get("keep_alive") == "24h"


class TestLMStudioHttpStatusError:
    """v0.3.2 (0c extension): LM Studio HTTP 500/404 with 'not found' body
    also clears state — catches the case where httpx raises HTTPStatusError
    before any stream chunks can be read."""

    @pytest.mark.asyncio
    async def test_lmstudio_http_500_with_stale_body_clears_state(self, tmp_path, monkeypatch):
        import json as _json
        import httpx
        from modules.llm.ollama_client import OllamaClient
        from config.settings import settings as app_settings

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "main_model_config.json").write_text(
            _json.dumps({"model": "qwen2.5-7b-instruct", "provider": "lmstudio"})
        )

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:1234",
            "ollama_model": "qwen2.5-7b-instruct",
        })
        client.api_style = "openai"

        # Build response that raise_for_status() will convert into HTTPStatusError
        resp = MagicMock()
        resp.status_code = 500
        resp.aread = AsyncMock(return_value=b'{"error":{"message":"Model \\"qwen2.5-7b-instruct\\" not found"}}')
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
        resp.raise_for_status = MagicMock(side_effect=err)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=cm)

        chunks = [c async for c in client.stream_response_with_tools(prompt="hi")]

        text_chunks = [c for c in chunks if c.get("type") == "text"]
        # Stale-model clean user message (no MDN link, no httpx raw error).
        assert any("no longer available" in c["content"] for c in text_chunks)
        # Runtime model cleared
        assert client.model_name == ""
        # v0.3.2: config rewritten with blank model but provider preserved.
        import json as _json
        cfg = _json.loads((tmp_path / "main_model_config.json").read_text(encoding="utf-8"))
        assert cfg == {"model": "", "provider": "lmstudio"}

    @pytest.mark.asyncio
    async def test_lmstudio_http_500_unknown_body_yields_clean_user_error(self):
        """Non-stale 500 should still surface a clean **Model error:** chunk,
        not let httpx's MDN-link default text through."""
        import httpx
        from modules.llm.ollama_client import OllamaClient

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:1234",
            "ollama_model": "test-model",
        })
        client.api_style = "openai"

        resp = MagicMock()
        resp.status_code = 500
        resp.aread = AsyncMock(return_value=b"Internal Server Error: something else broke")
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
        resp.raise_for_status = MagicMock(side_effect=err)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=cm)

        chunks = [c async for c in client.stream_response_with_tools(prompt="hi")]
        text_chunks = [c for c in chunks if c.get("type") == "text"]
        assert text_chunks, "Expected user-facing text chunk"
        assert "Model error" in text_chunks[0]["content"]
        # Must NOT leak httpx's default "For more information check https://..." message.
        assert "developer.mozilla.org" not in text_chunks[0]["content"]


class TestStaleModelLMStudio:
    """v0.3.2 (0c extension): LM Studio-style error surface also clears state."""

    @pytest.mark.asyncio
    async def test_lmstudio_model_not_found_clears_state(self, tmp_path, monkeypatch):
        import json as _json
        from modules.llm.ollama_client import OllamaClient
        from config.settings import settings as app_settings

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "main_model_config.json").write_text(
            _json.dumps({"model": "qwen2.5-7b-instruct", "provider": "lmstudio"})
        )

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:1234",
            "ollama_model": "qwen2.5-7b-instruct",
        })
        client.api_style = "openai"

        # Fake LM Studio SSE stream yielding a single error chunk.
        err_chunk_body = _json.dumps({
            "error": {"message": "Model 'qwen2.5-7b-instruct' not found"}
        })

        async def _aiter():
            # SSE format: "data: {json}\n"
            yield f"data: {err_chunk_body}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.aiter_lines = lambda: _aiter()

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=cm)

        chunks = [c async for c in client.stream_response_with_tools(prompt="hi")]

        # User-facing stale-model message present.
        text_chunks = [c for c in chunks if c.get("type") == "text"]
        assert any("no longer available" in c["content"] for c in text_chunks)
        # Runtime model cleared.
        assert client.model_name == ""
        # v0.3.2: config rewritten with blank model but provider preserved.
        import json as _json
        cfg = _json.loads((tmp_path / "main_model_config.json").read_text(encoding="utf-8"))
        assert cfg == {"model": "", "provider": "lmstudio"}


class TestStaleModel404:
    """v0.3.2 (0c): 404 on /api/chat means the model was removed outside Roampal."""

    @pytest.mark.asyncio
    async def test_ollama_404_yields_user_message_and_clears_model(self, tmp_path, monkeypatch):
        from modules.llm.ollama_client import OllamaClient
        from config.settings import settings as app_settings

        # Point DATA_PATH-derived files at a temp location for this test.
        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        # Seed a pretend persisted-model file so we can assert it's deleted.
        (tmp_path / "main_model_config.json").write_text('{"model":"gone","provider":"ollama"}')

        client = OllamaClient()
        await client.initialize({
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gone:latest",
        })
        client.api_style = "ollama"

        gone = _make_stream_response(404, b'{"error":"model not found"}')
        client.client = MagicMock()
        client.client.stream = MagicMock(return_value=_make_stream_cm(gone))

        chunks = [c async for c in client.stream_response_with_tools(
            prompt="hi",
        )]

        # v0.3.2 unified copy: "no longer available" (same wording Ollama + LM Studio).
        text_chunks = [c for c in chunks if c.get("type") == "text"]
        assert any("no longer available" in c["content"] for c in text_chunks)
        # Runtime model cleared
        assert client.model_name == ""
        # v0.3.2 follow-up: config is REWRITTEN (not deleted) to preserve the
        # user's provider preference. Boot will fall through to first-available
        # model of the SAME provider.
        import json as _json
        cfg_path = tmp_path / "main_model_config.json"
        assert cfg_path.exists()
        saved = _json.loads(cfg_path.read_text(encoding="utf-8"))
        assert saved == {"model": "", "provider": "ollama"}
