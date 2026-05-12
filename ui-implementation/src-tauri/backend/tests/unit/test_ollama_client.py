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


class TestCleanModelArtifacts:
    """v0.3.3 (Section 3): Harmony channel-token leakage fix."""

    def _client(self):
        from modules.llm.ollama_client import OllamaClient
        return OllamaClient()

    @pytest.mark.asyncio
    async def test_strips_channel_token(self):
        """<channel|> token must be removed from output."""
        client = self._client()
        result = client._clean_model_artifacts(
            "Let me check!<channel|>\nBased on records...",
            model="gemma4:31b",
        )
        assert "<channel|>" not in result
        assert "Let me check!" in result

    @pytest.mark.asyncio
    async def test_strips_pipe_delimited_tokens(self):
        """<|start|>, <|end|>, <|message|> tokens must be removed."""
        client = self._client()
        result = client._clean_model_artifacts(
            "<|start|>Hello<|end|>",
            model="gemma4:31b",
        )
        assert "<|start|>" not in result
        assert "<|end|>" not in result
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_strips_message_token(self):
        """<|message|> token must be removed."""
        client = self._client()
        result = client._clean_model_artifacts(
            "Sure!<|message|>Here is the answer.",
            model="qwen:7b",
        )
        assert "<|message|>" not in result

    @pytest.mark.asyncio
    async def test_preserves_normal_text(self):
        """Normal text without tokens passes through unchanged."""
        client = self._client()
        original = "This is a normal response with no artifacts."
        result = client._clean_model_artifacts(original, model="qwen:7b")
        assert result == original

    @pytest.mark.asyncio
    async def test_strips_multiple_tokens(self):
        """Multiple Harmony tokens in one string are all removed."""
        client = self._client()
        result = client._clean_model_artifacts(
            "<|start|><channel|>Hello<|end|>",
            model="gemma4:31b",
        )
        assert "<|" not in result
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_handles_empty_input(self):
        """Empty/None input returns as-is."""
        client = self._client()
        assert client._clean_model_artifacts("", model="x") == ""
        assert client._clean_model_artifacts(None, model="x") is None


class TestCleanModelArtifactsStreaming:
    """v0.3.3 hotfix: per-chunk streaming variant must NOT strip whitespace.

    The full _clean_model_artifacts call had `.strip()` which, when applied per
    streaming chunk, ate the leading space SentencePiece-tokenized models emit
    on every token. Result: words mashed together (`"Yo!What'sup?How's..."`).
    The streaming variant only strips Harmony control tokens — whitespace and
    prefix-removal are deferred to a final pass.
    """

    def _client(self):
        from modules.llm.ollama_client import OllamaClient
        return OllamaClient()

    def test_preserves_leading_space(self):
        """A chunk like ' the' (SentencePiece word boundary) must keep its leading space."""
        client = self._client()
        assert client._clean_model_artifacts_streaming(" the") == " the"

    def test_preserves_trailing_space(self):
        """Trailing whitespace within a chunk must survive too."""
        client = self._client()
        assert client._clean_model_artifacts_streaming("hello ") == "hello "

    def test_single_space_chunk_passes_through(self):
        """A chunk that IS just whitespace must not collapse to empty."""
        client = self._client()
        assert client._clean_model_artifacts_streaming(" ") == " "
        assert client._clean_model_artifacts_streaming("\n") == "\n"

    def test_concatenated_chunks_form_proper_sentence(self):
        """Regression: SentencePiece-style chunks must reassemble with spaces intact."""
        client = self._client()
        chunks = ["Yo", "!", " What", "'", "s", " up", "?"]
        result = "".join(client._clean_model_artifacts_streaming(c) for c in chunks)
        assert result == "Yo! What's up?"

    def test_strips_harmony_tokens_per_chunk(self):
        """Harmony tags within a chunk are still stripped — that part of Section 3 is kept."""
        client = self._client()
        assert client._clean_model_artifacts_streaming("Sure!<|message|>") == "Sure!"
        assert client._clean_model_artifacts_streaming("<channel|>foo") == "foo"

    def test_does_not_strip_answer_prefix(self):
        """Prefix removal ('Answer: ', etc.) is non-streaming-safe and must NOT fire here.

        A chunk like 'Answer: ' arriving mid-stream is just text, not a model prefix.
        """
        client = self._client()
        assert client._clean_model_artifacts_streaming("Answer: ") == "Answer: "

    def test_does_not_collapse_newlines(self):
        """The \\n{4,} -> \\n\\n\\n collapse is non-streaming-safe."""
        client = self._client()
        text = "\n\n\n\n\n"  # 5 newlines — full cleaner would collapse to 3
        assert client._clean_model_artifacts_streaming(text) == text

    def test_handles_empty_input(self):
        """Empty/None input returns as-is."""
        client = self._client()
        assert client._clean_model_artifacts_streaming("") == ""
        assert client._clean_model_artifacts_streaming(None) is None


class TestStripDataUrlPrefix:
    """v0.3.3 Defect 3: Ollama's multimodal `images` field wants raw base64."""

    def _strip(self):
        from modules.llm.ollama_client import _strip_data_url_prefix
        return _strip_data_url_prefix

    def test_strips_png_data_url(self):
        strip = self._strip()
        assert strip("data:image/png;base64,iVBORw0KGgo=") == "iVBORw0KGgo="

    def test_strips_jpeg_data_url(self):
        strip = self._strip()
        assert strip("data:image/jpeg;base64,/9j/4AAQSkZJRg==") == "/9j/4AAQSkZJRg=="

    def test_raw_base64_passes_through(self):
        """Already-stripped strings round-trip unchanged (idempotency)."""
        strip = self._strip()
        assert strip("iVBORw0KGgoAAAANSUhEUg==") == "iVBORw0KGgoAAAANSUhEUg=="

    def test_non_data_url_passes_through(self):
        """A regular https URL is not modified."""
        strip = self._strip()
        assert strip("https://example.com/image.png") == "https://example.com/image.png"

    def test_non_string_passes_through(self):
        from modules.llm.ollama_client import _strip_data_url_prefix
        assert _strip_data_url_prefix(None) is None
        assert _strip_data_url_prefix(123) == 123


class TestMultimodalPayloadShape:
    """v0.3.3 Defect 3: stream_response_with_tools must shape the multimodal
    message payload according to the active provider's API contract.

    - Ollama /api/chat: {"role": ..., "content": "<text>", "images": ["<raw base64>"]}
    - LM Studio /v1/chat/completions: {"role": ..., "content": [content_blocks]}

    Sending content_blocks to Ollama produces:
      "json: cannot unmarshal array into Go struct field ChatRequest.messages.content of type string"
    """

    def _capture_payload(self, api_style: str):
        """Build a client with the given api_style and instrument it so the
        first /api/chat (Ollama) or /v1/chat/completions (LM Studio) POST is
        captured without actually streaming. Returns the captured request JSON.
        """
        from modules.llm.ollama_client import OllamaClient

        client = OllamaClient()
        client.api_style = api_style
        # Force-initialize the httpx client attribute so the not-initialized
        # guard at the top of stream_response_with_tools doesn't short-circuit.
        client.client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_ollama_payload_uses_native_images_field(self):
        """Ollama path: content is a string, images is a sibling list of raw base64."""
        from modules.llm.ollama_client import _strip_data_url_prefix
        client = self._capture_payload("ollama")

        captured = {}

        async def fake_stream(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            raise StopAsyncIteration()  # bail before real streaming begins

        # The function builds `messages` BEFORE the httpx call; we can inspect
        # by patching post and catching the raise.
        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        try:
            agen = client.stream_response_with_tools(
                prompt="describe this",
                images=["data:image/png;base64,iVBORw0KGgo="],
                model="gemma4:31b",
            )
            async for _ in agen:
                break  # we only care about the request shape, not the stream output
        except Exception:
            pass

        assert captured.get("url", "").endswith("/api/chat"), captured
        msg = captured["json"]["messages"][-1]
        assert msg["role"] == "user"
        assert msg["content"] == "describe this"
        assert msg["images"] == ["iVBORw0KGgo="]  # data: prefix stripped

    @pytest.mark.asyncio
    async def test_lmstudio_payload_uses_openai_content_blocks(self):
        """LM Studio path: content is a list of OpenAI-style content blocks."""
        client = self._capture_payload("openai")

        captured = {}

        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        try:
            agen = client.stream_response_with_tools(
                prompt="describe this",
                images=["data:image/png;base64,iVBORw0KGgo="],
                model="some-lmstudio-model",
            )
            async for _ in agen:
                break
        except Exception:
            pass

        # OpenAI path posts to /v1/chat/completions
        assert "/v1/chat/completions" in captured.get("url", ""), captured
        msg = captured["json"]["messages"][-1]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "describe this"}
        assert msg["content"][1]["type"] == "image_url"
        # Data URL kept intact for LM Studio (OpenAI spec accepts data URLs)
        assert msg["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="

    @pytest.mark.asyncio
    async def test_history_with_images_field_normalizes_for_ollama(self):
        """v0.3.3 §4 Defect 5: history entries stored as {content: str, images: [data_url]}
        must be replayed in the right shape per provider. Previously, replayed history
        entries with content_blocks shape crashed Ollama mid-conversation."""
        client = self._capture_payload("ollama")

        captured = {}

        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        history = [
            {
                "role": "user",
                "content": "describe this",
                "images": ["data:image/png;base64,iVBORw0KGgo="],
            },
            {"role": "assistant", "content": "looks like a placeholder image"},
        ]

        try:
            agen = client.stream_response_with_tools(
                prompt="follow up question",
                history=history,
                model="gemma4:31b",
            )
            async for _ in agen:
                break
        except Exception:
            pass

        msgs = captured["json"]["messages"]
        # First user message in history should be Ollama-native shape (string content + raw base64 images)
        user_hist = next(m for m in msgs if m["role"] == "user" and m.get("images"))
        assert user_hist["content"] == "describe this"
        assert user_hist["images"] == ["iVBORw0KGgo="]  # prefix stripped
        # No content_blocks anywhere
        for m in msgs:
            assert not isinstance(m.get("content"), list), f"Ollama path must not emit list content: {m}"

    @pytest.mark.asyncio
    async def test_history_with_legacy_content_blocks_normalized_for_ollama(self):
        """Legacy sessions may have content_blocks in history. Replayed via the Ollama
        path, these must be converted to native shape (string content + images sibling)."""
        client = self._capture_payload("ollama")

        captured = {}

        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                ],
            },
        ]

        try:
            agen = client.stream_response_with_tools(
                prompt="anything",
                history=history,
                model="gemma4:31b",
            )
            async for _ in agen:
                break
        except Exception:
            pass

        msgs = captured["json"]["messages"]
        user_hist = next(m for m in msgs if m["role"] == "user" and m.get("content") == "what is this")
        assert user_hist["images"] == ["iVBORw0KGgo="]
        for m in msgs:
            assert not isinstance(m.get("content"), list), f"Ollama path must not emit list content: {m}"

    @pytest.mark.asyncio
    async def test_image_only_send_with_empty_prompt_still_emits_user_message(self):
        """v0.3.3 §4 Defect 6 fix: previously `if prompt:` dropped the entire user
        turn when text was empty, even when images were present. The model would
        receive only system context and respond generically without ever seeing
        the image. The branch must fire whenever prompt OR images is non-empty."""
        client = self._capture_payload("ollama")

        captured = {}

        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        try:
            agen = client.stream_response_with_tools(
                prompt="",  # image-only send, no text caption
                images=["data:image/png;base64,iVBORw0KGgo="],
                model="gemma4:31b",
            )
            async for _ in agen:
                break
        except Exception:
            pass

        msgs = captured["json"]["messages"]
        # User message must be present (this was the regression — it was being dropped)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1, f"Expected exactly 1 user message, got {len(user_msgs)}: {msgs}"
        assert user_msgs[0]["content"] == ""
        assert user_msgs[0]["images"] == ["iVBORw0KGgo="]

    @pytest.mark.asyncio
    async def test_ollama_no_images_uses_plain_string_content(self):
        """Text-only Ollama request keeps the original plain-string content shape."""
        client = self._capture_payload("ollama")

        captured = {}

        class _Ctx:
            async def __aenter__(self_):
                response = AsyncMock()
                response.raise_for_status = MagicMock()
                response.status_code = 200
                response.aiter_lines = MagicMock(return_value=_empty_aiter())
                return response

            async def __aexit__(self_, *a):
                return False

        async def _empty_aiter():
            if False:
                yield ""

        def fake_stream_ctor(method, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Ctx()

        client.client.stream = fake_stream_ctor

        try:
            agen = client.stream_response_with_tools(
                prompt="hello",
                images=None,
                model="gemma4:31b",
            )
            async for _ in agen:
                break
        except Exception:
            pass

        msg = captured["json"]["messages"][-1]
        assert msg["content"] == "hello"
        assert "images" not in msg  # no images key when no images supplied

