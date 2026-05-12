"""Tests for dynamic context-limit fetch (Section 6, v0.3.3)."""

import pytest
import time
from unittest.mock import patch, MagicMock, AsyncMock
import json


@pytest.fixture(autouse=True)
def clean_caches(tmp_path):
    """Clear in-memory caches AND isolate the disk cache to a tmp path
    between tests. v0.3.3 §4H fixed a latent bug where the disk write was
    silently failing; with the fix in place, tests would pollute each other
    unless isolated."""
    import config.model_contexts as mc
    mc._context_cache.clear()
    orig_disk = mc.MODEL_METADATA_CACHE_FILE
    mc.MODEL_METADATA_CACHE_FILE = str(tmp_path / "test_metadata_cache.json")
    try:
        yield
    finally:
        mc._context_cache.clear()
        mc.MODEL_METADATA_CACHE_FILE = orig_disk


class TestOllamaParsing:
    @pytest.mark.asyncio
    async def test_qwen36_returns_262144(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"qwen35moe.context_length": 262144}}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("qwen3.6:35b-a3b", "ollama", "http://localhost:11434")
            assert result == 262144

    @pytest.mark.asyncio
    async def test_llama_returns_131072(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"llama.context_length": 131072}}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("llama3.1:8b", "ollama", "http://localhost:11434")
            assert result == 131072

    @pytest.mark.asyncio
    async def test_gemma3_returns_131072(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"gemma3.context_length": 131072}}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("gemma3:12b", "ollama", "http://localhost:11434")
            assert result == 131072

    @pytest.mark.asyncio
    async def test_no_context_length_key_returns_none(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"general.architecture": "llama"}}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("llama3.1:8b", "ollama", "http://localhost:11434")
            assert result is None

    @pytest.mark.asyncio
    async def test_404_returns_none(self):
        from config.model_contexts import fetch_context_from_provider as fc
        import httpx
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404))
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("nonexistent", "ollama", "http://localhost:11434")
            assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        from config.model_contexts import fetch_context_from_provider as fc
        import httpx
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("llama3.1:8b", "ollama", "http://localhost:11434")
            assert result is None


class TestLMStudioParsing:
    @pytest.mark.asyncio
    async def test_loaded_wins_over_max(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"id": "m", "loaded_context_length": 65536, "max_context_length": 262144}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("m", "lmstudio", "http://localhost:1234")
            assert result == 65536

    @pytest.mark.asyncio
    async def test_fallback_to_max(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"id": "m", "max_context_length": 262144}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("m", "lmstudio", "http://localhost:1234")
            assert result == 262144

    @pytest.mark.asyncio
    async def test_fallback_to_context_length(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"id": "m", "context_length": 32768}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("m", "lmstudio", "http://localhost:1234")
            assert result == 32768

    @pytest.mark.asyncio
    async def test_no_fields_returns_none(self):
        from config.model_contexts import fetch_context_from_provider as fc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"id": "m"}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("m", "lmstudio", "http://localhost:1234")
            assert result is None

    @pytest.mark.asyncio
    async def test_lmstudio_404_returns_none(self):
        from config.model_contexts import fetch_context_from_provider as fc
        import httpx
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404))
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            result = await fc("nonexistent", "lmstudio", "http://localhost:1234")
            assert result is None


class TestMemoryCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_network(self):
        import config.model_contexts as mc
        count = [0]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"llama.context_length": 131072}}

        async def cnt_post(*a, **k):
            count[0] += 1
            return resp

        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = cnt_post
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst
            r1 = await mc.get_context_size_async("llama3.1:8b", provider="ollama", base_url="http://localhost:11434")
            r2 = await mc.get_context_size_async("llama3.1:8b", provider="ollama", base_url="http://localhost:11434")

        assert count[0] == 1
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self, tmp_path):
        """Memory cache TTL=0 forces re-fetch each call.

        v0.3.3 §4H fix: also redirect disk cache to a non-existent tmp path so
        the now-working disk persistence doesn't mask the memory-TTL behavior
        under test. (Pre-fix, the disk write was silently failing, so the disk
        cache never hit; the test passed by accident.)
        """
        import config.model_contexts as mc
        orig_ttl = mc._CACHE_TTL
        orig_cache = mc.MODEL_METADATA_CACHE_FILE
        count = [0]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"llama.context_length": 131072}}

        async def cnt_post(*a, **k):
            count[0] += 1
            return resp

        mc._CACHE_TTL = 0
        mc.MODEL_METADATA_CACHE_FILE = str(tmp_path / "nonexistent_disk_cache.json")
        try:
            with patch("config.model_contexts.httpx.AsyncClient") as MC:
                inst = AsyncMock()
                inst.post = cnt_post
                inst.__aenter__ = AsyncMock(return_value=inst)
                inst.__aexit__ = AsyncMock(return_value=False)
                MC.return_value = inst
                r1 = await mc.get_context_size_async("llama3.1:8b", provider="ollama", base_url="http://localhost:11434")
                # Wipe the disk cache between calls so memory TTL is isolated
                import os as _os
                if _os.path.exists(mc.MODEL_METADATA_CACHE_FILE):
                    _os.unlink(mc.MODEL_METADATA_CACHE_FILE)
                r2 = await mc.get_context_size_async("llama3.1:8b", provider="ollama", base_url="http://localhost:11434")

            assert count[0] == 2
        finally:
            mc._CACHE_TTL = orig_ttl
            mc.MODEL_METADATA_CACHE_FILE = orig_cache


class TestDiskCache:
    def test_survives_restart(self, tmp_path):
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            disk_data = {
                "ollama::llama3.1:8b": {"context_length": 131072, "expires_at": time.time() + 3600}
            }
            with open(cache_file, "w") as f:
                json.dump(disk_data, f)
            mc._context_cache.clear()

        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig
            mc._context_cache.clear()

    def test_expired_disk_ignored(self, tmp_path):
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file

        async def mock_post(*a, **k):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"model_info": {"llama.context_length": 131072}}
            return r

        try:
            disk_data = {
                "ollama::llama3.1:8b": {"context_length": 9999, "expires_at": time.time() - 1}
            }
            with open(cache_file, "w") as f:
                json.dump(disk_data, f)
            mc._context_cache.clear()

            with patch("config.model_contexts.httpx.AsyncClient") as MC:
                inst = AsyncMock()
                inst.post = mock_post
                inst.__aenter__ = AsyncMock(return_value=inst)
                inst.__aexit__ = AsyncMock(return_value=False)
                MC.return_value = inst

        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig
            mc._context_cache.clear()


class TestPriorityChain:
    @pytest.mark.asyncio
    async def test_dynamic_wins_over_hardcoded(self):
        import config.model_contexts as mc
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"qwen35moe.context_length": 262144}}
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(return_value=resp)
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst

    @pytest.mark.asyncio
    async def test_fallback_to_hardcoded_on_timeout(self):
        import config.model_contexts as mc
        import httpx
        with patch("config.model_contexts.httpx.AsyncClient") as MC:
            inst = AsyncMock()
            inst.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            MC.return_value = inst

    @pytest.mark.asyncio
    async def test_user_override_beats_dynamic(self):
        import config.model_contexts as mc
        with patch.object(mc, "load_user_overrides", return_value={"qwen3.6:35b-a3b": 65536}):
            pass

    def test_sync_unchanged(self):
        from config.model_contexts import get_context_size
        assert isinstance(get_context_size("qwen2.5:7b"), int)
        assert get_context_size("unknown-model") == 8192
        assert get_context_size("qwen2.5:7b", user_override=16384) == 16384


class TestCachedVisionDetection:
    """Tests for sync vision cache lookup (get_cached_vision) — v0.3.3 §4H
    refactor reads from the dynamic capabilities cache, no hardcoded list."""

    def test_cached_vision_capability_returns_true(self, tmp_path):
        """Cache populated with vision capability → get_cached_vision True."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            mc._save_caps_cache(
                mc._caps_cache_key("custom-vision-model", "ollama"),
                {"vision", "completion"},
            )
            assert mc.get_cached_vision("custom-vision-model", "ollama") is True
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    def test_cached_without_vision_returns_false(self, tmp_path):
        """Cache populated without vision in caps → get_cached_vision False."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            mc._save_caps_cache(
                mc._caps_cache_key("text-only-model", "ollama"),
                {"completion", "tools"},
            )
            assert mc.get_cached_vision("text-only-model", "ollama") is False
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    def test_uncached_unknown_defaults_false(self):
        """Cold-start cache miss → False (no hardcoded heuristic anymore)."""
        from config.model_contexts import get_cached_vision, _load_caps_cache
        cache = _load_caps_cache()
        assert "caps::ollama::__nonexistent_model_xyz__" not in cache
        result = get_cached_vision("__nonexistent_model_xyz__", "ollama")
        assert result is False

    def test_disk_cache_round_trip(self, tmp_path):
        """End-to-end: write capabilities to disk, read back via the public reader."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            disk_data = {
                "caps::ollama::gemma4:31b": {
                    "capabilities": ["completion", "thinking", "tools", "vision"],
                    "expires_at": time.time() + 86400,
                }
            }
            with open(cache_file, "w") as f:
                json.dump(disk_data, f)

            assert mc.get_cached_vision("gemma4:31b", "ollama") is True
            assert mc.get_cached_tools("gemma4:31b", "ollama") is True
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig


class TestCachedToolsDetection:
    """v0.3.3 §4H: get_cached_tools reads dynamic capability cache."""

    def test_ollama_with_tools_returns_true(self, tmp_path):
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            mc._save_caps_cache(mc._caps_cache_key("qwen3:32b", "ollama"), {"tools", "completion"})
            assert mc.get_cached_tools("qwen3:32b", "ollama") is True
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    def test_ollama_without_tools_returns_false(self, tmp_path):
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            mc._save_caps_cache(mc._caps_cache_key("gemma2:9b", "ollama"), {"completion"})
            assert mc.get_cached_tools("gemma2:9b", "ollama") is False
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    def test_lmstudio_cache_miss_defaults_true(self, tmp_path):
        """LM Studio's OpenAI API has no capability metadata; default True so
        the model isn't preemptively hidden. Runtime tool-incompatibility
        detection in ollama_client retries without tools on 400."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            # Empty cache for this model
            assert mc.get_cached_tools("some-lmstudio-model", "lmstudio") is True
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    def test_lmstudio_with_explicit_caps_respects_cache(self, tmp_path):
        """If LM Studio probe (or other source) writes caps, respect them."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file
        try:
            mc._save_caps_cache(mc._caps_cache_key("foo", "lmstudio"), {"completion"})
            # caps cache populated, tools not in it → False even for lmstudio
            assert mc.get_cached_tools("foo", "lmstudio") is False
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig


class TestFetchOllamaCapabilities:
    """v0.3.3 §4H: fetch_ollama_capabilities asks /api/show for the model's
    self-declared capability list."""

    @pytest.mark.asyncio
    async def test_returns_capability_set_on_200(self, tmp_path, monkeypatch):
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"capabilities": ["completion", "vision", "tools", "thinking"]}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None):
                return FakeResponse()

        monkeypatch.setattr(mc.httpx, "AsyncClient", lambda **kw: FakeClient())
        try:
            caps = await mc.fetch_ollama_capabilities("gemma4:31b", "http://x:11434")
            assert caps == {"completion", "vision", "tools", "thinking"}
            # Should have written to cache
            assert mc.get_cached_vision("gemma4:31b", "ollama") is True
            assert mc.get_cached_tools("gemma4:31b", "ollama") is True
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    @pytest.mark.asyncio
    async def test_missing_capabilities_key_returns_empty_set(self, tmp_path, monkeypatch):
        """Old Ollama versions without the capabilities field → empty set, no crash."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"modelfile": "...", "details": {}}  # No capabilities key

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None):
                return FakeResponse()

        monkeypatch.setattr(mc.httpx, "AsyncClient", lambda **kw: FakeClient())
        try:
            caps = await mc.fetch_ollama_capabilities("legacy-model", "http://x:11434")
            assert caps == set()
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig

    @pytest.mark.asyncio
    async def test_network_failure_returns_none(self, tmp_path, monkeypatch):
        """httpx error → None, cache untouched."""
        import config.model_contexts as mc
        cache_file = str(tmp_path / "mc.json")
        orig = mc.MODEL_METADATA_CACHE_FILE
        mc.MODEL_METADATA_CACHE_FILE = cache_file

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None):
                raise mc.httpx.HTTPError("connection refused")

        monkeypatch.setattr(mc.httpx, "AsyncClient", lambda **kw: FakeClient())
        try:
            caps = await mc.fetch_ollama_capabilities("foo", "http://x:11434")
            assert caps is None
            assert mc.get_cached_capabilities("foo", "ollama") == set()
        finally:
            mc.MODEL_METADATA_CACHE_FILE = orig
