"""
Tests for model_switcher.py — sidecar endpoints, model validation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Sidecar status
# ---------------------------------------------------------------------------

class TestSidecarStatus:
    @pytest.mark.asyncio
    async def test_status_enabled(self):
        from app.routers.model_switcher import sidecar_status
        request = MagicMock()
        request.app.state.sidecar_model = "gpt-oss:20b"
        request.app.state.sidecar_provider = "ollama"
        request.app.state.sidecar_last_error = ""
        result = await sidecar_status(request)
        assert result["enabled"] is True
        assert result["model"] == "gpt-oss:20b"
        assert result["provider"] == "ollama"
        assert result["last_error"] == ""

    @pytest.mark.asyncio
    async def test_status_disabled(self):
        from app.routers.model_switcher import sidecar_status
        request = MagicMock()
        request.app.state.sidecar_model = ""
        request.app.state.sidecar_provider = ""
        request.app.state.sidecar_last_error = ""
        result = await sidecar_status(request)
        assert result["enabled"] is False

    @pytest.mark.asyncio
    async def test_status_with_error(self):
        from app.routers.model_switcher import sidecar_status
        request = MagicMock()
        request.app.state.sidecar_model = "qwen2.5:7b"
        request.app.state.sidecar_provider = "lmstudio"
        request.app.state.sidecar_last_error = "Scoring failed"
        result = await sidecar_status(request)
        assert result["last_error"] == "Scoring failed"


# ---------------------------------------------------------------------------
# Sidecar disable
# ---------------------------------------------------------------------------

class TestSidecarDisable:
    @pytest.mark.asyncio
    async def test_disable_clears_state(self):
        from app.routers.model_switcher import sidecar_disable
        request = MagicMock()
        request.app.state.sidecar_model = "gpt-oss:20b"
        request.app.state.sidecar_provider = "ollama"
        request.app.state.sidecar_client = MagicMock()

        with patch("app.routers.model_switcher._save_sidecar_config"):
            result = await sidecar_disable(request)

        assert result["status"] == "ok"
        assert result["enabled"] is False
        assert request.app.state.sidecar_model == ""
        assert request.app.state.sidecar_client is None


# ---------------------------------------------------------------------------
# Model name validation
# ---------------------------------------------------------------------------

class TestModelValidation:
    def test_valid_model_names(self):
        from app.routers.model_switcher import _validate_model_name
        assert _validate_model_name("gpt-oss:20b") is True
        assert _validate_model_name("qwen2.5:7b") is True
        assert _validate_model_name("llama3.3:70b") is True

    def test_invalid_model_name_special_chars(self):
        from app.routers.model_switcher import _validate_model_name
        assert _validate_model_name("model; rm -rf /") is False

    def test_empty_model_name(self):
        from app.routers.model_switcher import _validate_model_name
        assert _validate_model_name("") is False

    def test_model_name_with_spaces(self):
        from app.routers.model_switcher import _validate_model_name
        assert _validate_model_name("model name with spaces") is False


# ---------------------------------------------------------------------------
# Sidecar set
# ---------------------------------------------------------------------------

class TestSidecarSet:
    def test_sidecar_set_request_model(self):
        """SidecarSetRequest validates fields."""
        from app.routers.model_switcher import SidecarSetRequest
        req = SidecarSetRequest(model="gpt-oss:20b", provider="ollama")
        assert req.model == "gpt-oss:20b"
        assert req.provider == "ollama"

    def test_sidecar_set_request_lmstudio(self):
        from app.routers.model_switcher import SidecarSetRequest
        req = SidecarSetRequest(model="qwen2.5-7b-instruct", provider="lmstudio")
        assert req.provider == "lmstudio"

    def test_save_sidecar_config_importable(self):
        """_save_sidecar_config function exists and is callable."""
        from app.routers.model_switcher import _save_sidecar_config
        assert callable(_save_sidecar_config)
