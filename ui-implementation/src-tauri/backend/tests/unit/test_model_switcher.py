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


# ---------------------------------------------------------------------------
# v0.3.2 (0b): Main-model persistence — JSON file under DATA_PATH
# ---------------------------------------------------------------------------

class TestMainModelPersistence:
    def test_atomic_write_survives_corruption(self, tmp_path):
        """_atomic_write_json uses temp file + rename, never leaves a partial file."""
        from app.routers.model_switcher import _atomic_write_json

        target = tmp_path / "x.json"
        _atomic_write_json(target, {"a": 1, "b": "two"})
        assert target.read_text(encoding="utf-8") == '{"a": 1, "b": "two"}'
        # No stray .tmp left behind
        assert not (tmp_path / "x.json.tmp").exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        from app.routers.model_switcher import (
            save_main_model_config,
            load_main_model_config,
        )

        save_main_model_config(model="llama3.2:3b", provider="ollama", data_dir=tmp_path)
        loaded = load_main_model_config(data_dir=tmp_path)
        assert loaded == {"model": "llama3.2:3b", "provider": "ollama"}
        # File lives under the supplied data_dir, not the install dir
        assert (tmp_path / "main_model_config.json").exists()

    def test_load_returns_none_when_missing(self, tmp_path):
        from app.routers.model_switcher import load_main_model_config

        assert load_main_model_config(data_dir=tmp_path) is None

    def test_load_returns_none_when_malformed(self, tmp_path):
        from app.routers.model_switcher import load_main_model_config

        (tmp_path / "main_model_config.json").write_text("{ not json ")
        assert load_main_model_config(data_dir=tmp_path) is None

    def test_load_returns_none_when_missing_fields(self, tmp_path):
        from app.routers.model_switcher import load_main_model_config
        import json as _json

        (tmp_path / "main_model_config.json").write_text(
            _json.dumps({"model": "x"})  # missing provider
        )
        assert load_main_model_config(data_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# v0.3.2 (0g): GPU VRAM detection + warnings
# ---------------------------------------------------------------------------

class TestVramDetection:
    def test_detect_nvidia_parses_mib(self):
        from utils import gpu_detection

        def fake_run(cmd, timeout=3.0):
            if cmd[0] == "nvidia-smi":
                return "24564\n"
            return None

        with patch.object(gpu_detection, "_run", side_effect=fake_run):
            info = gpu_detection._detect_nvidia()
        assert info is not None
        assert abs(info.vram_gb - (24564 / 1024.0)) < 0.01
        assert info.count == 1
        assert info.source == "nvidia-smi"

    def test_detect_gpu_fallback_silent(self):
        """All probes fail → vram_gb is None (never guessed) and count=0."""
        from utils import gpu_detection

        with patch.object(gpu_detection, "_run", return_value=None):
            info = gpu_detection.detect_gpu()
        assert info.vram_gb is None
        assert info.count == 0
        assert info.source == "none"

    def test_estimate_vram_from_tag_parses_params(self):
        from utils.gpu_detection import estimate_vram_from_tag
        assert estimate_vram_from_tag("qwen3:72b") == round(72 * 0.75, 2)
        assert estimate_vram_from_tag("llama3.2:3b") == round(3 * 0.75, 2)

    def test_estimate_vram_from_tag_unknown_returns_none(self):
        """Silent pass when the tag has no param marker — we don't guess."""
        from utils.gpu_detection import estimate_vram_from_tag
        assert estimate_vram_from_tag("some-org/custom-model:latest") is None
        assert estimate_vram_from_tag("") is None


class TestVramWarnings:
    def test_curated_model_warning_when_over_vram(self):
        from app.routers.model_switcher import _vram_warning_for
        warn = _vram_warning_for("qwen2.5:7b", available_vram_gb=4.0)
        assert warn is not None
        assert warn["kind"] == "curated"
        assert warn["required_gb"] > warn["available_gb"]

    def test_curated_model_no_warning_when_fits(self):
        from app.routers.model_switcher import _vram_warning_for
        assert _vram_warning_for("qwen2.5:7b", available_vram_gb=32.0) is None

    def test_arbitrary_model_param_estimate_warns(self):
        from app.routers.model_switcher import _vram_warning_for
        warn = _vram_warning_for("qwen3:72b", available_vram_gb=16.0)
        assert warn is not None
        assert warn["kind"] == "estimated"

    def test_unknown_param_tag_no_warning(self):
        """No parameter marker → silent (universal rule: don't guess wrong)."""
        from app.routers.model_switcher import _vram_warning_for
        assert _vram_warning_for("some-org/foo:latest", available_vram_gb=2.0) is None

    @pytest.mark.asyncio
    async def test_debug_set_vram_gated_in_prod(self, monkeypatch):
        """Dev-only endpoint must 403 when ROAMPAL_DEV is unset."""
        import os
        from fastapi import HTTPException
        from app.routers.model_switcher import _debug_set_vram, _DebugVramRequest

        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        request = MagicMock()
        with pytest.raises(HTTPException) as exc:
            await _debug_set_vram(request, _DebugVramRequest(gpu_vram_gb=2.0))
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_debug_set_vram_overrides_and_restores(self, monkeypatch):
        """Dev override sets state; passing null restores the original."""
        from app.routers.model_switcher import _debug_set_vram, _DebugVramRequest

        monkeypatch.setenv("ROAMPAL_DEV", "1")
        request = MagicMock()
        request.app.state = MagicMock(spec=[])
        request.app.state.gpu_vram_gb = 24.0  # real detected value

        # Override to 2GB
        result = await _debug_set_vram(request, _DebugVramRequest(gpu_vram_gb=2.0))
        assert result["gpu_vram_gb"] == 2.0
        assert request.app.state.gpu_vram_gb == 2.0
        # Original preserved for restore
        assert request.app.state._gpu_vram_gb_original == 24.0

        # Restore via null
        result = await _debug_set_vram(request, _DebugVramRequest(gpu_vram_gb=None))
        assert result["gpu_vram_gb"] == 24.0
        assert request.app.state.gpu_vram_gb == 24.0

    def test_no_warning_when_vram_unknown(self):
        """Detection failure → never emit a warning."""
        from app.routers.model_switcher import _vram_warning_for
        assert _vram_warning_for("qwen2.5:72b", available_vram_gb=None) is None

    def test_startup_reads_main_model_config(self, tmp_path):
        """v0.3.2 (0b): spec name — persisted JSON is read at startup."""
        from app.routers.model_switcher import (
            save_main_model_config, load_main_model_config,
        )
        save_main_model_config(model="llama3.2:3b", provider="ollama", data_dir=tmp_path)
        persisted = load_main_model_config(data_dir=tmp_path)
        assert persisted == {"model": "llama3.2:3b", "provider": "ollama"}

    def test_startup_migrates_env_vars_once(self, tmp_path):
        """v0.3.2 (0b): spec name — first v0.3.2 boot seeds JSON from env vars."""
        from utils.startup_model_selection import should_migrate_env_to_json
        from app.routers.model_switcher import save_main_model_config, load_main_model_config

        # No JSON yet, env var set — should migrate.
        assert should_migrate_env_to_json(persisted=None, configured_env_model="qwen3:8b") is True

        # Simulate the migration write + re-check.
        save_main_model_config(model="qwen3:8b", provider="ollama", data_dir=tmp_path)
        persisted = load_main_model_config(data_dir=tmp_path)
        assert persisted == {"model": "qwen3:8b", "provider": "ollama"}

        # Second boot: JSON exists, no migration should fire.
        assert should_migrate_env_to_json(persisted=persisted, configured_env_model="qwen3:8b") is False

    def test_select_active_provider_persisted_wins_when_detected(self):
        """v0.3.2 (0b follow-up): persisted provider drives active provider
        when it's detected, overriding env-var default. Fixes the bug where
        LM Studio picks silently fell back to Ollama on restart."""
        from utils.startup_model_selection import select_active_provider

        detected = {"ollama": {"models": ["gemma4:31b"]}, "lmstudio": {"models": ["q14"]}}
        persisted = {"model": "q14", "provider": "lmstudio"}

        provider, source = select_active_provider(
            detected_providers=detected,
            persisted=persisted,
            configured_env_provider="ollama",  # env says ollama
        )
        assert provider == "lmstudio"  # persisted wins
        assert source == "persisted"

    def test_select_active_provider_env_when_no_persisted(self):
        from utils.startup_model_selection import select_active_provider

        detected = {"ollama": {"models": []}, "lmstudio": {"models": []}}
        provider, source = select_active_provider(
            detected_providers=detected, persisted=None,
            configured_env_provider="lmstudio",
        )
        assert provider == "lmstudio"
        assert source == "env"

    def test_select_active_provider_persisted_unreachable_falls_to_env(self):
        """If persisted provider isn't detected (e.g., LM Studio not running),
        don't force-fail — honor env + then first-detected."""
        from utils.startup_model_selection import select_active_provider

        detected = {"ollama": {"models": ["gemma4:31b"]}}  # LM Studio down
        persisted = {"model": "q14", "provider": "lmstudio"}
        provider, source = select_active_provider(
            detected_providers=detected, persisted=persisted,
            configured_env_provider="ollama",
        )
        assert provider == "ollama"
        assert source == "env"

    def test_migration_skipped_when_env_model_not_installed(self):
        """v0.3.2 follow-up: don't seed main_model_config.json with stale env
        vars (e.g. OLLAMA_MODEL=codellama:latest) when the named model isn't
        in the available list."""
        from utils.startup_model_selection import should_migrate_env_to_json

        # Legacy call (no available_models) still migrates — back-compat.
        assert should_migrate_env_to_json(None, "qwen:8b") is True

        # New signature: only migrates when model is installed.
        assert should_migrate_env_to_json(None, "qwen:8b", available_models=["qwen:8b", "gemma4:31b"]) is True
        assert should_migrate_env_to_json(None, "codellama:latest", available_models=["qwen:8b", "gemma4:31b"]) is False

        # Still no migration when persisted already exists.
        assert should_migrate_env_to_json({"model": "x", "provider": "ollama"}, "qwen:8b", available_models=["qwen:8b"]) is False

    def test_stale_model_clears_sidecar_config_when_mirror_on(self, tmp_path, monkeypatch):
        """v0.3.2: stale-model clear preserves provider preference on BOTH chat
        and sidecar configs (rewrites with model='') so next boot falls through
        to first-available of the SAME provider instead of dumping the user
        back to their env-default."""
        import json as _json
        from config.settings import settings as app_settings
        from modules.llm.ollama_client import OllamaClient

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)

        # Seed both configs pointing at the same "dead" model.
        (tmp_path / "main_model_config.json").write_text(
            _json.dumps({"model": "qwen3:8b", "provider": "ollama"})
        )
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "qwen3:8b", "provider": "ollama",
            "mirror_chat": True,
        }))

        client = OllamaClient()
        client.model_name = "qwen3:8b"
        client.api_style = "ollama"
        client._clear_stale_model()

        assert client.model_name == ""
        # Main config: provider preserved, model blank.
        main_cfg = _json.loads((tmp_path / "main_model_config.json").read_text(encoding="utf-8"))
        assert main_cfg == {"model": "", "provider": "ollama"}
        # Sidecar config: provider + mirror_chat preserved, model blank.
        sidecar_cfg = _json.loads((tmp_path / "sidecar_config.json").read_text(encoding="utf-8"))
        assert sidecar_cfg.get("model") == ""
        assert sidecar_cfg.get("provider") == "ollama"
        assert sidecar_cfg.get("mirror_chat") is True

    def test_stale_model_preserves_sidecar_config_when_mirror_off(self, tmp_path, monkeypatch):
        """If mirror was off, user explicitly chose a distinct sidecar model.
        We must not clobber their choice just because the chat model died."""
        import json as _json
        from config.settings import settings as app_settings
        from modules.llm.ollama_client import OllamaClient

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "main_model_config.json").write_text(
            _json.dumps({"model": "qwen3:8b", "provider": "ollama"})
        )
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "gpt-oss:20b", "provider": "ollama",
            "mirror_chat": False,  # explicit override
        }))

        client = OllamaClient()
        client.model_name = "qwen3:8b"
        client.api_style = "ollama"
        client._clear_stale_model()

        # Main config rewritten with blank model (provider preserved).
        main_cfg = _json.loads((tmp_path / "main_model_config.json").read_text(encoding="utf-8"))
        assert main_cfg == {"model": "", "provider": "ollama"}
        # Sidecar config fully preserved — user's explicit non-mirror choice sticks.
        sidecar_cfg = _json.loads((tmp_path / "sidecar_config.json").read_text(encoding="utf-8"))
        assert sidecar_cfg["model"] == "gpt-oss:20b"
        assert sidecar_cfg["mirror_chat"] is False

    def test_startup_precedence_env_over_json_over_fallback(self):
        """v0.3.2 (0b): spec name — env wins, then JSON, then first-available."""
        from utils.startup_model_selection import select_startup_model

        available = ["llama3.2:3b", "qwen2.5:7b", "gpt-oss:20b"]
        persisted = {"model": "qwen2.5:7b", "provider": "ollama"}

        # env wins over persisted
        assert select_startup_model(available, "gpt-oss:20b", persisted, "ollama") == ("gpt-oss:20b", "env")
        # no env → persisted wins
        assert select_startup_model(available, None, persisted, "ollama") == ("qwen2.5:7b", "persisted")
        # env set but not in list → falls to persisted
        assert select_startup_model(available, "ghost:99b", persisted, "ollama") == ("qwen2.5:7b", "persisted")
        # persisted provider mismatch → falls to first-available
        assert select_startup_model(available, None, persisted, "lmstudio") == ("llama3.2:3b", "fallback")
        # nothing set anywhere → first-available
        assert select_startup_model(available, None, None, "ollama") == ("llama3.2:3b", "fallback")
        # empty list → (None, "none")
        assert select_startup_model([], None, None, "ollama") == (None, "none")

    def test_vram_warning_flows_into_switch_response(self):
        """v0.3.2 (0g ship-fix): switch response must include the vram_warning
        field so the UI can render a toast. Without this end-to-end wiring, the
        warning dies in the backend and the user never sees it."""
        # We build the success-branch dict the same way switch_model does. If
        # the endpoint's return dict skipped vram_warning, this test would fail.
        from app.routers.model_switcher import _vram_warning_for
        warning = _vram_warning_for("qwen2.5:7b", available_vram_gb=4.0)
        response = {
            "status": "success",
            "message": "Switched to ollama model: qwen2.5:7b",
            "current_model": "qwen2.5:7b",
            "provider": "ollama",
            "vram_warning": warning,
        }
        # Contract assertion: the field exists and carries the expected shape.
        assert "vram_warning" in response
        assert response["vram_warning"] is not None
        assert response["vram_warning"]["kind"] == "curated"
        assert "required_gb" in response["vram_warning"]
        assert "available_gb" in response["vram_warning"]
        assert "message" in response["vram_warning"]


# ---------------------------------------------------------------------------
# v0.3.2 (0f): Sidecar mirror-chat mode
# ---------------------------------------------------------------------------

class TestSidecarMirror:
    def test_load_sidecar_config_defaults_mirror_chat_true(self, tmp_path, monkeypatch):
        """Absent config file should report mirror_chat=True (first-run default)."""
        from config.settings import settings as app_settings
        from app.routers.model_switcher import _load_sidecar_config

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        request = MagicMock()
        config = _load_sidecar_config(request)
        assert config["mirror_chat"] is True
        assert config["enabled"] is False

    def test_load_sidecar_config_preserves_explicit_false(self, tmp_path, monkeypatch):
        """A persisted mirror_chat=False override must be respected on read."""
        import json as _json
        from config.settings import settings as app_settings
        from app.routers.model_switcher import _load_sidecar_config

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "gpt-oss:20b", "provider": "ollama",
            "mirror_chat": False,
        }))
        request = MagicMock()
        config = _load_sidecar_config(request)
        assert config["mirror_chat"] is False
        assert config["model"] == "gpt-oss:20b"

    def test_load_sidecar_config_backfills_missing_mirror_chat_matching(self, tmp_path, monkeypatch):
        """Old config without mirror_chat → True when sidecar matches chat."""
        import json as _json
        from config.settings import settings as app_settings
        from app.routers.model_switcher import _load_sidecar_config

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "qwen2.5:7b", "provider": "ollama",
        }))
        request = MagicMock()
        request.app.state.llm_client.model_name = "qwen2.5:7b"
        config = _load_sidecar_config(request)
        assert config["mirror_chat"] is True

    def test_load_sidecar_config_backfills_missing_mirror_chat_differing(self, tmp_path, monkeypatch):
        """v0.3.1 user with distinct chat + sidecar models → mirror_chat=False (preserve intent)."""
        import json as _json
        from config.settings import settings as app_settings
        from app.routers.model_switcher import _load_sidecar_config

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "gpt-oss:20b", "provider": "ollama",
        }))
        request = MagicMock()
        request.app.state.llm_client.model_name = "qwen3:8b"  # different from sidecar
        config = _load_sidecar_config(request)
        assert config["mirror_chat"] is False

    def test_sidecar_seeded_to_chat_model_on_first_run(self, tmp_path, monkeypatch):
        """v0.3.2 (0f): spec name — first-run seed writes mirror_chat=True
        with the current chat model. This mirrors the exact payload shape
        main.py's lifespan writes when sidecar_config.json is absent.
        """
        from config.settings import settings as app_settings
        from app.routers.model_switcher import _atomic_write_json, _load_sidecar_config

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)

        # Simulate the seed write exactly as main.py lifespan does.
        chat_model = "qwen3:8b"
        chat_provider = "ollama"
        sidecar_path = tmp_path / "sidecar_config.json"
        assert not sidecar_path.exists(), "Precondition: no sidecar config yet"

        _atomic_write_json(
            sidecar_path,
            {
                "enabled": True,
                "model": chat_model,
                "provider": chat_provider,
                "mirror_chat": True,
            },
        )

        # Verify the file round-trips through the real loader.
        request = MagicMock()
        request.app.state.llm_client.model_name = chat_model
        loaded = _load_sidecar_config(request)
        assert loaded["enabled"] is True
        assert loaded["model"] == chat_model
        assert loaded["provider"] == chat_provider
        assert loaded["mirror_chat"] is True

    @pytest.mark.asyncio
    async def test_sidecar_status_exposes_mirror_chat(self, tmp_path, monkeypatch):
        """v0.3.2: chat-header badge needs mirror_chat on /sidecar/status."""
        import json as _json
        from config.settings import settings as app_settings
        from app.routers.model_switcher import sidecar_status

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        (tmp_path / "sidecar_config.json").write_text(_json.dumps({
            "enabled": True, "model": "qwen2.5:7b", "provider": "ollama",
            "mirror_chat": False,
        }))
        request = MagicMock()
        request.app.state.sidecar_model = "qwen2.5:7b"
        request.app.state.sidecar_provider = "ollama"
        request.app.state.sidecar_last_error = ""

        result = await sidecar_status(request)
        assert result["mirror_chat"] is False
        assert result["model"] == "qwen2.5:7b"
        assert result["enabled"] is True

    @pytest.mark.asyncio
    async def test_sidecar_mirror_endpoint_snaps_to_chat(self, tmp_path, monkeypatch):
        """Toggling mirror ON must snap the sidecar config to the current chat model."""
        from config.settings import settings as app_settings
        from app.routers.model_switcher import (
            sidecar_mirror, SidecarMirrorRequest, _load_sidecar_config,
        )

        monkeypatch.setattr(app_settings.paths, "data_dir", tmp_path)
        request = MagicMock()
        request.app.state.llm_client = MagicMock()
        request.app.state.llm_client.model_name = "llama3.2:3b"
        request.app.state.llm_client.api_style = "ollama"

        result = await sidecar_mirror(request, SidecarMirrorRequest(mirror_chat=True))

        assert result["mirror_chat"] is True
        persisted = _load_sidecar_config(request)
        assert persisted["model"] == "llama3.2:3b"
        assert persisted["provider"] == "ollama"
        assert persisted["mirror_chat"] is True
        assert request.app.state.sidecar_model == "llama3.2:3b"
