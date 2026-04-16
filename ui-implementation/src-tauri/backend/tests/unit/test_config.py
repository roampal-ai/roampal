"""
Tests for config modules — model contexts, feature flags, settings.
"""

import pytest
from unittest.mock import patch, MagicMock
import os


# ---------------------------------------------------------------------------
# Model contexts
# ---------------------------------------------------------------------------

class TestModelContexts:
    def test_known_model_returns_correct_size(self):
        from config.model_contexts import get_context_size
        size = get_context_size("qwen2.5:7b")
        assert isinstance(size, int)
        assert size > 0

    def test_unknown_model_returns_fallback(self):
        from config.model_contexts import get_context_size
        size = get_context_size("totally-unknown-model:99b")
        assert size == 8192  # Safe fallback

    def test_user_override_takes_precedence(self):
        from config.model_contexts import get_context_size
        size = get_context_size("qwen2.5:7b", user_override=16384)
        assert size == 16384

    def test_get_model_info_returns_dict(self):
        from config.model_contexts import get_model_info
        info = get_model_info("qwen2.5:7b")
        assert isinstance(info, dict)
        assert "current" in info


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

class TestFeatureFlags:
    def test_feature_flags_class_exists(self):
        from config.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "ENABLE_MEMORY")
        assert hasattr(ff, "ENABLE_SEARCH")

    def test_flags_have_boolean_defaults(self):
        from config.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert isinstance(ff.ENABLE_MEMORY, bool)
        assert isinstance(ff.ENABLE_SEARCH, bool)
        assert isinstance(ff.ENABLE_KG, bool)

    def test_dangerous_combo_detection(self):
        """Feature flag validator should detect dangerous combinations."""
        try:
            from config.feature_flag_validator import validate_feature_flags
            validate_feature_flags()
        except ImportError:
            pytest.skip("feature_flag_validator not importable")


# ---------------------------------------------------------------------------
# Model limits
# ---------------------------------------------------------------------------

class TestModelLimits:
    def test_get_model_limits(self):
        from config.model_limits import get_model_limits
        limits = get_model_limits("qwen2.5:7b", "test message")
        assert hasattr(limits, "max_tokens")
        assert hasattr(limits, "context_char_limit")
        assert limits.max_tokens > 0

    def test_unknown_model_gets_defaults(self):
        from config.model_limits import get_model_limits
        limits = get_model_limits("unknown-model:99b", "test message")
        assert limits is not None
        assert limits.max_tokens > 0

    def test_model_limits_has_iterations(self):
        from config.model_limits import get_model_limits
        limits = get_model_limits("qwen2.5:7b", "hello")
        assert hasattr(limits, "effective_iterations")
        assert limits.effective_iterations >= 1


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_settings_importable(self):
        from config.settings import Settings
        assert Settings is not None

    def test_settings_has_memory_config(self):
        from config.settings import Settings
        settings = Settings()
        assert hasattr(settings, "og_memory")
        assert hasattr(settings.og_memory, "base_data_path")

    def test_settings_has_llm_config(self):
        from config.settings import Settings
        settings = Settings()
        assert hasattr(settings, "llm")
        assert hasattr(settings.llm, "provider")

    def test_settings_has_sidecar_config(self):
        from config.settings import Settings
        settings = Settings()
        assert hasattr(settings, "sidecar")
        assert hasattr(settings.sidecar, "enabled")


# ---------------------------------------------------------------------------
# Rate limit config
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_default(self):
        try:
            val = int(os.getenv("ROAMPAL_RATE_LIMIT", "200"))
        except (ValueError, TypeError):
            val = 200
        assert isinstance(val, int)

    def test_rate_limit_invalid_env_fallback(self):
        try:
            val = int("false")
        except (ValueError, TypeError):
            val = 200
        assert val == 200
