"""
Tests for utils/sidecar_tag_wrapper.py — v0.3.2 (Bug 4) regression coverage.

The factory returns an async wrapper that must read sidecar
client/model from app_state at CALL time, not at construction time.
Prior inline closure captured boot-time values into local vars and
silently failed whenever the user switched sidecar mid-session.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from utils.sidecar_tag_wrapper import make_llm_tag_extractor


class TestMakeLLMTagExtractor:
    """v0.3.2 (Bug 4): wrapper must dynamically track app_state sidecar swaps."""

    @pytest.mark.asyncio
    async def test_wrapper_picks_up_sidecar_swap(self):
        """Boot-time closure capture was the bug. Set wrapper up with one
        client; swap app_state.sidecar_client to a new one; next wrapper
        call must use the NEW client."""
        app_state = SimpleNamespace(
            sidecar_client="stale_client",
            sidecar_model="stale_model",
        )

        wrapper = make_llm_tag_extractor(app_state)

        # Mutate app_state as /sidecar/set or /sidecar/mirror would
        app_state.sidecar_client = "fresh_client"
        app_state.sidecar_model = "fresh_model"

        with patch(
            "utils.sidecar_tag_wrapper.extract_noun_tags", new=AsyncMock(return_value=["tag"])
        ) as mock_extract:
            await wrapper("some text")

        mock_extract.assert_awaited_once()
        call_kwargs = mock_extract.await_args.kwargs
        # Proves the wrapper read fresh values at call time, not stale ones
        assert call_kwargs["client"] == "fresh_client"
        assert call_kwargs["model"] == "fresh_model"

    @pytest.mark.asyncio
    async def test_wrapper_returns_empty_when_sidecar_cleared(self):
        """When user toggles /sidecar/disable, app_state.sidecar_client is
        set to None. Wrapper must return [] rather than crash on None client."""
        app_state = SimpleNamespace(sidecar_client="client", sidecar_model="model")
        wrapper = make_llm_tag_extractor(app_state)

        app_state.sidecar_client = None  # /sidecar/disable set this
        app_state.sidecar_model = ""

        tags = await wrapper("some text")
        assert tags == []

    @pytest.mark.asyncio
    async def test_wrapper_returns_empty_on_extraction_failure(self):
        """Connection errors, timeouts, JSON parse failures → [] (benchmark-aligned)
        with WARNING log, not crash upward."""
        app_state = SimpleNamespace(sidecar_client="c", sidecar_model="m")
        wrapper = make_llm_tag_extractor(app_state)

        with patch(
            "utils.sidecar_tag_wrapper.extract_noun_tags",
            new=AsyncMock(side_effect=ConnectionError("sidecar down")),
        ):
            tags = await wrapper("text")

        assert tags == []

    @pytest.mark.asyncio
    async def test_wrapper_normalizes_none_to_empty_list(self):
        """extract_noun_tags returns None when the LLM output can't be parsed;
        wrapper must convert None → [] so downstream `if tags:` checks work."""
        app_state = SimpleNamespace(sidecar_client="c", sidecar_model="m")
        wrapper = make_llm_tag_extractor(app_state)

        with patch(
            "utils.sidecar_tag_wrapper.extract_noun_tags",
            new=AsyncMock(return_value=None),
        ):
            tags = await wrapper("text")

        assert tags == []

    @pytest.mark.asyncio
    async def test_wrapper_passes_text_through_unchanged(self):
        """Regression guard: text arg must pass through untouched. If the
        wrapper ever starts sampling / truncating / wrapping text, the
        LLM-side prompt construction breaks in subtle ways."""
        app_state = SimpleNamespace(sidecar_client="c", sidecar_model="m")
        wrapper = make_llm_tag_extractor(app_state)

        with patch(
            "utils.sidecar_tag_wrapper.extract_noun_tags",
            new=AsyncMock(return_value=["a"]),
        ) as mock_extract:
            await wrapper("Logan went to Boston.")

        assert mock_extract.await_args.kwargs["text"] == "Logan went to Boston."
