"""
Tests for data_management.py — summarize tool, clear endpoints, error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Summarize scan
# ---------------------------------------------------------------------------

class TestSummarizeScan:
    @pytest.mark.asyncio
    async def test_scan_returns_result(self):
        from app.routers.data_management import summarize_scan

        request = MagicMock()
        memory = MagicMock()
        for name in ["working", "history", "patterns"]:
            adapter = MagicMock()
            adapter.get_all_ids = MagicMock(return_value=[])
            memory.collections[name] = adapter
        memory.collections = {
            "working": MagicMock(get_all_ids=MagicMock(return_value=[])),
            "history": MagicMock(get_all_ids=MagicMock(return_value=[])),
            "patterns": MagicMock(get_all_ids=MagicMock(return_value=[])),
        }
        request.app.state.memory = memory
        result = await summarize_scan(request)
        assert "total" in result

    @pytest.mark.asyncio
    async def test_summarize_requires_sidecar(self):
        from app.routers.data_management import summarize_run

        request = MagicMock()
        request.app.state.sidecar_client = None
        request.app.state.sidecar_model = ""
        request.app.state.memory = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await summarize_run(request)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Clear endpoints — verify they require memory system
# ---------------------------------------------------------------------------

class TestClearEndpoints:
    @pytest.mark.asyncio
    async def test_clear_working_requires_memory(self):
        from app.routers.data_management import clear_working_memory
        request = MagicMock()
        request.app.state.memory = None
        with pytest.raises(HTTPException) as exc_info:
            await clear_working_memory(request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_clear_memory_bank_requires_memory(self):
        from app.routers.data_management import clear_memory_bank
        request = MagicMock()
        request.app.state.memory = None
        with pytest.raises(HTTPException) as exc_info:
            await clear_memory_bank(request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_clear_history_requires_memory(self):
        from app.routers.data_management import clear_history
        request = MagicMock()
        request.app.state.memory = None
        with pytest.raises(HTTPException) as exc_info:
            await clear_history(request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_clear_patterns_requires_memory(self):
        from app.routers.data_management import clear_patterns
        request = MagicMock()
        request.app.state.memory = None
        with pytest.raises(HTTPException) as exc_info:
            await clear_patterns(request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_clear_requires_collection(self):
        """Clear should 404 if collection doesn't exist."""
        from app.routers.data_management import clear_working_memory
        request = MagicMock()
        memory = MagicMock()
        memory.collections = {}  # No working collection
        request.app.state.memory = memory
        with pytest.raises(HTTPException) as exc_info:
            await clear_working_memory(request)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_get_stats_no_memory(self):
        from app.routers.data_management import get_data_stats
        request = MagicMock()
        request.app.state.memory = None
        with pytest.raises(HTTPException) as exc_info:
            await get_data_stats(request)
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Summarize cancel
# ---------------------------------------------------------------------------

class TestSummarizeCancel:
    @pytest.mark.asyncio
    async def test_cancel_returns_ok(self):
        from app.routers.data_management import summarize_cancel
        result = await summarize_cancel()
        assert "cancel" in result["status"].lower()
