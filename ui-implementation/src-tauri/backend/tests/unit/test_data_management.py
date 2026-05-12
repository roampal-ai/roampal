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
# v0.3.3 Section 9 — bulk /clear/* uses delete_collection+recreate
# ---------------------------------------------------------------------------

class TestClearLeavesNoPhantoms:
    """v0.3.3 Section 9: bulk-clear endpoints must use delete_collection+recreate.

    The old pattern (collection.delete(ids=batch)) left HNSW phantoms — actual
    root cause of issue #8 for users who triggered bulk-delete from the GUI's
    "Delete data" tab. New pattern matches /clear/books.
    """

    @pytest.mark.asyncio
    async def test_clear_memory_bank_recreates_collection(self):
        from app.routers.data_management import clear_memory_bank

        adapter = MagicMock()
        adapter.collection_name = "roampal_memory_bank"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=5)

        memory = MagicMock()
        memory.collections = {"memory_bank": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_memory_bank(request)

        adapter.client.delete_collection.assert_called_once_with(name="roampal_memory_bank")
        adapter.client.get_or_create_collection.assert_called_once()
        assert result["deleted_count"] == 5

    @pytest.mark.asyncio
    async def test_clear_working_recreates_collection(self):
        from app.routers.data_management import clear_working_memory

        adapter = MagicMock()
        adapter.collection_name = "roampal_working"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=12)

        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_working_memory(request)

        adapter.client.delete_collection.assert_called_once_with(name="roampal_working")
        adapter.client.get_or_create_collection.assert_called_once()
        assert result["deleted_count"] == 12

    @pytest.mark.asyncio
    async def test_clear_history_recreates_collection(self):
        from app.routers.data_management import clear_history

        adapter = MagicMock()
        adapter.collection_name = "roampal_history"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=3)

        memory = MagicMock()
        memory.collections = {"history": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await clear_history(request)
        adapter.client.delete_collection.assert_called_once_with(name="roampal_history")

    @pytest.mark.asyncio
    async def test_clear_patterns_recreates_collection(self):
        from app.routers.data_management import clear_patterns

        adapter = MagicMock()
        adapter.collection_name = "roampal_patterns"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=7)

        memory = MagicMock()
        memory.collections = {"patterns": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await clear_patterns(request)
        adapter.client.delete_collection.assert_called_once_with(name="roampal_patterns")

    @pytest.mark.asyncio
    async def test_clear_empty_collection_skips_recreate(self):
        """count_before=0 → skip the destructive path entirely."""
        from app.routers.data_management import clear_working_memory

        adapter = MagicMock()
        adapter.client = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=0)

        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_working_memory(request)

        adapter.client.delete_collection.assert_not_called()
        assert result["deleted_count"] == 0


# ---------------------------------------------------------------------------
# v0.3.3 Section 9.1 — conversational-tier clears unlink _completion_state.json
# ---------------------------------------------------------------------------

class TestSection91CompletionStateReset:
    """v0.3.3 Section 9.1: working/history/patterns clears must also unlink
    <DATA_PATH>/mcp_sessions/_completion_state.json. memory_bank and books
    clears must leave it alone."""

    @staticmethod
    def _seed_state(tmp_path):
        sessions_dir = tmp_path / "mcp_sessions"
        sessions_dir.mkdir()
        state_file = sessions_dir / "_completion_state.json"
        state_file.write_text('{"conv_abc": {"scored_this_turn": true}}')
        return state_file

    @staticmethod
    def _build_adapter(name, count):
        adapter = MagicMock()
        adapter.collection_name = f"roampal_{name}"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=count)
        return adapter

    @pytest.mark.asyncio
    async def test_clear_working_unlinks_completion_state(self, tmp_path, monkeypatch):
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        state_file = self._seed_state(tmp_path)
        adapter = self._build_adapter("working", 3)
        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await data_management.clear_working_memory(request)

        assert not state_file.exists()

    @pytest.mark.asyncio
    async def test_clear_history_unlinks_completion_state(self, tmp_path, monkeypatch):
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        state_file = self._seed_state(tmp_path)
        adapter = self._build_adapter("history", 2)
        memory = MagicMock()
        memory.collections = {"history": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await data_management.clear_history(request)

        assert not state_file.exists()

    @pytest.mark.asyncio
    async def test_clear_patterns_unlinks_completion_state(self, tmp_path, monkeypatch):
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        state_file = self._seed_state(tmp_path)
        adapter = self._build_adapter("patterns", 1)
        memory = MagicMock()
        memory.collections = {"patterns": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await data_management.clear_patterns(request)

        assert not state_file.exists()

    @pytest.mark.asyncio
    async def test_clear_memory_bank_leaves_completion_state(self, tmp_path, monkeypatch):
        """memory_bank holds permanent atomic facts, not turn lifecycle — leave the file alone."""
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        state_file = self._seed_state(tmp_path)
        adapter = self._build_adapter("memory_bank", 5)
        memory = MagicMock()
        memory.collections = {"memory_bank": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await data_management.clear_memory_bank(request)

        assert state_file.exists()

    @pytest.mark.asyncio
    async def test_clear_books_leaves_completion_state(self, tmp_path, monkeypatch):
        """books are reference docs, unrelated to conversation lifecycle — leave the file alone."""
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        state_file = self._seed_state(tmp_path)
        adapter = self._build_adapter("books", 0)  # 0 skips nuke
        memory = MagicMock()
        memory.collections = {"books": adapter}

        # Minimal book_processor that hits the "nothing to do" branches:
        # db_path/metadata_dir/uploads_dir all report non-existent.
        book_processor = MagicMock()
        book_processor.db_path = MagicMock()
        book_processor.db_path.exists = MagicMock(return_value=False)
        empty_dir = tmp_path / "books_data"
        book_processor.data_dir = empty_dir

        request = MagicMock()
        request.app.state.memory = memory
        request.app.state.book_processor = book_processor

        # Patch ghost registry to avoid touching real settings/paths.
        ghost = MagicMock()
        ghost.clear = MagicMock(return_value=0)
        with patch("app.routers.data_management.get_ghost_registry", return_value=ghost):
            await data_management.clear_books(request)

        assert state_file.exists()

    @pytest.mark.asyncio
    async def test_clear_working_handles_missing_state_file(self, tmp_path, monkeypatch):
        """Clear succeeds even when the state file is already absent."""
        from app.routers import data_management
        monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

        # No state file seeded — directory may not even exist.
        adapter = self._build_adapter("working", 3)
        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        # Should not raise.
        result = await data_management.clear_working_memory(request)
        assert result["status"] == "success"


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
