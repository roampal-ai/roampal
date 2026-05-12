"""
Unit Tests for MemoryBankService

Tests the extracted memory bank operations.
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from modules.memory.memory_bank_service import MemoryBankService
from modules.memory.config import MemoryConfig


class TestMemoryBankServiceInit:
    """Test MemoryBankService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        collection = MagicMock()
        service = MemoryBankService(
            collection=collection,
            embed_fn=AsyncMock()
        )
        assert service.config is not None
        assert service.MAX_ITEMS == 500

    def test_init_with_custom_config(self):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        service = MemoryBankService(
            collection=MagicMock(),
            embed_fn=AsyncMock(),
            config=config
        )
        assert service.config.promotion_score_threshold == 0.8


class TestStore:
    """Test memory storage."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.collection = MagicMock()
        coll.collection.count = MagicMock(return_value=10)
        coll.upsert_vectors = AsyncMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_store_basic(self, service, mock_collection):
        """Should store memory with correct metadata."""
        doc_id = await service.store(
            text="User prefers dark mode",
            tags=["preference"]
        )

        assert doc_id.startswith("memory_bank_")
        mock_collection.upsert_vectors.assert_called_once()

        call_args = mock_collection.upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["text"] == "User prefers dark mode"
        assert metadata["status"] == "active"
        assert metadata["score"] == 1.0
        assert json.loads(metadata["tags"]) == ["preference"]

    @pytest.mark.asyncio
    async def test_store_with_importance_confidence(self, service, mock_collection):
        """Should store with custom importance/confidence."""
        await service.store(
            text="Critical info",
            tags=["identity"],
            importance=0.95,
            confidence=0.9
        )

        call_args = mock_collection.upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["importance"] == 0.95
        assert metadata["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_store_capacity_check(self, service, mock_collection):
        """Should reject when at capacity (active entries only, excludes archived)."""
        # Simulate 500 active entries via list_all_ids + get_fragment
        fake_ids = [f"memory_bank_{i:04d}" for i in range(500)]
        mock_collection.list_all_ids.return_value = fake_ids
        mock_collection.get_fragment.return_value = {
            "content": "existing",
            "metadata": {"status": "active"}
        }

        with pytest.raises(ValueError, match="capacity"):
            await service.store("Test", ["test"])


class TestUpdate:
    """Test memory update with archiving."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "old content",
            "metadata": {
                "text": "old content",
                "tags": '["identity"]',
                "importance": 0.7,
                "confidence": 0.7
            }
        })
        coll.upsert_vectors = AsyncMock()
        coll.collection = MagicMock()
        coll.collection.count = MagicMock(return_value=10)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_update_archives_old(self, service, mock_collection):
        """Should archive old version when updating."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content",
            reason="correction"
        )

        # Should have 2 upsert calls: archive + update
        assert mock_collection.upsert_vectors.call_count == 2

        # Check archive call
        archive_call = mock_collection.upsert_vectors.call_args_list[0]
        archive_id = archive_call[1]["ids"][0]
        archive_metadata = archive_call[1]["metadatas"][0]

        assert "archived" in archive_id
        assert archive_metadata["status"] == "archived"
        assert archive_metadata["archive_reason"] == "correction"

    @pytest.mark.asyncio
    async def test_update_preserves_metadata(self, service, mock_collection):
        """Should preserve original metadata fields."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content"
        )

        update_call = mock_collection.upsert_vectors.call_args_list[1]
        metadata = update_call[1]["metadatas"][0]

        assert metadata["importance"] == 0.7
        assert metadata["text"] == "new content"

    @pytest.mark.asyncio
    async def test_update_not_found_creates_new(self, service, mock_collection):
        """Should create new memory if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        doc_id = await service.update(
            doc_id="memory_bank_nonexistent",
            new_text="new memory"
        )

        assert doc_id.startswith("memory_bank_")


class TestArchive:
    """Test memory archiving."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "active"}
        })
        coll.update_fragment_metadata = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_archive_success(self, service, mock_collection):
        """Should archive memory successfully."""
        result = await service.archive(
            doc_id="memory_bank_test123",
            reason="outdated"
        )

        assert result is True
        mock_collection.update_fragment_metadata.assert_called_once()

        call_args = mock_collection.update_fragment_metadata.call_args
        metadata = call_args[0][1]
        assert metadata["status"] == "archived"
        assert metadata["archive_reason"] == "outdated"

    @pytest.mark.asyncio
    async def test_archive_not_found(self, service, mock_collection):
        """Should return False if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        result = await service.archive("nonexistent")
        assert result is False


class TestSearch:
    """Test memory search."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=[
            "memory_bank_1", "memory_bank_2", "memory_bank_3"
        ])

        def get_fragment_side_effect(doc_id):
            if doc_id == "memory_bank_1":
                return {
                    "content": "User name is John",
                    "metadata": {"status": "active", "tags": '["identity"]'}
                }
            elif doc_id == "memory_bank_2":
                return {
                    "content": "Prefers dark mode",
                    "metadata": {"status": "active", "tags": '["preference"]'}
                }
            elif doc_id == "memory_bank_3":
                return {
                    "content": "Old info",
                    "metadata": {"status": "archived", "tags": '["identity"]'}
                }
            return None

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_search_excludes_archived_by_default(self, service):
        """Should exclude archived memories by default."""
        results = await service.search()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_includes_archived_when_requested(self, service):
        """Should include archived when requested."""
        results = await service.search(include_archived=True)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_filters_by_tags(self, service):
        """Should filter by tags."""
        results = await service.search(tags=["identity"])
        assert len(results) == 1
        assert results[0]["metadata"]["tags"] == '["identity"]'


class TestRestore:
    """Test memory restoration."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "archived"}
        })
        coll.update_fragment_metadata = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_restore_success(self, service, mock_collection):
        """Should restore archived memory."""
        result = await service.restore("memory_bank_test123")

        assert result is True
        call_args = mock_collection.update_fragment_metadata.call_args
        metadata = call_args[0][1]
        assert metadata["status"] == "active"
        assert metadata["restored_by"] == "user"

    @pytest.mark.asyncio
    async def test_restore_not_found(self, service, mock_collection):
        """Should return False if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        result = await service.restore("nonexistent")
        assert result is False


class TestDelete:
    """v0.3.3 Section 8F: delete_permanent() requires force=True."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.delete_vectors = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_delete_permanent_with_force(self, service, mock_collection):
        """Should delete memory when force=True."""
        result = await service.delete_permanent("memory_bank_test123", force=True)

        assert result is True
        mock_collection.delete_vectors.assert_called_with(["memory_bank_test123"])

    @pytest.mark.asyncio
    async def test_delete_permanent_failure(self, service, mock_collection):
        """Should return False on error (with force=True)."""
        mock_collection.delete_vectors = MagicMock(side_effect=Exception("Test error"))

        result = await service.delete_permanent("memory_bank_test123", force=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_permanent_without_force_raises(self, service, mock_collection):
        """Calling without force=True must raise — guards future code from accidentally
        wiring user-facing paths to hard delete."""
        with pytest.raises(RuntimeError, match="force=True"):
            await service.delete_permanent("memory_bank_test123")
        mock_collection.delete_vectors.assert_not_called()


class TestSweepPhantoms:
    """v0.3.3 Section 8A: _sweep_phantoms removes HNSW-orphaned IDs."""

    def _service(self, ids, fragments):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=ids)
        coll.get_fragment = MagicMock(side_effect=lambda doc_id: fragments.get(doc_id))
        coll.delete_vectors = MagicMock()
        return MemoryBankService(collection=coll, embed_fn=AsyncMock()), coll

    def test_sweeps_phantom_ids(self):
        """IDs in list_all_ids() but with no fragment are phantoms."""
        service, coll = self._service(
            ids=["live1", "phantom", "live2"],
            fragments={"live1": {"content": "x"}, "live2": {"content": "y"}},
        )
        count = service._sweep_phantoms()
        assert count == 1
        coll.delete_vectors.assert_called_once_with(["phantom"])

    def test_no_phantoms_no_delete(self):
        """Clean collection: zero phantoms, no delete call."""
        service, coll = self._service(
            ids=["live1", "live2"],
            fragments={"live1": {"content": "x"}, "live2": {"content": "y"}},
        )
        count = service._sweep_phantoms()
        assert count == 0
        coll.delete_vectors.assert_not_called()

    def test_swallows_errors(self):
        """list_all_ids() raising must not propagate."""
        coll = MagicMock()
        coll.list_all_ids = MagicMock(side_effect=Exception("boom"))
        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        assert service._sweep_phantoms() == 0


class TestCleanupArchived:
    """v0.3.3 Section 8A: cleanup_archived hard-deletes archived entries,
    then sweeps the phantoms it just created."""

    def _service(self, ids, fragments):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=ids)
        coll.get_fragment = MagicMock(side_effect=lambda doc_id: fragments.get(doc_id))
        coll.delete_vectors = MagicMock()
        return MemoryBankService(collection=coll, embed_fn=AsyncMock()), coll

    def test_deletes_archived_then_sweeps(self):
        """Archived entries get hard-deleted, then _sweep_phantoms runs."""
        # After delete_vectors, simulate the IDs becoming phantoms by returning
        # None on the next get_fragment call for them.
        fragments = {
            "a1": {"metadata": {"status": "active"}},
            "a2": {"metadata": {"status": "archived"}},
            "a3": {"metadata": {"status": "archived"}},
        }
        service, coll = self._service(
            ids=["a1", "a2", "a3"], fragments=fragments
        )

        # First delete_vectors call deletes the archived entries.
        # Subsequent _sweep_phantoms() call invokes list_all_ids again — keep
        # returning the same ids; get_fragment will report the deleted ones
        # as phantoms.
        deleted_ids = []

        def delete_side_effect(ids_arg):
            deleted_ids.extend(ids_arg)
            for d in ids_arg:
                fragments[d] = None
        coll.delete_vectors.side_effect = delete_side_effect

        count = service.cleanup_archived()
        assert count == 2  # a2, a3 archived
        # delete_vectors called once for archives, then again from sweep
        assert coll.delete_vectors.call_count == 2
        first_call_args = coll.delete_vectors.call_args_list[0].args[0]
        assert sorted(first_call_args) == ["a2", "a3"]
        second_call_args = coll.delete_vectors.call_args_list[1].args[0]
        assert sorted(second_call_args) == ["a2", "a3"]  # the resulting phantoms

    def test_nothing_archived_skips_sweep(self):
        """If nothing is archived, no delete_vectors or sweep happens."""
        fragments = {
            "a1": {"metadata": {"status": "active"}},
            "a2": {"metadata": {"status": "active"}},
        }
        service, coll = self._service(ids=["a1", "a2"], fragments=fragments)

        count = service.cleanup_archived()
        assert count == 0
        coll.delete_vectors.assert_not_called()

    def test_list_all_ids_error_returns_zero(self):
        """list_all_ids() raising returns 0, no delete attempts."""
        coll = MagicMock()
        coll.list_all_ids = MagicMock(side_effect=Exception("boom"))
        coll.delete_vectors = MagicMock()
        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        assert service.cleanup_archived() == 0
        coll.delete_vectors.assert_not_called()


class TestMaybeCleanupArchived:
    """v0.3.3 Section 8D: capacity-pressure auto-cleanup gates."""

    def _service_with_state(self, active_count, total_count):
        """Build a service where _get_count returns `active_count` and
        list_all_ids returns IDs totaling `total_count`."""
        coll = MagicMock()
        # Synthesize total ids list; first `active_count` get "active" status,
        # the rest get "archived". _get_count walks list_all_ids() + get_fragment
        # and counts non-archived.
        ids = [f"id_{i}" for i in range(total_count)]
        fragments = {}
        for i, doc_id in enumerate(ids):
            status = "active" if i < active_count else "archived"
            fragments[doc_id] = {"metadata": {"status": status}}
        coll.list_all_ids = MagicMock(return_value=ids)
        coll.get_fragment = MagicMock(side_effect=lambda doc_id: fragments.get(doc_id))
        coll.delete_vectors = MagicMock()
        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        return service, coll

    def test_below_active_threshold_no_cleanup(self):
        """Active count under ACTIVE_THRESHOLD: never trigger cleanup."""
        service, coll = self._service_with_state(active_count=399, total_count=399)
        service._maybe_cleanup_archived()
        coll.delete_vectors.assert_not_called()

    def test_above_active_threshold_low_archived_ratio_no_cleanup(self):
        """Active >= 400 but archived ratio < 50%: still no cleanup."""
        # active=450, total=500 → archived=50, ratio=0.1 → skip
        service, coll = self._service_with_state(active_count=450, total_count=500)
        service._maybe_cleanup_archived()
        coll.delete_vectors.assert_not_called()

    def test_above_active_threshold_high_archived_ratio_triggers_cleanup(self):
        """Active >= 400 AND archived ratio >= 50%: cleanup fires."""
        # active=400, total=900 → archived=500, ratio≈0.56 → trigger
        service, coll = self._service_with_state(active_count=400, total_count=900)
        service._maybe_cleanup_archived()
        # cleanup_archived → delete_vectors for archived ids + sweep call
        assert coll.delete_vectors.called

    def test_errors_swallowed(self):
        """list_all_ids() raising must not propagate from auto-cleanup."""
        coll = MagicMock()
        coll.list_all_ids = MagicMock(side_effect=Exception("boom"))
        # Force the active count check to pass so we hit list_all_ids
        coll.get_fragment = MagicMock(return_value=None)
        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        # Should not raise
        service._maybe_cleanup_archived()


class TestListAll:
    """Test listing all memories."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=["m1", "m2", "m3"])

        def get_fragment_side_effect(doc_id):
            return {
                "content": f"content_{doc_id}",
                "metadata": {
                    "status": "active" if doc_id != "m3" else "archived",
                    "tags": '["identity"]' if doc_id == "m1" else '["preference"]'
                }
            }

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_list_all_excludes_archived(self, service):
        """Should exclude archived by default."""
        results = service.list_all()
        assert len(results) == 2

    def test_list_all_includes_archived(self, service):
        """Should include archived when requested."""
        results = service.list_all(include_archived=True)
        assert len(results) == 3

    def test_list_all_filters_tags(self, service):
        """Should filter by tags."""
        results = service.list_all(tags=["identity"])
        assert len(results) == 1


class TestStats:
    """Test statistics retrieval."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=["m1", "m2", "m3"])

        def get_fragment_side_effect(doc_id):
            if doc_id == "m1":
                return {
                    "content": "identity",
                    "metadata": {
                        "status": "active",
                        "tags": '["identity"]',
                        "importance": 0.9,
                        "confidence": 0.8
                    }
                }
            elif doc_id == "m2":
                return {
                    "content": "preference",
                    "metadata": {
                        "status": "active",
                        "tags": '["preference", "identity"]',
                        "importance": 0.7,
                        "confidence": 0.7
                    }
                }
            elif doc_id == "m3":
                return {
                    "content": "archived",
                    "metadata": {
                        "status": "archived",
                        "tags": '["old"]',
                        "importance": 0.5,
                        "confidence": 0.5
                    }
                }
            return None

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_get_stats(self, service):
        """Should return correct statistics."""
        stats = service.get_stats()

        assert stats["total"] == 3
        assert stats["active"] == 2
        assert stats["archived"] == 1
        assert stats["capacity"] == 500
        assert stats["tag_counts"]["identity"] == 2
        assert stats["tag_counts"]["preference"] == 1
        assert stats["avg_importance"] == 0.8  # (0.9 + 0.7) / 2
        assert stats["avg_confidence"] == 0.75  # (0.8 + 0.7) / 2


class TestIncrementMention:
    """Test mention count tracking."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"mentioned_count": 5}
        })
        coll.update_fragment_metadata = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_increment_mention(self, service, mock_collection):
        """Should increment mention count."""
        result = service.increment_mention("memory_bank_test123")

        assert result is True
        call_args = mock_collection.update_fragment_metadata.call_args
        metadata = call_args[0][1]
        assert metadata["mentioned_count"] == 6
        assert "last_mentioned" in metadata

    def test_increment_not_found(self, service, mock_collection):
        """Should return False if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        result = service.increment_mention("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
