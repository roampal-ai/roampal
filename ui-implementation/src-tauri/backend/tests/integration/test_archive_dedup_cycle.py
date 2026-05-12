"""
v0.3.3 Section 8G — Integration test for the full archive-then-add cycle.

Reproduces issue #8 end-to-end against real ChromaDB:
  add facts → archive some → add new facts that semantically overlap
  → confirm all new facts stored (no silent dedup against archived entries).

Mocks should NOT be used here: Section 8G's whole point is that unit-level
mocks don't model HNSW phantom behavior, which is the bug class we're
guarding against. The embedding service is the one exception — we inject
a deterministic stand-in so tests don't pull in ONNX models.
"""

import asyncio
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure backend/ is on the path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


class _DeterministicEmbedding:
    """Deterministic 768-d embedding for tests.

    Same text → same vector. Different text → near-orthogonal vector.
    Skips ONNX entirely so tests are fast and offline-safe.
    """

    def __init__(self, dim: int = 768):
        self._dim = dim

    async def embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand the 32-byte digest into a dim-length float vector in [-1, 1].
        vec = []
        i = 0
        while len(vec) < self._dim:
            byte = digest[i % len(digest)]
            vec.append((byte / 127.5) - 1.0)
            i += 1
        return vec

    def _load_model(self):
        """No-op: nothing to warm up."""
        return None


class TestArchiveDedupCycle:
    """Issue #8 regression: archived entries must not block dedup of new facts."""

    @pytest.fixture
    def temp_data_dir(self):
        d = tempfile.mkdtemp(prefix="roampal_archive_cycle_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    async def _build_mem(self, temp_data_dir):
        from modules.memory.unified_memory_system import UnifiedMemorySystem

        system = UnifiedMemorySystem(
            data_dir=temp_data_dir,
            use_server=False,
            embedding_service=_DeterministicEmbedding(),
        )
        await system.initialize()
        return system

    @pytest.mark.asyncio
    async def test_archive_then_add_identical_fact_succeeds(self, temp_data_dir):
        mem = await self._build_mem(temp_data_dir)
        """Marcus #8 repro: archive a fact, re-add identical content, new ID must be stored."""
        original_id = await mem.store_memory_bank(
            "User prefers blue over red", tags=["preference"]
        )
        assert original_id is not None

        # Find the document by listing all to grab the doc_id
        all_items = mem._memory_bank_service.list_all()
        assert len(all_items) == 1
        doc_id = all_items[0]["id"]

        # User-initiated delete (routes through archive per v0.3.3 Section 7)
        archived = await mem.user_delete_memory(doc_id)
        assert archived is True

        # Active count is now 0 (archived entries don't count)
        assert mem._memory_bank_service._get_count() == 0

        # Re-store identical content — must succeed; archived entry filtered from dedup
        new_id = await mem.store_memory_bank(
            "User prefers blue over red", tags=["preference"]
        )
        assert new_id is not None

        # Active count back to 1; the live entry is the freshly-stored one
        assert mem._memory_bank_service._get_count() == 1
        live_items = mem._memory_bank_service.list_all()
        assert len(live_items) == 1
        assert live_items[0]["metadata"]["text"] == "User prefers blue over red"

    @pytest.mark.asyncio
    async def test_active_duplicate_still_dedups(self, temp_data_dir):
        mem = await self._build_mem(temp_data_dir)
        """Sanity check: active entries still dedup normally (regression guard)."""
        id1 = await mem.store_memory_bank(
            "User lives in San Francisco", tags=["location"]
        )
        id2 = await mem.store_memory_bank(
            "User lives in San Francisco", tags=["location"]
        )
        # Both calls resolve to the same stored doc (one row in memory_bank)
        assert mem._memory_bank_service._get_count() == 1
        # Same content → same embedding → second store returns the existing id
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_archive_all_then_readd_all(self, temp_data_dir):
        mem = await self._build_mem(temp_data_dir)
        """Archive multiple entries, re-add same content for all — each must succeed."""
        await mem.store_memory_bank("User likes hiking", tags=["hobby"])
        await mem.store_memory_bank("User likes biking", tags=["hobby"])
        await mem.store_memory_bank("User likes climbing", tags=["hobby"])
        assert mem._memory_bank_service._get_count() == 3

        # Archive all three
        for item in mem._memory_bank_service.list_all():
            ok = await mem.user_delete_memory(item["id"])
            assert ok is True
        assert mem._memory_bank_service._get_count() == 0

        # Re-add same content — all three must store fresh
        await mem.store_memory_bank("User likes hiking", tags=["hobby"])
        await mem.store_memory_bank("User likes biking", tags=["hobby"])
        await mem.store_memory_bank("User likes climbing", tags=["hobby"])
        assert mem._memory_bank_service._get_count() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
