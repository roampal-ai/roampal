"""
Tests for memory_visualization_enhanced.py — the /api/memory/collections/{type}
endpoint the Memory Panel UI hits.

Section 0l regression coverage: `noun_tags` is stored as a JSON-encoded
string inside `metadata` by the sidecar write path. The endpoint must
parse it and flatten to a top-level `tags` array so the frontend
(MemoryPanelV2.tsx) can render tag pills without doing its own parsing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_request(items):
    """Build a MagicMock Request whose memory_collections.search returns `items`."""
    request = MagicMock()
    memory_collections = MagicMock()
    memory_collections.search = AsyncMock(
        return_value={"results": items, "total": len(items)}
    )
    request.app.state.memory_collections = memory_collections
    return request


class TestCollectionMemoriesFlattensNounTags:
    """v0.3.2 Section 0l: noun_tags JSON string must surface as memory.tags list."""

    @pytest.mark.asyncio
    async def test_collection_memories_flattens_noun_tags_json_string(self):
        """Sidecar-written noun_tags (JSON-encoded string) surfaces as a list."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_abc",
                "content": "sample",
                "metadata": {
                    "noun_tags": json.dumps(["calvin", "muscle car", "boston"]),
                    "timestamp": "2026-04-21T10:00:00",
                    "score": 0.7,
                    "uses": 2,
                },
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert len(result["memories"]) == 1
        memory = result["memories"][0]
        assert memory["tags"] == ["calvin", "muscle car", "boston"]

    @pytest.mark.asyncio
    async def test_collection_memories_handles_list_noun_tags(self):
        """Forgiving path: if noun_tags already came through as a list, pass through."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_xyz",
                "content": "sample",
                "metadata": {
                    "noun_tags": ["already", "a", "list"],
                    "timestamp": "2026-04-21T10:00:00",
                    "score": 0.5,
                },
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert result["memories"][0]["tags"] == ["already", "a", "list"]

    @pytest.mark.asyncio
    async def test_collection_memories_handles_missing_noun_tags(self):
        """Missing noun_tags key → memory.tags == []."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_no_tags",
                "content": "sample",
                "metadata": {
                    "timestamp": "2026-04-21T10:00:00",
                    "score": 0.5,
                },
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert result["memories"][0]["tags"] == []

    @pytest.mark.asyncio
    async def test_collection_memories_handles_empty_string_noun_tags(self):
        """Empty-string noun_tags → memory.tags == [], no 500."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_empty",
                "content": "sample",
                "metadata": {"noun_tags": "", "timestamp": "2026-04-21T10:00:00"},
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert result["memories"][0]["tags"] == []

    @pytest.mark.asyncio
    async def test_collection_memories_handles_malformed_json_noun_tags(self):
        """Invalid JSON string → memory.tags == [], no 500."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_malformed",
                "content": "sample",
                "metadata": {
                    "noun_tags": "not json at all {[}",
                    "timestamp": "2026-04-21T10:00:00",
                },
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert result["memories"][0]["tags"] == []

    @pytest.mark.asyncio
    async def test_collection_memories_handles_non_list_json_noun_tags(self):
        """JSON-valid but not a list (e.g. object) → memory.tags == []."""
        from app.routers.memory_visualization_enhanced import get_collection_memories

        items = [
            {
                "id": "history_object",
                "content": "sample",
                "metadata": {
                    "noun_tags": json.dumps({"not": "a list"}),
                    "timestamp": "2026-04-21T10:00:00",
                },
            }
        ]
        request = _make_request(items)
        result = await get_collection_memories(
            request=request, collection_type="history", limit=50, offset=0
        )

        assert result["memories"][0]["tags"] == []
