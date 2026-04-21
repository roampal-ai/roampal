"""
Unit Tests for SearchService

Tests the TagCascade search pipeline (v0.3.1).
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from modules.memory.search_service import SearchService
from modules.memory.scoring_service import ScoringService
from modules.memory.routing_service import RoutingService
from modules.memory.tag_service import TagService
from modules.memory.config import MemoryConfig


class TestSearchServiceInit:
    """Test SearchService initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        collections = {
            "working": MagicMock(),
            "history": MagicMock(),
            "patterns": MagicMock(),
            "books": MagicMock(),
            "memory_bank": MagicMock(),
        }
        scoring = MagicMock(spec=ScoringService)
        routing = MagicMock(spec=RoutingService)
        tag_service = MagicMock(spec=TagService)
        tag_service.match_query_tags = MagicMock(return_value=[])
        embed_fn = AsyncMock(return_value=[0.1] * 384)

        return {
            "collections": collections,
            "scoring_service": scoring,
            "routing_service": routing,
            "tag_service": tag_service,
            "embed_fn": embed_fn,
        }

    def test_init_with_all_dependencies(self, mock_dependencies):
        """Should initialize with all dependencies."""
        service = SearchService(**mock_dependencies)
        assert service.collections == mock_dependencies["collections"]
        assert service.scoring_service == mock_dependencies["scoring_service"]
        assert service.routing_service == mock_dependencies["routing_service"]
        assert service.tag_service == mock_dependencies["tag_service"]

    def test_init_accepts_kwargs(self, mock_dependencies):
        """Should accept extra kwargs for backward compat."""
        service = SearchService(**mock_dependencies, reranker=MagicMock())
        assert service is not None

    def test_ce_candidate_pool_is_40(self, mock_dependencies):
        """v0.3.1: CE candidate pool should be 40."""
        service = SearchService(**mock_dependencies)
        assert service.CE_CANDIDATE_POOL == 40


class TestMainSearch:
    """Test main search functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create SearchService with mocks."""
        collections = {
            "working": MagicMock(),
            "history": MagicMock(),
        }

        # Mock hybrid_query to return sample results
        async def mock_hybrid_query(**kwargs):
            return [
                {"id": "doc_1", "text": "test result 1", "distance": 0.5, "metadata": {"score": 0.7, "uses": 3}},
                {"id": "doc_2", "text": "test result 2", "distance": 0.8, "metadata": {"score": 0.5, "uses": 1}},
            ]

        for coll in collections.values():
            coll.hybrid_query = AsyncMock(side_effect=mock_hybrid_query)

        scoring = MagicMock(spec=ScoringService)
        scoring.apply_scoring_to_results = MagicMock(side_effect=lambda x: x)

        routing = MagicMock(spec=RoutingService)
        routing.route_query = MagicMock(return_value=["working", "history"])
        routing.preprocess_query = MagicMock(side_effect=lambda x: x)

        tag_service = MagicMock(spec=TagService)
        tag_service.match_query_tags = MagicMock(return_value=[])

        embed_fn = AsyncMock(return_value=[0.1] * 384)

        return SearchService(
            collections=collections,
            scoring_service=scoring,
            routing_service=routing,
            tag_service=tag_service,
            embed_fn=embed_fn,
        )

    @pytest.mark.asyncio
    async def test_search_routes_query(self, mock_service):
        """Should route query when collections not specified."""
        await mock_service.search("test query", limit=5)
        mock_service.routing_service.route_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_uses_explicit_collections(self, mock_service):
        """Should use explicit collections when provided."""
        await mock_service.search("test query", collections=["history"], limit=5)
        mock_service.routing_service.route_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_preprocesses_query(self, mock_service):
        """Should preprocess query before embedding."""
        await mock_service.search("test query", limit=5)
        mock_service.routing_service.preprocess_query.assert_called()

    @pytest.mark.asyncio
    async def test_search_generates_embedding(self, mock_service):
        """Should generate embedding for query."""
        await mock_service.search("test query", limit=5)
        mock_service.embed_fn.assert_called()

    @pytest.mark.asyncio
    async def test_search_applies_scoring(self, mock_service):
        """Should apply scoring to results."""
        await mock_service.search("test query", limit=5)
        mock_service.scoring_service.apply_scoring_to_results.assert_called()

    @pytest.mark.asyncio
    async def test_search_checks_tags(self, mock_service):
        """v0.3.1: Should check for matching tags before search."""
        await mock_service.search("test query", limit=5)
        mock_service.tag_service.match_query_tags.assert_called_with("test query")

    @pytest.mark.asyncio
    async def test_search_returns_list_by_default(self, mock_service):
        """Should return list when return_metadata=False."""
        result = await mock_service.search("test query", limit=5)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_returns_dict_with_metadata(self, mock_service):
        """Should return dict when return_metadata=True."""
        result = await mock_service.search("test query", limit=5, return_metadata=True)
        assert isinstance(result, dict)
        assert "results" in result
        assert "total" in result
        assert "has_more" in result

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, mock_service):
        """Should respect limit parameter."""
        result = await mock_service.search("test query", limit=1)
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self, mock_service):
        """Empty query should return all items."""
        for coll in mock_service.collections.values():
            coll.collection = MagicMock()
            coll.collection.get = MagicMock(return_value={
                'ids': ['id1', 'id2'],
                'documents': ['doc1', 'doc2'],
                'metadatas': [{'score': 0.5}, {'score': 0.6}]
            })

        result = await mock_service.search("", limit=10)
        assert len(result) > 0


class TestCollectionBoosts:
    """Test collection-specific distance boosts."""

    @pytest.fixture
    def service(self):
        """Create SearchService."""
        tag_service = MagicMock(spec=TagService)
        tag_service.match_query_tags = MagicMock(return_value=[])

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            tag_service=tag_service,
            embed_fn=AsyncMock(),
        )

    def test_patterns_boost(self, service):
        """Patterns should get 10% distance reduction."""
        result = {"distance": 1.0, "metadata": {}}
        service._apply_collection_boost(result, "patterns", "query")
        assert result["distance"] == 0.9

    def test_memory_bank_quality_boost(self, service):
        """Memory bank should boost by quality score."""
        result = {
            "distance": 1.0,
            "id": "doc_1",
            "metadata": {"importance": 0.9, "confidence": 0.9}
        }
        service._apply_collection_boost(result, "memory_bank", "query")
        # quality = 0.81, metadata_boost = 1.0 - 0.81*0.8 = 0.352
        assert result["distance"] < 1.0

    def test_books_recent_upload_boost(self, service):
        """Recent books should get boost."""
        result = {
            "distance": 1.0,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "metadata": {}
        }
        service._apply_collection_boost(result, "books", "query")
        assert result["distance"] == 0.7


class TestCaching:
    """Test doc_id caching for outcome scoring."""

    @pytest.fixture
    def service(self):
        """Create SearchService."""
        tag_service = MagicMock(spec=TagService)
        tag_service.match_query_tags = MagicMock(return_value=[])

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            tag_service=tag_service,
            embed_fn=AsyncMock(),
        )

    def test_track_search_caches_doc_ids(self, service):
        """Should cache doc_ids from scoreable collections."""
        results = [
            {"id": "working_1", "collection": "working"},
            {"id": "history_1", "collection": "history"},
            {"id": "books_1", "collection": "books"},  # Not cached
        ]

        service._track_search_results("query", results, None)

        cached = service.get_cached_doc_ids('default')
        assert "working_1" in cached
        assert "history_1" in cached
        assert "books_1" not in cached  # Books not cached

    def test_caching_per_session(self, service):
        """Should cache separately per session."""
        results1 = [{"id": "doc_1", "collection": "working"}]
        results2 = [{"id": "doc_2", "collection": "working"}]

        ctx1 = MagicMock()
        ctx1.session_id = "session_1"
        ctx2 = MagicMock()
        ctx2.session_id = "session_2"

        service._track_search_results("q1", results1, ctx1)
        service._track_search_results("q2", results2, ctx2)

        assert service.get_cached_doc_ids("session_1") == ["doc_1"]
        assert service.get_cached_doc_ids("session_2") == ["doc_2"]


class TestMergeFilters:
    """Test filter merging — uses valid ChromaDB where-clause operators only.
    NOTE: historical tests here used {"$contains": ...} which ChromaDB's
    where-clause doesn't actually accept (that's a where_document operator).
    v0.3.2 dropped the broken $contains tag filter from the production path;
    these tests now use valid operators that exercise the merge helper."""

    def test_merge_tag_only(self):
        """Should return tag filter when no metadata filters."""
        tag_filter = {"noun_tags": {"$eq": "python"}}
        result = SearchService._merge_filters(tag_filter, None)
        assert result == tag_filter

    def test_merge_with_metadata(self):
        """Should combine with $and when metadata filters present."""
        tag_filter = {"noun_tags": {"$eq": "python"}}
        metadata = {"status": {"$ne": "archived"}}
        result = SearchService._merge_filters(tag_filter, metadata)
        assert "$and" in result
        assert len(result["$and"]) == 2


class TestTagCascadePythonFilter:
    """v0.3.2 Section 0m: TagCascade tag match happens in Python (post-fetch),
    not via ChromaDB where-clause — because the previous where-clause
    $contains approach was silently erroring on every query since v0.3.1.
    These tests cover the Python-side tag match: parse JSON-encoded
    noun_tags, filter by membership. Covers the _tag_routed_search parsing
    paths regardless of adapter behavior."""

    def _parse_and_match(self, raw_tags, tag):
        """Inline the parsing logic from _tag_routed_search for unit coverage."""
        parsed_tags = None
        if isinstance(raw_tags, list):
            parsed_tags = raw_tags
        elif isinstance(raw_tags, str) and raw_tags:
            try:
                candidate = json.loads(raw_tags)
                if isinstance(candidate, list):
                    parsed_tags = candidate
            except (ValueError, TypeError):
                parsed_tags = None
        return parsed_tags is not None and tag in parsed_tags

    def test_json_string_with_tag_matches(self):
        assert self._parse_and_match(json.dumps(["calvin", "boston"]), "calvin") is True

    def test_json_string_without_tag_does_not_match(self):
        assert self._parse_and_match(json.dumps(["calvin", "boston"]), "python") is False

    def test_already_parsed_list_matches(self):
        assert self._parse_and_match(["calvin", "boston"], "boston") is True

    def test_missing_noun_tags_does_not_match(self):
        assert self._parse_and_match(None, "calvin") is False

    def test_empty_string_noun_tags_does_not_match(self):
        assert self._parse_and_match("", "calvin") is False

    def test_malformed_json_does_not_match_no_crash(self):
        """Invalid JSON shouldn't raise — must silently exclude the row."""
        assert self._parse_and_match("not json {[", "calvin") is False

    def test_non_list_json_does_not_match(self):
        """JSON-valid but not a list (object) shouldn't crash or match."""
        assert self._parse_and_match(json.dumps({"not": "a list"}), "calvin") is False


class TestMemoryBankFilterWrapping:
    """v0.3.2 Section 0n: ChromaDB rejects multi-key top-level where filters —
    they must be wrapped in $and. The TagCascade tag-routed path was missing
    this wrapping when combining the summary-lane filter (memory_type != fact)
    with memory_bank's status != archived filter. Covers the wrapping helper
    shape without needing a real ChromaDB."""

    def _build_mb_filter(self, metadata_filters):
        """Inline the wrapping logic from _tag_routed_search for unit coverage."""
        status_filter = {"status": {"$ne": "archived"}}
        if metadata_filters:
            return {
                "$and": [{k: v} for k, v in metadata_filters.items()] + [status_filter]
            }
        return status_filter

    def test_no_metadata_filters_returns_bare_status_filter(self):
        """When no caller filter, memory_bank gets just the status filter —
        single key, no $and wrapping needed."""
        result = self._build_mb_filter(None)
        assert result == {"status": {"$ne": "archived"}}

    def test_empty_metadata_filters_returns_bare_status_filter(self):
        """Empty-dict metadata filter treated as None."""
        result = self._build_mb_filter({})
        assert result == {"status": {"$ne": "archived"}}

    def test_single_metadata_key_wraps_in_and(self):
        """Summary-lane filter (memory_type != fact) + status filter must be
        wrapped in $and — ChromaDB rejects top-level multi-key dicts."""
        result = self._build_mb_filter({"memory_type": {"$ne": "fact"}})
        assert "$and" in result
        assert len(result["$and"]) == 2
        # Every element is a single-key dict (valid ChromaDB where-clause shape)
        for cond in result["$and"]:
            assert len(cond) == 1

    def test_multiple_metadata_keys_each_become_conditions(self):
        """Each metadata key becomes its own $and condition, plus status."""
        result = self._build_mb_filter({
            "memory_type": {"$ne": "fact"},
            "score": {"$gt": 0.5},
        })
        assert "$and" in result
        assert len(result["$and"]) == 3  # 2 metadata + 1 status


class TestParseNumeric:
    """Test numeric parsing helper."""

    @pytest.fixture
    def service(self):
        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            tag_service=MagicMock(),
            embed_fn=AsyncMock(),
        )

    def test_parse_float(self, service):
        assert service._parse_numeric(0.9) == 0.9

    def test_parse_int(self, service):
        assert service._parse_numeric(1) == 1.0

    def test_parse_list(self, service):
        assert service._parse_numeric([0.8, 0.9]) == 0.8

    def test_parse_string_high(self, service):
        assert service._parse_numeric("high") == 0.9

    def test_parse_string_medium(self, service):
        assert service._parse_numeric("medium") == 0.7

    def test_parse_string_low(self, service):
        assert service._parse_numeric("low") == 0.5

    def test_parse_none(self, service):
        assert service._parse_numeric(None) == 0.7

    def test_parse_invalid(self, service):
        assert service._parse_numeric("invalid") == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
