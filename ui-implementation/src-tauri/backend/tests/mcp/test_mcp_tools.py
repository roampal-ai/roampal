"""
Unit Tests for MCP Tool Handlers

Tests the MCP tool layer that wraps the underlying memory services.
These tests validate:
1. Tool schema compliance
2. Argument parsing and validation
3. Service delegation
4. Response formatting
5. Cache management (doc_id caching for outcome scoring)
6. Action-Effectiveness KG tracking

Ported from backend service tests to test the MCP interface layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


class TestSearchMemoryTool:
    """Tests for search_memory MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory system."""
        memory = MagicMock()
        memory.initialized = True

        async def mock_search(query, collections=None, limit=5, metadata_filters=None):
            return [
                {
                    "id": "doc_1",
                    "doc_id": "doc_1",
                    "content": f"Result for: {query}",
                    "text": f"Result for: {query}",
                    "collection": "working",
                    "metadata": {"score": 0.8, "uses": 3}
                }
            ]
        memory.search = AsyncMock(side_effect=mock_search)
        return memory

    @pytest.mark.asyncio
    async def test_search_basic_query(self, mock_memory):
        """Should delegate to memory.search with correct args."""
        from tests.mcp.mcp_tool_harness import call_search_memory

        result = await call_search_memory(
            memory=mock_memory,
            arguments={"query": "python programming"},
            session_id="test_session"
        )

        mock_memory.search.assert_called_once()
        call_args = mock_memory.search.call_args
        assert call_args.kwargs["query"] == "python programming"

    @pytest.mark.asyncio
    async def test_search_with_collections(self, mock_memory):
        """Should pass explicit collections."""
        from tests.mcp.mcp_tool_harness import call_search_memory

        result = await call_search_memory(
            memory=mock_memory,
            arguments={"query": "test", "collections": ["books", "history"]},
            session_id="test_session"
        )

        call_args = mock_memory.search.call_args
        assert call_args.kwargs["collections"] == ["books", "history"]

    @pytest.mark.asyncio
    async def test_search_with_limit(self, mock_memory):
        """Should respect limit parameter."""
        from tests.mcp.mcp_tool_harness import call_search_memory

        result = await call_search_memory(
            memory=mock_memory,
            arguments={"query": "test", "limit": 10},
            session_id="test_session"
        )

        call_args = mock_memory.search.call_args
        assert call_args.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_caches_doc_ids(self, mock_memory):
        """Should cache doc_ids for outcome scoring."""
        from tests.mcp.mcp_tool_harness import call_search_memory, get_search_cache

        await call_search_memory(
            memory=mock_memory,
            arguments={"query": "test query"},
            session_id="test_session"
        )

        cache = get_search_cache("test_session")
        assert cache is not None
        assert "doc_1" in cache["doc_ids"]
        assert cache["query"] == "test query"

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, mock_memory):
        """Should format results with metadata."""
        from tests.mcp.mcp_tool_harness import call_search_memory

        result = await call_search_memory(
            memory=mock_memory,
            arguments={"query": "test"},
            session_id="test_session"
        )

        assert "Found 1 result" in result
        assert "[working]" in result  # Collection tag
        assert "score:" in result  # Metadata

    @pytest.mark.asyncio
    async def test_search_uninitialized_memory(self, mock_memory):
        """Should handle uninitialized memory gracefully."""
        from tests.mcp.mcp_tool_harness import call_search_memory

        mock_memory.initialized = False

        result = await call_search_memory(
            memory=mock_memory,
            arguments={"query": "test"},
            session_id="test_session"
        )

        assert "No results" in result or "empty" in result.lower()


class TestAddToMemoryBankTool:
    """Tests for add_to_memory_bank MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.store_memory_bank = AsyncMock(return_value="memory_bank_123")
        return memory

    @pytest.mark.asyncio
    async def test_add_basic(self, mock_memory):
        """Should store content with defaults."""
        from tests.mcp.mcp_tool_harness import call_add_to_memory_bank

        result = await call_add_to_memory_bank(
            memory=mock_memory,
            arguments={"content": "User prefers dark mode"},
            session_id="test_session"
        )

        mock_memory.store_memory_bank.assert_called_once()
        call_args = mock_memory.store_memory_bank.call_args
        assert call_args.kwargs["text"] == "User prefers dark mode"
        assert "memory_bank_123" in result

    @pytest.mark.asyncio
    async def test_add_with_tags(self, mock_memory):
        """Should pass tags correctly."""
        from tests.mcp.mcp_tool_harness import call_add_to_memory_bank

        await call_add_to_memory_bank(
            memory=mock_memory,
            arguments={"content": "Test", "tags": ["preference", "ui"]},
            session_id="test_session"
        )

        call_args = mock_memory.store_memory_bank.call_args
        assert call_args.kwargs["tags"] == ["preference", "ui"]

    @pytest.mark.asyncio
    async def test_add_with_importance_confidence(self, mock_memory):
        """Should pass importance and confidence."""
        from tests.mcp.mcp_tool_harness import call_add_to_memory_bank

        await call_add_to_memory_bank(
            memory=mock_memory,
            arguments={
                "content": "Critical info",
                "importance": 0.95,
                "confidence": 0.9
            },
            session_id="test_session"
        )

        call_args = mock_memory.store_memory_bank.call_args
        assert call_args.kwargs["importance"] == 0.95
        assert call_args.kwargs["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_add_tracks_action(self, mock_memory):
        """Should track action for Action-Effectiveness KG."""
        from tests.mcp.mcp_tool_harness import call_add_to_memory_bank, get_action_cache, clear_all_caches

        # Clear caches to avoid pollution from other tests
        clear_all_caches()

        await call_add_to_memory_bank(
            memory=mock_memory,
            arguments={"content": "Test fact"},
            session_id="test_add_action"  # Use unique session ID
        )

        actions = get_action_cache("test_add_action")
        assert len(actions) > 0
        assert actions[0]["action_type"] == "create_memory"


class TestUpdateMemoryTool:
    """Tests for update_memory MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search_memory_bank = AsyncMock(return_value=[
            {"id": "memory_bank_123", "content": "old content"}
        ])
        memory.update_memory_bank = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_update_finds_and_updates(self, mock_memory):
        """Should find old memory and update it."""
        from tests.mcp.mcp_tool_harness import call_update_memory

        result = await call_update_memory(
            memory=mock_memory,
            arguments={
                "old_content": "old content",
                "new_content": "new content"
            },
            session_id="test_session"
        )

        # Should search for old content
        mock_memory.search_memory_bank.assert_called_once()

        # Should update with new content
        mock_memory.update_memory_bank.assert_called_once()
        call_args = mock_memory.update_memory_bank.call_args
        assert call_args.kwargs["doc_id"] == "memory_bank_123"
        assert call_args.kwargs["new_text"] == "new content"

    @pytest.mark.asyncio
    async def test_update_not_found(self, mock_memory):
        """Should return error when memory not found."""
        from tests.mcp.mcp_tool_harness import call_update_memory

        mock_memory.search_memory_bank = AsyncMock(return_value=[])

        result, is_error = await call_update_memory(
            memory=mock_memory,
            arguments={"old_content": "nonexistent", "new_content": "new"},
            session_id="test_session",
            return_error_flag=True
        )

        assert is_error is True
        assert "not found" in result.lower()


class TestArchiveMemoryTool:
    """Tests for archive_memory MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search_memory_bank = AsyncMock(return_value=[
            {"id": "memory_bank_123", "content": "to archive"}
        ])
        memory.archive_memory_bank = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_archive_finds_and_archives(self, mock_memory):
        """Should find memory and archive it."""
        from tests.mcp.mcp_tool_harness import call_archive_memory

        result = await call_archive_memory(
            memory=mock_memory,
            arguments={"content": "to archive"},
            session_id="test_session"
        )

        mock_memory.search_memory_bank.assert_called_once()
        mock_memory.archive_memory_bank.assert_called_once()

        call_args = mock_memory.archive_memory_bank.call_args
        assert call_args.kwargs["doc_id"] == "memory_bank_123"
        assert "memory_bank_123" in result

    @pytest.mark.asyncio
    async def test_archive_not_found(self, mock_memory):
        """Should return error when memory not found."""
        from tests.mcp.mcp_tool_harness import call_archive_memory

        mock_memory.search_memory_bank = AsyncMock(return_value=[])

        result, is_error = await call_archive_memory(
            memory=mock_memory,
            arguments={"content": "nonexistent"},
            session_id="test_session",
            return_error_flag=True
        )

        assert is_error is True


class TestGetContextInsightsTool:
    """Tests for get_context_insights MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.detect_context_type = AsyncMock(return_value="coding")
        memory.analyze_conversation_context = AsyncMock(return_value={
            "entities": ["python", "testing"],
            "patterns": []
        })
        memory.get_action_effectiveness = MagicMock(return_value={
            "search_memory|coding|books": {"success_rate": 0.85, "uses": 10}
        })
        memory.get_tier_recommendations = MagicMock(return_value=["books", "patterns"])
        memory.get_facts_for_entities = AsyncMock(return_value=[
            {"text": "User prefers pytest", "tags": ["preference"]}
        ])
        return memory

    @pytest.mark.asyncio
    async def test_context_returns_structured_response(self, mock_memory, tmp_path):
        """Should return structured context insights."""
        from tests.mcp.mcp_tool_harness import call_get_context_insights

        result = await call_get_context_insights(
            memory=mock_memory,
            arguments={"query": "how to write tests"},
            session_id="test_session",
            data_path=tmp_path
        )

        # Should contain known facts
        assert "User prefers pytest" in result or "MEMORY_BANK" in result

    @pytest.mark.asyncio
    async def test_context_detects_context_type(self, mock_memory, tmp_path):
        """Should detect context type from conversation."""
        from tests.mcp.mcp_tool_harness import call_get_context_insights

        await call_get_context_insights(
            memory=mock_memory,
            arguments={"query": "python code"},
            session_id="test_session",
            data_path=tmp_path
        )

        mock_memory.detect_context_type.assert_called()


class TestRecordResponseTool:
    """Tests for record_response MCP tool."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.store = AsyncMock(return_value="working_123")
        memory.record_outcome = AsyncMock()
        memory.record_action_outcome = AsyncMock()
        memory._update_kg_routing = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_record_stores_takeaway(self, mock_memory):
        """Should store key takeaway in working collection."""
        from tests.mcp.mcp_tool_harness import call_record_response

        result = await call_record_response(
            memory=mock_memory,
            arguments={
                "key_takeaway": "Learned about pytest fixtures",
                "outcome": "worked"
            },
            session_id="test_session"
        )

        mock_memory.store.assert_called()
        call_args = mock_memory.store.call_args
        assert "pytest fixtures" in call_args.kwargs["text"]
        assert call_args.kwargs["collection"] == "working"

    @pytest.mark.asyncio
    async def test_record_scores_cached_docs(self, mock_memory):
        """Should score cached doc_ids with outcome."""
        from tests.mcp.mcp_tool_harness import (
            call_search_memory, call_record_response, set_search_cache
        )

        # Pre-populate cache
        set_search_cache("test_session", {
            "doc_ids": ["doc_1", "doc_2"],
            "query": "test query",
            "collections": ["working"],
            "timestamp": datetime.now()
        })

        await call_record_response(
            memory=mock_memory,
            arguments={
                "key_takeaway": "Test summary",
                "outcome": "worked"
            },
            session_id="test_session"
        )

        # Should have called record_outcome for each cached doc
        assert mock_memory.record_outcome.call_count >= 2

    @pytest.mark.asyncio
    async def test_record_clears_caches(self, mock_memory):
        """Should clear caches after recording."""
        from tests.mcp.mcp_tool_harness import (
            call_record_response, set_search_cache, get_search_cache
        )

        set_search_cache("test_session", {
            "doc_ids": ["doc_1"],
            "query": "test",
            "collections": ["working"],
            "timestamp": datetime.now()
        })

        await call_record_response(
            memory=mock_memory,
            arguments={"key_takeaway": "Done", "outcome": "worked"},
            session_id="test_session"
        )

        # Cache should be cleared
        cache = get_search_cache("test_session")
        assert cache is None or len(cache.get("doc_ids", [])) == 0

    @pytest.mark.asyncio
    async def test_record_outcome_mapping(self, mock_memory):
        """Should map outcomes to correct scores."""
        from tests.mcp.mcp_tool_harness import call_record_response

        # Test worked -> 0.7 initial score
        await call_record_response(
            memory=mock_memory,
            arguments={"key_takeaway": "Test", "outcome": "worked"},
            session_id="test_session"
        )

        call_args = mock_memory.store.call_args
        metadata = call_args.kwargs.get("metadata", {})
        # Worked should have higher initial score
        assert metadata.get("score", 0.7) >= 0.7

    @pytest.mark.asyncio
    async def test_record_tracks_actions(self, mock_memory):
        """Should score cached actions with outcome."""
        from tests.mcp.mcp_tool_harness import (
            call_record_response, set_action_cache
        )

        # Pre-populate action cache
        set_action_cache("test_session", [{
            "action_type": "search_memory",
            "context_type": "coding",
            "outcome": "unknown",
            "collection": "books"
        }])

        await call_record_response(
            memory=mock_memory,
            arguments={"key_takeaway": "Test", "outcome": "worked"},
            session_id="test_session"
        )

        # Should have called record_action_outcome
        mock_memory.record_action_outcome.assert_called()


class TestToolSchemaCompliance:
    """Test that tools comply with their schema definitions."""

    def test_search_memory_required_params(self):
        """search_memory requires 'query' parameter."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        # Should pass with query
        assert validate_tool_args("search_memory", {"query": "test"}) is True

        # Should fail without query
        assert validate_tool_args("search_memory", {}) is False

    def test_add_to_memory_bank_required_params(self):
        """add_to_memory_bank requires 'content' parameter."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        assert validate_tool_args("add_to_memory_bank", {"content": "test"}) is True
        assert validate_tool_args("add_to_memory_bank", {}) is False

    def test_update_memory_required_params(self):
        """update_memory requires 'old_content' and 'new_content'."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        assert validate_tool_args("update_memory", {
            "old_content": "old",
            "new_content": "new"
        }) is True
        assert validate_tool_args("update_memory", {"old_content": "old"}) is False

    def test_archive_memory_required_params(self):
        """archive_memory requires 'content' parameter."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        assert validate_tool_args("archive_memory", {"content": "test"}) is True
        assert validate_tool_args("archive_memory", {}) is False

    def test_get_context_insights_required_params(self):
        """get_context_insights requires 'query' parameter."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        assert validate_tool_args("get_context_insights", {"query": "test"}) is True
        assert validate_tool_args("get_context_insights", {}) is False

    def test_record_response_required_params(self):
        """record_response requires 'key_takeaway' parameter."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        assert validate_tool_args("record_response", {"key_takeaway": "test"}) is True
        assert validate_tool_args("record_response", {}) is False

    def test_record_response_outcome_enum(self):
        """record_response 'outcome' must be valid enum value."""
        from tests.mcp.mcp_tool_harness import validate_tool_args

        # Valid outcomes
        for outcome in ["worked", "failed", "partial", "unknown"]:
            assert validate_tool_args("record_response", {
                "key_takeaway": "test",
                "outcome": outcome
            }) is True

        # Invalid outcome
        assert validate_tool_args("record_response", {
            "key_takeaway": "test",
            "outcome": "invalid"
        }) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
