"""
Pytest configuration for MCP tool tests.

Provides fixtures for testing MCP tools either:
1. Directly via the call_tool handler (unit testing)
2. Via MCP protocol simulation (integration testing)
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


@pytest.fixture
def mock_memory_system():
    """Create a mock UnifiedMemorySystem for testing MCP handlers."""
    memory = MagicMock()
    memory.initialized = True

    # Mock search
    async def mock_search(query, collections=None, limit=5, metadata_filters=None):
        return [
            {
                "id": f"doc_{i}",
                "doc_id": f"doc_{i}",
                "content": f"Test result {i} for query: {query}",
                "text": f"Test result {i} for query: {query}",
                "collection": collections[0] if collections else "working",
                "metadata": {
                    "score": 0.8 - i * 0.1,
                    "uses": 3 - i,
                    "timestamp": datetime.now().isoformat(),
                    "last_outcome": "worked"
                }
            }
            for i in range(min(limit, 3))
        ]
    memory.search = AsyncMock(side_effect=mock_search)

    # Mock store_memory_bank
    async def mock_store_memory_bank(text, tags=None, importance=0.7, confidence=0.7):
        return f"memory_bank_{hash(text) % 10000}"
    memory.store_memory_bank = AsyncMock(side_effect=mock_store_memory_bank)

    # Mock search_memory_bank
    async def mock_search_memory_bank(query, limit=1, include_archived=False):
        return [{"id": "memory_bank_123", "content": query}]
    memory.search_memory_bank = AsyncMock(side_effect=mock_search_memory_bank)

    # Mock update_memory_bank
    memory.update_memory_bank = AsyncMock()

    # Mock archive_memory_bank
    memory.archive_memory_bank = AsyncMock()

    # Mock detect_context_type
    memory.detect_context_type = AsyncMock(return_value="general")

    # Mock analyze_conversation_context
    memory.analyze_conversation_context = AsyncMock(return_value={
        "entities": [],
        "patterns": [],
        "context_type": "general"
    })

    # Mock get_action_effectiveness
    memory.get_action_effectiveness = MagicMock(return_value={})

    # Mock get_tier_recommendations
    memory.get_tier_recommendations = MagicMock(return_value=[])

    # Mock get_facts_for_entities
    memory.get_facts_for_entities = AsyncMock(return_value=[])

    # Mock store (for record_response)
    async def mock_store(text, collection="working", metadata=None):
        return f"{collection}_{hash(text) % 10000}"
    memory.store = AsyncMock(side_effect=mock_store)

    # Mock record_outcome
    memory.record_outcome = AsyncMock()

    # Mock record_action_outcome
    memory.record_action_outcome = AsyncMock()

    # Mock _update_kg_routing
    memory._update_kg_routing = AsyncMock()

    return memory


@pytest.fixture
def mock_mcp_caches():
    """Create fresh MCP caches for testing."""
    return {
        "search_cache": {},
        "action_cache": {},
        "first_tool_call": set()
    }


@pytest.fixture
def session_id():
    """Provide a test session ID."""
    return "test_session_123"


@pytest.fixture
def mock_data_path(tmp_path):
    """Provide a temporary data path for MCP session files."""
    mcp_sessions = tmp_path / "mcp_sessions"
    mcp_sessions.mkdir()
    return tmp_path
