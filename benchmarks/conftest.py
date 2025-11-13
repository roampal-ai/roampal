"""
Pytest configuration and shared fixtures for Roampal benchmarks.

This file provides:
- Isolated test ChromaDB instances
- Test memory system setup/teardown
- Synthetic test data fixtures
- Helper functions for benchmarking
"""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
import pytest

# Add backend to Python path for imports
BACKEND_PATH = Path(__file__).parent.parent / "ui-implementation" / "src-tauri" / "backend"
sys.path.insert(0, str(BACKEND_PATH))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="roampal_benchmark_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def test_memory_system(temp_data_dir):
    """
    Create an isolated UnifiedMemorySystem instance for testing.

    Uses ephemeral ChromaDB in temp directory, cleaned up after test.
    """
    from modules.memory.unified_memory_system import UnifiedMemorySystem

    # Create isolated memory system with temp data directory
    # use_server=False forces embedded ChromaDB (no server required)
    memory = UnifiedMemorySystem(data_dir=temp_data_dir, use_server=False)
    await memory.initialize()

    yield memory

    # Cleanup (ChromaDB will be removed with temp_dir)
    memory = None


@pytest.fixture
def high_importance_facts():
    """Synthetic high-importance memory bank facts for ranking tests."""
    return [
        {
            "content": "User's name is Logan and works at EverBright",
            "importance": 0.95,
            "confidence": 1.0,
            "tags": ["identity", "work"]
        },
        {
            "content": "User is building Roampal, an AI memory system with ChromaDB",
            "importance": 0.9,
            "confidence": 1.0,
            "tags": ["project", "technical"]
        },
        {
            "content": "User has MBA from WGU, completed in 4 months",
            "importance": 0.8,
            "confidence": 1.0,
            "tags": ["education", "achievement"]
        }
    ]


@pytest.fixture
def low_importance_facts():
    """Synthetic low-importance memory bank facts for ranking tests."""
    return [
        {
            "content": "User said 'cool' in a conversation once",
            "importance": 0.2,
            "confidence": 0.3,
            "tags": ["casual"]
        },
        {
            "content": "Random test data AAAAAAAAAA",
            "importance": 0.1,
            "confidence": 0.1,
            "tags": ["test"]
        }
    ]


@pytest.fixture
def test_book_content():
    """Synthetic book content for books search tests."""
    return {
        "title": "Roampal Architecture Guide",
        "author": "Test Author",
        "content": """
# Roampal Architecture

## Memory System
Roampal implements a 5-tier memory architecture:
1. Books - Permanent reference documentation
2. Memory Bank - User identity and preferences
3. Working - Current conversation context (24h retention)
4. History - Past conversations (30 day retention)
5. Patterns - Proven solutions (permanent)

## Knowledge Graphs
Two knowledge graphs power intelligent routing:
- Routing KG: Maps queries to collections
- Content KG: Tracks entity relationships and mentions

## Cold-Start Auto-Trigger
On message #1, the system automatically injects user profile from Content KG.
This ensures personalized responses without requiring manual memory search.
"""
    }


@pytest.fixture
def routing_test_queries():
    """Test queries with expected collection routing."""
    return {
        "python bug fix": ["patterns", "working"],
        "architecture documentation": ["books"],
        "who am I": ["memory_bank"],
        "recent conversation": ["history", "working"],
        "fix error message": ["patterns", "working"]
    }


# Helper functions

def calculate_precision_at_k(results: list, relevant_ids: set, k: int = 5) -> float:
    """Calculate precision@k for ranking tests."""
    if not results or k == 0:
        return 0.0

    top_k = results[:k]
    relevant_in_top_k = sum(1 for r in top_k if r.get('id') in relevant_ids)
    return relevant_in_top_k / k


def calculate_recall_at_k(results: list, relevant_ids: set, k: int = 5) -> float:
    """Calculate recall@k for search tests."""
    if not relevant_ids or not results:
        return 0.0

    top_k = results[:k]
    relevant_in_top_k = sum(1 for r in top_k if r.get('id') in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)
