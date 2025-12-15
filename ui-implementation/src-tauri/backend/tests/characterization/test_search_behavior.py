r"""
Characterization Tests for UnifiedMemorySystem.search()

These tests capture the CURRENT behavior of the search() method.
They serve as a regression safety net during refactoring.

Uses lazy imports inside fixtures to ensure path is set correctly.

Usage:
    cd C:/ROAMPAL-REFACTOR
    python -m pytest tests/characterization/ -v
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

import pytest


def clear_memory_modules():
    """Clear all cached modules.memory.* modules to force fresh import."""
    import sys
    to_delete = [key for key in sys.modules if key.startswith("modules.memory") or key == "modules"]
    for key in to_delete:
        del sys.modules[key]


def get_original_memory_system():
    """Import and return the original UnifiedMemorySystem.

    Uses importlib to ensure fresh import with correct path.
    """
    import sys

    # Clear cached modules first
    clear_memory_modules()

    # Remove refactor path if present and ensure original is first
    refactor_path = "C:/ROAMPAL-REFACTOR"
    backend_path = "C:/ROAMPAL/ui-implementation/src-tauri/backend"

    # Remove refactor path if it's there
    while refactor_path in sys.path:
        sys.path.remove(refactor_path)

    # Ensure original backend path is first
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    elif sys.path.index(backend_path) != 0:
        sys.path.remove(backend_path)
        sys.path.insert(0, backend_path)

    from modules.memory.unified_memory_system import UnifiedMemorySystem
    return UnifiedMemorySystem


def get_original_wilson_score():
    """Import and return the original wilson_score_lower function."""
    import sys

    # Clear cached modules first
    clear_memory_modules()

    # Remove refactor path if present and ensure original is first
    refactor_path = "C:/ROAMPAL-REFACTOR"
    backend_path = "C:/ROAMPAL/ui-implementation/src-tauri/backend"

    # Remove refactor path if it's there
    while refactor_path in sys.path:
        sys.path.remove(refactor_path)

    # Ensure original backend path is first
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    elif sys.path.index(backend_path) != 0:
        sys.path.remove(backend_path)
        sys.path.insert(0, backend_path)

    from modules.memory.scoring_service import wilson_score_lower
    return wilson_score_lower


# Characterization queries from the plan
CHARACTERIZATION_QUERIES = [
    "how do I search memory",
    "python async patterns",
    "",  # empty query
    "user preference for dark mode",
    "API rate limiting",
    "remember my name is John",
    "what books have I read",
    "coding best practices",
]


class TestSearchBehavior:
    """Capture current search() behavior for regression testing."""

    @pytest.fixture(scope="class")
    def memory_system(self):
        """Create a single instance for all tests in this class."""
        UnifiedMemorySystem = get_original_memory_system()
        # Use production data directory (read-only access)
        ms = UnifiedMemorySystem(
            data_dir="C:/ROAMPAL/ui-implementation/src-tauri/binaries/data",
            use_server=False  # Don't start embedding server
        )
        return ms

    @pytest.mark.asyncio
    async def test_search_returns_list(self, memory_system):
        """Search should return a list."""
        results = await memory_system.search("test query", limit=5)
        assert isinstance(results, list), f"Expected list, got {type(results)}"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory_system):
        """Search should respect the limit parameter."""
        for limit in [1, 3, 5, 10]:
            results = await memory_system.search("python", limit=limit)
            assert len(results) <= limit, f"Expected <= {limit} results, got {len(results)}"

    @pytest.mark.asyncio
    async def test_search_result_structure(self, memory_system):
        """Each search result should have expected fields."""
        results = await memory_system.search("test", limit=3)

        if results:  # Only check if we got results
            result = results[0]
            # Check for expected fields (actual API returns these)
            expected_fields = ["collection", "metadata", "distance"]
            for field in expected_fields:
                assert field in result, f"Missing field: {field}"
            # Text is in metadata or as 'text' field
            assert "text" in result or "text" in result.get("metadata", {}), \
                "Missing text in result"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, memory_system):
        """Empty query should not crash."""
        results = await memory_system.search("", limit=5)
        assert isinstance(results, list), "Empty query should return list"

    @pytest.mark.asyncio
    async def test_search_with_collection_filter(self, memory_system):
        """Search with explicit collection should only return from that collection."""
        collections = ["memory_bank", "working", "history", "patterns", "books"]

        for coll in collections:
            results = await memory_system.search(
                "test",
                collections=[coll],
                limit=3
            )
            for result in results:
                assert result.get("collection") == coll, \
                    f"Expected collection {coll}, got {result.get('collection')}"

    @pytest.mark.asyncio
    async def test_search_with_metadata(self, memory_system):
        """Search with return_metadata=True should include scoring details."""
        results = await memory_system.search(
            "python programming",
            limit=3,
            return_metadata=True
        )

        # When return_metadata is True, result structure may differ
        # Capture whatever the current behavior is
        assert results is not None

    @pytest.mark.asyncio
    async def test_search_ranking_consistency(self, memory_system):
        """Same query should return same ranking (deterministic)."""
        query = "python async patterns"

        results1 = await memory_system.search(query, limit=5)
        results2 = await memory_system.search(query, limit=5)

        # Extract doc IDs for comparison
        ids1 = [r.get("metadata", {}).get("id") or r.get("id") for r in results1]
        ids2 = [r.get("metadata", {}).get("id") or r.get("id") for r in results2]

        assert ids1 == ids2, "Search ranking should be deterministic"


class TestWilsonScore:
    """Test Wilson score calculation."""

    def test_wilson_score_import(self):
        """Wilson score function should be importable."""
        wilson_score_lower = get_original_wilson_score()
        assert callable(wilson_score_lower)

    def test_wilson_score_zero_uses(self):
        """Zero uses should return 0.5 (neutral)."""
        wilson_score_lower = get_original_wilson_score()
        score = wilson_score_lower(0, 0)
        assert score == 0.5, f"Expected 0.5 for zero uses, got {score}"

    def test_wilson_score_perfect_record(self):
        """Perfect record with few uses should be lower than proven record."""
        wilson_score_lower = get_original_wilson_score()

        # 1/1 = 100% but low confidence
        new_score = wilson_score_lower(1, 1)

        # 90/100 = 90% but high confidence
        proven_score = wilson_score_lower(90, 100)

        # Proven should beat lucky newcomer
        assert proven_score > new_score, \
            f"Proven ({proven_score}) should beat newcomer ({new_score})"

    def test_wilson_score_range(self):
        """Wilson score should always be between 0 and 1."""
        wilson_score_lower = get_original_wilson_score()

        test_cases = [
            (0, 1),
            (1, 1),
            (5, 10),
            (50, 100),
            (99, 100),
            (100, 100),
        ]

        for successes, total in test_cases:
            score = wilson_score_lower(successes, total)
            assert 0 <= score <= 1, f"Score {score} out of range for {successes}/{total}"


class TestMCPToolShapes:
    """Verify MCP tool response shapes match expected format."""

    @pytest.fixture(scope="class")
    def memory_system(self):
        UnifiedMemorySystem = get_original_memory_system()
        ms = UnifiedMemorySystem(
            data_dir="C:/ROAMPAL/ui-implementation/src-tauri/binaries/data",
            use_server=False
        )
        return ms

    @pytest.mark.asyncio
    async def test_search_memory_shape(self, memory_system):
        """search_memory MCP tool returns expected shape."""
        results = await memory_system.search("test", limit=5)

        # Should be a list
        assert isinstance(results, list)

        # Capture the shape for regression
        if results:
            result = results[0]
            shape = set(result.keys())
            # Log for baseline capture
            print(f"search_memory result keys: {shape}")

    @pytest.mark.asyncio
    async def test_get_context_insights_shape(self, memory_system):
        """get_cold_start_context returns expected shape."""
        result = await memory_system.get_cold_start_context(limit=5)

        # Can be None or string
        assert result is None or isinstance(result, str)
        print(f"get_cold_start_context type: {type(result)}")

    @pytest.mark.asyncio
    async def test_store_memory_bank_shape(self, memory_system):
        """store_memory_bank returns expected shape."""
        # Don't actually store - just verify the method signature exists
        import inspect
        sig = inspect.signature(memory_system.store_memory_bank)
        params = list(sig.parameters.keys())

        # Actual API uses 'text' not 'content'
        expected_params = ["text", "tags", "importance", "confidence"]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
