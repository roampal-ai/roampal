"""
Unit tests for GhostRegistry.

Tests the ghost tracking system that filters deleted book chunks from search results.
"""

import pytest
import tempfile
import json
from pathlib import Path

from modules.memory.ghost_registry import (
    GhostRegistry,
    get_ghost_registry,
    reset_ghost_registry
)


class TestGhostRegistry:
    """Tests for GhostRegistry class."""

    def test_init_without_data_dir(self):
        """Registry works in memory-only mode without data_dir."""
        registry = GhostRegistry()
        assert registry.count() == 0
        assert registry._file_path is None

    def test_init_with_data_dir(self, tmp_path):
        """Registry creates file path when data_dir provided."""
        registry = GhostRegistry(tmp_path)
        assert registry._file_path == tmp_path / "ghost_ids.json"
        assert registry.count() == 0

    def test_add_single_ghost(self, tmp_path):
        """Adding a ghost ID increases count and is_ghost returns True."""
        registry = GhostRegistry(tmp_path)

        added = registry.add(["chunk_123"])

        assert added == 1
        assert registry.count() == 1
        assert registry.is_ghost("chunk_123")
        assert not registry.is_ghost("chunk_456")

    def test_add_multiple_ghosts(self, tmp_path):
        """Adding multiple ghost IDs works correctly."""
        registry = GhostRegistry(tmp_path)

        added = registry.add(["chunk_1", "chunk_2", "chunk_3"])

        assert added == 3
        assert registry.count() == 3
        assert registry.is_ghost("chunk_1")
        assert registry.is_ghost("chunk_2")
        assert registry.is_ghost("chunk_3")

    def test_add_duplicates_not_counted(self, tmp_path):
        """Adding duplicate IDs doesn't increase count."""
        registry = GhostRegistry(tmp_path)

        registry.add(["chunk_1", "chunk_2"])
        added = registry.add(["chunk_2", "chunk_3"])  # chunk_2 is duplicate

        assert added == 1  # Only chunk_3 is new
        assert registry.count() == 3

    def test_filter_ghosts_removes_ghost_results(self, tmp_path):
        """filter_ghosts removes results with ghost IDs."""
        registry = GhostRegistry(tmp_path)
        registry.add(["ghost_1", "ghost_2"])

        results = [
            {"id": "ghost_1", "text": "deleted content"},
            {"id": "good_1", "text": "valid content"},
            {"id": "ghost_2", "text": "also deleted"},
            {"id": "good_2", "text": "also valid"},
        ]

        filtered = registry.filter_ghosts(results)

        assert len(filtered) == 2
        assert filtered[0]["id"] == "good_1"
        assert filtered[1]["id"] == "good_2"

    def test_filter_ghosts_uses_doc_id_fallback(self, tmp_path):
        """filter_ghosts checks 'doc_id' key if 'id' is missing."""
        registry = GhostRegistry(tmp_path)
        registry.add(["ghost_1"])

        results = [
            {"doc_id": "ghost_1", "text": "deleted"},
            {"doc_id": "good_1", "text": "valid"},
        ]

        filtered = registry.filter_ghosts(results)

        assert len(filtered) == 1
        assert filtered[0]["doc_id"] == "good_1"

    def test_filter_ghosts_empty_registry_returns_all(self, tmp_path):
        """filter_ghosts returns all results when registry is empty."""
        registry = GhostRegistry(tmp_path)

        results = [
            {"id": "chunk_1", "text": "content 1"},
            {"id": "chunk_2", "text": "content 2"},
        ]

        filtered = registry.filter_ghosts(results)

        assert len(filtered) == 2

    def test_clear_removes_all_ghosts(self, tmp_path):
        """clear() removes all ghost IDs."""
        registry = GhostRegistry(tmp_path)
        registry.add(["chunk_1", "chunk_2", "chunk_3"])

        cleared = registry.clear()

        assert cleared == 3
        assert registry.count() == 0
        assert not registry.is_ghost("chunk_1")

    def test_get_all_returns_list(self, tmp_path):
        """get_all() returns list of all ghost IDs."""
        registry = GhostRegistry(tmp_path)
        registry.add(["chunk_1", "chunk_2"])

        all_ghosts = registry.get_all()

        assert set(all_ghosts) == {"chunk_1", "chunk_2"}

    def test_persistence_save_and_load(self, tmp_path):
        """Ghost IDs persist to disk and load on init."""
        # Create registry and add ghosts
        registry1 = GhostRegistry(tmp_path)
        registry1.add(["chunk_1", "chunk_2"])

        # Create new registry instance - should load from file
        registry2 = GhostRegistry(tmp_path)

        assert registry2.count() == 2
        assert registry2.is_ghost("chunk_1")
        assert registry2.is_ghost("chunk_2")

    def test_persistence_file_format(self, tmp_path):
        """Saved file has expected JSON structure."""
        registry = GhostRegistry(tmp_path)
        registry.add(["chunk_1", "chunk_2"])

        file_path = tmp_path / "ghost_ids.json"
        with open(file_path) as f:
            data = json.load(f)

        assert "ghost_ids" in data
        assert set(data["ghost_ids"]) == {"chunk_1", "chunk_2"}


class TestGhostRegistrySingleton:
    """Tests for singleton pattern functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_ghost_registry()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_ghost_registry()

    def test_get_ghost_registry_creates_singleton(self):
        """get_ghost_registry returns same instance on repeated calls."""
        registry1 = get_ghost_registry()
        registry2 = get_ghost_registry()

        assert registry1 is registry2

    def test_get_ghost_registry_with_data_dir(self, tmp_path):
        """First call with data_dir initializes persistence."""
        registry = get_ghost_registry(tmp_path)
        registry.add(["test_chunk"])

        # File should exist
        assert (tmp_path / "ghost_ids.json").exists()

    def test_reset_ghost_registry_clears_singleton(self, tmp_path):
        """reset_ghost_registry allows creating new instance."""
        registry1 = get_ghost_registry(tmp_path)
        registry1.add(["chunk_1"])

        reset_ghost_registry()

        registry2 = get_ghost_registry()  # No data_dir - memory only
        assert registry2.count() == 0
        assert registry1 is not registry2
