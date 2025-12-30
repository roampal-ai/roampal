"""
Ghost Registry - Tracks deleted book chunk IDs to filter from search results.

v0.2.9: Fixes the ghost vector issue where deleted book chunks remain in
ChromaDB's HNSW index but return [No content] when retrieved.

Problem: ChromaDB's delete() removes records from SQLite but leaves vectors
in the HNSW binary index (data_level0.bin). Searches still find these
"ghost" vectors but content retrieval fails.

Solution: Track deleted chunk IDs in a JSON file and filter them out at
query time before returning results to users.
"""

import json
import logging
from pathlib import Path
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


class GhostRegistry:
    """
    Registry for tracking deleted book chunk IDs.

    Ghosts are chunk IDs that have been deleted from ChromaDB's SQLite store
    but remain as vectors in the HNSW index. This registry tracks them so
    they can be filtered out of search results.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the ghost registry.

        Args:
            data_dir: Path to data directory. ghost_ids.json will be stored here.
                     If None, ghosts are only tracked in memory.
        """
        self._ghosts: Set[str] = set()
        self._data_dir = data_dir
        self._file_path: Optional[Path] = None

        if data_dir:
            self._file_path = Path(data_dir) / "ghost_ids.json"
            self._load()

    def _load(self) -> None:
        """Load ghost IDs from disk."""
        if not self._file_path or not self._file_path.exists():
            return

        try:
            with open(self._file_path, 'r') as f:
                data = json.load(f)
                self._ghosts = set(data.get("ghost_ids", []))
            logger.debug(f"Loaded {len(self._ghosts)} ghost IDs from {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to load ghost registry: {e}")
            self._ghosts = set()

    def _save(self) -> None:
        """Save ghost IDs to disk."""
        if not self._file_path:
            return

        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, 'w') as f:
                json.dump({"ghost_ids": list(self._ghosts)}, f, indent=2)
            logger.debug(f"Saved {len(self._ghosts)} ghost IDs to {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to save ghost registry: {e}")

    def add(self, chunk_ids: List[str]) -> int:
        """
        Add chunk IDs to the ghost registry.

        Args:
            chunk_ids: List of chunk IDs that were deleted

        Returns:
            Number of new ghosts added (excludes duplicates)
        """
        before = len(self._ghosts)
        self._ghosts.update(chunk_ids)
        added = len(self._ghosts) - before

        if added > 0:
            self._save()
            logger.info(f"Added {added} chunk IDs to ghost registry (total: {len(self._ghosts)})")

        return added

    def is_ghost(self, chunk_id: str) -> bool:
        """
        Check if a chunk ID is a ghost (was deleted).

        Args:
            chunk_id: The chunk ID to check

        Returns:
            True if the chunk is a ghost, False otherwise
        """
        return chunk_id in self._ghosts

    def filter_ghosts(self, results: List[dict]) -> List[dict]:
        """
        Filter ghost results from a list of search results.

        Args:
            results: List of search result dicts with 'id' keys

        Returns:
            Filtered list with ghosts removed
        """
        if not self._ghosts:
            return results

        filtered = []
        removed = 0
        for r in results:
            chunk_id = r.get('id') or r.get('doc_id')
            if chunk_id and self.is_ghost(chunk_id):
                removed += 1
            else:
                filtered.append(r)

        if removed > 0:
            logger.debug(f"Filtered {removed} ghost results from search")

        return filtered

    def clear(self) -> int:
        """
        Clear all ghost IDs from the registry.

        Call this after nuking/recreating the books collection.

        Returns:
            Number of ghosts that were cleared
        """
        count = len(self._ghosts)
        self._ghosts.clear()
        self._save()

        if count > 0:
            logger.info(f"Cleared {count} ghost IDs from registry")

        return count

    def count(self) -> int:
        """Return the number of tracked ghosts."""
        return len(self._ghosts)

    def get_all(self) -> List[str]:
        """Return all tracked ghost IDs."""
        return list(self._ghosts)


# Module-level singleton for easy access
_registry: Optional[GhostRegistry] = None


def get_ghost_registry(data_dir: Optional[Path] = None) -> GhostRegistry:
    """
    Get the singleton ghost registry instance.

    Args:
        data_dir: Data directory path (only used on first call)

    Returns:
        The ghost registry singleton
    """
    global _registry
    if _registry is None:
        _registry = GhostRegistry(data_dir)
    return _registry


def reset_ghost_registry() -> None:
    """Reset the singleton (for testing)."""
    global _registry
    _registry = None
