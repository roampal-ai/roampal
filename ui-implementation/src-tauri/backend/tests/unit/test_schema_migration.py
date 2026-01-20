"""
Unit tests for ChromaDB schema migration.

Tests the _migrate_chromadb_schema() method in UnifiedMemorySystem
that handles upgrades from ChromaDB 0.4.x/0.5.x to 1.x.
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
import sqlite3
import tempfile


class TestSchemaMigration:
    """Tests for ChromaDB schema migration."""

    def test_migration_adds_missing_topic_columns(self, tmp_path):
        """Test that migration adds 'topic' column to both collections and segments tables."""
        # Create old-style ChromaDB schema (pre-1.x without 'topic' columns)
        chromadb_path = tmp_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create minimal tables WITHOUT topic columns
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dimension INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT
            )
        """)
        cursor.execute("INSERT INTO collections (id, name, dimension) VALUES ('test-id', 'test', 768)")
        cursor.execute("INSERT INTO segments (id, type, scope) VALUES ('seg-id', 'hnsw', 'VECTOR')")
        conn.commit()
        conn.close()

        # Verify topic columns don't exist
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(collections)")
        collections_before = {col[1] for col in cursor.fetchall()}
        cursor.execute("PRAGMA table_info(segments)")
        segments_before = {col[1] for col in cursor.fetchall()}
        conn.close()
        assert 'topic' not in collections_before
        assert 'topic' not in segments_before

        # Run migration
        from modules.memory.unified_memory_system import UnifiedMemorySystem
        ums = UnifiedMemorySystem(data_dir=str(tmp_path))
        ums._migrate_chromadb_schema()

        # Verify topic columns were added to both tables
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(collections)")
        collections_after = {col[1] for col in cursor.fetchall()}
        cursor.execute("PRAGMA table_info(segments)")
        segments_after = {col[1] for col in cursor.fetchall()}
        conn.close()

        assert 'topic' in collections_after
        assert 'topic' in segments_after

    def test_migration_is_idempotent(self, tmp_path):
        """Test that running migration multiple times is safe."""
        chromadb_path = tmp_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create schema WITH topic columns (already migrated)
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dimension INTEGER,
                topic TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT,
                topic TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Run migration multiple times - should not raise
        from modules.memory.unified_memory_system import UnifiedMemorySystem
        ums = UnifiedMemorySystem(data_dir=str(tmp_path))

        # Should not raise on first call
        ums._migrate_chromadb_schema()

        # Should not raise on second call (idempotent)
        ums._migrate_chromadb_schema()

        # Verify both tables still have topic column
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(collections)")
        collections_cols = {col[1] for col in cursor.fetchall()}
        cursor.execute("PRAGMA table_info(segments)")
        segments_cols = {col[1] for col in cursor.fetchall()}
        conn.close()

        assert 'topic' in collections_cols
        assert 'topic' in segments_cols

    def test_migration_skips_when_no_database(self, tmp_path):
        """Test that migration gracefully skips when no chroma.sqlite3 exists."""
        chromadb_path = tmp_path / "chromadb"
        chromadb_path.mkdir()
        # Note: NOT creating chroma.sqlite3

        from modules.memory.unified_memory_system import UnifiedMemorySystem
        ums = UnifiedMemorySystem(data_dir=str(tmp_path))

        # Should not raise - just skip
        ums._migrate_chromadb_schema()

    def test_migration_preserves_existing_data(self, tmp_path):
        """Test that migration doesn't lose existing collection data."""
        chromadb_path = tmp_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create old schema with data
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dimension INTEGER
            )
        """)
        cursor.execute("INSERT INTO collections (id, name, dimension) VALUES ('coll-1', 'roampal_books', 768)")
        cursor.execute("INSERT INTO collections (id, name, dimension) VALUES ('coll-2', 'roampal_memory_bank', 768)")
        conn.commit()
        conn.close()

        # Run migration
        from modules.memory.unified_memory_system import UnifiedMemorySystem
        ums = UnifiedMemorySystem(data_dir=str(tmp_path))
        ums._migrate_chromadb_schema()

        # Verify data still exists
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM collections ORDER BY name")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'roampal_books' in names
        assert 'roampal_memory_bank' in names

    def test_migration_handles_corrupted_database(self, tmp_path):
        """Test that migration handles corrupted database gracefully."""
        chromadb_path = tmp_path / "chromadb"
        chromadb_path.mkdir()
        sqlite_path = chromadb_path / "chroma.sqlite3"

        # Create corrupted file (not valid SQLite)
        sqlite_path.write_bytes(b"not a valid sqlite database")

        from modules.memory.unified_memory_system import UnifiedMemorySystem
        ums = UnifiedMemorySystem(data_dir=str(tmp_path))

        # Should not raise - just log warning
        ums._migrate_chromadb_schema()