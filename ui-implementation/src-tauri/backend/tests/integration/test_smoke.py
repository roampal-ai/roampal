"""
Smoke Tests for Roampal Desktop

Quick end-to-end tests to verify the app works after deployment.
Run these after building to catch obvious breakage.

Usage:
    pytest tests/integration/test_smoke.py -v
"""

import sys
import os
import tempfile
import shutil
import asyncio
from pathlib import Path
from datetime import datetime

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestBackendSmoke:
    """Smoke tests for backend startup and basic operations."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_smoke_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_chromadb_imports(self):
        """Verify ChromaDB imports correctly."""
        import chromadb
        assert chromadb.__version__.startswith("1."), f"Expected ChromaDB 1.x, got {chromadb.__version__}"

    def test_backend_modules_import(self):
        """Verify all critical backend modules import without error."""
        # These imports will fail if dependencies are broken
        from modules.memory.chromadb_adapter import ChromaDBAdapter
        from modules.memory.unified_memory_system import UnifiedMemorySystem
        from modules.memory.promotion_service import PromotionService

        assert ChromaDBAdapter is not None
        assert UnifiedMemorySystem is not None
        assert PromotionService is not None

    def test_chromadb_adapter_lifecycle(self, temp_data_dir):
        """Test ChromaDB adapter can initialize, add, query, and cleanup."""
        from modules.memory.chromadb_adapter import ChromaDBAdapter

        adapter = ChromaDBAdapter(
            persistence_directory=temp_data_dir,
            use_server=False
        )

        # Initialize
        asyncio.run(adapter.initialize(collection_name="smoke_test"))
        assert adapter.collection is not None

        # Add a document
        adapter.collection.add(
            ids=["smoke_1"],
            embeddings=[[0.1] * 768],
            documents=["Smoke test document"],
            metadatas=[{"text": "Smoke test", "score": 0.5}]
        )

        # Verify it exists
        result = adapter.get_fragment("smoke_1")
        assert result is not None, "Should find added document"

        # Delete it
        adapter.delete_vectors(["smoke_1"])

        # Verify deletion
        result = adapter.get_fragment("smoke_1")
        assert result is None, "Should be deleted"

        # Cleanup
        asyncio.run(adapter.cleanup())

    def test_memory_persistence_across_sessions(self, temp_data_dir):
        """Test that memories persist when closing and reopening."""
        from modules.memory.chromadb_adapter import ChromaDBAdapter

        # Session 1: Create and add
        adapter1 = ChromaDBAdapter(persistence_directory=temp_data_dir, use_server=False)
        asyncio.run(adapter1.initialize(collection_name="persist_test"))

        adapter1.collection.add(
            ids=["persist_1"],
            embeddings=[[0.2] * 768],
            documents=["This should persist"],
            metadatas=[{"text": "Persistent memory"}]
        )

        asyncio.run(adapter1.cleanup())
        del adapter1

        # Session 2: Reopen and verify
        adapter2 = ChromaDBAdapter(persistence_directory=temp_data_dir, use_server=False)
        asyncio.run(adapter2.initialize(collection_name="persist_test"))

        result = adapter2.get_fragment("persist_1")
        assert result is not None, "Memory should persist across sessions"

        asyncio.run(adapter2.cleanup())

    def test_unified_memory_system_init(self, temp_data_dir):
        """Test UnifiedMemorySystem can initialize in embedded mode."""
        from modules.memory.unified_memory_system import UnifiedMemorySystem

        # Force embedded mode via environment
        os.environ["CHROMADB_PERSIST_DIRECTORY"] = temp_data_dir
        os.environ["CHROMADB_USE_SERVER"] = "false"
        os.environ["ROAMPAL_CONTAINER"] = "false"

        # Create UMS with explicit embedded config
        ums = UnifiedMemorySystem(
            data_dir=temp_data_dir,
            use_server=False  # Force embedded mode
        )

        # Should be able to initialize (async)
        asyncio.run(ums.initialize())

        # Should have collections
        assert ums.collections is not None

        # Cleanup
        asyncio.run(ums.cleanup())


class TestReleaseBundle:
    """Tests specific to the release bundle."""

    def test_bundled_python_has_chromadb(self):
        """Verify bundled Python has ChromaDB 1.x (if running from bundle)."""
        import chromadb
        version = chromadb.__version__
        assert version.startswith("1."), f"ChromaDB should be 1.x, got {version}"

    def test_required_packages_available(self):
        """Verify all required packages can be imported."""
        required = [
            "chromadb",
            "sentence_transformers",
            "fastapi",
            "uvicorn",
            "httpx",
            "numpy",
            "pandas",
        ]

        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)

        assert not missing, f"Missing required packages: {missing}"


class TestMemoryOperations:
    """Test core memory operations work correctly."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_smoke_mem_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_and_search_memory(self, temp_data_dir):
        """Test adding a memory and searching for it."""
        from modules.memory.chromadb_adapter import ChromaDBAdapter

        adapter = ChromaDBAdapter(persistence_directory=temp_data_dir, use_server=False)
        asyncio.run(adapter.initialize(collection_name="search_test"))

        # Add test memories
        adapter.collection.add(
            ids=["mem_1", "mem_2", "mem_3"],
            embeddings=[[0.1] * 768, [0.5] * 768, [0.9] * 768],
            documents=["Python programming tips", "JavaScript async patterns", "Rust memory safety"],
            metadatas=[
                {"text": "Python programming tips", "score": 0.7},
                {"text": "JavaScript async patterns", "score": 0.6},
                {"text": "Rust memory safety", "score": 0.8}
            ]
        )

        # List all
        all_ids = adapter.list_all_ids()
        assert len(all_ids) == 3, f"Should have 3 memories, got {len(all_ids)}"

        # Get specific
        result = adapter.get_fragment("mem_2")
        assert result is not None
        assert "JavaScript" in result.get("content", "")

        asyncio.run(adapter.cleanup())

    def test_delete_memory_persists(self, temp_data_dir):
        """Test that deleting a memory persists to disk (the v0.2.10 fix)."""
        from modules.memory.chromadb_adapter import ChromaDBAdapter

        # Session 1: Add and delete
        adapter1 = ChromaDBAdapter(persistence_directory=temp_data_dir, use_server=False)
        asyncio.run(adapter1.initialize(collection_name="delete_test"))

        adapter1.collection.add(
            ids=["to_delete", "to_keep"],
            embeddings=[[0.1] * 768, [0.2] * 768],
            documents=["Delete me", "Keep me"],
            metadatas=[{"text": "Delete me"}, {"text": "Keep me"}]
        )

        # Delete one
        adapter1.delete_vectors(["to_delete"])

        # Verify immediate deletion
        assert adapter1.get_fragment("to_delete") is None
        assert adapter1.get_fragment("to_keep") is not None

        asyncio.run(adapter1.cleanup())
        del adapter1

        # Session 2: Verify deletion persisted
        adapter2 = ChromaDBAdapter(persistence_directory=temp_data_dir, use_server=False)
        asyncio.run(adapter2.initialize(collection_name="delete_test"))

        # This was the bug in v0.2.9 - deleted items would reappear
        assert adapter2.get_fragment("to_delete") is None, "Deleted item should NOT reappear!"
        assert adapter2.get_fragment("to_keep") is not None, "Kept item should still exist"

        asyncio.run(adapter2.cleanup())


class TestHealthCheck:
    """Test health check endpoints (if applicable)."""

    def test_can_create_fastapi_app(self):
        """Verify FastAPI app can be created."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok"}

        assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])