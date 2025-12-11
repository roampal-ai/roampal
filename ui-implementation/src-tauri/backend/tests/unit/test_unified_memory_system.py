"""
Unit Tests for UnifiedMemorySystem Facade

Tests the refactored facade that coordinates all services.
"""

import sys
sys.path.insert(0, "C:/ROAMPAL-REFACTOR")

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.config import MemoryConfig


class TestUnifiedMemorySystemInit:
    """Test initialization."""

    def test_init_creates_data_dir(self, tmp_path):
        """Should create data directory."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert (tmp_path / "data").exists()

    def test_init_with_custom_config(self, tmp_path):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        ums = UnifiedMemorySystem(
            data_dir=str(tmp_path / "data"),
            config=config
        )
        assert ums.config.promotion_score_threshold == 0.8
        assert ums.PROMOTION_SCORE_THRESHOLD == 0.8

    def test_init_conversation_id(self, tmp_path):
        """Should generate conversation ID."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert ums.conversation_id is not None
        assert len(ums.conversation_id) > 0

    def test_init_not_initialized(self, tmp_path):
        """Should not be initialized until initialize() called."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert not ums.initialized


class TestInitialize:
    """Test initialization process."""

    @pytest.fixture
    def mock_adapter_factory(self):
        """Create mock adapter factory."""
        def factory(name):
            adapter = MagicMock()
            adapter.initialize = AsyncMock()
            adapter.list_all_ids = MagicMock(return_value=[])
            adapter.get_fragment = MagicMock(return_value=None)
            adapter.query_vectors = AsyncMock(return_value=[])
            adapter.upsert_vectors = AsyncMock()
            return adapter
        return factory

    @pytest.fixture
    def ums(self, tmp_path, mock_adapter_factory):
        """Create UMS with mocked adapters."""
        ums = UnifiedMemorySystem(
            data_dir=str(tmp_path / "data"),
            chromadb_adapter_factory=mock_adapter_factory
        )
        return ums

    @pytest.mark.asyncio
    async def test_initialize_creates_collections(self, ums):
        """Should create all collections."""
        await ums.initialize()

        assert "books" in ums.collections
        assert "working" in ums.collections
        assert "history" in ums.collections
        assert "patterns" in ums.collections
        assert "memory_bank" in ums.collections

    @pytest.mark.asyncio
    async def test_initialize_creates_services(self, ums):
        """Should initialize all services."""
        await ums.initialize()

        assert ums._scoring_service is not None
        assert ums._kg_service is not None
        assert ums._routing_service is not None
        assert ums._search_service is not None
        assert ums._promotion_service is not None
        assert ums._outcome_service is not None
        assert ums._memory_bank_service is not None
        assert ums._context_service is not None

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, ums):
        """Should only initialize once."""
        await ums.initialize()
        first_kg = ums._kg_service

        await ums.initialize()
        assert ums._kg_service is first_kg


class TestStore:
    """Test store functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with all mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        working = MagicMock()
        working.upsert_vectors = AsyncMock()
        ums.collections = {"working": working}

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        return ums

    @pytest.mark.asyncio
    async def test_store_generates_doc_id(self, mock_ums):
        """Should generate document ID."""
        doc_id = await mock_ums.store("test text")

        assert doc_id.startswith("working_")
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_embeds_text(self, mock_ums):
        """Should embed the text."""
        await mock_ums.store("test text")

        mock_ums._embedding_service.embed_text.assert_called_with("test text")

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, mock_ums):
        """Should include custom metadata."""
        await mock_ums.store(
            "test text",
            metadata={"custom": "value"}
        )

        call_args = mock_ums.collections["working"].upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["custom"] == "value"
        assert metadata["text"] == "test text"


class TestSearch:
    """Test search functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with search mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock search service
        ums._search_service = MagicMock()
        ums._search_service.search = AsyncMock(return_value=[
            {"content": "result1", "metadata": {}},
            {"content": "result2", "metadata": {}}
        ])

        return ums

    @pytest.mark.asyncio
    async def test_search_delegates_to_service(self, mock_ums):
        """Should delegate to search service."""
        results = await mock_ums.search("test query")

        mock_ums._search_service.search.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_passes_collections(self, mock_ums):
        """Should pass collections to search service."""
        await mock_ums.search(
            "test query",
            collections=["patterns", "history"]
        )

        call_args = mock_ums._search_service.search.call_args
        assert call_args[1]["collections"] == ["patterns", "history"]


class TestRecordOutcome:
    """Test outcome recording."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with outcome mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock outcome service
        ums._outcome_service = MagicMock()
        ums._outcome_service.record_outcome = AsyncMock(return_value={"score": 0.7})

        return ums

    @pytest.mark.asyncio
    async def test_record_outcome_delegates(self, mock_ums):
        """Should delegate to outcome service."""
        await mock_ums.record_outcome(
            doc_id="working_123",
            outcome="worked"
        )

        mock_ums._outcome_service.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_outcome_with_reason(self, mock_ums):
        """Should pass failure reason."""
        await mock_ums.record_outcome(
            doc_id="working_123",
            outcome="failed",
            failure_reason="Test failure"
        )

        call_args = mock_ums._outcome_service.record_outcome.call_args
        assert call_args[1]["failure_reason"] == "Test failure"


class TestMemoryBankAPI:
    """Test memory bank API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with memory bank mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock memory bank service
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.store = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.update = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.archive = AsyncMock(return_value=True)
        ums._memory_bank_service.search = AsyncMock(return_value=[])
        ums._memory_bank_service.restore = AsyncMock(return_value=True)
        ums._memory_bank_service.delete = AsyncMock(return_value=True)

        return ums

    @pytest.mark.asyncio
    async def test_store_memory_bank(self, mock_ums):
        """Should delegate to memory bank service."""
        doc_id = await mock_ums.store_memory_bank(
            text="User prefers dark mode",
            tags=["preference"]
        )

        assert doc_id == "memory_bank_123"
        mock_ums._memory_bank_service.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_memory_bank(self, mock_ums):
        """Should delegate update."""
        await mock_ums.update_memory_bank(
            doc_id="memory_bank_123",
            new_text="Updated text"
        )

        mock_ums._memory_bank_service.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_memory_bank(self, mock_ums):
        """Should delegate archive."""
        result = await mock_ums.archive_memory_bank("memory_bank_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_search_memory_bank(self, mock_ums):
        """Should delegate search."""
        await mock_ums.search_memory_bank(query="test")
        mock_ums._memory_bank_service.search.assert_called_once()


class TestContextAPI:
    """Test context analysis API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with context mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock context service
        ums._context_service = MagicMock()
        ums._context_service.analyze_conversation_context = AsyncMock(return_value={
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        })

        return ums

    @pytest.mark.asyncio
    async def test_analyze_context(self, mock_ums):
        """Should delegate to context service."""
        context = await mock_ums.analyze_conversation_context(
            current_message="test",
            recent_conversation=[],
            conversation_id="conv123"
        )

        assert "relevant_patterns" in context
        mock_ums._context_service.analyze_conversation_context.assert_called_once()


class TestPromotionAPI:
    """Test promotion API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with promotion mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock promotion service
        ums._promotion_service = MagicMock()
        ums._promotion_service.promote_valuable_working_memory = AsyncMock(return_value=5)
        ums._promotion_service.cleanup_old_working_memory = AsyncMock(return_value=3)

        return ums

    @pytest.mark.asyncio
    async def test_promote_valuable(self, mock_ums):
        """Should delegate to promotion service."""
        count = await mock_ums.promote_valuable_working_memory()

        assert count == 5
        mock_ums._promotion_service.promote_valuable_working_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old(self, mock_ums):
        """Should delegate cleanup."""
        count = await mock_ums.cleanup_old_working_memory()

        assert count == 3


class TestSessionManagement:
    """Test session/conversation management."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with session mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock promotion service
        ums._promotion_service = MagicMock()
        ums._promotion_service.promote_valuable_working_memory = AsyncMock(return_value=0)

        return ums

    @pytest.mark.asyncio
    async def test_switch_conversation(self, mock_ums):
        """Should switch conversation ID."""
        old_id = mock_ums.conversation_id

        # Pass explicit new ID to avoid timestamp collision
        new_id = await mock_ums.switch_conversation("new_conv_123")

        assert new_id == "new_conv_123"
        assert mock_ums.conversation_id == new_id
        assert mock_ums.message_count == 0

    @pytest.mark.asyncio
    async def test_switch_conversation_promotes(self, mock_ums):
        """Should promote valuable memories when switching."""
        await mock_ums.switch_conversation()

        mock_ums._promotion_service.promote_valuable_working_memory.assert_called_once()

    def test_increment_message_count(self, mock_ums):
        """Should increment message count."""
        initial = mock_ums.message_count

        mock_ums.increment_message_count()

        assert mock_ums.message_count == initial + 1


class TestKGVisualization:
    """Test KG visualization API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with KG mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock KG service
        ums._kg_service = MagicMock()
        ums._kg_service.get_kg_entities = MagicMock(return_value=[
            {"id": "concept1", "weight": 10}
        ])
        ums._kg_service.get_kg_relationships = MagicMock(return_value=[
            {"source": "c1", "target": "c2", "weight": 1}
        ])
        ums._kg_service.knowledge_graph = {"routing_patterns": {}}

        return ums

    def test_get_kg_entities(self, mock_ums):
        """Should return entities from KG service."""
        entities = mock_ums.get_kg_entities()

        assert len(entities) == 1
        mock_ums._kg_service.get_kg_entities.assert_called_once()

    def test_get_kg_relationships(self, mock_ums):
        """Should return relationships from KG service."""
        rels = mock_ums.get_kg_relationships()

        assert len(rels) == 1

    def test_knowledge_graph_property(self, mock_ums):
        """Should expose knowledge graph."""
        kg = mock_ums.knowledge_graph

        assert "routing_patterns" in kg


class TestCleanup:
    """Test cleanup functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with cleanup mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        working = MagicMock()
        working.cleanup = AsyncMock()
        ums.collections = {"working": working}

        # Mock KG service
        ums._kg_service = MagicMock()
        ums._kg_service._save_kg_sync = MagicMock()

        return ums

    @pytest.mark.asyncio
    async def test_cleanup_saves_kg(self, mock_ums):
        """Should save KG on cleanup."""
        await mock_ums.cleanup()

        mock_ums._kg_service._save_kg_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_closes_collections(self, mock_ums):
        """Should cleanup collections."""
        await mock_ums.cleanup()

        mock_ums.collections["working"].cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
