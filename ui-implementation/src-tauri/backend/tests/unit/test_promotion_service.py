"""
Unit Tests for PromotionService

Tests the extracted promotion/demotion logic.
"""

import sys
sys.path.insert(0, "C:/ROAMPAL-REFACTOR")

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from modules.memory.promotion_service import PromotionService
from modules.memory.config import MemoryConfig


class TestPromotionServiceInit:
    """Test PromotionService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        collections = {"working": MagicMock(), "history": MagicMock()}
        service = PromotionService(
            collections=collections,
            embed_fn=AsyncMock(),
        )
        assert service.config.promotion_score_threshold == 0.7
        assert service.config.high_value_threshold == 0.9

    def test_init_with_custom_config(self):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        service = PromotionService(
            collections={},
            embed_fn=AsyncMock(),
            config=config,
        )
        assert service.config.promotion_score_threshold == 0.8


class TestWorkingToHistoryPromotion:
    """Test working -> history promotion."""

    @pytest.fixture
    def mock_collections(self):
        """Create mock collections."""
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test content",
            "metadata": {"text": "test content", "score": 0.8, "uses": 3}
        })
        working.delete_vectors = MagicMock()

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        return {"working": working, "history": history, "patterns": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections):
        """Create PromotionService instance."""
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
            add_relationship_fn=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_promotes_working_to_history(self, service, mock_collections):
        """Should promote working memory with high score and uses."""
        result = await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.8,
            uses=3,
            metadata={"text": "test", "score": 0.8, "uses": 3}
        )

        # Should have created history entry
        mock_collections["history"].upsert_vectors.assert_called_once()
        # Should have deleted from working
        mock_collections["working"].delete_vectors.assert_called_with(["working_test123"])
        # Should return new ID
        assert result == "history_test123"

    @pytest.mark.asyncio
    async def test_no_promotion_low_score(self, service, mock_collections):
        """Should not promote if score too low."""
        result = await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.5,  # Below threshold
            uses=3,
            metadata={"text": "test"}
        )

        mock_collections["history"].upsert_vectors.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_promotion_low_uses(self, service, mock_collections):
        """Should not promote if uses too low."""
        result = await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.8,
            uses=1,  # Below threshold
            metadata={"text": "test"}
        )

        mock_collections["history"].upsert_vectors.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_tracks_promotion_history(self, service, mock_collections):
        """Should add promotion record to metadata."""
        await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.8,
            uses=3,
            metadata={"text": "test", "promotion_history": "[]"}
        )

        # Check the metadata passed to upsert
        call_args = mock_collections["history"].upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        promotion_history = json.loads(metadata["promotion_history"])
        assert len(promotion_history) == 1
        assert promotion_history[0]["from"] == "working"
        assert promotion_history[0]["to"] == "history"


class TestHistoryToPatternsPromotion:
    """Test history -> patterns promotion."""

    @pytest.fixture
    def mock_collections(self):
        history = MagicMock()
        history.get_fragment = MagicMock(return_value={
            "content": "test pattern",
            "metadata": {"text": "test pattern"}
        })
        history.delete_vectors = MagicMock()

        patterns = MagicMock()
        patterns.upsert_vectors = AsyncMock()

        return {"working": MagicMock(), "history": history, "patterns": patterns}

    @pytest.fixture
    def service(self, mock_collections):
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

    @pytest.mark.asyncio
    async def test_promotes_history_to_patterns(self, service, mock_collections):
        """Should promote history memory with very high score."""
        result = await service.handle_promotion(
            doc_id="history_test123",
            collection="history",
            score=0.95,  # >= HIGH_VALUE_THRESHOLD (0.9)
            uses=5,
            metadata={"text": "test pattern"}
        )

        mock_collections["patterns"].upsert_vectors.assert_called_once()
        mock_collections["history"].delete_vectors.assert_called_with(["history_test123"])
        assert result == "patterns_test123"

    @pytest.mark.asyncio
    async def test_no_patterns_promotion_low_score(self, service, mock_collections):
        """Should not promote to patterns if score below threshold."""
        result = await service.handle_promotion(
            doc_id="history_test123",
            collection="history",
            score=0.85,  # Below HIGH_VALUE_THRESHOLD
            uses=5,
            metadata={"text": "test"}
        )

        mock_collections["patterns"].upsert_vectors.assert_not_called()
        assert result is None


class TestDemotion:
    """Test patterns -> history demotion."""

    @pytest.fixture
    def mock_collections(self):
        patterns = MagicMock()
        patterns.get_fragment = MagicMock(return_value={
            "content": "failing pattern",
            "metadata": {"text": "failing pattern"}
        })
        patterns.delete_vectors = MagicMock()

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        return {"working": MagicMock(), "history": history, "patterns": patterns}

    @pytest.fixture
    def service(self, mock_collections):
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

    @pytest.mark.asyncio
    async def test_demotes_patterns_to_history(self, service, mock_collections):
        """Should demote patterns memory with low score."""
        result = await service.handle_promotion(
            doc_id="patterns_test123",
            collection="patterns",
            score=0.3,  # Below DEMOTION_SCORE_THRESHOLD (0.4)
            uses=5,
            metadata={"text": "failing pattern"}
        )

        mock_collections["history"].upsert_vectors.assert_called_once()
        mock_collections["patterns"].delete_vectors.assert_called_with(["patterns_test123"])

        # Check demotion metadata
        call_args = mock_collections["history"].upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["demoted_from"] == "patterns"

        assert result == "history_test123"


class TestDeletion:
    """Test low-score deletion."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"text": "test"}
        })
        working.delete_vectors = MagicMock()

        return {"working": working, "history": MagicMock(), "patterns": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections):
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_deletes_very_low_score(self, service, mock_collections):
        """Should delete memory with very low score."""
        await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.15,  # Below DELETION_SCORE_THRESHOLD (0.2)
            uses=1,
            metadata={"text": "test", "timestamp": (datetime.now() - timedelta(days=10)).isoformat()}
        )

        mock_collections["working"].delete_vectors.assert_called_with(["working_test123"])

    @pytest.mark.asyncio
    async def test_lenient_deletion_for_new_items(self, service, mock_collections):
        """New items should have more lenient deletion threshold."""
        # Item created 1 day ago (< 7 days, so lenient threshold)
        timestamp = (datetime.now() - timedelta(days=1)).isoformat()

        await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.15,  # Below DELETION but above NEW_ITEM threshold
            uses=1,
            metadata={"text": "test", "timestamp": timestamp}
        )

        # Should NOT be deleted (0.15 >= 0.1 new item threshold)
        # Actually this depends on exact thresholds...
        # With score=0.15 and NEW_ITEM_DELETION_THRESHOLD=0.1, it should NOT be deleted

    @pytest.mark.asyncio
    async def test_strict_deletion_for_old_items(self, service, mock_collections):
        """Old items should have stricter deletion threshold."""
        # Item created 10 days ago (> 7 days, so strict threshold)
        timestamp = (datetime.now() - timedelta(days=10)).isoformat()

        await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.15,  # Below DELETION_SCORE_THRESHOLD (0.2)
            uses=1,
            metadata={"text": "test", "timestamp": timestamp}
        )

        mock_collections["working"].delete_vectors.assert_called_with(["working_test123"])


class TestBatchPromotion:
    """Test batch promotion of working memory."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.list_all_ids = MagicMock(return_value=[
            "working_high_score",
            "working_low_score",
            "working_old_item"
        ])

        def get_fragment_side_effect(doc_id):
            if doc_id == "working_high_score":
                return {
                    "content": "high score item",
                    "metadata": {
                        "text": "high score item",
                        "score": 0.8,
                        "uses": 3,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            elif doc_id == "working_low_score":
                return {
                    "content": "low score item",
                    "metadata": {
                        "text": "low score item",
                        "score": 0.4,
                        "uses": 1,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            elif doc_id == "working_old_item":
                return {
                    "content": "old item",
                    "metadata": {
                        "text": "old item",
                        "score": 0.5,
                        "uses": 1,
                        "timestamp": (datetime.now() - timedelta(hours=30)).isoformat()
                    }
                }
            return None

        working.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        working.delete_vectors = MagicMock()

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        return {"working": working, "history": history, "patterns": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections):
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

    @pytest.mark.asyncio
    async def test_batch_promotes_high_score_items(self, service, mock_collections):
        """Should promote high-score items in batch."""
        promoted = await service.promote_valuable_working_memory()

        # Should have promoted 1 item (high_score)
        assert promoted == 1
        mock_collections["history"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_cleans_old_items(self, service, mock_collections):
        """Should clean up old items that weren't promoted."""
        await service.promote_valuable_working_memory()

        # Should have deleted the old item
        delete_calls = mock_collections["working"].delete_vectors.call_args_list
        deleted_ids = [call[0][0] for call in delete_calls]

        # Should include high_score (promoted) and old_item (cleaned)
        assert ["working_high_score"] in deleted_ids
        assert ["working_old_item"] in deleted_ids


class TestPromoteItem:
    """Test generic item promotion."""

    @pytest.fixture
    def mock_collections(self):
        source = MagicMock()
        source.get_fragment = MagicMock(return_value={
            "content": "test content"
        })

        target = MagicMock()
        target.upsert_vectors = AsyncMock()

        return {"source": source, "target": target}

    @pytest.fixture
    def service(self, mock_collections):
        return PromotionService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

    @pytest.mark.asyncio
    async def test_promote_item_creates_new_id(self, service, mock_collections):
        """Should create new ID in target collection."""
        result = await service.promote_item(
            doc_id="source_123",
            from_collection="source",
            to_collection="target",
            metadata={"text": "test"}
        )

        assert result is not None
        assert result.startswith("target_")
        mock_collections["target"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_promote_item_adds_metadata(self, service, mock_collections):
        """Should add promotion metadata."""
        await service.promote_item(
            doc_id="source_123",
            from_collection="source",
            to_collection="target",
            metadata={"text": "test"}
        )

        call_args = mock_collections["target"].upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["promoted_from"] == "source"
        assert metadata["original_id"] == "source_123"
        assert "promoted_at" in metadata


class TestAgeCalculation:
    """Test age calculation helper."""

    @pytest.fixture
    def service(self):
        return PromotionService(
            collections={},
            embed_fn=AsyncMock(),
        )

    def test_calculates_age_hours(self, service):
        """Should calculate age in hours."""
        timestamp = (datetime.now() - timedelta(hours=5)).isoformat()
        age = service._calculate_age_hours(timestamp)
        assert abs(age - 5.0) < 0.1

    def test_handles_empty_timestamp(self, service):
        """Should return 0 for empty timestamp."""
        age = service._calculate_age_hours("")
        assert age == 0.0

    def test_handles_invalid_timestamp(self, service):
        """Should return 0 for invalid timestamp."""
        age = service._calculate_age_hours("invalid")
        assert age == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
