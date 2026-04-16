"""
Unit Tests for OutcomeService

Tests the extracted outcome recording logic.
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from modules.memory.outcome_service import OutcomeService
from modules.memory.config import MemoryConfig


class TestOutcomeServiceInit:
    """Test OutcomeService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        service = OutcomeService(collections={})
        assert service.config.promotion_score_threshold == 0.7

    def test_init_with_custom_config(self):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        service = OutcomeService(collections={}, config=config)
        assert service.config.promotion_score_threshold == 0.8

    def test_init_with_services(self):
        """Should accept promotion service (v0.3.1: KG removed)."""
        promo_mock = MagicMock()
        service = OutcomeService(
            collections={},
            promotion_service=promo_mock
        )
        assert service.promotion_service == promo_mock


class TestRecordOutcome:
    """Test outcome recording."""

    @pytest.fixture
    def mock_collections(self):
        """Create mock collections."""
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test content",
            "metadata": {
                "text": "test content",
                "score": 0.5,
                "uses": 0,
                "outcome_history": "[]"
            }
        })
        working.update_fragment_metadata = MagicMock()
        working.collection = MagicMock()
        working.collection.count = MagicMock(return_value=10)

        return {"working": working, "history": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections):
        """Create OutcomeService instance."""
        return OutcomeService(collections=mock_collections)

    @pytest.mark.asyncio
    async def test_record_worked_outcome(self, service, mock_collections):
        """Should increase score for worked outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked"
        )

        assert result is not None
        assert result["score"] > 0.5  # Score increased
        assert result["uses"] == 1
        assert result["last_outcome"] == "worked"
        mock_collections["working"].update_fragment_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_failed_outcome(self, service, mock_collections):
        """Should decrease score for failed outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="failed",
            failure_reason="Did not work"
        )

        assert result is not None
        assert result["score"] < 0.5  # Score decreased
        assert result["uses"] == 1  # v0.3.0: Uses now incremented for failure (supports Wilson score)
        assert result["last_outcome"] == "failed"

    @pytest.mark.asyncio
    async def test_record_partial_outcome(self, service, mock_collections):
        """Should slightly increase score for partial outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="partial"
        )

        assert result is not None
        assert result["score"] > 0.5  # Score slightly increased
        assert result["uses"] == 1  # Uses incremented for partial
        assert result["last_outcome"] == "partial"

    @pytest.mark.asyncio
    async def test_record_unknown_outcome(self, service, mock_collections):
        """Should handle unknown outcome (surfaced but not used). v0.3.1: -0.05 delta."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="unknown"
        )

        assert result is not None
        assert result["score"] == 0.45  # v0.3.1: -0.05 weak negative (was 0.0)
        assert result["uses"] == 1  # Uses still incremented
        assert result["last_outcome"] == "unknown"

    @pytest.mark.asyncio
    async def test_safeguard_books(self, service):
        """Should not score book chunks."""
        result = await service.record_outcome(
            doc_id="books_test123",
            outcome="worked"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_safeguard_memory_bank(self, service):
        """Should not score memory bank items."""
        result = await service.record_outcome(
            doc_id="memory_bank_test123",
            outcome="worked"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_not_found(self, service, mock_collections):
        """Should return None for non-existent document."""
        mock_collections["working"].get_fragment = MagicMock(return_value=None)

        result = await service.record_outcome(
            doc_id="working_nonexistent",
            outcome="worked"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_outcome_history_tracking(self, service, mock_collections):
        """Should track outcome history."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked"
        )

        history = json.loads(result["outcome_history"])
        assert len(history) == 1
        assert history[0]["outcome"] == "worked"

    @pytest.mark.asyncio
    async def test_failure_reason_tracking(self, service, mock_collections):
        """Should track failure reasons."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="failed",
            failure_reason="Test failure"
        )

        reasons = json.loads(result["failure_reasons"])
        assert len(reasons) == 1
        assert reasons[0]["reason"] == "Test failure"

    @pytest.mark.asyncio
    async def test_success_context_tracking(self, service, mock_collections):
        """Should track success contexts."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked",
            context={"topic": "test"}
        )

        contexts = json.loads(result["success_contexts"])
        assert len(contexts) == 1
        assert contexts[0]["topic"] == "test"


class TestScoreCalculation:
    """Test score calculation logic. v0.3.1: Flat deltas, no time-weight."""

    @pytest.fixture
    def service(self):
        return OutcomeService(collections={})

    def test_worked_increases_score(self, service):
        """Worked should increase score by +0.2."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "worked", 0.5, 0
        )
        assert delta == 0.2
        assert new_score == 0.7
        assert uses == 1
        assert success_delta == 1.0

    def test_failed_decreases_score(self, service):
        """Failed should decrease score by -0.3."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "failed", 0.5, 0
        )
        assert delta == -0.3
        assert new_score == 0.2
        assert uses == 1
        assert success_delta == 0.0

    def test_partial_slightly_increases(self, service):
        """Partial should increase score by +0.05."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "partial", 0.5, 0
        )
        assert delta == 0.05
        assert new_score == 0.55
        assert uses == 1
        assert success_delta == 0.5

    def test_score_capped_at_1(self, service):
        """Score should not exceed 1.0."""
        delta, new_score, uses, _ = service._calculate_score_update(
            "worked", 0.95, 0
        )
        assert new_score == 1.0

    def test_score_capped_at_0(self, service):
        """Score should not go below 0.0."""
        delta, new_score, uses, _ = service._calculate_score_update(
            "failed", 0.1, 0
        )
        assert new_score == 0.0

    def test_unknown_weak_negative(self, service):
        """Unknown should apply -0.05 delta (v0.3.1, matches core)."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "unknown", 0.5, 0
        )
        assert delta == -0.05
        assert new_score == 0.45
        assert uses == 1
        assert success_delta == 0.25

    def test_flat_deltas_no_time_weight(self, service):
        """v0.3.1: Deltas are flat regardless of memory age."""
        delta1, _, _, _ = service._calculate_score_update("worked", 0.5, 0)
        delta2, _, _, _ = service._calculate_score_update("worked", 0.5, 10)
        assert delta1 == delta2 == 0.2


class TestCountSuccesses:
    """Test success counting from history."""

    @pytest.fixture
    def service(self):
        return OutcomeService(collections={})

    def test_empty_returns_zero(self, service):
        """Empty history returns 0."""
        assert service.count_successes_from_history("") == 0
        assert service.count_successes_from_history("[]") == 0

    def test_worked_counts_as_one(self, service):
        """Worked outcomes count as 1."""
        history = json.dumps([{"outcome": "worked"}])
        assert service.count_successes_from_history(history) == 1.0

    def test_partial_counts_as_half(self, service):
        """Partial outcomes count as 0.5."""
        history = json.dumps([{"outcome": "partial"}])
        assert service.count_successes_from_history(history) == 0.5

    def test_failed_counts_as_zero(self, service):
        """Failed outcomes count as 0."""
        history = json.dumps([{"outcome": "failed"}])
        assert service.count_successes_from_history(history) == 0

    def test_unknown_counts_as_quarter(self, service):
        """Unknown outcomes count as 0.25 (weak negative)."""
        history = json.dumps([{"outcome": "unknown"}])
        assert service.count_successes_from_history(history) == 0.25

    def test_mixed_outcomes(self, service):
        """Mixed outcomes sum correctly."""
        history = json.dumps([
            {"outcome": "worked"},
            {"outcome": "partial"},
            {"outcome": "failed"},
            {"outcome": "worked"}
        ])
        assert service.count_successes_from_history(history) == 2.5

    def test_invalid_json_returns_zero(self, service):
        """Invalid JSON returns 0."""
        assert service.count_successes_from_history("invalid") == 0


class TestOutcomeStats:
    """Test outcome statistics retrieval."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {
                "score": 0.8,
                "uses": 5,
                "last_outcome": "worked",
                "outcome_history": json.dumps([
                    {"outcome": "worked"},
                    {"outcome": "worked"},
                    {"outcome": "partial"},
                    {"outcome": "failed"},
                    {"outcome": "worked"}
                ])
            }
        })
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections):
        return OutcomeService(collections=mock_collections)

    def test_get_outcome_stats(self, service):
        """Should return correct outcome stats."""
        stats = service.get_outcome_stats("working_test123")

        assert stats["doc_id"] == "working_test123"
        assert stats["collection"] == "working"
        assert stats["score"] == 0.8
        assert stats["uses"] == 5
        assert stats["last_outcome"] == "worked"
        assert stats["outcomes"]["worked"] == 3
        assert stats["outcomes"]["partial"] == 1
        assert stats["outcomes"]["failed"] == 1
        assert stats["total_outcomes"] == 5

    def test_get_stats_not_found(self, service, mock_collections):
        """Should return error for non-existent document."""
        mock_collections["working"].get_fragment = MagicMock(return_value=None)

        stats = service.get_outcome_stats("working_nonexistent")
        assert stats["error"] == "not_found"


class TestPromotionIntegration:
    """Test promotion service integration."""

    @pytest.fixture
    def mock_promotion_service(self):
        promo = MagicMock()
        promo.handle_promotion = AsyncMock()
        return promo

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"score": 0.5, "uses": 0, "outcome_history": "[]"}
        })
        working.update_fragment_metadata = MagicMock()
        working.collection = MagicMock()
        working.collection.count = MagicMock(return_value=10)
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections, mock_promotion_service):
        return OutcomeService(
            collections=mock_collections,
            promotion_service=mock_promotion_service
        )

    @pytest.mark.asyncio
    async def test_calls_promotion_handler(self, service, mock_promotion_service):
        """Should call promotion handler after outcome."""
        await service.record_outcome("working_test123", "worked")

        mock_promotion_service.handle_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_params_to_promotion(self, service, mock_promotion_service):
        """Should pass correct parameters to promotion handler."""
        await service.record_outcome("working_test123", "worked")

        call_args = mock_promotion_service.handle_promotion.call_args
        assert call_args[1]["doc_id"] == "working_test123"
        assert call_args[1]["collection"] == "working"
        assert "score" in call_args[1]
        assert "uses" in call_args[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
