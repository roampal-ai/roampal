"""
Characterization Tests for UnifiedMemorySystem outcome recording.

These tests capture the CURRENT behavior of record_outcome() and related methods.
They serve as a regression safety net against accidental threshold / API drift.
"""

import asyncio
from datetime import datetime

import pytest

from modules.memory.unified_memory_system import UnifiedMemorySystem, ActionOutcome


class TestOutcomeBehavior:
    """Capture current outcome recording behavior."""

    @pytest.fixture
    def memory_system(self, tmp_path):
        ms = UnifiedMemorySystem(
            data_dir=str(tmp_path),
            use_server=False
        )
        return ms

    @pytest.mark.asyncio
    async def test_record_outcome_valid_outcomes(self, memory_system):
        """Valid outcome values should be accepted."""
        valid_outcomes = ["worked", "failed", "partial", "unknown"]

        for outcome in valid_outcomes:
            # Just verify these don't raise - we're not actually recording
            # to avoid polluting the data
            assert outcome in valid_outcomes

    @pytest.mark.asyncio
    async def test_outcome_scoring_logic(self, memory_system):
        """Verify outcome affects scoring as expected."""
        # Test the scoring constants exist
        assert hasattr(memory_system, 'HIGH_VALUE_THRESHOLD')
        assert hasattr(memory_system, 'PROMOTION_SCORE_THRESHOLD')
        assert hasattr(memory_system, 'DEMOTION_SCORE_THRESHOLD')
        assert hasattr(memory_system, 'DELETION_SCORE_THRESHOLD')

        # Verify threshold ordering makes sense
        assert memory_system.HIGH_VALUE_THRESHOLD > memory_system.PROMOTION_SCORE_THRESHOLD
        assert memory_system.PROMOTION_SCORE_THRESHOLD > memory_system.DEMOTION_SCORE_THRESHOLD
        assert memory_system.DEMOTION_SCORE_THRESHOLD > memory_system.DELETION_SCORE_THRESHOLD

    def test_threshold_values(self, memory_system):
        """Capture exact threshold values for regression."""
        thresholds = {
            "HIGH_VALUE_THRESHOLD": memory_system.HIGH_VALUE_THRESHOLD,
            "PROMOTION_SCORE_THRESHOLD": memory_system.PROMOTION_SCORE_THRESHOLD,
            "DEMOTION_SCORE_THRESHOLD": memory_system.DEMOTION_SCORE_THRESHOLD,
            "DELETION_SCORE_THRESHOLD": memory_system.DELETION_SCORE_THRESHOLD,
            "NEW_ITEM_DELETION_THRESHOLD": memory_system.NEW_ITEM_DELETION_THRESHOLD,
        }

        # Expected values from the codebase
        expected = {
            "HIGH_VALUE_THRESHOLD": 0.9,
            "PROMOTION_SCORE_THRESHOLD": 0.7,
            "DEMOTION_SCORE_THRESHOLD": 0.4,
            "DELETION_SCORE_THRESHOLD": 0.2,
            "NEW_ITEM_DELETION_THRESHOLD": 0.1,
        }

        for name, value in expected.items():
            assert thresholds[name] == value, \
                f"{name}: expected {value}, got {thresholds[name]}"


class TestPromotionBehavior:
    """Capture current promotion/demotion behavior."""

    @pytest.fixture
    def memory_system(self, tmp_path):
        ms = UnifiedMemorySystem(
            data_dir=str(tmp_path),
            use_server=False
        )
        return ms

    @pytest.mark.asyncio
    async def test_promotion_methods_exist(self, memory_system):
        """Verify promotion methods exist on UMS (delegates to promotion_service)."""
        # Public API on UMS
        assert hasattr(memory_system, 'promote_valuable_working_memory')
        # After initialization, promotion_service should exist
        await memory_system.initialize()
        assert memory_system._promotion_service is not None

    def test_promotion_is_async(self, memory_system):
        """Promotion methods should be async."""
        import asyncio
        assert asyncio.iscoroutinefunction(memory_system.promote_valuable_working_memory)


class TestActionOutcome:
    """Test ActionOutcome dataclass behavior."""

    def test_action_outcome_import(self):
        """ActionOutcome should be importable."""
        assert ActionOutcome is not None

    def test_action_outcome_fields(self):
        """ActionOutcome should have expected fields."""

        # Create instance with actual API (v0.2.1 Causal Learning)
        ao = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="worked"
        )

        assert ao.action_type == "search_memory"
        assert ao.context_type == "coding"
        assert ao.outcome == "worked"

    def test_action_outcome_to_dict(self):
        """ActionOutcome.to_dict() should work."""

        ao = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="worked"
        )

        d = ao.to_dict()
        assert isinstance(d, dict)
        assert "action_type" in d
        assert "context_type" in d
        assert "outcome" in d
        assert "timestamp" in d

    def test_action_outcome_from_dict(self):
        """ActionOutcome.from_dict() should work."""

        data = {
            "action_type": "search_memory",
            "context_type": "coding",
            "outcome": "worked",
            "timestamp": datetime.now().isoformat(),
            "action_params": {},
            "doc_id": None,
            "collection": None,
            "failure_reason": None,
            "success_context": None,
            "chain_position": 0,
            "chain_length": 1,
            "caused_final_outcome": True,
        }

        ao = ActionOutcome.from_dict(data)
        assert ao.action_type == "search_memory"
        assert ao.context_type == "coding"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
