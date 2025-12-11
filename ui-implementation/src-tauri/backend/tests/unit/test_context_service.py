"""
Unit Tests for ContextService

Tests the extracted context analysis logic.
"""

import sys
sys.path.insert(0, "C:/ROAMPAL-REFACTOR")

import pytest
from unittest.mock import MagicMock, AsyncMock

from modules.memory.context_service import ContextService
from modules.memory.config import MemoryConfig


class TestContextServiceInit:
    """Test ContextService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        service = ContextService(collections={})
        assert service.config is not None
        assert service.kg_service is None

    def test_init_with_services(self):
        """Should accept KG service and embed function."""
        kg_mock = MagicMock()
        embed_mock = AsyncMock()
        service = ContextService(
            collections={},
            kg_service=kg_mock,
            embed_fn=embed_mock
        )
        assert service.kg_service == kg_mock
        assert service.embed_fn == embed_mock


class TestConceptExtraction:
    """Test concept extraction."""

    @pytest.fixture
    def service(self):
        return ContextService(collections={})

    def test_basic_extraction(self, service):
        """Should extract meaningful words."""
        concepts = service._basic_concept_extraction(
            "How do I configure Python logging?"
        )
        assert "configure" in concepts
        assert "python" in concepts
        assert "logging" in concepts

    def test_filters_stopwords(self, service):
        """Should filter common stopwords."""
        concepts = service._basic_concept_extraction(
            "The quick brown fox is a test"
        )
        assert "the" not in concepts
        assert "quick" in concepts
        assert "brown" in concepts

    def test_filters_short_words(self, service):
        """Should filter words shorter than 3 chars."""
        concepts = service._basic_concept_extraction(
            "I am a developer"
        )
        assert "developer" in concepts
        # Short words filtered
        assert len([c for c in concepts if len(c) < 3]) == 0

    def test_empty_text(self, service):
        """Should handle empty text."""
        concepts = service._basic_concept_extraction("")
        assert concepts == []

    def test_uses_kg_service_if_available(self):
        """Should use KG service for extraction when available."""
        kg_mock = MagicMock()
        kg_mock.extract_concepts = MagicMock(return_value=["test", "concepts"])

        service = ContextService(collections={}, kg_service=kg_mock)
        concepts = service._extract_concepts("test input")

        kg_mock.extract_concepts.assert_called_once_with("test input")
        assert concepts == ["test", "concepts"]


class TestPatternRecognition:
    """Test pattern recognition from past conversations."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.extract_concepts = MagicMock(return_value=["python", "logging", "config"])
        kg.get_problem_categories = MagicMock(return_value={
            "config_logging_python": ["patterns_doc123"]
        })
        return kg

    @pytest.fixture
    def mock_collections(self):
        patterns = MagicMock()
        patterns.get_fragment = MagicMock(return_value={
            "content": "Use logging.basicConfig() for simple setup",
            "metadata": {
                "score": 0.85,
                "uses": 5,
                "last_outcome": "worked"
            }
        })
        return {"patterns": patterns, "history": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections, mock_kg_service):
        return ContextService(
            collections=mock_collections,
            kg_service=mock_kg_service
        )

    @pytest.mark.asyncio
    async def test_finds_relevant_patterns(self, service):
        """Should find patterns from past conversations."""
        patterns = await service._find_relevant_patterns(
            ["python", "logging", "config"]
        )

        assert len(patterns) == 1
        assert patterns[0]["score"] == 0.85
        assert patterns[0]["uses"] == 5
        assert "success rate" in patterns[0]["insight"]

    @pytest.mark.asyncio
    async def test_filters_low_score_patterns(self, service, mock_collections):
        """Should filter patterns below threshold."""
        mock_collections["patterns"].get_fragment = MagicMock(return_value={
            "content": "Low score pattern",
            "metadata": {"score": 0.5, "uses": 1, "last_outcome": "partial"}
        })

        patterns = await service._find_relevant_patterns(
            ["python", "logging", "config"]
        )

        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_no_kg_service(self):
        """Should return empty list without KG service."""
        service = ContextService(collections={})
        patterns = await service._find_relevant_patterns(["test"])
        assert patterns == []


class TestFailureAwareness:
    """Test failure pattern detection."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.get_failure_patterns = MagicMock(return_value={
            "asyncio deadlock": [
                {"timestamp": "2024-01-15T10:00:00", "doc_id": "doc1"},
                {"timestamp": "2024-01-16T11:00:00", "doc_id": "doc2"}
            ]
        })
        return kg

    @pytest.fixture
    def service(self, mock_kg_service):
        return ContextService(collections={}, kg_service=mock_kg_service)

    def test_detects_failure_patterns(self, service):
        """Should detect related failure patterns."""
        failures = service._check_failure_patterns(["asyncio", "threading"])

        assert len(failures) >= 1
        assert failures[0]["outcome"] == "failed"
        assert "deadlock" in failures[0]["reason"]

    def test_no_matching_failures(self, service):
        """Should return empty for unrelated concepts."""
        failures = service._check_failure_patterns(["unrelated", "concepts"])
        assert len(failures) == 0


class TestTopicContinuity:
    """Test topic continuity detection."""

    @pytest.fixture
    def service(self):
        kg = MagicMock()
        kg.extract_concepts = MagicMock(side_effect=lambda x: x.lower().split()[:5])
        return ContextService(collections={}, kg_service=kg)

    def test_detects_continuation(self, service):
        """Should detect topic continuation."""
        recent = [
            {"role": "user", "content": "setup logging debug level"},
            {"role": "assistant", "content": "You can use..."},
        ]

        continuity = service._detect_topic_continuity(
            ["logging", "debug", "level"],
            recent
        )

        assert len(continuity) == 1
        assert continuity[0]["continuing"] is True
        assert "logging" in continuity[0]["common_concepts"]

    def test_detects_topic_shift(self, service):
        """Should detect topic shift."""
        recent = [
            {"role": "user", "content": "How do I configure logging?"},
            {"role": "assistant", "content": "You can use..."},
        ]

        continuity = service._detect_topic_continuity(
            ["database", "sql", "connection"],
            recent
        )

        assert len(continuity) == 1
        assert continuity[0]["continuing"] is False

    def test_empty_conversation(self, service):
        """Should handle empty conversation."""
        continuity = service._detect_topic_continuity(["test"], [])
        assert continuity == []


class TestProactiveInsights:
    """Test proactive insights generation."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.get_routing_patterns = MagicMock(return_value={
            "python": {
                "success_rate": 0.85,
                "best_collection": "patterns"
            },
            "config": {
                "success_rate": 0.6,  # Below threshold
                "best_collection": "history"
            }
        })
        return kg

    @pytest.fixture
    def service(self, mock_kg_service):
        return ContextService(collections={}, kg_service=mock_kg_service)

    def test_generates_insights_for_high_success(self, service):
        """Should generate insights for high-success patterns."""
        insights = service._get_proactive_insights(["python", "config"])

        # Only python should have insight (config below 0.7)
        assert len(insights) == 1
        assert insights[0]["concept"] == "python"
        assert "patterns" in insights[0]["recommendation"]

    def test_no_insights_below_threshold(self, service):
        """Should not generate insights for low-success patterns."""
        insights = service._get_proactive_insights(["config"])
        assert len(insights) == 0


class TestRepetitionDetection:
    """Test repetition detection."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.query_vectors = AsyncMock(return_value=[
            {
                "content": "How do I configure logging?",
                "metadata": {"conversation_id": "conv123"},
                "distance": 0.1  # Very similar
            }
        ])
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections):
        return ContextService(
            collections=mock_collections,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_detects_repetition(self, service):
        """Should detect similar recent messages."""
        repetitions = await service._detect_repetition(
            "How can I setup logging?",
            "conv123"
        )

        assert len(repetitions) == 1
        assert repetitions[0]["similarity"] > 0.85
        assert "similar" in repetitions[0]["insight"]

    @pytest.mark.asyncio
    async def test_ignores_different_conversation(self, service, mock_collections):
        """Should ignore items from different conversation."""
        mock_collections["working"].query_vectors = AsyncMock(return_value=[
            {
                "content": "Similar message",
                "metadata": {"conversation_id": "other_conv"},
                "distance": 0.1
            }
        ])

        repetitions = await service._detect_repetition(
            "Similar message",
            "conv123"
        )

        assert len(repetitions) == 0


class TestAnalyzeConversationContext:
    """Test full context analysis."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.extract_concepts = MagicMock(return_value=["test", "concept"])
        kg.get_problem_categories = MagicMock(return_value={})
        kg.get_failure_patterns = MagicMock(return_value={})
        kg.get_routing_patterns = MagicMock(return_value={})
        return kg

    @pytest.fixture
    def service(self, mock_kg_service):
        return ContextService(
            collections={},
            kg_service=mock_kg_service
        )

    @pytest.mark.asyncio
    async def test_returns_context_structure(self, service):
        """Should return proper context structure."""
        context = await service.analyze_conversation_context(
            "test message",
            [],
            "conv123"
        )

        assert "relevant_patterns" in context
        assert "past_outcomes" in context
        assert "topic_continuity" in context
        assert "proactive_insights" in context


class TestFindKnownSolutions:
    """Test known solution finding."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.extract_concepts = MagicMock(return_value=["python", "logging"])
        kg.get_problem_solutions = MagicMock(return_value={
            "logging_python": [
                {"doc_id": "patterns_123", "success_count": 5, "last_used": "2024-01-15"}
            ]
        })
        return kg

    @pytest.fixture
    def mock_collections(self):
        patterns = MagicMock()
        patterns.get_fragment = MagicMock(return_value={
            "content": "Use logging.basicConfig()",
            "metadata": {"score": 0.9}
        })
        return {"patterns": patterns}

    @pytest.fixture
    def service(self, mock_collections, mock_kg_service):
        return ContextService(
            collections=mock_collections,
            kg_service=mock_kg_service
        )

    @pytest.mark.asyncio
    async def test_finds_exact_match(self, service):
        """Should find exact problem match."""
        solutions = await service.find_known_solutions(
            "How do I setup python logging?"
        )

        assert len(solutions) == 1
        assert solutions[0]["is_known_solution"] is True
        assert solutions[0]["solution_success_count"] == 5

    @pytest.mark.asyncio
    async def test_boosts_known_solutions(self, service):
        """Should boost distance for known solutions."""
        solutions = await service.find_known_solutions(
            "How do I setup python logging?"
        )

        # Distance should be boosted (reduced by 50%)
        assert solutions[0]["distance"] == 0.5


class TestContextSummary:
    """Test context summary generation."""

    @pytest.fixture
    def service(self):
        return ContextService(collections={})

    def test_summary_with_patterns(self, service):
        """Should mention patterns in summary."""
        context = {
            "relevant_patterns": [{"text": "pattern1"}],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)
        assert "1 relevant pattern" in summary

    def test_summary_with_failures(self, service):
        """Should mention failures in summary."""
        context = {
            "relevant_patterns": [],
            "past_outcomes": [{"outcome": "failed"}],
            "topic_continuity": [],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)
        assert "Warning" in summary
        assert "failed" in summary

    def test_summary_with_continuity(self, service):
        """Should mention topic continuity."""
        context = {
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [{"continuing": True, "common_concepts": ["python"]}],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)
        assert "Continuing" in summary

    def test_summary_empty_context(self, service):
        """Should handle empty context."""
        context = {
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)
        assert "No significant context" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
