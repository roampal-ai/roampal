"""
Unit Tests for ContextService

Tests the extracted context analysis logic.
KG removed in v0.3.1 — KG-dependent methods return empty results.
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

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

    def test_init_with_embed_fn(self):
        """Should accept embed function."""
        embed_mock = AsyncMock()
        service = ContextService(
            collections={},
            embed_fn=embed_mock
        )
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
        assert len([c for c in concepts if len(c) < 3]) == 0

    def test_empty_text(self, service):
        """Should handle empty text."""
        concepts = service._basic_concept_extraction("")
        assert concepts == []

    def test_extract_concepts_uses_basic_extraction(self):
        """Should use basic extraction (KG removed)."""
        service = ContextService(collections={})
        concepts = service._extract_concepts("test input here")
        assert isinstance(concepts, list)
        assert len(concepts) > 0


class TestPatternRecognition:
    """Test pattern recognition — returns empty without KG (removed v0.3.1)."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_kg(self):
        """Should return empty list (KG removed)."""
        service = ContextService(collections={})
        patterns = await service._find_relevant_patterns(["test"])
        assert patterns == []


class TestFailureAwareness:
    """Test failure pattern detection — returns empty without KG (removed v0.3.1)."""

    def test_returns_empty_without_kg(self):
        """Should return empty (KG removed)."""
        service = ContextService(collections={})
        failures = service._check_failure_patterns(["asyncio", "threading"])
        assert failures == []


class TestTopicContinuity:
    """Test topic continuity detection."""

    @pytest.fixture
    def service(self):
        return ContextService(collections={})

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
    """Test proactive insights — returns empty without KG (removed v0.3.1)."""

    def test_returns_empty_without_kg(self):
        """Should return empty (KG removed)."""
        service = ContextService(collections={})
        insights = service._get_proactive_insights(["python", "config"])
        assert insights == []


class TestRepetitionDetection:
    """Test repetition detection."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.query_vectors = AsyncMock(return_value=[
            {
                "content": "How do I configure logging?",
                "metadata": {"conversation_id": "conv123"},
                "distance": 0.1
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
    def service(self):
        return ContextService(collections={})

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
    """Test known solution finding — returns empty without KG (removed v0.3.1)."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_kg(self):
        """Should return empty (KG removed)."""
        service = ContextService(collections={})
        solutions = await service.find_known_solutions("How do I setup python logging?")
        assert solutions == []


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
