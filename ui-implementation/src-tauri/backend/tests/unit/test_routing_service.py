"""
Unit Tests for RoutingService

Tests the extracted routing logic.
"""

import sys
sys.path.insert(0, "C:/ROAMPAL-REFACTOR")

import pytest
from unittest.mock import MagicMock, patch

from modules.memory.routing_service import RoutingService, ALL_COLLECTIONS
from modules.memory.knowledge_graph_service import KnowledgeGraphService
from modules.memory.config import MemoryConfig


class TestQueryPreprocessing:
    """Test query preprocessing and acronym expansion."""

    @pytest.fixture
    def mock_kg_service(self):
        """Create mock KG service."""
        mock = MagicMock(spec=KnowledgeGraphService)
        mock.knowledge_graph = {}
        mock.extract_concepts = MagicMock(return_value=[])
        return mock

    @pytest.fixture
    def service(self, mock_kg_service):
        """Create RoutingService instance."""
        return RoutingService(kg_service=mock_kg_service)

    def test_expand_single_acronym(self, service):
        """Should expand single acronym."""
        result = service.preprocess_query("What is API?")
        assert "application programming interface" in result.lower()
        assert "api" in result.lower()  # Original kept

    def test_expand_multiple_acronyms(self, service):
        """Should expand multiple acronyms."""
        result = service.preprocess_query("Use the SDK for ML")
        assert "software development kit" in result.lower()
        assert "machine learning" in result.lower()

    def test_no_expansion_for_unknown(self, service):
        """Should not expand unknown words."""
        result = service.preprocess_query("Hello world")
        assert result == "Hello world"

    def test_normalize_whitespace(self, service):
        """Should normalize whitespace."""
        result = service.preprocess_query("  multiple   spaces  here  ")
        assert "  " not in result

    def test_empty_query(self, service):
        """Empty query should return empty."""
        result = service.preprocess_query("")
        assert result == ""

    def test_case_insensitive_expansion(self, service):
        """Acronym expansion should be case insensitive."""
        result_lower = service.preprocess_query("api")
        result_upper = service.preprocess_query("API")
        assert "application programming interface" in result_lower.lower()
        assert "application programming interface" in result_upper.lower()

    def test_punctuation_handling(self, service):
        """Should handle acronyms with punctuation."""
        result = service.preprocess_query("What's the API?")
        assert "application programming interface" in result.lower()


class TestTierScoreCalculation:
    """Test tier score calculation for collections."""

    @pytest.fixture
    def mock_kg_service(self):
        """Create mock KG service with patterns."""
        mock = MagicMock(spec=KnowledgeGraphService)
        mock.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "history": {"successes": 8, "failures": 2, "partials": 0, "total": 10},
                        "books": {"successes": 3, "failures": 1, "partials": 0, "total": 4},
                    }
                },
                "django": {
                    "collections_used": {
                        "history": {"successes": 5, "failures": 0, "partials": 0, "total": 5},
                    }
                }
            }
        }
        mock.extract_concepts = MagicMock(return_value=["python", "django"])
        return mock

    @pytest.fixture
    def service(self, mock_kg_service):
        return RoutingService(kg_service=mock_kg_service)

    def test_tier_score_calculation(self, service):
        """Should calculate tier scores correctly."""
        scores = service.calculate_tier_scores(["python"])

        # Python in history: success_rate = 8/10 = 0.8, confidence = 10/10 = 1.0
        # tier_score = 0.8 * 1.0 = 0.8
        assert scores["history"] == pytest.approx(0.8, rel=0.01)

        # Python in books: success_rate = 3/4 = 0.75, confidence = 4/10 = 0.4
        # tier_score = 0.75 * 0.4 = 0.3
        assert scores["books"] == pytest.approx(0.3, rel=0.01)

    def test_tier_score_aggregation(self, service):
        """Should aggregate scores across multiple concepts."""
        scores = service.calculate_tier_scores(["python", "django"])

        # history should have python score + django score
        # python: 0.8, django: success_rate=5/5=1.0, confidence=5/10=0.5, tier=0.5
        # total for history: 0.8 + 0.5 = 1.3
        assert scores["history"] == pytest.approx(1.3, rel=0.01)

    def test_tier_score_empty_concepts(self, service):
        """Empty concepts should return zero scores."""
        scores = service.calculate_tier_scores([])
        assert all(score == 0.0 for score in scores.values())

    def test_tier_score_unknown_concept(self, service):
        """Unknown concept should not affect scores."""
        scores = service.calculate_tier_scores(["unknown_concept"])
        assert all(score == 0.0 for score in scores.values())

    def test_neutral_score_no_feedback(self, mock_kg_service):
        """Concepts with no success/failure feedback should get neutral rate."""
        mock_kg_service.knowledge_graph["routing_patterns"]["new_concept"] = {
            "collections_used": {
                "working": {"successes": 0, "failures": 0, "partials": 5, "total": 5}
            }
        }
        service = RoutingService(kg_service=mock_kg_service)
        scores = service.calculate_tier_scores(["new_concept"])

        # success_rate = 0.5 (neutral), confidence = 0.5
        # tier_score = 0.5 * 0.5 = 0.25
        assert scores["working"] == pytest.approx(0.25, rel=0.01)


class TestQueryRouting:
    """Test intelligent query routing."""

    @pytest.fixture
    def mock_kg_service(self):
        mock = MagicMock(spec=KnowledgeGraphService)
        mock.knowledge_graph = {"routing_patterns": {}}
        return mock

    def test_exploration_phase_no_patterns(self, mock_kg_service):
        """No patterns should trigger exploration phase (all collections)."""
        mock_kg_service.extract_concepts = MagicMock(return_value=["test"])
        service = RoutingService(kg_service=mock_kg_service)

        result = service.route_query("test query")
        assert result == ALL_COLLECTIONS

    def test_exploration_phase_low_score(self, mock_kg_service):
        """Low total score should trigger exploration phase."""
        mock_kg_service.knowledge_graph = {
            "routing_patterns": {
                "test": {
                    "collections_used": {
                        "history": {"successes": 1, "failures": 0, "partials": 0, "total": 1}
                    }
                }
            }
        }
        mock_kg_service.extract_concepts = MagicMock(return_value=["test"])
        service = RoutingService(kg_service=mock_kg_service)

        result = service.route_query("test query")
        # Score = 1.0 * 0.1 = 0.1 < 0.5 (exploration threshold)
        assert result == ALL_COLLECTIONS

    def test_medium_confidence_phase(self, mock_kg_service):
        """Medium score should select top 2-3 collections."""
        mock_kg_service.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "history": {"successes": 8, "failures": 2, "partials": 0, "total": 10},
                        "books": {"successes": 3, "failures": 1, "partials": 0, "total": 4},
                    }
                }
            }
        }
        mock_kg_service.extract_concepts = MagicMock(return_value=["python"])
        service = RoutingService(kg_service=mock_kg_service)

        result = service.route_query("python tutorial")
        # Total score = 0.8 + 0.3 = 1.1 (medium confidence)
        assert "history" in result
        assert len(result) <= 3

    def test_high_confidence_phase(self, mock_kg_service):
        """High score should select top 1-2 collections."""
        mock_kg_service.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "history": {"successes": 10, "failures": 0, "partials": 0, "total": 20},
                    }
                },
                "django": {
                    "collections_used": {
                        "history": {"successes": 10, "failures": 0, "partials": 0, "total": 20},
                    }
                },
                "web": {
                    "collections_used": {
                        "history": {"successes": 10, "failures": 0, "partials": 0, "total": 20},
                    }
                }
            }
        }
        mock_kg_service.extract_concepts = MagicMock(return_value=["python", "django", "web"])
        service = RoutingService(kg_service=mock_kg_service)

        result = service.route_query("python django web")
        # Total score = 1.0 * 3 = 3.0 (high confidence)
        assert len(result) <= 2
        assert "history" in result

    def test_no_concepts_searches_all(self, mock_kg_service):
        """Empty concepts should search all collections."""
        mock_kg_service.extract_concepts = MagicMock(return_value=[])
        service = RoutingService(kg_service=mock_kg_service)

        result = service.route_query("test")
        assert result == ALL_COLLECTIONS

    def test_routing_tracks_usage(self, mock_kg_service):
        """Routing should track usage in KG."""
        mock_kg_service.knowledge_graph = {
            "routing_patterns": {
                "test": {
                    "collections_used": {
                        "history": {"successes": 5, "failures": 0, "partials": 0, "total": 10}
                    }
                }
            }
        }
        mock_kg_service.extract_concepts = MagicMock(return_value=["test"])
        service = RoutingService(kg_service=mock_kg_service)

        service.route_query("test query")

        # Should have incremented total for used collections
        pattern = mock_kg_service.knowledge_graph["routing_patterns"]["test"]
        assert "last_used" in pattern


class TestTierRecommendations:
    """Test tier recommendations for insights."""

    @pytest.fixture
    def mock_kg_service(self):
        mock = MagicMock(spec=KnowledgeGraphService)
        mock.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "history": {"successes": 8, "failures": 2, "partials": 0, "total": 10}
                    }
                }
            }
        }
        return mock

    @pytest.fixture
    def service(self, mock_kg_service):
        return RoutingService(kg_service=mock_kg_service)

    def test_recommendations_empty_concepts(self, service):
        """Empty concepts should return exploration recommendations."""
        result = service.get_tier_recommendations([])
        assert result["confidence_level"] == "exploration"
        assert result["match_count"] == 0
        assert result["top_collections"] == ALL_COLLECTIONS

    def test_recommendations_with_matches(self, service):
        """Should count matched patterns."""
        result = service.get_tier_recommendations(["python"])
        assert result["match_count"] == 1
        assert result["total_score"] > 0

    def test_recommendations_confidence_levels(self, mock_kg_service):
        """Should set correct confidence level."""
        # High confidence setup
        mock_kg_service.knowledge_graph = {
            "routing_patterns": {
                f"concept_{i}": {
                    "collections_used": {
                        "history": {"successes": 10, "failures": 0, "partials": 0, "total": 20}
                    }
                }
                for i in range(5)
            }
        }
        service = RoutingService(kg_service=mock_kg_service)

        concepts = [f"concept_{i}" for i in range(5)]
        result = service.get_tier_recommendations(concepts)

        # 5 concepts * 1.0 each = 5.0 total score (high confidence)
        assert result["confidence_level"] == "high"

    def test_recommendations_returns_scores(self, service):
        """Should return top 3 collection scores."""
        result = service.get_tier_recommendations(["python"])
        assert "scores" in result
        assert len(result["scores"]) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
