"""
Unit Tests for RoutingService

v0.3.1: KG routing removed. Tests query preprocessing and route_query().
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest

from modules.memory.routing_service import RoutingService, ALL_COLLECTIONS


class TestQueryPreprocessing:
    """Test query preprocessing and acronym expansion."""

    @pytest.fixture
    def service(self):
        """Create RoutingService instance."""
        return RoutingService()

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


class TestQueryRouting:
    """Test query routing."""

    @pytest.fixture
    def service(self):
        return RoutingService()

    def test_route_query_returns_all_collections(self, service):
        """v0.3.1: route_query always returns all collections."""
        result = service.route_query("test query")
        assert result == ALL_COLLECTIONS

    def test_route_query_returns_copy(self, service):
        """Should return a copy, not the original list."""
        result = service.route_query("test")
        result.append("extra")
        assert "extra" not in ALL_COLLECTIONS

    def test_all_collections_has_five(self):
        """ALL_COLLECTIONS should have 5 collections."""
        assert len(ALL_COLLECTIONS) == 5
        assert "working" in ALL_COLLECTIONS
        assert "patterns" in ALL_COLLECTIONS
        assert "history" in ALL_COLLECTIONS
        assert "books" in ALL_COLLECTIONS
        assert "memory_bank" in ALL_COLLECTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
