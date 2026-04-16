"""
Tests for tag_service.py — v0.3.1 tag extraction, matching, and known tag index.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from modules.memory.tag_service import (
    extract_tags_regex,
    TagService,
    _dedup_substrings,
    _is_adjective_not_place,
    _NOISE_TAGS,
)


# ---------------------------------------------------------------------------
# extract_tags_regex — deterministic extraction
# ---------------------------------------------------------------------------

class TestExtractTagsRegex:
    def test_proper_nouns(self):
        tags = extract_tags_regex("Logan went to Boston for work.")
        assert "logan" in tags
        assert "boston" in tags

    def test_multi_word_proper_nouns(self):
        tags = extract_tags_regex("I visited New York City last summer.")
        assert "new york city" in tags or "new york" in tags

    def test_quoted_strings(self):
        tags = extract_tags_regex('He said "dark mode" is better.')
        assert "dark mode" in tags

    def test_filters_common_sentence_starters(self):
        tags = extract_tags_regex("The weather is nice. However it rained.")
        # "The" and "However" should be filtered as sentence starters
        assert "the" not in tags
        assert "however" not in tags

    def test_filters_noise_tags(self):
        tags = extract_tags_regex("The System has good Information about the Topic.")
        for tag in tags:
            assert tag not in _NOISE_TAGS

    def test_filters_nationality_adjectives(self):
        tags = extract_tags_regex("The American team played well.")
        assert "american" not in tags

    def test_preserves_place_names(self):
        """Short capitalized words that aren't adjective-suffixed should pass."""
        tags = extract_tags_regex("Logan visited Paris and Rome.")
        assert "paris" in tags or "rome" in tags

    def test_max_8_tags(self):
        text = "Alice Bob Carol Dave Eve Frank Grace Hank Ivan Jack met in London."
        tags = extract_tags_regex(text)
        assert len(tags) <= 8

    def test_sorted_by_length(self):
        tags = extract_tags_regex('I like "machine learning" and Python.')
        if len(tags) >= 2:
            assert len(tags[0]) >= len(tags[1])

    def test_empty_input(self):
        assert extract_tags_regex("") == []
        assert extract_tags_regex("ab") == []

    def test_no_proper_nouns(self):
        tags = extract_tags_regex("the quick brown fox jumped over the lazy dog.")
        assert len(tags) == 0

    def test_strips_file_extensions(self):
        tags = extract_tags_regex("Check Settings.py for the config.")
        if tags:
            assert not any(t.endswith(".py") for t in tags)

    def test_filters_pure_numbers(self):
        tags = extract_tags_regex("Version 42 was released.")
        assert "42" not in tags


# ---------------------------------------------------------------------------
# _dedup_substrings
# ---------------------------------------------------------------------------

class TestDedupSubstrings:
    def test_removes_component_words(self):
        tags = ["ford mustang", "ford", "mustang", "python"]
        result = _dedup_substrings(tags)
        assert "ford mustang" in result
        assert "ford" not in result
        assert "mustang" not in result
        assert "python" in result

    def test_no_multi_word_tags(self):
        tags = ["python", "rust", "go"]
        result = _dedup_substrings(tags)
        assert result == tags

    def test_empty(self):
        assert _dedup_substrings([]) == []


# ---------------------------------------------------------------------------
# _is_adjective_not_place
# ---------------------------------------------------------------------------

class TestIsAdjectiveNotPlace:
    def test_nationality_adjectives(self):
        assert _is_adjective_not_place("canadian") is True  # ends in "ian"
        assert _is_adjective_not_place("british") is True   # ends in "ish"
        assert _is_adjective_not_place("japanese") is True   # ends in "ese"

    def test_short_words_not_adjective(self):
        assert _is_adjective_not_place("paris") is False
        assert _is_adjective_not_place("rome") is False

    def test_non_suffix(self):
        assert _is_adjective_not_place("python") is False


# ---------------------------------------------------------------------------
# TagService — extract_tags (LLM + regex fallback)
# ---------------------------------------------------------------------------

class TestTagServiceExtractTags:
    def test_with_llm_success(self):
        llm_fn = MagicMock(return_value=["Python", "Logan", "dark mode"])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Logan likes Python dark mode")
        assert "python" in tags
        assert "logan" in tags
        llm_fn.assert_called_once()

    def test_llm_returns_empty_list(self):
        """LLM returns [] → return [] (no regex fallback, benchmark-aligned)."""
        llm_fn = MagicMock(return_value=[])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Logan likes Python")
        assert tags == []

    def test_llm_returns_none(self):
        """LLM returns None → return [] (benchmark-aligned)."""
        llm_fn = MagicMock(return_value=None)
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Logan likes Python")
        assert tags == []

    def test_llm_raises_exception(self):
        llm_fn = MagicMock(side_effect=Exception("LLM down"))
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Logan likes Python")
        assert tags == []

    def test_no_llm_falls_back_to_regex(self):
        service = TagService(llm_extract_fn=None)
        tags = service.extract_tags("Logan went to Boston.")
        assert "logan" in tags
        assert "boston" in tags

    def test_llm_tags_registered_in_known_tags(self):
        llm_fn = MagicMock(return_value=["Python", "FastAPI"])
        service = TagService(llm_extract_fn=llm_fn)
        service.extract_tags("text")
        assert "python" in service.known_tags
        assert "fastapi" in service.known_tags

    def test_normalize_filters_noise(self):
        llm_fn = MagicMock(return_value=["system", "memory", "Python", "Logan"])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("text")
        assert "system" not in tags
        assert "memory" not in tags
        assert "python" in tags

    def test_normalize_max_8(self):
        llm_fn = MagicMock(return_value=[f"tag{i}" for i in range(15)])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("text")
        assert len(tags) <= 8

    def test_set_llm_extract_fn(self):
        service = TagService(llm_extract_fn=None)
        # Initially falls back to regex
        tags1 = service.extract_tags("Logan went to Boston.")
        assert len(tags1) > 0

        # Set LLM function
        llm_fn = MagicMock(return_value=["custom tag"])
        service.set_llm_extract_fn(llm_fn)
        tags2 = service.extract_tags("anything")
        assert "custom tag" in tags2


# ---------------------------------------------------------------------------
# TagService — extract_tags_async
# ---------------------------------------------------------------------------

class TestTagServiceExtractTagsAsync:
    @pytest.mark.asyncio
    async def test_async_llm_success(self):
        async def async_llm(text):
            return ["async tag", "Python"]

        service = TagService(llm_extract_fn=async_llm)
        tags = await service.extract_tags_async("test text")
        assert "async tag" in tags
        assert "python" in tags

    @pytest.mark.asyncio
    async def test_async_llm_failure(self):
        async def async_llm(text):
            return []

        service = TagService(llm_extract_fn=async_llm)
        tags = await service.extract_tags_async("test text")
        assert tags == []

    @pytest.mark.asyncio
    async def test_sync_llm_in_async_context(self):
        llm_fn = MagicMock(return_value=["sync tag"])
        service = TagService(llm_extract_fn=llm_fn)
        tags = await service.extract_tags_async("test text")
        assert "sync tag" in tags


# ---------------------------------------------------------------------------
# TagService — match_query_tags
# ---------------------------------------------------------------------------

class TestMatchQueryTags:
    def test_exact_match(self):
        service = TagService()
        service._known_tags = {"python", "fastapi", "logan"}
        matches = service.match_query_tags("tell me about python")
        assert "python" in matches

    def test_word_boundary(self):
        """'log' should NOT match 'logan'."""
        service = TagService()
        service._known_tags = {"logan", "python"}
        matches = service.match_query_tags("check the log files")
        assert "logan" not in matches

    def test_no_known_tags(self):
        service = TagService()
        assert service.match_query_tags("anything") == []

    def test_empty_query(self):
        service = TagService()
        service._known_tags = {"python"}
        assert service.match_query_tags("") == []

    def test_multiple_matches(self):
        service = TagService()
        service._known_tags = {"python", "fastapi", "react"}
        matches = service.match_query_tags("python and fastapi setup")
        assert "python" in matches
        assert "fastapi" in matches
        assert "react" not in matches

    def test_max_8_matches(self):
        service = TagService()
        service._known_tags = {f"tag{i}" for i in range(20)}
        query = " ".join(f"tag{i}" for i in range(20))
        matches = service.match_query_tags(query)
        assert len(matches) <= 8

    def test_case_insensitive(self):
        service = TagService()
        service._known_tags = {"python", "logan"}
        matches = service.match_query_tags("PYTHON and LOGAN")
        assert "python" in matches
        assert "logan" in matches

    def test_sorted_by_length(self):
        service = TagService()
        service._known_tags = {"ai", "machine learning", "python"}
        matches = service.match_query_tags("ai machine learning python")
        if len(matches) >= 2:
            assert len(matches[0]) >= len(matches[1])


# ---------------------------------------------------------------------------
# TagService — known tag index
# ---------------------------------------------------------------------------

class TestKnownTagIndex:
    def test_rebuild_from_collections(self):
        """Simulate ChromaDB collection scan."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"noun_tags": json.dumps(["python", "fastapi"])},
                {"noun_tags": json.dumps(["react", "typescript"])},
                {"noun_tags": ""},
                None,
            ]
        }
        mock_adapter = MagicMock()
        mock_adapter.collection = mock_collection

        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})
        assert "python" in service.known_tags
        assert "fastapi" in service.known_tags
        assert "react" in service.known_tags

    def test_rebuild_handles_invalid_json(self):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"noun_tags": "not valid json"},
                {"noun_tags": json.dumps(["valid tag"])},
            ]
        }
        mock_adapter = MagicMock()
        mock_adapter.collection = mock_collection

        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})
        assert "valid tag" in service.known_tags

    def test_rebuild_handles_collection_error(self):
        mock_adapter = MagicMock()
        mock_adapter.collection = MagicMock()
        mock_adapter.collection.get.side_effect = Exception("DB error")

        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})
        assert service.known_tag_count == 0

    def test_add_known_tags(self):
        service = TagService()
        service.add_known_tags(["python", "fastapi"])
        assert "python" in service.known_tags
        assert service.known_tag_count == 2

    def test_known_tags_returns_copy(self):
        service = TagService()
        service._known_tags = {"python"}
        copy = service.known_tags
        copy.add("not_in_original")
        assert "not_in_original" not in service._known_tags
