"""
Tests for sidecar_service.py — v0.3.1 exchange scoring and fact extraction.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modules.memory.sidecar_service import (
    _extract_json,
    _build_scoring_prompt,
    score_exchange,
    extract_facts,
    extract_noun_tags,
    summarize_only,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json('{"summary": "hello", "outcome": "worked"}')
        assert result["summary"] == "hello"
        assert result["outcome"] == "worked"

    def test_markdown_fenced(self):
        text = '```json\n{"summary": "hello"}\n```'
        result = _extract_json(text)
        assert result["summary"] == "hello"

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"summary": "hello"}\n```'
        result = _extract_json(text)
        assert result["summary"] == "hello"

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"tags": ["python", "api"]} hope that helps'
        result = _extract_json(text)
        assert result["tags"] == ["python", "api"]

    def test_nested_braces(self):
        text = '{"memory_scores": {"mem_1": "worked", "mem_2": "unknown"}}'
        result = _extract_json(text)
        assert result["memory_scores"]["mem_1"] == "worked"

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_invalid_json(self):
        assert _extract_json("{invalid json}") is None

    def test_array_not_object(self):
        # _extract_json looks for { } only — arrays won't match direct parse returns list
        result = _extract_json('["a", "b"]')
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _build_scoring_prompt
# ---------------------------------------------------------------------------

class TestBuildScoringPrompt:
    def test_no_memories(self):
        prompt = _build_scoring_prompt("hi", "hello", "thanks", [])
        assert '"summary"' in prompt
        assert '"outcome"' in prompt
        assert "memory_scores" not in prompt
        assert "MEMORY SCORES" not in prompt

    def test_with_memories(self):
        memories = [
            {"id": "working_abc", "content": "User likes Python"},
            {"id": "history_def", "content": "User prefers dark mode"},
        ]
        prompt = _build_scoring_prompt("hi", "hello", "thanks", memories)
        assert "memory_scores" in prompt
        assert "working_abc" in prompt
        assert "history_def" in prompt
        assert "MEMORY SCORES" in prompt

    def test_7_rule_guide_present(self):
        memories = [{"id": "m1", "content": "test"}]
        prompt = _build_scoring_prompt("hi", "hello", "thanks", memories)
        assert "NOT about the topic" in prompt
        assert '"unknown"' in prompt
        assert '"worked"' in prompt
        assert '"failed"' in prompt
        assert '"partial"' in prompt

    def test_summary_instructions(self):
        prompt = _build_scoring_prompt("hi", "hello", "thanks", [])
        assert "300 chars" in prompt
        assert "first-person" in prompt.lower() or "future self" in prompt.lower()


# ---------------------------------------------------------------------------
# score_exchange
# ---------------------------------------------------------------------------

class TestScoreExchange:
    @pytest.mark.asyncio
    async def test_success_no_memories(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"summary": "User said hi and I responded", "outcome": "worked"}'
        )
        result = await score_exchange("hi", "hello there", "thanks", [], client, "qwen2.5:3b")
        assert result is not None
        assert result["summary"] == "User said hi and I responded"
        assert result["outcome"] == "worked"
        assert result["memory_scores"] == {}

    @pytest.mark.asyncio
    async def test_success_with_memories(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value=json.dumps({
                "summary": "Discussed Python setup with memory context",
                "outcome": "worked",
                "memory_scores": {"working_abc": "worked", "history_def": "unknown"},
            })
        )
        memories = [
            {"id": "working_abc", "content": "Python dev"},
            {"id": "history_def", "content": "Old info"},
        ]
        result = await score_exchange("help", "sure", "great", memories, client, "model")
        assert result["memory_scores"]["working_abc"] == "worked"
        assert result["memory_scores"]["history_def"] == "unknown"

    @pytest.mark.asyncio
    async def test_invalid_outcome_defaults_unknown(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"summary": "A valid summary here.", "outcome": "good"}'
        )
        result = await score_exchange("hi", "hello", "ok", [], client, "model")
        assert result["outcome"] == "unknown"

    @pytest.mark.asyncio
    async def test_short_summary_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"summary": "hi", "outcome": "worked"}')
        result = await score_exchange("hi", "hello", "ok", [], client, "model")
        assert result is None  # summary < 10 chars

    @pytest.mark.asyncio
    async def test_llm_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=None)
        result = await score_exchange("hi", "hello", "ok", [], client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        client = MagicMock()
        client.generate_response = AsyncMock(side_effect=Exception("LLM down"))
        result = await score_exchange("hi", "hello", "ok", [], client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_summary_truncated_to_2000(self):
        long_summary = "A" * 2500
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value=json.dumps({"summary": long_summary, "outcome": "worked"})
        )
        result = await score_exchange("hi", "hello", "ok", [], client, "model")
        assert len(result["summary"]) == 2000

    @pytest.mark.asyncio
    async def test_invalid_memory_score_defaults_unknown(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value=json.dumps({
                "summary": "A valid summary here.",
                "outcome": "worked",
                "memory_scores": {"m1": "great", "m2": "worked"},
            })
        )
        memories = [{"id": "m1", "content": "a"}, {"id": "m2", "content": "b"}]
        result = await score_exchange("hi", "hello", "ok", memories, client, "model")
        assert result["memory_scores"]["m1"] == "unknown"  # "great" is invalid
        assert result["memory_scores"]["m2"] == "worked"

    @pytest.mark.asyncio
    async def test_assistant_msg_truncated(self):
        """Verify long assistant messages are truncated before prompt building."""
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"summary": "Truncation test summary.", "outcome": "worked"}'
        )
        long_msg = "x" * 10000
        result = await score_exchange("hi", long_msg, "ok", [], client, "model")
        assert result is not None
        # v0.3.1.4: assistant_msg truncated to 8000 chars before prompt building
        call_args = client.generate_response.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "x" * 8001 not in prompt


# ---------------------------------------------------------------------------
# extract_facts
# ---------------------------------------------------------------------------

class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"facts": ["User prefers Python 3.13", "Logan is a data scientist"]}'
        )
        result = await extract_facts("some exchange text", client, "model")
        assert len(result) == 2
        assert "Python 3.13" in result[0]

    @pytest.mark.asyncio
    async def test_empty_facts(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"facts": []}')
        result = await extract_facts("hi how are you", client, "model")
        assert result is None  # empty list returns None

    @pytest.mark.asyncio
    async def test_strips_bullet_prefixes(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"facts": ["• User likes tea", "- User dislikes coffee", "1. User is tall"]}'
        )
        result = await extract_facts("exchange", client, "model")
        assert all(not f.startswith(("•", "-", "1")) for f in result)

    @pytest.mark.asyncio
    async def test_filters_short_facts(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"facts": ["ok", "yes", "User prefers dark mode for coding"]}'
        )
        result = await extract_facts("exchange", client, "model")
        assert len(result) == 1  # only the long fact survives

    @pytest.mark.asyncio
    async def test_max_10_facts(self):
        facts = [f"Fact number {i} with enough length to pass filter" for i in range(15)]
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=json.dumps({"facts": facts}))
        result = await extract_facts("exchange", client, "model")
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=None)
        result = await extract_facts("exchange", client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_list_facts_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"facts": "not a list"}')
        result = await extract_facts("exchange", client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_content_truncated_to_8000(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"facts": ["Content was very long indeed for testing"]}'
        )
        long_content = "z" * 10000
        result = await extract_facts(long_content, client, "model")
        assert result is not None
        # v0.3.1.4: fact-extraction content truncated to 8000 chars
        call_args = client.generate_response.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "z" * 8001 not in prompt


# ---------------------------------------------------------------------------
# extract_noun_tags
# ---------------------------------------------------------------------------

class TestExtractNounTags:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"tags": ["Logan", "Python", "dark mode"]}'
        )
        result = await extract_noun_tags("Logan likes Python with dark mode", client, "model")
        assert "logan" in result  # lowercased
        assert "python" in result
        assert len(result) <= 8

    @pytest.mark.asyncio
    async def test_filters_pronouns(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"tags": ["he", "she", "user", "Python", "the assistant"]}'
        )
        result = await extract_noun_tags("text", client, "model")
        assert "he" not in result
        assert "she" not in result
        assert "user" not in result
        assert "the assistant" not in result
        assert "python" in result

    @pytest.mark.asyncio
    async def test_max_8_tags(self):
        tags = [f"tag{i}" for i in range(15)]
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=json.dumps({"tags": tags}))
        result = await extract_noun_tags("text", client, "model")
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=None)
        result = await extract_noun_tags("text", client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_list_tags_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"tags": "not a list"}')
        result = await extract_noun_tags("text", client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_short_tags_filtered(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"tags": ["a", "python"]}')
        result = await extract_noun_tags("text", client, "model")
        assert "a" not in result
        assert "python" in result

    @pytest.mark.asyncio
    async def test_empty_tags_returns_none(self):
        """All tags filtered out → returns None."""
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"tags": ["he", "she", "i"]}')
        result = await extract_noun_tags("text", client, "model")
        assert result is None


# ---------------------------------------------------------------------------
# summarize_only
# ---------------------------------------------------------------------------

class TestSummarizeOnly:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        client.generate_response = AsyncMock(
            return_value='{"summary": "Discussed Python setup and virtual environments"}'
        )
        result = await summarize_only("long exchange content here", client, "model")
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value=None)
        result = await summarize_only("content", client, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_summary_key_returns_none(self):
        client = MagicMock()
        client.generate_response = AsyncMock(return_value='{"text": "oops wrong key"}')
        result = await summarize_only("content", client, "model")
        assert result is None
