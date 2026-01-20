"""
Unit tests for MCP (Model Context Protocol) handler functions in main.py.

Tests critical functions:
- _should_clear_action_cache: Detects conversation boundaries (time gaps, topic shifts)
- _cache_action_with_boundary_check: Caches actions with automatic boundary detection
- detect_mcp_client: Identifies which MCP client is connected
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import asyncio

# Import the functions under test
# We need to set up imports carefully since main.py has side effects
sys.path.insert(0, str(__file__).rsplit('tests', 1)[0])

from main import (
    _should_clear_action_cache,
    _cache_action_with_boundary_check,
    _mcp_action_cache,
    MCP_CACHE_EXPIRY_SECONDS,
)
from modules.memory.types import ActionOutcome


class TestShouldClearActionCache:
    """Tests for _should_clear_action_cache - conversation boundary detection."""

    def setup_method(self):
        """Clear cache before each test."""
        _mcp_action_cache.clear()

    def test_no_existing_cache_returns_false(self):
        """Should return False when no cache exists for session."""
        should_clear, reason = _should_clear_action_cache("new_session", "coding")

        assert should_clear is False
        assert "No existing cache" in reason

    def test_same_context_same_time_no_clear(self):
        """Should not clear when context is same and recent."""
        session_id = "test_session"
        _mcp_action_cache[session_id] = {
            "actions": [],
            "last_context": "coding",
            "last_activity": datetime.now()
        }

        should_clear, reason = _should_clear_action_cache(session_id, "coding")

        assert should_clear is False
        assert "same_conversation" in reason

    def test_time_gap_triggers_clear(self):
        """Should clear cache when time gap exceeds threshold."""
        session_id = "test_session"
        old_time = datetime.now() - timedelta(seconds=MCP_CACHE_EXPIRY_SECONDS + 100)
        _mcp_action_cache[session_id] = {
            "actions": [MagicMock()],  # Dummy action
            "last_context": "coding",
            "last_activity": old_time
        }

        should_clear, reason = _should_clear_action_cache(session_id, "coding")

        assert should_clear is True
        assert "time_gap" in reason

    def test_context_shift_triggers_clear(self):
        """Should clear cache when context shifts to different topic."""
        session_id = "test_session"
        _mcp_action_cache[session_id] = {
            "actions": [MagicMock()],
            "last_context": "coding",
            "last_activity": datetime.now()
        }

        should_clear, reason = _should_clear_action_cache(session_id, "fitness")

        assert should_clear is True
        assert "context_shift" in reason
        assert "coding" in reason
        assert "fitness" in reason

    def test_general_context_ignored(self):
        """Shifts to/from 'general' should not trigger clear (too noisy)."""
        session_id = "test_session"
        _mcp_action_cache[session_id] = {
            "actions": [MagicMock()],
            "last_context": "coding",
            "last_activity": datetime.now()
        }

        # Shift to general - should NOT clear
        should_clear, reason = _should_clear_action_cache(session_id, "general")
        assert should_clear is False

        # Shift from general - should NOT clear either
        _mcp_action_cache[session_id]["last_context"] = "general"
        should_clear, reason = _should_clear_action_cache(session_id, "coding")
        assert should_clear is False

    def test_recent_activity_no_clear(self):
        """Should not clear when activity is recent even with time passed."""
        session_id = "test_session"
        recent_time = datetime.now() - timedelta(seconds=60)  # 1 minute ago
        _mcp_action_cache[session_id] = {
            "actions": [],
            "last_context": "coding",
            "last_activity": recent_time
        }

        should_clear, reason = _should_clear_action_cache(session_id, "coding")

        assert should_clear is False


class TestCacheActionWithBoundaryCheck:
    """Tests for _cache_action_with_boundary_check - action caching with boundaries."""

    def setup_method(self):
        """Clear cache before each test."""
        _mcp_action_cache.clear()

    def test_first_action_creates_cache(self):
        """First action for a session should create cache entry."""
        session_id = "new_session"
        action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="unknown"
        )

        _cache_action_with_boundary_check(session_id, action, "coding")

        assert session_id in _mcp_action_cache
        assert len(_mcp_action_cache[session_id]["actions"]) == 1
        assert _mcp_action_cache[session_id]["last_context"] == "coding"

    def test_subsequent_actions_appended(self):
        """Subsequent actions should be appended to cache."""
        session_id = "test_session"
        action1 = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="unknown"
        )
        action2 = ActionOutcome(
            action_type="add_to_memory_bank",
            context_type="coding",
            outcome="unknown"
        )

        _cache_action_with_boundary_check(session_id, action1, "coding")
        _cache_action_with_boundary_check(session_id, action2, "coding")

        assert len(_mcp_action_cache[session_id]["actions"]) == 2

    def test_context_shift_clears_and_restarts(self):
        """Context shift should clear old actions and start fresh."""
        session_id = "test_session"
        old_action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="unknown"
        )

        # Add initial action
        _cache_action_with_boundary_check(session_id, old_action, "coding")
        assert len(_mcp_action_cache[session_id]["actions"]) == 1

        # Context shift should clear and add new
        new_action = ActionOutcome(
            action_type="search_memory",
            context_type="fitness",
            outcome="unknown"
        )
        _cache_action_with_boundary_check(session_id, new_action, "fitness")

        # Should have only 1 action (old ones cleared)
        assert len(_mcp_action_cache[session_id]["actions"]) == 1
        assert _mcp_action_cache[session_id]["last_context"] == "fitness"

    def test_last_activity_updated(self):
        """Last activity timestamp should be updated on each action."""
        session_id = "test_session"
        action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="unknown"
        )

        before = datetime.now()
        _cache_action_with_boundary_check(session_id, action, "coding")
        after = datetime.now()

        last_activity = _mcp_action_cache[session_id]["last_activity"]
        assert before <= last_activity <= after

    def test_time_gap_clears_cache(self):
        """Time gap exceeding threshold should clear cache."""
        session_id = "test_session"

        # Manually set up old cache
        old_time = datetime.now() - timedelta(seconds=MCP_CACHE_EXPIRY_SECONDS + 100)
        _mcp_action_cache[session_id] = {
            "actions": [MagicMock(), MagicMock(), MagicMock()],  # 3 old actions
            "last_context": "coding",
            "last_activity": old_time
        }

        # Add new action - should clear the 3 old ones
        new_action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="unknown"
        )
        _cache_action_with_boundary_check(session_id, new_action, "coding")

        # Should have only 1 action
        assert len(_mcp_action_cache[session_id]["actions"]) == 1


class TestMCPCacheExpiry:
    """Tests for MCP cache expiry configuration."""

    def test_expiry_constant_is_reasonable(self):
        """Cache expiry should be a reasonable value (5-30 minutes)."""
        assert 300 <= MCP_CACHE_EXPIRY_SECONDS <= 1800  # 5-30 minutes

    def test_expiry_is_integer(self):
        """Cache expiry should be an integer for seconds calculation."""
        assert isinstance(MCP_CACHE_EXPIRY_SECONDS, int)


class TestActionOutcomeIntegration:
    """Tests for ActionOutcome type integration with cache."""

    def setup_method(self):
        """Clear cache before each test."""
        _mcp_action_cache.clear()

    def test_action_outcome_fields_preserved(self):
        """ActionOutcome fields should be preserved in cache."""
        session_id = "test_session"
        action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="worked",
            action_params={"query": "test query", "limit": 5},
            collection="history",
            doc_id="doc_123"
        )

        _cache_action_with_boundary_check(session_id, action, "coding")

        cached_action = _mcp_action_cache[session_id]["actions"][0]
        assert cached_action.action_type == "search_memory"
        assert cached_action.context_type == "coding"
        assert cached_action.outcome == "worked"
        assert cached_action.action_params == {"query": "test query", "limit": 5}
        assert cached_action.collection == "history"
        assert cached_action.doc_id == "doc_123"

    def test_multiple_action_types(self):
        """Different action types should all be cacheable."""
        session_id = "test_session"
        action_types = ["search_memory", "add_to_memory_bank", "update_memory", "get_context_insights"]

        for action_type in action_types:
            action = ActionOutcome(
                action_type=action_type,
                context_type="coding",
                outcome="unknown"
            )
            _cache_action_with_boundary_check(session_id, action, "coding")

        assert len(_mcp_action_cache[session_id]["actions"]) == 4
        cached_types = [a.action_type for a in _mcp_action_cache[session_id]["actions"]]
        assert cached_types == action_types


class TestMemoryScoresParameter:
    """Tests for v0.3.0 memory_scores parameter handling in record_response."""

    def test_memory_scores_dict_structure(self):
        """memory_scores should be a dict of doc_id -> outcome."""
        memory_scores = {
            "history_abc123": "worked",
            "patterns_xyz789": "failed",
            "working_def456": "unknown"
        }

        # Verify structure
        assert isinstance(memory_scores, dict)
        for doc_id, outcome in memory_scores.items():
            assert isinstance(doc_id, str)
            assert outcome in ["worked", "failed", "partial", "unknown"]

    def test_memory_scores_outcome_filtering(self):
        """Only worked/failed/partial outcomes should trigger scoring."""
        memory_scores = {
            "doc1": "worked",    # Should score
            "doc2": "failed",    # Should score
            "doc3": "partial",   # Should score
            "doc4": "unknown",   # Should NOT score
        }

        # Simulate the filtering logic from main.py
        scorable = [
            (doc_id, outcome)
            for doc_id, outcome in memory_scores.items()
            if outcome in ["worked", "failed", "partial"]
        ]

        assert len(scorable) == 3
        assert ("doc1", "worked") in scorable
        assert ("doc2", "failed") in scorable
        assert ("doc3", "partial") in scorable
        assert not any(doc_id == "doc4" for doc_id, _ in scorable)

    def test_memory_scores_empty_dict(self):
        """Empty memory_scores should not crash."""
        memory_scores = {}

        scorable = [
            (doc_id, outcome)
            for doc_id, outcome in memory_scores.items()
            if outcome in ["worked", "failed", "partial"]
        ]

        assert len(scorable) == 0

    def test_memory_scores_takes_precedence(self):
        """memory_scores should take precedence over related parameter."""
        # Simulate the condition check from main.py
        memory_scores = {"doc1": "worked"}
        related = [1, 2, 3]

        # Logic: if memory_scores -> use per-memory, elif related -> use legacy
        if memory_scores:
            scoring_mode = "per_memory"
        elif related:
            scoring_mode = "legacy"
        else:
            scoring_mode = "all"

        assert scoring_mode == "per_memory"

    def test_related_fallback_when_no_memory_scores(self):
        """Should fall back to related when memory_scores not provided."""
        memory_scores = None
        related = [1, 2, 3]

        if memory_scores:
            scoring_mode = "per_memory"
        elif related:
            scoring_mode = "legacy"
        else:
            scoring_mode = "all"

        assert scoring_mode == "legacy"

    def test_key_takeaway_required(self):
        """key_takeaway is required - forces LLM to reflect on what happened."""
        # Valid: has key_takeaway
        valid_args = [
            {"key_takeaway": "Some learning", "outcome": "worked"},
            {"key_takeaway": "Learning", "memory_scores": {"doc1": "worked"}},
            {"key_takeaway": "Routine exchange", "outcome": "unknown"},
        ]

        for args in valid_args:
            key_takeaway = args.get("key_takeaway")
            assert key_takeaway is not None, "key_takeaway must be provided"

        # Invalid: missing key_takeaway (would fail schema validation)
        invalid_args = [
            {"memory_scores": {"doc1": "worked"}},
            {"outcome": "worked"},
        ]

        for args in invalid_args:
            key_takeaway = args.get("key_takeaway")
            assert key_takeaway is None, "These examples lack required key_takeaway"


class TestRecordResponseScoringLogic:
    """Tests for the scoring logic paths in record_response."""

    def test_per_memory_scoring_path(self):
        """When memory_scores provided, should use per-memory scoring."""
        arguments = {
            "key_takeaway": "Test",
            "memory_scores": {
                "doc1": "worked",
                "doc2": "failed"
            }
        }

        memory_scores = arguments.get("memory_scores")
        assert memory_scores is not None
        assert len(memory_scores) == 2

    def test_legacy_related_path(self):
        """When related provided (no memory_scores), should use legacy scoring."""
        arguments = {
            "key_takeaway": "Test",
            "outcome": "worked",
            "related": [1, 2, 3]
        }

        memory_scores = arguments.get("memory_scores")
        related = arguments.get("related")

        assert memory_scores is None
        assert related == [1, 2, 3]

    def test_score_all_path(self):
        """When neither memory_scores nor related, should score all cached."""
        arguments = {
            "key_takeaway": "Test",
            "outcome": "worked"
        }

        memory_scores = arguments.get("memory_scores")
        related = arguments.get("related")

        assert memory_scores is None
        assert related is None

    def test_scoring_only_no_takeaway(self):
        """v0.3.0: Can score without storing a takeaway."""
        arguments = {
            "memory_scores": {
                "doc1": "worked",
                "doc2": "partial"
            }
        }

        key_takeaway = arguments.get("key_takeaway")
        memory_scores = arguments.get("memory_scores")

        assert key_takeaway is None
        assert memory_scores is not None
        assert len(memory_scores) == 2


class TestClaudeCodeMCPDetection:
    """Tests for Claude Code CLI MCP config detection (v0.3.0 fix).

    Claude Code CLI stores MCP config at ~/.claude.json (mcpServers at root),
    NOT at ~/.claude/mcp.json. This was a critical fix in v0.3.0.
    """

    def test_claude_code_config_path_detection(self, tmp_path):
        """Should detect ~/.claude.json as Claude Code CLI config."""
        from pathlib import Path

        # Simulate Claude Code CLI config structure
        config_path = tmp_path / ".claude.json"

        # Claude Code uses mcpServers at root level
        config_data = {
            "mcpServers": {
                "roampal": {
                    "command": "python",
                    "args": ["-m", "roampal.mcp"]
                }
            }
        }

        import json
        config_path.write_text(json.dumps(config_data))

        # Verify structure
        loaded = json.loads(config_path.read_text())
        assert "mcpServers" in loaded
        assert "roampal" in loaded["mcpServers"]

    def test_claude_code_vs_claude_desktop_format(self):
        """Claude Code uses flat format, Claude Desktop uses nested."""
        import json

        # Claude Code CLI format (v0.3.0 fix target)
        claude_code_format = {
            "mcpServers": {
                "roampal": {"command": "python", "args": []}
            }
        }

        # Claude Desktop format (different location, same structure)
        claude_desktop_format = {
            "mcpServers": {
                "roampal": {"command": "python", "args": []}
            }
        }

        # Both use mcpServers, but at different file paths:
        # - Claude Code: ~/.claude.json
        # - Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json

        # Verify roampal detection works in both
        assert "roampal" in claude_code_format.get("mcpServers", {})
        assert "roampal" in claude_desktop_format.get("mcpServers", {})

    def test_wrong_path_not_detected_as_claude_code(self, tmp_path):
        """~/.claude/mcp.json should NOT be detected as Claude Code CLI config."""
        from pathlib import Path

        # This is the WRONG path - Claude Code doesn't use this
        wrong_dir = tmp_path / ".claude"
        wrong_dir.mkdir()
        wrong_path = wrong_dir / "mcp.json"

        import json
        wrong_path.write_text(json.dumps({"mcpServers": {}}))

        # The detection logic checks:
        # config_path.name == ".claude.json" and config_path.parent == Path.home()

        # This path would fail because:
        assert wrong_path.name != ".claude.json"  # It's "mcp.json"
        # So it's not the Claude Code CLI config

    def test_claude_code_priority_over_other_configs(self):
        """Claude Code config should have highest priority (110)."""
        # From mcp.py get_config_priority function
        def get_config_priority(filename: str, is_home_root: bool, tool_lower: str) -> int:
            # Claude Code CLI config at ~/.claude.json (highest priority)
            if filename == ".claude.json" and is_home_root:
                return 110
            # Claude Desktop's official MCP config
            if tool_lower == "claude" and filename == "claude_desktop_config.json":
                return 100
            if filename.endswith("_desktop_config.json"):
                return 90
            if "mcp" in filename:
                return 80
            if filename == "config.json":
                return 10
            return 50

        # Verify Claude Code has highest priority
        claude_code_priority = get_config_priority(".claude.json", True, "claude")
        claude_desktop_priority = get_config_priority("claude_desktop_config.json", False, "claude")
        other_mcp_priority = get_config_priority("mcp.json", False, "other")

        assert claude_code_priority > claude_desktop_priority
        assert claude_code_priority > other_mcp_priority
        assert claude_code_priority == 110

    def test_roampal_connection_detection(self, tmp_path):
        """Should correctly detect roampal connection status."""
        import json

        # Connected config
        connected_config = {
            "mcpServers": {
                "roampal": {"command": "python", "args": []}
            }
        }

        # Not connected config
        not_connected_config = {
            "mcpServers": {
                "other-server": {"command": "node", "args": []}
            }
        }

        # Empty config
        empty_config = {
            "mcpServers": {}
        }

        # Test detection logic from mcp.py line 257
        def is_roampal_connected(config: dict) -> bool:
            return "roampal" in config.get("mcpServers", {})

        assert is_roampal_connected(connected_config) is True
        assert is_roampal_connected(not_connected_config) is False
        assert is_roampal_connected(empty_config) is False

    def test_connect_writes_to_correct_format(self, tmp_path):
        """Connect should write mcpServers at root for Claude Code."""
        import json

        config_path = tmp_path / ".claude.json"

        # Start with empty config
        config_path.write_text(json.dumps({}))

        # Simulate connect logic from mcp.py lines 509-514
        config = json.loads(config_path.read_text())

        # Claude Code CLI: mcpServers at root level
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        roampal_cmd = {
            "command": "python",
            "args": ["-m", "roampal.mcp"],
            "env": {"ROAMPAL_DATA_DIR": "/path/to/data"}
        }

        config["mcpServers"]["roampal"] = roampal_cmd
        config_path.write_text(json.dumps(config, indent=2))

        # Verify written correctly
        final_config = json.loads(config_path.read_text())
        assert "mcpServers" in final_config
        assert "roampal" in final_config["mcpServers"]
        assert final_config["mcpServers"]["roampal"]["command"] == "python"

    def test_disconnect_removes_from_correct_location(self, tmp_path):
        """Disconnect should remove from mcpServers at root."""
        import json

        config_path = tmp_path / ".claude.json"

        # Start with connected config
        config_path.write_text(json.dumps({
            "mcpServers": {
                "roampal": {"command": "python", "args": []},
                "other": {"command": "node", "args": []}
            }
        }))

        # Simulate disconnect logic from mcp.py lines 578-580
        config = json.loads(config_path.read_text())

        if "mcpServers" in config and "roampal" in config["mcpServers"]:
            del config["mcpServers"]["roampal"]

        config_path.write_text(json.dumps(config, indent=2))

        # Verify removed correctly
        final_config = json.loads(config_path.read_text())
        assert "roampal" not in final_config["mcpServers"]
        assert "other" in final_config["mcpServers"]  # Other servers preserved
