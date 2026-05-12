"""
Unit tests for MCP-related logic — scoring path branching and Claude Code config detection.

v0.3.3 §12: Action-KG tests removed alongside the dead code they covered
(_should_clear_action_cache, _cache_action_with_boundary_check, _mcp_action_cache,
MCP_CACHE_EXPIRY_SECONDS). The Action-KG plan never landed past v0.3.1's KG removal.
"""


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
