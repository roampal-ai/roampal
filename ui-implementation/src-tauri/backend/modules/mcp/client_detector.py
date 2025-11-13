"""
MCP Client Detection
Auto-detect which MCP client is connecting to assign appropriate session IDs
"""

import os
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import psutil for process detection (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("[MCP] psutil not available, using environment-based detection only")


def detect_mcp_client() -> str:
    """
    Detect which MCP client is connecting.

    Returns:
        Session ID format: mcp_{client}_main

    Detection order:
        1. Environment variables
        2. Parent process name (if psutil available)
        3. Fallback to "unknown"
    """
    # Method 1: Environment variables
    if os.getenv("CLAUDE_DESKTOP"):
        return "mcp_claude_desktop_main"

    if os.getenv("CURSOR_IDE"):
        return "mcp_cursor_main"

    if os.getenv("CONTINUE_DEV"):
        return "mcp_continue_dev_main"

    if os.getenv("CLINE"):
        return "mcp_cline_main"

    # Method 2: Parent process detection (if psutil available)
    if PSUTIL_AVAILABLE:
        try:
            current_process = psutil.Process()
            parent = current_process.parent()

            if parent:
                parent_name = parent.name().lower()

                # Claude Desktop
                if "claude" in parent_name:
                    return "mcp_claude_desktop_main"

                # Cursor
                if "cursor" in parent_name:
                    return "mcp_cursor_main"

                # Continue.dev
                if "continue" in parent_name:
                    return "mcp_continue_dev_main"

                # Cline (formerly Claude Dev)
                if "cline" in parent_name or "claude-dev" in parent_name:
                    return "mcp_cline_main"

                # VS Code (might be running Continue or Cline)
                if "code" in parent_name or "vscode" in parent_name:
                    # Check for Continue.dev or Cline in grandparent
                    grandparent = parent.parent()
                    if grandparent:
                        gp_name = grandparent.name().lower()
                        if "continue" in gp_name:
                            return "mcp_continue_dev_main"
                        if "cline" in gp_name:
                            return "mcp_cline_main"

                    # Default to generic vscode
                    return "mcp_vscode_main"

                logger.debug(f"[MCP] Unknown parent process: {parent_name}")

        except Exception as e:
            logger.debug(f"[MCP] Error detecting parent process: {e}")

    # Method 3: Fallback
    logger.info("[MCP] Could not detect client, using 'unknown'")
    return "mcp_unknown_main"


def get_auto_session_id() -> str:
    """
    Get auto-detected session ID for current MCP client.
    Convenience wrapper for detect_mcp_client().

    Returns:
        Session ID string (e.g., "mcp_claude_desktop_main")
    """
    return detect_mcp_client()


# Client display names for logging/UI
CLIENT_NAMES = {
    "mcp_claude_desktop_main": "Claude Desktop",
    "mcp_cursor_main": "Cursor",
    "mcp_continue_dev_main": "Continue.dev",
    "mcp_cline_main": "Cline",
    "mcp_vscode_main": "VS Code",
    "mcp_unknown_main": "Unknown MCP Client"
}


def get_client_display_name(session_id: str) -> str:
    """
    Get human-readable display name for client.

    Args:
        session_id: Session ID (e.g., "mcp_claude_desktop_main")

    Returns:
        Display name (e.g., "Claude Desktop")
    """
    return CLIENT_NAMES.get(session_id, session_id)
