"""
MCP support modules for Roampal
Enables external LLMs to learn and contribute to memory like internal LLM
"""

from .client_detector import detect_mcp_client, get_auto_session_id

__all__ = ["detect_mcp_client", "get_auto_session_id"]
