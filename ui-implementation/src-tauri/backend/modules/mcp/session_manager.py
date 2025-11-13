"""
MCP Session Manager
Tracks MCP client conversations with AUTOMATIC outcome detection
Enables external LLMs to learn like internal LLM without manual intervention
"""

import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from filelock import FileLock

logger = logging.getLogger(__name__)


class MCPSessionManager:
    """
    Manages MCP client sessions with automatic outcome detection.

    This matches internal LLM behavior where outcome detection happens
    automatically on every message without manual intervention.
    """

    def __init__(self, data_path: Path, memory_system):
        """
        Initialize MCP session manager.

        Args:
            data_path: Root data directory
            memory_system: UnifiedMemorySystem instance (for outcome detection)
        """
        self.data_path = Path(data_path)
        self.memory = memory_system  # Need reference for automatic outcome detection
        self.mcp_sessions_dir = self.data_path / "mcp_sessions"
        self.mcp_sessions_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MCP] Session manager initialized with automatic outcome detection")
        logger.info(f"[MCP] Sessions directory: {self.mcp_sessions_dir}")

    async def record_exchange(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        doc_id: str,
        outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record exchange with OPTIONAL LLM-provided outcome.

        Args:
            session_id: MCP client session ID (e.g., "mcp_claude_desktop_main")
            user_msg: User's message
            assistant_msg: Assistant's response
            doc_id: ChromaDB document ID for this exchange
            outcome: Optional outcome from LLM ("worked", "failed", "partial")

        Returns:
            Dict with:
                - doc_id: Current exchange doc_id
                - previous_outcome: Detected outcome for previous exchange (if any)
                - previous_score_change: Score change for previous exchange
                - promoted: Whether previous exchange was promoted
                - promoted_to: Collection promoted to (if applicable)
        """
        logger.info(f"[MCP] DEBUG: record_exchange() called for session {session_id}, outcome={outcome}")

        # 1. MANUAL OUTCOME RECORDING (if LLM provided one)
        # The LLM can optionally pass an outcome for the CURRENT exchange
        outcome_info = {
            "outcome": None,
            "score_change": None,
            "promoted": False,
            "promoted_to": None
        }

        # SCORE PREVIOUS EXCHANGE (DISABLED - now handled in main.py before calling this)
        # The scoring logic moved to main.py so it can score BOTH:
        # 1. Previous exchange
        # 2. Retrieved memories from cache
        # Both get scored based on the same outcome (current user feedback)
        prev_outcome_info = {
            "outcome": None,
            "score_change": None
        }

        # 2. AUTOMATIC OUTCOME DETECTION (DISABLED)
        # This was causing 4+ minute delays - now relying on LLM to pass outcome manually
        if False:  # Disabled - was blocking for 4.5 minutes
            prev_assistant = await self.get_last_assistant_with_doc_id(session_id)
            if prev_assistant:
                try:
                    # Build conversation pair for outcome detection
                    outcome_conversation = [
                        {"role": "assistant", "content": prev_assistant["content"]},
                        {"role": "user", "content": user_msg}
                    ]

                    # Use same outcome detector as internal LLM
                    outcome_result = await self.memory.detect_conversation_outcome(
                        outcome_conversation
                    )

                    # Auto-score if outcome detected
                    if outcome_result.get("outcome") in ["worked", "failed", "partial"]:
                        # Get old score before updating
                        prev_doc_id = prev_assistant["doc_id"]

                        # Determine which collection the doc is in
                        prev_doc = None
                        for collection_name in ["working", "history", "patterns"]:
                            try:
                                prev_doc = self.memory.collections[collection_name].get_fragment(prev_doc_id)
                                if prev_doc:
                                    break
                            except:
                                continue

                        if prev_doc:
                            old_score = prev_doc.get("metadata", {}).get("score", 0.5)

                            # Record outcome (this updates score and checks promotion)
                            await self.memory.record_outcome(
                                doc_id=prev_doc_id,
                                outcome=outcome_result["outcome"]
                            )

                            # Get new score after updating
                            updated_doc = None
                            for collection_name in ["working", "history", "patterns"]:
                                try:
                                    updated_doc = self.memory.collections[collection_name].get_fragment(prev_doc_id)
                                    if updated_doc:
                                        # Check if doc was promoted (doc_id changed)
                                        if not updated_doc:
                                            # Doc was promoted, check other collections
                                            for promo_coll in ["history", "patterns"]:
                                                # Look for recently promoted docs with similar content
                                                # This is a simplified check - promotion changes doc_id
                                                pass
                                        break
                                except:
                                    continue

                            if updated_doc:
                                new_score = updated_doc.get("metadata", {}).get("score", old_score)
                                outcome_info = {
                                    "outcome": outcome_result["outcome"],
                                    "score_change": f"{old_score:.2f} → {new_score:.2f}",
                                    "confidence": outcome_result.get("confidence", 0.0),
                                    "promoted": False,  # Promotion detected in next search if doc_id changed
                                    "promoted_to": None
                                }

                                logger.info(
                                    f"[MCP] Auto-detected outcome '{outcome_result['outcome']}' "
                                    f"for {prev_doc_id} (score: {old_score:.2f} → {new_score:.2f})"
                                )
                            else:
                                # Document may have been promoted (doc_id changed)
                                outcome_info = {
                                    "outcome": outcome_result["outcome"],
                                    "score_change": f"{old_score:.2f} → [promoted]",
                                    "promoted": True,
                                    "promoted_to": "history or patterns"  # Can't determine exactly without search
                                }
                                logger.info(
                                    f"[MCP] Auto-detected outcome '{outcome_result['outcome']}' "
                                    f"for {prev_doc_id}, document may have been promoted"
                                )
                        else:
                            logger.warning(f"[MCP] Could not find previous document {prev_doc_id} for scoring")

                except Exception as e:
                    logger.error(f"[MCP] Error in automatic outcome detection: {e}", exc_info=True)

        # 2. Save current exchange to session file (only if not already present)
        # PERFORMANCE FIX: Removed FileLock (was deadlocking for 4+ minutes)
        # JSONL append is atomic enough for MCP use case
        logger.info(f"[MCP] DEBUG: Checking if exchange already exists in session file")
        session_file = self.mcp_sessions_dir / f"{session_id}.jsonl"

        # Check if this doc_id already exists in session file
        exchange_exists = False
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("doc_id") == doc_id:
                            exchange_exists = True
                            logger.info(f"[MCP] Exchange with doc_id {doc_id} already exists, skipping append")
                            break
                    except json.JSONDecodeError:
                        continue

        # Only append if this is a NEW exchange
        if not exchange_exists:
            logger.info(f"[MCP] DEBUG: Opening file to append new exchange: {session_file}")
            with open(session_file, "a", encoding="utf-8") as f:
                # User message
                f.write(json.dumps({
                    "role": "user",
                    "content": user_msg,
                    "timestamp": datetime.now().isoformat()
                }) + "\n")

                # Assistant message with doc_id link to ChromaDB
                f.write(json.dumps({
                    "role": "assistant",
                    "content": assistant_msg,
                    "doc_id": doc_id,  # Links to ChromaDB
                    "timestamp": datetime.now().isoformat()
                }) + "\n")

            logger.info(f"[MCP] DEBUG: File write completed successfully")

        # Return info about what was scored
        logger.info(f"[MCP] DEBUG: Returning result dict")

        return {
            "doc_id": doc_id,
            "previous_outcome": prev_outcome_info["outcome"],
            "previous_score_change": prev_outcome_info["score_change"],
            "promoted": False,  # Promotion happens in background task
            "promoted_to": None
        }

    async def get_last_assistant_with_doc_id(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        Get last assistant message with doc_id from session file.

        Args:
            session_id: MCP client session ID

        Returns:
            Dict with 'content' and 'doc_id', or None if not found
        """
        session_file = self.mcp_sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return None

        try:
            # PERFORMANCE FIX: Removed FileLock (was deadlocking)
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Read backwards to find last assistant message with doc_id
            for line in reversed(lines):
                try:
                    entry = json.loads(line.strip())
                    if entry.get("role") == "assistant" and entry.get("doc_id"):
                        return {
                            "content": entry["content"],
                            "doc_id": entry["doc_id"],
                            "timestamp": entry.get("timestamp")
                        }
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"[MCP] Error reading session file {session_file}: {e}")

        return None

    async def update_last_assistant_doc_id(self, session_id: str, doc_id: str) -> bool:
        """
        Update the doc_id for the last assistant message in session file.

        Args:
            session_id: MCP client session ID
            doc_id: ChromaDB document ID to link

        Returns:
            True if updated successfully, False otherwise
        """
        session_file = self.mcp_sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            logger.warning(f"[MCP] Session file not found: {session_file}")
            return False

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find last assistant message and update its doc_id
            for i in range(len(lines) - 1, -1, -1):
                try:
                    entry = json.loads(lines[i].strip())
                    if entry.get("role") == "assistant":
                        entry["doc_id"] = doc_id
                        lines[i] = json.dumps(entry) + "\n"
                        break
                except json.JSONDecodeError:
                    continue

            # Write back
            with open(session_file, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"[MCP] Updated last assistant message with doc_id: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"[MCP] Error updating doc_id: {e}")
            return False

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages from session file.

        Args:
            session_id: MCP client session ID
            limit: Maximum number of messages to return (None = all messages)

        Returns:
            List of message dicts (role, content, timestamp, doc_id)
        """
        session_file = self.mcp_sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return []

        lock_file = str(session_file) + ".lock"
        messages = []

        try:
            with FileLock(lock_file, timeout=10):
                with open(session_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            # Apply limit if specified
            if limit is not None:
                lines = lines[-limit:]

            # Parse all messages
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    messages.append({
                        "role": entry.get("role"),
                        "content": entry.get("content"),
                        "timestamp": entry.get("timestamp"),
                        "doc_id": entry.get("doc_id")  # May be None for user messages
                    })
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"[MCP] Error reading session messages: {e}")

        return messages

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history from session file.
        Alias for get_session_messages() with default limit.

        Args:
            session_id: MCP client session ID
            limit: Maximum number of messages to return

        Returns:
            List of message dicts (role, content, timestamp)
        """
        return await self.get_session_messages(session_id, limit=limit)

    def list_sessions(self) -> List[str]:
        """
        List all MCP session IDs.

        Returns:
            List of session IDs (e.g., ["mcp_claude_desktop_main", "mcp_cursor_main"])
        """
        if not self.mcp_sessions_dir.exists():
            return []

        sessions = []
        for file in self.mcp_sessions_dir.glob("*.jsonl"):
            sessions.append(file.stem)  # Filename without .jsonl extension

        return sorted(sessions)

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: MCP client session ID

        Returns:
            Dict with message_count, first_message_time, last_message_time
        """
        session_file = self.mcp_sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return {
                "message_count": 0,
                "first_message_time": None,
                "last_message_time": None
            }

        lock_file = str(session_file) + ".lock"

        try:
            with FileLock(lock_file, timeout=10):
                with open(session_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            if not lines:
                return {
                    "message_count": 0,
                    "first_message_time": None,
                    "last_message_time": None
                }

            first_entry = json.loads(lines[0].strip())
            last_entry = json.loads(lines[-1].strip())

            return {
                "message_count": len(lines),
                "first_message_time": first_entry.get("timestamp"),
                "last_message_time": last_entry.get("timestamp"),
                "exchanges": len([l for l in lines if '"role": "assistant"' in l])
            }

        except Exception as e:
            logger.error(f"[MCP] Error getting session stats: {e}")
            return {
                "message_count": 0,
                "first_message_time": None,
                "last_message_time": None
            }
