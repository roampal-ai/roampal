"""
Session management endpoints for conversation persistence
"""
import logging
import json
from typing import List, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]

class SessionLoadResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

@router.get("/list")
async def list_sessions(request: Request) -> SessionListResponse:
    """
    List all available conversations

    Note: Returns 'session_id' key for backward compatibility with existing UI.
    This is equivalent to 'conversation_id' used in other endpoints.
    """
    memory = request.app.state.memory
    try:
        # Use AppData paths, not bundled data folder
        sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")
        sessions = []

        if sessions_dir.exists():
            # Include both active and archived sessions
            session_files = list(sessions_dir.glob("*.jsonl"))
            archive_dir = sessions_dir / "archive"
            if archive_dir.exists():
                session_files.extend(archive_dir.glob("*.jsonl"))

            for session_file in session_files:
                try:
                    # Read first and last message to get summary
                    with open(session_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if lines:
                            first_msg = json.loads(lines[0])
                            last_msg = json.loads(lines[-1]) if len(lines) > 1 else first_msg

                            # Check if there's a title stored in metadata
                            title = first_msg.get("metadata", {}).get("title", None)
                            if not title:
                                # Handle both old format (user/assistant) and new format (role/content)
                                # Old format: {user: "...", assistant: "..."}
                                # New format: {role: "user", content: "..."}
                                if "content" in first_msg:
                                    title = first_msg.get("content", "New Chat")[:50]
                                elif "user" in first_msg:
                                    title = first_msg.get("user", "New Chat")[:50]
                                else:
                                    title = "New Chat"

                            # Get content from either format
                            def get_content(msg):
                                if "content" in msg:
                                    return msg.get("content", "")
                                elif "user" in msg:
                                    return msg.get("user", "")
                                elif "assistant" in msg:
                                    return msg.get("assistant", "")
                                return ""

                            # Get timestamp - use file modification time as fallback
                            import os
                            file_mtime = os.path.getmtime(session_file)

                            # Ensure consistent timestamp format (always use unix timestamp as float)
                            first_timestamp = first_msg.get("timestamp")
                            last_timestamp = last_msg.get("timestamp")

                            # Convert to float if string ISO format, else use file_mtime
                            if isinstance(first_timestamp, str):
                                try:
                                    from datetime import datetime
                                    # Parse ISO string as local time (strip timezone markers)
                                    first_timestamp = datetime.fromisoformat(first_timestamp.split('+')[0].replace('Z', '')).timestamp()
                                except:
                                    first_timestamp = file_mtime
                            elif not first_timestamp:
                                first_timestamp = file_mtime

                            if isinstance(last_timestamp, str):
                                try:
                                    from datetime import datetime
                                    # Parse ISO string as local time (strip timezone markers)
                                    last_timestamp = datetime.fromisoformat(last_timestamp.split('+')[0].replace('Z', '')).timestamp()
                                except:
                                    last_timestamp = file_mtime
                            elif not last_timestamp:
                                last_timestamp = file_mtime

                            sessions.append({
                                "session_id": session_file.stem,
                                "title": title,
                                "message_count": len(lines),
                                "first_message": get_content(first_msg)[:100],
                                "last_message": get_content(last_msg)[:100],
                                "timestamp": first_timestamp,
                                "last_updated": last_timestamp
                            })
                except Exception as e:
                    logger.warning(f"Error reading session {session_file}: {e}")

        # Sort by last updated
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)

        return SessionListResponse(sessions=sessions)

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
async def load_session(session_id: str, request: Request) -> SessionLoadResponse:
    """Load a specific session"""
    memory = request.app.state.memory
    try:
        # Validate session_id format to prevent path traversal
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Use AppData paths, not bundled data folder
        sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")
        session_file = (sessions_dir / f"{session_id}.jsonl").resolve()
        sessions_dir = sessions_dir.resolve()
        
        # Verify the resolved path is within sessions directory
        try:
            session_file.relative_to(sessions_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        if not session_file.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        messages = []
        with open(session_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    msg_data = json.loads(line)

                    # Handle both old format (user/assistant fields) and new format (role/content)
                    if "role" in msg_data and "content" in msg_data:
                        # New format
                        messages.append({
                            "role": msg_data.get("role"),
                            "content": msg_data.get("content"),
                            "timestamp": msg_data.get("timestamp"),
                            "metadata": msg_data.get("metadata", {}),
                            "citations": msg_data.get("citations", [])
                        })
                    else:
                        # Old format - has user and/or assistant fields in same entry
                        if "user" in msg_data:
                            messages.append({
                                "role": "user",
                                "content": msg_data.get("user"),
                                "timestamp": msg_data.get("timestamp"),
                                "metadata": msg_data.get("metadata", {})
                            })
                        if "assistant" in msg_data:
                            messages.append({
                                "role": "assistant",
                                "content": msg_data.get("assistant"),
                                "timestamp": msg_data.get("timestamp"),
                                "metadata": msg_data.get("metadata", {}),
                                "citations": msg_data.get("citations", [])
                            })

        return SessionLoadResponse(session_id=session_id, messages=messages)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a session and clean up associated memories"""
    memory = request.app.state.memory
    try:
        # Validate session_id format to prevent path traversal
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Use AppData paths, not bundled data folder
        sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")
        session_file = (sessions_dir / f"{session_id}.jsonl").resolve()
        sessions_dir = sessions_dir.resolve()
        
        # Verify the resolved path is within sessions directory
        try:
            session_file.relative_to(sessions_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        if not session_file.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Reset active conversation if deleting it
        if hasattr(request.app.state, 'memory') and request.app.state.memory:
            if request.app.state.memory.conversation_id == session_id:
                request.app.state.memory.conversation_id = None
                logger.info(f"Reset active conversation (was being deleted)")

            # Delete associated memories from ChromaDB
            try:
                deleted_count = await request.app.state.memory.delete_by_conversation(session_id)
                logger.info(f"Deleted {deleted_count} memories for conversation {session_id}")
            except Exception as mem_error:
                logger.warning(f"Failed to delete memories for {session_id}: {mem_error}")
                # Continue with file deletion even if memory cleanup fails

        # Delete the session file
        session_file.unlink()

        return {
            "status": "success",
            "message": f"Session {session_id} deleted",
            "memories_deleted": deleted_count if 'deleted_count' in locals() else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))