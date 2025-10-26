import json
import logging
import os
from typing import List, Dict, Any, Optional, Union
import datetime
import aiofiles
from pathlib import Path
from filelock import FileLock

from core.interfaces.memory_adapter_interface import MemoryAdapterInterface
from config.settings import settings, DATA_PATH  # Import settings
# Simplified paths - no user isolation needed
def get_loopsmith_path(resource):
    return Path(DATA_PATH) / resource

def get_session_path(session_id):
    return get_loopsmith_path("sessions") / f"{session_id}.jsonl"

logger = logging.getLogger(__name__)

class FileMemoryAdapter(MemoryAdapterInterface):
    def __init__(self):
        self.data_path: Optional[Path] = None
        self.arbitrary_data_dir: Optional[Path] = None
        self.sessions_dir: Optional[Path] = None
        self.initialized = False
        logger.debug("FileMemoryAdapter instance created (uninitialized).")

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.initialized:
            return
        if not config:
            raise ValueError("Config must be provided to FileMemoryAdapter.")
        
        base_data_path_str = config.get("base_data_path")
        if not base_data_path_str:
            raise ValueError("'base_data_path' must be provided in FileMemoryAdapter config.")
        
        self.data_path = Path(base_data_path_str).resolve()
        arbitrary_subdir = config.get("arbitrary_store_subdir", "arbitrary_store")
        
        # Check if base_data_path already contains arbitrary_store to avoid nesting
        if "arbitrary_store" in str(self.data_path):
            self.arbitrary_data_dir = self.data_path
        else:
            self.arbitrary_data_dir = self.data_path / arbitrary_subdir
        
        # --- Simplified Path Logic ---
        self.sessions_dir = get_loopsmith_path("sessions")

        # Create all necessary directories on startup
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.arbitrary_data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.initialized = True
        logger.info(f"FileMemoryAdapter initialized at path: {self.data_path}")
        logger.info(f"Session logs will be written to and read from: {self.sessions_dir}")

    # --- PUBLIC METHOD ---
    def get_sessions_directory(self) -> Path:
        """
        Returns the absolute path to the directory containing session logs.
        This is the method that other services will call for the correct path.
        """
        if not self.sessions_dir:
            raise RuntimeError("FileMemoryAdapter not initialized.")
        return self.sessions_dir

    def _get_conversation_file_path(self, session_id: str) -> Path:
        """
        Returns the full path to a session's conversation file.

        Note: Parameter named 'session_id' for backward compatibility.
        This is equivalent to 'conversation_id' used elsewhere in the system.
        """
        return self.get_sessions_directory() / f"{session_id}.jsonl"

    async def save_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves a turn of the conversation to the session's JSONL file.
        This now correctly writes 'role' and 'content' fields.

        Note: Parameter named 'session_id' for backward compatibility.
        This is equivalent to 'conversation_id' used elsewhere in the system.
        """
        file_path = self._get_conversation_file_path(session_id)
        now = datetime.datetime.now().isoformat()
        
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            if user_input:
                turn_data = {
                    "session_id": session_id,
                    "role": "user",
                    "content": user_input,
                    "timestamp": now,
                    "metadata": metadata or {}
                }
                # Extract images from metadata if present
                if metadata and "images" in metadata:
                    turn_data["images"] = metadata["images"]
                await f.write(json.dumps(turn_data) + "\n")
            if assistant_response:
                turn_data = {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": now,
                    "metadata": metadata or {}
                }
                # Assistant responses might also have images
                if metadata and "response_images" in metadata:
                    turn_data["images"] = metadata["response_images"]
                await f.write(json.dumps(turn_data) + "\n")

    async def load_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        file_path = self._get_conversation_file_path(session_id)
        if not file_path.exists():
            return []
        
        lines = []
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        return lines[-limit:] if limit else lines

    async def list_all_session_ids(self) -> List[str]:
        """Lists all session IDs by finding all conversation files in the dedicated sessions subfolder."""
        sessions_dir = self.get_sessions_directory()
        if not sessions_dir.exists():
            return []
        return [f.stem for f in sessions_dir.glob("*.jsonl")]

    async def update_session_title(self, session_id: str, title: str) -> None:
        """
        Update the title in the first message metadata of a session file.
        This modifies the JSONL file in-place by updating the first line with atomic writes.

        NOTE: Caller must handle concurrency control. This method does not lock internally
        to avoid deadlocks with per-conversation locks in the service layer.
        """
        session_file = self._get_conversation_file_path(session_id)

        if not session_file.exists():
            logger.warning(f"Cannot update title: session file {session_id} does not exist")
            return

        try:
            def update_title_sync():
                """Synchronous file update with atomic writes (NO locking - caller handles)"""
                # Read all lines
                with open(session_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines:
                    logger.warning(f"Cannot update title: session file {session_id} is empty")
                    return

                # Parse first line and update metadata
                first_msg = json.loads(lines[0].strip())
                if "metadata" not in first_msg:
                    first_msg["metadata"] = {}
                first_msg["metadata"]["title"] = title

                # Write to temp file first (atomic operation)
                temp_file = session_file.with_suffix('.title_tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(first_msg, ensure_ascii=False) + '\n')
                    for line in lines[1:]:
                        f.write(line)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                temp_file.replace(session_file)

            # Run in thread pool to avoid blocking
            import asyncio
            await asyncio.to_thread(update_title_sync)

            logger.debug(f"Updated title for session {session_id}: {title}")

        except Exception as e:
            logger.error(f"Failed to update title for session {session_id}: {e}", exc_info=True)

    # --- Other existing methods below are untouched ---
    def _get_goals_path(self, session_id: Optional[str] = None) -> Path:
        if not self.data_path:
            raise RuntimeError("FileMemoryAdapter not initialized with data_path.")
        return self.data_path / "goals.txt"

    def _get_values_path(self, session_id: Optional[str] = None) -> Path:
        if not self.data_path:
            raise RuntimeError("FileMemoryAdapter not initialized with data_path.")
        return self.data_path / "values.txt"

    async def add_goal(self, session_id: str, goal: str) -> None:
        goals_path = self._get_goals_path(session_id)
        current_goals = []
        if goals_path.exists():
            async with aiofiles.open(goals_path, "r", encoding='utf-8') as f:
                content = await f.read()
                current_goals = [g.strip() for g in content.splitlines() if g.strip()]
        if goal not in current_goals:
            current_goals.append(goal)
            async with aiofiles.open(goals_path, "w", encoding='utf-8') as f:
                await f.write("\n".join(current_goals) + "\n")

    async def get_goals(self, session_id: str) -> List[str]:
        goals_path = self._get_goals_path(session_id)
        if goals_path.exists():
            async with aiofiles.open(goals_path, "r", encoding='utf-8') as f:
                content = await f.read()
            return [g.strip() for g in content.splitlines() if g.strip()]
        return []

    async def store_arbitrary_data(self, key: str, data: Any) -> bool:
        if not self.arbitrary_data_dir:
            raise RuntimeError("Adapter not initialized.")
        safe_key = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in key) + ".json"
        file_path = self.arbitrary_data_dir / safe_key
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))
        return True

    async def retrieve_arbitrary_data(self, key: str) -> Optional[Any]:
        if not self.arbitrary_data_dir:
            raise RuntimeError("Adapter not initialized.")
        safe_key = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in key) + ".json"
        file_path = self.arbitrary_data_dir / safe_key
        if file_path.exists():
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                data_str = await f.read()
            return json.loads(data_str)
        return None

    # --- Implement the abstract search_memories method ---
    async def search_memories(self, session_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memories based on a query embedding using a simple text similarity approach."""
        if not self.sessions_dir or not self.sessions_dir.exists():
            return []

        results = []
        for file_path in self.sessions_dir.glob("*.jsonl"):
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    async for line in f:
                        if line.strip():
                            memory = json.loads(line)
                            # Simple similarity check: compare content text (basic approach)
                            if "content" in memory:
                                results.append(memory)
            except Exception as e:
                logger.warning(f"Error reading memory file {file_path}: {e}")
                continue

        # Sort by timestamp (newest first) as a basic ranking, limit to top_k
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:top_k]

    # --- Unused methods placeholders ---
    async def remove_goal(self, session_id: str, goal_to_remove: str) -> bool: return False
    async def add_value(self, session_id: str, value: str) -> None: pass
    async def get_values(self, session_id: str) -> List[str]: return []
    async def remove_value(self, session_id: str, value_to_remove: str) -> bool: return False
    async def delete_arbitrary_data(self, key: str) -> bool: return False
    def _get_learnings_path(self) -> Path: raise NotImplementedError
    async def add_learning(self, learning_content: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]: raise NotImplementedError
    async def get_learnings(self, **kwargs) -> List[Dict[str, Any]]: return []
    async def close(self) -> None: pass
