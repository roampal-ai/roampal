"""
Session Cleanup Manager - Prevents memory bloat from unlimited session accumulation
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


class SessionCleanupManager:
    """
    Manages session lifecycle to prevent memory bloat.
    Uses LRU (Least Recently Used) eviction policy.
    """

    # Production configuration
    MAX_SESSIONS = 100  # Maximum number of concurrent sessions
    MAX_SESSION_AGE_HOURS = 72  # Sessions older than 3 days are eligible for cleanup
    CLEANUP_INTERVAL_MINUTES = 30  # Run cleanup every 30 minutes
    ARCHIVE_RETENTION_DAYS = 90  # Archives older than 90 days are permanently deleted

    def __init__(self, sessions_dir: Path, max_sessions: int = MAX_SESSIONS):
        self.sessions_dir = sessions_dir
        self.max_sessions = max_sessions
        self.active_sessions: OrderedDict[str, float] = OrderedDict()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the cleanup manager and start background cleanup task"""
        # Load existing sessions
        await self._load_existing_sessions()

        # Start background cleanup task
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"Session cleanup manager initialized with max_sessions={self.max_sessions}")

    async def _load_existing_sessions(self):
        """Load existing session files and their modification times"""
        if not self.sessions_dir.exists():
            return

        for session_file in self.sessions_dir.glob("*.jsonl"):
            try:
                mtime = session_file.stat().st_mtime
                session_id = session_file.stem
                self.active_sessions[session_id] = mtime
            except Exception as e:
                logger.warning(f"Failed to load session {session_file}: {e}")

        logger.info(f"Loaded {len(self.active_sessions)} existing sessions")

        # Perform initial cleanup if needed
        await self._cleanup_old_sessions()

    async def register_session(self, session_id: str):
        """Register a new or accessed session"""
        # Move to end (most recently used)
        if session_id in self.active_sessions:
            self.active_sessions.move_to_end(session_id)
        else:
            self.active_sessions[session_id] = time.time()

        # Check if we need to evict old sessions
        await self._evict_if_needed()

    async def _evict_if_needed(self):
        """Evict oldest sessions if we exceed max_sessions"""
        while len(self.active_sessions) > self.max_sessions:
            # Get oldest session (first in OrderedDict)
            oldest_session_id = next(iter(self.active_sessions))
            await self._remove_session(oldest_session_id)
            logger.info(f"Evicted session {oldest_session_id} (LRU policy, exceeded {self.max_sessions} sessions)")

    async def _remove_session(self, session_id: str):
        """Remove a session file and its tracking"""
        session_file = self.sessions_dir / f"{session_id}.jsonl"

        # Archive instead of delete (move to archive directory)
        archive_dir = self.sessions_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        try:
            if session_file.exists():
                archive_file = archive_dir / session_file.name
                session_file.rename(archive_file)
                logger.debug(f"Archived session {session_id} to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")

        # Remove from tracking
        self.active_sessions.pop(session_id, None)

    async def _cleanup_old_sessions(self):
        """Clean up sessions older than MAX_SESSION_AGE_HOURS"""
        current_time = time.time()
        max_age_seconds = self.MAX_SESSION_AGE_HOURS * 3600

        sessions_to_remove = []
        for session_id, last_access in list(self.active_sessions.items()):
            if current_time - last_access > max_age_seconds:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self._remove_session(session_id)
            logger.info(f"Cleaned up old session {session_id} (older than {self.MAX_SESSION_AGE_HOURS} hours)")

    async def _cleanup_archived_sessions(self):
        """Delete archived sessions older than ARCHIVE_RETENTION_DAYS"""
        archive_dir = self.sessions_dir / "archive"
        if not archive_dir.exists():
            return

        current_time = time.time()
        max_archive_age_seconds = self.ARCHIVE_RETENTION_DAYS * 24 * 3600
        deleted_count = 0

        try:
            for archive_file in archive_dir.glob("*.jsonl"):
                try:
                    file_age = current_time - archive_file.stat().st_mtime
                    if file_age > max_archive_age_seconds:
                        archive_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old archive: {archive_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete archive {archive_file.name}: {e}")

            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} archived sessions older than {self.ARCHIVE_RETENTION_DAYS} days")

        except Exception as e:
            logger.error(f"Error cleaning archived sessions: {e}")

    async def _cleanup_loop(self):
        """Background task that periodically cleans up old sessions and archives"""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_MINUTES * 60)
                await self._cleanup_old_sessions()
                await self._cleanup_archived_sessions()  # Also cleanup old archives
                logger.debug(f"Cleanup task completed. Active sessions: {len(self.active_sessions)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_session_count(self) -> int:
        """Get the current number of active sessions"""
        return len(self.active_sessions)

    def get_session_stats(self) -> Dict:
        """Get statistics about sessions"""
        if not self.active_sessions:
            return {
                "active_sessions": 0,
                "max_sessions": self.max_sessions,
                "oldest_session_age_hours": 0,
                "newest_session_age_hours": 0
            }

        current_time = time.time()
        ages = [(current_time - t) / 3600 for t in self.active_sessions.values()]

        return {
            "active_sessions": len(self.active_sessions),
            "max_sessions": self.max_sessions,
            "oldest_session_age_hours": max(ages),
            "newest_session_age_hours": min(ages),
            "cleanup_interval_minutes": self.CLEANUP_INTERVAL_MINUTES,
            "max_session_age_hours": self.MAX_SESSION_AGE_HOURS
        }

    async def shutdown(self):
        """Shutdown the cleanup manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Session cleanup manager shut down")