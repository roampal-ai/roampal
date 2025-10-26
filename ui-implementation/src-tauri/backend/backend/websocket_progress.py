"""
WebSocket Progress Tracking for Book Processing
Provides real-time progress updates for long-running tasks
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for progress tracking"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.task_progress: Dict[str, Dict] = {}
        self.cancelled_tasks: Set[str] = set()

    async def connect(self, websocket: WebSocket, task_id: str):
        """Accept and register a WebSocket connection for a task"""
        await websocket.accept()
        self.active_connections[task_id].add(websocket)
        logger.info(f"WebSocket connected for task {task_id}")

        # Send current progress if task exists
        if task_id in self.task_progress:
            await websocket.send_json(self.task_progress[task_id])

    def disconnect(self, websocket: WebSocket, task_id: str):
        """Remove a WebSocket connection"""
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket disconnected for task {task_id}")

    async def send_progress(self, task_id: str, data: dict):
        """Send progress update to all connected clients for a task"""
        if task_id in self.active_connections:
            logger.info(f"Sending progress update to {len(self.active_connections[task_id])} connections for task {task_id}: {data.get('status', 'unknown')}")
            disconnected = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(data)
                    # Add small delay for completion messages to ensure delivery
                    if data.get('status') == 'completed':
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to send to connection: {e}")
                    disconnected.add(connection)

            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[task_id].discard(conn)
        else:
            logger.warning(f"No active connections for task {task_id} when trying to send: {data.get('status', 'unknown')}")

    def update_progress(self, task_id: str, data: dict):
        """Update stored progress for a task"""
        if task_id not in self.task_progress:
            self.task_progress[task_id] = {
                "task_id": task_id,
                "started_at": datetime.now().isoformat()
            }
        self.task_progress[task_id].update(data)
        self.task_progress[task_id]["updated_at"] = datetime.now().isoformat()

    def complete_task(self, task_id: str, success: bool = True, error: Optional[str] = None):
        """Mark a task as complete"""
        if task_id in self.task_progress:
            self.task_progress[task_id].update({
                "status": "completed" if success else "failed",
                "completed_at": datetime.now().isoformat(),
                "error": error
            })

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.task_progress:
            self.cancelled_tasks.add(task_id)
            self.task_progress[task_id]["status"] = "cancelled"
            return True
        return False

    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled"""
        return task_id in self.cancelled_tasks

# Global manager instance
manager = ConnectionManager()

def update_progress(task_id: str, data: dict):
    """Update progress for a task (synchronous wrapper)"""
    manager.update_progress(task_id, data)

def initialize_task(book_id: str) -> str:
    """Initialize a new processing task and return its ID"""
    task_id = str(uuid.uuid4())
    manager.update_progress(task_id, {
        "task_id": task_id,
        "book_id": book_id,
        "status": "initializing",
        "progress": 0,
        "message": "Task created"
    })
    logger.info(f"Initialized task {task_id} for book {book_id}")
    return task_id

async def send_progress_update(
    task_id: str,
    status: str = "processing",
    message: str = "",
    progress: float = 0,
    stage: Optional[str] = None,
    current_chunk: Optional[int] = None,
    total_chunks: Optional[int] = None,
    error: Optional[str] = None
):
    """Send a progress update for a task"""
    update_data = {
        "task_id": task_id,
        "status": status,
        "message": message,
        "progress": progress
    }

    # Add optional fields if provided
    if stage:
        update_data["stage"] = stage
    if current_chunk is not None:
        update_data["current_chunk"] = current_chunk
    if total_chunks is not None:
        update_data["total_chunks"] = total_chunks
    if error:
        update_data["error"] = error

    # Update stored progress
    manager.update_progress(task_id, update_data)

    # Send to connected clients
    await manager.send_progress(task_id, update_data)

    logger.debug(f"Progress update for task {task_id}: {status} - {progress}%")

def get_task_status(task_id: str) -> dict:
    """Get the current status of a task"""
    if task_id in manager.task_progress:
        return manager.task_progress[task_id]
    return {"status": "not_found", "task_id": task_id}

def cancel_task(task_id: str) -> bool:
    """Cancel a running task"""
    return manager.cancel_task(task_id)

def is_task_cancelled(task_id: str) -> bool:
    """Check if a task has been cancelled"""
    return manager.is_cancelled(task_id)

async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for progress tracking"""
    await manager.connect(websocket, task_id)
    try:
        while True:
            # Keep connection alive, wait for disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)