"""
Action Status WebSocket Manager
Handles real-time action status updates for transparency system
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Set, List, Any
from fastapi import WebSocket
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class ActionStatusManager:
    """Manages WebSocket connections for action status updates"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.action_history: Dict[str, List[Dict]] = defaultdict(list)
        self.citations: Dict[str, List[Dict]] = defaultdict(list)
        self.pending_approvals: Dict[str, List[Dict]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a WebSocket connection for a session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Action WebSocket connected for session {session_id}")

        # Send connection confirmation
        await self.send_status(session_id, {
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, session_id: str):
        """Remove a WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Action WebSocket disconnected for session {session_id}")

    async def send_status(self, session_id: str, data: dict):
        """Send status update to a specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send status to session {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast_action_status(
        self,
        session_id: str,
        action_type: str,
        description: str,
        status: str = "executing",
        detail: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Broadcast an action status update to the session"""
        action_id = str(uuid.uuid4())[:8]

        action_data = {
            "type": "action-status-update",
            "action_id": action_id,
            "action_type": action_type,
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

        if detail:
            action_data["detail"] = detail
        if metadata:
            action_data["metadata"] = metadata

        # Store in history
        self.action_history[session_id].append(action_data)

        # Send to client
        await self.send_status(session_id, action_data)

        # Auto-complete after delay for executing actions
        if status == "executing":
            asyncio.create_task(self._auto_complete_action(session_id, action_id))

    async def _auto_complete_action(self, session_id: str, action_id: str, delay: float = 2.0):
        """Auto-complete an action after a delay"""
        await asyncio.sleep(delay)
        await self.send_status(session_id, {
            "type": "action-status-update",
            "action_id": action_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        })

    async def send_citation(
        self,
        session_id: str,
        citation_id: int,
        source: str,
        confidence: float,
        collection: str,
        text: Optional[str] = None
    ):
        """Send a citation reference"""
        citation_data = {
            "type": "citation",
            "citation_id": citation_id,
            "source": source,
            "confidence": confidence,
            "collection": collection,
            "timestamp": datetime.now().isoformat()
        }

        if text:
            citation_data["text"] = text[:200]  # Truncate for preview

        self.citations[session_id].append(citation_data)
        await self.send_status(session_id, citation_data)

    async def send_code_change(
        self,
        session_id: str,
        file_path: str,
        diff: str,
        change_id: str,
        description: str
    ):
        """Send a code change preview"""
        change_data = {
            "type": "code-change",
            "change_id": change_id,
            "file_path": file_path,
            "diff": diff,
            "description": description,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }

        await self.send_status(session_id, change_data)

    async def request_approval(
        self,
        session_id: str,
        operations: List[Dict],
        risk_level: str = "low"
    ):
        """Request approval for operations"""
        approval_id = str(uuid.uuid4())[:8]
        approval_data = {
            "type": "approval-request",
            "approval_id": approval_id,
            "operations": operations,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }

        self.pending_approvals[session_id].append(approval_data)
        await self.send_status(session_id, approval_data)

        return approval_id

    def get_action_summary(self, session_id: str, message_id: Optional[str] = None) -> Dict:
        """Get summary of actions for a session or specific message"""
        actions = self.action_history.get(session_id, [])
        citations = self.citations.get(session_id, [])

        # Filter by message_id if provided
        if message_id:
            # This would require tracking message_id with actions
            # For now, return recent actions
            actions = actions[-10:] if len(actions) > 10 else actions
            citations = citations[-5:] if len(citations) > 5 else citations

        return {
            "actions": actions,
            "citations": citations,
            "action_count": len(actions),
            "citation_count": len(citations)
        }

    def clear_session_history(self, session_id: str):
        """Clear action history for a session"""
        if session_id in self.action_history:
            del self.action_history[session_id]
        if session_id in self.citations:
            del self.citations[session_id]
        if session_id in self.pending_approvals:
            del self.pending_approvals[session_id]

# Global manager instance
action_manager = ActionStatusManager()