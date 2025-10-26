"""
Response Enhancer for Transparency System
Adds structured metadata to chat responses including actions, citations, and code changes
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseEnhancer:
    """Enhances chat responses with transparency metadata"""

    def __init__(self):
        self.action_history = []
        self.citations = []
        self.code_changes = []

    def track_action(
        self,
        action_type: str,
        description: str,
        detail: Optional[str] = None,
        status: str = "completed",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Track an action that was performed"""
        action = {
            "action_id": str(uuid.uuid4())[:8],
            "action_type": action_type,
            "description": description,
            "detail": detail,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.action_history.append(action)
        return action

    def add_citation(
        self,
        source: str,
        confidence: float,
        collection: str,
        text: Optional[str] = None
    ) -> int:
        """Add a citation and return its ID"""
        citation_id = len(self.citations) + 1
        citation = {
            "citation_id": citation_id,
            "source": source,
            "confidence": confidence,
            "collection": collection,
            "text": text[:200] if text else None,
            "timestamp": datetime.now().isoformat()
        }
        self.citations.append(citation)
        return citation_id

    def add_code_change(
        self,
        file_path: str,
        diff: str,
        description: str
    ) -> str:
        """Add a code change preview"""
        change_id = str(uuid.uuid4())[:8]
        change = {
            "change_id": change_id,
            "file_path": file_path,
            "diff": diff,
            "description": description,
            "status": "pending",
            "lines_added": diff.count('\n+') if diff else 0,
            "lines_removed": diff.count('\n-') if diff else 0
        }
        self.code_changes.append(change)
        return change_id

    def enhance_response(
        self,
        response: Dict[str, Any],
        include_actions: bool = True,
        include_citations: bool = True,
        include_code_changes: bool = True
    ) -> Dict[str, Any]:
        """Enhance a response with transparency metadata"""

        # Start with original response
        enhanced = response.copy()

        # Add action history
        if include_actions and self.action_history:
            enhanced["actions"] = self.action_history.copy()

        # Add citations
        if include_citations and self.citations:
            enhanced["citations"] = self.citations.copy()

            # If response text contains citation markers, ensure they're preserved
            if "response" in enhanced and self.citations:
                response_text = enhanced["response"]
                for citation in self.citations:
                    cit_id = citation["citation_id"]
                    # Add citation markers if not already present
                    marker = f"[{cit_id}]"
                    if marker not in response_text and citation.get("text"):
                        # Find where to insert citation based on relevance
                        # This is simplified - in production would use NLP
                        pass

        # Add code changes
        if include_code_changes and self.code_changes:
            enhanced["code_changes"] = self.code_changes.copy()

        # Add pending approvals if there are risky operations
        pending_approvals = []
        for change in self.code_changes:
            if change.get("status") == "pending":
                if any(risky in change.get("file_path", "")
                      for risky in [".env", "config", "settings", "secrets"]):
                    pending_approvals.append({
                        "operation": f"Modify {change['file_path']}",
                        "risk_level": "high",
                        "change_id": change["change_id"]
                    })

        if pending_approvals:
            enhanced["pending_approvals"] = pending_approvals

        # Add transparency metadata
        enhanced["transparency"] = {
            "version": "1.0",
            "action_count": len(self.action_history),
            "citation_count": len(self.citations),
            "code_change_count": len(self.code_changes),
            "has_approvals": len(pending_approvals) > 0
        }

        return enhanced

    def clear(self):
        """Clear all tracked data"""
        self.action_history.clear()
        self.citations.clear()
        self.code_changes.clear()

    def track_memory_operation(
        self,
        operation: str,
        query: Optional[str] = None,
        collection: Optional[str] = None,
        results_count: int = 0
    ):
        """Track a memory system operation"""
        descriptions = {
            "search": f"Searched memory: {query[:50]}..." if query else "Searched memory",
            "store": f"Stored in {collection}" if collection else "Stored memory",
            "get_failures": f"Checked failures for: {query}" if query else "Checked failures",
            "get_recent": "Retrieved recent memories",
            "pattern_match": f"Found pattern match for: {query[:30]}..." if query else "Pattern matching"
        }

        detail = None
        if results_count > 0:
            detail = f"{results_count} results found"

        return self.track_action(
            action_type=f"memory_{operation}",
            description=descriptions.get(operation, f"Memory operation: {operation}"),
            detail=detail,
            metadata={"collection": collection, "query": query}
        )

    def format_response_with_citations(self, response_text: str) -> str:
        """Add citation markers to response text"""
        if not self.citations:
            return response_text

        # This is a simplified implementation
        # In production, would use NLP to find relevant positions
        formatted = response_text

        # Add citation numbers at the end of relevant sentences
        for citation in self.citations:
            cit_id = citation["citation_id"]
            marker = f"[{cit_id}]"

            # Simple heuristic: add citation at end if not present
            if marker not in formatted:
                # Find a good position (end of sentence mentioning key terms)
                # Simplified: just append to end for now
                if citation.get("text") and len(citation["text"]) > 20:
                    # Would normally use similarity matching here
                    pass

        return formatted