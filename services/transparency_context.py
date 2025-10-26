"""
Transparency Context - Clean integration for tracking AI operations
Passed through call chain to collect operations without coupling
"""

import uuid
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SensitiveDataFilter:
    """Filter for removing sensitive data from tracked information"""

    # Patterns for common sensitive data
    PATTERNS = [
        # API Keys and tokens
        (r'(api[_-]?key|apikey|api_secret|secret[_-]?key)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})', '[API_KEY_REDACTED]'),
        (r'(token|access[_-]?token|auth[_-]?token)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})', '[TOKEN_REDACTED]'),
        (r'Bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer [REDACTED]'),

        # Passwords
        (r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?[^\s"\',]+', '[PASSWORD_REDACTED]'),

        # AWS Keys
        (r'AKIA[0-9A-Z]{16}', '[AWS_ACCESS_KEY_REDACTED]'),
        (r'[0-9a-zA-Z/+=]{40}', '[AWS_SECRET_KEY_REDACTED]'),

        # Private keys
        (r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', '[PRIVATE_KEY_REDACTED]'),

        # Credit cards
        (r'\b(?:\d[ -]*?){13,19}\b', '[CREDIT_CARD_REDACTED]'),

        # Social Security Numbers
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),

        # Email addresses (optional - may want to keep for debugging)
        # (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_REDACTED]'),

        # Database connection strings
        (r'(mongodb|postgres|mysql|redis|amqp)://[^\s]+', '[DB_CONNECTION_REDACTED]'),

        # JWT tokens
        (r'eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+', '[JWT_REDACTED]'),
    ]

    @classmethod
    def filter_text(cls, text: str) -> str:
        """Remove sensitive data from text"""
        if not text:
            return text

        filtered_text = text
        for pattern, replacement in cls.PATTERNS:
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)

        return filtered_text

    @classmethod
    def filter_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from dictionary"""
        if not data:
            return data

        filtered = {}
        sensitive_keys = {'password', 'secret', 'token', 'key', 'auth', 'credential', 'api_key', 'apikey'}

        for key, value in data.items():
            # Check if key name suggests sensitive data
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = '[REDACTED]'
            elif isinstance(value, str):
                filtered[key] = cls.filter_text(value)
            elif isinstance(value, dict):
                filtered[key] = cls.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [cls.filter_text(item) if isinstance(item, str) else item for item in value]
            else:
                filtered[key] = value

        return filtered


@dataclass
class Action:
    """Single AI action/operation with reasoning"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: str = ""
    description: str = ""
    detail: Optional[str] = None
    status: str = "completed"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Chain of thought fields
    reasoning: Optional[str] = None
    confidence: float = 1.0
    alternatives_considered: List[str] = field(default_factory=list)
    decision_rationale: Optional[str] = None
    performance: Dict[str, Any] = field(default_factory=dict)
    parallel_with: List[str] = field(default_factory=list)


@dataclass
class Citation:
    """Memory citation reference with influence tracking"""
    citation_id: int
    source: str
    confidence: float
    collection: str
    text: Optional[str] = None
    doc_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Influence tracking
    influence: Dict[str, Any] = field(default_factory=dict)
    user_preference_match: bool = False


@dataclass
class CodeChange:
    """Proposed code change"""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    diff: str = ""
    description: str = ""
    status: str = "pending"
    risk_level: str = "low"

@dataclass
class ThinkingEvent:
    """Chain of thought thinking event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""  # query_analysis, tool_selection, memory_search, response_planning
    thought: str = ""
    reasoning: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    decision: Optional[str] = None
    confidence: float = 0.0
    parallel_with: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TransparencyContext:
    """
    Context object passed through operations to collect transparency data.
    Clean, minimal, and doesn't create coupling between components.
    Tracks LLM thinking process and tool execution for transparency.
    """

    def __init__(self, session_id: Optional[str] = None, enable_tracking: bool = True):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.enable_tracking = enable_tracking
        self.actions: List[Action] = []
        self.citations: List[Citation] = []
        self.code_changes: List[CodeChange] = []
        self.thinking_events: List[ThinkingEvent] = []
        self._citation_counter = 0
        self._parallel_operations: Dict[str, List[str]] = {}  # parent_id -> [child_ids]

    def track_action(
        self,
        action_type: str,
        description: str,
        detail: Optional[str] = None,
        status: str = "completed",
        metadata: Optional[Dict] = None,
        reasoning: Optional[str] = None,
        confidence: float = 1.0,
        alternatives: Optional[List[str]] = None,
        decision_rationale: Optional[str] = None
    ) -> Action:
        """Track an action performed by the AI with reasoning"""
        if not self.enable_tracking:
            return None

        # Filter sensitive data
        description = SensitiveDataFilter.filter_text(description)
        detail = SensitiveDataFilter.filter_text(detail) if detail else None
        reasoning = SensitiveDataFilter.filter_text(reasoning) if reasoning else None
        decision_rationale = SensitiveDataFilter.filter_text(decision_rationale) if decision_rationale else None
        metadata = SensitiveDataFilter.filter_dict(metadata) if metadata else {}

        action = Action(
            action_type=action_type,
            description=description,
            detail=detail,
            status=status,
            metadata=metadata,
            reasoning=reasoning,
            confidence=confidence,
            alternatives_considered=alternatives or [],
            decision_rationale=decision_rationale
        )
        self.actions.append(action)
        logger.debug(f"Tracked action: {action_type} - {description}")
        return action

    def track_memory_search(
        self,
        query: str,
        collections: List[str],
        results_count: int,
        confidence_scores: Optional[List[float]] = None,
        reasoning: Optional[str] = None
    ) -> Action:
        """Track a memory search operation with reasoning"""
        collections_str = ", ".join(collections) if collections else "all"
        # Don't generate synthetic reasoning - let it come from the LLM

        return self.track_action(
            action_type="memory_search",
            description=f"Searched: {query[:50]}{'...' if len(query) > 50 else ''}",
            detail=f"{results_count} results from {collections_str}",
            metadata={
                "query": query,
                "collections": collections,
                "results_count": results_count,
                "avg_confidence": sum(confidence_scores)/len(confidence_scores) if confidence_scores else None,
                "relevance_scores": confidence_scores[:5] if confidence_scores else []
            },
            reasoning=reasoning,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            alternatives=["web_search", "documentation_lookup"] if results_count < 3 else [],
            decision_rationale="Local memory has recent relevant context" if results_count > 0 else "No local context found"
        )

    def track_memory_store(
        self,
        text: str,
        collection: str,
        doc_id: str
    ) -> Action:
        """Track a memory store operation"""
        return self.track_action(
            action_type="memory_store",
            description=f"Stored in {collection}",
            detail=f"{text[:50]}{'...' if len(text) > 50 else ''}",
            metadata={
                "collection": collection,
                "doc_id": doc_id,
                "text_length": len(text)
            }
        )

    def add_citation(
        self,
        source: str,
        confidence: float,
        collection: str,
        text: Optional[str] = None,
        doc_id: Optional[str] = None,
        influence_reason: Optional[str] = None
    ) -> int:
        """Add a citation from memory with influence tracking"""
        if not self.enable_tracking:
            return 0

        # Filter sensitive data
        source = SensitiveDataFilter.filter_text(source)
        text = SensitiveDataFilter.filter_text(text) if text else None
        influence_reason = SensitiveDataFilter.filter_text(influence_reason) if influence_reason else None

        self._citation_counter += 1

        # Auto-generate influence tracking
        influence = {
            "how_used": influence_reason or "Applied to current response",
            "impact_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
            "confidence_boost": confidence * 0.2
        }

        if collection == "patterns":
            influence["specific_application"] = "Applied proven solution pattern"
        elif collection == "history":
            influence["specific_application"] = "Used previous conversation context"
        else:
            influence["specific_application"] = "Referenced for context"

        citation = Citation(
            citation_id=self._citation_counter,
            source=source,
            confidence=confidence,
            collection=collection,
            text=text[:200] if text else None,
            doc_id=doc_id,
            influence=influence,
            user_preference_match=collection == "patterns"  # Patterns match user preferences
        )
        self.citations.append(citation)
        return self._citation_counter

    def track_tool_use(
        self,
        tool_name: str,
        operation: str,
        target: Optional[str] = None,
        result: Optional[str] = None
    ) -> Action:
        """Track usage of a tool (file ops, code execution, etc)"""
        # Filter sensitive data
        operation = SensitiveDataFilter.filter_text(operation)
        target = SensitiveDataFilter.filter_text(target) if target else None
        result = SensitiveDataFilter.filter_text(result) if result else None

        descriptions = {
            "read_file": f"Read file: {target}",
            "write_file": f"Write file: {target}",
            "execute_code": f"Executed {target} code",
            "run_command": f"Ran command: {operation[:30]}",
            "analyze_project": f"Analyzed: {target}",
            "run_tests": f"Ran tests: {target}"
        }

        return self.track_action(
            action_type=f"tool_{tool_name}",
            description=descriptions.get(tool_name, f"Used {tool_name}"),
            detail=result[:100] if result else None,
            metadata={
                "tool": tool_name,
                "operation": operation,
                "target": target
            }
        )

    def track_background_process(
        self,
        process: str,
        description: str,
        metadata: Optional[Dict] = None
    ) -> Action:
        """Track background processes (auto-promotion, embedding, etc)"""
        return self.track_action(
            action_type=f"background_{process}",
            description=description,
            status="executing",
            metadata=metadata
        )

    def add_code_change(
        self,
        file_path: str,
        diff: str,
        description: str,
        risk_level: str = "low"
    ) -> str:
        """Add a proposed code change"""
        if not self.enable_tracking:
            return ""

        # Filter sensitive data from diff and description
        diff = SensitiveDataFilter.filter_text(diff)
        description = SensitiveDataFilter.filter_text(description)

        change = CodeChange(
            file_path=file_path,
            diff=diff,
            description=description,
            risk_level=risk_level
        )
        self.code_changes.append(change)
        return change.change_id

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked operations with full transparency"""
        return {
            "session_id": self.session_id,
            "actions": [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "description": a.description,
                    "detail": a.detail,
                    "status": a.status,
                    "timestamp": a.timestamp,
                    "metadata": a.metadata,
                    # Chain of thought fields
                    "reasoning": a.reasoning,
                    "confidence": a.confidence,
                    "alternatives_considered": a.alternatives_considered,
                    "decision_rationale": a.decision_rationale,
                    "performance": a.performance,
                    "parallel_with": a.parallel_with
                }
                for a in self.actions
            ],
            "citations": [
                {
                    "citation_id": c.citation_id,
                    "source": c.source,
                    "confidence": c.confidence,
                    "collection": c.collection,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    # Chain of thought fields
                    "influence": c.influence,
                    "user_preference_match": c.user_preference_match
                }
                for c in self.citations
            ],
            "code_changes": [
                {
                    "change_id": cc.change_id,
                    "file_path": cc.file_path,
                    "diff": cc.diff,
                    "description": cc.description,
                    "status": cc.status,
                    "risk_level": cc.risk_level
                }
                for cc in self.code_changes
            ],
            "thinking_events": [
                {
                    "event_id": t.event_id,
                    "category": t.category,
                    "thought": t.thought,
                    "reasoning": t.reasoning,
                    "alternatives": t.alternatives,
                    "decision": t.decision,
                    "confidence": t.confidence,
                    "parallel_with": t.parallel_with,
                    "timestamp": t.timestamp
                }
                for t in self.thinking_events
            ],
            "thinking_tree": self.get_thinking_tree(),
            "parallel_operations": self._parallel_operations,
            "transparency": {
                "version": "3.0",  # Transparency context version
                "action_count": len(self.actions),
                "citation_count": len(self.citations),
                "code_change_count": len(self.code_changes),
                "thinking_event_count": len(self.thinking_events),
                "tracking_enabled": self.enable_tracking,
                "glass_box_enabled": True
            }
        }

    def format_response_with_citations(self, response_text: str) -> str:
        """Add citation markers to response text"""
        if not self.citations:
            return response_text

        # For now, append citations at the end
        # In future, could use NLP to insert at relevant positions
        if self.citations:
            # Add subtle citation markers
            for citation in self.citations[:3]:  # Limit to top 3
                response_text += f" [{citation.citation_id}]"

        return response_text

    def track_thinking(
        self,
        category: str,
        thought: str,
        reasoning: str,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0
    ) -> ThinkingEvent:
        """Track AI's chain of thought process"""
        if not self.enable_tracking:
            return None

        # Filter sensitive data
        thought = SensitiveDataFilter.filter_text(thought)
        reasoning = SensitiveDataFilter.filter_text(reasoning)

        event = ThinkingEvent(
            category=category,
            thought=thought,
            reasoning=reasoning,
            alternatives=alternatives or [],
            confidence=confidence
        )
        self.thinking_events.append(event)
        logger.debug(f"Tracked thinking: {category} - {thought[:50]}...")
        return event

    def track_decision(
        self,
        decision_type: str,
        chosen: str,
        alternatives: List[Dict[str, Any]],
        rationale: str,
        confidence: float
    ) -> ThinkingEvent:
        """Track a decision point in the chain of thought"""
        return self.track_thinking(
            category=f"decision_{decision_type}",
            thought=f"Deciding on {decision_type}",
            reasoning=rationale,
            alternatives=alternatives,
            confidence=confidence
        )

    def track_parallel_operation(
        self,
        operation_id: str,
        parent_id: Optional[str],
        operation_type: str,
        status: str = "started"
    ) -> None:
        """Track parallel operations for visualization"""
        if not self.enable_tracking:
            return

        if parent_id:
            if parent_id not in self._parallel_operations:
                self._parallel_operations[parent_id] = []
            self._parallel_operations[parent_id].append(operation_id)

        # Find related operations running in parallel
        parallel_ops = []
        for parent, children in self._parallel_operations.items():
            if operation_id in children:
                parallel_ops.extend([c for c in children if c != operation_id])

        # Update the action or thinking event with parallel info
        for action in self.actions:
            if action.action_id == operation_id:
                action.parallel_with = parallel_ops
                break

        for event in self.thinking_events:
            if event.event_id == operation_id:
                event.parallel_with = parallel_ops
                break

        logger.debug(f"Tracked parallel op: {operation_id} (parent: {parent_id}, type: {operation_type})")

    def get_thinking_tree(self) -> Dict[str, Any]:
        """Get structured thinking tree for visualization"""
        tree = {
            "root": "User Query",
            "branches": []
        }

        # Group thinking events by category
        categories = {}
        for event in self.thinking_events:
            if event.category not in categories:
                categories[event.category] = []
            categories[event.category].append({
                "id": event.event_id,
                "thought": event.thought,
                "reasoning": event.reasoning,
                "confidence": event.confidence,
                "alternatives": event.alternatives,
                "parallel_with": event.parallel_with,
                "timestamp": event.timestamp
            })

        # Build tree structure
        for category, events in categories.items():
            branch = {
                "category": category,
                "icon": self._get_category_icon(category),
                "events": events
            }
            tree["branches"].append(branch)

        return tree

    def _get_category_icon(self, category: str) -> str:
        """Get icon for thinking category"""
        icons = {
            "query_analysis": "🧠",
            "tool_selection": "🛠️",
            "memory_search": "🔍",
            "response_planning": "📋",
            "pattern_recognition": "💡",
            "decision": "⚡"
        }
        for key, icon in icons.items():
            if key in category:
                return icon
        return "🤔"

    def clear(self):
        """Clear all tracked data"""
        self.actions.clear()
        self.citations.clear()
        self.code_changes.clear()
        self.thinking_events.clear()
        self._parallel_operations.clear()
        self._citation_counter = 0