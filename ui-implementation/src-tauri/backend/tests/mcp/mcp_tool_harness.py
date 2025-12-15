"""
MCP Tool Test Harness

Provides isolated test functions that simulate MCP tool calls without
requiring the full MCP server infrastructure. This allows testing the
tool handler logic directly.

Design based on Clean Architecture (books collection reference):
- Isolates tool handlers from MCP protocol
- Enables testing without network/server overhead
- Matches the interface contracts from main.py
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Try to import the real ActionOutcome, fall back to local definition
try:
    from modules.memory.unified_memory_system import ActionOutcome
except ImportError:
    @dataclass
    class ActionOutcome:
        """Mirror of the ActionOutcome from modules/memory/types.py"""
        action_type: str
        context_type: str
        outcome: str
        timestamp: datetime = field(default_factory=datetime.now)
        action_params: Dict[str, Any] = field(default_factory=dict)
        doc_id: Optional[str] = None
        collection: Optional[str] = None
        failure_reason: Optional[str] = None
        success_context: Optional[Dict[str, Any]] = None
        chain_position: int = 0
        chain_length: int = 1
        caused_final_outcome: bool = True

# Simulate the module-level caches from main.py
_test_search_cache: Dict[str, Dict] = {}
_test_action_cache: Dict[str, List[Dict]] = {}
_test_first_tool_call: set = set()


# =============================================================================
# Schema Validation
# =============================================================================

TOOL_SCHEMAS = {
    "search_memory": {
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "collections": {"type": "array"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "metadata": {"type": "object"}
        }
    },
    "add_to_memory_bank": {
        "required": ["content"],
        "properties": {
            "content": {"type": "string"},
            "tags": {"type": "array"},
            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        }
    },
    "update_memory": {
        "required": ["old_content", "new_content"],
        "properties": {
            "old_content": {"type": "string"},
            "new_content": {"type": "string"}
        }
    },
    "archive_memory": {
        "required": ["content"],
        "properties": {
            "content": {"type": "string"}
        }
    },
    "get_context_insights": {
        "required": ["query"],
        "properties": {
            "query": {"type": "string"}
        }
    },
    "record_response": {
        "required": ["key_takeaway"],
        "properties": {
            "key_takeaway": {"type": "string"},
            "outcome": {
                "type": "string",
                "enum": ["worked", "failed", "partial", "unknown"]
            }
        }
    },
    "score_response": {
        "required": ["outcome"],
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["worked", "failed", "partial", "unknown"]
            }
        }
    }
}


def validate_tool_args(tool_name: str, arguments: Dict[str, Any]) -> bool:
    """Validate tool arguments against schema."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return False

    # Check required params
    for required in schema.get("required", []):
        if required not in arguments:
            return False

    # Check enum constraints
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            if "enum" in prop_schema:
                if value not in prop_schema["enum"]:
                    return False

    return True


# =============================================================================
# Cache Management (mirrors main.py caches)
# =============================================================================

def get_search_cache(session_id: str) -> Optional[Dict]:
    """Get cached search results for session."""
    return _test_search_cache.get(session_id)


def set_search_cache(session_id: str, cache: Dict):
    """Set search cache for session."""
    _test_search_cache[session_id] = cache


def clear_search_cache(session_id: str):
    """Clear search cache for session."""
    if session_id in _test_search_cache:
        del _test_search_cache[session_id]


def get_action_cache(session_id: str) -> List[Dict]:
    """Get cached actions for session."""
    return _test_action_cache.get(session_id, [])


def set_action_cache(session_id: str, actions: List[Dict]):
    """Set action cache for session."""
    _test_action_cache[session_id] = actions


def clear_action_cache(session_id: str):
    """Clear action cache for session."""
    if session_id in _test_action_cache:
        del _test_action_cache[session_id]


def clear_all_caches():
    """Clear all test caches."""
    _test_search_cache.clear()
    _test_action_cache.clear()
    _test_first_tool_call.clear()


# =============================================================================
# Tool Handlers (isolated from MCP server)
# =============================================================================

async def call_search_memory(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default",
    context_type: str = "general"
) -> str:
    """
    Simulate search_memory tool call.

    Mirrors the handler logic from main.py lines 1048-1191.
    """
    query = arguments.get("query")
    collections = arguments.get("collections", None)
    if collections == ["all"]:
        collections = None
    limit = int(arguments.get("limit", 5)) if arguments.get("limit") else 5
    metadata = arguments.get("metadata", None)

    # Check initialization
    if not memory.initialized:
        return f"No results found for '{query}' in all collections.\n\nNote: Memory system is empty."

    # Call search
    results = await memory.search(
        query=query,
        collections=collections,
        limit=limit,
        metadata_filters=metadata
    )

    # Cache doc_ids
    cached_doc_ids = []
    result_collections = set()
    if results:
        for r in results:
            metadata_r = r.get('metadata', {})
            collection = r.get('collection') or metadata_r.get('collection', 'unknown')
            doc_id = r.get('doc_id') or r.get('id')
            result_collections.add(collection)
            if doc_id:
                cached_doc_ids.append(doc_id)

    _test_search_cache[session_id] = {
        "doc_ids": cached_doc_ids,
        "query": query,
        "collections": list(result_collections),
        "timestamp": datetime.now()
    }

    # Track action
    collections_used = list(result_collections) if result_collections else (collections if collections else ["all"])
    if session_id not in _test_action_cache:
        _test_action_cache[session_id] = []

    for coll in collections_used:
        action = {
            "action_type": "search_memory",
            "context_type": context_type,
            "outcome": "unknown",
            "action_params": {"query": query, "limit": limit},
            "collection": coll if coll != "all" else None,
            "doc_id": cached_doc_ids[0] if cached_doc_ids else None
        }
        _test_action_cache[session_id].append(action)

    # Format results
    if not results:
        return f"No results found for '{query}'."

    text = f"Found {len(results)} result(s) for '{query}':\n\n"
    for i, r in enumerate(results[:5], 1):
        content = r.get('content') or r.get('text', '')
        metadata_r = r.get('metadata', {})
        collection = r.get('collection') or metadata_r.get('collection', 'unknown')
        score = metadata_r.get('score')
        uses = metadata_r.get('uses', 0)

        meta_parts = []
        if score is not None:
            meta_parts.append(f"score:{score:.2f}")
        if uses > 0:
            meta_parts.append(f"uses:{uses}")

        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        text += f"{i}. [{collection}]{meta_line} {content}\n\n"

    return text


async def call_add_to_memory_bank(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default",
    context_type: str = "general"
) -> str:
    """
    Simulate add_to_memory_bank tool call.

    Mirrors the handler logic from main.py lines 1193-1219.
    """
    content = arguments.get("content")
    tags = arguments.get("tags", [])
    importance = arguments.get("importance", 0.7)
    confidence = arguments.get("confidence", 0.7)

    doc_id = await memory.store_memory_bank(
        text=content,
        tags=tags,
        importance=importance,
        confidence=confidence
    )

    # Track action
    if session_id not in _test_action_cache:
        _test_action_cache[session_id] = []

    action = {
        "action_type": "create_memory",
        "context_type": context_type,
        "outcome": "unknown",
        "action_params": {"content_preview": content[:50]},
        "doc_id": doc_id,
        "collection": "memory_bank"
    }
    _test_action_cache[session_id].append(action)

    return f"Added to memory bank (ID: {doc_id})"


async def call_update_memory(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default",
    context_type: str = "general",
    return_error_flag: bool = False
) -> str | Tuple[str, bool]:
    """
    Simulate update_memory tool call.

    Mirrors the handler logic from main.py lines 1221-1256.
    """
    old_content = arguments.get("old_content", "")
    new_content = arguments.get("new_content", "")

    results = await memory.search_memory_bank(query=old_content, limit=1, include_archived=False)

    if results:
        doc_id = results[0].get("id")
        await memory.update_memory_bank(doc_id=doc_id, new_text=new_content, reason="mcp_update")

        # Track action
        if session_id not in _test_action_cache:
            _test_action_cache[session_id] = []

        action = {
            "action_type": "update_memory",
            "context_type": context_type,
            "outcome": "unknown",
            "action_params": {"new_content_preview": new_content[:50]},
            "doc_id": doc_id,
            "collection": "memory_bank"
        }
        _test_action_cache[session_id].append(action)

        text = f"Updated memory (ID: {doc_id})"
        if return_error_flag:
            return text, False
        return text
    else:
        text = "Memory not found for update"
        if return_error_flag:
            return text, True
        return text


async def call_archive_memory(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default",
    context_type: str = "general",
    return_error_flag: bool = False
) -> str | Tuple[str, bool]:
    """
    Simulate archive_memory tool call.

    Mirrors the handler logic from main.py lines 1258-1292.
    """
    content = arguments.get("content", "")

    results = await memory.search_memory_bank(query=content, limit=1, include_archived=False)

    if results:
        doc_id = results[0].get("id")
        await memory.archive_memory_bank(doc_id=doc_id, reason="mcp_archive")

        # Track action
        if session_id not in _test_action_cache:
            _test_action_cache[session_id] = []

        action = {
            "action_type": "archive_memory",
            "context_type": context_type,
            "outcome": "unknown",
            "action_params": {"content_preview": content[:50]},
            "doc_id": doc_id,
            "collection": "memory_bank"
        }
        _test_action_cache[session_id].append(action)

        text = f"Archived memory (ID: {doc_id})"
        if return_error_flag:
            return text, False
        return text
    else:
        text = "Memory not found for archiving"
        if return_error_flag:
            return text, True
        return text


async def call_get_context_insights(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default",
    data_path=None
) -> str:
    """
    Simulate get_context_insights tool call.

    Mirrors the handler logic from main.py lines 1294-1457.
    """
    query = arguments.get("query", "")

    # Detect context type
    context_type = await memory.detect_context_type(
        system_prompts=[],
        recent_messages=[]
    )

    # Get conversation context
    org_context = await memory.analyze_conversation_context(
        current_message=query,
        recent_conversation=[],
        conversation_id=session_id
    )

    # Get action effectiveness stats (simplified - real implementation iterates over actions/collections)
    action_stats = {}
    try:
        # Try calling with required args (real system needs context_type and action_type)
        for action in ["search_memory", "create_memory"]:
            stats = memory.get_action_effectiveness("general", action)
            if stats:
                action_stats[f"{action}|general"] = stats
    except (TypeError, AttributeError):
        # Mock mode - just return empty stats
        pass

    # Get tier recommendations (requires concepts param in real system)
    tier_recs = []
    try:
        tier_recs = memory.get_tier_recommendations(concepts=["general"])
    except (TypeError, AttributeError):
        pass

    # Get relevant facts
    facts = await memory.get_facts_for_entities([])

    # Build response
    text = f"Known Context for '{query}':\n\n"

    if facts:
        text += "**Memory Bank:**\n"
        for fact in facts:
            text += f"- {fact.get('text', '')}\n"
        text += "\n"

    if tier_recs:
        text += f"**Recommended Collections:** {', '.join(tier_recs)}\n\n"

    if action_stats:
        text += "**Tool Stats:**\n"
        for key, stats in action_stats.items():
            text += f"- {key}: {stats.get('success_rate', 0):.0%} success\n"

    return text


async def call_record_response(
    memory,
    arguments: Dict[str, Any],
    session_id: str = "default"
) -> str:
    """
    Simulate record_response tool call.

    Mirrors the handler logic from main.py lines 1459-1593.
    """
    key_takeaway = arguments.get("key_takeaway", "")
    outcome = arguments.get("outcome", "unknown")

    # Map outcome to initial score
    score_map = {
        "worked": 0.7,
        "failed": 0.2,
        "partial": 0.55,
        "unknown": 0.5
    }
    initial_score = score_map.get(outcome, 0.5)

    # Store takeaway
    doc_id = await memory.store(
        text=key_takeaway,
        collection="working",
        metadata={"score": initial_score, "outcome": outcome}
    )

    # Score cached memories
    cached = _test_search_cache.get(session_id, {})
    cached_doc_ids = cached.get("doc_ids", [])
    scored_count = 0

    for cdoc_id in cached_doc_ids:
        await memory.record_outcome(cdoc_id, outcome)
        scored_count += 1

    # Score cached actions
    actions = _test_action_cache.get(session_id, [])
    for action_dict in actions:
        action_dict["outcome"] = outcome
        # Convert dict to ActionOutcome if real memory system expects it
        try:
            action_obj = ActionOutcome(
                action_type=action_dict["action_type"],
                context_type=action_dict["context_type"],
                outcome=action_dict["outcome"],
                action_params=action_dict.get("action_params"),
                collection=action_dict.get("collection"),
                doc_id=action_dict.get("doc_id")
            )
            await memory.record_action_outcome(action_obj)
        except (TypeError, AttributeError):
            # Mock mode - just pass the dict
            await memory.record_action_outcome(action_dict)

    # Update KG routing
    query = cached.get("query", "")
    collections = cached.get("collections", [])
    for coll in collections:
        await memory._update_kg_routing(query, coll, outcome)

    # Clear caches
    clear_search_cache(session_id)
    clear_action_cache(session_id)

    return f"Recorded: {key_takeaway[:50]}... (scored {scored_count} memories)"


# =============================================================================
# Test Utilities
# =============================================================================

def create_mock_results(count: int = 3, collection: str = "working") -> List[Dict]:
    """Create mock search results for testing."""
    return [
        {
            "id": f"doc_{i}",
            "doc_id": f"doc_{i}",
            "content": f"Test result {i}",
            "text": f"Test result {i}",
            "collection": collection,
            "metadata": {
                "score": 0.9 - i * 0.1,
                "uses": 5 - i,
                "timestamp": datetime.now().isoformat()
            }
        }
        for i in range(count)
    ]
