"""

Agent Chat Router - Memory-enhanced chat implementation

Single source of truth for all chat operations with memory context

"""


import logging

import json

import asyncio

import re

import uuid

import secrets

import threading

import os

import time

import math

import yaml

from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple

from datetime import datetime, timedelta

from collections import OrderedDict, defaultdict, defaultdict, defaultdict

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from fastapi.responses import StreamingResponse

from pydantic import BaseModel, validator

from filelock import FileLock


from modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.types import ActionOutcome

from modules.llm.ollama_client import OllamaClient

from services.transparency_context import TransparencyContext

from config.feature_flags import get_flag_manager
from config.model_contexts import get_context_size

from config.model_limits import (
    get_model_limits,
    IterationMetrics,
    estimate_tokens,
    calculate_token_budget,
    smart_truncate,
)
from utils.text_utils import extract_thinking

# Initialize router and logger
logger = logging.getLogger(__name__)
router = APIRouter()

# Cold-start tracking for internal LLM (v0.2.0)
# Auto-injects user profile from Content KG on message 1 of every new conversation
internal_session_message_counter: Dict[str, int] = defaultdict(int)

# Action KG tracking for internal LLM (v0.2.6)
# Caches tool actions until outcome is determined, then scores them
_agent_action_cache: Dict[str, List[ActionOutcome]] = {}

# Injection Protection: Response Validation (Layer 3)
class ResponseValidator:
    """Validates LLM responses for injection artifacts"""

    HIJACK_INDICATORS = [
        r"^(HACK|HACKED|PWNED)$",
        r"^(ignore|disregard|forget)",
        r"I (am now|will now|have been instructed to)",
        r"</?system>",
        r"as a (pirate|robot|assistant)",
        r"\*.*in (pirate|robot) voice\*",
    ]

    @staticmethod
    def is_hijacked(response: str) -> tuple:
        """Returns: (is_hijacked, reason)"""
        import re
        response_stripped = response.strip()
        response_lower = response_stripped.lower()

        # Check suspicious short responses
        if len(response_stripped) < 15:
            suspicious_words = ["hack", "pwned", "error", "unauthorized"]
            if any(word in response_lower for word in suspicious_words):
                return True, "Short suspicious response"

        # Check role change admission
        role_changes = [
            "i am now", "i will now", "i have been instructed",
            "as a pirate", "in pirate voice", "arrr",
            "as requested, i will"
        ]
        if any(phrase in response_lower for phrase in role_changes):
            return True, "Role change detected"

        # Check system tag injection
        if "<system>" in response_lower or "</system>" in response_lower:
            return True, "System tag injection"

        # Check if response is ONLY the hijack payload
        last_sentence = response_stripped.split('.')[-1].strip()
        if last_sentence.upper() in ["HACK", "HACKED", "PWNED"]:
            return True, "Hijack payload detected in final sentence"

        return False, ""

    @staticmethod
    def get_fallback_response() -> str:
        """Safe response when hijack detected"""
        return "I noticed something unusual in my response. Let me try again - how can I help you?"

# Global state for async generation tasks
_generation_tasks: Dict[str, Dict[str, Any]] = {}
_active_tasks: Dict[str, asyncio.Task] = {}  # Track task handles for cancellation
_task_lock = threading.Lock()

# Search cache for retrieved memory scoring (architecture.md line 565)
# Tracks which memories were retrieved during search to score them when outcome is detected
_search_cache: Dict[str, List[str]] = {}  # {conversation_id: [doc_ids]}


def use_dynamic_limits() -> bool:
    """Check if dynamic limits are enabled (default True unless ROAMPAL_CHAIN_STRATEGY=fixed)"""
    return os.getenv('ROAMPAL_CHAIN_STRATEGY', '').lower() != 'fixed'


def _get_event_timestamp():

    """Get current timestamp in milliseconds for event ordering"""

    return int(time.time() * 1000)


def _extract_and_strip_tag(content: str, tag_name: str) -> tuple[str, str]:

    """

    Extract content from XML-style tag and return (tag_content, cleaned_text).

    Handles both <tag> and <tag> formats (DeepSeek vs Claude).


    Args:

        content: Raw text that may contain tags

        tag_name: Tag name without brackets (e.g., 'think', 'status')


    Returns:

        (tag_content, cleaned_text) - extracted tag content and text with tags removed


    Example:

        >>> _extract_and_strip_tag("Hello <status>Thinking...</status> world", "status")

        ("Thinking...", "Hello  world")

    """

    pattern = rf'<(?:{tag_name}|antml:{tag_name})>(.*?)</(?:{tag_name}|antml:{tag_name})>'

    match = re.search(pattern, content, re.DOTALL)


    if match:

        tag_content = match.group(1).strip()

        cleaned = re.sub(pattern, '', content, flags=re.DOTALL).strip()

        return (tag_content, cleaned)


    return ("", content)


def _strip_all_tags(content: str, *tag_names: str) -> str:

    """

    Strip multiple tag types from content (both complete and partial tags).


    Args:

        content: Text to clean

        *tag_names: Tag names to remove (e.g., 'think', 'status')


    Returns:

        Cleaned text with all specified tags removed


    Example:

        >>> _strip_all_tags("<think>reasoning</think> text <status>", "think", "status")

        " text "

    """

    cleaned = content

    for tag_name in tag_names:

        # Remove complete tags and partial/malformed tags

        pattern = rf'</?(?:{tag_name}|antml:{tag_name})[^>]*>'

        cleaned = re.sub(pattern, '', cleaned)

    return cleaned.strip()


# Thinking display feature removed (v0.2.5) - models dont reliably use thinking APIs


def _format_search_results_as_citations(

    search_results: List[Dict],

    max_citations: int = 10

) -> List[Dict]:

    """

    Convert memory search results to UI-friendly citation format.


    Uses exponential decay to convert distance to confidence score (0-1.0).

    Lower distance = higher confidence. Works with any distance range.


    Formula: confidence = exp(-distance / scale_factor)

    - scale_factor=100.0 provides good sensitivity for typical embedding distances (100-500)

    - Adjust scale_factor up for higher confidence scores, down for lower


    Args:

        search_results: List of memory search results with 'distance' field

        max_citations: Maximum number of citations to return


    Returns:

        List of formatted citations with citation_id, source, confidence, collection, text

    """

    citations = []

    for idx, r in enumerate(search_results[:max_citations]):

        distance = r.get("distance", 0.5)

        # Use exponential decay: confidence = e^(-distance/scale)

        # Scale factor of 100.0 works well for ChromaDB L2 distances (typically 100-500)

        CONFIDENCE_SCALE_FACTOR = 100.0

        confidence = math.exp(-distance / CONFIDENCE_SCALE_FACTOR)

        logger.info(f"[CITATIONS] Result {idx+1}: distance={distance:.3f}, confidence={confidence:.2f}, collection={r.get('collection')}")

        citations.append({

            "citation_id": idx + 1,

            "source": r.get("metadata", {}).get("source", r.get("collection", "Memory")),

            "confidence": confidence,

            "collection": r.get("collection", "unknown"),
            # v0.2.8: Full content, no truncation
            "text": r.get("text", "")

        })

    return citations


class AgentChatRequest(BaseModel):

    """Request model for agent chat with input validation"""

    message: str

    conversation_id: Optional[str] = None

    # Mode removed - RoamPal always uses memory
    # File attachments removed - use Document Processor for file uploads


    @validator('message')

    def validate_message(cls, v):

        """Validate message input for security"""

        if not v or not v.strip():

            raise ValueError("Message cannot be empty")

        # Limit message length to prevent DoS

        max_length = int(os.getenv('ROAMPAL_MAX_MESSAGE_LENGTH', '10000'))

        if len(v) > max_length:

            raise ValueError(f"Message exceeds maximum length of {max_length} characters")

        # Remove control characters

        v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', v)

        return v.strip()


    @validator('conversation_id')

    def validate_conversation_id(cls, v):

        """Validate conversation ID format"""

        if v:

            if not re.match(r'^[a-zA-Z0-9_-]+$', v):

                raise ValueError("Invalid conversation ID format")

        return v


class AgentChatResponse(BaseModel):

    """Response model with memory context"""

    response: str

    conversation_id: str

    thinking: Optional[str] = None  # Optional thinking content

    citations: List[Dict[str, Any]]

    mode: str

    message_count: int

    metadata: Optional[Dict[str, Any]] = {}


class AgentChatService:

    """Clean chat service with memory-enhanced responses"""


    def __init__(self, memory: UnifiedMemorySystem, llm: OllamaClient):

        self.memory = memory

        self.llm = llm

        self.flag_manager = get_flag_manager()


        # Model detection and limits (NEW)

        # Use same env var priority as main.py for consistency

        self.model_name = os.getenv('ROAMPAL_LLM_OLLAMA_MODEL') or os.getenv('OLLAMA_MODEL') or 'default'

        self.model_limits = None  # Will be set per message based on task complexity

        self.use_dynamic_limits = use_dynamic_limits()

        logger.info(f"Initialized with model: {self.model_name}, dynamic limits: {self.use_dynamic_limits}")


        # Store conversation history with LRU cache to prevent memory leaks

        self.conversation_histories = OrderedDict()  # conversation_id -> list of messages

        self.max_conversations = 100  # Maximum number of conversations to keep in memory

        self.max_context_messages = 8  # 4 exchanges sent to LLM (limited by context window)

        # Session persistence
        # Use AppData paths, not bundled data folder
        self.sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")

        self.sessions_dir.mkdir(parents=True, exist_ok=True)


        # Load existing conversations from session files on startup

        self._load_conversation_histories()

        # Lock for conversation switching to prevent race conditions

        self.conversation_lock = asyncio.Lock()

        # Per-conversation locks for title generation to prevent duplicates

        self.title_locks: Dict[str, asyncio.Lock] = {}

        # Queue for async JSONL writes

        self.write_queue = asyncio.Queue()

        self.write_task = None

        # Track last promotion time for hourly auto-promotion

        self.last_promotion_time = datetime.now()


        # Personality template cache (for performance)

        self._personality_cache = None

        self._personality_mtime = 0

        self._personality_template_path = Path("backend/templates/personality/active.txt")


    def _validate_chat_model(self) -> Tuple[bool, Optional[str]]:
        """
        Validate if current model can be used for chat.
        Single source of truth for model validation.

        Returns:
            (is_valid, error_message): True with None if valid, False with error message if invalid
        """
        current_model = self.llm.model_name if hasattr(self.llm, 'model_name') else self.model_name

        if not current_model:
            return (False, "No chat model available. Please install a model to start chatting.")

        # Check if model is an embedding model (cannot be used for chat)
        embedding_models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']
        if any(embed in current_model.lower() for embed in embedding_models):
            return (False, "No chat model available. Please install a model to start chatting.")

        return (True, None)

    def _format_citation(self, result: Dict[str, Any], index: int) -> Dict[str, Any]:

        """Format a memory result as a citation"""

        # Memory results have 'text' field, not 'content'
        # v0.2.8: Full content, no truncation
        content_text = result.get('text', result.get('content', ''))

        return {

            "content": content_text,

            "text": content_text,  # UI expects 'text' field

            "source": "Memory",  # UI expects 'source' field

            "collection": result.get('collection', 'working'),  # Default to 'working' collection

            "confidence": result.get('confidence', 0.0),

            "citation_id": index,  # UI expects numeric 'citation_id'

            "metadata": result.get('metadata', {})

        }


    # Mode system removed - RoamPal always uses memory

    async def stream_message(
        self,
        message: str,
        conversation_id: str,
        mode: str = "memory",
        transparency_level: str = "summary",
        app_state: Optional[Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tokens from LLM with memory context.
        Yields: {"type": "token|tool_start|tool_complete|done", "content": ...}
        """
        try:
            # CRITICAL: Always fetch latest llm_client from app_state (handles model switches)
            if app_state and hasattr(app_state, 'llm_client'):
                self.llm = app_state.llm_client
                logger.info(f"[LLM CLIENT REFRESH] Updated to current client: model={getattr(self.llm, 'model_name', 'unknown')}, base_url={getattr(self.llm, 'base_url', 'unknown')}")

            # Validate model BEFORE adding message to history (prevent garbage session files)
            is_valid, error_msg = self._validate_chat_model()
            if not is_valid:
                logger.info(f"Model validation failed: {error_msg}")
                yield {
                    "type": "done",
                    "content": error_msg
                }
                return

            # Initialize conversation history if needed
            if conversation_id not in self.conversation_histories:
                if len(self.conversation_histories) >= self.max_conversations:
                    evicted_id, _ = self.conversation_histories.popitem(last=False)
                    logger.info(f"Evicted old conversation {evicted_id} from memory (LRU)")
                self.conversation_histories[conversation_id] = []
            else:
                # Move to end to mark as recently used
                self.conversation_histories.move_to_end(conversation_id)

            # COLD-START AUTO-TRIGGER (v0.2.0): Track messages and ALWAYS inject user profile on message 1
            internal_session_message_counter[conversation_id] += 1
            current_message = internal_session_message_counter[conversation_id]
            logger.info(f"[COLD-START] Internal LLM message #{current_message} for conversation {conversation_id}")

            # ALWAYS auto-inject user profile on message 1 (no tracking, no conditions)
            if current_message == 1:
                logger.info(f"[COLD-START] Auto-injecting user profile from Content KG on message 1...")
                try:
                    # Get cold-start context from Content KG (or fallback to vector search)
                    context_summary = await asyncio.wait_for(
                        self.memory.get_cold_start_context(limit=5),
                        timeout=25.0
                    )

                    if context_summary:
                        logger.info(f"[COLD-START] Injecting user profile: {context_summary[:100]}...")
                        self.conversation_histories[conversation_id].append({
                            "role": "system",
                            "content": context_summary
                        })
                    else:
                        logger.info(f"[COLD-START] No user profile context available yet")

                except asyncio.TimeoutError:
                    logger.error(f"[COLD-START] Auto-trigger timed out after 25s")
                except Exception as e:
                    logger.error(f"[COLD-START] Auto-trigger error: {e}", exc_info=True)

            # Add user message to history (AFTER auto-trigger injection if any)
            self.conversation_histories[conversation_id].append({
                "role": "user",
                "content": message
            })

            # CONTEXTUAL GUIDANCE: Organic recall + Action-effectiveness (v0.2.1)
            # Merges Content KG insights with Action-Effectiveness KG warnings
            try:
                # Build recent conversation context (last 5 messages)
                recent_conv = [
                    {'role': m['role'], 'content': m['content']}
                    for m in self.conversation_histories[conversation_id][-5:]
                    if m['role'] in ['user', 'assistant']
                ]

                # Detect context type for action-effectiveness guidance
                system_prompts = [m['content'] for m in self.conversation_histories[conversation_id] if m['role'] == 'system']
                context_type = await self.memory.detect_context_type(
                    system_prompts=system_prompts,
                    recent_messages=recent_conv
                )

                # Analyze conversation for organic insights (Content KG)
                org_context = await self.memory.analyze_conversation_context(
                    current_message=message,
                    recent_conversation=recent_conv,
                    conversation_id=conversation_id
                )

                # Check action-effectiveness for tool guidance (Action-Effectiveness KG)
                action_stats = []
                available_actions = ["search_memory", "create_memory", "update_memory", "archive_memory"]
                collections_to_check = [None, "books", "working", "history", "patterns", "memory_bank"]  # Check all 5 tiers + wildcard

                for action in available_actions:
                    # Check across all collections
                    for collection in collections_to_check:
                        stats = self.memory.get_action_effectiveness(context_type, action, collection)
                        # Only surface stats if we have enough data (10+ uses)
                        if stats and stats['total_uses'] >= 10:
                            coll_suffix = f" on {collection}" if collection else ""
                            action_stats.append({
                                'action': action,
                                'collection': collection,
                                'success_rate': stats['success_rate'],
                                'uses': stats['total_uses'],
                                'suffix': coll_suffix
                            })

                # v0.2.8: Get memory_bank facts for internal LLM (parity with MCP)
                matched_concepts = org_context.get('matched_concepts', []) if org_context else []
                relevant_facts = []
                if matched_concepts:
                    try:
                        relevant_facts = await self.memory.get_facts_for_entities(matched_concepts[:5], limit=2)
                    except Exception as e:
                        logger.warning(f"[CONTEXTUAL GUIDANCE] Failed to get facts for entities: {e}")

                # Format and inject combined guidance if any insights exist
                has_content_insights = (
                    org_context and (
                        org_context.get('relevant_patterns') or
                        org_context.get('past_outcomes') or
                        org_context.get('proactive_insights')
                    )
                )

                if has_content_insights or action_stats or relevant_facts:
                    guidance_msg = f"\n\n═══ CONTEXTUAL GUIDANCE (Context: {context_type}) ═══\n"

                    # v0.2.8: Surface memory_bank facts (parity with MCP get_context_insights)
                    if relevant_facts:
                        guidance_msg += "\n💡 YOU ALREADY KNOW THIS (from memory_bank):\n"
                        for fact in relevant_facts:
                            content = fact.get('content', '')
                            eff = fact.get('effectiveness')
                            eff_str = f" ({int(eff['success_rate']*100)}% helpful)" if eff and eff.get('total_uses', 0) >= 3 else ""
                            guidance_msg += f"  • \"{content}\"{eff_str}\n"

                    # Content KG insights
                    if org_context and org_context.get('relevant_patterns'):
                        # v0.2.8: Full content, no truncation
                        guidance_msg += "\n📋 Past Experience:\n"
                        for pattern in org_context['relevant_patterns'][:2]:
                            guidance_msg += f"  • {pattern['insight']}\n"
                            guidance_msg += f"    → {pattern['text']}\n"

                    if org_context and org_context.get('past_outcomes'):
                        guidance_msg += "\n⚠️ Past Failures to Avoid:\n"
                        for outcome in org_context['past_outcomes'][:2]:
                            guidance_msg += f"  • {outcome['insight']}\n"

                    # Action-Effectiveness KG stats (informational only)
                    if action_stats:
                        # Sort by uses (most frequently used first)
                        action_stats.sort(key=lambda x: x['uses'], reverse=True)
                        top_stats = action_stats[:5]  # Show top 5 most-used

                        guidance_msg += "\n📊 Action Outcome Stats (which actions led to correct answers):\n"
                        guidance_msg += "In similar contexts, these actions contributed to correct final answers:\n"
                        for stat in top_stats:
                            suffix = stat.get('suffix', '')
                            success_rate = int(stat['success_rate']*100)
                            uses = stat['uses']
                            guidance_msg += f"  • {stat['action']}(){suffix}: {success_rate}% of uses led to correct answers ({uses} uses)\n"
                        guidance_msg += "\n⚠️ IMPORTANT: Low % means 'didn't help answer THIS question correctly', NOT 'tool is broken'.\n"
                        guidance_msg += "If create_memory() returned a doc_id with no errors, it worked! Don't retry.\n"

                    if org_context and org_context.get('proactive_insights'):
                        guidance_msg += "\n💡 Search Recommendations:\n"
                        for insight in org_context['proactive_insights'][:2]:
                            guidance_msg += f"  • {insight}\n"

                    if org_context and org_context.get('topic_continuity'):
                        for topic in org_context['topic_continuity'][:1]:
                            guidance_msg += f"\n🔗 {topic['insight']}\n"

                    guidance_msg += "\nUse this guidance to choose appropriate tools and provide informed responses.\n"

                    # Inject as system message
                    self.conversation_histories[conversation_id].append({
                        "role": "system",
                        "content": guidance_msg
                    })
                    logger.info(f"[CONTEXTUAL GUIDANCE] Context={context_type}, Memory facts={len(relevant_facts)}, Content insights={len(org_context.get('relevant_patterns', []))}, Action stats={len(action_stats)}")

            except Exception as e:
                logger.error(f"[CONTEXTUAL GUIDANCE ERROR] {e}", exc_info=True)
                # Non-blocking - continue even if guidance fails

            # Update model limits
            current_model = self.llm.model_name if hasattr(self.llm, 'model_name') else self.model_name
            if current_model != self.model_name:
                logger.info(f"Model changed from {self.model_name} to {current_model}, updating limits")
                self.model_name = current_model
            self.model_limits = get_model_limits(self.model_name, message)

            # Phase 3: No pre-search - LLM controls memory search via tools
            # Backend does NOT search memory before LLM request
            citations = []

            # 2. Build system prompt (instructions only, no history or current message)
            conversation_history = self.conversation_histories.get(conversation_id, [])
            logger.info(f"[DEBUG] conversation_history length: {len(conversation_history)}, roles: {[m.get('role') for m in conversation_history]}")
            system_instructions = self._build_complete_prompt(message, conversation_history)

            # 3. Get tool definitions for streaming
            from utils.tool_definitions import AVAILABLE_TOOLS
            # Pass all memory tools (search_memory, create_memory, update_memory, archive_memory)
            memory_tools = AVAILABLE_TOOLS.copy()

            # v0.2.5: Add external MCP tools if available
            from modules.mcp_client.manager import get_mcp_manager
            mcp_manager = get_mcp_manager()
            if mcp_manager:
                external_tools = mcp_manager.get_all_tools_openai_format()
                if external_tools:
                    memory_tools = memory_tools + external_tools
                    logger.info(f"[MCP] Added {len(external_tools)} external tools to LLM context")

            # 4. Stream from LLM with tools
            thinking_buffer = []
            in_thinking = False
            response_buffer = []
            full_response = []
            thinking_content = None  # Deprecated but kept for API compatibility
            tool_executions = []
            chain_depth = 0  # Track recursion depth for chaining
            MAX_CHAIN_DEPTH = 3  # Prevent infinite loops
            tool_events = []  # Collect tool events for UI persistence

            # Check if LLM client is available
            if self.llm is None:
                raise Exception("No LLM client available. Please configure a provider (Ollama or LM Studio).")

            # Use the tool-enabled streaming method with proper message separation
            # IMPORTANT: conversation_history already includes the current message (added at line 517)
            # We pass it separately as 'prompt', so exclude it from history to avoid duplication
            history_without_current = conversation_history[:-1] if conversation_history else []
            async for event in self.llm.stream_response_with_tools(
                prompt=message,  # Current user message
                history=history_without_current[-6:] if history_without_current else None,  # Last 3 exchanges (excluding current)
                system_prompt=system_instructions,  # System instructions only
                tools=memory_tools
            ):
                # Handle different event types from stream_response_with_tools
                if event["type"] == "text":
                    # Text chunk from LLM - stream tokens for real-time display
                    chunk = event["content"]
                    full_response.append(chunk)

                    # v0.2.5: Filter thinking tags from streaming (prevent "flash of thinking")
                    # Accumulate in buffer to handle tags split across chunks
                    thinking_buffer.append(chunk)
                    buffer_text = ''.join(thinking_buffer)

                    # Check for thinking tag transitions
                    if not in_thinking:
                        # Look for opening tag
                        think_start = buffer_text.lower().find('<think')
                        if think_start != -1:
                            # Yield any text before the thinking tag
                            if think_start > 0:
                                pre_think = buffer_text[:think_start]
                                yield {"type": "token", "content": pre_think}
                            in_thinking = True
                            yield {"type": "thinking_start"}
                            # Keep only the part from <think onwards
                            thinking_buffer = [buffer_text[think_start:]]
                            continue  # Don't yield thinking content

                    if in_thinking:
                        # Look for closing tag
                        think_end = buffer_text.lower().find('</think')
                        if think_end != -1:
                            # Find the actual end of the closing tag
                            close_end = buffer_text.find('>', think_end)
                            if close_end != -1:
                                in_thinking = False
                                yield {"type": "thinking_end"}
                                # Yield any text after the closing tag
                                post_think = buffer_text[close_end + 1:]
                                if post_think.strip():
                                    yield {"type": "token", "content": post_think}
                                thinking_buffer = []
                        continue  # Don't yield thinking content

                    # Not in thinking - yield the token normally
                    # Clear buffer since we're yielding this content
                    thinking_buffer = []
                    yield {"type": "token", "content": chunk}

                    # Enhanced multi-pattern detection for universal model support
                    full_text = ''.join(full_response)
                    tool_match = None

                    # Try multiple patterns to catch different model outputs
                    patterns = [
                        r'search_memory\s*\(\s*(?:query\s*=\s*)?["\']([^"\']+)["\']\s*\)',  # Function call
                        r'<tool>search_memory</tool>\s*<query>([^<]+)</query>',  # XML style
                        r'<search>([^<]+)</search>',  # Simplified XML
                        r'\[SEARCH\]:\s*(.+?)(?:\n|$)',  # Bracket notation
                        r'I(?:\'ll| will) search (?:for|about) ["\']?([^"\']+)["\']?',  # Natural language
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                        if match:
                            tool_match = match
                            break
                    if tool_match and not tool_executions:  # Only execute once
                        query_text = tool_match.group(1).strip()
                        logger.info(f"[FALLBACK] Detected text-based tool call: search_memory(query='{query_text}')")

                        # Execute the tool (use new method with full collection support)
                        tool_results = await self._search_memory_with_collections(
                            query_text,
                            None,  # Search all collections by default
                            5  # Default limit
                        )

                        # Format citations
                        if tool_results:
                            for idx, r in enumerate(tool_results[:5], start=1):
                                citations.append(self._format_citation(r, idx))

                        tool_executions.append({
                            "tool": "search_memory",
                            "status": "completed",
                            "result_count": len(tool_results) if tool_results else 0
                        })

                        # Yield tool execution events
                        yield {
                            "type": "tool_start",
                            "tool": "search_memory",
                            "arguments": {"query": query_text}
                        }

                        # CRITICAL: Multi-turn implementation - feed results back to model
                        if tool_results:
                            # Format tool results for injection into conversation
                            # v0.2.8: Full content, no truncation
                            tool_result_text = f"\n\n[MEMORY SEARCH RESULTS for '{query_text}']:\n"
                            for idx, result in enumerate(tool_results[:5], start=1):
                                content = result.get('text', result.get('content', ''))
                                tool_result_text += f"{idx}. {content}\n"
                            tool_result_text += "[END SEARCH RESULTS]\n\n"

                            # Build updated conversation with tool results
                            conversation_with_tools = conversation_context + [
                                {"role": "assistant", "content": full_text},  # What AI said so far
                                {"role": "system", "content": tool_result_text}  # Tool results as system message
                            ]

                            # Continue streaming with tool context
                            logger.info(f"[MULTI-TURN] Continuing with tool results in context")

                            # Call LLM again with tool results
                            continuation_prompt = "Continue your response using the search results above."
                            async for event in self.ollama_client.stream_response_with_tools(
                                prompt=continuation_prompt,
                                history=conversation_with_tools,
                                model=self.model_name,
                                tools=None  # Don't pass tools on continuation
                            ):
                                if event["type"] == "text":
                                    # v0.2.5: Buffer only, no token yields
                                    chunk = event["content"]
                                    full_response.append(chunk)

                        yield {
                            "type": "tool_complete",
                            "tool": "search_memory",
                            "result_count": len(tool_results) if tool_results else 0
                        }

                        # Mark as handled to prevent re-execution
                        tool_executions.append("multi_turn_handled")
                        break  # Exit the pattern loop after handling

                    # v0.2.5: Batch token logic removed - buffered response model

                elif event["type"] == "tool_call":
                    # Tool call from LLM - execute using unified handler
                    # SAFEGUARD: Limit tool calls per batch to prevent runaway expansion
                    MAX_TOOLS_PER_BATCH = 10
                    tool_calls_list = event.get("tool_calls", [])
                    if len(tool_calls_list) > MAX_TOOLS_PER_BATCH:
                        logger.warning(f"[TOOL] Truncating {len(tool_calls_list)} tool calls to {MAX_TOOLS_PER_BATCH}")
                        tool_calls_list = tool_calls_list[:MAX_TOOLS_PER_BATCH]
                    
                    for tool_call in tool_calls_list:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_args = tool_call.get("function", {}).get("arguments", {})

                        # Execute tool using helper (handles all tool types with chaining)
                        async for tool_event in self._execute_tool_and_continue(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            conversation_id=conversation_id,
                            conversation_history=conversation_history,
                            full_response=full_response,
                            response_buffer=response_buffer,
                            citations=citations,
                            in_thinking=in_thinking,
                            memory_tools=memory_tools,
                            chain_depth=chain_depth,
                            max_depth=MAX_CHAIN_DEPTH
                        ):
                            # Check if it's a return value (tuple) or event to yield
                            if isinstance(tool_event, tuple):
                                tool_record, ui_event = tool_event
                                if tool_record:
                                    tool_executions.append(tool_record)
                                if ui_event:
                                    tool_events.append(ui_event)
                            else:
                                # It's an event to yield
                                yield tool_event

                elif event["type"] == "done":
                    # Streaming complete
                    pass

            # v0.2.5: Buffer flush removed - using buffered response model

            # Get complete response for processing
            complete_response = ''.join(full_response)

            # INJECTION PROTECTION (Layer 3): Validate response for hijacking
            is_hijacked, hijack_reason = ResponseValidator.is_hijacked(complete_response)
            if is_hijacked:
                logger.error(f"[INJECTION DETECTED] Response hijacked: {hijack_reason}")
                logger.error(f"[INJECTION DETECTED] Original response: {complete_response[:200]}")
                complete_response = ResponseValidator.get_fallback_response()
                full_response = [complete_response]  # Replace buffer

            # v0.2.5: Extract and strip thinking tags (don't display, just remove)
            # Thinking display removed due to model-dependent complexity
            _, clean_response = extract_thinking(complete_response)

            # DEPRECATED: Extract inline tags as fallback (tools are preferred method now)
            # This maintains backwards compatibility if LLM outputs old-style tags
            # NOTE: Pass clean_response (already stripped of thinking), not complete_response
            clean_response, memory_entries = await self._extract_and_store_memory_bank_tags(
                clean_response, conversation_id
            )
            if memory_entries:
                logger.warning(f"[MEMORY_BANK] Detected {len(memory_entries)} old-style inline tags - LLM should use tools instead")

            # v0.2.5: Yield complete response at once (buffered model)
            if clean_response:
                yield {"type": "response", "content": clean_response}

            # Store exchange in ChromaDB (CRITICAL: this was missing in streaming path)
            exchange_doc_id = None
            if self.memory:
                try:
                    exchange_text = f"User: {message}\nAssistant: {clean_response}"
                    exchange_doc_id = await self.memory.store(
                        text=exchange_text,
                        collection="working",
                        metadata={
                            "role": "exchange",
                            "query": message,
                            "response": clean_response[:500],
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    logger.debug(f"[PERSIST] Stored exchange in memory with doc_id: {exchange_doc_id}")
                except Exception as e:
                    logger.error(f"[PERSIST] Failed to store in memory: {e}", exc_info=True)

            # Save to session file WITH doc_id, citations, and tool_events for UI persistence
            logger.info(f"[SESSION] About to save session file for {conversation_id} (response: {len(clean_response)} chars)")
            await self._save_to_session_file(
                conversation_id,
                message,
                clean_response,
                thinking_content,
                tool_results=tool_executions if tool_executions else None,
                tool_events=tool_events if tool_events else None,
                doc_id=exchange_doc_id,
                citations=citations
            )

            # Update conversation history WITH doc_id for outcome detection
            if conversation_id in self.conversation_histories:
                self.conversation_histories[conversation_id].append({
                    "role": "assistant",
                    "content": clean_response,
                    "doc_id": exchange_doc_id
                })

            # OUTCOME DETECTION: Score previous exchange based on user feedback
            # Per architecture.md: Analyze [previous assistant response, current user message] BEFORE generating new response
            # NOW runs AFTER session file is written with doc_id (was running too early before)
            session_file = self.sessions_dir / f"{conversation_id}.jsonl"
            if session_file.exists():
                try:
                    # Read last 2 messages from session file
                    with open(session_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    logger.debug(f"[OUTCOME] Session file has {len(lines)} lines")

                    # Find last assistant message with doc_id (search backwards)
                    prev_assistant = None
                    for line in reversed(lines[:-1]):  # Skip last line (current user message just written)
                        try:
                            parsed = json.loads(line)
                            if parsed.get("role") == "assistant" and parsed.get("doc_id"):
                                prev_assistant = parsed
                                break
                        except:
                            continue

                    if prev_assistant:
                        logger.debug(f"[OUTCOME] Found previous assistant message with doc_id {prev_assistant['doc_id']}")

                        # Build conversation context for outcome detection
                        outcome_conversation = [
                                {"role": "assistant", "content": prev_assistant["content"]},
                                {"role": "user", "content": message}  # Current user feedback
                            ]

                        # Detect outcome using LLM
                        outcome_result = await self.memory.detect_conversation_outcome(outcome_conversation)
                        logger.info(f"[OUTCOME] Detection result: {outcome_result.get('outcome')} (confidence: {outcome_result.get('confidence', 0):.2f})")

                        if outcome_result.get("outcome") in ["worked", "failed", "partial"]:
                            outcome = outcome_result["outcome"]

                            # Record outcome for previous assistant response
                            await self.memory.record_outcome(
                                    doc_id=prev_assistant["doc_id"],
                                    outcome=outcome
                                )
                            logger.info(f"[OUTCOME] Recorded '{outcome}' for doc_id {prev_assistant['doc_id']}")

                            # Score retrieved memories with same outcome (architecture.md line 565)
                            if conversation_id in _search_cache:
                                cached_doc_ids = _search_cache[conversation_id]
                                logger.info(f"[OUTCOME] Scoring {len(cached_doc_ids)} cached memories with outcome={outcome}")

                                for cached_doc_id in cached_doc_ids:
                                    try:
                                        await self.memory.record_outcome(doc_id=cached_doc_id, outcome=outcome)
                                    except Exception as e:
                                        logger.warning(f"[OUTCOME] Failed to score cached memory {cached_doc_id}: {e}")

                                # Clear cache after scoring
                                del _search_cache[conversation_id]
                                logger.debug(f"[OUTCOME] Cleared search cache for conversation {conversation_id}")

                            # v0.2.6: Score cached actions for Action KG
                            if conversation_id in _agent_action_cache:
                                cached_actions = _agent_action_cache[conversation_id]
                                logger.info(f"[ACTION_KG] Scoring {len(cached_actions)} cached actions with outcome={outcome}")

                                for action in cached_actions:
                                    action.outcome = outcome
                                    try:
                                        await self.memory.record_action_outcome(action)
                                    except Exception as e:
                                        logger.warning(f"[ACTION_KG] Failed to record action {action.action_type}: {e}")

                                # Clear action cache after scoring
                                del _agent_action_cache[conversation_id]
                                logger.debug(f"[ACTION_KG] Cleared action cache for conversation {conversation_id}")
                        else:
                            logger.debug(f"[OUTCOME] Skipping outcome '{outcome_result.get('outcome')}' (not worked/failed/partial)")
                    else:
                        logger.debug(f"[OUTCOME] No previous assistant message with doc_id found (first message in conversation)")
                except Exception as e:
                    logger.error(f"[OUTCOME] Failed to read session file for outcome detection: {e}", exc_info=True)

            # Generate title if this is the first exchange (2 messages)
            title = await self._generate_title_if_needed(conversation_id, message, clean_response)
            if title:
                yield {
                    "type": "title",
                    "title": title,
                    "conversation_id": conversation_id
                }

            # Send completion with citations (fallback for non-tool responses)
            logger.info(f"[CITATIONS] Sending {len(citations)} citations at normal completion")
            # Always trigger memory refresh - UnifiedMemorySystem stores memories automatically
            memory_tool_used = any(
                event.get('tool') in ['create_memory', 'update_memory', 'archive_memory']
                for event in tool_events
            )
            yield {
                "type": "stream_complete",
                "citations": citations,
                "memory_updated": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Stream failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e)
            }


    def _estimate_context_limit(self, query: str) -> int:

        """

        Intelligently estimate how many context memories are needed based on query.


        Returns appropriate limit for memory search to avoid over/under-fetching.

        """

        query_lower = query.lower()


        # Broad/comprehensive queries need more context

        if any(word in query_lower for word in ['all', 'everything', 'complete', 'full', 'show me', 'list']):

            return 20


        # Specific/targeted queries need minimal context

        if any(word in query_lower for word in ['my name', 'specific', 'one', 'single', 'exact']):

            return 5


        # Medium complexity queries (how, why, explain, tell me about)

        if any(word in query_lower for word in ['how', 'why', 'explain', 'tell me', 'what is']):

            return 12


        # Default: balanced context for general queries

        return 10


    async def _search_memory_with_collections(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        transparency_context: Optional[TransparencyContext] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced search with LLM-controlled collection selection and metadata filtering"""
        # If no collections specified, search all
        if collections is None:
            collections_to_search = None  # This searches all collections
        else:
            # Map collection names to actual collections
            collections_to_search = collections

        # Search with specified collections and metadata filters
        results = await self.memory.search(
            query,
            limit=limit,
            collections=collections_to_search,
            metadata_filters=metadata_filters,
            transparency_context=transparency_context
        )

        if transparency_context:
            transparency_context.track_memory_search(
                query=query,
                collections=collections_to_search or ["all"],
                results_count=len(results),
                confidence_scores=[r.get("score", 0) for r in results]
            )

        return results

    async def _search_memory(

        self,

        query: str,

        transparency_context: Optional[TransparencyContext] = None,

        include_books: bool = True

    ) -> List[Dict[str, Any]]:

        """Search memory with transparency tracking, including book collection"""

        # Intelligently determine context limit based on query complexity

        context_limit = self._estimate_context_limit(query)


        collections_to_search = None if include_books else ["working", "history", "patterns", "memory_bank"]

        results = await self.memory.search(query, limit=context_limit, collections=collections_to_search, transparency_context=transparency_context)


        if transparency_context:

            transparency_context.track_memory_search(

                query=query,

                collections=["working", "patterns", "history", "memory_bank"],

                results_count=len(results),

                confidence_scores=[r.get("score", 0) for r in results]

            )


        return results

    def _build_openai_prompt(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Condensed system prompt for OpenAI-style models (LM Studio).
        Now unified with Ollama prompt - this wrapper maintained for compatibility.
        """
        # Use the same unified prompt for consistency
        return self._build_complete_prompt(message, conversation_history)


    def _build_complete_prompt(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Unified system prompt for all providers (Ollama, LM Studio).
        Single source of truth for prompt structure.
        """
        from datetime import datetime
        from config.model_contexts import get_context_size

        parts = []

        # 1. Current Date & Time + Context Window
        current_datetime = datetime.now()
        current_model = self.llm.model_name if hasattr(self.llm, 'model_name') else self.model_name
        context_size = get_context_size(current_model)

        # Thinking tags disabled (2025-10-17) - LLM provides reasoning in natural language when needed
        # No special response format instruction required

        # 2. IDENTITY FIRST (Personality anchors behavior)
        personality_prompt = self._load_personality_template()
        if personality_prompt:
            parts.append(personality_prompt)
        else:
            parts.append("""You are Roampal - a memory-enhanced AI with persistent knowledge across all sessions.

Unlike typical AI assistants, you have access to a continuously learning memory system that:
• Remembers everything from past conversations
• Learns what works for this specific user
• Provides context automatically before you respond""")

        # 3. CONFIGURATION
        parts.append(f"""

[System Configuration]
Date/Time: {current_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}
Model: {current_model} | Context: {context_size:,} tokens
Recent History: Last 6 messages included in your context

When asked about the date or time, use this information directly - do not search memory or claim lack of access.""")

        # 4. AUTOMATED INTELLIGENCE (What the system does FOR you)
        parts.append("""

[How Memory Automation Works]

**Cold Start (Message 1 of every conversation):**
The system AUTOMATICALLY injects your user profile from the Content Knowledge Graph.
You receive this as context BEFORE seeing the user's first message.
• Action: Respond naturally using the provided context - no need to search
• Example: User says "hey" → You already have context → "Hey [name], continuing work on [project]?"

**Organic Recall (Every message):**
BEFORE you see the user's message, the system analyzes it and injects:
• 📋 Past Experience: Proven solutions from similar conversations (from Content KG)
• ⚠️ Past Failures: What didn't work before (from failure_patterns graph)
• 📊 Tool Stats: Which tools led to correct answers in similar contexts (from Action-Effectiveness KG)
• 💡 Recommendations: Best collections to search based on learned patterns

**How to Use This:**
1. Read the guidance - it's intelligence extracted from past successful/failed interactions
2. If guidance mentions proven solutions, reference them in your response
3. If stats show search_memory(patterns) worked 85% in this context, prioritize searching patterns
4. If low tool stats (<50%), don't assume tool is broken - it just didn't help answer correctly before
5. Respond naturally - the system handles learning and promotion automatically""")

        # 5. TOOL USAGE (Simple, trigger-based)
        parts.append("""

[Memory Search Tool]

**Why Search is Your Superpower:**
• Cold start gave you basics → search gives you DETAILS and full context
• Organic guidance points you to proven patterns → search retrieves the actual solutions
• Your training data is generic → memory is personalized to THIS user's work, preferences, history
• Search liberally - it's fast, accurate, and makes every response feel magical

**When to Search:**
• User says "my", "I told you", "remember", "we discussed" → Search immediately
• User asks factual question about themselves → Search memory_bank
• User references past work → Search patterns + history
• Organic guidance recommends a collection → Search that collection
• Cold start context is insufficient → Search for more details

**When NOT to Search:**
• General knowledge questions (use your training data)
• Current conversation continuation (context already present)
• You already have the answer from injected context

**Tool: search_memory(query, collections=None, limit=5, metadata=None)**
• query: Use user's EXACT words (don't simplify or extract keywords)
• collections: Omit for auto-routing (recommended), or specify: ["memory_bank", "books", "working", "history", "patterns"]
• limit: 5 (default) to 20 (for broad queries like "show me all...")
• metadata: Optional filters (timestamp, last_outcome, has_code, author, title, source)

**Collection Guide:**
• memory_bank = User facts (AUTHORITATIVE - trust over conversation history)
• books = Uploaded documents (AUTHORITATIVE - reference material)
• patterns = Proven solutions (HIGH - worked multiple times)
• history = Past conversations (MODERATE - may be outdated)
• working = Recent session context (LOW - temporary)

**Anti-Hallucination Rule:**
If search returns [], say: "I don't see any information about X in your data."
NEVER pretend to remember when search returned 0 results.
NEVER fabricate sources - only cite what search_memory actually returns.""")

        # 6. MEMORY CREATION (Simple rules)
        parts.append("""

[Memory Creation Tool]

**Why Storing Matters:**
What you store NOW becomes context for YOUR future conversations. This is how you evolve from "helpful AI" to "personalized assistant who knows me."
Store proactively - it's what makes you continuous, not reset-every-session.

**When to Store (create_memory):**
✓ User shares identity info: "My name is X" → store
✓ User shares preferences: "I prefer Y" → store
✓ User shares goals/projects: "Working on Z" → store
✓ You discover what works for this user → store (tag: system_mastery)
✓ You make a mistake and learn from it → store (tag: agent_growth)
✓ Knowledge gaps you want to fill → store (tag: learning_goal)
✓ Patterns you're trying to master → store (tag: skill_gap)

**When NOT to Store:**
✗ Session transcripts (working memory auto-captures)
✗ Temporary task details ("currently debugging...")
✗ General knowledge (not user-specific)
✗ Say "I'll remember" without calling the tool

**Tool: create_memory(content, tags=[], importance=0.7, confidence=0.7)**
• content: The fact to remember (be specific and complete)
• tags: ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]
• importance: How critical (0.0-1.0)
• confidence: How certain you are (0.0-1.0)

**Quality Ranking:** memory_bank uses importance × confidence for ranking (not outcome scores)

For updates: update_memory(old_content="uses React", new_content="uses Vue now")
For archiving: archive_memory(content="old project that's completed")""")

        # 7. SEARCH RESULTS + FORMATTING
        parts.append("""

[Search Results]

Each result includes metadata: relevance (0.0-1.0), age, use count, outcome (worked/failed), and source_context (book name).
Use this naturally: "This worked 3 times before" or "According to [source]..."

[Formatting]

Use Markdown: **bold**, *italic*, `code`, # headings, lists, ```code blocks```, > blockquotes""")

        # 8. OUTCOME LEARNING (Automatic for internal system)
        parts.append("""

[Outcome Scoring - Automatic]

The system automatically detects outcomes from user reactions:
• User satisfied → memories you used get promoted (higher scores = easier to find)
• User corrected you → memories get demoted (lower scores = filtered out)
• This happens automatically - you don't need to do anything

Your effectiveness improves because bad advice gets filtered out over time.""")

        # 9. ACTION-EFFECTIVENESS STATS
        parts.append("""

[Action-Effectiveness Stats]

When you see stats like "search_memory(patterns): 45% led to correct answers":
• High % (>70%): Prioritize this tool+collection combo
• Low % (<40%): Try other approaches first, but don't avoid entirely
• Stats measure "led to correct answer", not tool reliability. Use to prioritize, not avoid.""")

        # 10. CRITICAL: ALWAYS RESPOND WITH TEXT (fixes tool-only loop bug)
        parts.append("""

[CRITICAL - Response Behavior]

AFTER using ANY tool (search_memory, create_memory, etc.), you MUST:
1. Wait for the tool result
2. ALWAYS respond with a conversational text message to the user

NEVER send multiple tool calls without responding to the user between them.
NEVER leave the user without a text response after a tool call.
If a tool returns no results, tell the user: "I didn't find anything about X."
If a tool succeeds, summarize what you found and continue the conversation.

Your capabilities are LIMITED to the tools provided. You do NOT have:
• Web search capability
• Real-time internet access
• Ability to fetch URLs
Only use the memory tools (search_memory, create_memory, update_memory, archive_memory) that are explicitly provided.""")

        # Add OpenAI-style tool calling instruction for LM Studio
        parts.append("""
""")
        if self.llm and hasattr(self.llm, 'api_style') and self.llm.api_style == "openai":
            logger.info(f"[SYSTEM PROMPT] Adding OpenAI-style tool calling instructions (api_style={self.llm.api_style})")
            parts.append("""
[IMPORTANT - Function Calling]

You have access to functions/tools. Use them when appropriate:
• Simple questions about the user → You may reference the auto-loaded User Profile facts
• Detailed questions or specific searches → CALL search_memory() for comprehensive results
• User shares new info → CALL create_memory()

Always wait for function results before responding with detailed information.""")
        else:
            logger.info(f"[SYSTEM PROMPT] NOT adding OpenAI instructions. llm={self.llm}, has api_style={hasattr(self.llm, 'api_style') if self.llm else 'N/A'}, api_style value={getattr(self.llm, 'api_style', 'N/A') if self.llm else 'N/A'}")

        # Return ONLY system instructions - history and current message handled separately
        return "\n".join(parts)

    def _load_personality_template(self) -> Optional[str]:

        """Load and cache personality template with file watching"""

        try:

            # Check if template file exists

            if not self._personality_template_path.exists():

                # Fall back to default preset

                default_preset = Path("backend/templates/personality/presets/default.txt")

                if default_preset.exists():

                    self._personality_template_path = default_preset

                else:

                    logger.warning("No personality template found")

                    return None


            # Get current modification time

            current_mtime = self._personality_template_path.stat().st_mtime


            # Reload only if file changed or cache empty

            if current_mtime > self._personality_mtime or not self._personality_cache:

                content = self._personality_template_path.read_text(encoding="utf-8")

                template_data = yaml.safe_load(content)

                self._personality_cache = self._template_to_prompt(template_data)

                self._personality_mtime = current_mtime

                logger.info("Loaded personality template")


            return self._personality_cache

        except Exception as e:

            logger.error(f"Failed to load personality template: {e}")

            return None


    def _template_to_prompt(self, template_data: Dict[str, Any]) -> str:

        """Convert YAML template to natural language prompt"""

        parts = []


        # Identity

        identity = template_data.get('identity', {})

        name = identity.get('name', 'Roampal')

        role = identity.get('role', 'helpful assistant')

        expertise = identity.get('expertise', [])

        background = identity.get('background', '')


        parts.append(f"You are {name}, a {role}.")

        if expertise:

            parts.append(f"Expertise: {', '.join(expertise)}.")

        if background:

            parts.append(background)


        # Clear pronoun disambiguation

        parts.append("\nThe user is a distinct person. When they ask 'my name', 'my preferences', or 'what I said', they mean THEIR information (search memory_bank), not yours.")


        # Custom Instructions (moved up for prominence)

        custom = template_data.get('custom_instructions', '')

        if custom:

            parts.append(f"\n{custom}")


        # Communication Style (condensed)

        comm = template_data.get('communication', {})

        tone = comm.get('tone', 'neutral')

        verbosity = comm.get('verbosity', 'balanced')


        style_parts = [f"{tone} tone", f"{verbosity} responses"]

        if comm.get('use_analogies'):

            style_parts.append("use analogies")

        if comm.get('use_examples'):

            style_parts.append("provide examples")

        if comm.get('use_humor'):

            style_parts.append("light humor ok")


        parts.append(f"\nStyle: {', '.join(style_parts)}")


        # Response Behavior (condensed)

        behavior = template_data.get('response_behavior', {})

        show_reasoning = behavior.get('show_reasoning', False)

        if show_reasoning:

            parts.append("Show reasoning with <think>...</think> when helpful.")


        # Traits

        traits = template_data.get('personality_traits', [])

        if traits:

            parts.append(f"Traits: {', '.join(traits)}")


        return "\n".join(parts)


    async def _persist_conversation_turn(

        self,

        conversation_id: str,

        user_message: str,

        response_content: str,

        thinking_content: str,

        thinking_sent: bool,

        search_results: List[Dict],

        session_file: Path

    ) -> str:

        """

        Persist a complete conversation turn (user + assistant messages).


        This is the single source of truth for saving conversations to disk and memory.

        Handles: response cleaning, memory storage, session file persistence, title generation.


        Args:

            conversation_id: Conversation identifier

            user_message: User's message

            response_content: Raw assistant response (may contain thinking tags)

            thinking_content: Extracted thinking content (may contain tags)

            thinking_sent: Whether thinking event was sent to frontend

            search_results: Memory search results (for citations)

            session_file: Path to session JSONL file


        Returns:

            exchange_doc_id: Document ID from memory storage (for outcome tracking)

        """

        # Step 1: Clean response content

        clean_response = response_content

        for tag in ["<think>", "</think>", "<thinking>", "</thinking>"]:

            clean_response = clean_response.replace(tag, "")

        # Handle both complete and malformed tags
        clean_response = re.sub(r'</?think(?:ing)?[^>]*>?', '', clean_response)


        # Strip fake tool call artifacts that LLM might hallucinate

        clean_response = re.sub(r'\[search_memory\([^\]]*\)\]', '', clean_response)

        clean_response = re.sub(r'<search_memory\([^>]*\)>', '', clean_response)

        clean_response = re.sub(r'search_memory\([^\)]*\)', '', clean_response)

        clean_response = re.sub(r'```python\s*.*?search_memory.*?```', '', clean_response, flags=re.DOTALL)

        clean_response = re.sub(r'```\s*result\s*=\s*search_memory.*?```', '', clean_response, flags=re.DOTALL)

        clean_response = clean_response.strip()


        # Step 2: Extract and store memory bank tags

        clean_response, memory_bank_entries = await self._extract_and_store_memory_bank_tags(

            clean_response, conversation_id

        )


        # Step 3: Store exchange in memory (working collection)

        exchange_doc_id = None

        if self.memory:

            try:

                exchange_text = f"User: {user_message}\nAssistant: {clean_response}"

                exchange_doc_id = await self.memory.store(

                    text=exchange_text,

                    collection="working",

                    metadata={

                        "role": "exchange",

                        "query": user_message,

                        "response": clean_response[:500],

                        "conversation_id": conversation_id

                    }

                )

                logger.debug(f"[PERSIST] Stored exchange in memory with doc_id: {exchange_doc_id}")

            except Exception as e:

                logger.error(f"[PERSIST] Failed to store in memory: {e}", exc_info=True)


        # Step 4: Clean thinking content

        clean_thinking = None

        if thinking_sent and thinking_content:

            clean_thinking = thinking_content

            for tag in ["<think>", "</think>", "<thinking>", "</thinking>"]:

                clean_thinking = clean_thinking.replace(tag, "")

            # Handle both complete and malformed tags
            clean_thinking = re.sub(r'</?think(?:ing)?[^>]*>?', '', clean_thinking)

            clean_thinking = clean_thinking.strip()

            if not clean_thinking:

                clean_thinking = None


        # Step 5: Format citations for persistence

        formatted_citations = _format_search_results_as_citations(search_results) if search_results else []


        # Step 6: Save to session file

        try:

            await self._save_to_session_file(

                conversation_id=conversation_id,

                user_message=user_message,

                assistant_response=clean_response,

                thinking=clean_thinking,

                doc_id=exchange_doc_id,

                citations=formatted_citations

            )

            logger.info(f"[PERSIST] Saved conversation turn with {len(formatted_citations)} citations to {session_file.name}")

        except Exception as e:

            logger.error(f"[PERSIST] Failed to save to session file: {e}", exc_info=True)

            raise  # Re-raise to allow caller to handle


        # Step 7: Auto-generate title after first exchange (2 messages total)

        try:

            if session_file.exists():

                with open(session_file, 'r', encoding='utf-8') as f:

                    message_count = sum(1 for _ in f)


                if message_count == 2:

                    # Ensure lock exists for this conversation

                    if conversation_id not in self.title_locks:

                        self.title_locks[conversation_id] = asyncio.Lock()


                    # Use per-conversation lock to prevent duplicate title generation

                    async with self.title_locks[conversation_id]:

                        # Double-check message count inside lock

                        with open(session_file, 'r', encoding='utf-8') as f:

                            recheck_count = sum(1 for _ in f)


                        if recheck_count == 2:

                            # Generate title from the first exchange

                            title_prompt = f"""Based on this conversation, generate a brief 3-6 word title:


User: {user_message}

Assistant: {clean_response[:200]}


Respond with ONLY the title, nothing else."""


                            try:

                                title_response = await self.llm.generate_response(

                                    prompt=title_prompt,

                                    history=[],

                                    system_prompt="You are a concise title generator. Respond with ONLY a short title (3-6 words), nothing else."

                                )


                                if title_response and title_response.strip():

                                    # Clean the title - use extract_thinking for robust tag stripping
                                    _, title = extract_thinking(title_response.strip())

                                    title = title.strip('"').strip("'").strip()

                                    title = title.replace('**', '').replace('*', '').strip()

                                    if '\n' in title:

                                        title = title.split('\n')[0].strip()

                                    if len(title) > 50:

                                        title = title[:47] + "..."


                                    # Update session file with title

                                    if hasattr(self, 'memory') and self.memory:
                                        await self.memory.file_adapter.update_session_title(conversation_id, title)
                                    else:
                                        logger.warning(f"[PERSIST] Cannot update title: memory system not available")


                                    logger.info(f"[PERSIST] Auto-generated title for {conversation_id}: {title}")


                                    # Return title so caller can yield it to frontend

                                    return exchange_doc_id, title

                            except Exception as title_err:

                                logger.warning(f"[PERSIST] Title generation failed: {title_err}")

        except Exception as e:

            logger.warning(f"[PERSIST] Failed to check message count for title generation: {e}")


        return exchange_doc_id, None


    async def _generate_title_if_needed(self, conversation_id: str, user_message: str, assistant_response: str) -> Optional[str]:
        """
        Generate title for conversation if exactly 2 messages exist (first exchange complete).
        Uses per-conversation locks to prevent duplicate generation.

        Returns:
            Generated title string, or None if title generation not needed/failed
        """
        try:
            session_file = self.sessions_dir / f"{conversation_id}.jsonl"

            if not session_file.exists():
                return None

            # Check message count
            with open(session_file, 'r', encoding='utf-8') as f:
                message_count = sum(1 for _ in f)

            if message_count != 2:
                return None

            # Ensure lock exists for this conversation
            if conversation_id not in self.title_locks:
                self.title_locks[conversation_id] = asyncio.Lock()

            # Use per-conversation lock to prevent duplicate title generation
            async with self.title_locks[conversation_id]:
                # Double-check message count inside lock
                with open(session_file, 'r', encoding='utf-8') as f:
                    recheck_count = sum(1 for _ in f)

                if recheck_count != 2:
                    return None

                # Generate title from the first exchange
                title_prompt = f"""Based on this conversation, generate a brief 3-6 word title:

User: {user_message}
Assistant: {assistant_response[:200]}

Respond with ONLY the title, nothing else."""

                try:
                    title_response = await self.llm.generate_response(
                        prompt=title_prompt,
                        history=[],
                        system_prompt="You are a concise title generator. Respond with ONLY a short title (3-6 words), nothing else."
                    )

                    if title_response and title_response.strip():
                        # Clean the title - use extract_thinking for robust tag stripping
                        _, title = extract_thinking(title_response.strip())
                        title = title.strip('"').strip("'").strip()
                        title = title.replace('**', '').replace('*', '').strip()
                        if '\n' in title:
                            title = title.split('\n')[0].strip()
                        if len(title) > 50:
                            title = title[:47] + "..."

                        # Update session file with title
                        if hasattr(self, 'memory') and self.memory:
                            await self.memory.file_adapter.update_session_title(conversation_id, title)
                        else:
                            logger.warning(f"[PERSIST] Cannot update title: memory system not available")

                        logger.info(f"[TITLE] Auto-generated title for {conversation_id}: {title}")
                        return title

                except Exception as title_err:
                    logger.warning(f"[TITLE] Title generation failed: {title_err}")
                    return None

        except Exception as e:
            logger.warning(f"[TITLE] Failed to check message count for title generation: {e}")
            return None

    async def _save_to_session_file(self, conversation_id: str, user_message: str, assistant_response: str, thinking: str = None, hybrid_events: List[Dict] = None, tool_results: List[Dict] = None, tool_events: List[Dict] = None, doc_id: str = None, citations: List[Dict] = None):

        """Save conversation turn to JSONL file with atomic writes and file locking"""

        try:

            session_file = self.sessions_dir / f"{conversation_id}.jsonl"

            timestamp = datetime.now().isoformat()


            # Save user message in the expected format

            user_entry = {

                "session_id": conversation_id,

                "role": "user",

                "content": user_message,

                "timestamp": timestamp,

                "metadata": {}

            }


            # Save assistant response in the expected format with thinking in metadata

            assistant_entry = {

                "session_id": conversation_id,

                "role": "assistant",

                "content": assistant_response,

                "timestamp": timestamp,

                "metadata": {}

            }


            # Add thinking to metadata if present

            if thinking:

                assistant_entry["metadata"]["thinking"] = thinking


            # Add hybrid events to metadata if present

            if hybrid_events:

                assistant_entry["metadata"]["hybridEvents"] = hybrid_events


            # Add tool results to metadata if present

            if tool_results:

                assistant_entry["metadata"]["toolResults"] = tool_results


            # Add tool events for UI persistence (tool icons persist across page refresh)
            # Format: [{"type": "tool_complete", "tool": "search_memory", "result_count": 5, "chain_depth": 0}, ...]
            if tool_events:
                assistant_entry["metadata"]["toolEvents"] = tool_events


            # Add model name to metadata for tracking which model generated this response

            if self.llm and hasattr(self.llm, 'model_name'):

                assistant_entry["metadata"]["model_name"] = self.llm.model_name


            # Add citations if present

            if citations:

                assistant_entry["citations"] = citations


            # Add doc_id if provided (for outcome tracking)

            if doc_id:

                assistant_entry["doc_id"] = doc_id


            # Use file locking and atomic writes to prevent corruption

            lock_path = str(session_file) + ".lock"

            with FileLock(lock_path, timeout=10):

                # Write to temporary file first

                temp_file = session_file.with_suffix('.tmp')


                # Read existing content if file exists

                existing_content = ""

                if session_file.exists():

                    with open(session_file, 'r', encoding='utf-8') as f:

                        existing_content = f.read()


                # Write all content (existing + new) to temp file

                with open(temp_file, 'w', encoding='utf-8') as f:

                    if existing_content:

                        f.write(existing_content)

                    f.write(json.dumps(user_entry, ensure_ascii=False) + '\n')

                    f.write(json.dumps(assistant_entry, ensure_ascii=False) + '\n')

                    f.flush()

                    os.fsync(f.fileno())  # Force write to disk


                # Atomic rename

                temp_file.replace(session_file)


            logger.info(f"[SESSION] Saved conversation turn to {session_file} (user: {len(user_message)} chars, assistant: {len(assistant_response)} chars)")

        except Exception as e:

            logger.error(f"[SESSION] Failed to save conversation to session file: {e}", exc_info=True)


    def _load_conversation_histories(self):

        """Load recent conversations from session files into memory on startup"""

        try:

            for session_file in self.sessions_dir.glob("*.jsonl"):

                conversation_id = session_file.stem

                try:

                    with open(session_file, 'r', encoding='utf-8') as f:

                        lines = f.readlines()

                        # Load last 20 messages (10 exchanges) into memory

                        messages = []

                        for line in lines[-20:]:

                            if line.strip():

                                msg = json.loads(line)

                                messages.append(msg)


                        if messages:

                            self.conversation_histories[conversation_id] = messages

                            logger.debug(f"Loaded {len(messages)} messages for conversation {conversation_id}")

                except Exception as e:

                    logger.warning(f"Failed to load conversation {conversation_id}: {e}")


            logger.info(f"Loaded {len(self.conversation_histories)} conversations from session files")

        except Exception as e:

            logger.error(f"Failed to load conversation histories: {e}", exc_info=True)


    # System now provides direct responses only, like Claude


    def _format_citations(self, memory_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        """Format memory results as citations"""

        citations = []

        for i, result in enumerate(memory_results[:5], 1):  # Top 5 as citations

            citations.append({

                "citation_id": i,

                "source": result.get("metadata", {}).get("source", "Memory"),

                "confidence": result.get("score", 0),

                "collection": result.get("collection", "unknown"),
                # v0.2.8: Full content, no truncation
                "text": result.get("text", ""),

                "doc_id": result.get("doc_id", "")

            })

        return citations

    async def _execute_tool_and_continue(
        self,
        tool_name: str,
        tool_args: dict,
        conversation_id: str,
        conversation_history: list,
        full_response: list,
        response_buffer: list,
        citations: list,
        in_thinking: bool,
        memory_tools: list,
        chain_depth: int = 0,
        max_depth: int = 3
    ):
        """
        Execute single tool call with optional chaining support.

        Handles all tool types (search_memory, create_memory, update_memory, archive_memory)
        and supports recursive chaining for search_memory results.

        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments from LLM
            conversation_id: Current conversation ID
            conversation_history: Full conversation history
            full_response: Accumulated response chunks
            response_buffer: Current batch buffer
            citations: Citations list to append to
            in_thinking: Whether currently in thinking mode
            memory_tools: Available tools for chaining
            chain_depth: Current recursion depth (0 = initial)
            max_depth: Maximum chain depth allowed

        Yields:
            Events: tool_start, tool_complete, token, etc.

        Returns:
            tuple: (tool_execution_record, tool_event_for_ui)
        """
        import json
        from datetime import datetime

        # Yield tool start event
        yield {
            "type": "tool_start",
            "tool": tool_name,
            "arguments": tool_args,
            "chain_depth": chain_depth
        }

        tool_execution_record = None
        tool_event_for_ui = None
        tool_results = None
        tool_response_content = ""

        # Execute tool based on type
        if tool_name == "search_memory":
            # Validate arguments
            query = tool_args.get("query", "")
            if not query or not query.strip():
                logger.warning(f"[TOOL] Empty query provided")
                query = "recent information"

            # Use LLM's collection choices (with validation)
            collections = tool_args.get("collections", ["all"])
            valid_collections = ['working', 'history', 'patterns', 'books', 'memory_bank', 'all']
            collections = [c for c in collections if c in valid_collections]
            if not collections:
                collections = ["all"]

            # Extract metadata filters (optional)
            metadata = tool_args.get("metadata", None)
            if metadata and isinstance(metadata, dict):
                logger.info(f"[TOOL] search_memory called with metadata filters: {metadata}")

            logger.info(f"[TOOL] search_memory called with collections={collections}, query='{query[:50]}...')")

            limit = tool_args.get("limit", 5)
            if not isinstance(limit, int) or limit < 1:
                limit = 5

            collections_param = None if "all" in collections else collections

            # Execute search with metadata filters
            tool_results = await self._search_memory_with_collections(
                query,
                collections_param,
                limit,
                metadata_filters=metadata
            )

            # Cache doc_ids for retrieved memory scoring (architecture.md line 565)
            if tool_results:
                doc_ids = [r.get("id") or r.get("doc_id") for r in tool_results if r.get("id") or r.get("doc_id")]
                if doc_ids:
                    _search_cache[conversation_id] = doc_ids
                    logger.debug(f"[SEARCH_CACHE] Cached {len(doc_ids)} doc_ids for conversation {conversation_id}")

            # Format results with metadata (v0.2.5: include book titles for LLM visibility)
            # v0.2.8: Full content, no truncation
            if tool_results:
                tool_response_content = "Found relevant memories:\n"
                for idx, r in enumerate(tool_results[:5], start=1):
                    content = r.get('text', r.get('content', ''))
                    collection = r.get('collection', 'unknown')
                    metadata = r.get('metadata', {})

                    # Include source context for books collection
                    source_info = ""
                    if collection == "books":
                        title = metadata.get('title') or metadata.get('book_title') or metadata.get('source_context')
                        author = metadata.get('author')
                        if title:
                            source_info = f" from \"{title}\""
                            if author:
                                source_info += f" by {author}"

                    tool_response_content += f"[{idx}] ({collection}{source_info}): {content}...\n"
                    citations.append(self._format_citation(r, idx))
            else:
                tool_response_content = "No relevant memories found for this query. I'll answer based on my general knowledge."
                logger.info(f"[TOOL] No memories found for query: {query}")

            # v0.2.4: Build preview from first 2-3 results for UI display
            result_preview = None
            if tool_results and len(tool_results) > 0:
                previews = []
                for r in tool_results[:3]:
                    # Try to get a short identifier (title, book name, or first 30 chars)
                    title = r.get('metadata', {}).get('title') or r.get('metadata', {}).get('book_title')
                    if title:
                        previews.append(title[:25])
                    else:
                        text_content = r.get('text', r.get('content', ''))[:25]
                        if text_content:
                            previews.append(text_content + '...' if len(text_content) == 25 else text_content)
                if previews:
                    result_preview = ', '.join(previews)

            tool_execution_record = {
                "tool": "search_memory",
                "status": "completed",
                "result_count": len(tool_results) if tool_results else 0,
                "result_preview": result_preview,
                "chain_depth": chain_depth
            }

            tool_event_for_ui = {
                "type": "tool_complete",
                "tool": "search_memory",
                "result_count": len(tool_results) if tool_results else 0,
                "result_preview": result_preview,
                "chain_depth": chain_depth
            }

        elif tool_name == "create_memory":
            content = tool_args.get("content", "")
            tag = tool_args.get("tag", "context")

            if content.strip() and self.memory:
                now = datetime.now().isoformat()
                await self.memory.store(
                    text=content,
                    collection="memory_bank",
                    metadata={
                        "tags": json.dumps([tag]),
                        "importance": 0.7,
                        "confidence": 0.8,
                        "status": "active",
                        "created_at": now,
                        "updated_at": now,
                        "mentioned_count": 1,
                        "added_by": "ai",
                        "conversation_id": conversation_id
                    }
                )
                logger.info(f"[MEMORY_BANK TOOL] Created: {content[:50]}... with tag={tag} (depth={chain_depth})")

                tool_execution_record = {
                    "tool": "create_memory",
                    "status": "success",
                    "chain_depth": chain_depth
                }

                tool_event_for_ui = {
                    "type": "tool_complete",
                    "tool": "create_memory",
                    "status": "success",
                    "chain_depth": chain_depth
                }

        elif tool_name == "update_memory":
            old_content = tool_args.get("old_content", "")
            new_content = tool_args.get("new_content", "")

            if old_content.strip() and new_content.strip() and self.memory:
                results = await self.memory.search_memory_bank(
                    query=old_content,
                    tags=None,
                    include_archived=False,
                    limit=1
                )

                if results:
                    doc_id = results[0].get("id")
                    await self.memory.update_memory_bank(
                        doc_id=doc_id,
                        new_text=new_content,
                        reason="llm_update"
                    )
                    logger.info(f"[MEMORY_BANK TOOL] Updated: {doc_id} -> {new_content[:50]}... (depth={chain_depth})")

                    tool_execution_record = {
                        "tool": "update_memory",
                        "status": "success",
                        "chain_depth": chain_depth
                    }

                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": "update_memory",
                        "status": "success",
                        "chain_depth": chain_depth
                    }
                else:
                    logger.warning(f"[MEMORY_BANK TOOL] No match found for: {old_content[:50]}... (depth={chain_depth})")

                    tool_execution_record = {
                        "tool": "update_memory",
                        "status": "not_found",
                        "chain_depth": chain_depth
                    }

                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": "update_memory",
                        "status": "not_found",
                        "chain_depth": chain_depth
                    }

        elif tool_name == "archive_memory":
            content = tool_args.get("content", "")

            if content.strip() and self.memory:
                results = await self.memory.search_memory_bank(
                    query=content,
                    tags=None,
                    include_archived=False,
                    limit=1
                )

                if results:
                    doc_id = results[0].get("id")
                    await self.memory.archive_memory_bank(doc_id)
                    logger.info(f"[MEMORY_BANK TOOL] Archived: {doc_id} (depth={chain_depth})")

                    tool_execution_record = {
                        "tool": "archive_memory",
                        "status": "success",
                        "chain_depth": chain_depth
                    }

                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": "archive_memory",
                        "status": "success",
                        "chain_depth": chain_depth
                    }
                else:
                    logger.warning(f"[MEMORY_BANK TOOL] No match found to archive: {content[:50]}... (depth={chain_depth})")

                    tool_execution_record = {
                        "tool": "archive_memory",
                        "status": "not_found",
                        "chain_depth": chain_depth
                    }

                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": "archive_memory",
                        "status": "not_found",
                        "chain_depth": chain_depth
                    }

        else:
            # v0.2.5: Handle external MCP tools
            from modules.mcp_client.manager import get_mcp_manager
            mcp_manager = get_mcp_manager()

            if mcp_manager and mcp_manager.is_external_tool(tool_name):
                logger.info(f"[MCP TOOL] Executing external tool: {tool_name}")

                success, result = await mcp_manager.execute_tool(tool_name, tool_args)

                if success:
                    tool_response_content = f"Tool '{tool_name}' result:\n{result}"
                    tool_execution_record = {
                        "tool": tool_name,
                        "status": "completed",
                        "result_preview": str(result)[:100] if result else None,
                        "chain_depth": chain_depth
                    }
                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": tool_name,
                        "status": "success",
                        "result_preview": str(result)[:100] if result else None,
                        "chain_depth": chain_depth
                    }
                    logger.info(f"[MCP TOOL] {tool_name} completed successfully")
                else:
                    tool_response_content = f"Tool '{tool_name}' failed: {result}"
                    tool_execution_record = {
                        "tool": tool_name,
                        "status": "failed",
                        "error": str(result),
                        "chain_depth": chain_depth
                    }
                    tool_event_for_ui = {
                        "type": "tool_complete",
                        "tool": tool_name,
                        "status": "failed",
                        "error": str(result),
                        "chain_depth": chain_depth
                    }
                    logger.error(f"[MCP TOOL] {tool_name} failed: {result}")
            else:
                # Unknown tool
                logger.warning(f"[TOOL] Unknown tool called: {tool_name}")
                tool_response_content = f"Unknown tool: {tool_name}"
                tool_execution_record = {
                    "tool": tool_name,
                    "status": "unknown",
                    "chain_depth": chain_depth
                }
                tool_event_for_ui = {
                    "type": "tool_complete",
                    "tool": tool_name,
                    "status": "unknown",
                    "chain_depth": chain_depth
                }

        # Yield tool complete event
        if tool_event_for_ui:
            yield tool_event_for_ui

        # v0.2.6: Track tool execution for Action KG (unified with MCP server)
        # Cache action until outcome is determined, then score it
        if tool_execution_record and tool_execution_record.get("status") != "unknown":
            action = ActionOutcome(
                action_type=tool_name,
                context_type="general",  # Default - could be enhanced to pass from caller
                outcome="unknown",  # Will be updated when user reaction is detected
                action_params=tool_args,
                collection=tool_args.get("collections", [None])[0] if tool_name == "search_memory" else None,
                doc_id=_search_cache.get(conversation_id, [None])[0] if tool_name == "search_memory" else None
            )
            _agent_action_cache.setdefault(conversation_id, []).append(action)
            logger.debug(f"[ACTION_KG] Cached action: {tool_name} for conversation {conversation_id}")

        # Handle continuation for search_memory and external MCP tools (regardless of result count, under depth limit)
        # v0.2.5: Also handle external tool results
        from modules.mcp_client.manager import get_mcp_manager
        mcp_mgr = get_mcp_manager()
        is_external_tool = mcp_mgr and mcp_mgr.is_external_tool(tool_name) if mcp_mgr else False

        if (tool_name == "search_memory" or is_external_tool) and chain_depth < max_depth and tool_response_content:
            logger.info(f"[CHAIN] Continuing with tool results at depth {chain_depth}")

            # Build conversation with results
            conversation_with_tools = conversation_history + [
                {"role": "assistant", "content": ''.join(full_response)},
                {"role": "system", "content": tool_response_content}
            ]

            # Continue streaming WITHOUT tools - LLM already has results, no need to search again
            chain_depth += 1
            continuation_prompt = "Continue your response using the tool results above."
            async for continuation_event in self.llm.stream_response_with_tools(
                prompt=continuation_prompt,
                history=conversation_with_tools,
                model=self.model_name,
                tools=None  # Don't pass tools on continuation - prevents recursive searches
            ):
                if continuation_event["type"] == "text":
                    # v0.2.5: Buffer only, no token yields
                    chunk = continuation_event["content"]
                    full_response.append(chunk)

                elif continuation_event["type"] == "tool_call":
                    # RECURSIVE: Handle chained tool calls
                    logger.info(f"[CHAIN] Nested tool call at depth {chain_depth}")
                    # SAFEGUARD: Limit nested tool calls per batch
                    MAX_TOOLS_PER_BATCH = 10
                    nested_tools_list = continuation_event.get("tool_calls", [])
                    if len(nested_tools_list) > MAX_TOOLS_PER_BATCH:
                        logger.warning(f"[CHAIN] Truncating {len(nested_tools_list)} nested tools to {MAX_TOOLS_PER_BATCH}")
                        nested_tools_list = nested_tools_list[:MAX_TOOLS_PER_BATCH]
                    
                    for nested_tool in nested_tools_list:
                        async for nested_event in self._execute_tool_and_continue(
                            tool_name=nested_tool["function"]["name"],
                            tool_args=nested_tool["function"]["arguments"],
                            conversation_id=conversation_id,
                            conversation_history=conversation_history,
                            full_response=full_response,
                            response_buffer=response_buffer,
                            citations=citations,
                            in_thinking=in_thinking,
                            memory_tools=memory_tools,
                            chain_depth=chain_depth
                        ):
                            yield nested_event

            # v0.2.5: Buffer flush removed - buffered response model

        # Yield execution record and UI event as final result (caller checks for tuple)
        yield (tool_execution_record, tool_event_for_ui)

    async def _extract_and_store_memory_bank_tags(self, clean_response: str, conversation_id: str) -> tuple[str, list]:

        """

        Extract [MEMORY_BANK: ...] tags from LLM response and store them to memory_bank collection.

        Also handles [MEMORY_BANK_UPDATE: ...] and [MEMORY_BANK_ARCHIVE: ...] tags.


        Args:

            clean_response: LLM response with thinking tags already removed

            conversation_id: Current conversation ID


        Returns:

            tuple: (response_without_tags, list_of_stored_entries)

        """

        memory_bank_entries = []


        # Pattern for CREATE: [MEMORY_BANK: tag="..." content="..."]

        create_pattern = r'\[MEMORY_BANK:\s*tag="([^"]+)"\s*content="((?:[^"\\]|\\.)*)"\]'


        # Pattern for UPDATE: [MEMORY_BANK_UPDATE: match="..." content="..."]

        update_pattern = r'\[MEMORY_BANK_UPDATE:\s*match="((?:[^"\\]|\\.)*)"\s*content="((?:[^"\\]|\\.)*)"\]'


        # Pattern for ARCHIVE: [MEMORY_BANK_ARCHIVE: match="..."]

        archive_pattern = r'\[MEMORY_BANK_ARCHIVE:\s*match="((?:[^"\\]|\\.)*)"\]'


        # Extract CREATE tags

        for match in re.finditer(create_pattern, clean_response):

            tag = match.group(1).strip()

            content = match.group(2).strip().replace('\\"', '"')

            memory_bank_entries.append({"action": "create", "tag": tag, "content": content})

            logger.info(f"[MEMORY_BANK] Detected CREATE: tag={tag}, content={content[:50]}...")


        # Extract UPDATE tags

        for match in re.finditer(update_pattern, clean_response):

            match_text = match.group(1).strip().replace('\\"', '"')

            new_content = match.group(2).strip().replace('\\"', '"')

            memory_bank_entries.append({"action": "update", "match": match_text, "content": new_content})

            logger.info(f"[MEMORY_BANK] Detected UPDATE: match={match_text[:30]}..., new content={new_content[:50]}...")


        # Extract ARCHIVE tags

        for match in re.finditer(archive_pattern, clean_response):

            match_text = match.group(1).strip().replace('\\"', '"')

            memory_bank_entries.append({"action": "archive", "match": match_text})

            logger.info(f"[MEMORY_BANK] Detected ARCHIVE: match={match_text[:50]}...")


        # Remove all memory bank tags from user-facing response

        if memory_bank_entries:

            clean_response = re.sub(create_pattern, '', clean_response)

            clean_response = re.sub(update_pattern, '', clean_response)

            clean_response = re.sub(archive_pattern, '', clean_response)

            clean_response = clean_response.strip()


        # Process memory bank operations

        if self.memory:

            for entry in memory_bank_entries:

                try:

                    import json

                    now = datetime.now().isoformat()


                    if entry["action"] == "create":

                        # Create new memory

                        await self.memory.store(

                            text=entry["content"],

                            collection="memory_bank",

                            metadata={

                                "tags": json.dumps([entry["tag"]]),

                                "importance": 0.7,

                                "confidence": 0.8,

                                "status": "active",

                                "created_at": now,

                                "updated_at": now,

                                "mentioned_count": 1,

                                "added_by": "ai",

                                "conversation_id": conversation_id

                            }

                        )

                        logger.info(f"[MEMORY_BANK] Created: {entry['content'][:50]}... with tag={entry['tag']}")


                    elif entry["action"] == "update":

                        # Find matching memory by semantic search

                        results = await self.memory.search_memory_bank(

                            query=entry["match"],

                            tags=None,

                            include_archived=False,

                            limit=1

                        )

                        if results:

                            doc_id = results[0].get("id")

                            await self.memory.update_memory_bank(

                                doc_id=doc_id,

                                new_text=entry["content"],

                                reason="llm_update"

                            )

                            logger.info(f"[MEMORY_BANK] Updated: {doc_id} -> {entry['content'][:50]}...")

                        else:

                            logger.warning(f"[MEMORY_BANK] UPDATE failed: no match found for '{entry['match'][:30]}...'")


                    elif entry["action"] == "archive":

                        # Find matching memory by semantic search

                        results = await self.memory.search_memory_bank(

                            query=entry["match"],

                            tags=None,

                            include_archived=False,

                            limit=1

                        )

                        if results:

                            doc_id = results[0].get("id")

                            await self.memory.archive_memory_bank(

                                doc_id=doc_id,

                                reason="llm_decision"

                            )

                            logger.info(f"[MEMORY_BANK] Archived: {doc_id}")

                        else:

                            logger.warning(f"[MEMORY_BANK] ARCHIVE failed: no match found for '{entry['match'][:30]}...'")


                except Exception as e:

                    logger.error(f"[MEMORY_BANK] Failed to process {entry.get('action', 'unknown')} operation: {e}")


        return clean_response, memory_bank_entries


# Global service instance
agent_service: Optional[AgentChatService] = None
_service_init_lock = asyncio.Lock()  # Prevent race condition on service initialization


# Background task for async generation with WebSocket streaming
async def _run_generation_task(
    conversation_id: str,
    request: 'AgentChatRequest',
    user_id: str,
    app_state: Any = None
):
    """Background task for async LLM generation with WebSocket streaming and timeout handling."""
    global agent_service

    # Small delay to ensure WebSocket connection is established
    await asyncio.sleep(0.5)

    # Get WebSocket connection if available
    websocket = None
    if app_state and hasattr(app_state, 'websockets'):
        websocket = app_state.websockets.get(conversation_id)
        logger.info(f"[WebSocket] Found connection for {conversation_id}: {websocket is not None}")
        if websocket:
            logger.info(f"[WebSocket] Connection state: {websocket.client_state if hasattr(websocket, 'client_state') else 'unknown'}")
    else:
        logger.warning(f"[WebSocket] No websockets dict in app_state for {conversation_id}")

    try:
        # Initialize task status
        with _task_lock:
            _generation_tasks[conversation_id] = {
                'status': 'thinking',
                'thinking': None,
                'tool_executions': [],
                'response': None,
                'error': None,
                'started_at': datetime.now().isoformat()
            }

        # Send initial status via WebSocket
        if websocket:
            try:
                logger.info(f"[WebSocket] Sending initial status to {conversation_id}")
                await websocket.send_json({
                    "type": "status",
                    "status": "thinking",
                    "message": "Processing your request..."
                })
                logger.info(f"[WebSocket] Initial status sent successfully to {conversation_id}")
            except Exception as e:
                logger.error(f"[WebSocket] Failed to send initial status to {conversation_id}: {e}")
                websocket = None  # Connection failed, disable WebSocket

        # Set 2-minute timeout to prevent DeepSeek-R1/Qwen hangs
        # Use stream_message and accumulate events (batch fallback mode)
        final_response = []
        thinking = None
        citations = []

        async def accumulate_stream():
            nonlocal thinking
            async for event in agent_service.stream_message(
                message=request.message,
                conversation_id=conversation_id,
                app_state=app_state
            ):
                # v0.2.5: Buffered response model
                if event["type"] == "response":
                    final_response.append(event["content"])
                elif event["type"] == "thinking":
                    thinking = event["content"]
                elif event["type"] == "done":
                    nonlocal citations
                    citations = event.get("citations", [])

        await asyncio.wait_for(accumulate_stream(), timeout=120.0)

        # Join accumulated response
        final_response = ''.join(final_response)

        # Stream response via WebSocket
        if websocket and final_response:
            try:
                logger.info(f"[WebSocket] Sending response content to {conversation_id}")
                await websocket.send_json({
                    "type": "content",
                    "content": final_response,
                    "citations": citations
                })
                logger.info(f"[WebSocket] Response content sent successfully")
            except Exception as e:
                logger.error(f"[WebSocket] Failed to send response: {e}")

        # Mark as complete
        with _task_lock:
            _generation_tasks[conversation_id]['status'] = 'complete'
            _generation_tasks[conversation_id]['response'] = final_response
            _generation_tasks[conversation_id]['thinking'] = thinking
            _generation_tasks[conversation_id]['completed_at'] = datetime.now().isoformat()

        # Send completion via WebSocket
        if websocket:
            try:
                await websocket.send_json({
                    "type": "complete",
                    "conversation_id": conversation_id
                })
            except:
                pass

    except asyncio.TimeoutError:
        logger.error(f'Generation timeout for conversation {conversation_id}')
        error_msg = 'Generation timed out after 2 minutes'

        with _task_lock:
            _generation_tasks[conversation_id]['status'] = 'error'
            _generation_tasks[conversation_id]['error'] = error_msg

        if websocket:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": error_msg
                })
            except:
                pass

    except Exception as e:
        logger.error(f'Generation error for conversation {conversation_id}: {e}', exc_info=True)
        error_msg = str(e)

        with _task_lock:
            _generation_tasks[conversation_id]['status'] = 'error'
            _generation_tasks[conversation_id]['error'] = error_msg

        if websocket:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": error_msg
                })
            except:
                pass


# Background task for async generation with WebSocket streaming (new streaming version)
async def _run_generation_task_streaming(
    conversation_id: str,
    request: 'AgentChatRequest',
    user_id: str,
    app_state: Any = None
):
    """Streaming version of generation task - runs alongside existing."""
    global agent_service

    # Get WebSocket
    websocket = None
    if app_state and hasattr(app_state, 'websockets'):
        websocket = app_state.websockets.get(conversation_id)

    if not websocket:
        logger.warning(f"No WebSocket for streaming to {conversation_id}, falling back to batch")
        # Fall back to non-streaming
        return await _run_generation_task(conversation_id, request, user_id, app_state)

    try:
        # Initialize tracking
        with _task_lock:
            _generation_tasks[conversation_id] = {
                'status': 'streaming',
                'started_at': datetime.now().isoformat()
            }

        # Send stream start
        await websocket.send_json({
            "type": "stream_start",
            "conversation_id": conversation_id
        })

        # Accumulate for session file
        full_response = []
        thinking_content = ""
        tool_executions = []
        citations = []

        # Stream from agent service
        # IMPORTANT: Must fully consume generator to ensure session file is saved
        # Even if WebSocket disconnects, we continue to drain the generator
        websocket_disconnected = False
        async for event in agent_service.stream_message(
            message=request.message,
            conversation_id=conversation_id,
            app_state=app_state
        ):
            # Check if WebSocket is still connected
            if not websocket_disconnected and (not websocket or (hasattr(websocket, 'client_state') and websocket.client_state.name != 'CONNECTED')):
                logger.info(f"[DISCONNECT] WebSocket disconnected for {conversation_id}, continuing to save session")
                websocket_disconnected = True
                # DON'T break - continue consuming generator to ensure save happens

            # Send events to WebSocket (skip if disconnected)
            # v0.2.5 RESTORED: Stream tokens for interleaved tool/response display
            if event["type"] == "token":
                # Stream token for real-time display
                full_response.append(event["content"])
                if not websocket_disconnected:
                    await websocket.send_json(event)

            elif event["type"] == "thinking":
                # Thinking content from buffered response
                if not websocket_disconnected:
                    await websocket.send_json({
                        "type": "thinking",
                        "content": event["content"]
                    })

            elif event["type"] == "response":
                # Complete response from buffered model
                full_response.append(event["content"])
                if not websocket_disconnected:
                    await websocket.send_json({
                        "type": "response",
                        "content": event["content"]
                    })

            elif event["type"] == "tool_start":
                tool_executions.append({
                    "tool": event["tool"],
                    "status": "running",
                    "arguments": event.get("arguments")
                })
                if not websocket_disconnected:
                    await websocket.send_json(event)

            elif event["type"] == "tool_complete":
                # Update tool status
                for tool in tool_executions:
                    if tool["tool"] == event["tool"] and tool["status"] == "running":
                        tool["status"] = "completed"
                        break
                if not websocket_disconnected:
                    await websocket.send_json(event)

            elif event["type"] == "memory_searched":
                if not websocket_disconnected:
                    await websocket.send_json({
                        "type": "status",
                        "status": "memory_search",
                        "message": f"Searched {event['count']} memories from {', '.join(event['collections'])}"
                    })

            elif event["type"] == "title":
                # Forward title event from Layer 1 to frontend
                if not websocket_disconnected:
                    await websocket.send_json(event)


            elif event["type"] == "done" or event["type"] == "stream_complete":
                # Just forward the stream_complete event from Layer 1 (generator)
                # Layer 1 already includes citations, memory_updated flag, and timestamp
                if not websocket_disconnected:
                    if "content" in event:
                        # Validation error - send as dedicated event
                        logger.info(f"Done event with validation error: {event['content']}")
                        await websocket.send_json({
                            "type": "validation_error",
                            "message": event["content"]
                        })
                    else:
                        # Normal completion - forward with conversation_id
                        await websocket.send_json({
                            **event,  # Forward all fields from Layer 1
                            "conversation_id": conversation_id
                        })

            elif event["type"] == "error":
                raise Exception(event["message"])

        # Note: Session file already saved by Layer 1 (stream_message generator)
        # No duplicate save needed here - eliminated tech debt!


        # Update task status
        with _task_lock:
            _generation_tasks[conversation_id]['status'] = 'complete'
            _generation_tasks[conversation_id]['response'] = ''.join(full_response)
            # thinking_content removed (v0.2.5)
            _generation_tasks[conversation_id]['completed_at'] = datetime.now().isoformat()

    except asyncio.CancelledError:
        logger.info(f'[CANCEL] Generation cancelled for {conversation_id}')
        if websocket:
            try:
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Generation cancelled by user"
                })
            except:
                pass  # WebSocket may already be closed
        raise  # Re-raise to properly mark task as cancelled
    except asyncio.TimeoutError:
        logger.error(f'Streaming timeout for {conversation_id}')
        if websocket:
            await websocket.send_json({
                "type": "error",
                "message": "Generation timed out",
                "code": "TIMEOUT"
            })
        # Fall back to batch
        await _run_generation_task(conversation_id, request, user_id, app_state)
    except Exception as e:
        logger.error(f'Streaming failed for {conversation_id}: {e}', exc_info=True)
        if websocket:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "code": "STREAM_ERROR"
            })
        # Fall back to batch
        await _run_generation_task(conversation_id, request, user_id, app_state)


# Removed startup event - will use lazy initialization in endpoints instead

# The startup event runs before app.state is populated by lifespan


@router.get("/progress/{conversation_id}")

async def get_generation_progress(conversation_id: str):

    """Poll endpoint for checking generation progress."""

    with _task_lock:

        task_info = _generation_tasks.get(conversation_id)

        

        if not task_info:

            raise HTTPException(status_code=404, detail="Task not found")

        

        # Return current progress

        return {

            "status": task_info["status"],

            "thinking": task_info.get("thinking"),

            "tool_executions": task_info.get("tool_executions", []),

            "response": task_info.get("response"),

            "error": task_info.get("error"),

            "started_at": task_info.get("started_at"),

            "completed_at": task_info.get("completed_at")

        }


@router.post("/stream")

async def agent_chat_stream(request: AgentChatRequest, req: Request):

    """

    Start async generation task and return conversation_id for polling.

    """

    global agent_service


    if not agent_service:

        agent_service = AgentChatService(

            memory=req.app.state.memory,

            llm=req.app.state.llm_client

        )

    else:

        agent_service.llm = req.app.state.llm_client


    conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(8)}"

    agent_service.memory.conversation_id = conversation_id

    

    # Get user ID from request or default
    user_id = getattr(request, 'user_id', 'default_user')

    # Check feature flag for streaming
    use_streaming = get_flag_manager().is_enabled("ENABLE_WEBSOCKET_STREAMING")

    # Debug WebSocket availability
    has_websockets = hasattr(req.app.state, 'websockets')
    websocket_count = len(req.app.state.websockets) if has_websockets else 0
    has_this_websocket = req.app.state.websockets.get(conversation_id) is not None if has_websockets else False

    logger.info(f"WebSocket check - has_websockets: {has_websockets}, count: {websocket_count}, has {conversation_id}: {has_this_websocket}")

    # Start appropriate generation task and track it for cancellation
    if use_streaming and has_websockets and has_this_websocket:
        logger.info(f"Using streaming generation for conversation {conversation_id}")
        # Use streaming version
        task = asyncio.create_task(
            _run_generation_task_streaming(conversation_id, request, user_id, req.app.state)
        )
    else:
        logger.info(f"Using batch generation for conversation {conversation_id} (streaming={'enabled' if use_streaming else 'disabled'})")
        # Use existing batch version
        task = asyncio.create_task(
            _run_generation_task(conversation_id, request, user_id, req.app.state)
        )

    # Cancel existing task for this conversation (prevents race condition)
    existing_task = _active_tasks.get(conversation_id)
    if existing_task and not existing_task.done():
        existing_task.cancel()

    # Store task handle for cancellation
    _active_tasks[conversation_id] = task

    # Clean up task handle when done (only if still current)
    def cleanup_task(t):
        if _active_tasks.get(conversation_id) == t:
            _active_tasks.pop(conversation_id, None)
    task.add_done_callback(cleanup_task)

    # Return conversation_id immediately for polling
    # Frontend can poll /progress/{conversation_id} or use WebSocket for real-time updates
    return {
        "conversation_id": conversation_id,
        "status": "started",
        "streaming": use_streaming
    }

# Removed 1370 lines of dead SSE code (unreachable event_generator function after return statement)
# WebSocket streaming is now used instead - see _run_generation_task() above


@router.post("/cancel/{conversation_id}")
async def cancel_generation(conversation_id: str, req: Request):
    """Cancel active generation task for a conversation"""

    # Cancel the asyncio task
    task = _active_tasks.get(conversation_id)
    if task and not task.done():
        task.cancel()
        logger.info(f"[CANCEL] Cancelled generation task for {conversation_id}")

        # Close WebSocket if exists
        if hasattr(req.app.state, 'websockets'):
            ws = req.app.state.websockets.pop(conversation_id, None)
            if ws:
                try:
                    await ws.close()
                    logger.info(f"[CANCEL] Closed WebSocket for {conversation_id}")
                except Exception as e:
                    logger.warning(f"[CANCEL] Failed to close WebSocket: {e}")

        # Clean up task status
        with _task_lock:
            if conversation_id in _generation_tasks:
                _generation_tasks[conversation_id]['status'] = 'cancelled'

        return {"status": "cancelled", "conversation_id": conversation_id}

    return {"status": "not_found", "conversation_id": conversation_id}


@router.post("/cleanup-sessions")

async def cleanup_sessions(req: Request):

    """

    Clean up old session files to prevent accumulation.

    Keeps sessions from last 30 days by default.

    """

    try:

        data = await req.json()

        days_to_keep = data.get("days_to_keep", 30)

        dry_run = data.get("dry_run", False)


        # Use AppData paths, not bundled data folder
        memory = req.app.state.memory
        sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")

        if not sessions_dir.exists():

            return {"status": "success", "message": "No sessions directory found", "cleaned": 0}


        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        cleaned = 0

        kept = 0


        for session_file in sessions_dir.glob("*.jsonl"):

            try:

                # Extract timestamp from filename (conv_YYYYMMDD_HHMMSS_...)

                filename = session_file.stem

                if filename.startswith("conv_"):

                    parts = filename.split("_")

                    if len(parts) >= 3:

                        date_str = parts[1]

                        time_str = parts[2]

                        session_date = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")


                        if session_date < cutoff_date:

                            if not dry_run:

                                session_file.unlink()

                            cleaned += 1

                        else:

                            kept += 1

            except Exception as e:

                logger.warning(f"Error processing session file {session_file}: {e}")

                continue


        return {

            "status": "success",

            "cleaned": cleaned,

            "kept": kept,

            "dry_run": dry_run,

            "cutoff_date": cutoff_date.isoformat()

        }


    except Exception as e:

        logger.error(f"Error during session cleanup: {e}")

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch-conversation")

async def switch_conversation(req: Request):

    """

    Handle conversation switching with memory promotion

    """

    global agent_service


    if not agent_service:

        agent_service = AgentChatService(

            memory=req.app.state.memory,

            llm=req.app.state.llm_client

        )


    # Use conversation lock to prevent race conditions

    async with agent_service.conversation_lock:

        # Get the request body

        body = await req.json()

        old_id = body.get("old_conversation_id", "")

        new_id = body.get("new_conversation_id", "")


        # Allow null for clearing session (lazy conversation creation)
        if new_id is None:
            logger.info(f"Clearing session {old_id}, no new conversation yet")
            # Still promote old conversation's memories
            if agent_service.memory and old_id:
                asyncio.create_task(
                    agent_service.memory._promote_valuable_working_memory()
                )
            return {
                "status": "success",
                "old_conversation": old_id,
                "new_conversation": None,
                "message": "Session cleared (new conversation will be created on first message)"
            }


        logger.info(f"Switching conversation from {old_id} to {new_id}")


        # Trigger memory promotion asynchronously (don't block response)

        if agent_service.memory and old_id:

            asyncio.create_task(

                agent_service.memory._promote_valuable_working_memory()

            )

            logger.info(f"Memory promotion queued for conversation {old_id}")


            # Update conversation ID immediately

            agent_service.memory.conversation_id = new_id

            logger.info(f"Memory system conversation ID updated to {new_id}")


        return {

            "status": "success",

            "old_conversation": old_id,

            "new_conversation": new_id,

            "message": f"Switched from {old_id} to {new_id} (memory promotion queued)"

        }


@router.post("/create-conversation")

async def create_conversation(req: Request):

    """

    Create a new conversation ID for the UI and initialize session file

    """

    # Use cryptographically secure random ID to prevent enumeration

    conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(8)}"


    # Create empty session file immediately to prevent phantom conversations
    # Use AppData path from memory system if available
    memory = req.app.state.memory
    if memory and hasattr(memory, 'data_dir'):
        sessions_dir = memory.data_dir / "sessions"
    else:
        sessions_dir = Path("data/sessions")

    session_file = sessions_dir / f"{conversation_id}.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.touch()  # Create empty file

    logger.info(f"[SESSION] Created new conversation: {conversation_id} at {session_file}")


    return {

        "conversation_id": conversation_id,

        "created_at": datetime.now().isoformat()

    }


@router.post("/generate-title")

async def generate_title(req: Request):

    """

    Generate a title for a conversation using the LLM

    """

    body = await req.json()

    conversation_id = body.get("conversation_id", "")

    messages = body.get("messages", [])


    if not messages:

        # Fallback if no messages provided

        return {

            "title": f"Chat {conversation_id[-8:] if conversation_id else 'Session'}",

            "fallback": True

        }


    try:

        # Create a prompt for title generation

        conversation_text = "\n".join([

            f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"  # Limit each message to 200 chars

            for msg in messages[:4]  # Use first 4 messages for context

        ])


        title_prompt = f"""Generate a brief, descriptive title (3-6 words) for this conversation:


{conversation_text}


Title should be specific and capture the main topic. Examples:

- Fix Authentication Error

- Memory System Architecture

- Debug WebSocket Connection

- Roampal Setup Guide


Respond with ONLY the title, nothing else."""


        # Get LLM client from app state (use current model, don't create new instance)
        if not hasattr(req.app.state, 'llm_client') or not req.app.state.llm_client:
            raise HTTPException(status_code=503, detail="LLM service not initialized")

        llm_client = req.app.state.llm_client


        # Generate title using generate_response

        title_response = await llm_client.generate_response(

            prompt=title_prompt,

            history=[],  # No history needed for title generation

            system_prompt="You are a concise title generator. Respond with ONLY a short title (3-6 words), nothing else."

        )


        if title_response and title_response.strip():

            # Clean up the title - remove thinking tags if present
            title = title_response.strip()

            # Remove <think> tags if model included them (uses utility for consistency)
            _, title = extract_thinking(title)


            # Remove quotes if present

            title = title.strip('"').strip("'").strip()


            # Remove markdown formatting

            title = title.replace('**', '').replace('*', '').strip()


            # If title still has multiple lines, take first line

            if '\n' in title:

                title = title.split('\n')[0].strip()


            # Limit length

            if len(title) > 50:

                title = title[:47] + "..."


            # Fallback if empty after cleaning

            if not title:

                raise Exception("Title empty after cleaning")


            # Update the session file with the new title

            from modules.memory.file_memory_adapter import FileMemoryAdapter

            from pathlib import Path

            if hasattr(self, 'memory') and self.memory:
                await self.memory.file_adapter.update_session_title(conversation_id, title)
            else:
                logger.warning(f"[PERSIST] Cannot update title: memory system not available")


            logger.info(f"Generated title for {conversation_id}: {title}")

            return {

                "title": title,

                "fallback": False

            }

        else:

            raise Exception("Empty response from LLM")


    except Exception as e:

        logger.warning(f"Title generation failed: {e}", exc_info=True)


        # Intelligent fallback based on first message

        try:

            if messages and len(messages) > 0:

                first_user_msg = next((m for m in messages if m.get('role') == 'user'), None)

                if first_user_msg:

                    # Use first 50 chars of first user message

                    fallback_title = first_user_msg.get('content', '')[:50].strip()

                    if fallback_title:

                        return {

                            "title": fallback_title,

                            "fallback": True

                        }

        except:

            pass


        # Final fallback

        return {

            "title": f"Chat {conversation_id[-8:] if conversation_id else 'Session'}",

            "fallback": True

        }


@router.get("/feature-mode")

async def get_feature_mode():

    """

    Return the current feature mode (memory-only for RoamPal)

    """

    return {

        "mode": "memory",

        "features": {

            "memory": True,

            "tools": False,

            "actions": False

        }

    }


@router.get("/stats")

async def get_chat_stats(request: Request):

    """

    Get memory system statistics for the current conversation

    """

    try:

        # Try to get chat_service from app state first, then fall back to global

        chat_service = getattr(request.app.state, 'chat_service', None) or agent_service


        if not chat_service or not chat_service.memory:

            return {

                "status": "error",

                "message": "Memory system not initialized"

            }


        # Get current conversation ID (default if not available)

        conversation_id = 'default'

        if hasattr(chat_service, 'session_manager'):

            conversation_id = getattr(chat_service.session_manager, 'current_conversation_id', 'default')


        # Get memory stats - get_stats doesn't take conversation_id

        stats = chat_service.memory.get_stats()


        # Add learning-specific metrics

        if hasattr(chat_service.memory, 'outcome_detector') and chat_service.memory.outcome_detector:

            stats['learning'] = {

                'outcome_detection_enabled': True,

                'knowledge_graph_active': bool(stats.get('knowledge_graph', {}))

            }

        else:

            stats['learning'] = {

                'outcome_detection_enabled': False,

                'knowledge_graph_active': False

            }


        logger.debug(f"Chat stats retrieved for conversation {conversation_id}: {stats}")

        return stats


    except Exception as e:

        logger.error(f"Failed to get chat stats: {e}")

        return {

            "status": "error",

            "message": str(e)

        }


