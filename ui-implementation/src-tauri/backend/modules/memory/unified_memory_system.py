"""
UnifiedMemorySystem Facade - Refactored Version

This is the coordinating facade that delegates to individual services.
Extracted from the original 4,746-line monolith into composable services.

Services:
- ScoringService: Wilson score, dynamic weights
- KnowledgeGraphService: Routing KG, content graph, relationships
- RoutingService: Query routing, tier scores
- SearchService: Hybrid search, entity boost
- PromotionService: Memory promotion/demotion
- OutcomeService: Outcome recording, score updates
- MemoryBankService: User identity/preferences
- ContextService: Conversation context analysis
"""

import logging
import math
import re
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from .config import MemoryConfig
from .types import CollectionName, MemoryResult, SearchMetadata, ActionOutcome
from .scoring_service import ScoringService
from .knowledge_graph_service import KnowledgeGraphService
from .routing_service import RoutingService
from .search_service import SearchService
from .promotion_service import PromotionService
from .outcome_service import OutcomeService
from .memory_bank_service import MemoryBankService
from .context_service import ContextService

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff for async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator


class UnifiedMemorySystem:
    """
    Facade for the unified memory system.

    Coordinates all memory operations through specialized services.

    5 Collections:
    - books: Uploaded reference material (never decays)
    - working: Current session context (session-scoped)
    - history: Past conversations (auto-promoted to patterns)
    - patterns: Proven solutions (what actually worked)
    - memory_bank: Persistent project/user context (LLM-controlled)
    """

    def __init__(
        self,
        data_dir: str = "./data",
        use_server: bool = True,
        llm_service: Any = None,
        embedding_service: Any = None,
        file_adapter: Any = None,
        chromadb_adapter_factory: Any = None,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize UnifiedMemorySystem.

        Args:
            data_dir: Base data directory
            use_server: Whether to use ChromaDB server mode
            llm_service: Optional LLM service for scoring
            embedding_service: Embedding service (injected for testing)
            file_adapter: File adapter (injected for testing)
            chromadb_adapter_factory: Factory for ChromaDB adapters
            config: Memory configuration
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.config = config or MemoryConfig()
        self.use_server = use_server
        self.llm_service = llm_service
        self.initialized = False

        # Expose thresholds for backward compatibility
        self.HIGH_VALUE_THRESHOLD = self.config.high_value_threshold
        self.PROMOTION_SCORE_THRESHOLD = self.config.promotion_score_threshold
        self.DEMOTION_SCORE_THRESHOLD = self.config.demotion_score_threshold
        self.DELETION_SCORE_THRESHOLD = self.config.deletion_score_threshold
        self.NEW_ITEM_DELETION_THRESHOLD = self.config.new_item_deletion_threshold

        # Injected dependencies (or lazy-loaded)
        self._embedding_service = embedding_service
        self._file_adapter = file_adapter
        self._chromadb_adapter_factory = chromadb_adapter_factory

        # Collections
        self.collections: Dict[str, Any] = {}

        # Conversation tracking
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_context = {}
        self.message_count = 0
        self._promotion_lock = asyncio.Lock()
        self._background_tasks = []

        # Services (initialized lazily after collections are set up)
        self._scoring_service: Optional[ScoringService] = None
        self._kg_service: Optional[KnowledgeGraphService] = None
        self._routing_service: Optional[RoutingService] = None
        self._search_service: Optional[SearchService] = None
        self._promotion_service: Optional[PromotionService] = None
        self._outcome_service: Optional[OutcomeService] = None
        self._memory_bank_service: Optional[MemoryBankService] = None
        self._context_service: Optional[ContextService] = None

    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from modules.embedding.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @embedding_service.setter
    def embedding_service(self, value):
        """Allow injection of embedding service for testing."""
        self._embedding_service = value

    @property
    def file_adapter(self):
        """Lazy load file adapter."""
        if self._file_adapter is None:
            from modules.memory.file_memory_adapter import FileMemoryAdapter
            self._file_adapter = FileMemoryAdapter()
        return self._file_adapter

    async def initialize(self):
        """Initialize all components and services."""
        if self.initialized:
            return

        logger.info("Initializing UnifiedMemorySystem...")

        # Initialize file adapter first (needed for title generation, session management)
        from modules.memory.file_memory_adapter import FileMemoryAdapter
        self._file_adapter = FileMemoryAdapter()
        await self._file_adapter.initialize({"base_data_path": str(self.data_dir)})

        # v0.2.10: Migrate ChromaDB schema before initialization
        # Handles upgrades from ChromaDB 0.4.x to 1.x
        self._migrate_chromadb_schema()

        # Initialize collections
        await self._initialize_collections()

        # Initialize services
        self._init_services()

        self.initialized = True
        logger.info("UnifiedMemorySystem initialized")

    async def _initialize_collections(self):
        """Initialize ChromaDB collections."""
        collection_names = ["books", "working", "history", "patterns", "memory_bank"]

        if self._chromadb_adapter_factory:
            # Use injected factory (for testing)
            for name in collection_names:
                self.collections[name] = self._chromadb_adapter_factory(name)
        else:
            # Default production initialization
            from modules.memory.chromadb_adapter import ChromaDBAdapter

            for name in collection_names:
                self.collections[name] = ChromaDBAdapter(
                    persistence_directory=str(self.data_dir / "chromadb"),
                    use_server=self.use_server
                )
                await self.collections[name].initialize(collection_name=f"roampal_{name}")

    def _migrate_chromadb_schema(self):
        """
        Migrate ChromaDB schema for compatibility across versions.

        v0.2.10: ChromaDB 1.x added 'topic' column to collections table.
        Users upgrading from ChromaDB 0.4.x/0.5.x will have old schema.
        This safely adds missing columns without affecting existing data.
        """
        import sqlite3

        chromadb_path = self.data_dir / "chromadb"
        sqlite_path = chromadb_path / "chroma.sqlite3"

        if not sqlite_path.exists():
            logger.debug("No existing ChromaDB - skipping migration")
            return

        try:
            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.cursor()

            # Columns added in ChromaDB 1.x that may be missing
            migrations_needed = []

            # Check collections table
            cursor.execute("PRAGMA table_info(collections)")
            collections_columns = {col[1] for col in cursor.fetchall()}
            if 'topic' not in collections_columns:
                migrations_needed.append(('collections', 'topic', 'TEXT'))

            # Check segments table (also needs 'topic' in ChromaDB 1.x)
            cursor.execute("PRAGMA table_info(segments)")
            segments_columns = {col[1] for col in cursor.fetchall()}
            if 'topic' not in segments_columns:
                migrations_needed.append(('segments', 'topic', 'TEXT'))

            # Apply migrations
            for table, column, col_type in migrations_needed:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    logger.info(f"ChromaDB migration: Added {column} to {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        pass  # Column already exists, safe to ignore
                    else:
                        raise

            conn.commit()
            conn.close()
            logger.debug("ChromaDB schema migration completed")

        except Exception as e:
            logger.warning(f"ChromaDB schema migration failed (non-fatal): {e}")
            # Non-fatal: if migration fails, ChromaDB may still work
            # or will fail with a more specific error later

    def _init_services(self):
        """Initialize all extracted services."""
        # Scoring service (no dependencies)
        self._scoring_service = ScoringService(self.config)

        # KG service
        self._kg_service = KnowledgeGraphService(
            kg_path=self.data_dir / "knowledge_graph.json",
            content_graph_path=self.data_dir / "content_graph.json",
            relationships_path=self.data_dir / "memory_relationships.json",
            config=self.config
        )

        # Routing service
        self._routing_service = RoutingService(
            kg_service=self._kg_service,
            config=self.config
        )

        # Search service
        self._search_service = SearchService(
            collections=self.collections,
            embed_fn=self._embed_text,
            scoring_service=self._scoring_service,
            routing_service=self._routing_service,
            kg_service=self._kg_service,
            config=self.config
        )

        # Promotion service
        self._promotion_service = PromotionService(
            collections=self.collections,
            embed_fn=self._embed_text,
            add_relationship_fn=self._add_relationship,
            config=self.config
        )

        # Outcome service
        self._outcome_service = OutcomeService(
            collections=self.collections,
            kg_service=self._kg_service,
            promotion_service=self._promotion_service,
            config=self.config
        )

        # Memory bank service
        if "memory_bank" in self.collections:
            self._memory_bank_service = MemoryBankService(
                collection=self.collections["memory_bank"],
                embed_fn=self._embed_text,
                search_fn=self.search,
                config=self.config
            )

        # Context service
        self._context_service = ContextService(
            collections=self.collections,
            kg_service=self._kg_service,
            embed_fn=self._embed_text,
            config=self.config
        )

    async def _embed_text(self, text: str) -> List[float]:
        """Embed text using the embedding service."""
        return await self.embedding_service.embed_text(text)

    async def _add_relationship(self, doc_id: str, rel_type: str, data: Dict[str, Any]):
        """Add a relationship to the KG service."""
        if self._kg_service:
            self._kg_service.add_relationship(doc_id, rel_type, data)

    async def _generate_contextual_prefix(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
        collection: str
    ) -> str:
        """
        Generate context-aware prefix for better retrieval (Anthropic Contextual Retrieval, Sep 2024).
        Reduces retrieval failures by 49% (67% with reranking).

        Args:
            text: Original memory text
            metadata: Memory metadata
            collection: Collection name

        Returns:
            Contextualized text with prefix explaining what this memory is about
        """
        # Skip contextual prefix for very short text or if LLM service unavailable
        if len(text) < 50 or not self.llm_service:
            return text

        try:
            # Build context from metadata
            context_parts = []

            if metadata:
                # Extract conversation context
                if metadata.get("conversation_id"):
                    context_parts.append(f"Conversation: {metadata['conversation_id']}")

                # Extract tags/categories
                if metadata.get("tags"):
                    tags = metadata["tags"]
                    if isinstance(tags, list):
                        context_parts.append(f"Tags: {', '.join(tags[:3])}")

                # Extract importance/purpose
                if metadata.get("importance"):
                    importance = metadata["importance"]
                    if importance >= 0.9:
                        context_parts.append("High importance")

            # Add collection type
            collection_context = {
                "memory_bank": "user memory",
                "patterns": "proven solution pattern",
                "working": "recent conversation",
                "history": "past conversation",
                "books": "reference material"
            }
            context_parts.append(collection_context.get(collection, collection))

            # Build prompt
            context_str = ", ".join(context_parts)
            prompt = f"""Given this context and memory chunk, write ONE concise sentence explaining what this memory is about.

Context: {context_str}
Chunk: {text[:300]}

Prefix (one sentence, max 20 words):"""

            # Use fast model for speed (timeout 5s to avoid blocking)
            try:
                response = await asyncio.wait_for(
                    self.llm_service.generate(prompt, max_tokens=50),
                    timeout=5.0
                )
                prefix = response.strip().strip('"').strip("'")

                # Validate prefix is reasonable
                if len(prefix) > 10 and len(prefix) < 200:
                    contextual_text = f"{prefix} {text}"
                    logger.debug(f"[CONTEXTUAL] Generated prefix: {prefix[:50]}...")
                    return contextual_text

            except asyncio.TimeoutError:
                logger.debug("[CONTEXTUAL] LLM timeout, using original text")

        except Exception as e:
            logger.debug(f"[CONTEXTUAL] Prefix generation failed: {e}, using original text")

        # Fallback to original text
        return text

    # ==================== Core API ====================

    @with_retry(max_attempts=3, delay=0.5)
    async def store(
        self,
        text: str,
        collection: CollectionName = "working",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store text in a collection.

        Args:
            text: Text to store
            collection: Target collection
            metadata: Optional metadata

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        doc_id = f"{collection}_{uuid.uuid4().hex[:8]}"

        # Build metadata
        final_metadata = {
            "text": text,
            "content": text,
            "score": 0.5,  # Initial score
            "uses": 0,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": self.conversation_id,
            **(metadata or {})
        }

        # Generate contextual embedding (Anthropic Contextual Retrieval)
        contextual_text = await self._generate_contextual_prefix(text, final_metadata, collection)
        embedding = await self._embed_text(contextual_text)

        # Store in collection
        await self.collections[collection].upsert_vectors(
            ids=[doc_id],
            vectors=[embedding],
            metadatas=[final_metadata]
        )

        logger.debug(f"Stored in {collection}: {doc_id}")
        return doc_id

    async def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0,
        return_metadata: bool = False,
        use_hybrid: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None,  # v0.2.9: Expose metadata filtering
        transparency_context: Optional[Any] = None  # v0.2.9: Transparency tracking
    ) -> List[Dict[str, Any]]:
        """
        Search across collections.

        Args:
            query: Search query
            collections: Collections to search (None = auto-route)
            limit: Maximum results
            return_metadata: Include search metadata
            use_hybrid: Use hybrid scoring
            metadata_filters: ChromaDB where filters (v0.2.9)
            transparency_context: Optional context for tracking (v0.2.9)

        Returns:
            List of results
        """
        if not self.initialized:
            await self.initialize()

        return await self._search_service.search(
            query=query,
            collections=collections,
            limit=limit,
            return_metadata=return_metadata,
            metadata_filters=metadata_filters,
            transparency_context=transparency_context
        )

    async def detect_conversation_outcome(
        self,
        conversation: List[Dict[str, Any]],
        surfaced_memories: Optional[Dict[int, str]] = None,
        llm_marks: Optional[Dict[int, str]] = None  # v0.2.12 Fix #7
    ) -> Dict[str, Any]:
        """
        Detect outcome from a conversation exchange.

        Uses LLM to analyze if the assistant's response was helpful based on user feedback.

        Args:
            conversation: List of turns [{role, content}, ...] - typically [assistant, user]
            surfaced_memories: v0.2.12 - Optional {position: content} for selective scoring
            llm_marks: v0.2.12 Fix #7 - Main LLM's attribution {pos: '👍'/'👎'/'➖'}

        Returns:
            {
                "outcome": "worked|failed|partial|unknown",
                "confidence": 0.0-1.0,
                "indicators": ["signals"],
                "reasoning": "brief explanation",
                "used_positions": [1, 3],  # v0.2.12: which memories were actually used
                "upvote": [1],              # v0.2.12 Fix #7: positions to upvote
                "downvote": [2]             # v0.2.12 Fix #7: positions to downvote
            }
        """
        if not self.llm_service:
            logger.debug("No LLM service for outcome detection")
            return {
                "outcome": "unknown",
                "confidence": 0.0,
                "indicators": [],
                "reasoning": "No LLM service available",
                "used_positions": [],
                "upvote": [],
                "downvote": []
            }

        # Lazy import to avoid circular dependency
        from modules.advanced.outcome_detector import OutcomeDetector

        # Use cached detector or create new one
        if not hasattr(self, '_outcome_detector') or self._outcome_detector is None:
            self._outcome_detector = OutcomeDetector(self.llm_service)

        # v0.2.12 Fix #7: Pass surfaced memories and llm_marks for causal scoring
        return await self._outcome_detector.analyze(conversation, surfaced_memories, llm_marks)

    async def record_outcome(
        self,
        doc_id: str,
        outcome: Literal["worked", "failed", "partial"],
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record outcome for a document.

        Args:
            doc_id: Document ID
            outcome: Outcome type
            failure_reason: Reason for failure
            context: Additional context
        """
        if not self.initialized:
            await self.initialize()

        await self._outcome_service.record_outcome(
            doc_id=doc_id,
            outcome=outcome,
            failure_reason=failure_reason,
            context=context
        )

    # ==================== Memory Bank API ====================

    async def store_memory_bank(
        self,
        text: str,
        tags: List[str],
        importance: float = 0.7,
        confidence: float = 0.7
    ) -> str:
        """Store memory in memory_bank collection."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.store(
            text=text,
            tags=tags,
            importance=importance,
            confidence=confidence
        )

    async def update_memory_bank(
        self,
        doc_id: str,
        new_text: str,
        reason: str = "llm_update"
    ) -> str:
        """Update memory_bank item."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.update(
            doc_id=doc_id,
            new_text=new_text,
            reason=reason
        )

    async def archive_memory_bank(
        self,
        doc_id: str,
        reason: str = "llm_decision"
    ) -> bool:
        """Archive memory_bank item."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.archive(
            doc_id=doc_id,
            reason=reason
        )

    async def search_memory_bank(
        self,
        query: str = None,
        tags: List[str] = None,
        include_archived: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search memory_bank collection."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.search(
            query=query,
            tags=tags,
            include_archived=include_archived,
            limit=limit
        )

    async def user_restore_memory(self, doc_id: str) -> bool:
        """User restores archived memory."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.restore(doc_id)

    async def user_delete_memory(self, doc_id: str) -> bool:
        """User permanently deletes memory."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.delete(doc_id)

    # ==================== Context API ====================

    async def analyze_conversation_context(
        self,
        current_message: str,
        recent_conversation: List[Dict[str, Any]],
        conversation_id: str
    ) -> Dict[str, Any]:
        """Analyze conversation context."""
        if not self.initialized:
            await self.initialize()

        return await self._context_service.analyze_conversation_context(
            current_message=current_message,
            recent_conversation=recent_conversation,
            conversation_id=conversation_id
        )

    # ==================== Promotion API ====================

    async def promote_valuable_working_memory(self) -> int:
        """Promote valuable working memory to history."""
        if not self.initialized:
            await self.initialize()

        return await self._promotion_service.promote_valuable_working_memory()

    async def cleanup_old_working_memory(self) -> int:
        """Clean up old working memory items."""
        if not self.initialized:
            await self.initialize()

        return await self._promotion_service.cleanup_old_working_memory()

    async def cleanup_action_kg_for_doc_ids(self, doc_ids: List[str]) -> int:
        """
        Clean up Action KG examples referencing deleted documents.

        Args:
            doc_ids: List of document IDs to clean up

        Returns:
            Number of examples cleaned
        """
        if not self._kg_service:
            return 0
        return await self._kg_service.cleanup_action_kg_for_doc_ids(doc_ids)

    # ==================== Session Management ====================

    async def switch_conversation(self, new_conversation_id: Optional[str] = None) -> str:
        """
        Switch to a new conversation.

        Args:
            new_conversation_id: New conversation ID (auto-generated if None)

        Returns:
            New conversation ID
        """
        # Promote valuable memories before switching
        async with self._promotion_lock:
            await self.promote_valuable_working_memory()

        # Switch conversation
        old_id = self.conversation_id
        self.conversation_id = new_conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_count = 0
        self.session_context = {}

        logger.info(f"Switched conversation: {old_id} -> {self.conversation_id}")
        return self.conversation_id

    def increment_message_count(self):
        """Increment message count and trigger auto-promotion if needed."""
        self.message_count += 1

        # Auto-promote every 20 messages
        if self.message_count % 20 == 0:
            asyncio.create_task(self._auto_promote())

    async def _auto_promote(self):
        """Auto-promote valuable memories."""
        async with self._promotion_lock:
            promoted = await self.promote_valuable_working_memory()
            if promoted > 0:
                logger.info(f"Auto-promoted {promoted} memories at message {self.message_count}")

    # ==================== Cleanup ====================


    # ==================== Context Detection API ====================

    async def detect_context_type(
        self,
        system_prompts: Optional[List[str]] = None,
        recent_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        LLM-BASED SESSION TYPE CLASSIFICATION.
        Returns session types like: "learning", "recall", "coding_help", etc.
        """
        if not self.llm_service:
            logger.debug("[CONTEXT] No LLM service - using 'general' context")
            return "general"

        context_parts = []
        if system_prompts:
            context_parts.extend(system_prompts[:2])
        if recent_messages:
            for msg in recent_messages[-3:]:
                if isinstance(msg, dict):
                    context_parts.append(msg.get('content', ''))
                else:
                    context_parts.append(str(msg))

        if not context_parts:
            return "general"

        conversation_text = "\n".join(context_parts)[:800]

        prompt = f"""Classify this conversation's SESSION TYPE in 1-2 words (lowercase, underscore-separated).

Examples: coding_help, fitness_tracking, creative_writing, project_planning, learning, recall, general_chat

Conversation:
{conversation_text}

Session type (1-2 words only):"""

        try:
            response = await asyncio.wait_for(
                self.llm_service.generate(prompt, max_tokens=10),
                timeout=3.0
            )
            topic = response.strip().lower().replace(" ", "_").replace("-", "_")
            topic = re.sub(r'[^\w]', '_', topic, flags=re.UNICODE)
            topic = topic.strip('_')
            if 0 < len(topic) < 30:
                logger.debug(f"[CONTEXT] Classified as: {topic}")
                return topic
            return "general"
        except asyncio.TimeoutError:
            return "general"
        except Exception as e:
            logger.debug(f"[CONTEXT] Classification failed: {e}")
            return "general"

    def get_action_effectiveness(
        self,
        context_type: str,
        action_type: str,
        collection: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get effectiveness stats for an action in a specific context."""
        if self._kg_service:
            kg = self._kg_service.knowledge_graph
            key = f"{context_type}|{action_type}|{collection or '*'}"
            return kg.get("context_action_effectiveness", {}).get(key)
        return None

    def get_tier_recommendations(self, concepts: List[str]) -> Dict[str, Any]:
        """Query Routing KG for best collections given concepts."""
        if self._routing_service:
            return self._routing_service.get_tier_recommendations(concepts)
        return {
            "top_collections": ["working", "patterns", "history", "books", "memory_bank"],
            "match_count": 0,
            "confidence_level": "exploration"
        }

    async def get_facts_for_entities(self, entities: List[str], limit: int = 2) -> List[Dict[str, Any]]:
        """Query Content KG to retrieve matching memory_bank facts."""
        facts = []
        for entity in entities:
            if len(facts) >= limit:
                break
            try:
                results = await self.search(
                    query=entity,
                    collections=["memory_bank"],
                    limit=2
                )
                for result in results:
                    if len(facts) >= limit:
                        break
                    doc_id = result.get("id") or result.get("doc_id")
                    content = result.get("text") or result.get("content", "")
                    if any(f["doc_id"] == doc_id for f in facts):
                        continue
                    # Get doc effectiveness from search service
                    effectiveness = None
                    if self._search_service and doc_id:
                        effectiveness = self._search_service.get_doc_effectiveness(doc_id)
                    # Filter out consistently failing facts
                    if effectiveness and effectiveness.get("total_uses", 0) >= 3:
                        if effectiveness.get("success_rate", 0.5) < 0.4:
                            continue
                    facts.append({
                        "doc_id": doc_id,
                        "content": content,
                        "entity": entity,
                        "effectiveness": effectiveness
                    })
            except Exception as e:
                logger.warning(f"[FACTS] Failed to get facts for entity '{entity}': {e}")
                continue
        return facts


    # ==================== Cold Start & Backup API ====================

    async def get_cold_start_context(self, limit: int = 5) -> Tuple[Optional[str], List[str], List[Dict]]:
        """Generate cold-start context from memory_bank, patterns, and history.

        Returns:
            Tuple of (formatted_context, doc_ids, raw_context) for outcome scoring.
            v0.2.12: Now returns doc_ids and raw context for selective outcome scoring.
        """
        all_context = []
        memory_bank_limit = max(3, limit - 2)

        # STEP 1: Get memory_bank facts via search
        try:
            results = await self.search(
                query="user identity name projects current work goals preferences",
                collections=["memory_bank"],
                limit=memory_bank_limit
            )
            for r in results:
                all_context.append({
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("text", ""),
                    "source": "memory_bank"
                })
        except Exception as e:
            logger.warning(f"[COLD-START] memory_bank search failed: {e}")

        # STEP 2: Get top pattern
        try:
            pattern_results = await self.search(
                query="proven solution effective approach",
                collections=["patterns"],
                limit=1
            )
            for r in pattern_results:
                all_context.append({
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("text", ""),
                    "source": "patterns"
                })
        except Exception as e:
            logger.debug(f"[COLD-START] patterns search failed: {e}")

        # STEP 3: Get recent history
        try:
            history_results = await self.search(
                query="recent conversation context",
                collections=["history"],
                limit=1
            )
            for r in history_results:
                all_context.append({
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("text", ""),
                    "source": "history"
                })
        except Exception as e:
            logger.debug(f"[COLD-START] history search failed: {e}")

        if not all_context:
            return None, [], []

        # v0.2.12: Extract doc_ids for outcome scoring
        doc_ids = [r.get("id") for r in all_context if r.get("id")]
        formatted = self._format_cold_start_results(all_context)
        return formatted, doc_ids, all_context

    def _format_cold_start_results(self, results: List[Dict]) -> Optional[str]:
        """Format cold-start context with injection protection."""
        if not results:
            return None

        # Filter suspicious content
        safe_results = [
            r for r in results
            if not any(x in (r.get("content") or "").lower()
                for x in ["ignore all previous", "ignore instructions"])
        ][:10]

        if not safe_results:
            return None

        memory_bank_items = [r for r in safe_results if r.get("source") == "memory_bank"]
        pattern_items = [r for r in safe_results if r.get("source") == "patterns"]
        history_items = [r for r in safe_results if r.get("source") == "history"]

        sections = []
        if memory_bank_items:
            sections.append("[User Profile]:\n" + "\n".join([
                f"- {(r.get('content') or '')[:250]}" for r in memory_bank_items
            ]))
        if pattern_items:
            sections.append("[Proven Patterns]:\n" + "\n".join([
                f"- {(r.get('content') or '')[:200]}" for r in pattern_items
            ]))
        if history_items:
            sections.append("[Recent Context]:\n" + "\n".join([
                f"- {(r.get('content') or '')[:200]}" for r in history_items
            ]))

        return "\n\n".join(sections)

    async def record_action_outcome(self, action) -> None:
        """Record action-level outcome with context awareness (Causal Learning)."""
        if not self._kg_service:
            return

        key = f"{action.context_type}|{action.action_type}|{action.collection or '*'}"
        kg = self._kg_service.knowledge_graph

        if "context_action_effectiveness" not in kg:
            kg["context_action_effectiveness"] = {}

        if key not in kg["context_action_effectiveness"]:
            kg["context_action_effectiveness"][key] = {
                "successes": 0, "failures": 0, "partials": 0,
                "success_rate": 0.0, "total_uses": 0,
                "first_seen": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "examples": []
            }

        stats = kg["context_action_effectiveness"][key]

        if action.outcome == "worked":
            stats["successes"] += 1
        elif action.outcome == "failed":
            stats["failures"] += 1
        else:
            stats["partials"] += 1

        stats["total_uses"] += 1
        stats["last_used"] = datetime.now().isoformat()

        total = stats["successes"] + stats["failures"] + stats["partials"]
        if total > 0:
            weighted = stats["successes"] + (stats["partials"] * 0.5)
            stats["success_rate"] = weighted / total

        example = {
            "timestamp": action.timestamp.isoformat() if hasattr(action, 'timestamp') else datetime.now().isoformat(),
            "outcome": action.outcome,
            "doc_id": getattr(action, 'doc_id', None)
        }
        stats["examples"] = (stats.get("examples", []) + [example])[-5:]

        await self._kg_service._debounced_save_kg()

        if hasattr(action, 'doc_id') and action.doc_id:
            await self.record_outcome(
                doc_id=action.doc_id,
                outcome=action.outcome,
                failure_reason=getattr(action, 'failure_reason', None)
            )

    async def _update_kg_routing(self, query: str, collection: str, outcome: str) -> None:
        """Update KG routing patterns based on outcome."""
        if not self._kg_service or not query:
            return
        await self._kg_service.update_kg_routing(query, collection, outcome)

    async def export_backup(self) -> Dict[str, Any]:
        """Export memory system state for backup."""
        backup = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "conversation_id": self.conversation_id,
            "stats": self.get_stats()
        }

        if self._kg_service:
            backup["knowledge_graph"] = self._kg_service.knowledge_graph
            backup["relationships"] = self._kg_service.relationships

        backup["collections"] = {}
        for name, adapter in self.collections.items():
            try:
                count = adapter.collection.count()
                backup["collections"][name] = {"count": count}
            except:
                backup["collections"][name] = {"count": 0}

        return backup

    async def import_backup(self, backup_data: Dict[str, Any]) -> bool:
        """Import memory system state from backup."""
        try:
            if self._kg_service:
                if "knowledge_graph" in backup_data:
                    self._kg_service.knowledge_graph = backup_data["knowledge_graph"]
                    await self._kg_service._save_kg()
                if "relationships" in backup_data:
                    self._kg_service.relationships = backup_data["relationships"]
                    self._kg_service._save_relationships_sync()

            logger.info(f"Restored backup from {backup_data.get('timestamp', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return False

    async def _cleanup_kg_dead_references(self) -> int:
        """Remove doc_id references that no longer exist in collections."""
        if not self._kg_service:
            return 0

        cleaned = 0
        kg = self._kg_service.knowledge_graph

        for problem_key in list(kg.get("problem_categories", {}).keys()):
            doc_ids = kg["problem_categories"][problem_key]
            valid_ids = [d for d in doc_ids if self._doc_exists(d)]
            if len(valid_ids) < len(doc_ids):
                cleaned += len(doc_ids) - len(valid_ids)
                if valid_ids:
                    kg["problem_categories"][problem_key] = valid_ids
                else:
                    del kg["problem_categories"][problem_key]

        for problem_sig in list(kg.get("problem_solutions", {}).keys()):
            solutions = kg["problem_solutions"][problem_sig]
            valid = [s for s in solutions if self._doc_exists(s.get("doc_id"))]
            if len(valid) < len(solutions):
                cleaned += len(solutions) - len(valid)
                if valid:
                    kg["problem_solutions"][problem_sig] = valid
                else:
                    del kg["problem_solutions"][problem_sig]

        if cleaned > 0:
            logger.info(f"KG cleanup: removed {cleaned} dead references")
            await self._kg_service._save_kg()

        return cleaned

    def _doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in any collection."""
        if not doc_id:
            return False
        for name in ["working", "history", "patterns", "memory_bank", "books"]:
            if doc_id.startswith(f"{name}_"):
                try:
                    adapter = self.collections.get(name)
                    if adapter:
                        result = adapter.collection.get(ids=[doc_id])
                        return bool(result and result.get("ids"))
                except:
                    pass
        return False

    
    def _route_query(self, query: str) -> List[str]:
        """Route query to appropriate collections (delegates to routing service)."""
        if self._routing_service:
            return self._routing_service.route_query(query)
        return ["working", "patterns", "history", "books", "memory_bank"]

    async def cleanup(self):
        """Clean shutdown."""
        logger.info("Shutting down UnifiedMemorySystem...")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Save KG
        if self._kg_service:
            self._kg_service._save_kg_sync()

        # Cleanup collections
        for name, adapter in self.collections.items():
            try:
                await adapter.cleanup()
            except Exception:
                pass

        logger.info("UnifiedMemorySystem shutdown complete")

    # ==================== Backward Compatibility ====================

    @property
    def knowledge_graph(self) -> Dict[str, Any]:
        """Expose knowledge graph for backward compatibility."""
        if self._kg_service:
            return self._kg_service.knowledge_graph
        return {}

    def get_outcome_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get outcome stats for a document."""
        if self._outcome_service:
            return self._outcome_service.get_outcome_stats(doc_id)
        return {"error": "service_not_initialized"}

    async def get_working_context(self, limit: int = 5) -> Dict[str, Any]:
        """Get recent working context."""
        if not self.initialized:
            await self.initialize()

        # Search working memory for recent items
        results = await self.search(
            query="",
            collections=["working"],
            limit=limit
        )

        return {
            "conversation_id": self.conversation_id,
            "message_count": self.message_count,
            "recent_items": results
        }

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all items for a conversation."""
        deleted_count = 0

        for coll_name, adapter in self.collections.items():
            try:
                all_ids = adapter.list_all_ids()
                to_delete = []

                for doc_id in all_ids:
                    doc = adapter.get_fragment(doc_id)
                    if doc and doc.get("metadata", {}).get("conversation_id") == conversation_id:
                        to_delete.append(doc_id)

                if to_delete:
                    adapter.delete_vectors(to_delete)
                    deleted_count += len(to_delete)
                    logger.info(f"Deleted {len(to_delete)} items from {coll_name}")

            except Exception as e:
                logger.error(f"Error deleting from {coll_name}: {e}")

        return deleted_count

    async def search_books(
        self,
        query: str,
        limit: int = 5,
        title_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search books collection specifically."""
        results = await self.search(
            query=query,
            collections=["books"],
            limit=limit * 2  # Get extra for filtering
        )

        if title_filter:
            results = [
                r for r in results
                if title_filter.lower() in r.get("metadata", {}).get("title", "").lower()
            ]

        return results[:limit]

    async def save_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a conversation turn to working memory."""
        turn_metadata = {
            "role": role,
            "conversation_id": self.conversation_id,
            "message_number": self.message_count,
            **(metadata or {})
        }

        self.increment_message_count()

        return await self.store(
            text=content,
            collection="working",
            metadata=turn_metadata
        )

    async def ingest_book(self, file_path: str, title: str) -> int:
        """
        Ingest a book file into the books collection.

        Args:
            file_path: Path to the book file
            title: Title of the book

        Returns:
            Number of chunks ingested
        """
        if not self.initialized:
            await self.initialize()

        # Use file adapter to read and chunk the file
        chunks = await self.file_adapter.read_file(file_path, chunk_size=1000)

        ingested = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"books_{uuid.uuid4().hex[:8]}"

            metadata = {
                "title": title,
                "source_file": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "uploaded_at": datetime.now().isoformat()
            }

            embedding = await self._embed_text(chunk)

            await self.collections["books"].upsert_vectors(
                ids=[doc_id],
                vectors=[embedding],
                metadatas=[{
                    "text": chunk,
                    "content": chunk,
                    **metadata
                }]
            )
            ingested += 1

        logger.info(f"Ingested book '{title}' with {ingested} chunks")
        return ingested

    # ==================== KG Visualization API ====================

    async def get_kg_entities(self, filter_text: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """Get KG entities for visualization."""
        if self._kg_service:
            entities = await self._kg_service.get_kg_entities(limit=limit)
            # Apply filter if provided
            if filter_text:
                filter_lower = filter_text.lower()
                entities = [e for e in entities if filter_lower in str(e).lower()]
            return entities
        return []

    def get_kg_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get KG relationships for visualization."""
        if self._kg_service:
            return self._kg_service.get_kg_relationships(limit)
        return []


    # ==================== Stats API ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "conversation_id": self.conversation_id,
            "collections": {},
            "outcomes": {},
            "knowledge_graph": {},
            "decay": {},
            "status": "active"
        }

        # Collection counts
        for name, adapter in self.collections.items():
            try:
                stats["collections"][name] = adapter.collection.count()
            except:
                stats["collections"][name] = 0

        # KG stats
        if self._kg_service:
            kg = self._kg_service.knowledge_graph
            stats["knowledge_graph"] = {
                "routing_patterns": len(kg.get("routing_patterns", {})),
                "failure_patterns": len(kg.get("failure_patterns", {})),
                "problem_categories": len(kg.get("problem_categories", {})),
                "problem_solutions": len(kg.get("problem_solutions", {})),
                "solution_patterns": len(kg.get("solution_patterns", {}))
            }

        return stats

# Backward compatibility alias
UMS = UnifiedMemorySystem
