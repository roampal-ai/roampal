"""
UnifiedMemorySystem Facade - Refactored Version

This is the coordinating facade that delegates to individual services.
Extracted from the original 4,746-line monolith into composable services.

Services:
- ScoringService: Wilson score, dynamic weights
- RoutingService: Query routing, tier scores
- SearchService: TagCascade retrieval, CE reranking
- PromotionService: Memory promotion/demotion
- OutcomeService: Outcome recording, score updates
- MemoryBankService: User identity/preferences
- ContextService: Conversation context analysis
- TagService: Noun tag extraction + word-boundary matching
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
from .routing_service import RoutingService
from .search_service import SearchService
from .promotion_service import PromotionService
from .outcome_service import OutcomeService
from .memory_bank_service import MemoryBankService
from .context_service import ContextService
from .tag_service import TagService

logger = logging.getLogger(__name__)

# v0.3.0: Cold start tag priorities - one fact per category (ported from roampal-core v0.2.7)
TAG_PRIORITIES = [
    "identity",
    "preference",
    "goal",
    "project",
    "system_mastery",
    "agent_growth",
]


def _first_sentence(text: str, max_chars: int = 300) -> str:
    """
    Extract first sentence from text, capped at max_chars.

    v0.3.0: Ported from roampal-core. Used for cold start truncation - prevents
    massive facts from overwhelming context. Full facts still available via search.
    """
    if not text:
        return ""
    # Find first sentence ending
    for end_char in [". ", ".\n", "!", "?"]:
        idx = text.find(end_char)
        if idx > 0:
            first = text[: idx + 1].strip()
            if len(first) <= max_chars:
                return first
            break
    # No sentence ending found or sentence too long - truncate
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


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
                        wait_time = delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
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
        config: Optional[MemoryConfig] = None,
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

        # v0.3.0: Health check for corrupted collections (auto-repairs working)
        await self._health_check_collections()

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
                    use_server=self.use_server,
                )
                await self.collections[name].initialize(
                    collection_name=f"roampal_{name}"
                )

    def _migrate_chromadb_schema(self):
        """
        Migrate ChromaDB schema for compatibility across versions.

        v0.2.10: ChromaDB 1.x added 'topic' column to collections table.
        Users upgrading from ChromaDB 0.4.x/0.5.x will have old schema.
        This safely adds missing columns without affecting existing data.

        v0.3.0: Added transaction wrapper and validation to prevent corruption.
        """
        import sqlite3

        chromadb_path = self.data_dir / "chromadb"
        sqlite_path = chromadb_path / "chroma.sqlite3"

        if not sqlite_path.exists():
            logger.debug("No existing ChromaDB - skipping migration")
            return

        conn = None
        try:
            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.cursor()

            # v0.3.0: Start transaction explicitly
            cursor.execute("BEGIN TRANSACTION")

            # Columns added in ChromaDB 1.x that may be missing
            migrations_needed = []

            # Check collections table
            cursor.execute("PRAGMA table_info(collections)")
            collections_columns = {col[1] for col in cursor.fetchall()}
            if "topic" not in collections_columns:
                migrations_needed.append(("collections", "topic", "TEXT"))

            # Check segments table (also needs 'topic' in ChromaDB 1.x)
            cursor.execute("PRAGMA table_info(segments)")
            segments_columns = {col[1] for col in cursor.fetchall()}
            if "topic" not in segments_columns:
                migrations_needed.append(("segments", "topic", "TEXT"))

            # Apply migrations within transaction
            for table, column, col_type in migrations_needed:
                try:
                    cursor.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                    )
                    logger.info(f"ChromaDB migration: Added {column} to {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        pass  # Column already exists, safe to ignore
                    else:
                        # v0.3.0: Rollback on any error
                        cursor.execute("ROLLBACK")
                        raise

            # v0.3.0: Validate migration before commit
            cursor.execute("PRAGMA table_info(collections)")
            final_cols = {col[1] for col in cursor.fetchall()}
            if "topic" not in final_cols:
                cursor.execute("ROLLBACK")
                raise RuntimeError(
                    "Migration validation failed: 'topic' column not added to collections"
                )

            cursor.execute("COMMIT")
            logger.debug("ChromaDB schema migration completed and validated")

        except Exception as e:
            logger.warning(f"ChromaDB schema migration failed (non-fatal): {e}")
            # v0.3.0: Ensure rollback on any error
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except:
                    pass
        finally:
            if conn:
                conn.close()

    async def _health_check_collections(self):
        """
        v0.3.0: Check health of all collections and repair if needed.

        Detects corruption from failed migrations or version upgrades.
        Can repair working collection by recreating it (temporary data, 24h TTL).

        v0.3.0 FIX: Don't use query_texts for health check - it requires embedding
        function and will fail if dimensions mismatch, causing false positives.
        Use count() and peek() instead - these don't require embeddings.
        """
        logger.info("[HEALTH] Running collection health check...")
        issues_found = []

        for name, adapter in self.collections.items():
            try:
                # Test basic operations (count doesn't require embeddings)
                count = adapter.collection.count()
                logger.debug(f"[HEALTH] {name}: {count} items")

                # v0.3.0 FIX #2: Skip peek on empty collections - they're not corrupt, just empty
                # "Nothing found on disk" error was falsely triggering repair on empty working collection
                if count == 0:
                    logger.debug(
                        f"[HEALTH] {name} is empty (count=0), skipping peek test"
                    )
                    continue

                # Test peek capability (doesn't require embeddings, unlike query)
                # This catches actual corruption without triggering on embedding mismatches
                try:
                    adapter.collection.peek(limit=1)
                except Exception as peek_error:
                    # Only flag as issue if it's actual corruption, not dimension mismatch
                    error_str = str(peek_error).lower()
                    if "dimension" not in error_str and "embedding" not in error_str:
                        issues_found.append(
                            {
                                "collection": name,
                                "error": f"Peek failed: {peek_error}",
                                "recoverable": name
                                == "working",  # Only working is easily recoverable
                            }
                        )
                        logger.warning(
                            f"[HEALTH] {name} collection peek failed: {peek_error}"
                        )
                    else:
                        # Embedding dimension mismatch is not corruption - skip repair
                        logger.debug(
                            f"[HEALTH] {name} has embedding dimension mismatch (not corruption)"
                        )

            except Exception as e:
                # Only flag as issue if it's actual corruption
                error_str = str(e).lower()
                if "dimension" not in error_str and "embedding" not in error_str:
                    issues_found.append(
                        {
                            "collection": name,
                            "error": str(e),
                            "recoverable": name == "working",
                        }
                    )
                    logger.warning(f"[HEALTH] {name} collection check failed: {e}")
                else:
                    logger.debug(
                        f"[HEALTH] {name} has embedding dimension mismatch (not corruption)"
                    )

        if issues_found:
            logger.warning(f"[HEALTH] Found {len(issues_found)} collection issues")

            # Auto-repair working collection (it's temporary data anyway)
            for issue in issues_found:
                if issue["collection"] == "working" and issue["recoverable"]:
                    logger.info("[HEALTH] Attempting to repair working collection...")
                    try:
                        await self._repair_working_collection()
                        logger.info("[HEALTH] Working collection repaired successfully")
                    except Exception as repair_error:
                        logger.error(
                            f"[HEALTH] Failed to repair working collection: {repair_error}"
                        )
        else:
            logger.info("[HEALTH] All collections healthy")

        return issues_found

    async def _repair_working_collection(self):
        """
        Repair corrupted working collection by recreating it.

        Working memory is temporary (24h TTL), so data loss is acceptable.
        This is better than leaving users with a broken app.
        """
        import shutil

        working_adapter = self.collections.get("working")
        if not working_adapter:
            return

        chromadb_path = self.data_dir / "chromadb"

        try:
            # Get ChromaDB client to delete collection properly
            client = working_adapter.collection._client
            client.delete_collection("roampal_working")
            logger.info("[REPAIR] Deleted corrupted working collection")

            # Recreate collection
            from modules.memory.chromadb_adapter import ChromaDBAdapter

            self.collections["working"] = ChromaDBAdapter(
                persistence_directory=str(chromadb_path), use_server=self.use_server
            )
            await self.collections["working"].initialize(
                collection_name="roampal_working"
            )
            logger.info("[REPAIR] Recreated working collection")

        except Exception as e:
            logger.error(f"[REPAIR] Failed to repair working collection: {e}")
            raise

    def _init_services(self):
        """Initialize all extracted services."""
        # Scoring service (no dependencies)
        self._scoring_service = ScoringService(self.config)

        # v0.3.1: Tag service (replaces KG for retrieval)
        self._tag_service = TagService()

        # Routing service (v0.3.1: KG removed, returns all collections)
        self._routing_service = RoutingService(config=self.config)

        # Search service (v0.3.1: TagCascade, tag_service replaces kg_service)
        self._search_service = SearchService(
            collections=self.collections,
            embed_fn=self._embed_text,
            scoring_service=self._scoring_service,
            routing_service=self._routing_service,
            tag_service=self._tag_service,
            config=self.config,
        )

        # Rebuild known tag index from existing ChromaDB metadata
        self._tag_service.rebuild_known_tags(self.collections)

        # Promotion service
        self._promotion_service = PromotionService(
            collections=self.collections,
            embed_fn=self._embed_text,
            config=self.config,
        )

        # Outcome service (v0.3.1: KG removed — tags handle routing)
        self._outcome_service = OutcomeService(
            collections=self.collections,
            promotion_service=self._promotion_service,
            config=self.config,
        )

        # Memory bank service
        if "memory_bank" in self.collections:
            self._memory_bank_service = MemoryBankService(
                collection=self.collections["memory_bank"],
                embed_fn=self._embed_text,
                search_fn=self.search,
                config=self.config,
            )

        # Context service
        self._context_service = ContextService(
            collections=self.collections,
            embed_fn=self._embed_text,
            config=self.config,
        )

    async def _embed_text(self, text: str) -> List[float]:
        """Embed text using the embedding service."""
        return await self.embedding_service.embed_text(text)

    async def _generate_contextual_prefix(
        self, text: str, metadata: Optional[Dict[str, Any]], collection: str
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
                "books": "reference material",
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
                    self.llm_service.generate(prompt, max_tokens=50), timeout=5.0
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
            logger.debug(
                f"[CONTEXTUAL] Prefix generation failed: {e}, using original text"
            )

        # Fallback to original text
        return text

    # ==================== Core API ====================

    @with_retry(max_attempts=3, delay=0.5)
    async def store(
        self,
        text: str,
        collection: CollectionName = "working",
        metadata: Optional[Dict[str, Any]] = None,
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
            **(metadata or {}),
        }

        # Generate contextual embedding (Anthropic Contextual Retrieval)
        contextual_text = await self._generate_contextual_prefix(
            text, final_metadata, collection
        )
        embedding = await self._embed_text(contextual_text)

        # Store in collection
        await self.collections[collection].upsert_vectors(
            ids=[doc_id], vectors=[embedding], metadatas=[final_metadata]
        )

        # v0.3.1: Extract noun_tags for TagCascade retrieval
        # Only extract if not already provided in metadata (benchmark-aligned: tags extracted at store time)
        if hasattr(self, "_tag_service") and self._tag_service:
            # Check if noun_tags already provided in metadata
            if "noun_tags" not in final_metadata:
                try:
                    tags = await self._tag_service.extract_tags_async(text)
                    if tags:
                        self.collections[collection].update_fragment_metadata(
                            doc_id, {"noun_tags": json.dumps(tags)}
                        )
                except Exception as e:
                    logger.warning(f"Tag extraction failed for {doc_id}: {e}")

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
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,  # v0.2.9: Expose metadata filtering
        transparency_context: Optional[Any] = None,  # v0.2.9: Transparency tracking
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
            transparency_context=transparency_context,
        )

    async def detect_conversation_outcome(
        self,
        conversation: List[Dict[str, Any]],
        surfaced_memories: Optional[Dict[int, str]] = None,
        llm_marks: Optional[Dict[int, str]] = None,  # v0.2.12 Fix #7
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
                "downvote": [],
            }

        # Lazy import to avoid circular dependency
        from modules.advanced.outcome_detector import OutcomeDetector

        # Use cached detector or create new one
        if not hasattr(self, "_outcome_detector") or self._outcome_detector is None:
            self._outcome_detector = OutcomeDetector(self.llm_service)

        # v0.2.12 Fix #7: Pass surfaced memories and llm_marks for causal scoring
        return await self._outcome_detector.analyze(
            conversation, surfaced_memories, llm_marks
        )

    async def record_outcome(
        self,
        doc_id: str,
        outcome: Literal["worked", "failed", "partial"],
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
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
            context=context,
        )

    # ==================== Memory Bank API ====================

    async def store_memory_bank(
        self,
        text: str,
        tags: List[str],
        noun_tags: Optional[List[str]] = None,
        importance: float = 0.7,
        confidence: float = 0.7,
    ) -> str:
        """Store memory in memory_bank collection."""
        if not self.initialized:
            await self.initialize()

        doc_id = await self._memory_bank_service.store(
            text=text, tags=tags, importance=importance, confidence=confidence
        )

        # v0.3.1: noun_tags for TagCascade retrieval
        if hasattr(self, "_tag_service") and self._tag_service:
            try:
                actual_tags = (
                    noun_tags if noun_tags else self._tag_service.extract_tags(text)
                )
                if actual_tags:
                    self.collections["memory_bank"].update_fragment_metadata(
                        doc_id, {"noun_tags": json.dumps(actual_tags)}
                    )
                    self._tag_service.add_known_tags(actual_tags)
            except Exception as e:
                logger.warning(f"Tag extraction failed for memory_bank {doc_id}: {e}")

        return doc_id

    async def update_memory_bank(
        self, doc_id: str, new_text: str, reason: str = "llm_update"
    ) -> str:
        """Update memory_bank item."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.update(
            doc_id=doc_id, new_text=new_text, reason=reason
        )

    async def archive_memory_bank(
        self, doc_id: str, reason: str = "llm_decision"
    ) -> bool:
        """Archive memory_bank item."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.archive(doc_id=doc_id, reason=reason)

    async def search_memory_bank(
        self,
        query: str = None,
        tags: List[str] = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search memory_bank collection."""
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.search(
            query=query, tags=tags, include_archived=include_archived, limit=limit
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
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Analyze conversation context."""
        if not self.initialized:
            await self.initialize()

        return await self._context_service.analyze_conversation_context(
            current_message=current_message,
            recent_conversation=recent_conversation,
            conversation_id=conversation_id,
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

    async def cleanup_old_history(self) -> int:
        """Clean up history items older than 30 days (v0.3.1, matches core)."""
        if not self.initialized:
            await self.initialize()

        return await self._promotion_service.cleanup_old_history()

    # ==================== Session Management ====================

    async def switch_conversation(
        self, new_conversation_id: Optional[str] = None
    ) -> str:
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
        self.conversation_id = new_conversation_id or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
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
                logger.info(
                    f"Auto-promoted {promoted} memories at message {self.message_count}"
                )

    # ==================== Cleanup ====================

    # ==================== Context Detection API ====================

    async def detect_context_type(
        self,
        system_prompts: Optional[List[str]] = None,
        recent_messages: Optional[List[Dict[str, str]]] = None,
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
                    context_parts.append(msg.get("content", ""))
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
                self.llm_service.generate(prompt, max_tokens=10), timeout=3.0
            )
            topic = response.strip().lower().replace(" ", "_").replace("-", "_")
            topic = re.sub(r"[^\w]", "_", topic, flags=re.UNICODE)
            topic = topic.strip("_")
            if 0 < len(topic) < 30:
                logger.debug(f"[CONTEXT] Classified as: {topic}")
                return topic
            return "general"
        except asyncio.TimeoutError:
            return "general"
        except Exception as e:
            logger.debug(f"[CONTEXT] Classification failed: {e}")
            return "general"

    def get_tier_recommendations(self, concepts: List[str]) -> Dict[str, Any]:
        """Query Routing KG for best collections given concepts."""
        if self._routing_service:
            return self._routing_service.get_tier_recommendations(concepts)
        return {
            "top_collections": [
                "working",
                "patterns",
                "history",
                "books",
                "memory_bank",
            ],
            "match_count": 0,
            "confidence_level": "exploration",
        }

    async def get_facts_for_entities(
        self, entities: List[str], limit: int = 2
    ) -> List[Dict[str, Any]]:
        """Query Content KG to retrieve matching memory_bank facts."""
        facts = []
        for entity in entities:
            if len(facts) >= limit:
                break
            try:
                results = await self.search(
                    query=entity, collections=["memory_bank"], limit=2
                )
                for result in results:
                    if len(facts) >= limit:
                        break
                    doc_id = result.get("id") or result.get("doc_id")
                    content = result.get("text") or result.get("content", "")
                    if any(f["doc_id"] == doc_id for f in facts):
                        continue
                    facts.append(
                        {
                            "doc_id": doc_id,
                            "content": content,
                            "entity": entity,
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"[FACTS] Failed to get facts for entity '{entity}': {e}"
                )
                continue
        return facts

    # ==================== Cold Start & Context Injection ====================

    async def _build_cold_start_profile(
        self, mode: str = "internal"
    ) -> Tuple[Optional[str], List[str], List[Dict]]:
        """
        Build the cold start user profile injection.

        v0.3.0: Ported from roampal-core. Lean but rich - one fact per tag category:
        1. Get all facts from memory_bank, sorted by quality (importance * confidence)
        2. Pick ONE highest-quality fact per tag category (identity, preference, goal, etc.)
        3. Format as <roampal-user-profile> with one line per category

        Note: We don't prompt LLM to ask for user's name - small models don't reliably
        follow through on storing it. Users can share their name naturally.

        Args:
            mode: Kept for API compatibility but no longer affects behavior.

        Returns:
            Tuple of (formatted_profile, doc_ids, raw_facts) for outcome scoring.
        """
        if not self._memory_bank_service:
            return None, [], []

        try:
            # Get all facts from memory_bank, sorted by quality (importance first, then confidence)
            all_memory_bank = self._memory_bank_service.list_all(include_archived=False)
            logger.info(
                f"[COLD-START] Found {len(all_memory_bank)} total memory_bank facts"
            )

            sorted_facts = sorted(
                all_memory_bank,
                key=lambda f: (
                    f.get("metadata", {}).get("importance", 0.5),
                    f.get("metadata", {}).get("confidence", 0.5),
                ),
                reverse=True,
            )

            # Pick HIGHEST QUALITY fact for EACH tag category (one per tag)
            all_facts = []
            seen_tags = set()
            for fact in sorted_facts:
                tags_raw = fact.get("metadata", {}).get("tags", [])
                if isinstance(tags_raw, str):
                    try:
                        tags = json.loads(tags_raw) if tags_raw else []
                    except:
                        tags = []
                else:
                    tags = tags_raw or []

                # Find which priority tag this fact matches (if any)
                for tag in TAG_PRIORITIES:
                    if tag in tags and tag not in seen_tags:
                        all_facts.append(fact)
                        seen_tags.add(tag)
                        break  # One fact per tag

                # Stop once we have one fact per tag category
                if len(seen_tags) == len(TAG_PRIORITIES):
                    break

            # Check if user has NO identity at all
            identity_content = []
            for fact in all_facts:
                content = (
                    fact.get("text")
                    or fact.get("content")
                    or fact.get("metadata", {}).get("content", "")
                )
                tags_raw = fact.get("metadata", {}).get("tags", [])
                if isinstance(tags_raw, str):
                    try:
                        tags = json.loads(tags_raw) if tags_raw else []
                    except:
                        tags = []
                else:
                    tags = tags_raw or []
                if "identity" in tags:
                    identity_content.append(content)

            logger.info(
                f"[COLD-START] Identity check: {len(identity_content)} identity-tagged facts found"
            )
            # Note: We no longer prompt LLM to ask for name - small models don't reliably follow through.
            # If user shares their name naturally, they can store it. No forced identity prompts.

            # Build narrative profile - one line per category (compact ~200 chars)
            tag_labels = {
                "identity": "Identity",
                "preference": "Preference",
                "goal": "Goal",
                "project": "Project",
                "system_mastery": "System Mastery",
                "agent_growth": "Agent Growth",
            }

            # Group facts by their primary tag, keeping only FIRST fact per category
            category_facts = {}
            for fact in all_facts:
                content = (
                    fact.get("text")
                    or fact.get("content")
                    or fact.get("metadata", {}).get("content", "")
                )
                tags_raw = fact.get("metadata", {}).get("tags", [])
                if isinstance(tags_raw, str):
                    try:
                        tags = json.loads(tags_raw) if tags_raw else []
                    except:
                        tags = []
                else:
                    tags = tags_raw or []

                for tag in TAG_PRIORITIES:
                    if tag in tags and tag not in category_facts:
                        category_facts[tag] = content
                        break

            # Build compact narrative
            profile_parts = ["<roampal-user-profile>"]
            for tag in TAG_PRIORITIES:
                if tag in category_facts:
                    profile_parts.append(
                        f"{tag_labels[tag]}: {_first_sentence(category_facts[tag])}"
                    )
            profile_parts.append("</roampal-user-profile>")

            # Extract doc_ids for outcome scoring
            doc_ids = [f.get("id") for f in all_facts if f.get("id")]
            raw_facts = [
                {
                    "id": f.get("id"),
                    "content": f.get("text") or f.get("content", ""),
                    "source": "memory_bank",
                }
                for f in all_facts
            ]

            logger.info(
                f"[COLD-START] {len(all_facts)} facts, {len(category_facts)} categories"
            )
            return "\n".join(profile_parts), doc_ids, raw_facts

        except Exception as e:
            logger.error(f"[COLD-START] Error building profile: {e}")
            return None, [], []

    async def get_context_for_injection(
        self,
        query: str,
        conversation_id: str = None,
        recent_conversation: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get context to inject into LLM prompt for organic recall.

        v0.3.1: Two-lane retrieval matching benchmark (4 summaries + 4 facts = 8).
        TagCascade pipeline handles both lanes. No nursery slot (benchmark: p=1.0).

        Args:
            query: The user's message
            conversation_id: Current conversation ID
            recent_conversation: Recent messages for continuity

        Returns:
            Dict with memories, doc_ids for scoring, and formatted injection
        """
        if not self.initialized:
            await self.initialize()

        result = {
            "memories": [],
            "user_facts": [],
            "formatted_injection": "",
            "doc_ids": [],
        }

        # 0. Fetch always_inject memories (core identity - always included)
        always_inject_memories = self._memory_bank_service.get_always_inject()
        if always_inject_memories:
            result["user_facts"] = always_inject_memories
            for mem in always_inject_memories:
                if mem.get("id"):
                    result["doc_ids"].append(mem["id"])

        # v0.3.1: Two-lane retrieval (4 summaries + 4 facts = 8)
        all_collections = ["working", "patterns", "history", "memory_bank"]

        # Lane 1: summaries/context (4 slots)
        # memory_bank items included (no memory_type field, $ne "fact" includes them)
        summary_results = await self.search(
            query=query,
            limit=4,
            collections=all_collections,
            metadata_filters={"memory_type": {"$ne": "fact"}},
        )

        # Lane 2: facts (4 slots)
        # memory_bank excluded naturally (no items have memory_type: "fact")
        fact_results = await self.search(
            query=query,
            limit=4,
            collections=all_collections,
            metadata_filters={"memory_type": "fact"},
        )

        # Merge: 4 summaries + 4 facts
        all_results = summary_results + fact_results
        top_memories = [m for m in all_results if m.get("content") or m.get("text")]

        logger.info(
            f"[CONTEXT INJECTION] {len(summary_results)} summaries + {len(fact_results)} facts = {len(top_memories)} memories"
        )

        result["memories"] = top_memories
        result["relevant_memories"] = top_memories  # Alias for selective scoring
        result["doc_ids"] = [m.get("id") for m in top_memories if m.get("id")]
        result["formatted_injection"] = self._format_context_injection(result)

        return result

    @staticmethod
    def _humanize_age(iso_timestamp: str) -> str:
        """Convert ISO timestamp to human-readable relative age like '2d', '5h'."""
        if not iso_timestamp:
            return ""
        try:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            now = datetime.now()
            delta = now - dt
            if delta.total_seconds() < 0:
                return "now"
            days = delta.days
            hours = delta.seconds // 3600
            minutes = delta.seconds // 60
            if days > 365:
                return f"{days // 365}y"
            elif days > 30:
                return f"{days // 30}mo"
            elif days > 0:
                return f"{days}d"
            elif hours > 0:
                return f"{hours}h"
            elif minutes > 0:
                return f"{minutes}m"
            else:
                return "now"
        except Exception:
            return ""

    def _format_context_injection(self, context: Dict[str, Any]) -> str:
        """
        Format context for injection into LLM prompt.

        v0.3.1: Aligned with roampal-core format. Shows doc_id, age, wilson,
        uses, last_outcome. Separates summaries from facts. LLM can reference
        memories by [id:...] for scoring.
        """
        parts = []
        user_name = None

        # Find identity-tagged facts in memory_bank (no always_inject required)
        all_facts = self._memory_bank_service.list_all(include_archived=False)
        for fact in all_facts:
            # Check for identity tag
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    tags = []
            else:
                tags = tags_raw or []

            if "identity" not in tags:
                continue

            content = (
                fact.get("content")
                or fact.get("text")
                or fact.get("metadata", {}).get("text", "")
            )
            content_lower = content.lower()

            # Look for name patterns
            if (
                "name is" in content_lower
                or "i'm " in content_lower
                or "i am " in content_lower
            ):
                match = re.search(r"name is (\w+)", content, re.IGNORECASE)
                if match:
                    user_name = match.group(1)
                    break
                match = re.search(r"i[''`]?m (\w+)|i am (\w+)", content, re.IGNORECASE)
                if match:
                    user_name = match.group(1) or match.group(2)
                    break

        memories = context.get("memories", [])

        if user_name or memories:
            parts.append(
                "You have persistent memory about this user via Roampal. "
                "Memory tags: wilson:N% = reliability from past scoring, "
                "used:Nx = times retrieved, last:worked/failed/partial/unknown = "
                "whether this memory was *helpful* last time (not whether a task succeeded). "
                "[id:...] tags can be looked up with search_memory(id=...). "
                "Memories may be outdated or wrong. Verify before treating as ground truth. "
                "The context below was retrieved from past conversations. "
                "If the user references past interactions or asks if you remember them, "
                "use this context — you DO remember."
            )
            parts.append("")
            parts.append("═══ KNOWN CONTEXT ═══")

            if user_name:
                parts.append(f"User: {user_name}")

            # v0.3.1: Separate summaries and facts (matching core format)
            summaries = []
            facts = []
            for mem in memories:
                mem_type = mem.get("metadata", {}).get("memory_type", "")
                if mem_type == "fact":
                    facts.append(mem)
                else:
                    summaries.append(mem)

            def _format_mem(mem):
                content = (
                    mem.get("content")
                    or mem.get("text")
                    or mem.get("metadata", {}).get("text", "")
                )
                metadata = mem.get("metadata", {})
                collection = mem.get("collection", "unknown")
                doc_id = mem.get("id", "")

                # Age from timestamp
                created_at = metadata.get("created_at") or metadata.get("timestamp") or ""
                age = self._humanize_age(created_at)

                uses = int(metadata.get("uses", 0))
                wilson = mem.get("wilson_score", 0)
                last_outcome = metadata.get("last_outcome", "")

                tag_parts = []
                if age:
                    tag_parts.append(age)
                tag_parts.append(collection)
                if collection == "books":
                    tag_parts.append("reference")
                elif uses > 0:
                    tag_parts.append(f"wilson:{wilson:.0%}")
                    tag_parts.append(f"used:{uses}x")
                    if last_outcome:
                        tag_parts.append(f"last:{last_outcome}")

                id_str = f" [id:{doc_id}]" if doc_id else ""
                return f"• {content}{id_str} ({', '.join(tag_parts)})"

            if summaries:
                for s in summaries:
                    parts.append(_format_mem(s))

            if facts:
                parts.append("")
                parts.append(
                    "Facts (auto-extracted from conversation — use for direction, "
                    "not authority. Verify before citing as true):"
                )
                for f in facts:
                    parts.append(_format_mem(f))

            parts.append("═══ END CONTEXT ═══")
            parts.append("")
            formatted = "\n".join(parts)
            logger.info(
                f"[CONTEXT INJECTION] Formatted {len(memories)} memories into {len(formatted)} chars"
            )
            return formatted

        logger.info(
            f"[CONTEXT INJECTION] No user_name and no memories - returning empty"
        )
        return ""

    # Legacy alias for backward compatibility
    async def get_cold_start_context(
        self, limit: int = 5, mode: str = "internal"
    ) -> Tuple[Optional[str], List[str], List[Dict]]:
        """Legacy wrapper - use _build_cold_start_profile() instead.

        Args:
            limit: Unused, kept for API compatibility
            mode: "internal" for Desktop (create_memory), "mcp" for MCP server (add_to_memory_bank)
        """
        return await self._build_cold_start_profile(mode=mode)

    async def record_action_outcome(self, action) -> None:
        """No-op stub — KG action tracking removed in v0.3.1."""
        pass

    async def _update_kg_routing(self, query: str, collection: str, outcome: str) -> None:
        """No-op stub — KG routing removed in v0.3.1."""
        pass

    async def export_backup(self) -> Dict[str, Any]:
        """Export memory system state for backup."""
        backup = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "conversation_id": self.conversation_id,
            "stats": self.get_stats(),
        }

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
            logger.info(
                f"Restored backup from {backup_data.get('timestamp', 'unknown')}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return False

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

        # Cleanup collections
        for name, adapter in self.collections.items():
            try:
                await adapter.cleanup()
            except Exception:
                pass

        logger.info("UnifiedMemorySystem shutdown complete")

    # ==================== Backward Compatibility ====================

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
        results = await self.search(query="", collections=["working"], limit=limit)

        return {
            "conversation_id": self.conversation_id,
            "message_count": self.message_count,
            "recent_items": results,
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
                    if (
                        doc
                        and doc.get("metadata", {}).get("conversation_id")
                        == conversation_id
                    ):
                        to_delete.append(doc_id)

                if to_delete:
                    adapter.delete_vectors(to_delete)
                    deleted_count += len(to_delete)
                    logger.info(f"Deleted {len(to_delete)} items from {coll_name}")

            except Exception as e:
                logger.error(f"Error deleting from {coll_name}: {e}")

        return deleted_count

    async def search_books(
        self, query: str, limit: int = 5, title_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search books collection specifically."""
        results = await self.search(
            query=query,
            collections=["books"],
            limit=limit * 2,  # Get extra for filtering
        )

        if title_filter:
            results = [
                r
                for r in results
                if title_filter.lower()
                in r.get("metadata", {}).get("title", "").lower()
            ]

        return results[:limit]

    async def save_conversation_turn(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a conversation turn to working memory."""
        turn_metadata = {
            "role": role,
            "conversation_id": self.conversation_id,
            "message_number": self.message_count,
            **(metadata or {}),
        }

        self.increment_message_count()

        return await self.store(
            text=content, collection="working", metadata=turn_metadata
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
                "uploaded_at": datetime.now().isoformat(),
            }

            embedding = await self._embed_text(chunk)

            await self.collections["books"].upsert_vectors(
                ids=[doc_id],
                vectors=[embedding],
                metadatas=[{"text": chunk, "content": chunk, **metadata}],
            )
            ingested += 1

        logger.info(f"Ingested book '{title}' with {ingested} chunks")
        return ingested

    # ==================== KG Visualization API ====================

    # ==================== Stats API ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "conversation_id": self.conversation_id,
            "collections": {},
            "outcomes": {},
            "decay": {},
            "status": "active",
        }

        # Collection counts
        for name, adapter in self.collections.items():
            try:
                stats["collections"][name] = adapter.collection.count()
            except:
                stats["collections"][name] = 0

        return stats


# Backward compatibility alias
UMS = UnifiedMemorySystem
