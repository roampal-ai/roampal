"""
THE UNIFIED MEMORY SYSTEM FOR ROAMPAL
Memory-first architecture for intelligent conversation with learning capabilities
"""

import logging
import json
import uuid
import re
import asyncio
import math
from typing import List, Dict, Any, Optional, Literal, Set
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from filelock import FileLock

from modules.memory.chromadb_adapter import ChromaDBAdapter
from modules.memory.file_memory_adapter import FileMemoryAdapter
from modules.memory.content_graph import ContentGraph
from modules.embedding.embedding_service import EmbeddingService
from config.feature_flags import is_enabled, get_flag
from services.metrics_service import track_performance, get_metrics

# Import advanced modules conditionally
try:
    from modules.advanced.outcome_detector import OutcomeDetector
except ImportError:
    OutcomeDetector = None
try:
    from modules.memory.outcome_tracker import OutcomeTracker
except ImportError:
    OutcomeTracker = None

# Hybrid routing: KG-based automatic routing + optional LLM override
# LLM can explicitly specify collections OR let _route_query() use learned KG patterns
AutonomousRouter = None  # Legacy import, no longer needed

logger = logging.getLogger(__name__)

CollectionName = Literal["books", "working", "history", "patterns", "memory_bank"]


def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff for async functions"""
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
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator


class UnifiedMemorySystem:
    """
    THE single memory system for Roampal.
    Handles storage, search, learning, and promotion.

    5 Collections:
    - books: Uploaded reference material (never decays)
    - working: Current session context (session-scoped)
    - history: Past conversations (auto-promoted to patterns)
    - patterns: Proven solutions (what actually worked)
    - memory_bank: Persistent project/user context (LLM-controlled, never decays)

    PRODUCTION THRESHOLDS:
    - HIGH_VALUE_THRESHOLD: 0.9 (memories with this score are preserved beyond retention period)
    - PROMOTION_SCORE_THRESHOLD: 0.7 (minimum score for promotion to patterns)
    - DEMOTION_SCORE_THRESHOLD: 0.3 (below this, patterns demote to history)
    - DELETION_SCORE_THRESHOLD: 0.2 (below this, history items are deleted)

    PROMOTION TRIGGER PRECEDENCE (highest to lowest priority):
    1. Manual: Via switch-conversation API call (immediate)
    2. Auto-20-messages: Every 20 messages (immediate)
    3. Hourly: Background task every hour (delayed)
    """

    # Production-defined thresholds
    HIGH_VALUE_THRESHOLD = 0.9  # Memories above this are preserved beyond retention
    PROMOTION_SCORE_THRESHOLD = 0.7  # Minimum score for working->history or history->patterns
    DEMOTION_SCORE_THRESHOLD = 0.3  # Below this, patterns demote to history
    DELETION_SCORE_THRESHOLD = 0.2  # Below this, history items are deleted
    NEW_ITEM_DELETION_THRESHOLD = 0.1  # More lenient for items < 7 days old

    def __init__(self, data_dir: str = "./data", use_server: bool = True, llm_service=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Core components
        self.embedding_service = EmbeddingService()
        self.file_adapter = FileMemoryAdapter()

        # Advanced components (if enabled)
        self.outcome_detector = None
        if OutcomeDetector and is_enabled("ENABLE_OUTCOME_DETECTION"):
            self.outcome_detector = OutcomeDetector(
                llm_service if is_enabled("ENABLE_LLM_OUTCOME_DETECTION") else None
            )
            logger.info("OutcomeDetector enabled")

        # Outcome tracker for persistent outcome storage
        self.outcome_tracker = None
        if OutcomeTracker:
            self.outcome_tracker = OutcomeTracker(str(data_dir))
            logger.info("OutcomeTracker enabled")

        # Router removed - LLM has direct control via tool parameters
        self.router = None
        logger.info("Direct LLM collection control enabled (no router gatekeeping)")


        # LLM service for scoring (if enabled)
        self.llm_service = llm_service
        if llm_service and is_enabled("ENABLE_LLM_MEMORY_SCORING"):
            logger.info("LLM memory scoring enabled")

        # One adapter per collection
        self.collections: Dict[str, ChromaDBAdapter] = {}
        self.use_server = use_server
        self.initialized = False
        self._background_tasks = []  # Track background tasks for cleanup

        # Conversation tracking (standardized from session_id)
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_context = {}  # Persistent context across conversation
        self.message_count = 0  # Track messages for auto-promotion
        self._promotion_lock = asyncio.Lock()  # Prevent race conditions in auto-promotion

        # Knowledge graph save debouncing to prevent excessive file writes
        self._kg_save_pending = False
        self._kg_save_task: Optional[asyncio.Task] = None

        # Knowledge graph for routing (query patterns → collection routing)
        self.kg_path = self.data_dir / "knowledge_graph.json"
        self.knowledge_graph = self._load_kg()

        # CRITICAL: Content Knowledge Graph (entity relationships from memory content)
        # This is a CORE FEATURE - do not disable or remove
        # Provides the green/purple nodes in KG visualization
        self.content_graph_path = self.data_dir / "content_graph.json"
        self.content_graph = self._load_content_graph()
        logger.info(f"Content KG initialized with {len(self.content_graph.entities)} entities")

        # Memory relationships
        self.relationships_path = self.data_dir / "memory_relationships.json"
        self.relationships = self._load_relationships()

    def _load_content_graph(self) -> ContentGraph:
        """
        Load Content Knowledge Graph from disk.

        CRITICAL: This is a core feature for entity relationship mapping.
        Do not disable or remove - required for dual KG visualization.
        """
        if self.content_graph_path.exists():
            try:
                return ContentGraph.load_from_file(str(self.content_graph_path))
            except Exception as e:
                logger.warning(f"Failed to load content graph, creating new: {e}")
        return ContentGraph()

    def _load_kg(self) -> Dict[str, Any]:
        """Load knowledge graph routing patterns"""
        default_kg = {
            "routing_patterns": {},     # concept -> best_collection
            "success_rates": {},        # collection -> success_rate
            "failure_patterns": {},     # concept -> failure_reasons
            "problem_categories": {},   # problem_type -> preferred_collections
            "problem_solutions": {},    # problem_signature -> [solution_ids]
            "solution_patterns": {}     # pattern_hash -> {problem, solution, success_rate}
        }

        if self.kg_path.exists():
            try:
                with open(self.kg_path, 'r') as f:
                    loaded_kg = json.load(f)
                    # Ensure all required keys exist
                    for key in default_kg:
                        if key not in loaded_kg:
                            loaded_kg[key] = default_kg[key]
                    return loaded_kg
            except:
                pass
        return default_kg

    def _load_relationships(self) -> Dict[str, Any]:
        """Load memory relationships"""
        if self.relationships_path.exists():
            try:
                with open(self.relationships_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "related": {},      # doc_id -> [related_doc_ids]
            "evolution": {},    # doc_id -> {parent, children}
            "conflicts": {}     # doc_id -> [conflicting_doc_ids]
        }

    def _save_kg_sync(self):
        """
        Synchronous save both routing KG and content KG.

        CRITICAL: Saves both graphs atomically to maintain consistency.
        Do not remove content graph save - it's required for entity tracking.
        """
        # Save routing KG
        lock_path = str(self.kg_path) + ".lock"
        try:
            with FileLock(lock_path, timeout=10):
                self.kg_path.parent.mkdir(exist_ok=True, parents=True)
                # Write to temp file first then rename (atomic operation)
                temp_path = self.kg_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.knowledge_graph, f, indent=2)
                temp_path.replace(self.kg_path)
        except PermissionError as e:
            logger.error(f"Permission denied saving routing KG: {e}")
        except Exception as e:
            logger.error(f"Failed to save routing KG: {e}", exc_info=True)

        # CRITICAL: Save content KG (entity relationships)
        try:
            self.content_graph.save_to_file(str(self.content_graph_path))
        except Exception as e:
            logger.error(f"Failed to save content KG: {e}", exc_info=True)

    async def _save_kg(self):
        """Save knowledge graph asynchronously"""
        import asyncio
        await asyncio.to_thread(self._save_kg_sync)

    async def _debounced_save_kg(self):
        """Debounce KG saves to batch within 5-second window to reduce file I/O"""
        # Cancel existing pending save task
        if self._kg_save_task and not self._kg_save_task.done():
            self._kg_save_task.cancel()
            try:
                await self._kg_save_task
            except asyncio.CancelledError:
                pass

        # Create new delayed save task
        async def delayed_save():
            try:
                await asyncio.sleep(5)  # Wait 5 seconds to batch multiple updates
                await self._save_kg()
                self._kg_save_pending = False
            except asyncio.CancelledError:
                pass

        self._kg_save_pending = True
        self._kg_save_task = asyncio.create_task(delayed_save())

    def _save_relationships_sync(self):
        """Synchronous save memory relationships - to be called in thread with file locking"""
        lock_path = str(self.relationships_path) + ".lock"
        try:
            with FileLock(lock_path, timeout=10):
                self.relationships_path.parent.mkdir(exist_ok=True, parents=True)
                # Atomic write
                temp_path = self.relationships_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.relationships, f, indent=2)
                temp_path.replace(self.relationships_path)
        except PermissionError as e:
            logger.error(f"Permission denied saving relationships: {e}")
        except Exception as e:
            logger.error(f"Failed to save relationships: {e}", exc_info=True)

    async def _save_relationships(self):
        """Save memory relationships asynchronously"""
        import asyncio
        await asyncio.to_thread(self._save_relationships_sync)

    def set_llm_service(self, llm_service):
        """
        Inject LLM service after initialization (for outcome detection).
        Called after LLM client is ready in main.py startup.
        """
        self.llm_service = llm_service
        if self.outcome_detector:
            self.outcome_detector.llm_service = llm_service
            logger.info("LLM service injected into OutcomeDetector")

    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return

        logger.info("Initializing UnifiedMemorySystem...")

        # Initialize file adapter for session logs
        await self.file_adapter.initialize({
            "base_data_path": str(self.data_dir)
        })

        # Create ChromaDB collections
        collection_names = ["books", "working", "history", "patterns", "memory_bank"]

        for name in collection_names:
            adapter = ChromaDBAdapter(
                persistence_directory=str(self.data_dir / "chromadb"),
                use_server=self.use_server
            )
            await adapter.initialize(collection_name=f"roampal_{name}")
            self.collections[name] = adapter
            logger.info(f"Initialized {name} collection")

        # Track last promotion check time
        self._last_promotion_check = datetime.now()

        # Background promotion task is started in main.py (runs every 30 minutes)
        # Removed duplicate _background_promotion_loop to eliminate tech debt

        # Start startup cleanup (non-blocking)
        asyncio.create_task(self._startup_cleanup())

        self.initialized = True
        logger.info("UnifiedMemorySystem ready")

    async def store(
        self,
        text: str,
        collection: CollectionName = "history",
        metadata: Optional[Dict[str, Any]] = None,
        transparency_context: Optional[Any] = None
    ) -> str:
        """
        Store text in memory.

        Args:
            text: Content to store
            collection: Target collection (default: history)
            metadata: Additional metadata
            transparency_context: Optional context for tracking operations

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        # Track operation if context provided
        if transparency_context and hasattr(transparency_context, 'track_memory_store'):
            transparency_context.track_memory_store(
                text=text,
                collection=collection,
                doc_id="pending"  # Will update after store
            )

        # Generate ID and embedding
        doc_id = f"{collection}_{uuid.uuid4().hex[:8]}_{datetime.now().timestamp()}"
        embedding = await self.embedding_service.embed_text(text)

        # Prepare metadata with enhanced context
        final_metadata = {
            "text": text,
            "content": text,
            "collection": collection,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "uses": 0,
            "last_outcome": "unknown",
            "failure_reasons": json.dumps([]),  # JSON string for ChromaDB
            "success_contexts": json.dumps([]),  # JSON string for ChromaDB
            "problem_signature": metadata.get("query", "") if metadata else "",
            "original_context": metadata.get("query", "") if metadata else "",
            "promotion_history": json.dumps([]),  # JSON string for ChromaDB
            "persist_session": False  # Can survive session clear
        }

        # Only add score for outcome-based collections (working, history, patterns)
        # memory_bank and books use pure distance-based ranking (no stored score)
        if collection in ["working", "history", "patterns"]:
            final_metadata["score"] = 0.5  # Neutral start for outcome-based scoring

        if metadata:
            final_metadata.update(metadata)

        # Store in ChromaDB with retry
        @with_retry(max_attempts=3, delay=0.5)
        async def store_with_retry():
            await self.collections[collection].upsert_vectors(
                ids=[doc_id],
                vectors=[embedding],
                metadatas=[final_metadata]
            )

        await store_with_retry()

        # Auto-promotion every 20 messages (with lock to prevent race conditions)
        if collection == "working":
            async with self._promotion_lock:
                self.message_count += 1
                if self.message_count >= 20:
                    logger.info(f"Auto-promoting after {self.message_count} messages")

                    # Track auto-promotion if transparency context available
                    if transparency_context and hasattr(transparency_context, 'track_background_process'):
                        transparency_context.track_background_process(
                            process="auto_promotion",
                            description=f"Auto-promoting valuable memories after {self.message_count} messages",
                            metadata={"message_count": self.message_count, "trigger": "20_messages"}
                        )

                    # Fire and forget - don't block on promotion
                    # Capture current conversation_id to ensure correct context
                    current_conv_id = self.conversation_id
                    task = asyncio.create_task(
                        self._promote_valuable_working_memory(conversation_id=current_conv_id)
                    )
                    # Add error callback to log failures
                    task.add_done_callback(lambda t: self._handle_promotion_error(t))

                    # Track completion
                    if transparency_context:
                        transparency_context.track_action(
                            action_type="auto_promotion_triggered",
                            description=f"Triggered auto-promotion after {self.message_count} messages",
                            status="completed"
                        )

                    self.message_count = 0

        logger.debug(f"Stored {doc_id} in {collection}")
        return doc_id

    @track_performance("memory_search")
    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        collections: Optional[List[CollectionName]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        transparency_context: Optional[Any] = None
    ) -> Any:
        """
        Search memory with intelligent routing and optional metadata filtering.

        Args:
            query: Search query
            limit: Max results
            collections: Override automatic routing
            metadata_filters: ChromaDB where filters for exact metadata matching
                Examples:
                - {"title": "architecture"} - Books by exact title
                - {"author": "Smith"} - Books by author
                - {"has_code": True} - Book chunks containing code
                - {"source": "mcp_claude"} - Learnings from MCP
                - {"last_outcome": "worked"} - Successful learnings only
                - {"title": "architecture", "has_code": True} - Combined filters

        Returns:
            Ranked results
        """
        if not self.initialized:
            await self.initialize()

        # Use KG to route query if collections not specified
        if collections is None:
            collections = self._route_query(query)

        # Check for known problem→solution patterns
        known_solutions = await self._find_known_solutions(query)

        # Special handling for empty query - return all items
        if not query or query.strip() == "":
            all_results = []
            for coll_name in collections:
                if coll_name not in self.collections:
                    continue

                # Get all items from collection using a simple get
                adapter = self.collections[coll_name]

                # For working memory, get ALL items (no session filter for visualization)
                if coll_name == "working":
                    # Get all working memory items (for visualization/admin purposes)
                    try:
                        # Use ChromaDB's get() method to fetch all documents
                        collection_obj = adapter.collection
                        # Fetch ALL items from DB (ChromaDB max is 100000), then paginate after sorting
                        items = collection_obj.get(limit=100000)

                        # Format results
                        for i in range(len(items['ids'])):
                            metadata = items['metadatas'][i] if i < len(items['metadatas']) else {}
                            # Only include score if it exists in metadata
                            result = {
                                'id': items['ids'][i],
                                'content': items['documents'][i] if i < len(items['documents']) else '',
                                'text': items['documents'][i] if i < len(items['documents']) else '',
                                'metadata': metadata,
                                'collection': coll_name
                            }
                            # Working memory should always have score, but check anyway
                            if 'score' in metadata:
                                result['score'] = metadata['score']
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error getting all items from {coll_name}: {e}")
                else:
                    # Get all items from other collections
                    try:
                        collection_obj = adapter.collection
                        # Fetch ALL items from DB (ChromaDB max is 100000), then paginate after sorting
                        items = collection_obj.get(limit=100000)

                        # Format results
                        for i in range(len(items['ids'])):
                            metadata = items['metadatas'][i] if i < len(items['metadatas']) else {}
                            # Only include score if it exists in metadata
                            # memory_bank/books don't have scores (use distance-based later)
                            result = {
                                'id': items['ids'][i],
                                'content': items['documents'][i] if i < len(items['documents']) else '',
                                'text': items['documents'][i] if i < len(items['documents']) else '',
                                'metadata': metadata,
                                'collection': coll_name
                            }
                            # Only add score if it exists (working/history/patterns have it)
                            if 'score' in metadata:
                                result['score'] = metadata['score']
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error getting all items from {coll_name}: {e}")

            # Sort by timestamp for working memory items
            all_results.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)

            # Return paginated results
            paginated_results = all_results[offset:offset + limit]
            if return_metadata:
                return {
                    "results": paginated_results,
                    "total": len(all_results),
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + limit) < len(all_results)
                }
            else:
                # Backward compatibility - return list
                return paginated_results

        # TIER 2: Generate query embedding with error handling
        try:
            query_embedding = await self.embedding_service.embed_text(query)
        except Exception as e:
            logger.error(f"[MEMORY] Embedding generation failed for query '{query}': {e}", exc_info=True)
            # Return empty results if embedding fails
            if return_metadata:
                return {"results": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}
            else:
                return []

        # Track search start if context provided
        if transparency_context and hasattr(transparency_context, 'track_memory_search'):
            action = transparency_context.track_action(
                action_type="memory_search",
                description=f"Searching: {query[:50]}{'...' if len(query) > 50 else ''}",
                detail=f"Collections: {', '.join(collections)}",
                status="executing"
            )

        # Search specified collections
        all_results = []
        for coll_name in collections:
            if coll_name not in self.collections:
                continue

            # Working memory searches globally across all conversations
            if coll_name == "working":
                # Get ALL working memory across all conversations
                results = await self.collections[coll_name].query_vectors(
                    query_vector=query_embedding,
                    top_k=limit * 3,  # Get more for better ranking
                    filters=metadata_filters
                )

                # Add recency metadata for ALL results (no conversation filter)
                for r in results:
                    metadata = r.get("metadata", {})
                    if metadata.get("timestamp"):
                        try:
                            timestamp = datetime.fromisoformat(metadata["timestamp"])
                            minutes_ago = (datetime.now() - timestamp).total_seconds() / 60

                            # Add human-readable recency
                            if minutes_ago < 1:
                                metadata["recency"] = "just now"
                            elif minutes_ago < 5:
                                metadata["recency"] = f"{int(minutes_ago)} minutes ago"
                            elif minutes_ago < 60:
                                metadata["recency"] = f"{int(minutes_ago)} minutes ago"
                            else:
                                hours_ago = minutes_ago / 60
                                metadata["recency"] = f"{int(hours_ago)} hours ago"

                            # Also add for sorting
                            r["minutes_ago"] = minutes_ago
                        except:
                            r["minutes_ago"] = 999  # Unknown time = old

                # Sort by BOTH recency and relevance
                # Recent + relevant = best
                for r in results:
                    # Combine: lower distance (more relevant) + fewer minutes ago (more recent)
                    r["combined_score"] = r.get("distance", 1.0) + (r.get("minutes_ago", 999) / 100)

                results.sort(key=lambda x: x["combined_score"])
                results = results[:limit]
            else:
                # Books collection: get more results and boost recent uploads
                if coll_name == "books":
                    results = await self.collections[coll_name].query_vectors(
                        query_vector=query_embedding,
                        top_k=limit * 3,  # Get 3x results from books
                        filters=metadata_filters
                    )
                # Memory bank: exclude archived items by default (unless explicitly requested)
                elif coll_name == "memory_bank":
                    # Build filters: combine user filters + archived exclusion
                    memory_bank_filters = metadata_filters.copy() if metadata_filters else {}
                    # Only exclude archived if user hasn't explicitly set status filter
                    if "status" not in memory_bank_filters:
                        memory_bank_filters["status"] = {"$ne": "archived"}  # ChromaDB not-equals filter

                    results = await self.collections[coll_name].query_vectors(
                        query_vector=query_embedding,
                        top_k=limit,
                        filters=memory_bank_filters
                    )
                else:
                    results = await self.collections[coll_name].query_vectors(
                        query_vector=query_embedding,
                        top_k=limit,
                        filters=metadata_filters
                    )

            # Add source collection and boost based on collection type
            for r in results:
                r["collection"] = coll_name
                # Boost patterns slightly
                if coll_name == "patterns":
                    r["distance"] = r.get("distance", 1.0) * 0.9
                # Boost memory_bank by importance × confidence
                elif coll_name == "memory_bank":
                    metadata = r.get("metadata", {})
                    importance = metadata.get("importance", 0.7)
                    confidence = metadata.get("confidence", 0.7)
                    quality_score = importance * confidence

                    # Reduce distance for high-quality memories (0.5 = 50% max boost)
                    # quality_score=1.0 → 50% distance reduction → ranks much higher
                    # quality_score=0.5 → 25% distance reduction → moderate boost
                    # quality_score=0.0 → 0% distance reduction → no boost
                    r["distance"] = r.get("distance", 1.0) * (1.0 - quality_score * 0.5)
                # Boost recently uploaded books (within last 7 days)
                elif coll_name == "books" and r.get("upload_timestamp"):
                    from datetime import datetime, timedelta
                    try:
                        upload_time = datetime.fromisoformat(r["upload_timestamp"])
                        age_days = (datetime.utcnow() - upload_time).days
                        if age_days <= 7:
                            # Strong boost for recent uploads (30% better score)
                            r["distance"] = r.get("distance", 1.0) * 0.7
                    except:
                        pass

            all_results.extend(results)

        # Add known solutions to the beginning (they're already boosted)
        if known_solutions:
            # Remove duplicates (known solutions might also be in regular search)
            existing_ids = {r.get("id") for r in all_results}
            unique_known = [s for s in known_solutions if s.get("id") not in existing_ids]
            all_results = unique_known + all_results

        # LEARNING-AWARE RANKING: Combine embedding distance with learned scores
        for r in all_results:
            metadata = r.get("metadata", {})
            learned_score = metadata.get("score", 0.5)  # Default to neutral 0.5
            distance = r.get("distance", 1.0)

            # Convert distance to similarity (lower distance = higher similarity)
            embedding_similarity = 1.0 - min(distance, 1.0)

            # Combine: 70% embedding similarity + 30% learned score
            # This ensures embeddings still matter most, but learning influences ranking
            combined_score = (0.7 * embedding_similarity) + (0.3 * learned_score)

            # Store as negative distance for sorting (higher score = lower "distance")
            r["final_rank_score"] = combined_score
            r["original_distance"] = distance  # Keep for debugging

        # Sort by combined score (higher is better)
        all_results.sort(key=lambda x: x.get("final_rank_score", 0.0), reverse=True)

        # Track query for KG learning (only for returned results)
        paginated_results = all_results[offset:offset + limit]
        for result in paginated_results:
            # Only track if collection field exists
            if "collection" in result and "id" in result:
                self._track_usage(query, result["collection"], result["id"])

        # Add KG hints to results (expose learned patterns to LLM)
        for result in paginated_results:
            doc_id = result.get("id")
            metadata = result.get("metadata", {})

            # Check if this pattern has KG history
            if doc_id in self.knowledge_graph.get("success_rates", {}):
                stats = self.knowledge_graph["success_rates"][doc_id]
                if isinstance(stats, dict):
                    # New format: true success rate tracking
                    success_rate = stats.get("success_rate", 0.0)
                    success_count = stats.get("success_count", 0)
                    failure_count = stats.get("failure_count", 0)
                    total_uses = success_count + failure_count + stats.get("partial_count", 0)
                    if total_uses > 0:
                        result["kg_hint"] = f"This pattern: {success_count}/{total_uses} successful ({int(success_rate*100)}%)"
                else:
                    # Old format: just a score
                    success_rate = stats
                    uses = metadata.get("uses", 0)
                    if uses > 0:
                        result["kg_hint"] = f"Score: {success_rate:.2f} ({uses} uses)"

            # Check for routing pattern hints
            for concept, pattern in self.knowledge_graph.get("routing_patterns", {}).items():
                if concept.lower() in query.lower():
                    result["kg_routing_hint"] = f"Similar queries ({concept}) had {int(pattern.get('success_rate', 0)*100)}% success rate"
                    break

        # Track search completion and add citations if context provided
        if transparency_context:
            # Complete the search action
            if hasattr(transparency_context, 'track_memory_search'):
                # Use exponential decay for confidence calculation (same as agent_chat.py)
                CONFIDENCE_SCALE_FACTOR = 100.0
                confidence_scores = [math.exp(-r.get("distance", 0.5) / CONFIDENCE_SCALE_FACTOR) for r in paginated_results]
                transparency_context.track_memory_search(
                    query=query,
                    collections=collections,
                    results_count=len(paginated_results),
                    confidence_scores=confidence_scores
                )

            # Add citations for top results
            if hasattr(transparency_context, 'add_citation'):
                for i, result in enumerate(paginated_results[:3]):  # Top 3 results
                    metadata = result.get("metadata", {})
                    # Use exponential decay for confidence calculation (same as agent_chat.py)
                    CONFIDENCE_SCALE_FACTOR = 100.0
                    confidence = math.exp(-result.get("distance", 0.5) / CONFIDENCE_SCALE_FACTOR)
                    transparency_context.add_citation(
                        source=metadata.get("text", "")[:100],
                        confidence=confidence,
                        collection=result.get("collection", "unknown"),
                        text=metadata.get("text"),
                        doc_id=result.get("id")
                    )

        # Return paginated results with metadata
        if return_metadata:
            return {
                "results": paginated_results,
                "total": len(all_results),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < len(all_results)
            }
        else:
            # Backward compatibility - return list
            return paginated_results

    async def search_books(
        self,
        query: str,
        chunk_type: str = None,
        has_code: bool = None,
        code_language: str = None,
        n_results: int = 5
    ) -> List[dict]:
        """
        Enhanced book search with metadata filtering.

        Args:
            query: Search query text
            chunk_type: Filter by chunk type ("code", "prose", "mixed")
            has_code: Filter by presence of code (True/False)
            code_language: Filter by programming language
            n_results: Number of results to return

        Returns:
            List of search results with enhanced context
        """
        if "books" not in self.collections:
            logger.warning("Books collection not initialized")
            return []

        # Build ChromaDB where clause
        where = {}

        if chunk_type:
            where["chunk_type"] = chunk_type

        if has_code is not None:
            where["has_code"] = has_code

        if code_language:
            where["code_language"] = code_language

        try:
            # Search with optional filters
            results = await self.collections["books"].query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more candidates for re-ranking
                where=where if where else None
            )

            if not results or not results.get("ids"):
                return []

            # Convert to standardized format with enhanced context
            formatted_results = []
            for i in range(min(n_results, len(results["ids"][0]))):
                meta = results["metadatas"][0][i]
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": meta,
                    "distance": results["distances"][0][i],
                    "context": {
                        "type": meta.get("chunk_type", "unknown"),
                        "topic": meta.get("primary_topic", "general"),
                        "position": f"{int(meta.get('doc_position', 0) * 100)}% through doc",
                        "has_code": meta.get("has_code", False),
                        "language": meta.get("code_language")
                    }
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in enhanced book search: {e}", exc_info=True)
            return []

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all memories associated with a specific conversation"""
        try:
            deleted_count = 0

            # Delete from all collections
            for collection_name, adapter in self.collections.items():
                try:
                    # ChromaDB delete by metadata filter
                    result = adapter.collection.delete(
                        where={"conversation_id": conversation_id}
                    )

                    if result:
                        deleted_count += len(result) if isinstance(result, list) else 1
                        logger.info(f"Deleted memories from {collection_name} for conversation {conversation_id}")

                except Exception as e:
                    logger.warning(f"Failed to delete from {collection_name}: {e}")
                    continue

            logger.info(f"Deleted {deleted_count} total memories for conversation {conversation_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting memories for conversation {conversation_id}: {e}", exc_info=True)
            return 0

    async def get_cold_start_context(self, limit: int = 5) -> Optional[str]:
        """
        Generate cold-start user profile from Content KG top entities.

        Uses Content KG to find most important entities (by mention count),
        then retrieves their source memory_bank documents.

        Args:
            limit: Maximum number of memory_bank facts to include

        Returns:
            Formatted string with top memory_bank facts, or None if unavailable
        """
        if not self.content_graph or len(self.content_graph.entities) == 0:
            logger.info("[COLD-START] Content KG empty, falling back to vector search")
            # Fallback to vector search
            try:
                results = await self.search(
                    query="user identity name projects current work goals",
                    collections=["memory_bank"],
                    limit=limit
                )
                return self._format_cold_start_results(results)
            except Exception as e:
                logger.error(f"[COLD-START] Fallback search failed: {e}")
                return None

        # Get top entities by mentions (most important)
        top_entities = self.content_graph.get_all_entities(min_mentions=1)[:10]

        if not top_entities:
            logger.info("[COLD-START] No entities in Content KG yet")
            return None

        logger.info(f"[COLD-START] Top entities: {[e['entity'] for e in top_entities[:5]]}")

        # Collect unique memory_bank document IDs from top entities
        seen_ids = set()
        memory_ids = []

        for entity in top_entities:
            for doc_id in entity.get("documents", []):
                # Only include memory_bank documents
                if doc_id not in seen_ids and doc_id.startswith("memory_bank_"):
                    seen_ids.add(doc_id)
                    memory_ids.append(doc_id)
                    if len(memory_ids) >= limit:
                        break
            if len(memory_ids) >= limit:
                break

        if not memory_ids:
            logger.info("[COLD-START] No memory_bank documents found in top entities")
            return None

        logger.info(f"[COLD-START] Retrieving {len(memory_ids)} memory_bank documents: {memory_ids}")

        # Retrieve actual memory_bank documents by ID
        memories = []
        try:
            adapter = self.collections["memory_bank"]
            result = await adapter.get_vectors_by_ids(memory_ids)

            logger.info(f"[COLD-START] ChromaDB result keys: {result.keys() if result else 'None'}")

            # ChromaDB returns {ids: [...], documents: [...], metadatas: [...]}
            # Note: documents can be None if IDs don't exist
            documents = result.get('documents') if result else None
            if documents is None:
                documents = []

            logger.info(f"[COLD-START] Documents count: {len(documents)}")

            if result and documents:
                for i, doc_id in enumerate(result.get("ids", [])):
                    if i < len(documents):
                        memories.append({
                            "id": doc_id,
                            "content": documents[i],
                            "text": documents[i],
                            "metadata": result.get("metadatas", [])[i] if i < len(result.get("metadatas", [])) else {}
                        })
                logger.info(f"[COLD-START] Retrieved {len(memories)} memory_bank documents")
            else:
                logger.warning(f"[COLD-START] ChromaDB returned empty or no documents (requested IDs may not exist)")
        except Exception as e:
            logger.error(f"[COLD-START] Failed to retrieve memory_bank documents: {e}", exc_info=True)
            return None

        # If Content KG had entities but retrieval returned 0 documents (stale IDs), fall back to vector search
        if not memories:
            logger.warning(f"[COLD-START] Content KG entities point to non-existent documents (stale data), falling back to vector search")
            try:
                results = await self.search(
                    query="user identity name projects current work goals",
                    collections=["memory_bank"],
                    limit=5
                )
                return self._format_cold_start_results(results)
            except Exception as e:
                logger.error(f"[COLD-START] Fallback vector search failed: {e}")
                return None

        return self._format_cold_start_results(memories)

    def _smart_truncate(self, text: str, max_len: int = 250) -> str:
        """Truncate text at sentence/word boundary, not mid-word"""
        if len(text) <= max_len:
            return text

        truncated = text[:max_len]

        # Try sentence boundary first (period + space)
        last_period = truncated.rfind('. ')
        if last_period > max_len * 0.6:  # At least 60% of target length
            return truncated[:last_period + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space] + '...'

        return truncated + '...'

    def _format_cold_start_results(self, results: List[Dict]) -> Optional[str]:
        """Format cold-start memories into system message with injection protection"""
        if not results:
            return None

        # INJECTION PROTECTION (Layer 4): Filter suspicious content
        safe_results = [
            r for r in results
            if not any(
                x in (r.get("content") or r.get("text", "")).lower()
                for x in ["ignore all previous", "hacked", "pwned", "ignore instructions"]
            )
        ][:10]  # Limit to top 10 after filtering

        if not safe_results:
            logger.warning("[COLD-START] All results filtered by Layer 4 injection protection")
            return None

        logger.info(f"[COLD-START] Formatted {len(safe_results)} safe memory_bank facts")

        context_summary = "📋 **User Profile** (auto-loaded from your most important stored facts):\n" + "\n".join([
            f"• {self._smart_truncate((r.get('content') or r.get('text', '')).replace(chr(10), ' '), 250)}"
            for r in safe_results
        ])

        return context_summary

    def _calculate_tier_scores(self, concepts: List[str]) -> Dict[str, float]:
        """
        Calculate tier scores for each collection based on learned patterns.
        Implements architecture.md tier scoring formula:

        tier_score = success_rate * confidence
        where:
          success_rate = successes / (successes + failures)
          confidence = min(total_uses / 10, 1.0)

        Returns dict mapping collection_name → total_score
        """
        collection_scores = {
            "working": 0.0,
            "patterns": 0.0,
            "history": 0.0,
            "books": 0.0,
            "memory_bank": 0.0
        }

        # Aggregate scores across all concepts
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern_data = self.knowledge_graph["routing_patterns"][concept]
                collections_used = pattern_data.get("collections_used", {})

                for collection, stats in collections_used.items():
                    successes = stats.get("successes", 0)
                    failures = stats.get("failures", 0)
                    partials = stats.get("partials", 0)
                    total_uses = successes + failures + partials

                    # Calculate success_rate (exclude partials from denominator)
                    if successes + failures > 0:
                        success_rate = successes / (successes + failures)
                    else:
                        success_rate = 0.5  # Neutral for no confirmed outcomes

                    # Calculate confidence (reaches 1.0 after 10 uses)
                    confidence = min(total_uses / 10.0, 1.0)

                    # Tier score
                    tier_score = success_rate * confidence

                    # Add to collection's total score
                    collection_scores[collection] += tier_score

        return collection_scores

    def _route_query(self, query: str) -> List[str]:
        """
        Intelligent routing using learned KG patterns.
        Implements architecture.md specification with learning phases:

        Phase 1 (Exploration): total_score < 0.5 → search all 5 collections
        Phase 2 (Medium Confidence): 0.5 ≤ total_score < 2.0 → search top 2-3 collections
        Phase 3 (High Confidence): total_score ≥ 2.0 → search top 1-2 collections

        Returns list of collection names to search.
        """
        # Extract concepts from query
        concepts = self._extract_concepts(query)

        if not concepts:
            logger.debug(f"[Routing] No concepts extracted, searching all collections")
            return ["working", "patterns", "history", "books", "memory_bank"]

        # Calculate tier scores for each collection
        collection_scores = self._calculate_tier_scores(concepts)

        # Calculate total score (sum of all collection scores)
        total_score = sum(collection_scores.values())

        # Sort collections by score (highest first)
        sorted_collections = sorted(
            collection_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply routing thresholds
        if total_score < 0.5:
            # EXPLORATION PHASE: No learned patterns yet, search everything
            selected = ["working", "patterns", "history", "books", "memory_bank"]
            logger.info(f"[Routing] Exploration phase (score={total_score:.2f}): searching all collections")

        elif total_score < 2.0:
            # MEDIUM CONFIDENCE: Search top 2-3 collections
            # Take top collections with score > 0.1, up to 3
            selected = [
                coll for coll, score in sorted_collections[:3]
                if score > 0.1
            ]
            if not selected:
                selected = [sorted_collections[0][0]]  # At least take top 1
            logger.info(f"[Routing] Medium confidence (score={total_score:.2f}): searching {selected}")

        else:
            # HIGH CONFIDENCE: Search top 1-2 collections
            # Take top collections with score > 0.5, up to 2
            selected = [
                coll for coll, score in sorted_collections[:2]
                if score > 0.5
            ]
            if not selected:
                selected = [sorted_collections[0][0]]  # At least take top 1
            logger.info(f"[Routing] High confidence (score={total_score:.2f}): searching {selected}")

        # Log concept extraction and scores for debugging
        logger.debug(f"[Routing] Concepts: {concepts[:5]}...")
        logger.debug(f"[Routing] Scores: {dict(sorted_collections[:3])}")

        # Track usage for KG visualization (increment 'total' for used patterns)
        # This makes MCP-searched patterns visible in UI even without explicit outcome feedback
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern = self.knowledge_graph["routing_patterns"][concept]
                collections_used = pattern.get("collections_used", {})

                # Increment total for each collection that was selected for search
                for collection in selected:
                    if collection in collections_used:
                        collections_used[collection]["total"] = collections_used[collection].get("total", 0) + 1
                    else:
                        # Initialize if this collection not tracked yet
                        collections_used[collection] = {
                            "successes": 0,
                            "failures": 0,
                            "partials": 0,
                            "total": 1
                        }

                # Update last_used timestamp
                pattern["last_used"] = datetime.now().isoformat()

        return selected

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract N-grams (unigrams, bigrams, trigrams) from text for KG routing.
        Implements architecture.md specification for concept extraction.
        """
        concepts = set()

        # Normalize and tokenize
        text_lower = text.lower()
        # Remove punctuation except hyphens and underscores
        text_clean = re.sub(r'[^\w\s\-_]', ' ', text_lower)
        words = text_clean.split()

        # Stop words (expanded set)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "for", "of",
            "with", "in", "on", "at", "by", "from", "as", "be", "this", "that",
            "it", "i", "you", "we", "they", "my", "your", "our", "their", "what",
            "when", "where", "how", "why", "can", "could", "would", "should"
        }

        # Filter stop words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

        # 1. Extract UNIGRAMS (single words)
        for word in filtered_words:
            if len(word) > 3:  # Only meaningful words
                concepts.add(word)

        # 2. Extract BIGRAMS (2-word phrases)
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]}_{filtered_words[i+1]}"
            concepts.add(bigram)

        # 3. Extract TRIGRAMS (3-word phrases)
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]}_{filtered_words[i+1]}_{filtered_words[i+2]}"
            concepts.add(trigram)

        # 4. Extract technical patterns (CamelCase, snake_case, ErrorTypes)
        technical_patterns = [
            r'[A-Z][a-z]+(?:[A-Z][a-z]+)+',  # CamelCase
            r'[a-z]+_[a-z]+(?:_[a-z]+)*',     # snake_case
            r'\b\w+Error\b',  # ErrorTypes
            r'\b\w+Exception\b',  # ExceptionTypes
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(m.lower() for m in matches)

        # Return up to 15 concepts (prioritize longer n-grams for specificity)
        sorted_concepts = sorted(concepts, key=lambda x: len(x.split('_')), reverse=True)
        return sorted_concepts[:15]

    def _track_usage(self, query: str, collection: str, doc_id: str):
        """Track which collection was used for which query"""
        concepts = self._extract_concepts(query)
        for concept in concepts:
            if concept not in self.knowledge_graph["routing_patterns"]:
                self.knowledge_graph["routing_patterns"][concept] = {
                    "collections_used": {},
                    "best_collection": collection,
                    "success_rate": 0.5
                }
            # Will be updated when outcome is recorded

    async def record_outcome(
        self,
        doc_id: str,
        outcome: Literal["worked", "failed", "partial"],
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record outcome and trigger learning.

        Args:
            doc_id: Document that was used
            outcome: Whether it worked
        """
        # Get document to find collection (needed for KG routing update)
        collection_name = None
        doc = None

        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                collection_name = coll_name
                doc = adapter.get_fragment(doc_id)
                break

        if not doc:
            logger.warning(f"Document {doc_id} not found")
            return

        # UPDATE KG ROUTING FIRST - even for books/memory_bank
        # This allows KG to learn which collections answer which queries
        metadata = doc.get("metadata", {})
        problem_text = metadata.get("query", "")
        if problem_text and collection_name:
            await self._update_kg_routing(problem_text, collection_name, outcome)
            logger.info(f"[KG] Updated routing for '{problem_text[:50]}' → {collection_name} (outcome={outcome})")

        # SAFEGUARD: Books are reference material, not scorable memories
        # But we still updated KG routing above so system learns to route to books
        if doc_id.startswith("books_"):
            logger.info(f"[KG] Learned routing pattern for books, but skipping score update (static reference material)")
            return

        # SAFEGUARD: Memory bank is user identity/facts, not scorable patterns
        # But we still updated KG routing above so system learns to route to memory_bank
        if doc_id.startswith("memory_bank_"):
            logger.info(f"[KG] Learned routing pattern for memory_bank, but skipping score update (persistent user facts)")
            return

        # Outcome tracking is active and learning from conversation patterns

        # Time-weighted score update
        metadata = doc.get("metadata", {})
        current_score = metadata.get("score", 0.5)
        uses = metadata.get("uses", 0)

        # Calculate time weight (recent outcomes matter more)
        last_used = metadata.get("last_used")
        if last_used:
            age_days = (datetime.now() - datetime.fromisoformat(last_used)).days
            time_weight = 1.0 / (1 + age_days / 30)  # Decay over month
        else:
            time_weight = 1.0

        if outcome == "worked":
            score_delta = 0.2 * time_weight
            new_score = min(1.0, current_score + score_delta)
            uses += 1
            if context:
                contexts = json.loads(metadata.get("success_contexts", "[]"))
                contexts.append(context)
                metadata["success_contexts"] = json.dumps(contexts)
        elif outcome == "failed":
            score_delta = -0.3 * time_weight
            new_score = max(0.0, current_score + score_delta)
            if failure_reason:
                reasons = json.loads(metadata.get("failure_reasons", "[]"))
                reasons.append({
                    "reason": failure_reason,
                    "timestamp": datetime.now().isoformat()
                })
                metadata["failure_reasons"] = json.dumps(reasons)
        else:  # partial
            score_delta = 0.05 * time_weight
            new_score = min(1.0, current_score + score_delta)
            uses += 1

        # Log score update for transparency/debugging
        logger.info(
            f"Score update [{collection_name}]: {current_score:.2f} → {new_score:.2f} "
            f"(outcome={outcome}, delta={score_delta:+.2f}, time_weight={time_weight:.2f}, uses={uses})"
        )

        # Update outcome history
        outcome_history = json.loads(metadata.get("outcome_history", "[]"))
        outcome_history.append({
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "reason": failure_reason
        })
        outcome_history = outcome_history[-10:]  # Keep last 10

        metadata.update({
            "score": new_score,
            "uses": uses,
            "last_outcome": outcome,
            "last_used": datetime.now().isoformat(),
            "outcome_history": json.dumps(outcome_history)
        })

        # Update in ChromaDB
        self.collections[collection_name].update_fragment_metadata(doc_id, metadata)

        # Handle promotion/demotion with dynamic thresholds
        collection_size = self.collections[collection_name].collection.count()
        await self._handle_promotion(doc_id, collection_name, new_score, uses, metadata, collection_size)

        # Update KG with both problem and solution for proper relationship building
        # Note: routing patterns already updated above (before safeguard checks)
        problem_text = metadata.get("query", "")
        solution_text = doc.get("content", "")

        # Build relationships and patterns when outcome is positive
        if outcome == "worked" and problem_text and solution_text:
            # Extract concepts from both problem and solution
            problem_concepts = self._extract_concepts(problem_text)
            solution_concepts = self._extract_concepts(solution_text)
            all_concepts = list(set(problem_concepts + solution_concepts))

            # Build relationships between all concepts
            self._build_concept_relationships(all_concepts)

            # Update problem categories
            problem_key = "_".join(sorted(problem_concepts)[:3])  # Key from first 3 concepts
            if problem_key not in self.knowledge_graph["problem_categories"]:
                self.knowledge_graph["problem_categories"][problem_key] = []
            if doc_id not in self.knowledge_graph["problem_categories"][problem_key]:
                self.knowledge_graph["problem_categories"][problem_key].append(doc_id)

            # Update solution patterns
            solution_key = f"solution_{doc_id}"
            self.knowledge_graph["solution_patterns"][solution_key] = {
                "solution": solution_text[:200],  # First 200 chars
                "success_rate": new_score,
                "problems_solved": [problem_key],
                "concepts": solution_concepts[:5]  # Top 5 concepts
            }

            # Track true success rates per fragment (not just current score)
            if doc_id not in self.knowledge_graph["success_rates"]:
                self.knowledge_graph["success_rates"][doc_id] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "partial_count": 0,
                    "success_rate": 0.0
                }

            stats = self.knowledge_graph["success_rates"][doc_id]
            stats["success_count"] += 1
            total = stats["success_count"] + stats["failure_count"] + stats["partial_count"]
            if total > 0:
                stats["success_rate"] = stats["success_count"] / total

            # Save KG after updates (debounced)
            await self._debounced_save_kg()

        # Track failure patterns when something doesn't work
        elif outcome == "failed":
            # Track fragment failure statistics
            if doc_id not in self.knowledge_graph["success_rates"]:
                self.knowledge_graph["success_rates"][doc_id] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "partial_count": 0,
                    "success_rate": 0.0
                }

            stats = self.knowledge_graph["success_rates"][doc_id]
            stats["failure_count"] += 1
            total = stats["success_count"] + stats["failure_count"] + stats["partial_count"]
            if total > 0:
                stats["success_rate"] = stats["success_count"] / total

            # Track failure patterns
            if failure_reason:
                failure_key = failure_reason[:50]
                if failure_key not in self.knowledge_graph["failure_patterns"]:
                    self.knowledge_graph["failure_patterns"][failure_key] = []
                self.knowledge_graph["failure_patterns"][failure_key].append({
                    "doc_id": doc_id,
                    "timestamp": datetime.now().isoformat(),
                    "problem": problem_text[:100]
                })

            await self._debounced_save_kg()

        # Track partial outcomes
        elif outcome == "partial":
            if doc_id not in self.knowledge_graph["success_rates"]:
                self.knowledge_graph["success_rates"][doc_id] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "partial_count": 0,
                    "success_rate": 0.0
                }

            stats = self.knowledge_graph["success_rates"][doc_id]
            stats["partial_count"] += 1
            total = stats["success_count"] + stats["failure_count"] + stats["partial_count"]
            if total > 0:
                stats["success_rate"] = stats["success_count"] / total

            await self._debounced_save_kg()

        # Track problem→solution patterns when something works
        if outcome == "worked":
            await self._track_problem_solution(doc_id, metadata, context)

        logger.info(f"Outcome recorded: {doc_id} -> {outcome} (score: {new_score:.2f})")

    async def _promote_item(
        self,
        doc_id: str,
        from_collection: str,
        to_collection: str,
        metadata: Dict[str, Any]
    ):
        """
        Move an item from one collection to another.
        """
        try:
            # Get the document from source collection
            doc = self.collections[from_collection].get_fragment(doc_id)
            if not doc:
                logger.warning(f"Cannot promote {doc_id}: not found in {from_collection}")
                return

            # Add promotion metadata
            metadata["promoted_from"] = from_collection
            metadata["promoted_at"] = datetime.now().isoformat()
            metadata["original_id"] = doc_id

            # Store in target collection
            new_id = f"{to_collection}_{uuid.uuid4().hex[:8]}"
            await self.store(
                text=doc.get("content", ""),
                collection=to_collection,
                metadata=metadata
            )

            # Remove from source collection (optional - could keep for history)
            # self.collections[from_collection].delete_fragment(doc_id)

            logger.info(f"Promoted {doc_id} from {from_collection} to {to_collection} as {new_id}")

        except Exception as e:
            logger.error(f"Failed to promote item: {e}")

    async def _handle_promotion(
        self,
        doc_id: str,
        collection: str,
        score: float,
        uses: int,
        metadata: Dict[str, Any],
        collection_size: int = 0
    ):
        """Handle automatic promotion/demotion using outcome-based thresholds"""

        # Get full document for evaluation
        doc = self.collections[collection].get_fragment(doc_id)
        if not doc:
            logger.warning(f"Cannot evaluate {doc_id}: document not found")
            return

        # Fast-track: working -> patterns (exceptional cases with proven track record)
        if collection == "working" and score >= self.HIGH_VALUE_THRESHOLD and uses >= 3:
            # Check for consecutive successful outcomes
            outcome_history = json.loads(metadata.get("outcome_history", "[]"))
            recent_outcomes = [o.get("outcome") for o in outcome_history[-3:]]

            if recent_outcomes.count("worked") >= 3:
                # Three consecutive successes = fast-track to patterns
                logger.info(f"Fast-track promotion: {doc_id} from working → patterns "
                           f"(score: {score:.2f}, {uses} uses, 3+ consecutive successes)")

                new_id = doc_id.replace("working_", "patterns_")
                promotion_record = {
                    "from": "working",
                    "to": "patterns",
                    "timestamp": datetime.now().isoformat(),
                    "score": score,
                    "uses": uses,
                    "fast_tracked": True
                }

                promotion_history = json.loads(metadata.get("promotion_history", "[]"))
                promotion_history.append(promotion_record)
                metadata["promotion_history"] = json.dumps(promotion_history)
                metadata["promoted_from"] = "working"
                metadata["fast_tracked"] = True

                await self.collections["patterns"].upsert_vectors(
                    ids=[new_id],
                    vectors=[await self.embedding_service.embed_text(metadata["text"])],
                    metadatas=[metadata]
                )
                self.collections["working"].delete_vectors([doc_id])
                await self._add_relationship(new_id, "evolution", {"parent": doc_id})

                logger.info(f"Fast-tracked {doc_id} to patterns")
                return  # Exit early

        # Promotion: working -> history
        if collection == "working" and score >= 0.7 and uses >= 2:
            new_id = doc_id.replace("working_", "history_")
            promotion_record = {
                "from": "working",
                "to": "history",
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "uses": uses
            }
            promotion_history = json.loads(metadata.get("promotion_history", "[]"))
            promotion_history.append(promotion_record)
            metadata["promotion_history"] = json.dumps(promotion_history)
            metadata["promoted_from"] = "working"

            await self.collections["history"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embedding_service.embed_text(metadata["text"])],
                metadatas=[metadata]
            )
            self.collections["working"].delete_vectors([doc_id])
            await self._add_relationship(new_id, "evolution", {"parent": doc_id})

            logger.info(f"Promoted {doc_id} from working → history (score: {score:.2f}, uses: {uses})")

        # Promotion: history -> patterns
        elif collection == "history" and score >= self.HIGH_VALUE_THRESHOLD and uses >= 3:
            new_id = doc_id.replace("history_", "patterns_")
            promotion_record = {
                "from": "history",
                "to": "patterns",
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "uses": uses
            }
            promotion_history = json.loads(metadata.get("promotion_history", "[]"))
            promotion_history.append(promotion_record)
            metadata["promotion_history"] = json.dumps(promotion_history)
            metadata["promoted_from"] = "history"

            await self.collections["patterns"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embedding_service.embed_text(metadata["text"])],
                metadatas=[metadata]
            )
            self.collections["history"].delete_vectors([doc_id])
            await self._add_relationship(new_id, "evolution", {"parent": doc_id})

            logger.info(f"Promoted {doc_id} from history → patterns (score: {score:.2f}, uses: {uses})")

        # Demotion: patterns -> history
        elif collection == "patterns" and score < self.DEMOTION_SCORE_THRESHOLD:
            new_id = doc_id.replace("patterns_", "history_")
            await self.collections["history"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embedding_service.embed_text(metadata["text"])],
                metadatas=[{**metadata, "demoted_from": "patterns"}]
            )
            self.collections["patterns"].delete_vectors([doc_id])
            logger.info(f"Demoted {doc_id} to history (score: {score:.2f})")

        # Deletion: score too low
        elif score < self.DELETION_SCORE_THRESHOLD:
            # Give new items a chance, delete persistent failures
            age_days = 0
            if metadata.get("timestamp"):
                try:
                    age_days = (datetime.now() - datetime.fromisoformat(metadata["timestamp"])).days
                except:
                    pass

            # Newer items get more lenient threshold
            deletion_threshold = self.DELETION_SCORE_THRESHOLD if age_days > 7 else self.NEW_ITEM_DELETION_THRESHOLD

            if score < deletion_threshold:
                self.collections[collection].delete_vectors([doc_id])
                logger.info(f"Deleted {doc_id} from {collection} (score {score:.2f} < threshold {deletion_threshold})")

    def _build_concept_relationships(self, concepts: List[str]):
        """Build relationships between co-occurring concepts"""
        if "relationships" not in self.knowledge_graph:
            self.knowledge_graph["relationships"] = {}

        # Build relationships between all concept pairs
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Create bidirectional relationship key (sorted for consistency)
                rel_key = "|".join(sorted([concept1, concept2]))

                if rel_key not in self.knowledge_graph["relationships"]:
                    self.knowledge_graph["relationships"][rel_key] = {
                        "co_occurrence": 0,
                        "success_together": 0,
                        "failure_together": 0
                    }

                # Increment co-occurrence
                self.knowledge_graph["relationships"][rel_key]["co_occurrence"] += 1

    async def _update_kg_routing(self, query: str, collection: str, outcome: str):
        """Update KG routing patterns and relationships based on outcome"""
        if not query:
            return

        concepts = self._extract_concepts(query)

        # Build relationships between concepts
        self._build_concept_relationships(concepts)

        for concept in concepts:
            if concept not in self.knowledge_graph["routing_patterns"]:
                self.knowledge_graph["routing_patterns"][concept] = {
                    "collections_used": {},
                    "best_collection": collection,
                    "success_rate": 0.5
                }

            pattern = self.knowledge_graph["routing_patterns"][concept]

            # Track collection performance
            if collection not in pattern["collections_used"]:
                pattern["collections_used"][collection] = {
                    "successes": 0,
                    "failures": 0,
                    "total": 0
                }

            stats = pattern["collections_used"][collection]
            stats["total"] += 1

            if outcome == "worked":
                stats["successes"] += 1
            elif outcome == "failed":
                stats["failures"] += 1

            # Update best collection
            best_collection = collection
            best_rate = 0.0

            for coll_name, coll_stats in pattern["collections_used"].items():
                # Calculate success rate: successes / (successes + failures)
                # Excludes "partial" and "unknown" outcomes per v0.1.6 spec (architecture.md:648)
                total_with_feedback = coll_stats["successes"] + coll_stats["failures"]

                if total_with_feedback > 0:
                    rate = coll_stats["successes"] / total_with_feedback
                else:
                    rate = 0.5  # Neutral baseline (50%) for untested patterns

                if rate > best_rate:
                    best_rate = rate
                    best_collection = coll_name

            pattern["best_collection"] = best_collection
            # Default to 0.5 if no collections have been tested with explicit feedback
            pattern["success_rate"] = best_rate if best_rate > 0 else 0.5

        # Update relationship outcomes
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                rel_key = "|".join(sorted([concept1, concept2]))
                if rel_key in self.knowledge_graph.get("relationships", {}):
                    rel_data = self.knowledge_graph["relationships"][rel_key]
                    if outcome == "worked":
                        rel_data["success_together"] += 1
                    elif outcome == "failed":
                        rel_data["failure_together"] += 1

        # Save KG with proper await (debounced)
        await self._debounced_save_kg()

    async def analyze_conversation_context(
        self,
        current_message: str,
        recent_conversation: List[Dict[str, Any]],
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Analyze conversation context for organic memory injection.
        Returns relevant patterns, past outcomes, and contextual insights.
        """
        context = {
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        }

        try:
            # Extract concepts from current message
            current_concepts = self._extract_concepts(current_message)

            # 1. PATTERN RECOGNITION: Find similar past conversations
            if current_concepts:
                # Search patterns collection for similar concept combinations
                pattern_signature = "_".join(sorted(current_concepts[:3]))

                if pattern_signature in self.knowledge_graph.get("problem_categories", {}):
                    past_solutions = self.knowledge_graph["problem_categories"][pattern_signature]

                    for doc_id in past_solutions[:2]:  # Top 2 past solutions
                        # Get the actual document
                        for coll_name in ["patterns", "history"]:
                            if coll_name in self.collections:
                                doc = self.collections[coll_name].get_fragment(doc_id)
                                if doc:
                                    metadata = doc.get("metadata", {})
                                    score = metadata.get("score", 0.5)
                                    uses = metadata.get("uses", 0)
                                    last_outcome = metadata.get("last_outcome", "unknown")

                                    # Only include proven patterns (matched to promotion threshold)
                                    if score >= self.PROMOTION_SCORE_THRESHOLD and last_outcome == "worked":
                                        context["relevant_patterns"].append({
                                            "text": doc.get("content", "")[:200],
                                            "score": score,
                                            "uses": uses,
                                            "collection": coll_name,
                                            "insight": f"Based on {uses} past use(s), this approach had a {int(score*100)}% success rate"
                                        })
                                    break

            # 2. FAILURE AWARENESS: Check if similar attempts failed before
            failure_patterns = self.knowledge_graph.get("failure_patterns", {})
            for failure_key, failures in failure_patterns.items():
                # Check if current message relates to past failures
                if any(concept in failure_key.lower() for concept in current_concepts):
                    recent_failures = [f for f in failures if f.get("timestamp", "")][-2:]  # Last 2
                    for failure in recent_failures:
                        context["past_outcomes"].append({
                            "outcome": "failed",
                            "reason": failure_key,
                            "when": failure.get("timestamp", ""),
                            "insight": f"Note: Similar approach failed before due to: {failure_key[:80]}"
                        })

            # 3. TOPIC CONTINUITY: Check if switching topics or continuing
            if recent_conversation and len(recent_conversation) >= 2:
                # Get last user message
                last_messages = [msg for msg in recent_conversation[-3:] if msg.get("role") == "user"]
                if last_messages:
                    last_message = last_messages[-1].get("content", "")
                    last_concepts = self._extract_concepts(last_message)

                    # Check concept overlap
                    overlap = set(current_concepts) & set(last_concepts)
                    if overlap:
                        context["topic_continuity"].append({
                            "continuing": True,
                            "common_concepts": list(overlap),
                            "insight": f"Continuing discussion about: {', '.join(list(overlap)[:3])}"
                        })
                    else:
                        context["topic_continuity"].append({
                            "continuing": False,
                            "insight": "Topic shift detected - loading new context"
                        })

            # 4. PROACTIVE INSIGHTS: Check success rates for proposed approaches
            for concept in current_concepts[:3]:
                if concept in self.knowledge_graph.get("routing_patterns", {}):
                    pattern = self.knowledge_graph["routing_patterns"][concept]
                    success_rate = pattern.get("success_rate", 0)
                    best_collection = pattern.get("best_collection", "unknown")

                    if success_rate > 0.7:
                        context["proactive_insights"].append({
                            "concept": concept,
                            "success_rate": success_rate,
                            "recommendation": f"For '{concept}', check {best_collection} collection (historically {int(success_rate*100)}% effective)"
                        })

            # 5. REPETITION DETECTION: Check if user asked similar question recently
            if "working" in self.collections:
                working_items = await self.collections["working"].query_vectors(
                    query_vector=await self.embedding_service.embed_text(current_message),
                    top_k=3
                )

                similar_recent = []
                for item in working_items:
                    metadata = item.get("metadata", {})
                    if metadata.get("conversation_id") == conversation_id:
                        # Check if it's recent (within last 10 messages)
                        similarity = 1.0 / (1.0 + item.get("distance", 1.0))
                        if similarity > 0.85:  # Very similar
                            similar_recent.append({
                                "text": item.get("content", "")[:150],
                                "similarity": similarity,
                                "insight": f"You mentioned something similar recently (message similarity: {int(similarity*100)}%)"
                            })

                if similar_recent:
                    context["proactive_insights"].extend(similar_recent[:1])  # Just show most similar

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")

        return context

    async def _find_known_solutions(self, query: str) -> List[Dict[str, Any]]:
        """Find known solutions for similar problems"""
        try:
            if not query:
                return []

            # Extract concepts from the query
            query_concepts = self._extract_concepts(query)
            query_signature = "_".join(sorted(query_concepts[:5]))

            known_solutions = []

            # Ensure problem_solutions exists in knowledge graph
            if "problem_solutions" not in self.knowledge_graph:
                self.knowledge_graph["problem_solutions"] = {}

            # Look for exact problem matches
            if query_signature in self.knowledge_graph["problem_solutions"]:
                solutions = self.knowledge_graph["problem_solutions"][query_signature]

                # Sort by success count and recency
                sorted_solutions = sorted(
                    solutions,
                    key=lambda x: (x.get("success_count", 0), x.get("last_used", "")),
                    reverse=True
                )

                # Add top solutions to results
                for solution in sorted_solutions[:3]:
                    doc_id = solution.get("doc_id")
                    if doc_id:
                        # Try to find the actual document
                        for coll_name, adapter in self.collections.items():
                            if doc_id.startswith(coll_name):
                                doc = adapter.get_fragment(doc_id)
                                if doc:
                                    # Boost the score for known solutions
                                    doc["distance"] = doc.get("distance", 1.0) * 0.5  # 50% boost
                                    doc["is_known_solution"] = True
                                    doc["solution_success_count"] = solution.get("success_count", 0)
                                    known_solutions.append(doc)
                                    logger.info(f"Found known solution: {doc_id} (used {solution['success_count']} times)")
                                    break

            # Also check for partial matches (3+ concept overlap)
            for problem_sig, solutions in self.knowledge_graph["problem_solutions"].items():
                if problem_sig != query_signature:
                    problem_concepts_stored = set(problem_sig.split("_"))
                    overlap = len(set(query_concepts) & problem_concepts_stored)

                    if overlap >= 3:  # Significant overlap
                        for solution in solutions[:1]:  # Take best from partial matches
                            doc_id = solution.get("doc_id")
                            if doc_id and doc_id not in [s.get("id") for s in known_solutions]:
                                for coll_name, adapter in self.collections.items():
                                    if doc_id.startswith(coll_name):
                                        doc = adapter.get_fragment(doc_id)
                                        if doc:
                                            doc["distance"] = doc.get("distance", 1.0) * 0.7  # 30% boost
                                            doc["is_partial_solution"] = True
                                            doc["concept_overlap"] = overlap
                                            known_solutions.append(doc)
                                            break

            return known_solutions

        except Exception as e:
            logger.error(f"Error finding known solutions: {e}")
            return []

    async def _track_problem_solution(self, doc_id: str, metadata: Dict[str, Any], context: Optional[Dict[str, Any]]):
        """Track successful problem→solution patterns for future reuse"""
        try:
            # Extract problem signature from the original query/context
            problem_text = metadata.get("original_context", "") or metadata.get("query", "")
            solution_text = metadata.get("text", "")

            if not problem_text or not solution_text:
                return

            # Create problem signature (simplified concepts)
            problem_concepts = self._extract_concepts(problem_text)
            problem_signature = "_".join(sorted(problem_concepts[:5]))  # Top 5 concepts

            if not problem_signature:
                return

            # Store problem→solution mapping
            if problem_signature not in self.knowledge_graph["problem_solutions"]:
                self.knowledge_graph["problem_solutions"][problem_signature] = []

            solution_entry = {
                "doc_id": doc_id,
                "solution": solution_text[:500],  # Store abbreviated solution
                "success_count": 1,
                "last_used": datetime.now().isoformat(),
                "contexts": [context] if context else []
            }

            # Check if this solution already exists for this problem
            existing_solutions = self.knowledge_graph["problem_solutions"][problem_signature]
            solution_found = False

            for existing in existing_solutions:
                if existing["doc_id"] == doc_id:
                    # Update existing solution
                    existing["success_count"] += 1
                    existing["last_used"] = datetime.now().isoformat()
                    if context and context not in existing.get("contexts", []):
                        existing.setdefault("contexts", []).append(context)
                    solution_found = True
                    break

            if not solution_found:
                existing_solutions.append(solution_entry)

            # Track solution patterns (for pattern matching)
            pattern_hash = f"{problem_signature}::{doc_id}"
            if pattern_hash not in self.knowledge_graph["solution_patterns"]:
                self.knowledge_graph["solution_patterns"][pattern_hash] = {
                    "problem": problem_text[:200],
                    "solution": solution_text[:200],
                    "success_count": 0,
                    "failure_count": 0,
                    "contexts": []
                }

            pattern = self.knowledge_graph["solution_patterns"][pattern_hash]
            pattern["success_count"] += 1
            pattern["success_rate"] = pattern["success_count"] / (pattern["success_count"] + pattern["failure_count"])

            # Save updated KG with proper await (debounced)
            await self._debounced_save_kg()

            logger.info(f"Tracked problem→solution pattern: {problem_signature[:30]}... -> {doc_id}")

        except Exception as e:
            logger.error(f"Error tracking problem→solution: {e}")

    # Removed _background_promotion_loop - Background promotion is handled in main.py
    # This eliminates duplicate background task tech debt (was running both 30min and 60min tasks)

    async def cleanup_old_working_memory(self):
        """Clean up working memory items older than 24 hours"""
        try:
            working_adapter = self.collections.get("working")
            if not working_adapter:
                return 0

            cleaned_count = 0
            all_ids = working_adapter.list_all_ids()

            for doc_id in all_ids:
                doc = working_adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    timestamp_str = metadata.get("timestamp", "")

                    # Calculate age
                    try:
                        if timestamp_str:
                            doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            age_hours = (datetime.now() - doc_time).total_seconds() / 3600

                            if age_hours > 24:
                                working_adapter.delete_vectors([doc_id])
                                cleaned_count += 1
                                logger.info(f"Cleaned up old working memory {doc_id} (age: {age_hours:.1f}h)")
                    except Exception as e:
                        logger.warning(f"Could not parse timestamp for {doc_id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old working memory items")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up working memory: {e}")
            return 0

    def _handle_promotion_error(self, task: asyncio.Task):
        """Handle errors from async promotion tasks"""
        try:
            task.result()  # Will raise if task failed
        except Exception as e:
            logger.error(f"Auto-promotion task failed: {e}", exc_info=True)

    async def _promote_valuable_working_memory(self, conversation_id: Optional[str] = None):
        """
        Promote valuable working memory to history collection.

        Args:
            conversation_id: Optional filter to only promote memories from specific conversation
        """
        try:
            working_adapter = self.collections.get("working")
            if not working_adapter:
                return

            promoted_count = 0
            checked_count = 0

            # Get all working memory items
            all_ids = working_adapter.list_all_ids()
            for doc_id in all_ids:
                doc = working_adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})

                    # Note: Cross-conversation promotion is ALLOWED (working memory is global)
                    # Valuable memories from any conversation should promote to history/patterns
                    # The conversation_id is preserved in metadata for traceability

                    checked_count += 1

                    # Get text for promotion evaluation
                    text = metadata.get("text", "")

                    # Check promotion criteria
                    score = metadata.get("score", 0.5)
                    uses = metadata.get("uses", 0)
                    timestamp_str = metadata.get("timestamp", "")

                    # Calculate age
                    try:
                        if timestamp_str:
                            doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            age_hours = (datetime.now() - doc_time).total_seconds() / 3600
                        else:
                            age_hours = 0
                    except:
                        age_hours = 0

                    # Promote if: high score AND (used multiple times OR old enough)
                    if score >= self.PROMOTION_SCORE_THRESHOLD and (uses >= 2 or age_hours >= 2):
                        # Promote to History
                        new_id = doc_id.replace("working_", "history_")
                        await self.collections["history"].upsert_vectors(
                            ids=[new_id],
                            vectors=[await self.embedding_service.embed_text(text)],
                            metadatas=[{
                                **metadata,
                                "promoted_from": "working",
                                "promotion_time": datetime.now().isoformat(),
                                "promotion_reason": "hourly_check"
                            }]
                        )

                        # Delete from working
                        working_adapter.delete_vectors([doc_id])
                        promoted_count += 1
                        logger.info(f"Promoted {doc_id} to history (score: {score:.2f}, uses: {uses}, age: {age_hours:.1f}h)")

                    # CLEANUP: Remove items older than 24 hours that weren't promoted
                    elif age_hours > 24:
                        # Delete old items that didn't make the cut
                        working_adapter.delete_vectors([doc_id])
                        logger.info(f"Cleaned up old working memory {doc_id} (age: {age_hours:.1f}h, score: {score:.2f})")

            if promoted_count > 0:
                logger.info(f"Hourly promotion: checked {checked_count}, promoted {promoted_count} memories")

        except Exception as e:
            logger.error(f"Error in hourly promotion: {e}")

    async def _check_implicit_outcomes(self, last_response: Dict[str, Any], new_user_input: str):
        """Detect implicit outcomes from user behavior patterns"""
        if not last_response or not new_user_input:
            return

        doc_id = last_response.get('doc_id')
        if not doc_id:
            return

        # Time since last response
        time_diff = (datetime.now() - last_response['timestamp']).total_seconds()

        # Implicit success signals
        outcome = None
        confidence = 0.5
        reason = None

        # Pattern 1: Enhanced positive outcome detection
        positive_words = [
            # Original patterns
            'thanks', 'perfect', 'great', 'works', 'working',
            # Common success expressions
            'awesome', 'excellent', 'brilliant', 'amazing', 'fantastic',
            # Confirmation phrases
            'that worked', 'it works', 'fixed it', 'solved it', 'got it',
            # Relief expressions
            'finally', 'exactly what i needed', 'thats it', "that's it",
            # Completion indicators
            'done', 'sorted', 'resolved', 'success', 'successful',
            # Gratitude variations
            'thank you', 'ty', 'thx', 'much appreciated', 'helpful',
            # Problem resolution
            'no more errors', 'working now', 'all good', 'all set'
        ]

        # Context patterns that indicate progression (success implied)
        success_context = [
            # Implementation success
            'now it', 'now i can', 'able to', 'managed to',
            # Testing success
            'tests pass', 'builds successfully', 'no errors', 'compiles',
            # User moving on
            'next question', 'another issue', 'different problem',
            # Building on solution
            'what about', 'now how', 'next step',
            'also need to', 'additionally', 'can i also'
        ]

        user_lower = new_user_input.lower()
        if (any(word in user_lower for word in positive_words) or
            any(phrase in user_lower for phrase in success_context)):
            outcome = "worked"
            confidence = 0.8
            reason = "positive_feedback"

        # Pattern 2: Topic change after solution (>30s delay + completely different content)
        elif time_diff > 30:
            prev_concepts = set(self._extract_concepts(last_response.get('user_input', '')))
            new_concepts = set(self._extract_concepts(new_user_input))
            overlap = len(prev_concepts & new_concepts) / max(len(prev_concepts), 1)

            if overlap < 0.2:  # Less than 20% concept overlap
                outcome = "worked"
                confidence = 0.7
                reason = "topic_change_after_delay"

        # Pattern 3: Continued error discussion (same error mentioned)
        elif 'error' in new_user_input.lower() and 'error' in last_response.get('content', '').lower():
            # Extract error patterns
            import re
            prev_errors = re.findall(r'(?:error|exception):\s*([^\n]+)', last_response.get('content', ''), re.IGNORECASE)
            new_errors = re.findall(r'(?:error|exception):\s*([^\n]+)', new_user_input, re.IGNORECASE)

            if prev_errors and new_errors and any(pe in ne or ne in pe for pe in prev_errors for ne in new_errors):
                outcome = "failed"
                confidence = 0.8
                reason = "same_error_persists"

        # Pattern 4: Quick follow-up question (< 10s, similar topic)
        elif time_diff < 10:
            prev_concepts = set(self._extract_concepts(last_response.get('user_input', '')))
            new_concepts = set(self._extract_concepts(new_user_input))
            overlap = len(prev_concepts & new_concepts) / max(len(prev_concepts), 1)

            if overlap > 0.6:  # High concept overlap
                outcome = "partial"
                confidence = 0.6
                reason = "quick_followup"

        # Pattern 5: "Still not working" or similar negative feedback
        elif any(phrase in new_user_input.lower() for phrase in ['still not', 'doesn\'t work', 'not working', 'same error', 'still getting']):
            outcome = "failed"
            confidence = 0.9
            reason = "explicit_failure"

        # Record the implicit outcome if detected (no confidence gate - trust the detection)
        if outcome:
            logger.info(f"Implicit outcome detected: {outcome} (confidence: {confidence}, reason: {reason})")
            await self.record_outcome(
                doc_id=doc_id,
                outcome=outcome,
                failure_reason=reason if outcome == "failed" else None,
                context={"implicit": True, "confidence": confidence, "reason": reason}
            )


    async def clear_old_history(self, days: int = 30):
        """Clear History items older than specified days (unless high value)"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days)

            history_adapter = self.collections.get("history")
            if not history_adapter:
                return

            deleted_count = 0
            preserved_count = 0

            all_ids = history_adapter.list_all_ids()
            for doc_id in all_ids:
                doc = history_adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    timestamp_str = metadata.get("timestamp", "")
                    score = metadata.get("score", 0.5)

                    # Parse timestamp
                    try:
                        if timestamp_str:
                            doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        else:
                            # No timestamp, consider it old
                            doc_time = datetime.now() - timedelta(days=days+1)
                    except:
                        doc_time = datetime.now() - timedelta(days=days+1)

                    # Delete if older than cutoff AND not high value
                    if doc_time < cutoff_time:
                        if score < self.HIGH_VALUE_THRESHOLD:  # Only delete if not high-value
                            await history_adapter.delete_vectors([doc_id])
                            deleted_count += 1
                            logger.info(f"Deleted old history item {doc_id} (age>{days}d, score={score:.2f})")
                        else:
                            preserved_count += 1
                            logger.info(f"Preserved valuable history {doc_id} (score={score:.2f})")

            logger.info(f"History cleanup: deleted {deleted_count}, preserved {preserved_count} high-value items")
            return deleted_count, preserved_count

        except Exception as e:
            logger.error(f"Error clearing old history: {e}")
            return 0, 0

    async def clear_old_working_memory(self, hours: int = 24):
        """Clear Working memory older than specified hours and promote valuable content"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)

            working_adapter = self.collections.get("working")
            if not working_adapter:
                return

            promoted_count = 0
            deleted_count = 0

            all_ids = working_adapter.list_all_ids()
            for doc_id in all_ids:
                doc = working_adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    timestamp_str = metadata.get("timestamp", "")

                    # Parse timestamp
                    try:
                        if timestamp_str:
                            doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        else:
                            # No timestamp, consider it old
                            doc_time = datetime.now() - timedelta(hours=hours+1)
                    except:
                        doc_time = datetime.now() - timedelta(hours=hours+1)

                    # If older than cutoff
                    if doc_time < cutoff_time:
                        score = metadata.get("score", 0.5)
                        uses = metadata.get("uses", 0)
                        text = metadata.get("text", "")

                        # Promote if valuable (high score and used multiple times)
                        if score >= 0.7 and uses >= 2 and text:
                            # Promote to History
                            new_id = doc_id.replace("working_", "history_")
                            await self.collections["history"].upsert_vectors(
                                ids=[new_id],
                                vectors=[doc.get("embedding")],
                                metadatas=[{
                                    **metadata,
                                    "promoted_from": "working",
                                    "promotion_time": datetime.now().isoformat()
                                }]
                            )
                            promoted_count += 1

                        # Delete from working
                        working_adapter.delete_vectors([doc_id])
                        deleted_count += 1

            logger.info(f"Cleared {deleted_count} old working memories, promoted {promoted_count} valuable ones")

        except Exception as e:
            logger.error(f"Error clearing old working memory: {e}")

    async def load_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load conversation history from file"""
        try:
            return await self.file_adapter.load_conversation_history(
                session_id=conversation_id,
                limit=limit
            )
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            return []

    async def clear_session(self, preserve_important: bool = True, promote_valuable: bool = True):
        """Switch to a new conversation ID without clearing Working memory
        Working memory only clears based on 24-hour retention policy"""
        # Just update conversation ID, don't clear Working memory
        # Working memory persists across conversations and only clears after 24 hours

        # New conversation ID
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Switched to new conversation: {self.conversation_id} (Working memory persists)")

    async def ingest_book(self, file_path: str, title: str) -> int:
        """Ingest a book/document into the books collection (DISABLED - removed in RoamPal)"""
        logger.info("Book ingestion available via book upload API")
        return 0

    async def get_recent_conversation(self, conversation_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation turns from working memory"""
        if not conversation_id:
            conversation_id = self.conversation_id

        if "working" not in self.collections:
            return []

        # Get all working memory for this conversation
        conversation = []
        for doc_id in self.collections["working"].list_all_ids():
            doc = self.collections["working"].get_fragment(doc_id)
            if doc and doc.get("metadata", {}).get("conversation_id") == conversation_id:
                metadata = doc.get("metadata", {})
                conversation.append({
                    "text": metadata.get("text", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "role": metadata.get("role", "")
                })

        # Sort by timestamp
        conversation.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")

        # Return last N turns
        return conversation[-limit:] if len(conversation) > limit else conversation

    async def get_working_context(self, limit: int = 5) -> Dict[str, Any]:
        """Get current working memory context summary"""
        context = {
            "conversation_id": self.conversation_id,
            "recent_turns": [],
            "current_task": None,
            "mentioned_errors": [],
            "code_snippets": 0
        }

        if "working" not in self.collections:
            return context

        # Get recent working memory
        working_docs = []
        for doc_id in self.collections["working"].list_all_ids()[-limit*2:]:
            doc = self.collections["working"].get_fragment(doc_id)
            if doc:
                working_docs.append(doc)

        # Analyze context
        for doc in working_docs:
            metadata = doc.get("metadata", {})
            text = metadata.get("text", "")

            # Track conversation
            if metadata.get("role"):
                context["recent_turns"].append({
                    "role": metadata["role"],
                    "text": text[:100],
                    "task_type": metadata.get("task_type")
                })

            # Track task type
            if metadata.get("task_type"):
                context["current_task"] = metadata["task_type"]

            # Track errors mentioned
            if "error" in text.lower() or "exception" in text.lower():
                import re
                errors = re.findall(r'(?:error|exception):\s*([^\n]+)', text, re.IGNORECASE)
                context["mentioned_errors"].extend(errors[:2])

            # Count code
            if metadata.get("contains_code"):
                context["code_snippets"] += 1

        return context

    async def save_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a conversation turn to working memory with proper metadata for context retrieval.
        """
        conversation_id = conversation_id or self.conversation_id

        # Store user message
        await self.store(
            text=user_input,
            collection="working",
            metadata={
                "role": "user",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )

        # Store assistant response
        await self.store(
            text=assistant_response,
            collection="working",
            metadata={
                "role": "assistant",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )

        # Also save to file adapter for persistence
        await self.file_adapter.save_conversation_turn(
            session_id=conversation_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata
        )

    # Removed duplicate method - using the one defined at line 1230

    async def detect_conversation_outcome(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect outcome from conversation using OutcomeDetector if enabled.

        Args:
            conversation: List of conversation turns

        Returns:
            Detection results with outcome, confidence, indicators
        """
        if not self.outcome_detector:
            return {
                "outcome": "unknown",
                "confidence": 0.0,
                "indicators": [],
                "detector_enabled": False
            }

        # Use outcome detector
        result = await self.outcome_detector.analyze(conversation)
        result["detector_enabled"] = True

        # Track problem-solution patterns (no confidence gate - trust the detector)
        if result["outcome"] == "worked":
            # Extract problem and solution from conversation
            user_messages = [m for m in conversation if m.get("role") == "user"]
            assistant_messages = [m for m in conversation if m.get("role") == "assistant"]

            if user_messages and assistant_messages:
                problem = user_messages[0].get("content", "")
                solution = assistant_messages[-1].get("content", "")

                # Use existing problem-solution tracking
                if problem and solution:
                    await self._track_problem_solution(
                        doc_id=f"detected_{uuid.uuid4().hex[:8]}",
                        metadata={"text": solution, "query": problem},
                        context={"detection": result}
                    )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "conversation_id": self.conversation_id,
            "collections": {},
            "kg_patterns": len(self.knowledge_graph.get("routing_patterns", {})),
            "knowledge_graph": {
                "routing_patterns": len(self.knowledge_graph.get("routing_patterns", {})),
                "failure_patterns": len(self.knowledge_graph.get("failure_patterns", {})),
                "problem_categories": len(self.knowledge_graph.get("problem_categories", {})),
                "problem_solutions": len(self.knowledge_graph.get("problem_solutions", {})),
                "solution_patterns": len(self.knowledge_graph.get("solution_patterns", {}))
            },
            "relationships": {
                "related": len(self.relationships.get("related", {})),
                "evolution": len(self.relationships.get("evolution", {})),
                "conflicts": len(self.relationships.get("conflicts", {}))
            }
        }

        for name, adapter in self.collections.items():
            try:
                stats["collections"][name] = adapter.collection.count()
            except:
                stats["collections"][name] = 0

        return stats

    async def _add_relationship(self, doc_id: str, rel_type: str, rel_data: Any):
        """Add a relationship for a document"""
        if rel_type not in self.relationships:
            self.relationships[rel_type] = {}
        self.relationships[rel_type][doc_id] = rel_data
        # Save relationships with proper await
        await self._save_relationships()

    def _get_related_docs(self, doc_id: str) -> List[str]:
        """Get all related document IDs"""
        related = set()

        # Direct relationships
        if doc_id in self.relationships.get("related", {}):
            related.update(self.relationships["related"][doc_id])

        # Evolution chain
        if doc_id in self.relationships.get("evolution", {}):
            evo = self.relationships["evolution"][doc_id]
            if "parent" in evo:
                related.add(evo["parent"])
            if "children" in evo:
                related.update(evo["children"])

        return list(related)

    async def mark_persistent(self, doc_id: str):
        """Mark a document to persist across sessions"""
        for collection in self.collections.values():
            doc = collection.get_fragment(doc_id)
            if doc:
                metadata = doc.get("metadata", {})
                metadata["persist_session"] = True
                collection.update_fragment_metadata(doc_id, metadata)
                logger.info(f"Marked {doc_id} as persistent")
                return True
        return False

    async def export_backup(self) -> Dict[str, Any]:
        """Export memory system state for backup"""
        backup = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "conversation_id": self.conversation_id,
            "knowledge_graph": self.knowledge_graph,
            "relationships": self.relationships,
            "stats": self.get_stats()
        }

        # Export collection metadata (not the actual vectors - too large)
        backup["collections"] = {}
        for name, adapter in self.collections.items():
            try:
                count = adapter.collection.count()
                backup["collections"][name] = {"count": count}
            except:
                backup["collections"][name] = {"count": 0}

        return backup

    async def import_backup(self, backup_data: Dict[str, Any]) -> bool:
        """Import memory system state from backup"""
        try:
            # Restore knowledge graph
            if "knowledge_graph" in backup_data:
                self.knowledge_graph = backup_data["knowledge_graph"]
                await self._save_kg()  # Immediate save for restore operation

            # Restore relationships
            if "relationships" in backup_data:
                self.relationships = backup_data["relationships"]
                await self._save_relationships()

            logger.info(f"Restored backup from {backup_data.get('timestamp', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return False

    async def _startup_cleanup(self):
        """Clean up stale data from previous sessions"""
        try:
            logger.info("Running startup cleanup...")

            # Clean working memory older than 24h
            await self.cleanup_old_working_memory()

            # Clean history older than 30 days
            await self.clear_old_history(days=30)

            # Clean dead KG references
            await self._cleanup_kg_dead_references()

            logger.info("Startup cleanup complete")
        except Exception as e:
            logger.error(f"Error in startup cleanup: {e}")

    async def _cleanup_kg_dead_references(self):
        """Remove doc_id references that no longer exist in collections"""
        try:
            cleaned = 0

            # Clean problem_categories
            for problem_key, doc_ids in list(self.knowledge_graph["problem_categories"].items()):
                valid_ids = [doc_id for doc_id in doc_ids if self._doc_exists(doc_id)]
                if len(valid_ids) < len(doc_ids):
                    cleaned += len(doc_ids) - len(valid_ids)
                    if valid_ids:
                        self.knowledge_graph["problem_categories"][problem_key] = valid_ids
                    else:
                        del self.knowledge_graph["problem_categories"][problem_key]

            # Clean problem_solutions
            for problem_sig, solutions in list(self.knowledge_graph["problem_solutions"].items()):
                valid_solutions = [s for s in solutions if self._doc_exists(s.get("doc_id"))]
                if len(valid_solutions) < len(solutions):
                    cleaned += len(solutions) - len(valid_solutions)
                    if valid_solutions:
                        self.knowledge_graph["problem_solutions"][problem_sig] = valid_solutions
                    else:
                        del self.knowledge_graph["problem_solutions"][problem_sig]

            # Clean routing_patterns (remove patterns with 0 total uses)
            for concept, pattern in list(self.knowledge_graph["routing_patterns"].items()):
                collections_used = pattern.get("collections_used", {})
                total_uses = sum(stats.get("total", 0) for stats in collections_used.values())
                if total_uses == 0:
                    del self.knowledge_graph["routing_patterns"][concept]
                    cleaned += 1

            if cleaned > 0:
                logger.info(f"KG cleanup: removed {cleaned} dead references")
                await self._save_kg()  # Immediate save for cleanup operation

            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning KG dead references: {e}")
            return 0

    def _doc_exists(self, doc_id: str) -> bool:
        """Check if document exists in any collection"""
        if not doc_id:
            return False

        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                try:
                    doc = adapter.get_fragment(doc_id)
                    return doc is not None
                except:
                    return False
        return False

    # ============================================
    # MEMORY BANK OPERATIONS (5th Collection)
    # ============================================

    async def store_memory_bank(
        self,
        text: str,
        tags: List[str],
        importance: float = 0.7,
        confidence: float = 0.7
    ) -> str:
        """
        Store user memory in memory_bank collection.
        LLM has full autonomy to create/manage memories.

        Args:
            text: Memory content
            tags: List of tags (soft guidelines: identity, preference, project, context, goal)
            importance: 0.0-1.0 (how critical is this memory)
            confidence: 0.0-1.0 (how sure are we about this)

        Returns:
            Document ID
        """
        # CAPACITY CHECK - prevent unbounded growth
        MAX_MEMORY_BANK_ITEMS = 500  # Reasonable limit for single-user system
        try:
            current_count = self.collections["memory_bank"].collection.count()
            if current_count >= MAX_MEMORY_BANK_ITEMS:
                error_msg = (
                    f"Memory bank at capacity ({current_count}/{MAX_MEMORY_BANK_ITEMS}). "
                    "Please archive or delete old memories in Settings > Memory Bank."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        except ValueError:
            # Re-raise capacity errors
            raise
        except Exception as e:
            # Ignore count check errors, continue with storage
            logger.warning(f"Could not check memory_bank capacity (continuing anyway): {e}")

        doc_id = f"memory_bank_{uuid.uuid4().hex[:8]}"

        await self.store(
            text=text,
            collection="memory_bank",
            metadata={
                "tags": json.dumps(tags),
                "importance": importance,
                "confidence": confidence,
                "score": 1.0,  # Never decays
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "mentioned_count": 1,
                "last_mentioned": datetime.now().isoformat()
            }
        )

        # CRITICAL: Extract entities from memory_bank content for Content KG
        # This builds the user's personal knowledge graph (green/purple nodes in UI)
        # Do NOT remove - core feature for entity relationship mapping
        try:
            timestamp = datetime.now().isoformat()
            self.content_graph.add_entities_from_text(text, doc_id, timestamp)
            # Save content graph asynchronously (debounced like routing KG)
            await self._debounced_save_kg()  # This now saves both KGs
            logger.debug(f"Extracted entities from memory_bank item: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to extract entities for content KG: {e}", exc_info=True)

        logger.info(f"Stored memory_bank item: {text[:50]}... (tags: {tags})")
        return doc_id

    async def update_memory_bank(
        self,
        doc_id: str,
        new_text: str,
        reason: str = "llm_update"
    ) -> str:
        """
        Update memory with auto-archiving of old version.
        LLM can freely update memories, old versions preserved automatically.

        Args:
            doc_id: Memory to update
            new_text: New content
            reason: Why updating (for audit trail)

        Returns:
            Document ID
        """
        # Get current version
        old_doc = self.collections["memory_bank"].get_fragment(doc_id)
        if not old_doc:
            logger.warning(f"Memory {doc_id} not found, creating new")
            return await self.store_memory_bank(new_text, tags=["updated"])

        # Auto-archive old version (with timestamp to prevent collisions)
        archive_id = f"{doc_id}_archived_{int(datetime.now().timestamp())}"
        # Re-embed old text for archiving (get_fragment doesn't return embeddings)
        old_text = old_doc.get("metadata", {}).get("content", old_doc.get("metadata", {}).get("text", ""))
        old_embedding = await self.embedding_service.embed_text(old_text)
        await self.collections["memory_bank"].upsert_vectors(
            ids=[archive_id],
            vectors=[old_embedding],
            metadatas=[{
                **old_doc.get("metadata", {}),
                "status": "archived",
                "original_id": doc_id,
                "archive_reason": reason,
                "archived_at": datetime.now().isoformat()
            }]
        )

        # Update in-place (overwrite)
        new_embedding = await self.embedding_service.embed_text(new_text)
        old_metadata = old_doc.get("metadata", {})

        await self.collections["memory_bank"].upsert_vectors(
            ids=[doc_id],
            vectors=[new_embedding],
            metadatas=[{
                **old_metadata,
                "text": new_text,
                "content": new_text,
                "updated_at": datetime.now().isoformat(),
                "update_reason": reason
            }]
        )

        # CRITICAL: Update content graph with new entities
        # Remove old entities, extract new ones
        try:
            self.content_graph.remove_entity_mention(doc_id)
            timestamp = datetime.now().isoformat()
            self.content_graph.add_entities_from_text(new_text, doc_id, timestamp)
            await self._debounced_save_kg()
            logger.debug(f"Updated content KG for memory_bank item: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to update content KG: {e}", exc_info=True)

        logger.info(f"Updated memory_bank item {doc_id}: {reason}")
        return doc_id

    async def archive_memory_bank(
        self,
        doc_id: str,
        reason: str = "llm_decision"
    ) -> bool:
        """
        Archive memory (soft delete).
        LLM can archive outdated/irrelevant memories.

        Args:
            doc_id: Memory to archive
            reason: Why archiving

        Returns:
            Success status
        """
        doc = self.collections["memory_bank"].get_fragment(doc_id)
        if not doc:
            return False

        metadata = doc.get("metadata", {})
        metadata["status"] = "archived"
        metadata["archive_reason"] = reason
        metadata["archived_at"] = datetime.now().isoformat()

        self.collections["memory_bank"].update_fragment_metadata(doc_id, metadata)

        # CRITICAL: Remove entity mentions from content graph when archived
        try:
            self.content_graph.remove_entity_mention(doc_id)
            await self._debounced_save_kg()
            logger.debug(f"Removed content KG entities for archived memory: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to update content KG on archive: {e}", exc_info=True)

        logger.info(f"Archived memory_bank item {doc_id}: {reason}")
        return True

    async def search_memory_bank(
        self,
        query: str = None,
        tags: List[str] = None,
        include_archived: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search memory_bank collection with filtering.

        Args:
            query: Semantic search query (None = get all)
            tags: Filter by tags
            include_archived: Include archived memories
            limit: Max results

        Returns:
            List of memories
        """
        # Search memory_bank collection
        if query:
            results = await self.search(
                query=query,
                collections=["memory_bank"],
                limit=limit * 2  # Get extra for filtering
            )
        else:
            # Get all items
            results = await self.search(
                query="",
                collections=["memory_bank"],
                limit=limit * 2
            )

        # Filter by status and tags
        filtered = []
        for r in results:
            metadata = r.get("metadata", {})
            status = metadata.get("status", "active")

            # Skip archived unless requested
            if status == "archived" and not include_archived:
                continue

            # Filter by tags if specified
            if tags:
                doc_tags = json.loads(metadata.get("tags", "[]"))
                if not any(tag in doc_tags for tag in tags):
                    continue

            filtered.append(r)

        return filtered[:limit]

    async def user_restore_memory(self, doc_id: str) -> bool:
        """
        User manually restores archived memory.
        Called from Settings UI.

        Args:
            doc_id: Memory to restore

        Returns:
            Success status
        """
        doc = self.collections["memory_bank"].get_fragment(doc_id)
        if not doc:
            return False

        metadata = doc.get("metadata", {})
        metadata["status"] = "active"
        metadata["restored_at"] = datetime.now().isoformat()
        metadata["restored_by"] = "user"

        self.collections["memory_bank"].update_fragment_metadata(doc_id, metadata)
        logger.info(f"User restored memory: {doc_id}")
        return True

    async def user_delete_memory(self, doc_id: str) -> bool:
        """
        User permanently deletes memory (hard delete).
        Called from Settings UI.

        Args:
            doc_id: Memory to delete

        Returns:
            Success status
        """
        try:
            self.collections["memory_bank"].delete_vectors([doc_id])
            logger.info(f"User permanently deleted memory: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {doc_id}: {e}")
            return False

    async def get_kg_entities(self, filter_text: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get entities from DUAL knowledge graph (Routing KG + Content KG merged).

        CRITICAL: This merges both graphs to provide complete entity view.
        - Routing KG: Query patterns → collection routing
        - Content KG: Entity relationships from memory_bank content
        - Entities in both graphs get source="both" (purple nodes in UI)

        NOTE: Reloads KG from disk to pick up changes from MCP process.
        """
        # Reload KG from disk to pick up changes from MCP process (different process writes to same file)
        self.knowledge_graph = self._load_kg()

        entities_map = {}  # entity_name -> entity_data

        # STEP 1: Get routing KG entities (query-based patterns)
        for concept, pattern in self.knowledge_graph.get("routing_patterns", {}).items():
            if filter_text and filter_text.lower() not in concept.lower():
                continue

            # Count routing connections
            routing_connections = 0
            for rel_key in self.knowledge_graph.get("relationships", {}).keys():
                if concept in rel_key.split("|"):
                    routing_connections += 1

            # Get total usage across all collections
            collections_used = pattern.get("collections_used", {})
            total_usage = sum(c.get("total", 0) for c in collections_used.values())

            entities_map[concept] = {
                "entity": concept,
                "source": "routing",  # Will be updated if also in content KG
                "routing_connections": routing_connections,
                "content_connections": 0,
                "total_connections": routing_connections,
                "success_rate": pattern.get("success_rate", 0.5),
                "best_collection": pattern.get("best_collection"),
                "collections_used": collections_used,
                "usage_count": total_usage,
                "mentions": 0,  # From content KG
                "last_used": pattern.get("last_used"),
                "created_at": pattern.get("created_at")
            }

        # STEP 2: Get content KG entities (memory-based relationships)
        # CRITICAL: Do not skip this step - provides green/purple nodes in UI
        content_entities = self.content_graph.get_all_entities(min_mentions=1)
        for entity_data in content_entities:
            entity_name = entity_data["entity"]

            if filter_text and filter_text.lower() not in entity_name.lower():
                continue

            # Count content connections
            content_rels = self.content_graph.get_entity_relationships(entity_name, min_strength=0.0)
            content_connections = len(content_rels)

            if entity_name in entities_map:
                # Entity exists in BOTH graphs → source="both" (purple node)
                entities_map[entity_name]["source"] = "both"
                entities_map[entity_name]["content_connections"] = content_connections
                entities_map[entity_name]["total_connections"] += content_connections
                entities_map[entity_name]["mentions"] = entity_data["mentions"]
            else:
                # Entity only in content KG → source="content" (green node)
                entities_map[entity_name] = {
                    "entity": entity_name,
                    "source": "content",  # Content KG only
                    "routing_connections": 0,
                    "content_connections": content_connections,
                    "total_connections": content_connections,
                    "success_rate": 0.5,  # Neutral (no routing data)
                    "best_collection": "memory_bank",  # Content entities are from memory_bank
                    "collections_used": {"memory_bank": {"total": entity_data["mentions"]}},
                    "usage_count": entity_data["mentions"],
                    "mentions": entity_data["mentions"],
                    "last_used": entity_data.get("last_seen"),
                    "created_at": entity_data.get("first_seen")
                }

        # Convert to list and sort by usage
        entities = list(entities_map.values())
        entities.sort(key=lambda x: x["usage_count"], reverse=True)
        return entities[:limit]

    async def get_kg_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity (DUAL KG merged).

        CRITICAL: Merges routing + content relationships for complete view.
        """
        relationships_map = {}  # related_entity -> relationship_data

        # STEP 1: Get routing KG relationships
        for rel_key, rel_data in self.knowledge_graph.get("relationships", {}).items():
            concepts = rel_key.split("|")
            if entity in concepts:
                related = concepts[1] if concepts[0] == entity else concepts[0]
                relationships_map[related] = {
                    "related_entity": related,
                    "source": "routing",  # Will update if also in content
                    "strength": rel_data.get("co_occurrence", 0),
                    "total_strength": rel_data.get("co_occurrence", 0),
                    "success_together": rel_data.get("success_together", 0),
                    "failure_together": rel_data.get("failure_together", 0),
                    "content_strength": 0  # From content KG
                }

        # STEP 2: Get content KG relationships
        # CRITICAL: Do not skip - provides entity relationship visualization
        content_rels = self.content_graph.get_entity_relationships(entity, min_strength=0.0)
        for rel_data in content_rels:
            related = rel_data["related_entity"]  # Fixed: was "entity", should be "related_entity"
            content_strength = rel_data["strength"]

            if related in relationships_map:
                # Relationship exists in BOTH graphs
                relationships_map[related]["source"] = "both"
                relationships_map[related]["content_strength"] = content_strength
                relationships_map[related]["total_strength"] += content_strength
            else:
                # Relationship only in content KG
                relationships_map[related] = {
                    "related_entity": related,
                    "source": "content",  # Content KG only
                    "strength": 0,  # No routing data
                    "total_strength": content_strength,
                    "success_together": 0,
                    "failure_together": 0,
                    "content_strength": content_strength
                }

        relationships = list(relationships_map.values())
        relationships.sort(key=lambda x: x["total_strength"], reverse=True)
        return relationships

    async def cleanup(self):
        """Clean shutdown"""
        logger.info("Shutting down UnifiedMemorySystem...")

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Save KG and relationships (immediate save on cleanup)
        await self._save_kg()  # Final cleanup save, not debounced
        await self._save_relationships()

        # Cleanup adapters
        for name, adapter in self.collections.items():
            try:
                await adapter.cleanup()
            except:
                pass

        logger.info("UnifiedMemorySystem shutdown complete")