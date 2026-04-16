"""
Search Service - TagCascade retrieval with ONNX cross-encoder reranking.

v0.3.1: Tags-first cascade replaces Wilson+CE blend.
Benchmark-validated on LoCoMo: 27.3% Hit@1 (p<0.0001 vs Wilson).

Responsibilities:
- Tag-routed search (overlap cascade + cosine fill)
- Cross-encoder reranking (raw CE score, no Wilson blend)
- Collection-specific boosts
- Document caching for outcome scoring
"""

import asyncio
import json
import logging
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from .config import MemoryConfig
from .scoring_service import ScoringService
from .routing_service import RoutingService, ALL_COLLECTIONS
from .tag_service import TagService
from .ghost_registry import get_ghost_registry

logger = logging.getLogger(__name__)


# Type aliases
CollectionName = Literal["working", "patterns", "history", "books", "memory_bank"]


class SearchService:
    """
    TagCascade retrieval with ONNX cross-encoder reranking.

    v0.3.1 Pipeline:
    1. Match query nouns against known tag index (word-boundary regex)
    2. Tag-routed search: per-tag ChromaDB queries, overlap counting, tier-fill pool
    3. Cosine fill remaining pool slots from unfiltered search
    4. CE reranks pool — raw CE score as final ranking (NO Wilson blend)
    """

    # Cross-encoder model for reranking (ONNX, lazy-loaded)
    CE_HF_REPO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    CE_ONNX_FILE = "onnx/model_O4.onnx"
    CE_TOKENIZER_FILE = "tokenizer.json"
    CE_CANDIDATE_POOL = 40  # v0.3.1: 40 candidates for CE (was 20)

    def __init__(
        self,
        collections: Dict[str, Any],  # ChromaDBAdapter instances
        scoring_service: ScoringService,
        routing_service: RoutingService,
        tag_service: TagService,
        embed_fn: Callable[[str], Any],  # Async function to embed text
        config: Optional[MemoryConfig] = None,
        **kwargs,  # Accept and ignore kg_service for backward compat
    ):
        self.collections = collections
        self.scoring_service = scoring_service
        self.routing_service = routing_service
        self.tag_service = tag_service
        self.embed_fn = embed_fn
        self.config = config or MemoryConfig()

        # Cross-encoder reranker (ONNX, lazy-loaded on first search)
        self._ce_session = None
        self._ce_tokenizer = None
        self._ce_loaded = False

        # Cache for doc_ids per session (for outcome scoring)
        self._cached_doc_ids: Dict[str, List[str]] = {}

    def _load_ce(self):
        """Lazy-load cross-encoder ONNX model on first use."""
        if self._ce_loaded:
            return self._ce_session is not None
        self._ce_loaded = True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=self.CE_HF_REPO, filename=self.CE_ONNX_FILE)
            tokenizer_path = hf_hub_download(repo_id=self.CE_HF_REPO, filename=self.CE_TOKENIZER_FILE)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 0  # auto-detect

            self._ce_session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._ce_tokenizer = Tokenizer.from_file(tokenizer_path)
            self._ce_tokenizer.enable_padding()
            self._ce_tokenizer.enable_truncation(max_length=256)

            logger.info(f"Cross-encoder loaded (ONNX): {self.CE_HF_REPO}")
            return True
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable: {e}. Falling back to cosine-only.")
            self._ce_session = None
            self._ce_tokenizer = None
            return False

    def _ce_predict(self, pairs: list) -> list:
        """Run cross-encoder inference on query-document pairs via ONNX."""
        import numpy as np

        # Use encode_batch for uniform padding across all pairs
        encoded = self._ce_tokenizer.encode_batch(
            [(q, d) for q, d in pairs]
        )
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        session_inputs = {inp.name for inp in self._ce_session.get_inputs()}
        if "token_type_ids" in session_inputs:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._ce_session.run(None, feeds)
        logits = outputs[0]
        if logits.ndim == 2:
            logits = logits[:, 0]
        return logits.tolist()

    # =========================================================================
    # Date Filter Helpers (v0.3.1, ported from core)
    # =========================================================================

    DATE_FIELDS = ('timestamp', 'last_used', 'created_at', 'first_seen', 'last_seen')

    def _extract_date_filters(
        self, filters: Optional[Dict[str, Any]]
    ) -> tuple:
        """Separate date filters (post-query) from ChromaDB filters."""
        if not filters:
            return None, None
        chromadb_filters = {}
        date_filters = {}
        for key, value in filters.items():
            if key in self.DATE_FIELDS:
                date_filters[key] = value
            else:
                chromadb_filters[key] = value
        return chromadb_filters or None, date_filters or None

    def _apply_date_filters(
        self, results: List[Dict], date_filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply date filtering in Python (ISO strings sort alphabetically)."""
        filtered = results
        for field, condition in date_filters.items():
            if isinstance(condition, str):
                if len(condition) == 10:  # YYYY-MM-DD
                    filtered = [
                        r for r in filtered
                        if r.get('metadata', {}).get(field, '').startswith(condition)
                    ]
                else:
                    filtered = [
                        r for r in filtered
                        if r.get('metadata', {}).get(field) == condition
                    ]
            elif isinstance(condition, dict):
                for op, val in condition.items():
                    if op == '$gte':
                        filtered = [r for r in filtered if r.get('metadata', {}).get(field, '') >= val]
                    elif op == '$gt':
                        filtered = [r for r in filtered if r.get('metadata', {}).get(field, '') > val]
                    elif op == '$lte':
                        filtered = [r for r in filtered if r.get('metadata', {}).get(field, '') <= val]
                    elif op == '$lt':
                        filtered = [r for r in filtered if r.get('metadata', {}).get(field, '') < val]
        return filtered

    # =========================================================================
    # Main Search
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        collections: Optional[List[CollectionName]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        transparency_context: Optional[Any] = None
    ) -> Union[List[Dict], Dict]:
        """
        Search memory with tag-routed retrieval and CE reranking.

        v0.3.1: TagCascade — tags-first cascade fills candidate pool,
        CE reranks with raw score (no Wilson blend).
        """
        if collections is None:
            collections = self.routing_service.route_query(query)

        # Special handling for empty query - return all items
        if not query or query.strip() == "":
            return await self._search_all(
                collections, limit, offset, return_metadata
            )

        # Preprocess query for better retrieval
        processed_query = self.routing_service.preprocess_query(query)

        # Generate query embedding
        try:
            query_embedding = await self.embed_fn(processed_query)
        except Exception as e:
            logger.error(f"Embedding generation failed for query '{query}': {e}")
            if return_metadata:
                return {"results": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}
            return []

        # Track search start if context provided
        if transparency_context and hasattr(transparency_context, 'track_action'):
            transparency_context.track_action(
                action_type="memory_search",
                description=f"Searching: {query[:50]}{'...' if len(query) > 50 else ''}",
                detail=f"Collections: {', '.join(collections)}",
                status="executing"
            )

        # v0.3.1: Date filter separation (date fields handled post-query in Python)
        chromadb_filters, date_filters = self._extract_date_filters(metadata_filters)
        fetch_limit = limit * 3 if date_filters else limit

        # v0.3.1: Tag-routed search
        # v0.3.1.1: Skip tag routing for fact lane — facts have no tags,
        # retrieved via cosine + CE only (matching benchmark two-lane architecture)
        is_fact_lane = (
            chromadb_filters
            and chromadb_filters.get("memory_type") == "fact"
        )
        matched_tags = [] if is_fact_lane else self.tag_service.match_query_tags(query)

        if matched_tags:
            all_results = await self._tag_routed_search(
                query_embedding, processed_query, collections,
                fetch_limit, matched_tags, chromadb_filters
            )
        else:
            # No tags matched (or fact lane) -> straight cosine
            all_results = await self._search_collections(
                query_embedding, processed_query, collections, fetch_limit, chromadb_filters
            )

        # Date filters (applied in Python after retrieval)
        if date_filters:
            all_results = self._apply_date_filters(all_results, date_filters)

        # Apply scoring (Wilson computed as metadata, not used for ranking)
        all_results = self.scoring_service.apply_scoring_to_results(all_results)

        # v0.3.1 TagCascade: CE reranks pool, raw CE score is final ranking
        all_results = self._rerank_with_ce(query, all_results)

        # Track and paginate
        paginated_results = all_results[offset:offset + limit]
        self._track_search_results(query, paginated_results, transparency_context)

        if return_metadata:
            return {
                "results": paginated_results,
                "total": len(all_results),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < len(all_results)
            }
        return paginated_results

    # =========================================================================
    # Tag-Routed Search (v0.3.1 TagCascade)
    # =========================================================================

    async def _tag_routed_search(
        self,
        query_embedding: List[float],
        processed_query: str,
        collections: List[str],
        limit: int,
        matched_tags: List[str],
        metadata_filters: Optional[Dict]
    ) -> List[Dict]:
        """
        Tag-routed search with overlap counting.

        For each matched tag, query ChromaDB with noun_tags filter.
        Count how many tags each result matches (overlap count).
        Fill candidate pool from highest overlap tier down.
        Cosine-fill remaining slots from unfiltered search.
        """
        result_overlaps: Dict[str, Dict] = {}  # doc_id -> {result, overlap_count}

        # Tag-filtered queries per collection
        for coll_name in collections:
            if coll_name not in self.collections:
                continue
            adapter = self.collections[coll_name]

            for tag in matched_tags:
                try:
                    # ChromaDB $contains with quoted tag prevents substring matches
                    tag_filter = {"noun_tags": {"$contains": f'"{tag}"'}}

                    # Merge with user metadata filters
                    mb_filters = metadata_filters
                    if coll_name == "memory_bank":
                        mb_filters = dict(metadata_filters) if metadata_filters else {}
                        mb_filters.setdefault("status", {"$ne": "archived"})
                    merged = self._merge_filters(tag_filter, mb_filters)

                    results = await adapter.hybrid_query(
                        query_vector=query_embedding,
                        query_text=processed_query,
                        top_k=limit * 2,
                        filters=merged
                    )

                    for r in results:
                        doc_id = r.get("id", "")
                        r["collection"] = coll_name

                        if doc_id in result_overlaps:
                            result_overlaps[doc_id]["overlap_count"] += 1
                            # Keep better distance
                            if r.get("distance", 1.0) < result_overlaps[doc_id]["result"].get("distance", 1.0):
                                result_overlaps[doc_id]["result"] = r
                        else:
                            result_overlaps[doc_id] = {
                                "result": r,
                                "overlap_count": 1
                            }
                except Exception as e:
                    logger.warning(f"Tag search failed for '{tag}' in {coll_name}: {e}")

        # Apply collection boosts and tag_overlap_count to tag-matched results
        for entry in result_overlaps.values():
            r = entry["result"]
            self._apply_collection_boost(r, r.get("collection", ""), processed_query)
            r["tag_overlap_count"] = entry["overlap_count"]

        # v0.3.1 TagCascade: tier-filling pool construction
        pool_size = self.CE_CANDIDATE_POOL
        pool: List[Dict] = []
        seen_ids: set = set()

        if result_overlaps:
            candidates_list = list(result_overlaps.values())
            max_overlap = max(e["overlap_count"] for e in candidates_list)

            for tier in range(max_overlap, 0, -1):
                tier_cands = [
                    e for e in candidates_list
                    if e["overlap_count"] == tier and e["result"].get("id") not in seen_ids
                ]
                # Within tier: sort by cosine distance ascending (closest first)
                tier_cands.sort(key=lambda e: e["result"].get("distance", 1.0))

                for e in tier_cands:
                    pool.append(e["result"])
                    seen_ids.add(e["result"].get("id"))
                    if len(pool) >= pool_size:
                        break
                if len(pool) >= pool_size:
                    break

        # Cosine fill remaining slots from unfiltered search
        if len(pool) < pool_size:
            cosine_results = await self._search_collections(
                query_embedding, processed_query, collections, limit, metadata_filters
            )
            for r in cosine_results:
                doc_id = r.get("id", "")
                if doc_id not in seen_ids:
                    self._apply_collection_boost(r, r.get("collection", ""), processed_query)
                    r["tag_overlap_count"] = 0
                    pool.append(r)
                    seen_ids.add(doc_id)
                    if len(pool) >= pool_size:
                        break

        return pool

    @staticmethod
    def _merge_filters(
        tag_filter: Dict, metadata_filters: Optional[Dict]
    ) -> Optional[Dict]:
        """Merge tag filter with user metadata filters using ChromaDB $and."""
        if not metadata_filters:
            return tag_filter
        conditions = []
        for k, v in tag_filter.items():
            conditions.append({k: v})
        for k, v in metadata_filters.items():
            conditions.append({k: v})
        return {"$and": conditions}

    # =========================================================================
    # Cosine Search (fallback when no tags match)
    # =========================================================================

    async def _search_all(
        self,
        collections: List[str],
        limit: int,
        offset: int,
        return_metadata: bool
    ) -> Union[List[Dict], Dict]:
        """Handle empty query - return all items."""
        all_results = []

        for coll_name in collections:
            if coll_name not in self.collections:
                continue

            try:
                adapter = self.collections[coll_name]
                collection_obj = adapter.collection
                items = collection_obj.get(limit=100000)

                for i in range(len(items['ids'])):
                    metadata = items['metadatas'][i] if i < len(items['metadatas']) else {}
                    result = {
                        'id': items['ids'][i],
                        'content': items['documents'][i] if i < len(items['documents']) else '',
                        'text': items['documents'][i] if i < len(items['documents']) else '',
                        'metadata': metadata,
                        'collection': coll_name
                    }
                    if 'score' in metadata:
                        result['score'] = metadata['score']
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error getting all items from {coll_name}: {e}")

        # Sort by timestamp
        all_results.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)

        paginated_results = all_results[offset:offset + limit]

        # v0.3.1: Add recency metadata so Memory Panel shows relative ages
        self._add_recency_metadata(paginated_results)

        if return_metadata:
            return {
                "results": paginated_results,
                "total": len(all_results),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < len(all_results)
            }
        return paginated_results

    async def _search_collections(
        self,
        query_embedding: List[float],
        processed_query: str,
        collections: List[str],
        limit: int,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """Search specified collections with cosine similarity (parallel)."""
        valid_collections = [c for c in collections if c in self.collections]
        if not valid_collections:
            return []

        coll_results = await asyncio.gather(*(
            self._search_single_collection(
                coll_name, query_embedding, processed_query, limit, metadata_filters
            )
            for coll_name in valid_collections
        ), return_exceptions=True)

        all_results = []
        for coll_name, results in zip(valid_collections, coll_results):
            if isinstance(results, Exception):
                logger.warning(f"Search failed for {coll_name}: {results}")
                continue

            for r in results:
                r["collection"] = coll_name
                self._apply_collection_boost(r, coll_name, processed_query)

            all_results.extend(results)

        return all_results

    async def _search_single_collection(
        self,
        coll_name: str,
        query_embedding: List[float],
        processed_query: str,
        limit: int,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """Search a single collection with timeout protection."""
        adapter = self.collections[coll_name]
        multiplier = self.config.search_multiplier

        # Build filters for memory_bank (exclude archived)
        filters = metadata_filters
        if coll_name == "memory_bank":
            status_filter = {"status": {"$ne": "archived"}}
            if metadata_filters:
                # v0.3.1: Wrap in $and when combining with existing filters
                filters = {"$and": [{k: v} for k, v in metadata_filters.items()] + [status_filter]}
            else:
                filters = status_filter

        # v0.2.10: Timeout protection for ChromaDB queries
        QUERY_TIMEOUT = 15.0
        try:
            results = await asyncio.wait_for(
                adapter.hybrid_query(
                    query_vector=query_embedding,
                    query_text=processed_query,
                    top_k=limit * multiplier,
                    filters=filters
                ),
                timeout=QUERY_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"[SearchService] Query timeout after {QUERY_TIMEOUT}s for collection {coll_name}")
            return []

        # Add recency metadata for working memory
        if coll_name == "working":
            self._add_recency_metadata(results)

        return results

    def _apply_collection_boost(self, result: Dict, coll_name: str, query: str):
        """Apply collection-specific distance boosts."""
        # Patterns get slight boost
        if coll_name == "patterns":
            result["distance"] = result.get("distance", 1.0) * 0.9

        # Memory bank: boost by importance * confidence (cold start quality)
        elif coll_name == "memory_bank":
            metadata = result.get("metadata", {})
            importance = self._parse_numeric(metadata.get("importance", 0.7))
            confidence = self._parse_numeric(metadata.get("confidence", 0.7))
            quality_score = importance * confidence

            metadata_boost = 1.0 - quality_score * 0.8
            result["distance"] = result.get("distance", 1.0) * metadata_boost

        # Books: boost recent uploads
        elif coll_name == "books":
            if result.get("upload_timestamp"):
                try:
                    upload_time = datetime.fromisoformat(result["upload_timestamp"])
                    age_days = (datetime.utcnow() - upload_time).days
                    if age_days <= 7:
                        result["distance"] = result.get("distance", 1.0) * 0.7
                except Exception:
                    pass

    def _add_recency_metadata(self, results: List[Dict]):
        """Add recency metadata to working memory results."""
        for r in results:
            metadata = r.get("metadata", {})
            if metadata.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                    minutes_ago = (datetime.now() - timestamp).total_seconds() / 60

                    if minutes_ago < 1:
                        metadata["recency"] = "just now"
                    elif minutes_ago < 60:
                        metadata["recency"] = f"{int(minutes_ago)} minutes ago"
                    else:
                        hours_ago = minutes_ago / 60
                        metadata["recency"] = f"{int(hours_ago)} hours ago"

                    r["minutes_ago"] = minutes_ago
                except Exception:
                    r["minutes_ago"] = 999

    def _parse_numeric(self, value: Any) -> float:
        """Parse value to float, handling various formats."""
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else 0.7
        elif isinstance(value, str):
            level_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
            return level_map.get(value.lower(), 0.7)
        try:
            return float(value) if value else 0.7
        except (ValueError, TypeError):
            return 0.7

    # =========================================================================
    # Cross-Encoder Reranking (v0.3.1: raw CE score, no Wilson blend)
    # =========================================================================

    def _rerank_with_ce(self, query: str, results: list) -> list:
        """
        Rerank results using cross-encoder.

        v0.3.1 TagCascade: Raw CE score as final ranking signal.
        No Wilson blend — benchmark showed Wilson hurts retrieval at every stage
        (p<0.001 in all configs). Wilson stays as metadata for display only.
        """
        if not self._load_ce() or not results:
            return results

        candidates = results[:self.CE_CANDIDATE_POOL]
        remainder = results[self.CE_CANDIDATE_POOL:]

        pairs = [[query, r.get("text", r.get("content", r.get("metadata", {}).get("text", "")))] for r in candidates]
        try:
            ce_scores = self._ce_predict(pairs)
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return results

        for i, r in enumerate(candidates):
            r["ce_score"] = ce_scores[i]
            r["final_rank_score"] = ce_scores[i]  # Raw CE, no Wilson blend

        # Sort by raw CE score
        candidates.sort(key=lambda x: x.get("final_rank_score", 0), reverse=True)

        return candidates + remainder

    # =========================================================================
    # Tracking and Caching
    # =========================================================================

    def _track_search_results(
        self,
        query: str,
        results: List[Dict],
        transparency_context: Optional[Any]
    ):
        """Track search results for outcome scoring."""
        # Cache doc_ids for outcome scoring
        session_id = 'default'
        if transparency_context and hasattr(transparency_context, 'session_id'):
            session_id = transparency_context.session_id

        cached_doc_ids = []
        for result in results:
            collection = result.get("collection", "")
            if collection in ["working", "history", "patterns"]:
                doc_id = result.get("id")
                if doc_id:
                    cached_doc_ids.append(doc_id)

        self._cached_doc_ids[session_id] = cached_doc_ids
        if cached_doc_ids:
            logger.debug(f"Cached {len(cached_doc_ids)} doc_ids for outcome scoring")

        # Track with transparency context
        if transparency_context and hasattr(transparency_context, 'track_memory_search'):
            confidence_scores = [math.exp(-r.get("distance", 0.5) / 100.0) for r in results]
            transparency_context.track_memory_search(
                query=query,
                collections=[r.get("collection") for r in results[:1]],
                results_count=len(results),
                confidence_scores=confidence_scores
            )

    def _get_fragment(self, collection: str, doc_id: str) -> Optional[Dict]:
        """Get a document fragment by collection and ID."""
        if collection in self.collections:
            return self.collections[collection].get_fragment(doc_id)
        return None

    def get_cached_doc_ids(self, session_id: str = 'default') -> List[str]:
        """Get cached doc_ids for a session."""
        return self._cached_doc_ids.get(session_id, [])

    # =========================================================================
    # Book Search (Specialized)
    # =========================================================================

    async def search_books(
        self,
        query: str,
        chunk_type: Optional[str] = None,
        has_code: Optional[bool] = None,
        code_language: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Enhanced book search with metadata filtering.

        Args:
            query: Search query text
            chunk_type: Filter by chunk type ("code", "prose", "mixed")
            has_code: Filter by presence of code
            code_language: Filter by programming language
            n_results: Number of results to return

        Returns:
            List of search results with enhanced context
        """
        if "books" not in self.collections:
            logger.warning("Books collection not initialized")
            return []

        # Build where clause
        where = {}
        if chunk_type:
            where["chunk_type"] = chunk_type
        if has_code is not None:
            where["has_code"] = has_code
        if code_language:
            where["code_language"] = code_language

        try:
            results = await self.collections["books"].query(
                query_texts=[query],
                n_results=n_results * 2,
                where=where if where else None
            )

            if not results or not results.get("ids"):
                return []

            # Format results
            formatted_results = []
            for i in range(min(n_results, len(results["ids"][0]))):
                result = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                    "collection": "books"
                }
                formatted_results.append(result)

            # v0.2.9: Filter out ghost results (deleted chunks still in HNSW index)
            from config.settings import settings
            ghost_registry = get_ghost_registry(settings.paths.data_dir)
            filtered_results = ghost_registry.filter_ghosts(formatted_results)

            return filtered_results

        except Exception as e:
            logger.error(f"Book search failed: {e}")
            return []
