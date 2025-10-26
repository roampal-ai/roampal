"""
LLM-Enhanced Knowledge Graph Routing
Provides intelligent collection routing with heuristic fallback

Features:
- LLM-based semantic routing using KG stats as context
- 5-minute TTL cache to prevent redundant LLM calls
- Graceful fallback to rule-based routing if LLM unavailable
- Zero breaking changes - drop-in replacement for _route_query
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """
    LLM-enhanced routing with caching and heuristic fallback.

    Workflow:
    1. Check cache (5min TTL)
    2. Try LLM routing with KG context
    3. Fallback to heuristic if LLM fails/unavailable
    """

    def __init__(self, knowledge_graph: Dict, llm_service=None, cache_ttl: int = 300):
        """
        Initialize intelligent router.

        Args:
            knowledge_graph: Reference to UnifiedMemorySystem.knowledge_graph
            llm_service: LLM service for semantic routing (optional)
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5min)
        """
        self.knowledge_graph = knowledge_graph
        self.llm_service = llm_service
        self.cache_ttl = cache_ttl
        self._routing_cache: Dict[str, tuple] = {}  # query_hash -> (collections, timestamp)

    async def route_query(self, query: str, concepts: List[str]) -> List[str]:
        """
        Intelligently route query to best collections.

        Args:
            query: User's query string
            concepts: Extracted concepts from query

        Returns:
            List of collection names to search, ordered by relevance
        """
        # Check cache first (performance optimization)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._routing_cache:
            collections, timestamp = self._routing_cache[cache_key]
            if (time.time() - timestamp) < self.cache_ttl:
                logger.debug(f"[IntelligentRouter] Cache hit: {collections}")
                return collections

        # Try LLM routing if available
        if self.llm_service:
            try:
                collections = await self._llm_route(query, concepts)
                if collections:
                    # Cache successful LLM routing
                    self._routing_cache[cache_key] = (collections, time.time())
                    logger.info(f"[IntelligentRouter] LLM routing: {collections}")
                    return collections
            except Exception as e:
                logger.warning(f"[IntelligentRouter] LLM routing failed, using heuristic: {e}")

        # Fallback to heuristic routing
        collections = self._heuristic_route(concepts)
        logger.debug(f"[IntelligentRouter] Heuristic routing: {collections}")
        return collections

    async def _llm_route(self, query: str, concepts: List[str]) -> Optional[List[str]]:
        """
        LLM-based routing using KG statistics as context.

        Returns:
            List of collection names if confident (>0.7), None otherwise
        """
        # Gather KG stats for LLM context
        routing_context = []
        for concept in concepts[:5]:  # Limit to top 5 concepts
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern = self.knowledge_graph["routing_patterns"][concept]
                routing_context.append({
                    "concept": concept,
                    "best_collection": pattern.get("best_collection"),
                    "success_rate": pattern.get("success_rate", 0.5),
                    "total_uses": sum(
                        stats.get("total", 0)
                        for stats in pattern.get("collections_used", {}).values()
                    )
                })

        # Build LLM prompt
        prompt = f"""Analyze this query and recommend 2-4 memory collections to search, prioritized by relevance.

Query: "{query}"

Available Collections:
- **working**: Current conversation context (24h retention, recency matters most)
- **patterns**: Proven solutions with high success rates (permanent storage)
- **history**: Past conversation attempts and learnings (30d retention)
- **books**: Reference documentation and guides (permanent storage)
- **memory_bank**: Persistent user/project context (permanent storage)

Knowledge Graph Stats:
{json.dumps(routing_context, indent=2) if routing_context else "No prior routing data for these concepts"}

Consider:
- Query semantics (is this architectural, debugging, how-to, or informational?)
- Concept success rates (which collections worked before?)
- Collection purposes (recency vs proven patterns vs reference docs)

Return ONLY valid JSON:
{{
    "collections": ["collection1", "collection2"],
    "reasoning": "brief explanation of why these collections",
    "confidence": 0.0-1.0
}}"""

        try:
            # Call LLM with structured output
            response = await self.llm_service.generate_structured(
                prompt=prompt,
                response_format={"type": "json_object"}
            )

            result = json.loads(response)

            # Validate response structure
            if not all(k in result for k in ["collections", "reasoning", "confidence"]):
                logger.warning(f"[IntelligentRouter] Invalid LLM response structure: {result}")
                return None

            # Check confidence threshold
            confidence = result.get("confidence", 0)
            if confidence <= 0.7:
                logger.debug(f"[IntelligentRouter] Low confidence ({confidence}), using heuristic")
                return None

            # Validate collections are valid
            valid_collections = {"working", "patterns", "history", "books", "memory_bank"}
            collections = [c for c in result.get("collections", []) if c in valid_collections]

            if not collections:
                logger.warning(f"[IntelligentRouter] No valid collections in LLM response: {result}")
                return None

            logger.info(f"[IntelligentRouter] LLM decision (conf={confidence:.2f}): {collections} | Reason: {result.get('reasoning', 'N/A')}")
            return collections

        except json.JSONDecodeError as e:
            logger.warning(f"[IntelligentRouter] Failed to parse LLM JSON response: {e}")
            return None
        except Exception as e:
            logger.debug(f"[IntelligentRouter] LLM routing error: {e}")
            return None

    def _heuristic_route(self, concepts: List[str]) -> List[str]:
        """
        Rule-based fallback routing (existing logic).

        Algorithm:
        1. Check KG routing patterns for known concepts
        2. If concept has >60% success rate in a collection, prioritize it
        3. Default: comprehensive search across all collections

        Args:
            concepts: Extracted concepts from query

        Returns:
            List of collection names to search
        """
        # Check KG learned routing patterns
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern = self.knowledge_graph["routing_patterns"][concept]
                success_rate = pattern.get("success_rate", 0)

                if success_rate > 0.6:
                    best_collection = pattern.get("best_collection", "patterns")
                    # Return working first (recency), then best collection, then history
                    return ["working", best_collection, "history"]

        # Default: comprehensive search, ordered by priority
        # working (recency) → patterns (proven) → history (learning) → books (reference)
        return ["working", "patterns", "history", "books"]

    def clear_cache(self):
        """Clear routing cache (useful for testing or after KG updates)"""
        self._routing_cache.clear()
        logger.info("[IntelligentRouter] Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        now = time.time()
        active_entries = sum(
            1 for (_, timestamp) in self._routing_cache.values()
            if (now - timestamp) < self.cache_ttl
        )

        return {
            "total_entries": len(self._routing_cache),
            "active_entries": active_entries,
            "expired_entries": len(self._routing_cache) - active_entries,
            "cache_ttl_seconds": self.cache_ttl
        }
