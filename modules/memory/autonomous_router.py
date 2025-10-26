"""
LLM-Autonomous Knowledge Graph Routing - Production Ready
Full LLM autonomy with outcome-based learning - ZERO hard-coded overrides

Philosophy:
- LLM makes ALL routing decisions (no confidence thresholds)
- LLM knows the schema (no validation filters)
- Outcomes teach, not arbitrary rules
- Heuristic ONLY for technical LLM failures (not distrust)

Performance:
- 5-minute cache for repeated queries
- Outcome tracking for continuous learning
- Accuracy stats visible to LLM for self-improvement
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AutonomousRouter:
    """
    Fully autonomous LLM-based routing.

    Zero training wheels - LLM decisions are final.
    """

    def __init__(self, knowledge_graph: Dict, llm_service=None, cache_ttl: int = 300):
        """
        Initialize autonomous router.

        Args:
            knowledge_graph: Reference to KG (shared state)
            llm_service: LLM for routing decisions
            cache_ttl: Cache TTL in seconds (default: 300)
        """
        self.knowledge_graph = knowledge_graph
        self.llm_service = llm_service
        self.cache_ttl = cache_ttl
        self._routing_cache: Dict[str, tuple] = {}  # query_hash -> (collections, timestamp, reasoning)

        # Outcome tracking for learning
        self.routing_outcomes: Dict[str, Dict] = {}  # decision_id -> decision_data
        self.llm_accuracy_stats = {
            "total_decisions": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
            "accuracy_rate": 0.0
        }

    async def route_query(self, query: str, concepts: List[str]) -> List[str]:
        """
        Route query using LLM autonomy.

        Args:
            query: User query
            concepts: Extracted concepts

        Returns:
            Collections to search (LLM decision, no overrides)
        """
        # Check cache (performance)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._routing_cache:
            collections, timestamp, reasoning = self._routing_cache[cache_key]
            if (time.time() - timestamp) < self.cache_ttl:
                logger.debug(f"[AutonomousRouter] Cache hit: {collections}")
                return collections

        # LLM routing (autonomous)
        if self.llm_service:
            try:
                result = await self._llm_route(query, concepts)

                # Trust LLM completely - NO validation
                collections = result.get("collections", [])
                if not collections:
                    # LLM returned empty - use its emergency suggestion or default
                    collections = result.get("emergency_fallback", ["working", "patterns", "history", "books"])

                reasoning = result.get("reasoning", "")

                # Cache decision
                self._routing_cache[cache_key] = (collections, time.time(), reasoning)

                # Track for outcome learning
                self._log_decision(query, collections, result)

                logger.info(f"[AutonomousRouter] LLM: {collections} | {reasoning[:100]}")
                return collections

            except Exception as e:
                logger.error(f"[AutonomousRouter] LLM technical failure: {e}")
                return self._emergency_fallback(concepts)
        else:
            logger.warning("[AutonomousRouter] No LLM available, using fallback")
            return self._emergency_fallback(concepts)

    async def _llm_route(self, query: str, concepts: List[str]) -> Dict[str, Any]:
        """
        Let LLM decide autonomously - zero overrides.

        Returns:
            LLM decision (trusted completely)
        """
        # Gather KG context (information, not constraints)
        routing_context = []
        for concept in concepts[:10]:
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

        # Build autonomous prompt
        prompt = f"""You are the autonomous routing intelligence for a memory system. You have FULL autonomy - your decisions are final.

Query: "{query}"

Available Collections (you know these):
- **working**: Current conversation context (24h retention, highest recency)
- **patterns**: Proven solutions (permanent, high reliability)
- **history**: Past conversations and attempts (30d retention, learning from failures)
- **books**: Reference documentation (permanent, architectural knowledge)
- **memory_bank**: User/project context (permanent, identity and preferences)

Your Performance Stats:
- Accuracy: {self.llm_accuracy_stats['accuracy_rate']:.1%}
- Successful decisions: {self.llm_accuracy_stats['successful_outcomes']}
- Learning from: {self.llm_accuracy_stats['failed_outcomes']} failures

Knowledge Graph Context (hints, not rules):
{json.dumps(routing_context, indent=2) if routing_context else "No prior routing data - use your best judgment"}

Recent Routing Failures (learn from these):
{json.dumps(self.knowledge_graph.get("routing_failures", [])[-3:], indent=2) if self.knowledge_graph.get("routing_failures") else "No failures yet"}

Instructions:
1. Analyze query semantics (is this architectural? debugging? how-to? conceptual?)
2. Consider KG stats as hints (you can override them based on context)
3. Choose 1-5 collections based on your autonomous judgment
4. If uncertain, express it in reasoning (we learn from outcomes, not guesses)
5. Your decision is FINAL - no validation, no overrides

Return JSON:
{{
    "collections": ["collection1", "collection2", ...],
    "reasoning": "why you chose these collections and your confidence level",
    "uncertainty_notes": "any concerns or alternative approaches you considered (optional)"
}}"""

        # Call LLM - trust response completely
        response = await self.llm_service.generate_structured(
            prompt=prompt,
            response_format={"type": "json_object"}
        )

        result = json.loads(response)

        # NO validation - trust LLM knows the schema
        # NO confidence threshold - LLM decides when uncertain
        return result

    def _log_decision(self, query: str, collections: List[str], llm_result: Dict) -> str:
        """Log routing decision for outcome-based learning."""
        decision_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()

        self.routing_outcomes[decision_id] = {
            "query": query,
            "collections": collections,
            "reasoning": llm_result.get("reasoning", ""),
            "uncertainty": llm_result.get("uncertainty_notes", ""),
            "timestamp": time.time(),
            "outcome": None  # Updated later
        }

        self.llm_accuracy_stats["total_decisions"] += 1
        return decision_id

    def record_routing_outcome(self, query: str, worked: bool, user_feedback: Optional[str] = None):
        """
        Record outcome of routing decision for learning.

        Args:
            query: Original query
            worked: Did routing lead to good answer?
            user_feedback: Optional user feedback
        """
        # Find decision
        for decision_id, decision in self.routing_outcomes.items():
            if decision["query"] == query and decision["outcome"] is None:
                # Update outcome
                decision["outcome"] = "success" if worked else "failure"
                decision["user_feedback"] = user_feedback
                decision["outcome_timestamp"] = time.time()

                # Update stats
                if worked:
                    self.llm_accuracy_stats["successful_outcomes"] += 1
                else:
                    self.llm_accuracy_stats["failed_outcomes"] += 1

                total = self.llm_accuracy_stats["successful_outcomes"] + self.llm_accuracy_stats["failed_outcomes"]
                if total > 0:
                    self.llm_accuracy_stats["accuracy_rate"] = self.llm_accuracy_stats["successful_outcomes"] / total

                logger.info(f"[AutonomousRouter] Outcome: {'SUCCESS' if worked else 'FAILURE'} | "
                           f"Accuracy: {self.llm_accuracy_stats['accuracy_rate']:.1%} "
                           f"({self.llm_accuracy_stats['successful_outcomes']}/{total})")

                # Learn from failures
                if not worked:
                    self._analyze_failure(decision)

                break

    def _analyze_failure(self, decision: Dict):
        """Analyze routing failure for future learning."""
        logger.warning(f"[AutonomousRouter] Learning from failure:\n"
                      f"  Query: {decision['query']}\n"
                      f"  Chosen: {decision['collections']}\n"
                      f"  Reasoning: {decision['reasoning']}\n"
                      f"  Uncertainty: {decision.get('uncertainty', 'None')}")

        # Store failure for LLM to see in future prompts
        if "routing_failures" not in self.knowledge_graph:
            self.knowledge_graph["routing_failures"] = []

        self.knowledge_graph["routing_failures"].append({
            "query": decision["query"],
            "chosen_collections": decision["collections"],
            "reasoning": decision["reasoning"],
            "timestamp": decision["timestamp"]
        })

        # Keep last 50 failures
        self.knowledge_graph["routing_failures"] = self.knowledge_graph["routing_failures"][-50:]

    def _emergency_fallback(self, concepts: List[str]) -> List[str]:
        """
        Emergency fallback ONLY for technical LLM failures.
        NOT used for "low confidence" - this is a safety net.
        """
        logger.warning("[AutonomousRouter] EMERGENCY FALLBACK (LLM unavailable)")

        # Simple KG lookup
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern = self.knowledge_graph["routing_patterns"][concept]
                best = pattern.get("best_collection", "patterns")
                return ["working", best, "history"]

        # Ultimate safe default
        return ["working", "patterns", "history", "books"]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        return {
            "llm_accuracy": self.llm_accuracy_stats,
            "total_decisions": len(self.routing_outcomes),
            "cache_size": len(self._routing_cache),
            "recent_failures": [
                {
                    "query": d["query"],
                    "collections": d["collections"],
                    "reasoning": d["reasoning"]
                }
                for d in list(self.routing_outcomes.values())[-5:]
                if d.get("outcome") == "failure"
            ]
        }

    def clear_cache(self):
        """Clear routing cache."""
        self._routing_cache.clear()
        logger.info("[AutonomousRouter] Cache cleared")
