"""
OutcomeService - Extracted from UnifiedMemorySystem

Handles outcome recording, score updates, and learning from feedback.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal, Callable, Awaitable

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class OutcomeService:
    """
    Service for recording outcomes and updating memory scores.

    Extracted from UnifiedMemorySystem.record_outcome and related methods.
    Handles:
    - Score updates (flat deltas, no time-weight)
    - Outcome history tracking
    - Promotion/demotion triggering
    - Batch cleanup of old working memory (every 50 scores)

    v0.3.1: KG routing removed (matches core v0.4.5). Tags handle routing.
    v0.3.1: Time-weight removed (matches core v0.3.6). A score is a score.
    """

    def __init__(
        self,
        collections: Dict[str, Any],
        promotion_service: Any = None,
        config: Optional[MemoryConfig] = None,
        **kwargs,  # Accept and ignore kg_service for backward compat
    ):
        """
        Initialize OutcomeService.

        Args:
            collections: Dict of collection name -> adapter
            promotion_service: PromotionService for promotion handling
            config: Memory configuration
        """
        self.collections = collections
        self.promotion_service = promotion_service
        self.config = config or MemoryConfig()
        self._score_count = 0
        self._cleanup_interval = 50  # Trigger cleanup every N scores

    async def record_outcome(
        self,
        doc_id: str,
        outcome: Literal["worked", "failed", "partial", "unknown"],
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Record outcome and trigger learning.

        Args:
            doc_id: Document that was used
            outcome: Whether it worked
            failure_reason: Reason for failure (if applicable)
            context: Additional context for learning

        Returns:
            Updated metadata or None if document not found
        """
        # Find collection and document
        collection_name = None
        doc = None

        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                collection_name = coll_name
                doc = adapter.get_fragment(doc_id)
                break

        # SAFEGUARD: Books are reference material, not scorable memories
        if doc_id.startswith("books_") or doc_id.startswith("book_"):
            logger.info(f"Skipping score update for books/{doc_id} (static reference material)")
            return None

        # v0.3.1: If doc not found, check if it was promoted to the next collection
        if not doc and collection_name:
            promotion_targets = {"working": "history", "history": "patterns"}
            target_coll = promotion_targets.get(collection_name)
            if target_coll and target_coll in self.collections:
                try:
                    target_adapter = self.collections[target_coll]
                    if target_adapter.collection:
                        promoted_results = target_adapter.collection.get(
                            where={"original_id": doc_id},
                            include=["metadatas", "documents", "embeddings"],
                            limit=1,
                        )
                        if promoted_results and promoted_results.get("ids") and len(promoted_results["ids"]) > 0:
                            promoted_id = promoted_results["ids"][0]
                            doc = target_adapter.get_fragment(promoted_id)
                            if doc:
                                doc_id = promoted_id
                                collection_name = target_coll
                                logger.info(f"Found promoted doc: {doc_id} in {collection_name}")
                except Exception as e:
                    logger.debug(f"Promoted doc lookup failed for {doc_id}: {e}")

        if not doc:
            logger.debug(f"Document {doc_id} not found (may have been promoted or expired)")
            return None

        # Calculate score update (flat deltas, no time-weight — matches core v0.3.6+)
        metadata = doc.get("metadata", {})
        current_score = metadata.get("score", 0.5)
        uses = metadata.get("uses", 0)
        success_count = metadata.get("success_count", 0.0)  # v0.3.0: Track cumulative successes

        score_delta, new_score, uses, success_delta = self._calculate_score_update(
            outcome, current_score, uses
        )
        success_count += success_delta  # v0.3.0: Accumulate successes (no cap)

        # Update context tracking
        if outcome == "worked" and context:
            contexts = json.loads(metadata.get("success_contexts", "[]"))
            contexts.append(context)
            metadata["success_contexts"] = json.dumps(contexts)
        elif outcome == "failed" and failure_reason:
            reasons = json.loads(metadata.get("failure_reasons", "[]"))
            reasons.append({
                "reason": failure_reason,
                "timestamp": datetime.now().isoformat()
            })
            metadata["failure_reasons"] = json.dumps(reasons)

        # Update outcome history
        outcome_history = json.loads(metadata.get("outcome_history", "[]"))
        outcome_history.append({
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "reason": failure_reason
        })
        outcome_history = outcome_history[-10:]  # Keep last 10

        # Update metadata
        metadata.update({
            "score": new_score,
            "uses": uses,
            "success_count": success_count,  # v0.3.0: Cumulative successes for Wilson score
            "last_outcome": outcome,
            "last_used": datetime.now().isoformat(),
            "outcome_history": json.dumps(outcome_history)
        })

        # Persist to collection
        self.collections[collection_name].update_fragment_metadata(doc_id, metadata)

        logger.info(
            f"Score update [{collection_name}]: {current_score:.2f} → {new_score:.2f} "
            f"(outcome={outcome}, delta={score_delta:+.2f}, uses={uses})"
        )

        # Handle promotion/demotion if service available
        if self.promotion_service:
            collection_size = self.collections[collection_name].collection.count()
            await self.promotion_service.handle_promotion(
                doc_id=doc_id,
                collection=collection_name,
                score=new_score,
                uses=uses,
                metadata=metadata,
                collection_size=collection_size
            )

        # Batch cleanup: trigger every N scores (matches core v0.4.5)
        self._score_count += 1
        if self._score_count % self._cleanup_interval == 0 and self.promotion_service:
            cleaned = await self.promotion_service.cleanup_old_working_memory(max_age_hours=24.0)
            if cleaned > 0:
                logger.info(f"Batch cleanup triggered: removed {cleaned} old working memories")
            cleaned_history = await self.promotion_service.cleanup_old_history(max_age_hours=720.0)
            if cleaned_history > 0:
                logger.info(f"Batch cleanup triggered: removed {cleaned_history} old history items")

        logger.info(f"Outcome recorded: {doc_id} -> {outcome} (score: {new_score:.2f})")
        return metadata

    def _calculate_score_update(
        self,
        outcome: str,
        current_score: float,
        uses: int,
    ) -> tuple:
        """
        Calculate score delta and new values.

        Returns:
            Tuple of (score_delta, new_score, new_uses, success_delta)

        v0.3.1: Flat deltas, no time-weight (matches core v0.3.6+).
        - worked: +0.2 raw, +1.0 success, +1 use
        - partial: +0.05 raw, +0.5 success, +1 use
        - failed: -0.3 raw, +0.0 success, +1 use
        - unknown: -0.05 raw, +0.25 success, +1 use
        """
        if outcome == "worked":
            score_delta = 0.2
            new_score = min(1.0, current_score + score_delta)
            uses += 1
            success_delta = 1.0
        elif outcome == "failed":
            score_delta = -0.3
            new_score = max(0.0, current_score + score_delta)
            uses += 1
            success_delta = 0.0
        elif outcome == "partial":
            score_delta = 0.05
            new_score = min(1.0, current_score + score_delta)
            uses += 1
            success_delta = 0.5
        elif outcome == "unknown":
            # v0.3.1: -0.05 raw (matches core v0.2.9+). Surfaced but unused = weak negative.
            score_delta = -0.05
            new_score = max(0.0, current_score + score_delta)
            uses += 1
            success_delta = 0.25
        else:
            # Guard - invalid outcomes don't affect score
            logger.warning(f"Unexpected outcome '{outcome}' - no score change")
            return 0.0, current_score, uses, 0.0

        return score_delta, new_score, uses, success_delta

    def count_successes_from_history(self, outcome_history_json: str) -> float:
        """
        Count successes from outcome history JSON.

        Args:
            outcome_history_json: JSON string of outcome history

        Returns:
            Weighted success count (worked=1, partial=0.5, unknown=0.25)
        """
        if not outcome_history_json or outcome_history_json == "[]":
            return 0

        try:
            history = json.loads(outcome_history_json)
            successes = 0.0
            for entry in history:
                outcome = entry.get("outcome", "")
                if outcome == "worked":
                    successes += 1.0
                elif outcome == "partial":
                    successes += 0.5
                elif outcome == "unknown":
                    successes += 0.25  # v0.3.0: weak negative signal
            return successes
        except json.JSONDecodeError:
            return 0

    def get_outcome_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get outcome statistics for a document.

        Args:
            doc_id: Document ID

        Returns:
            Dict with outcome stats
        """
        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                doc = adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    outcome_history = json.loads(metadata.get("outcome_history", "[]"))

                    worked = sum(1 for o in outcome_history if o.get("outcome") == "worked")
                    failed = sum(1 for o in outcome_history if o.get("outcome") == "failed")
                    partial = sum(1 for o in outcome_history if o.get("outcome") == "partial")

                    return {
                        "doc_id": doc_id,
                        "collection": coll_name,
                        "score": metadata.get("score", 0.5),
                        "uses": metadata.get("uses", 0),
                        "last_outcome": metadata.get("last_outcome"),
                        "outcomes": {
                            "worked": worked,
                            "failed": failed,
                            "partial": partial
                        },
                        "total_outcomes": len(outcome_history)
                    }

        return {"doc_id": doc_id, "error": "not_found"}
