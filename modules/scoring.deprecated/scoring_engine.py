import logging
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from core.interfaces.scoring_engine_interface import ScoringEngineInterface, ScoringStrategyInterface
from .strategies.basic_scoring_strategy import BasicScoringStrategy

# --- FIXED: Correct import paths for vector DB and embedding service
from backend.modules.memory.chromadb_adapter import ChromaDBAdapter
from backend.modules.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

REINFORCED_FILE = "memory/reinforced_patterns.json"
DECAY_INTERVAL_DAYS = 30  # Decay score if unused after 30 days

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

async def async_store_json(file_path, data):
    _ensure_dir(file_path)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: json.dump(data, open(file_path, "w", encoding="utf-8"), indent=2))

async def async_load_json(file_path):
    if not os.path.exists(file_path):
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: json.load(open(file_path, "r", encoding="utf-8")))

async def store_reinforced_pattern(query: str, response: str, soul_item_ids: list, score: float):
    entry = {
        "query": query,
        "top_response": response,
        "soul_item_ids": soul_item_ids,
        "score": score,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    data = await async_load_json(REINFORCED_FILE)
    data.append(entry)
    await async_store_json(REINFORCED_FILE, data)

async def find_similar_patterns(query: str, min_score: float = 0.7) -> List[Dict[str, Any]]:
    data = await async_load_json(REINFORCED_FILE)
    return [
        entry for entry in data
        if query.lower() in entry["query"].lower() and entry["score"] >= min_score
    ]

class ScoringEngine(ScoringEngineInterface):
    def __init__(
        self,
        vector_adapter: Optional[ChromaDBAdapter] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.strategy: Optional[ScoringStrategyInterface] = None
        self.config: Dict[str, Any] = {}
        self.initialized: bool = False
        self.vector_adapter = vector_adapter
        self.embedding_service = embedding_service
        logger.debug("ScoringEngine instance created (uninitialized).")

    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        if self.initialized:
            return
        self.config = config or {}
        strategy_name = self.config.get("strategy")
        if not strategy_name:
            raise ValueError("ScoringEngine requires a 'strategy' name in config.")

        if strategy_name.lower() == "basic":
            self.strategy = BasicScoringStrategy()
        elif strategy_name.lower() == "none":
            self.strategy = None
        else:
            raise ValueError(f"Unknown scoring strategy: '{strategy_name}'")

        if self.strategy:
            await self.strategy.initialize(self.config)

        self.initialized = True
        logger.info(f"ScoringEngine initialized with strategy: {strategy_name if self.strategy else 'none'}")

    async def record_interaction(self, interaction_type: str, success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        if not self.initialized or not self.strategy:
            return
        await self.strategy.record_and_score_interaction(interaction_type, success, details)

        # Patch: Score fragments directly in ChromaDB
        if details and self.vector_adapter and "fragment_id" in details:
            frag_id = details.get("fragment_id")
            score = details.get("score")
            if frag_id and score is not None:
                self.vector_adapter.update_fragment_score(frag_id, score)

        # Reinforce high-scoring completions (score threshold adjustable)
        if details and details.get("score", 0) >= 0.8:
            query = details.get("query", "")
            response = details.get("response", "")
            soul_item_ids = details.get("soul_item_ids", [])
            await store_reinforced_pattern(query, response, soul_item_ids, details["score"])

    async def decay_fragments(self):
        """Decay score of unused fragments in the vector DB (if integrated)."""
        if not self.vector_adapter:
            logger.warning("Vector DB adapter not set. Skipping decay.")
            return

        cutoff = datetime.utcnow() - timedelta(days=DECAY_INTERVAL_DAYS)
        fragments = self.vector_adapter.get_all_fragments()
        decayed = 0
        for frag in fragments:
            last_used = frag.get("last_used")
            try:
                if last_used and datetime.fromisoformat(str(last_used).rstrip("Z")) < cutoff:
                    old_score = frag.get("score", 1.0)
                    new_score = max(0, old_score * 0.9)
                    self.vector_adapter.update_fragment_score(frag["id"], new_score)
                    decayed += 1
            except Exception as e:
                logger.error(f"Error decaying fragment {frag['id']}: {e}", exc_info=True)
        logger.info(f"Decayed {decayed} fragments not used since {cutoff.date()}.")

    async def reinforce_fragment(self, fragment_id: str, boost: float = 0.2):
        """Boost score for a fragment after positive interaction."""
        if not self.vector_adapter:
            logger.warning("Vector DB adapter not set. Skipping reinforcement.")
            return
        frag = self.vector_adapter.get_fragment(fragment_id)
        if frag:
            old_score = frag.get("metadata", {}).get("score", 0.0)
            new_score = min(1.0, old_score + boost)
            self.vector_adapter.update_fragment_score(fragment_id, new_score)
            logger.info(f"Fragment {fragment_id} reinforced. New score: {new_score}")

    async def penalize_fragment(self, fragment_id: str, penalty: float = 0.05):
        """Penalize a fragment for a failed/contradicted use."""
        if not self.vector_adapter:
            logger.warning("Vector DB adapter not set. Skipping penalty.")
            return
        frag = self.vector_adapter.get_fragment(fragment_id)
        if frag:
            old_score = frag.get("metadata", {}).get("score", 0.0)
            new_score = max(0, old_score - penalty)
            self.vector_adapter.update_fragment_score(fragment_id, new_score)
            logger.info(f"Fragment {fragment_id} penalized. New score: {new_score}")

    async def get_score(self, fragment_id: str) -> Optional[float]:
        """Get the current score for a fragment."""
        if not self.vector_adapter:
            return None
        frag = self.vector_adapter.get_fragment(fragment_id)
        if frag:
            return frag.get("metadata", {}).get("score", 0.0)
        return None

    async def get_all_scores(self) -> Dict[str, float]:
        """Return mapping {fragment_id: score} for all fragments."""
        if not self.vector_adapter:
            return {}
        fragments = self.vector_adapter.get_all_fragments()
        return {frag["id"]: frag.get("score", 0.0) for frag in fragments}

    async def persist_scores(self) -> None:
        if self.strategy and hasattr(self.strategy, "persist_scores"):
            await self.strategy.persist_scores()

    async def score(self, query_text: str, soul_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.initialized or not self.strategy:
            return []
        try:
            scored = await self.strategy.score(query_text, soul_items)
            # Patch: Persist new scores for fragments in ChromaDB
            if self.vector_adapter:
                for item in scored:
                    frag_id = item.get("id")
                    score = item.get("score")
                    if frag_id and score is not None:
                        self.vector_adapter.update_fragment_score(frag_id, score)
            return scored
        except Exception as e:
            logger.error(f"ScoringEngine.score: Error while scoring: {e}", exc_info=True)
            return []

    async def select_top_n(self, scored_items: List[Dict[str, Any]], n: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.initialized or not self.strategy:
            return []
        return self.strategy.select_top_n(scored_items, n or 3)

    async def record_injection(self, chunk_id: str, injected_tokens: int, score: float) -> None:
        if self.strategy and hasattr(self.strategy, "record_injection"):
            await self.strategy.record_injection(chunk_id, injected_tokens, score)
        # Patch: Also write updated score for chunk
        if self.vector_adapter:
            self.vector_adapter.update_fragment_score(chunk_id, score)

    async def batch_score_fragments(self, query_texts: List[str], soul_items_batch: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Batch scoring for book ingest or mass update."""
        if not self.initialized or not self.strategy:
            return []
        result = []
        for query, items in zip(query_texts, soul_items_batch):
            result.append(await self.score(query, items))
        return result

    async def update_fragment_usage(self, fragment_id: str):
        """Update the last_used timestamp of a fragment."""
        if self.vector_adapter:
            now_str = datetime.utcnow().isoformat() + "Z"
            self.vector_adapter.update_fragment_metadata(fragment_id, {"last_used": now_str})

    async def retrieve_fragment(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a fragment by ID (for auditing or live display)."""
        if self.vector_adapter:
            return self.vector_adapter.get_fragment(fragment_id)
        return None

    # Optional: Expose retrieval for reinforced patterns (could be called elsewhere)
    async def retrieve_reinforced_patterns(self, query: str, min_score: float = 0.7) -> List[Dict[str, Any]]:
        return await find_similar_patterns(query, min_score)
