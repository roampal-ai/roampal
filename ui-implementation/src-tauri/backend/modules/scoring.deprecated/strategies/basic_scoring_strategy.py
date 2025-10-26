import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from backend.core.interfaces.scoring_engine_interface import ScoringStrategyInterface

logger = logging.getLogger(__name__)

class ScoringStrategyException(Exception):
    """Custom exception for scoring strategy errors."""
    pass

class BasicScoringStrategy(ScoringStrategyInterface):
    def __init__(self):
        self.scores: Dict[str, Dict[str, Any]] = {}
        self.fragments: Dict[str, Dict[str, Any]] = {}
        self.scores_filepath: Optional[Path] = None
        self.initialized = False
        logger.debug("BasicScoringStrategy instance created.")

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.initialized:
            logger.info("BasicScoringStrategy already initialized.")
            return

        if config:
            filepath_str = config.get("data_path")
            if filepath_str:
                self.scores_filepath = Path(filepath_str).resolve()
                logger.info(f"BasicScoringStrategy scores file path set to: {self.scores_filepath}")
                await self._load_scores()
            else:
                logger.error("scores_filepath (as 'data_path') not provided in BasicScoringStrategy config.")
                raise ScoringStrategyException(
                    "scores_filepath (as 'data_path') must be provided in BasicScoringStrategy config."
                )
        else:
            logger.error("Config not provided to BasicScoringStrategy initialize method.")
            raise ScoringStrategyException("Config must be provided to BasicScoringStrategy.")

        self.initialized = True
        logger.info(f"BasicScoringStrategy initialized. Scores file: {self.scores_filepath}")

    async def _load_scores(self) -> None:
        if self.scores_filepath and self.scores_filepath.exists():
            try:
                loaded_data = json.loads(self.scores_filepath.read_text(encoding="utf-8"))
                if isinstance(loaded_data, dict):
                    self.scores = loaded_data.get("actions", {}) if "actions" in loaded_data else loaded_data
                    self.fragments = loaded_data.get("fragments", {})
                    logger.info(f"Loading scores from {self.scores_filepath}")
                else:
                    logger.warning(
                        f"Scores file {self.scores_filepath} has incorrect format. Starting with empty scores."
                    )
                    self.scores = {}
                    self.fragments = {}
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from scores file {self.scores_filepath}. "
                    "Starting with empty scores.", exc_info=True
                )
                self.scores = {}
                self.fragments = {}
            except Exception as e:
                logger.error(
                    f"Error loading scores from {self.scores_filepath}: {e}. "
                    "Starting with empty scores.", exc_info=True
                )
                self.scores = {}
                self.fragments = {}
        else:
            logger.info(f"Scores file {self.scores_filepath} not found or path is None. Starting with empty scores.")
            self.scores = {}
            self.fragments = {}

    async def _save_scores(self) -> None:
        if self.scores_filepath:
            try:
                self.scores_filepath.parent.mkdir(parents=True, exist_ok=True)
                # Save both action and fragment scores, backward compatible
                save_obj = {"actions": self.scores, "fragments": self.fragments}
                self.scores_filepath.write_text(json.dumps(save_obj, indent=2), encoding="utf-8")
                logger.info(f"Scores snapshot persisted to {self.scores_filepath}")
            except Exception as e:
                logger.error(f"Error saving scores to {self.scores_filepath}: {e}", exc_info=True)
        else:
            logger.warning("Scores filepath not set. Cannot persist scores.")

    async def record_and_score_interaction(
        self,
        interaction_type: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.initialized:
            logger.warning("BasicScoringStrategy not initialized. Cannot record interaction.")
            return

        # Action-based scoring (existing)
        logger.info(f"Recording interaction: type='{interaction_type}', success={success}")
        if interaction_type not in self.scores:
            self.scores[interaction_type] = {"success": 0, "failure": 0, "total": 0, "rate": 0.0}

        self.scores[interaction_type]["total"] = self.scores[interaction_type].get("total", 0) + 1
        if success:
            self.scores[interaction_type]["success"] = self.scores[interaction_type].get("success", 0) + 1
        else:
            self.scores[interaction_type]["failure"] = self.scores[interaction_type].get("failure", 0) + 1

        current_successes = self.scores[interaction_type].get("success", 0)
        current_total = self.scores[interaction_type].get("total", 0)
        if current_total > 0:
            self.scores[interaction_type]["rate"] = round(current_successes / current_total, 4)
        else:
            self.scores[interaction_type]["rate"] = 0.0

        # --- NEW: Fragment scoring (per-neuron)
        if details:
            fragment_ids = []
            if "fragment_id" in details:
                fragment_ids = [details["fragment_id"]]
            elif "fragment_ids" in details:
                fragment_ids = details["fragment_ids"]
            elif "soul_items" in details and isinstance(details["soul_items"], list):
                fragment_ids = [item.get("id") for item in details["soul_items"] if "id" in item]

            for fragment_id in fragment_ids:
                if not fragment_id:
                    continue
                if fragment_id not in self.fragments:
                    self.fragments[fragment_id] = {"success": 0, "failure": 0, "total": 0, "rate": 0.0}
                self.fragments[fragment_id]["total"] = self.fragments[fragment_id].get("total", 0) + 1
                if success:
                    self.fragments[fragment_id]["success"] = self.fragments[fragment_id].get("success", 0) + 1
                else:
                    self.fragments[fragment_id]["failure"] = self.fragments[fragment_id].get("failure", 0) + 1
                cs = self.fragments[fragment_id].get("success", 0)
                ct = self.fragments[fragment_id].get("total", 0)
                self.fragments[fragment_id]["rate"] = round(cs / ct, 4) if ct > 0 else 0.0
        # Note: we do NOT auto-save here. Call `persist_scores()` when you want to write out.

    async def get_current_score(self, interaction_type: str) -> Optional[float]:
        if not self.initialized:
            return None
        return self.scores.get(interaction_type, {}).get("rate")

    async def get_all_scores(self) -> Dict[str, float]:
        if not self.initialized:
            return {}
        # Combine both action rates and fragment rates
        all_scores = {k: v.get("rate", 0.0) for k, v in self.scores.items() if isinstance(v, dict)}
        all_scores.update({f"fragment:{k}": v.get("rate", 0.0) for k, v in self.fragments.items() if isinstance(v, dict)})
        return all_scores

    async def persist_scores(self) -> None:
        if not self.initialized:
            logger.warning("BasicScoringStrategy not initialized. Cannot persist scores.")
            return
        await self._save_scores()

    # ──────────────────────────────────────────────────────────────────────────
    async def score(
        self,
        query_text: str,
        soul_items: List[Dict[str, Any]],
        *,
        max_items: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self.initialized:
            logger.warning("BasicScoringStrategy not initialized. Returning empty score list.")
            return []
        # Placeholder: return [] until you implement real scoring logic
        return []
    # ──────────────────────────────────────────────────────────────────────────
