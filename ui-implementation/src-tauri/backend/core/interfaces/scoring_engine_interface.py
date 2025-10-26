# backend/core/interfaces/scoring_engine_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ScoringStrategyInterface(ABC):
    """
    Interface for different scoring strategies.
    Each strategy will implement how scores are calculated, stored, and retrieved.
    """
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the scoring strategy with its configuration."""
        pass

    @abstractmethod
    async def record_and_score_interaction(
        self, 
        interaction_type: str, 
        success: bool, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Records an interaction and updates the relevant scores based on the strategy.
        'interaction_type' could be an enum member's value or a descriptive string.
        'details' can contain any relevant context, like length of input/output, specific entities, etc.
        """
        pass

    @abstractmethod
    async def get_current_score(self, interaction_type: str) -> Optional[float]:
        """Retrieves the current score for a specific interaction type."""
        pass

    @abstractmethod
    async def get_all_scores(self) -> Dict[str, float]:
        """Retrieves all current scores managed by the strategy."""
        pass
    
    @abstractmethod
    async def persist_scores(self) -> None:
        """Persists the current scores to a file or database if the strategy requires it."""
        pass


class ScoringEngineInterface(ABC):
    """
    Interface for the main ScoringEngine.
    The engine uses a specific ScoringStrategyInterface implementation.
    """
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the ScoringEngine, including loading its configured strategy.
        The 'config' dictionary should contain a 'strategy' key indicating which
        strategy to load (e.g., "basic") and any strategy-specific settings.
        """
        pass

    @abstractmethod
    async def record_interaction(
        self, 
        interaction_type: str, 
        success: bool, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Public method to record an interaction. This will be delegated to the active strategy.
        """
        pass

    @abstractmethod
    async def get_score(self, interaction_type: str) -> Optional[float]:
        """Retrieves the current score for a specific interaction type from the active strategy."""
        pass

    @abstractmethod
    async def get_all_scores(self) -> Dict[str, float]:
        """Retrieves all current scores from the active strategy."""
        pass

    @abstractmethod
    async def persist_scores(self) -> None:
        """Commands the active strategy to persist its scores."""
        pass
