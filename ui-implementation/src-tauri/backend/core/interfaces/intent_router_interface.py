# C:\RoampalAI\backend\core\interfaces\intent_router_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.types.common_types import Action, Message  # Updated to use correct relative path

class IntentRouterInterface(ABC):
    """
    Abstract Base Class for an intent router.
    Determines the user's intent and the corresponding action Roampal should take.
    """

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the intent router.
        This might involve loading rules, models, or other configurations.

        Args:
            config: (Optional) A dictionary for router-specific configurations.
        """
        pass

    @abstractmethod
    async def determine_action(
        self,
        user_input: str,
        conversation_history: List[Message],  # Changed to List[Message]
        available_tools: Optional[List[str]] = None  # For future tool use integration
    ) -> Action:  # Return type is now the Action TypedDict
        """
        Determines the primary action Roampal should take based on the input.

        Args:
            user_input: The latest input from the user.
            conversation_history: The history of the current conversation.
            available_tools: (Optional) A list of tools Roampal has access to.

        Returns:
            An Action object specifying the type of action and any parameters.
        """
        pass