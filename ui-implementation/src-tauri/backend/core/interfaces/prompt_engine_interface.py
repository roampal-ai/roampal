# backend/core/interfaces/prompt_engine_interface.py
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

# We'll likely define more specific Pydantic models for context elements
# in core/types/common_types.py later (e.g., UserProfile, ToolDescription, WebSearchResult)
# For now, Dict[str, Any] offers flexibility for the context.

class PromptContext(Dict[str, Any]):
    """
    A specialized dictionary to hold various pieces of information
    that can be injected into a prompt. This provides a clear type hint
    for what the prompt engine expects.
    Examples of keys: "user_input", "current_date", "goals", "values",
                      "conversation_history", "web_search_results",
                      "available_tools", "persona_specific_data".
    """
    pass


class PromptEngineInterface(ABC):
    """
    Abstract Base Class for a prompt engineering system.
    Defines how various pieces of information are assembled into a final
    prompt string to be sent to an LLM.
    """

    @abstractmethod
    async def initialize(
        self,
        template_directory: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes the prompt engine.
        This might involve loading base templates, persona configurations,
        or other settings.

        Args:
            template_directory: The file system path to the directory where
                                prompt templates are stored.
            config: (Optional) A dictionary for any other engine-specific
                    configurations.
        """
        pass

    @abstractmethod
    async def build_prompt(
        self,
        user_input: str,
        context: PromptContext,
        system_prompt_name: Optional[str] = "default_system",
        # We could add more granular control if needed, e.g.,
        # user_prompt_template_name: Optional[str] = "default_user_wrapper"
    ) -> str:
        """
        Constructs the final prompt string to be sent to the LLM.

        This method will typically:
        1. Load a base system prompt template (identified by system_prompt_name).
        2. Inject relevant data from the `context` dictionary into the system prompt.
           This context can include things like Roampal's goals, values,
           current date, relevant memories, web search results, etc.
        3. Incorporate the `user_input` appropriately, possibly wrapping it or
           placing it within a larger conversational structure defined by the templates.

        Args:
            user_input: The raw input string from the user.
            context: A PromptContext object containing all dynamic data to be
                     potentially included in the prompt.
            system_prompt_name: (Optional) An identifier for the specific system
                                prompt template to use (e.g., "default_system",
                                "coding_persona_system"). Defaults to "default_system".

        Returns:
            The fully constructed prompt string.
        """
        pass

    # Optional: Add methods for managing prompt fragments or personas if needed directly
    # in the interface, though much of this can be handled by how `build_prompt`
    # uses the `context` and `system_prompt_name`.
    #
    # @abstractmethod
    # async def list_available_personas(self) -> List[str]:
    #     pass
    #
    # @abstractmethod
    # async def get_persona_details(self, persona_name: str) -> Optional[Dict[str, Any]]:
    #     pass
