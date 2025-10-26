# backend/core/interfaces/llm_client_interface.py
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from abc import ABC, abstractmethod


class LLMClientInterface(ABC):
    """
    Abstract Base Class for Large Language Model clients.
    Defines the common interface for interacting with different LLM backends
    (e.g., Ollama, OpenAI, Claude, OpenChat).
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initializes the LLM client with necessary backend-specific configuration.
        This method should be called before any other operations.

        Args:
            config: A dictionary containing settings required by the specific LLM client,
                    such as API keys, base URLs, model names, timeouts, etc.
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        format: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generates a single, complete text response from the LLM.

        Args:
            prompt: The user's input or the fully constructed prompt to send to the LLM.
            history: (Optional) A list of previous conversation turns.
            format: (Optional) The desired response format (e.g., "json").
            model: (Optional) The model to use (for multi-model support).
            system_prompt: (Optional) A special system message for behavior shaping.

        Returns:
            The LLM's generated response as a string.
        """
        pass

    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams the LLM's response token by token as an asynchronous generator.

        Args:
            prompt: The user's input or the fully constructed prompt.
            history: (Optional) A list of previous conversation turns.
            model: (Optional) The model to use.
            system_prompt: (Optional) A system message to inject before chat begins.

        Yields:
            Strings, where each string is a token or a small chunk of the LLM's response.
        """
        if False:
            yield

    @abstractmethod
    async def close(self) -> None:
        """
        Performs any cleanup and closes connections or releases resources
        held by the LLM client (e.g., HTTP client sessions).
        """
        pass
