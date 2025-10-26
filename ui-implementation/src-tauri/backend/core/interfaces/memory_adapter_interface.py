# backend/core/interfaces/memory_adapter_interface.py
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

# We might later define specific Pydantic models in core/types/common_types.py
# for conversation turns or memory entries.
# For now, List[Dict[str, str]] for history and Any for generic data.

class MemoryAdapterInterface(ABC):
    """
    Abstract Base Class for memory storage and retrieval adapters.
    Defines a common interface for various memory backends like flat files,
    SQLite, or vector databases.
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initializes the memory adapter with necessary backend-specific configuration.
        This method should be called before any other operations.

        Args:
            config: A dictionary containing settings required by the specific adapter,
                    such as file paths, database connection strings, etc.
        """
        pass

    @abstractmethod
    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Saves a single turn of a conversation (user input and assistant response).

        Args:
            session_id: A unique identifier for the conversation session.
            user_message: The content of the user's message.
            assistant_message: The content of the assistant's response.
            metadata: (Optional) Additional metadata to store with the conversation turn
                      (e.g., timestamps, interaction IDs, scores).

        Returns:
            True if saving was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def load_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Loads the recent conversation history for a given session.
        The history should be ordered from oldest to newest turn.

        Args:
            session_id: The unique identifier for the conversation session.
            limit: The maximum number of recent conversation turns to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a conversation
            turn (e.g., {"role": "user", "content": "..."} or
            {"role": "assistant", "content": "..."}).
            Returns an empty list if no history is found or an error occurs.
        """
        pass

    @abstractmethod
    async def get_goals(self) -> List[str]:
        """
        Retrieves the list of Roampal's defined goals.

        Returns:
            A list of strings, where each string is a goal.
            Returns an empty list if no goals are found or an error occurs.
        """
        pass

    @abstractmethod
    async def add_goal(self, goal: str) -> bool:
        """
        Adds a new goal to Roampal's list of goals.
        Should avoid adding duplicate goals if possible.

        Args:
            goal: The goal string to add.

        Returns:
            True if the goal was added successfully, False otherwise (e.g., if it's a duplicate
            and duplicates are not allowed, or an error occurred).
        """
        pass

    @abstractmethod
    async def remove_goal(self, goal: str) -> bool:
        """
        Removes a goal from Roampal's list of goals.

        Args:
            goal: The goal string to remove.

        Returns:
            True if the goal was removed successfully, False otherwise (e.g., if the goal
            was not found or an error occurred).
        """
        pass

    @abstractmethod
    async def get_values(self) -> List[str]:
        """
        Retrieves the list of Roampal's defined values.

        Returns:
            A list of strings, where each string is a value.
            Returns an empty list if no values are found or an error occurs.
        """
        pass

    @abstractmethod
    async def add_value(self, value: str) -> bool:
        """
        Adds a new value to Roampal's list of values.
        Should avoid adding duplicate values if possible.

        Args:
            value: The value string to add.

        Returns:
            True if the value was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def remove_value(self, value: str) -> bool:
        """
        Removes a value from Roampal's list of values.

        Args:
            value: The value string to remove.

        Returns:
            True if the value was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]: # The 'Any' here could be a Pydantic model for a generic memory entry
        """
        Searches through stored memories (e.g., past conversations, documents)
        for entries relevant to the query.
        This is more applicable to semantic search in vector databases but can be
        adapted for keyword search in simpler backends.

        Args:
            query: The search query string.
            top_k: The maximum number of relevant memory entries to return.
            filters: (Optional) A dictionary of filters to apply to the search
                     (e.g., filter by session_id, date range, metadata tags).

        Returns:
            A list of memory entries. The structure of each entry depends on
            what is stored and how it's retrieved. For simple backends, this might
            be less relevant or return full conversation turns that match keywords.
        """
        pass

    @abstractmethod
    async def store_arbitrary_data(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Stores an arbitrary piece of data associated with a unique key.
        Useful for miscellaneous persistent information.

        Args:
            key: The unique string key to identify the data.
            data: The data to store (can be any serializable type).
            metadata: (Optional) Additional metadata about the data.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    async def retrieve_arbitrary_data(self, key: str) -> Optional[Any]:
        """
        Retrieves arbitrary data stored with the given key.

        Args:
            key: The unique string key of the data to retrieve.

        Returns:
            The retrieved data, or None if the key is not found or an error occurs.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Performs any cleanup and closes connections or releases resources
        held by the memory adapter (e.g., database connections, file handles).
        This method should be called when the application is shutting down
        or when the adapter is no longer needed.
        """
        pass
