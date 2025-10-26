# backend/core/interfaces/book_processor_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal

class BookProcessorInterface(ABC):
    """
    Interface for services that process and retrieve information from books.
    """

    @abstractmethod
    async def retrieve_from_ingested_books(
        self,
        query_text: str,
        book_ids: Optional[List[str]] = None,
        content_types_to_search: Optional[List[Literal["chunks", "chunk_summaries", "full_summary", "models", "quotes"]]] = None,
        max_results_per_book: int = 3,
        max_total_results: int = 10,
        min_keyword_match_percent: float = 0.3 
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant snippets from ingested books based on a query.
        Each snippet dictionary should ideally contain:
        - book_id: str
        - book_title: str
        - chapter_title: Optional[str]
        - chunk_id: Optional[str]
        - source_type: str (e.g., "chunk_text", "chunk_summary", "model", "quote")
        - text_snippet: str
        - matched_keywords: List[str]
        - relevance_score: float
        """
        pass

    @abstractmethod
    async def list_books(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all books in the registry with their metadata.
        Useful for OG commands or allowing users to see the available library.
        """
        pass

    @abstractmethod
    async def _get_book_metadata(self, book_id: str) -> Optional[Dict[str, Any]]:
        """
        Internal helper to get metadata for a specific book.
        May be used by other interface methods if they need detailed book info.
        Marked as protected ('_') suggesting primary use within implementing classes
        or closely related services, but exposed via interface for potential specific needs.
        """
        pass

    # Add other methods from the concrete BookProcessor here IF OGChatService
    # (or other services depending on this interface) needs to call them directly.
    # For now, retrieval and listing are the primary interactions anticipated from outside.
