# backend/core/interfaces/vector_db_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDBInterface(ABC):
    """
    Interface for a vector database client, handling storage and retrieval of text embeddings.
    """
    @abstractmethod
    async def initialize(self, collection_name: str, embedding_model_name: str):
        """Initializes the database and ensures a collection exists."""
        pass

    @abstractmethod
    async def upsert_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Adds or updates vectors in the database."""
        pass

    @abstractmethod
    async def query_vectors(self, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Queries the database for the most similar vectors."""
        pass
