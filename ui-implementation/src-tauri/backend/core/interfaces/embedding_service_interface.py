"""
Embedding Service Interface for RoampalAI

This interface defines the contract for embedding services that convert text to vector representations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class EmbeddingServiceInterface(ABC):
    """
    Interface for embedding services that convert text to vector representations.
    
    This interface provides:
    - Text embedding functionality
    - Model metadata access
    - Embedding dimension information
    - Service configuration
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the embedding model being used."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings produced by this service."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get the version of the embedding service."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector representation.
        
        Args:
            text: The text string to embed
            
        Returns:
            List of floats representing the text embedding
            
        Raises:
            Exception: If embedding fails
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vector representations.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings, one for each input text
            
        Raises:
            Exception: If embedding fails
        """
        pass
    
    @abstractmethod
    def get_embedding_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the embedding service and model.
        
        Returns:
            Dictionary containing metadata about the embedding service
        """
        pass
    
    @abstractmethod
    async def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct format and dimension.
        
        Args:
            embedding: The embedding to validate
            
        Returns:
            True if the embedding is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate the similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        pass 