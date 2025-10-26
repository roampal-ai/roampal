"""
Unified Embedding Configuration
Ensures consistent embeddings across all services
"""

from typing import Optional
import os

class EmbeddingConfig:
    """
    Single source of truth for embedding configuration.
    Uses sentence-transformers/all-MiniLM-L6-v2 as the universal model.
    """
    
    # Primary embedding model - consistent across all services
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Embedding dimensions (384 for MiniLM-L6)
    EMBEDDING_DIM = 384
    
    # Service configuration
    USE_EMBEDDING_SERVICE = True  # Always use port 8004 service when available
    EMBEDDING_SERVICE_URL = "http://localhost:8004"
    
    # Cache configuration
    CACHE_SIZE = 5000
    CACHE_TTL_SECONDS = 1800  # 30 minutes
    
    # Fallback behavior
    ALLOW_FALLBACK = False  # Disable fallbacks to ensure consistency
    
    @classmethod
    def get_embedding_endpoint(cls) -> str:
        """Get the unified embedding endpoint."""
        if cls.USE_EMBEDDING_SERVICE:
            return f"{cls.EMBEDDING_SERVICE_URL}/embed"
        raise ValueError("Embedding service is required but not configured")
    
    @classmethod
    def validate_embedding(cls, embedding: list) -> bool:
        """Validate that an embedding has the correct dimensions."""
        return len(embedding) == cls.EMBEDDING_DIM

# Global instance
embedding_config = EmbeddingConfig()