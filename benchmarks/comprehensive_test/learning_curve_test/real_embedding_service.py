"""
Real Embedding Service using sentence-transformers
Provides actual semantic similarity for statistical testing
"""

from sentence_transformers import SentenceTransformer
from typing import List


class RealEmbeddingService:
    """Generate real semantic embeddings using sentence-transformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
                       - Fast, good quality, 384 dimensions
                       - Alternative: 'all-mpnet-base-v2' (768d, slower, better)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate semantic embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # sentence-transformers uses synchronous encoding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()
