"""
Real Embedding Service using sentence-transformers.

This wrapper provides embeddings for benchmark tests that need
to directly compare vector similarity without going through
the full UnifiedMemorySystem embedding pipeline.
"""

import sys
from pathlib import Path

# Add backend to path to use the actual embedding service
backend_path = Path(__file__).parent.parent.parent.parent / "ui-implementation" / "src-tauri" / "backend"
sys.path.insert(0, str(backend_path))

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class RealEmbeddingService:
    """
    Embedding service that uses the same model as the production system.
    Uses paraphrase-multilingual-mpnet-base-v2 for consistency.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """Initialize with the specified model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        # Check for bundled model first (like production does)
        # The bundled model is in HuggingFace cache format with snapshots
        models_dir = Path(__file__).parent.parent.parent.parent / "ui-implementation" / "src-tauri" / "binaries" / "models"
        bundled_path = models_dir / model_name / "snapshots"

        model_path = None
        if bundled_path.exists():
            # Get the first (and only) snapshot directory
            snapshots = list(bundled_path.iterdir())
            if snapshots:
                model_path = str(snapshots[0])

        if model_path:
            self.model = SentenceTransformer(model_path)
        else:
            # Fall back to downloading from HuggingFace
            self.model = SentenceTransformer(model_name)

        self.model_name = model_name

    def encode(self, texts, **kwargs):
        """
        Encode texts into embeddings.

        Args:
            texts: Single text string or list of strings
            **kwargs: Additional args passed to SentenceTransformer.encode()

        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, **kwargs)

    async def aencode(self, texts, **kwargs):
        """Async wrapper for encode (runs synchronously internally)."""
        return self.encode(texts, **kwargs)

    async def embed_text(self, text: str):
        """
        Embed a single text string.
        Compatible with UnifiedMemorySystem's embedding_service interface.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    async def embed_texts(self, texts):
        """
        Embed multiple texts.
        Compatible with UnifiedMemorySystem's embedding_service interface.

        Args:
            texts: List of text strings

        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return [e.tolist() for e in embeddings]

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
