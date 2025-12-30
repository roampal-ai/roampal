# backend/modules/embedding/embedding_service.py

import logging
import numpy as np
from typing import List, Dict, Any
import sys
import os
from pathlib import Path
import threading

# Add the backend directory to sys.path if not already there
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.interfaces.embedding_service_interface import EmbeddingServiceInterface

logger = logging.getLogger(__name__)

class EmbeddingService(EmbeddingServiceInterface):
    def __init__(self):
        self._model_name = "paraphrase-multilingual-mpnet-base-v2"
        self._version = "1.4"
        self._embedding_dim = 768
        self.model = None

        # Embedding cache to avoid regenerating identical embeddings
        self._cache = {}  # {text_hash: embedding}
        self._max_cache_size = 200  # Limit memory usage

        logger.info(f"EmbeddingService initialized (model will load on first use): {self._model_name}")

    def _load_bundled_model(self):
        """Load the bundled paraphrase-multilingual-mpnet-base-v2 model with timeout protection."""
        try:
            from sentence_transformers import SentenceTransformer

            # Path from embedding_service.py: embedding/ -> modules/ -> backend/ -> release/ -> binaries/
            bundled_cache = Path(__file__).parent.parent.parent.parent / "binaries" / "models" / "paraphrase-multilingual-mpnet-base-v2"

            model_path = None
            if bundled_cache.exists():
                # Read the snapshot ID from refs/main
                ref_file = bundled_cache / "refs" / "main"
                if ref_file.exists():
                    snapshot_id = ref_file.read_text().strip()
                    snapshot_path = bundled_cache / "snapshots" / snapshot_id

                    if snapshot_path.exists():
                        logger.info(f"Loading bundled embedding model from snapshot: {snapshot_path}")
                        model_path = str(snapshot_path)
                    else:
                        logger.warning(f"Snapshot path not found: {snapshot_path}")

            if model_path is None:
                # Fallback to download (development mode only)
                logger.warning("Bundled model not found, downloading from HuggingFace (dev mode)")
                model_path = 'paraphrase-multilingual-mpnet-base-v2'

            # Load model with timeout protection to prevent deadlocks
            result = [None]
            error = [None]

            def load_model():
                try:
                    result[0] = SentenceTransformer(model_path)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
            thread.join(timeout=120)  # 2 minute timeout for model loading

            if thread.is_alive():
                logger.error("Model loading timed out after 120 seconds")
                raise RuntimeError("Embedding model loading timed out")

            if error[0]:
                raise error[0]

            self.model = result[0]
            logger.info(f"Embedding model loaded successfully: {self._model_name}")

        except ImportError as e:
            logger.error(f"Cannot import sentence_transformers: {e}")
            raise RuntimeError("sentence_transformers is required for embeddings") from e
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model being used."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings produced by this service."""
        return self._embedding_dim

    @property
    def version(self) -> str:
        """Get the version of the embedding service."""
        return self._version

    def get_embedding_metadata(self) -> Dict[str, Any]:
        """Get metadata about the embedding service and model."""
        return {
            "model_name": self._model_name,
            "version": self._version,
            "embedding_dim": self._embedding_dim,
            "bundled": True
        }

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector representation with caching."""
        if not isinstance(text, str) or not text.strip():
            logger.warning("Attempted to embed empty or non-string text. Returning zero vector.")
            return [0.0] * self._embedding_dim

        # Lazy load model on first use
        if self.model is None:
            self._load_bundled_model()

        # Check cache first (using hash to save memory)
        import hashlib
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        if cache_key in self._cache:
            logger.debug(f"Embedding cache HIT for: {text[:50]}...")
            return self._cache[cache_key]

        # Generate embedding using bundled model
        try:
            # Truncate text to avoid token length issues (roughly 400 tokens = 2000 chars)
            if len(text) > 2000:
                text = text[:2000]
                logger.warning(f"Truncated text to 2000 characters for embedding")

            # Add timeout protection to prevent indefinite hangs
            result = [None]
            error = [None]

            def encode_with_timeout():
                try:
                    result[0] = self.model.encode(text).tolist()
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=encode_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=30)  # 30 second timeout

            if thread.is_alive():
                logger.error("Embedding generation timed out after 30 seconds, returning zero vector")
                return [0.0] * self._embedding_dim

            if error[0]:
                raise error[0]

            embedding = result[0]

            # Verify dimension (should be 768 natively)
            if len(embedding) != self._embedding_dim:
                logger.warning(f"Dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}. Padding/trimming.")
                embedding = (embedding + [0.0] * (self._embedding_dim - len(embedding)))[:self._embedding_dim]

            # Store in cache (FIFO eviction if at capacity)
            if len(self._cache) >= self._max_cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            return [0.0] * self._embedding_dim

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings into vector representations."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    async def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding has the correct format and dimension."""
        if not isinstance(embedding, list):
            return False
        if len(embedding) != self._embedding_dim:
            return False
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
        return True

    async def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate the similarity between two embeddings using cosine similarity."""
        try:
            # Convert to numpy arrays for calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

