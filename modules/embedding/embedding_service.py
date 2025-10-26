# backend/modules/embedding/embedding_service.py

import logging
import json
import numpy as np
from typing import List, Optional, Dict, Any
import httpx
import sys
import os

# Add the backend directory to sys.path if not already there
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from config.settings import settings
from sentence_transformers import SentenceTransformer  # For HF fallback
from core.interfaces.embedding_service_interface import EmbeddingServiceInterface

logger = logging.getLogger(__name__)

class EmbeddingService(EmbeddingServiceInterface):
    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or "nomic-embed-text"
        self._version = "1.3"
        self.ollama_base_url = settings.llm.ollama_base_url
        self._embedding_dim = 768  # Ollama default, will be updated to 384 if using HF fallback
        self.use_ollama = True
        self.hf_model = None
        self.client = httpx.AsyncClient(base_url=self.ollama_base_url)

        # Embedding cache to avoid regenerating identical embeddings
        self._cache = {}  # {text_hash: embedding}
        self._max_cache_size = 200  # Limit memory usage

        logger.info(f"EmbeddingService created using base LLM: {self._model_name}")

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
            "use_ollama": self.use_ollama
        }

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector representation with caching."""
        if not isinstance(text, str) or not text.strip():
            logger.warning("Attempted to embed empty or non-string text. Returning zero vector.")
            return [0.0] * self._embedding_dim

        # Check cache first (using hash to save memory)
        import hashlib
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        if cache_key in self._cache:
            logger.debug(f"Embedding cache HIT for: {text[:50]}...")
            return self._cache[cache_key]

        # Cache miss - generate embedding
        if self.use_ollama:
            try:
                response = await self.client.post(
                    "/api/embeddings",
                    json={"model": self._model_name, "prompt": text},
                    timeout=settings.llm.ollama_request_timeout_seconds
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                if not isinstance(embedding, list):
                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                if len(embedding) != self._embedding_dim:
                    logger.warning(f"Dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}. Padding/trimming.")
                    embedding = (embedding + [0.0] * (self._embedding_dim - len(embedding)))[:self._embedding_dim]

                # Store in cache (LRU-style: remove oldest if at capacity)
                if len(self._cache) >= self._max_cache_size:
                    # Simple FIFO eviction
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = embedding

                return embedding
            except httpx.HTTPStatusError as http_err:
                if http_err.response.status_code == 404:
                    logger.warning("Ollama embeddings endpoint not found; disabling Ollama and switching to HF fallback.")
                    self.use_ollama = False
                else:
                    logger.error(f"HTTP error in embed_text: {http_err}; falling back to HF.")
            except Exception as e:
                logger.error(f"Error generating embedding with Ollama: {e}", exc_info=True)
                logger.warning("Falling back to HF due to Ollama failure.")

        # Fallback to HuggingFace
        embedding = await self._hf_embed(text)

        # Cache the HF result too
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = embedding

        return embedding

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

    async def _hf_embed(self, text: str) -> List[float]:
        """Fallback embedding using HuggingFace SentenceTransformer."""
        try:
            if self.hf_model is None:
                # Use a multilingual model with longer context support
                self.hf_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Keep 768 dimensions to match Ollama - we'll pad the 384-dim output
            
            # Truncate text to avoid token length issues (roughly 400 tokens = 2000 chars)
            if len(text) > 2000:
                text = text[:2000]
                logger.warning(f"Truncated text to 2000 characters for embedding")
            
            embedding = self.hf_model.encode(text).tolist()
            
            # Pad HuggingFace 384-dim embeddings to 768-dim to match Ollama
            if len(embedding) == 384:
                # Pad with zeros to reach 768 dimensions
                embedding = embedding + [0.0] * (768 - 384)
                logger.debug(f"Padded HF embedding from 384 to 768 dimensions")
            elif len(embedding) != self._embedding_dim:
                logger.warning(f"HF embedding dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}. Padding/trimming.")
                embedding = (embedding + [0.0] * (self._embedding_dim - len(embedding)))[:self._embedding_dim]
            
            return embedding
        except Exception as e:
            logger.error(f"HF fallback embedding failed: {e}")
            return [0.0] * self._embedding_dim

    async def _chat_based_embed(self, text: str) -> List[float]:
        """Fallback embedding using chat-based approach."""
        try:
            prompt = (
                f"Generate a numerical embedding vector (list of {self._embedding_dim} floats) "
                f"for this text: {text}. Output ONLY a JSON array of floats, e.g., [0.1, 0.2, ...]."
            )
            payload = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                embedding_str = response_data["message"]["content"]
                embedding = json.loads(embedding_str)
                if (
                    isinstance(embedding, list)
                    and len(embedding) == self._embedding_dim
                    and all(isinstance(e, (int, float)) for e in embedding)
                ):
                    return embedding
            logger.warning("Chat fallback embedding invalid; returning zeros.")
            return [0.0] * self._embedding_dim
        except Exception as e:
            logger.error(f"Chat fallback embedding failed: {e}")
            return [0.0] * self._embedding_dim
