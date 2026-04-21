# backend/modules/embedding/embedding_service.py
"""
Embedding Service — ONNX Runtime backend.

v0.3.1: Replaced sentence-transformers + PyTorch with direct ONNX inference.
Same model (paraphrase-multilingual-mpnet-base-v2), same 768d vectors,
same ChromaDB collections — zero user-facing change. Existing embeddings
stay compatible.
"""

import asyncio
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Add the backend directory to sys.path if not already there
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.interfaces.embedding_service_interface import EmbeddingServiceInterface

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    Tokenizer = None
    hf_hub_download = None
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

# HuggingFace repo for the ONNX-exported model
HF_REPO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ONNX_FILE = "onnx/model_O4.onnx"
TOKENIZER_FILE = "tokenizer.json"
EMBEDDING_DIM = 768


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling — average token embeddings weighted by attention mask."""
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


class EmbeddingService(EmbeddingServiceInterface):
    def __init__(self):
        self._model_name = "paraphrase-multilingual-mpnet-base-v2"
        self._version = "1.5"  # v0.3.1: ONNX backend
        self._embedding_dim = EMBEDDING_DIM
        self._session = None
        self._tokenizer = None

        # Embedding cache to avoid regenerating identical embeddings
        self._cache = {}
        self._max_cache_size = 200

        logger.info(f"EmbeddingService initialized (ONNX, model loads on first use): {self._model_name}")

    def _load_model(self):
        """Download (if needed) and load the ONNX model + tokenizer."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime/tokenizers not installed. "
                "Run: pip install onnxruntime tokenizers huggingface-hub"
            )

        # Check for bundled model first
        bundled_cache = Path(__file__).parent.parent.parent.parent / "binaries" / "models" / self._model_name
        bundled_onnx = None
        bundled_tokenizer = None

        if bundled_cache.exists():
            ref_file = bundled_cache / "refs" / "main"
            if ref_file.exists():
                snapshot_id = ref_file.read_text().strip()
                snapshot_path = bundled_cache / "snapshots" / snapshot_id
                onnx_path = snapshot_path / "onnx" / "model_O4.onnx"
                tok_path = snapshot_path / "tokenizer.json"
                if onnx_path.exists() and tok_path.exists():
                    bundled_onnx = str(onnx_path)
                    bundled_tokenizer = str(tok_path)
                    logger.info(f"Loading bundled ONNX model from: {snapshot_path}")

        if bundled_onnx:
            model_path = bundled_onnx
            tokenizer_path = bundled_tokenizer
        else:
            logger.info(f"Downloading ONNX model: {self._model_name}")
            model_path = hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILE)
            tokenizer_path = hf_hub_download(repo_id=HF_REPO, filename=TOKENIZER_FILE)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 0  # auto-detect

        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=128)

        logger.info(f"Embedding model loaded (ONNX): {self._model_name}")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Tokenize and run ONNX inference, return normalized embeddings."""
        encoded = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        session_inputs = {inp.name for inp in self._session.get_inputs()}
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in session_inputs:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feeds)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        pooled = _mean_pool(token_embeddings, attention_mask.astype(np.float32))
        return _normalize(pooled)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def version(self) -> str:
        return self._version

    def get_embedding_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "version": self._version,
            "embedding_dim": self._embedding_dim,
            "backend": "onnx",
            "bundled": True
        }

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector representation with caching."""
        if not isinstance(text, str) or not text.strip():
            logger.warning("Attempted to embed empty or non-string text. Returning zero vector.")
            return [0.0] * self._embedding_dim

        # Lazy load model on first use
        if self._session is None:
            self._load_model()

        # Check cache first
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Truncate to avoid token length issues
            if len(text) > 2000:
                text = text[:2000]

            # v0.3.2: Offload blocking ONNX inference so the event loop
            # stays free for WebSocket sends etc.
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(None, self._encode, [text])
            embedding = vectors[0].tolist()

            # Verify dimension
            if len(embedding) != self._embedding_dim:
                logger.warning(f"Dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}")
                embedding = (embedding + [0.0] * (self._embedding_dim - len(embedding)))[:self._embedding_dim]

            # Store in cache (FIFO eviction)
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
        if self._session is None:
            self._load_model()

        valid_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        try:
            # v0.3.2: Offload blocking ONNX inference.
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(None, self._encode, valid_texts)
            return vectors.tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [await self.embed_text(t) for t in texts]

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
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
