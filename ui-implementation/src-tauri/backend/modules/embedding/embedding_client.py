"""
Embedding Client for communicating with the standalone embedding service
Falls back to local embedding generation if service is unavailable
"""
import logging
from typing import List, Optional
import aiohttp
import asyncio
from modules.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class EmbeddingClient:
    """
    Client for the standalone embedding service.
    Provides automatic fallback to local embedding generation if service is unavailable.
    """
    
    def __init__(
        self,
        service_url: str = "http://localhost:8004",
        fallback_to_local: bool = True,
        timeout: int = 30
    ):
        self.service_url = service_url.rstrip('/')
        self.fallback_to_local = fallback_to_local
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.local_service: Optional[EmbeddingService] = None
        self.service_available = False
        
        # Track service availability
        self.last_check_time = 0
        self.check_interval = 60  # Check every 60 seconds
        
    async def initialize(self):
        """Initialize the client and check service availability"""
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=600
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Check if service is available
        await self._check_service_health()
        
        # Initialize local fallback if needed
        if self.fallback_to_local and not self.service_available:
            logger.info("Embedding service not available, initializing local fallback")
            self.local_service = EmbeddingService()
    
    async def _check_service_health(self) -> bool:
        """Check if the embedding service is healthy"""
        try:
            async with self.session.get(
                f"{self.service_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Embedding service healthy: {data}")
                    self.service_available = True
                    return True
        except Exception as e:
            logger.debug(f"Embedding service not available: {e}")
        
        self.service_available = False
        return False
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        Tries service first, falls back to local if configured.
        """
        # Periodically check service availability
        import time
        current_time = time.time()
        if current_time - self.last_check_time > self.check_interval:
            await self._check_service_health()
            self.last_check_time = current_time
        
        # Try service if available
        if self.service_available:
            try:
                async with self.session.post(
                    f"{self.service_url}/embed",
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["embedding"]
                    else:
                        logger.warning(f"Embedding service returned {response.status}")
            except asyncio.TimeoutError:
                logger.warning("Embedding service timeout")
                self.service_available = False
            except Exception as e:
                logger.warning(f"Embedding service error: {e}")
                self.service_available = False
        
        # Fallback to local if configured
        if self.fallback_to_local:
            if not self.local_service:
                self.local_service = EmbeddingService()
            
            logger.debug("Using local embedding generation")
            return await self.local_service.embed_text(text)
        
        logger.error("No embedding service available and fallback disabled")
        return None
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Uses batch endpoint for better performance.
        """
        # Try service if available
        if self.service_available:
            try:
                async with self.session.post(
                    f"{self.service_url}/embed/batch",
                    json={"texts": texts},
                    timeout=aiohttp.ClientTimeout(total=self.timeout * 2)  # Double timeout for batch
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["embeddings"]
            except Exception as e:
                logger.warning(f"Batch embedding service error: {e}")
                self.service_available = False
        
        # Fallback to local sequential processing
        if self.fallback_to_local:
            if not self.local_service:
                self.local_service = EmbeddingService()
            
            logger.debug(f"Using local batch embedding for {len(texts)} texts")
            embeddings = []
            for text in texts:
                embedding = await self.local_service.embed_text(text)
                embeddings.append(embedding if embedding else [0.0] * 768)
            return embeddings
        
        return [[0.0] * 768] * len(texts)  # Return zero vectors as last resort
    
    async def get_cache_stats(self) -> Optional[dict]:
        """Get cache statistics from the service"""
        if not self.service_available:
            return None
        
        try:
            async with self.session.get(
                f"{self.service_url}/cache/stats",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Failed to get cache stats: {e}")
        
        return None
    
    async def clear_cache(self) -> bool:
        """Clear the service cache"""
        if not self.service_available:
            return False
        
        try:
            async with self.session.delete(
                f"{self.service_url}/cache",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Failed to clear cache: {e}")
        
        return False
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()

# Global client instance
_embedding_client: Optional[EmbeddingClient] = None

async def get_embedding_client() -> EmbeddingClient:
    """Get or create the global embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        await _embedding_client.initialize()
    return _embedding_client