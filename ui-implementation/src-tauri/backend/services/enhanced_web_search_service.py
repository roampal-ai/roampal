# Enhanced Web Search Service
# Integrates enhanced web search with the main backend and data layer

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from modules.web_search.enhanced_web_scraper import EnhancedWebScraper
from modules.embedding.embedding_service import EmbeddingService
from modules.memory.chromadb_adapter import ChromaDBAdapter
from config.enhanced_web_search_config import get_enhanced_web_search_config, get_shard_web_search_config
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedWebSearchService:
    """
    Enhanced web search service that integrates with the data layer.
    
    Features:
    - Shard-specific web search
    - Memory-based query optimization
    - Data layer integration for search results
    - Scalable per-shard architecture
    - Human-like browsing patterns preserved
    """
    
    def __init__(self):
        self.enhanced_scraper: Optional[EnhancedWebScraper] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.chromadb_adapters: Dict[str, ChromaDBAdapter] = {}
        self.config = get_enhanced_web_search_config()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the enhanced web search service."""
        try:
            logger.info("Initializing Enhanced Web Search Service...")
            
            # Initialize embedding service
            self.embedding_service = EmbeddingService()
            
            # Initialize ChromaDB adapters for different shards
            await self._initialize_shard_adapters()
            
            # Initialize enhanced web scraper
            self.enhanced_scraper = EnhancedWebScraper()
            await self.enhanced_scraper.initialize(
                service_client_config={
                    "playwright_service_url": self.config.playwright_service_url,
                    "request_timeout_seconds": self.config.request_timeout_seconds,
                    "default_search_engine": self.config.default_search_engine.value
                },
                embedding_service=self.embedding_service
            )
            
            self.initialized = True
            logger.info("Enhanced Web Search Service initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Web Search Service: {e}")
            raise
    
    async def _initialize_shard_adapters(self):
        """Initialize ChromaDB adapters for different shards."""
        try:
            # Initialize adapters for known shards
            shard_ids = ["og", "trader", "creator", "service", "ecommerce"]
            for shard_id in shard_ids:
                vector_db_dir = settings.paths.get_vector_db_dir(shard_id)
                self.chromadb_adapters[shard_id] = ChromaDBAdapter(
                    persistence_directory=str(vector_db_dir)
                )
            logger.info(f"Initialized ChromaDB adapters for {len(shard_ids)} shards")
        except Exception as e:
            logger.warning(f"Failed to initialize some shard adapters: {e}")
    
    async def search(
        self,
        query: str,
        shard_id: str = "og",
        num_results: int = 5,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform enhanced web search with data layer integration.
        
        Args:
            query: Search query
            shard_id: Target shard for data layer integration
            num_results: Number of results to return
            user_profile: User profile for personalization
            
        Returns:
            Dictionary containing search results and metadata
        """
        if not self.initialized:
            raise RuntimeError("Enhanced Web Search Service not initialized")
        
        try:
            # Get shard-specific configuration
            shard_config = get_shard_web_search_config(shard_id)
            if not shard_config:
                logger.warning(f"No configuration found for shard {shard_id}, using default")
                shard_config = get_shard_web_search_config("og")
            
            # Cultural context removed - simplified system
            
            # Set user profile if provided
            if user_profile:
                self.enhanced_scraper.set_user_profile(user_profile)
            
            # Set current shard
            self.enhanced_scraper.set_current_shard(shard_id)
            
            # Perform the search
            logger.info(f"Calling enhanced_scraper.search with query: '{query}', num_results: {num_results}, shard_id: {shard_id}")
            try:
                search_results = await self.enhanced_scraper.search(
                    query=query,
                    num_results=num_results,
                    shard_id=shard_id
                )
                logger.info(f"Enhanced scraper returned {len(search_results)} results")
            except Exception as e:
                logger.error(f"Enhanced scraper search failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                search_results = []
            
            # Filter results based on relevance threshold
            logger.info(f"Filtering {len(search_results)} results with threshold {shard_config.relevance_threshold}")
            for i, result in enumerate(search_results):
                relevance_score = result.get("relevance_score", 0.5)
                logger.info(f"Result {i+1}: relevance_score = {relevance_score}")
            
            filtered_results = self._filter_results_by_relevance(
                search_results, 
                shard_config.relevance_threshold
            )
            
            # Prepare response
            response = {
                "query": query,
                "shard_id": shard_id,
                "results": filtered_results,
                "total_results": len(search_results),
                "filtered_results": len(filtered_results),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "relevance_threshold": shard_config.relevance_threshold,
                    "memory_enabled": shard_config.memory_enabled
                }
            }
            
            logger.info(f"Enhanced search completed for shard {shard_id}: {len(filtered_results)} results")
            return response
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return {
                "query": query,
                "shard_id": shard_id,
                "results": [],
                "total_results": 0,
                "filtered_results": 0,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "relevance_threshold": 0.6,
                    "memory_enabled": True,
                    "error": str(e)
                }
            }
    
    def _filter_results_by_relevance(self, results: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Filter results based on relevance threshold."""
        filtered = []
        for result in results:
            relevance_score = result.get("relevance_score", 0.5)
            if relevance_score >= threshold:
                filtered.append(result)
        return filtered
    
    async def scrape_url(
        self,
        url: str,
        shard_id: str = "og"
    ) -> Dict[str, Any]:
        """
        Scrape URL content with data layer integration.
        
        Args:
            url: URL to scrape
            shard_id: Target shard for data layer integration
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        if not self.initialized:
            raise RuntimeError("Enhanced Web Search Service not initialized")
        
        try:
            # Set current shard
            self.enhanced_scraper.set_current_shard(shard_id)
            
            # Scrape the URL
            content = await self.enhanced_scraper.scrape_url_content(url, shard_id)
            
            response = {
                "url": url,
                "shard_id": shard_id,
                "content": content,
                "content_length": len(content) if content else 0,
                "timestamp": datetime.now().isoformat(),
                "success": content is not None
            }
            
            logger.info(f"URL scraping completed for shard {shard_id}: {len(content) if content else 0} characters")
            return response
            
        except Exception as e:
            logger.error(f"URL scraping failed: {e}")
            return {
                "url": url,
                "shard_id": shard_id,
                "content": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def search_with_memory_context(
        self,
        query: str,
        shard_id: str = "og",
        memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform search with additional memory context.
        
        Args:
            query: Search query
            shard_id: Target shard
            memory_context: Additional memory context to consider
            
        Returns:
            Enhanced search results with memory context
        """
        if not self.initialized:
            raise RuntimeError("Enhanced Web Search Service not initialized")
        
        try:
            # Enhance query with memory context if provided
            enhanced_query = query
            if memory_context:
                context_text = memory_context.get("text", "")
                if context_text:
                    enhanced_query = f"{query} {context_text}"
            
            # Perform search
            search_response = await self.search(
                query=enhanced_query,
                shard_id=shard_id
            )
            
            # Add memory context to response
            search_response["memory_context"] = memory_context
            search_response["original_query"] = query
            search_response["enhanced_query"] = enhanced_query
            
            return search_response
            
        except Exception as e:
            logger.error(f"Search with memory context failed: {e}")
            return {
                "query": query,
                "shard_id": shard_id,
                "results": [],
                "error": str(e),
                "memory_context": memory_context,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_search_history(
        self,
        shard_id: str = "og",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get search history for a specific shard.
        
        Args:
            shard_id: Target shard
            limit: Maximum number of history items to return
            
        Returns:
            List of search history items
        """
        try:
            if shard_id not in self.chromadb_adapters:
                logger.warning(f"No ChromaDB adapter for shard {shard_id}")
                return []
            
            adapter = self.chromadb_adapters[shard_id]
            
            # Query for search history
            results = await adapter.query_vectors(
                query_vector=None,
                top_k=limit,
                filters={"source": "web_search"}
            )
            
            # Format history items
            history = []
            for result in results:
                history.append({
                    "query": result.get("meta", {}).get("query", ""),
                    "timestamp": result.get("meta", {}).get("timestamp", ""),
                    "relevance_score": result.get("meta", {}).get("relevance_score", 0.0)
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []
    
    async def close(self) -> None:
        """Close the enhanced web search service."""
        if self.enhanced_scraper:
            await self.enhanced_scraper.close()
        logger.info("Enhanced Web Search Service closed")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the enhanced web search service."""
        return {
            "initialized": self.initialized,
            "config": {
                "playwright_service_url": self.config.playwright_service_url,
                "default_search_engine": self.config.default_search_engine.value,
                "enable_data_layer_integration": self.config.enable_data_layer_integration,
                "enable_memory_optimization": self.config.enable_memory_optimization
            },
            "shards": list(self.chromadb_adapters.keys()),
            "timestamp": datetime.now().isoformat()
        } 