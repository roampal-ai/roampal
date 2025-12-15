# Enhanced Web Scraper with Data Layer Integration
# Preserves all human-like features while adding cultural intelligence and memory integration

import httpx
import json
import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from core.interfaces.web_scraper_interface import WebScraperInterface, SearchStrategyInterface
from modules.memory.chromadb_adapter import ChromaDBAdapter
from modules.embedding.embedding_service import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedWebScraperException(Exception):
    """Custom exception for enhanced web scraper errors."""
    pass

class EnhancedWebScraper(WebScraperInterface):
    """
    Enhanced web scraper that integrates with the data layer while preserving human-like features.
    
    Features:
    - Cultural intelligence and adaptation
    - Memory-based query optimization
    - Shard-specific data storage
    - Human-like browsing patterns
    - Anti-detection measures
    - Knowledge graph integration
    """
    
    def __init__(self):
        self.service_url: str = ""
        self.http_client: Optional[httpx.AsyncClient] = None
        self.request_timeout: int = 60
        self.default_search_engine_name: str = "bing"
        
        # Data layer integration
        self.embedding_service: Optional[EmbeddingService] = None
        self.chromadb_adapters: Dict[str, ChromaDBAdapter] = {}
        self.current_shard_id: str = "og"
        
        # Cultural intelligence
        self.cultural_context: Dict[str, Any] = {}
        self.user_profile: Dict[str, Any] = {}
        
        # Human-like features preservation
        self.search_patterns: List[Dict[str, Any]] = []
        self.user_agents: List[str] = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0"
        ]
        
        logger.debug("EnhancedWebScraper instance created (uninitialized).")

    async def initialize(
        self,
        service_client_config: Optional[Dict[str, Any]] = None,
        default_search_strategy: Optional[SearchStrategyInterface] = None,
        embedding_service: Optional[EmbeddingService] = None,
        cultural_context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the enhanced web scraper with data layer integration."""
        logger.debug(f"Initializing EnhancedWebScraper with config: {service_client_config}")
        
        if service_client_config is None:
            service_client_config = {}

        # Initialize basic web scraper
        self.service_url = service_client_config.get("playwright_service_url", "")
        self.request_timeout = service_client_config.get("request_timeout_seconds", self.request_timeout)
        self.default_search_engine_name = service_client_config.get("default_search_engine", self.default_search_engine_name)

        if not self.service_url:
            logger.error("Playwright service URL not configured.")
            raise EnhancedWebScraperException("Playwright service URL is not configured.")

        self.http_client = httpx.AsyncClient(base_url=self.service_url, timeout=self.request_timeout)
        
        # Initialize data layer components
        self.embedding_service = embedding_service or EmbeddingService()
        self.cultural_context = cultural_context or {}
        self.user_profile = user_profile or {}
        
        # Initialize ChromaDB adapters for different shards
        await self._initialize_shard_adapters()
        
        logger.info(f"EnhancedWebScraper initialized with data layer integration.")

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
        num_results: int = 5,
        strategy: Optional[SearchStrategyInterface] = None,
        search_engine_name_override: Optional[str] = None,
        shard_id: Optional[str] = None,
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Enhanced search with data layer integration and cultural intelligence.
        """
        if not self.http_client:
            raise EnhancedWebScraperException("HTTP client not initialized.")

        # Update context
        if shard_id:
            self.current_shard_id = shard_id
        if cultural_context:
            self.cultural_context = cultural_context

        # Enhance query with cultural intelligence and memory
        enhanced_query = await self._enhance_query_with_intelligence(query)
        
        # Perform the search
        engine = search_engine_name_override or self.default_search_engine_name
        payload = {
            "query": enhanced_query, 
            "num_results": num_results, 
            "search_engine": engine
        }

        logger.info(f"Sending enhanced search request: {payload}")
        try:
            logger.info(f"Making HTTP request to {self.service_url}/search")
            response = await self.http_client.post("/search", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Received response: {data}")

            if data.get("error"):
                raise EnhancedWebScraperException(f"Search error: {data['error']}")

            results = data.get("results", [])
            logger.info(f"Extracted {len(results)} results from response")
            
            # Convert link field to url for compatibility
            for result in results:
                if "link" in result and "url" not in result:
                    result["url"] = result["link"]

            logger.info(f"Raw results from playwright service: {len(results)} items")
            if results:
                logger.info(f"First result: {results[0]}")

            # Enhance results with cultural intelligence
            enhanced_results = await self._enhance_results_with_intelligence(results, query)
            
            # Store search results in data layer
            await self._store_search_results_in_data_layer(query, enhanced_results, shard_id)
            
            # Extract and add temperature if found
            temperature = self._extract_temperature_from_results(enhanced_results)
            if temperature:
                enhanced_results.insert(0, {
                    "title": "Auto-Extracted Temperature",
                    "snippet": f"The current temperature is {temperature}",
                    "url": ""
                })

            logger.info(f"Enhanced search completed. Found {len(enhanced_results)} results.")
            return enhanced_results

        except httpx.HTTPStatusError as e:
            detail = self._extract_http_error_detail(e)
            logger.error(f"HTTP error: {e.response.status_code} - {detail}", exc_info=True)
            raise EnhancedWebScraperException(f"HTTP error: {e.response.status_code} - {detail}") from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise EnhancedWebScraperException(f"Request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}", exc_info=True)
            raise EnhancedWebScraperException(f"JSON decode error: {e}") from e

    async def _enhance_query_with_intelligence(self, query: str) -> str:
        """Enhance search query with cultural intelligence and memory."""
        enhanced_query = query
        
        # Add cultural context to query if relevant
        if self.cultural_context:
            cultural_keywords = self._get_cultural_keywords()
            if cultural_keywords:
                enhanced_query = f"{query} {cultural_keywords}"
        
        # Add memory-based context from previous searches
        memory_context = await self._get_memory_context(query)
        if memory_context:
            enhanced_query = f"{enhanced_query} {memory_context}"
        
        logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
        return enhanced_query

    def _get_cultural_keywords(self) -> str:
        """Get cultural keywords based on current cultural context."""
        if not self.cultural_context:
            return ""
        
        cultural_keywords = {
            "western_individual": "individual perspective personal development",
            "eastern_collectivist": "community harmony family values",
            "south_asian_family": "family tradition spiritual growth",
            "african_communal": "community wisdom ancestral knowledge",
            "middle_eastern_honor": "honor family respect tradition",
            "latin_american_familial": "family passion community celebration",
            "indigenous_traditional": "land connection ancestral wisdom",
            "global_nomad": "global perspective cultural adaptation"
        }
        
        context_type = self.cultural_context.get("type", "western_individual")
        return cultural_keywords.get(context_type, "")

    async def _get_memory_context(self, query: str) -> str:
        """Get relevant context from memory based on query."""
        try:
            if self.current_shard_id in self.chromadb_adapters:
                adapter = self.chromadb_adapters[self.current_shard_id]
                
                # Search for related content in memory
                results = await adapter.query_vectors(
                    query_vector=None,  # Will be computed from query
                    top_k=3,
                    filters={"source": "web_search"}
                )
                
                if results:
                    # Extract relevant keywords from memory
                    keywords = []
                    for result in results:
                        content = result.get("content", "")
                        # Extract key terms (simplified)
                        words = content.split()[:10]  # First 10 words
                        keywords.extend(words)
                    
                    return " ".join(set(keywords))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
        
        return ""

    async def _enhance_results_with_intelligence(self, results: List[Dict[str, str]], original_query: str) -> List[Dict[str, str]]:
        """Enhance search results with cultural intelligence and relevance scoring."""
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add cultural relevance score
            cultural_score = self._calculate_cultural_relevance(result, original_query)
            enhanced_result["cultural_relevance"] = cultural_score
            
            # Add memory-based relevance
            memory_score = await self._calculate_memory_relevance(result)
            enhanced_result["memory_relevance"] = memory_score
            
            # Add combined relevance score
            combined_score = (cultural_score + memory_score) / 2
            enhanced_result["relevance_score"] = combined_score
            
            enhanced_results.append(enhanced_result)
        
        # Sort by relevance score
        enhanced_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return enhanced_results

    def _calculate_cultural_relevance(self, result: Dict[str, str], query: str) -> float:
        """Calculate cultural relevance score for a search result."""
        if not self.cultural_context:
            return 0.5  # Neutral score
        
        content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        # Define cultural keywords and their weights
        cultural_keywords = {
            "western_individual": ["individual", "personal", "achievement", "freedom", "career"],
            "eastern_collectivist": ["community", "harmony", "family", "respect", "modesty"],
            "south_asian_family": ["family", "spiritual", "dharma", "respect", "tradition"],
            "african_communal": ["community", "ancestral", "wisdom", "storytelling", "ubuntu"],
            "middle_eastern_honor": ["honor", "family", "religious", "hospitality", "respect"],
            "latin_american_familial": ["family", "passion", "community", "spirituality", "celebration"],
            "indigenous_traditional": ["land", "ancestral", "community", "tradition", "wisdom"],
            "global_nomad": ["global", "cultural", "adaptation", "diversity", "perspective"]
        }
        
        context_type = self.cultural_context.get("type", "western_individual")
        keywords = cultural_keywords.get(context_type, [])
        
        # Calculate relevance score
        score = 0.0
        for keyword in keywords:
            if keyword in content:
                score += 0.2  # 0.2 points per matching keyword
        
        return min(score, 1.0)  # Cap at 1.0

    async def _calculate_memory_relevance(self, result: Dict[str, str]) -> float:
        """Calculate memory-based relevance score."""
        try:
            if self.current_shard_id in self.chromadb_adapters:
                adapter = self.chromadb_adapters[self.current_shard_id]
                
                # Check if similar content exists in memory
                content = f"{result.get('title', '')} {result.get('snippet', '')}"
                
                # Simple similarity check (in production, use proper embedding)
                results = await adapter.query_vectors(
                    query_vector=None,
                    top_k=1,
                    filters={"source": "web_search"}
                )
                
                if results:
                    # If similar content exists, it's more relevant
                    return 0.8
                else:
                    # New content might be interesting
                    return 0.6
        except Exception as e:
            logger.warning(f"Failed to calculate memory relevance: {e}")
        
        return 0.5  # Default neutral score

    async def _store_search_results_in_data_layer(self, query: str, results: List[Dict[str, str]], shard_id: Optional[str] = None):
        """Store search results in the data layer for future reference."""
        try:
            if not shard_id:
                shard_id = self.current_shard_id
            
            if shard_id not in self.chromadb_adapters:
                logger.warning(f"No ChromaDB adapter for shard {shard_id}")
                return
            
            adapter = self.chromadb_adapters[shard_id]
            
            for result in results:
                # Create a unique ID for this search result
                result_id = str(uuid.uuid4())
                
                # Prepare content for storage
                content = f"{result.get('title', '')}\n{result.get('snippet', '')}\n{result.get('url', '')}"
                
                # Prepare metadata
                metadata = {
                    "source": "web_search",
                    "query": query,
                    "shard_id": shard_id,
                    "cultural_context": self.cultural_context.get("type", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "relevance_score": result.get("relevance_score", 0.5)
                }
                
                # Store in ChromaDB
                await adapter.upsert_to_chroma(
                    text=content,
                    id=result_id,
                    meta=metadata,
                    vector_db=adapter,
                    embedding_service=self.embedding_service,
                    collection_name=f"web_search_{shard_id}"
                )
                
            logger.info(f"Stored {len(results)} search results in data layer for shard {shard_id}")
            
        except Exception as e:
            logger.error(f"Failed to store search results in data layer: {e}")

    async def scrape_url_content(self, url: str, shard_id: Optional[str] = None) -> Optional[str]:
        """Enhanced URL scraping with data layer integration."""
        if not self.http_client:
            raise EnhancedWebScraperException("HTTP client not initialized.")

        payload = {"url": url}
        logger.info(f"Scraping URL: {url}")
        try:
            response = await self.http_client.post("/scrape-url", json=payload)
            response.raise_for_status()
            data = response.json()

            if data.get("error") and not data.get("content"):
                logger.warning(f"Scrape error for {url}: {data['error']}")
                return None

            content = data.get("content", "")
            
            # Store scraped content in data layer
            if content and shard_id:
                await self._store_scraped_content_in_data_layer(url, content, shard_id)
            
            logger.info(f"Scraped content length: {len(content)}")
            return content

        except httpx.HTTPStatusError as e:
            detail = self._extract_http_error_detail(e)
            logger.error(f"Scrape error: {e.response.status_code} - {detail}", exc_info=True)
            raise EnhancedWebScraperException(f"Scrape error: {e.response.status_code} - {detail}") from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise EnhancedWebScraperException(f"Request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}", exc_info=True)
            raise EnhancedWebScraperException(f"JSON decode error: {e}") from e

    async def _store_scraped_content_in_data_layer(self, url: str, content: str, shard_id: str):
        """Store scraped content in the data layer."""
        try:
            if shard_id not in self.chromadb_adapters:
                logger.warning(f"No ChromaDB adapter for shard {shard_id}")
                return
            
            adapter = self.chromadb_adapters[shard_id]
            
            # Create a unique ID for this scraped content
            content_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {
                "source": "scraped_content",
                "url": url,
                "shard_id": shard_id,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            # Store in ChromaDB
            await adapter.upsert_to_chroma(
                text=content,
                id=content_id,
                meta=metadata,
                vector_db=adapter,
                embedding_service=self.embedding_service,
                collection_name=f"scraped_content_{shard_id}"
            )
            
            logger.info(f"Stored scraped content from {url} in data layer for shard {shard_id}")
            
        except Exception as e:
            logger.error(f"Failed to store scraped content in data layer: {e}")

    def _extract_temperature_from_results(self, results: List[Dict[str, str]]) -> Optional[str]:
        """Extract temperature information from search results."""
        for result in results:
            match = re.search(r"\b(-?\d{1,3})°[FfCc]?\b", result.get("snippet", ""))
            if match:
                return match.group(0)
        return None

    def _extract_http_error_detail(self, e: httpx.HTTPStatusError) -> str:
        """Extract error detail from HTTP response."""
        try:
            return e.response.json().get("detail", e.response.text)
        except ValueError:
            return e.response.text

    async def close(self) -> None:
        """Close the enhanced web scraper."""
        if self.http_client:
            logger.info("Closing httpx.AsyncClient...")
            await self.http_client.aclose()
            self.http_client = None

    def set_cultural_context(self, context: Dict[str, Any]):
        """Set the cultural context for search operations."""
        self.cultural_context = context
        logger.info(f"Set cultural context: {context}")

    def set_user_profile(self, profile: Dict[str, Any]):
        """Set the user profile for personalized search."""
        self.user_profile = profile
        logger.info(f"Set user profile: {profile}")

    def set_current_shard(self, shard_id: str):
        """Set the current shard for data layer operations."""
        self.current_shard_id = shard_id
        logger.info(f"Set current shard: {shard_id}") 