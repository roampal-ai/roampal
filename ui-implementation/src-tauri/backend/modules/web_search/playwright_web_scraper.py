# C:\RoampalAI\backend\modules\web_search\playwright_web_scraper.py

import httpx
import json
import logging
import re
from typing import List, Dict, Any, Optional

from core.interfaces.web_scraper_interface import WebScraperInterface, SearchStrategyInterface  # Corrected import path
logger = logging.getLogger(__name__)

class PlaywrightServiceException(Exception):
    """Custom exception for errors when communicating with the Playwright Service."""
    pass

class PlaywrightWebScraper(WebScraperInterface):
    def __init__(self):
        self.service_url: str = ""
        self.http_client: Optional[httpx.AsyncClient] = None
        self.request_timeout: int = 60
        self.default_search_engine_name: str = "bing"
        logger.debug("PlaywrightWebScraper (HTTP Client) instance created (uninitialized).")

    async def initialize(
        self,
        service_client_config: Optional[Dict[str, Any]] = None,
        default_search_strategy: Optional[SearchStrategyInterface] = None
    ) -> None:
        logger.debug(f"Attempting to initialize PlaywrightWebScraper with config: {service_client_config}")
        if service_client_config is None:
            service_client_config = {}

        self.service_url = service_client_config.get("playwright_service_url", "")
        self.request_timeout = service_client_config.get("request_timeout_seconds", self.request_timeout)
        self.default_search_engine_name = service_client_config.get("default_search_engine", self.default_search_engine_name)

        if not self.service_url:
            logger.error("Playwright service URL not configured.")
            raise PlaywrightServiceException("Playwright service URL is not configured.")

        self.http_client = httpx.AsyncClient(base_url=self.service_url, timeout=self.request_timeout)
        logger.info(f"PlaywrightWebScraper initialized. URL: {self.service_url}, Timeout: {self.request_timeout}, Engine: {self.default_search_engine_name}")

    async def search(
        self,
        query: str,
        num_results: int = 5,
        strategy: Optional[SearchStrategyInterface] = None,
        search_engine_name_override: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if not self.http_client:
            raise PlaywrightServiceException("HTTP client not initialized.")

        # Sanitize query to remove quotes, normalize whitespace, and clean formatting
        sanitized_query = self._sanitize_search_query(query)
        
        engine = search_engine_name_override or self.default_search_engine_name
        payload = {"query": sanitized_query, "num_results": num_results, "search_engine": engine}

        logger.info(f"Sending search request: {payload}")
        try:
            response = await self.http_client.post("/search", json=payload)
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise PlaywrightServiceException(f"Search error: {data['error']}")

            results = data.get("results", [])

            temperature = self._extract_temperature_from_results(results)
            if temperature:
                results.insert(0, {
                    "title": "Auto-Extracted Temperature",
                    "snippet": f"The current temperature is {temperature}",
                    "url": ""
                })

            logger.info(f"Received {len(results)} search results.")
            return results

        except httpx.HTTPStatusError as e:
            detail = self._extract_http_error_detail(e)
            logger.error(f"HTTP error: {e.response.status_code} - {detail}", exc_info=True)
            raise PlaywrightServiceException(f"HTTP error: {e.response.status_code} - {detail}") from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise PlaywrightServiceException(f"Request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}", exc_info=True)
            raise PlaywrightServiceException(f"JSON decode error: {e}") from e

    async def scrape_url_content(self, url: str) -> Optional[str]:
        if not self.http_client:
            raise PlaywrightServiceException("HTTP client not initialized.")

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
            logger.info(f"Scraped content length: {len(content)}")
            return content

        except httpx.HTTPStatusError as e:
            detail = self._extract_http_error_detail(e)
            logger.error(f"Scrape error: {e.response.status_code} - {detail}", exc_info=True)
            raise PlaywrightServiceException(f"Scrape error: {e.response.status_code} - {detail}") from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise PlaywrightServiceException(f"Request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}", exc_info=True)
            raise PlaywrightServiceException(f"JSON decode error: {e}") from e

    def _sanitize_search_query(self, query: str) -> str:
        """Clean and sanitize search query for optimal search results."""
        if not query or not query.strip():
            return ""
            
        # Remove surrounding quotes
        cleaned = query.strip().strip('"\'')
        
        # Normalize whitespace (replace newlines, tabs, multiple spaces with single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove any remaining problematic characters that could break search
        cleaned = cleaned.strip()
        
        logger.debug(f"Sanitized query: '{query}' → '{cleaned}'")
        return cleaned

    def _extract_temperature_from_results(self, results: List[Dict[str, str]]) -> Optional[str]:
        for result in results:
            match = re.search(r"\b(-?\d{1,3})°[FfCc]?\b", result.get("snippet", ""))
            if match:
                return match.group(0)
        return None

    def _extract_http_error_detail(self, e: httpx.HTTPStatusError) -> str:
        try:
            return e.response.json().get("detail", e.response.text)
        except ValueError:
            return e.response.text

    async def close(self) -> None:
        if self.http_client:
            logger.info("Closing httpx.AsyncClient...")
            await self.http_client.aclose()
            self.http_client = None