# backend/core/interfaces/web_scraper_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# We might later define a Pydantic model for SearchResult in core/types/common_types.py
# e.g., class SearchResult(BaseModel): title: str; link: str; snippet: Optional[str] = None

class SearchStrategyInterface(ABC):
    """
    Abstract Base Class for search engine specific strategies.
    Defines how to construct search URLs and parse results for a particular
    search engine (e.g., Bing, Startpage, Google).
    """

    @abstractmethod
    def get_search_url(self, query: str, page_number: int = 1) -> str:
        """
        Constructs the search URL for the given query and page number
        for the specific search engine.

        Args:
            query: The search query string.
            page_number: The page number of the search results (default is 1).

        Returns:
            The fully formed URL for the search query.
        """
        pass

    @abstractmethod
    async def parse_results(self, html_content: str) -> List[Dict[str, str]]:
        """
        Parses the HTML content of a search engine results page to extract
        structured search result data.

        Args:
            html_content: The HTML content (as a string) of the search results page.

        Returns:
            A list of dictionaries, where each dictionary represents a search result
            and should contain at least 'title' and 'link' keys. An optional
            'snippet' key can also be included.
            Example: [{"title": "...", "link": "...", "snippet": "..."}, ...]
            Returns an empty list if no results are found or parsing fails.
        """
        pass


class WebScraperInterface(ABC):
    """
    Abstract Base Class for web scraping operations.
    Defines a common interface for performing web searches and scraping
    content from URLs, abstracting the underlying scraping tool (e.g., Playwright).
    """

    @abstractmethod
    async def initialize(
        self,
        playwright_config: Optional[Dict[str, Any]] = None,
        default_search_strategy: Optional[SearchStrategyInterface] = None
    ) -> None:
        """
        Initializes the web scraper.
        This might involve launching a browser instance (e.g., Playwright),
        setting up timeouts, and configuring a default search strategy.

        Args:
            playwright_config: (Optional) A dictionary containing configurations
                               specific to the Playwright setup (e.g., browser type,
                               headless mode, user agent, timeouts).
            default_search_strategy: (Optional) An instance of a SearchStrategyInterface
                                     to be used as the default for search operations if
                                     no specific strategy is provided to the search method.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 5,
        strategy: Optional[SearchStrategyInterface] = None
    ) -> List[Dict[str, str]]:
        """
        Performs a web search for the given query using either the provided
        strategy or the default search strategy.

        Args:
            query: The search query string.
            num_results: The desired approximate number of search results. The scraper
                         will attempt to fetch at least this many, but the actual
                         number may vary based on search engine output.
            strategy: (Optional) An instance of SearchStrategyInterface to use for
                      this specific search. If None, the default strategy set during
                      initialization will be used.

        Returns:
            A list of dictionaries, where each dictionary represents a search result
            (as defined by SearchStrategyInterface.parse_results).
            Returns an empty list if the search fails or no results are found.
        """
        pass

    @abstractmethod
    async def scrape_url_content(
        self,
        url: str,
        # Optional: Add parameters for more sophisticated scraping, e.g.,
        # content_selectors: Optional[List[str]] = None,
        # extract_metadata: bool = False,
        # render_javascript: bool = True # If the underlying tool supports it
    ) -> Optional[str]:
        """
        Scrapes the primary textual content from a given URL.
        The goal is to extract the main article or body content, stripping
        away boilerplate like navigation, ads, footers, etc., if possible.

        Args:
            url: The URL of the web page to scrape.

        Returns:
            A string containing the extracted textual content of the page,
            or None if scraping fails or no meaningful content is found.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Performs any cleanup and closes resources held by the web scraper,
        such as browser instances or network connections.
        This method should be called when the application is shutting down
        or when the scraper is no longer needed.
        """
        pass
