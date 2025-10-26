# backend/modules/web_search/strategies/bing_search_strategy.py
from typing import List, Dict, Optional
from core.interfaces.web_scraper_interface import SearchStrategyInterface
from bs4 import BeautifulSoup, Tag

class BingSearchStrategy(SearchStrategyInterface):
    def get_search_url(self, query: str, page_number: int = 1) -> str:
        query_encoded = query.replace(' ', '+')
        first_result = (page_number - 1) * 10 + 1
        return f"https://www.bing.com/search?q={query_encoded}&first={first_result}&adlt=strict"

    async def parse_results(self, html_content: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        processed_links = set()
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Parse organic web results
        for item in soup.select('li.b_algo'):
            parsed = self._parse_organic_result(item)
            if parsed and parsed['link'] not in processed_links:
                results.append(parsed)
                processed_links.add(parsed['link'])
        
        # Parse news card results if present
        for item in soup.select('div.news-card.newsitem'):
            parsed = self._parse_news_card(item)
            if parsed and parsed['link'] not in processed_links:
                results.append(parsed)
                processed_links.add(parsed['link'])
        
        return results
    
    def _parse_organic_result(self, item: Tag) -> Optional[Dict[str, str]]:
        """Parse standard web search result."""
        title_tag = item.select_one('h2 a')
        if not title_tag:
            return None
        
        title = title_tag.get_text(strip=True)
        link = title_tag.get('href', '#')
        
        # Try multiple snippet selectors
        snippet = ""
        for selector in ['p.b_lineclamp3', 'p.b_lineclamp2', 'div.b_caption p', '.b_algoSlug']:
            snippet_tag = item.select_one(selector)
            if snippet_tag:
                snippet = snippet_tag.get_text(strip=True)
                break
        
        if not snippet:
            snippet = "No description available"
        
        return {"title": title, "link": link, "snippet": snippet}
    
    def _parse_news_card(self, item: Tag) -> Optional[Dict[str, str]]:
        """Parse news card result."""
        title_tag = item.select_one('a.title')
        if not title_tag:
            return None
        
        title = title_tag.get_text(strip=True)
        link = title_tag.get('href', '#')
        
        snippet_tag = item.select_one('div.snippet')
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No description available"
        
        return {"title": title, "link": link, "snippet": snippet}
