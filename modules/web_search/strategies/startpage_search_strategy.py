# backend/modules/web_search/strategies/startpage_search_strategy.py
from typing import List, Dict, Optional
from core.interfaces.web_scraper_interface import SearchStrategyInterface
from bs4 import BeautifulSoup, Tag

class StartpageSearchStrategy(SearchStrategyInterface):
    def get_search_url(self, query: str, page_number: int = 1) -> str:
        query_encoded = query.replace(' ', '+')
        return f"https://www.startpage.com/sp/search?query={query_encoded}&page={page_number}"

    async def parse_results(self, html_content: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        soup = BeautifulSoup(html_content, 'lxml')
        
        # StartPage uses w-gl class for web results
        for item in soup.select('.w-gl__result'):
            parsed = self._parse_result(item)
            if parsed:
                results.append(parsed)
        
        return results
    
    def _parse_result(self, item: Tag) -> Optional[Dict[str, str]]:
        """Parse StartPage search result."""
        # Title and link in h3 a tag
        title_tag = item.select_one('h3 a.w-gl__result-title')
        if not title_tag:
            # Try alternative selector
            title_tag = item.select_one('a.w-gl__result-title')
        
        if not title_tag:
            return None
        
        title = title_tag.get_text(strip=True)
        link = title_tag.get('href', '#')
        
        # Description/snippet
        snippet_tag = item.select_one('.w-gl__description')
        if not snippet_tag:
            # Try alternative selector
            snippet_tag = item.select_one('p.w-gl__result-desc')
        
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No description available"
        
        return {"title": title, "link": link, "snippet": snippet}
