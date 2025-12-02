"""
HTML Extractor using BeautifulSoup
Extracts text content from HTML files
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class HTMLExtractor(BaseExtractor):
    """
    Extracts text from HTML files using BeautifulSoup.

    Features:
    - Removes scripts, styles, and other non-content elements
    - Preserves heading hierarchy (converted to markdown)
    - Extracts title from <title> tag
    - Handles common encodings

    Limitations:
    - Dynamic/JS-rendered content not captured
    - Complex layouts may lose structure
    """

    SUPPORTED_EXTENSIONS = ['.html', '.htm']

    # Tags to completely remove (including their content)
    REMOVE_TAGS = ['script', 'style', 'noscript', 'iframe', 'svg', 'canvas']

    # Tags that indicate structural breaks
    BLOCK_TAGS = ['p', 'div', 'section', 'article', 'header', 'footer',
                  'nav', 'aside', 'main', 'li', 'tr', 'blockquote', 'pre']

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from HTML file"""
        self._validate_file(file_path)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ExtractionError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

        warnings = []

        # Read file with encoding detection
        content_raw, encoding = self._read_with_encoding(file_path)

        if encoding and encoding.lower() not in ['utf-8', 'ascii']:
            warnings.append(f"Detected encoding: {encoding}")

        try:
            soup = BeautifulSoup(content_raw, 'html.parser')
        except Exception as e:
            raise ExtractionError(f"Failed to parse HTML: {e}")

        # Extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract meta description if available
        meta_desc = None
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')

        # Remove unwanted elements
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract and structure text
        text_parts = []

        # Handle headings with markdown conversion
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    heading.replace_with(f"\n{'#' * i} {heading_text}\n")

        # Get body content or full soup if no body
        body = soup.find('body') or soup

        # Extract text with structure preservation
        content = self._extract_structured_text(body)

        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()

        if not content:
            warnings.append("No text content extracted from HTML")

        logger.info(f"Extracted {len(content)} chars from {file_path.name}")

        return ExtractedDocument(
            content=content,
            format_type='html',
            title=title if title and title.strip() else None,
            extraction_warnings=warnings,
            original_path=str(file_path),
            extra_metadata={
                'meta_description': meta_desc,
                'encoding': encoding,
            }
        )

    def _read_with_encoding(self, file_path: Path) -> tuple:
        """Read file with encoding detection"""
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), 'utf-8'
        except UnicodeDecodeError:
            pass

        # Try chardet detection
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw = f.read()
                detected = chardet.detect(raw)
                encoding = detected.get('encoding', 'utf-8')
                return raw.decode(encoding, errors='replace'), encoding
        except ImportError:
            pass

        # Fallback with error replacement
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(), 'utf-8'

    def _extract_structured_text(self, element) -> str:
        """Extract text while preserving some structure"""
        texts = []

        for child in element.children:
            if hasattr(child, 'name'):
                # It's a tag
                if child.name in self.BLOCK_TAGS:
                    text = child.get_text(separator=' ', strip=True)
                    if text:
                        texts.append(text)
                        texts.append('')  # Add blank line after block
                else:
                    # Inline element - get text
                    text = child.get_text(separator=' ', strip=True)
                    if text:
                        texts.append(text)
            else:
                # NavigableString
                text = str(child).strip()
                if text:
                    texts.append(text)

        return '\n'.join(texts)
