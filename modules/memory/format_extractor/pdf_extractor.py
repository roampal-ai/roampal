"""
PDF Extractor using PyMuPDF (fitz)
Extracts text content and metadata from PDF files
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    Extracts text from PDF files using PyMuPDF.

    Features:
    - Text extraction from all pages
    - Metadata extraction (title, author, creation date)
    - Handles multi-column layouts reasonably well
    - Detects scanned/image-only PDFs and warns

    Limitations:
    - No OCR support (text-based PDFs only)
    - Password-protected PDFs not supported
    """

    SUPPORTED_EXTENSIONS = ['.pdf']

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text and metadata from PDF"""
        self._validate_file(file_path)

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ExtractionError("PyMuPDF not installed. Run: pip install pymupdf")

        warnings = []
        text_parts = []

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                raise ExtractionError(f"PDF is password-protected: {file_path.name}")
            raise ExtractionError(f"Failed to open PDF: {e}")

        try:
            # Extract metadata
            metadata = doc.metadata or {}
            title = metadata.get('title') or None
            author = metadata.get('author') or None
            creation_date = metadata.get('creationDate') or None

            # Clean up creation date format if present
            if creation_date and creation_date.startswith('D:'):
                # PDF date format: D:YYYYMMDDHHmmSS
                try:
                    creation_date = f"{creation_date[2:6]}-{creation_date[6:8]}-{creation_date[8:10]}"
                except (IndexError, ValueError):
                    pass

            # Extract text from each page
            total_pages = len(doc)
            pages_with_text = 0

            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text")

                if page_text.strip():
                    pages_with_text += 1
                    # Add page marker for context
                    text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                    text_parts.append(page_text)

            # Check if PDF might be scanned/image-only
            if pages_with_text == 0:
                warnings.append("No text extracted - PDF may be scanned/image-only (OCR not supported)")
            elif pages_with_text < total_pages * 0.5:
                warnings.append(f"Only {pages_with_text}/{total_pages} pages had extractable text")

            content = "\n".join(text_parts).strip()

            # Log extraction stats
            logger.info(f"Extracted {len(content)} chars from {pages_with_text}/{total_pages} pages of {file_path.name}")

            return ExtractedDocument(
                content=content,
                format_type='pdf',
                title=title if title and title.strip() else None,
                author=author if author and author.strip() else None,
                creation_date=creation_date,
                extraction_warnings=warnings,
                original_path=str(file_path),
                extra_metadata={
                    'total_pages': total_pages,
                    'pages_with_text': pages_with_text,
                }
            )

        finally:
            doc.close()
