"""
PDF Extractor using pypdf (BSD-licensed)
Extracts text content and metadata from PDF files
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    Extracts text from PDF files using pypdf.

    Features:
    - Text extraction from all pages
    - Metadata extraction (title, author, creation date)
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
            from pypdf import PdfReader
        except ImportError:
            raise ExtractionError("pypdf not installed. Run: pip install pypdf")

        warnings = []
        text_parts = []

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                raise ExtractionError(f"PDF is password-protected: {file_path.name}")
            raise ExtractionError(f"Failed to open PDF: {e}")

        # Extract metadata
        meta = reader.metadata
        title = (meta.title if meta and meta.title else None)
        author = (meta.author if meta and meta.author else None)
        creation_date = None
        if meta and meta.creation_date:
            try:
                creation_date = meta.creation_date.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Extract text from each page
        total_pages = len(reader.pages)
        pages_with_text = 0

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages_with_text += 1
                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                text_parts.append(page_text)

        # Check if PDF might be scanned/image-only
        if pages_with_text == 0:
            warnings.append("No text extracted - PDF may be scanned/image-only (OCR not supported)")
        elif pages_with_text < total_pages * 0.5:
            warnings.append(f"Only {pages_with_text}/{total_pages} pages had extractable text")

        content = "\n".join(text_parts).strip()

        logger.info(f"Extracted {len(content)} chars from {pages_with_text}/{total_pages} pages of {file_path.name}")

        return ExtractedDocument(
            content=content,
            format_type='pdf',
            title=title if title and str(title).strip() else None,
            author=author if author and str(author).strip() else None,
            creation_date=creation_date,
            extraction_warnings=warnings,
            original_path=str(file_path),
            extra_metadata={
                'total_pages': total_pages,
                'pages_with_text': pages_with_text,
            }
        )
