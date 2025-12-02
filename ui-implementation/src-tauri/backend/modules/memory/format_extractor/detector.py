"""
Format Detector and Router
Detects document format and routes to appropriate extractor
"""

import logging
from pathlib import Path
from typing import Dict, Type, Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError
from .pdf_extractor import PDFExtractor
from .docx_extractor import DocxExtractor
from .excel_extractor import ExcelExtractor
from .csv_extractor import CSVExtractor
from .html_extractor import HTMLExtractor
from .rtf_extractor import RTFExtractor

logger = logging.getLogger(__name__)


class FormatDetector:
    """
    Detects document format and provides appropriate extractor.

    Usage:
        detector = FormatDetector()
        extractor = detector.get_extractor(file_path)
        result = extractor.extract(file_path)

    Or use the convenience function:
        result = detect_and_extract(file_path)
    """

    # Map of extensions to extractor classes
    EXTRACTOR_MAP: Dict[str, Type[BaseExtractor]] = {
        # PDF
        '.pdf': PDFExtractor,

        # Word
        '.docx': DocxExtractor,

        # Excel
        '.xlsx': ExcelExtractor,
        '.xls': ExcelExtractor,

        # CSV/TSV
        '.csv': CSVExtractor,
        '.tsv': CSVExtractor,

        # HTML
        '.html': HTMLExtractor,
        '.htm': HTMLExtractor,

        # RTF
        '.rtf': RTFExtractor,
    }

    # Plain text formats (no extraction needed, just read)
    PLAIN_TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.text'}

    @classmethod
    def get_supported_extensions(cls) -> set:
        """Get all supported file extensions"""
        return set(cls.EXTRACTOR_MAP.keys()) | cls.PLAIN_TEXT_EXTENSIONS

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file format is supported"""
        ext = file_path.suffix.lower()
        return ext in cls.EXTRACTOR_MAP or ext in cls.PLAIN_TEXT_EXTENSIONS

    @classmethod
    def is_plain_text(cls, file_path: Path) -> bool:
        """Check if file is plain text (no extraction needed)"""
        return file_path.suffix.lower() in cls.PLAIN_TEXT_EXTENSIONS

    @classmethod
    def get_extractor(cls, file_path: Path) -> Optional[BaseExtractor]:
        """
        Get the appropriate extractor for a file.

        Returns:
            Extractor instance or None for plain text files
        """
        ext = file_path.suffix.lower()

        if ext in cls.PLAIN_TEXT_EXTENSIONS:
            return None  # No extraction needed

        extractor_class = cls.EXTRACTOR_MAP.get(ext)
        if extractor_class:
            return extractor_class()

        raise ExtractionError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: {', '.join(sorted(cls.get_supported_extensions()))}"
        )

    @classmethod
    def detect_format(cls, file_path: Path) -> str:
        """
        Detect the format type of a file.

        Returns:
            Format string (e.g., 'pdf', 'docx', 'txt')
        """
        ext = file_path.suffix.lower()

        if ext in cls.PLAIN_TEXT_EXTENSIONS:
            return 'text'

        # Map extension to format type
        format_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.csv': 'csv',
            '.tsv': 'csv',
            '.html': 'html',
            '.htm': 'html',
            '.rtf': 'rtf',
        }

        return format_map.get(ext, 'unknown')


def detect_and_extract(file_path: Path) -> ExtractedDocument:
    """
    Convenience function to detect format and extract content.

    Args:
        file_path: Path to the document

    Returns:
        ExtractedDocument with content and metadata

    Raises:
        ExtractionError: If extraction fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ExtractionError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ExtractionError(f"Not a file: {file_path}")

    # Handle plain text files directly
    if FormatDetector.is_plain_text(file_path):
        return _extract_plain_text(file_path)

    # Get appropriate extractor
    extractor = FormatDetector.get_extractor(file_path)
    if extractor is None:
        raise ExtractionError(f"No extractor available for: {file_path.suffix}")

    logger.info(f"Extracting {file_path.name} using {extractor.__class__.__name__}")

    return extractor.extract(file_path)


def _extract_plain_text(file_path: Path) -> ExtractedDocument:
    """
    Extract content from plain text files.
    Handles encoding detection.
    """
    warnings = []

    # Try UTF-8 first
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        encoding = 'utf-8'
    except UnicodeDecodeError:
        # Try chardet
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw = f.read()
                detected = chardet.detect(raw)
                encoding = detected.get('encoding', 'utf-8')
                content = raw.decode(encoding, errors='replace')
                warnings.append(f"Detected encoding: {encoding}")
        except ImportError:
            # Fallback
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                encoding = 'utf-8'
                warnings.append("Encoding detection unavailable, using UTF-8 with replacement")

    # Determine format type
    ext = file_path.suffix.lower()
    format_type = 'markdown' if ext in ['.md', '.markdown'] else 'text'

    return ExtractedDocument(
        content=content,
        format_type=format_type,
        extraction_warnings=warnings,
        original_path=str(file_path),
        extra_metadata={
            'encoding': encoding,
        }
    )
