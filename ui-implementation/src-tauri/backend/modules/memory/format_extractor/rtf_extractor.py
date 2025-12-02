"""
RTF Extractor using striprtf
Extracts text content from RTF (Rich Text Format) files
"""

import logging
from pathlib import Path

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class RTFExtractor(BaseExtractor):
    """
    Extracts text from RTF files using striprtf.

    Features:
    - Strips RTF formatting codes
    - Preserves paragraph structure
    - Handles various RTF encodings

    Limitations:
    - Images and embedded objects skipped
    - Complex formatting lost
    - Tables may lose structure
    """

    SUPPORTED_EXTENSIONS = ['.rtf']

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from RTF file"""
        self._validate_file(file_path)

        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            raise ExtractionError("striprtf not installed. Run: pip install striprtf")

        warnings = []

        # Read RTF content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                rtf_content = f.read()
        except Exception as e:
            raise ExtractionError(f"Failed to read RTF file: {e}")

        # Convert RTF to plain text
        try:
            content = rtf_to_text(rtf_content)
        except Exception as e:
            raise ExtractionError(f"Failed to parse RTF content: {e}")

        # Clean up the content
        if content:
            # Remove excessive whitespace while preserving paragraph breaks
            lines = content.split('\n')
            cleaned_lines = []
            prev_empty = False

            for line in lines:
                line = line.strip()
                if not line:
                    if not prev_empty:
                        cleaned_lines.append('')
                        prev_empty = True
                else:
                    cleaned_lines.append(line)
                    prev_empty = False

            content = '\n'.join(cleaned_lines).strip()

        if not content:
            warnings.append("No text content extracted from RTF")

        logger.info(f"Extracted {len(content)} chars from {file_path.name}")

        return ExtractedDocument(
            content=content,
            format_type='rtf',
            extraction_warnings=warnings,
            original_path=str(file_path),
        )
