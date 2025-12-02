"""
Base classes and dataclasses for format extraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when document extraction fails"""
    pass


@dataclass
class ExtractedDocument:
    """
    Standardized output from any format extractor.
    Contains extracted text and metadata for SmartBookProcessor.
    """
    content: str
    format_type: str

    # Metadata extracted from document (if available)
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None

    # For tabular data
    is_tabular: bool = False
    row_count: Optional[int] = None
    column_count: Optional[int] = None

    # Extraction info
    extraction_warnings: List[str] = field(default_factory=list)
    original_path: Optional[str] = None

    # Additional metadata
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def has_content(self) -> bool:
        """Check if extraction produced meaningful content"""
        return bool(self.content and self.content.strip())

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get all metadata as a dictionary for SmartBookProcessor"""
        metadata = {
            'format_type': self.format_type,
            'is_tabular': self.is_tabular,
        }
        if self.title:
            metadata['extracted_title'] = self.title
        if self.author:
            metadata['extracted_author'] = self.author
        if self.creation_date:
            metadata['creation_date'] = self.creation_date
        if self.row_count is not None:
            metadata['row_count'] = self.row_count
        if self.column_count is not None:
            metadata['column_count'] = self.column_count
        if self.extraction_warnings:
            metadata['extraction_warnings'] = self.extraction_warnings
        metadata.update(self.extra_metadata)
        return metadata


class BaseExtractor(ABC):
    """
    Abstract base class for all format extractors.
    Each extractor converts a specific format to ExtractedDocument.
    """

    # Subclasses should define supported extensions
    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract text content from a file.

        Args:
            file_path: Path to the file to extract

        Returns:
            ExtractedDocument with content and metadata

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if this extractor can handle the given file"""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def _validate_file(self, file_path: Path) -> None:
        """Common file validation"""
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ExtractionError(f"Not a file: {file_path}")
        if file_path.stat().st_size == 0:
            raise ExtractionError(f"File is empty: {file_path}")
