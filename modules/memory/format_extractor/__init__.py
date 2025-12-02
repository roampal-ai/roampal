"""
Format Extractor Module for Roampal v0.2.3
Converts various document formats to plain text for processing by SmartBookProcessor
"""

from .base import ExtractedDocument, BaseExtractor, ExtractionError
from .detector import FormatDetector, detect_and_extract
from .pdf_extractor import PDFExtractor
from .docx_extractor import DocxExtractor
from .excel_extractor import ExcelExtractor
from .csv_extractor import CSVExtractor
from .html_extractor import HTMLExtractor
from .rtf_extractor import RTFExtractor

__all__ = [
    'ExtractedDocument',
    'BaseExtractor',
    'ExtractionError',
    'FormatDetector',
    'detect_and_extract',
    'PDFExtractor',
    'DocxExtractor',
    'ExcelExtractor',
    'CSVExtractor',
    'HTMLExtractor',
    'RTFExtractor',
]
