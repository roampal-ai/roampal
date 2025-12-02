"""
CSV Extractor using pandas
Extracts tabular data from CSV files with row-based chunking
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class CSVExtractor(BaseExtractor):
    """
    Extracts data from CSV files using pandas.

    Features:
    - Auto-detects delimiter (comma, tab, semicolon, pipe)
    - Auto-detects encoding using chardet
    - Row-based text conversion for semantic search
    - Handles quoted fields and escapes

    Limitations:
    - Very large files (>100K rows) may be slow
    - Complex nested structures not supported
    """

    SUPPORTED_EXTENSIONS = ['.csv', '.tsv']

    # Rows per chunk for semantic search optimization
    ROWS_PER_CHUNK = 50

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract data from CSV file"""
        self._validate_file(file_path)

        try:
            import pandas as pd
        except ImportError:
            raise ExtractionError("pandas not installed. Run: pip install pandas")

        warnings = []

        # Detect encoding
        encoding = self._detect_encoding(file_path)
        if encoding and encoding.lower() not in ['utf-8', 'ascii']:
            warnings.append(f"Detected encoding: {encoding}")

        # Detect delimiter
        delimiter = self._detect_delimiter(file_path, encoding)

        try:
            df = pd.read_csv(
                str(file_path),
                encoding=encoding or 'utf-8',
                sep=delimiter,
                on_bad_lines='warn',  # Don't fail on malformed lines
                low_memory=False,  # Avoid mixed type warnings
            )
        except Exception as e:
            raise ExtractionError(f"Failed to read CSV: {e}")

        if df.empty:
            warnings.append("CSV file is empty or has no parseable data")
            return ExtractedDocument(
                content="",
                format_type='csv',
                is_tabular=True,
                row_count=0,
                column_count=0,
                extraction_warnings=warnings,
                original_path=str(file_path),
            )

        # Convert to searchable text
        content = self._dataframe_to_text(df)

        logger.info(f"Extracted {len(df)} rows x {len(df.columns)} cols from {file_path.name}")

        return ExtractedDocument(
            content=content,
            format_type='csv',
            is_tabular=True,
            row_count=len(df),
            column_count=len(df.columns),
            extraction_warnings=warnings,
            original_path=str(file_path),
            extra_metadata={
                'delimiter': delimiter,
                'encoding': encoding,
                'columns': list(df.columns),
            }
        )

    def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """Detect file encoding using chardet"""
        try:
            import chardet
        except ImportError:
            return 'utf-8'

        try:
            with open(file_path, 'rb') as f:
                # Read first 10KB for detection
                raw = f.read(10240)
                result = chardet.detect(raw)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'

    def _detect_delimiter(self, file_path: Path, encoding: Optional[str] = None) -> str:
        """Auto-detect CSV delimiter"""
        try:
            with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                sample = f.read(4096)

            # Count occurrences of common delimiters
            delimiters = {
                ',': sample.count(','),
                '\t': sample.count('\t'),
                ';': sample.count(';'),
                '|': sample.count('|'),
            }

            # Return the most common delimiter
            best = max(delimiters, key=delimiters.get)
            return best if delimiters[best] > 0 else ','

        except Exception:
            return ','

    def _dataframe_to_text(self, df) -> str:
        """
        Convert DataFrame to searchable text format.
        Uses row-based format with column headers for context.
        """
        parts = []

        # Header with column names
        parts.append(f"Columns: {' | '.join(str(c) for c in df.columns)}")
        parts.append("")

        # Convert data to string, handling NaN
        df_str = df.astype(str).replace('nan', '').replace('NaN', '')

        # Output rows with periodic header reminders
        for i, (idx, row) in enumerate(df_str.iterrows()):
            # Remind of headers every ROWS_PER_CHUNK rows
            if i > 0 and i % self.ROWS_PER_CHUNK == 0:
                parts.append(f"\n[Row {i+1}+ - Columns: {' | '.join(str(c) for c in df.columns)}]")

            # Format row as "Col1: val1 | Col2: val2 | ..."
            row_parts = []
            for col in df.columns:
                val = row[col]
                if val and val.strip():
                    row_parts.append(f"{col}: {val}")

            if row_parts:
                parts.append(" | ".join(row_parts))

        return "\n".join(parts)
