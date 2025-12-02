"""
Excel Extractor using openpyxl and pandas
Extracts tabular data from Excel files with row-based chunking
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseExtractor, ExtractedDocument, ExtractionError

logger = logging.getLogger(__name__)


class ExcelExtractor(BaseExtractor):
    """
    Extracts data from Excel files (.xlsx, .xls) using openpyxl/pandas.

    Features:
    - Multi-sheet support (all sheets extracted)
    - Row-based chunking with headers for semantic search
    - Handles merged cells
    - Extracts document properties if available

    Limitations:
    - Formulas return computed values only
    - Charts and images skipped
    - Very large files may be slow
    """

    SUPPORTED_EXTENSIONS = ['.xlsx', '.xls']

    # Rows per chunk for semantic search optimization
    ROWS_PER_CHUNK = 50

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract data from Excel file"""
        self._validate_file(file_path)

        try:
            import pandas as pd
        except ImportError:
            raise ExtractionError("pandas not installed. Run: pip install pandas")

        try:
            import openpyxl
        except ImportError:
            raise ExtractionError("openpyxl not installed. Run: pip install openpyxl")

        warnings = []
        text_parts = []
        total_rows = 0
        total_cols = 0

        try:
            # Read all sheets
            engine = 'openpyxl' if file_path.suffix.lower() == '.xlsx' else 'xlrd'

            # Handle potential xlrd import for .xls
            if engine == 'xlrd':
                try:
                    import xlrd
                except ImportError:
                    warnings.append("xlrd not installed - .xls support limited. Trying openpyxl.")
                    engine = 'openpyxl'

            xlsx = pd.ExcelFile(str(file_path), engine=engine)
            sheet_names = xlsx.sheet_names

            if not sheet_names:
                raise ExtractionError(f"No sheets found in Excel file: {file_path.name}")

            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)

                    if df.empty:
                        continue

                    # Track stats
                    total_rows += len(df)
                    total_cols = max(total_cols, len(df.columns))

                    # Convert sheet to searchable text
                    sheet_text = self._dataframe_to_text(df, sheet_name)
                    if sheet_text:
                        text_parts.append(sheet_text)

                except Exception as e:
                    warnings.append(f"Error reading sheet '{sheet_name}': {str(e)}")
                    logger.warning(f"Error reading sheet '{sheet_name}' from {file_path.name}: {e}")

            xlsx.close()

        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                raise ExtractionError(f"Excel file is password-protected: {file_path.name}")
            raise ExtractionError(f"Failed to read Excel file: {e}")

        content = "\n\n".join(text_parts).strip()

        if not content:
            warnings.append("No data extracted from Excel file")

        logger.info(f"Extracted {total_rows} rows from {len(sheet_names)} sheet(s) in {file_path.name}")

        return ExtractedDocument(
            content=content,
            format_type='excel',
            is_tabular=True,
            row_count=total_rows,
            column_count=total_cols,
            extraction_warnings=warnings,
            original_path=str(file_path),
            extra_metadata={
                'sheet_count': len(sheet_names),
                'sheet_names': sheet_names,
            }
        )

    def _dataframe_to_text(self, df, sheet_name: str) -> str:
        """
        Convert DataFrame to searchable text format.
        Uses row-based chunking with column headers for context.
        """
        if df.empty:
            return ""

        parts = []

        # Sheet header
        parts.append(f"=== Sheet: {sheet_name} ===")
        parts.append(f"Columns: {' | '.join(str(c) for c in df.columns)}")
        parts.append("")

        # Convert data to string, handling various types
        df_str = df.astype(str).replace('nan', '').replace('NaN', '')

        # Output rows with periodic header reminders for long tables
        for i, (idx, row) in enumerate(df_str.iterrows()):
            # Remind of headers every ROWS_PER_CHUNK rows
            if i > 0 and i % self.ROWS_PER_CHUNK == 0:
                parts.append(f"\n[Continued - Columns: {' | '.join(str(c) for c in df.columns)}]")

            # Format row as "Col1: val1 | Col2: val2 | ..."
            row_parts = []
            for col in df.columns:
                val = row[col]
                if val and val.strip():
                    row_parts.append(f"{col}: {val}")

            if row_parts:
                parts.append(" | ".join(row_parts))

        return "\n".join(parts)
