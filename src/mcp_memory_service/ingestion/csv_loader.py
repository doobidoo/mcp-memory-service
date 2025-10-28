# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CSV document loader for tabular data files.
"""

import csv
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional
import asyncio
import io

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


class CSVLoader(DocumentLoader):
    """
    Document loader for CSV data files.

    Features:
    - Automatic delimiter and header detection
    - Converts rows to text with column context
    - Handles large files with row-based chunking
    - Preserves table structure in metadata
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize CSV loader.

        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['csv']

        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=False,  # CSV doesn't have paragraphs
            respect_sentence_boundaries=False,   # CSV doesn't have sentences
            min_chunk_size=10  # Allow smaller chunks for tabular data
        ))

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given CSV file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this loader can process the CSV file
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        extension = file_path.suffix.lower().lstrip('.')
        return extension in self.supported_extensions

    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a CSV file.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional options:
                - has_header: Whether file has headers (auto-detected if not specified)
                - delimiter: CSV delimiter (auto-detected if not specified)
                - quotechar: Quote character (default: ")
                - encoding: Text encoding (auto-detected if not specified)
                - max_rows_per_chunk: Maximum rows to include in each chunk
                - include_row_numbers: Whether to include row numbers in output

        Yields:
            DocumentChunk objects containing extracted text and metadata

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV file can't be parsed or processed
        """
        await self.validate_file(file_path)

        has_header = kwargs.get('has_header', None)  # Auto-detect if None
        delimiter = kwargs.get('delimiter', None)    # Auto-detect if None
        quotechar = kwargs.get('quotechar', '"')
        encoding = kwargs.get('encoding', None)      # Auto-detect if None
        max_rows_per_chunk = kwargs.get('max_rows_per_chunk', 50)
        include_row_numbers = kwargs.get('include_row_numbers', True)

        logger.info(f"Extracting chunks from CSV file: {file_path}")

        try:
            # Read CSV data
            rows, detected_header, detected_delimiter, detected_encoding = await self._read_csv_file(
                file_path, has_header, delimiter, quotechar, encoding
            )

            if not rows:
                logger.warning(f"CSV file {file_path} appears to be empty")
                return

            # Prepare headers
            headers = detected_header if detected_header else [f"col_{i+1}" for i in range(len(rows[0]))]

            # Convert rows to text chunks
            text_content = self._rows_to_text(
                rows, headers, max_rows_per_chunk, include_row_numbers
            )

            # Create base metadata
            base_metadata = self.get_base_metadata(file_path)
            base_metadata.update({
                'encoding': detected_encoding,
                'content_type': 'csv',
                'delimiter': detected_delimiter,
                'quotechar': quotechar,
                'has_header': bool(detected_header),
                'column_count': len(headers),
                'row_count': len(rows),
                'headers': headers,
                'max_rows_per_chunk': max_rows_per_chunk,
                'include_row_numbers': include_row_numbers
            })

            # Chunk the text content
            chunks = self.chunker.chunk_text(text_content, base_metadata)

            for i, (chunk_text, chunk_metadata) in enumerate(chunks):
                yield DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=i,
                    source_file=file_path
                )

        except Exception as e:
            logger.error(f"Error extracting from CSV file {file_path}: {type(e).__name__} - {str(e)}")
            raise ValueError(f"Failed to extract CSV content: {str(e)}") from e

    async def _read_csv_file(
        self,
        file_path: Path,
        has_header: Optional[bool],
        delimiter: Optional[str],
        quotechar: str,
        encoding: Optional[str]
    ) -> tuple:
        """
        Read and parse CSV file with automatic detection.

        Args:
            file_path: Path to the CSV file
            has_header: Whether file has headers
            delimiter: CSV delimiter
            quotechar: Quote character
            encoding: Text encoding

        Returns:
            Tuple of (rows, headers, detected_delimiter, detected_encoding)
        """
        def _read_sync():
            # Auto-detect encoding
            detected_encoding = encoding
            if detected_encoding is None:
                try:
                    # Try UTF-8 first
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(1024)
                    detected_encoding = 'utf-8'
                except UnicodeDecodeError:
                    # Fallback to other encodings
                    encodings_to_try = ['utf-16', 'utf-32', 'latin-1', 'cp1252']
                    for enc in encodings_to_try:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                sample = f.read(1024)
                            detected_encoding = enc
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # Last resort
                        detected_encoding = 'utf-8'

            # Read full file content
            with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
                content = f.read()

            # Auto-detect delimiter if not specified
            detected_delimiter = delimiter
            if detected_delimiter is None:
                detected_delimiter = self._detect_delimiter(content)

            # Parse CSV
            csv_reader = csv.reader(
                io.StringIO(content),
                delimiter=detected_delimiter,
                quotechar=quotechar
            )

            rows = list(csv_reader)

            # Remove empty rows
            rows = [row for row in rows if any(cell.strip() for cell in row)]

            if not rows:
                return [], None, detected_delimiter, detected_encoding

            # Auto-detect headers if not specified
            detected_header = None
            if has_header is None:
                # Simple heuristic: if first row contains mostly strings and no numbers,
                # assume it's a header
                first_row = rows[0]
                if len(first_row) > 1:  # Need at least 2 columns
                    non_numeric_count = sum(1 for cell in first_row if not self._is_numeric(cell))
                    if non_numeric_count >= len(first_row) * 0.7:  # 70% non-numeric
                        detected_header = first_row
                        rows = rows[1:]  # Remove header from data rows
            elif has_header:
                detected_header = rows[0]
                rows = rows[1:]

            return rows, detected_header, detected_delimiter, detected_encoding

        # Run file reading in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_sync)

    def _detect_delimiter(self, content: str) -> str:
        """
        Auto-detect CSV delimiter by analyzing sample content.

        Args:
            content: CSV content sample

        Returns:
            Detected delimiter character
        """
        delimiters = [',', ';', '\t', '|', ':']

        # Sample first few lines
        lines = content.split('\n')[:5]
        if len(lines) < 2:
            return ','  # Default fallback

        # Count occurrences of each delimiter in each line
        delimiter_counts = {}
        for delimiter in delimiters:
            counts = []
            for line in lines:
                count = line.count(delimiter)
                counts.append(count)
            # Use the delimiter that appears consistently across lines
            if len(set(counts)) == 1 and counts[0] > 0:
                delimiter_counts[delimiter] = counts[0]

        # Return delimiter with highest consistent count
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)

        return ','  # Default fallback

    def _is_numeric(self, value: str) -> bool:
        """
        Check if a string value represents a number.

        Args:
            value: String value to check

        Returns:
            True if value appears to be numeric
        """
        try:
            float(value.replace(',', '').replace(' ', ''))
            return True
        except ValueError:
            return False

    def _rows_to_text(
        self,
        rows: List[List[str]],
        headers: List[str],
        max_rows_per_chunk: int,
        include_row_numbers: bool
    ) -> str:
        """
        Convert CSV rows to formatted text.

        Args:
            rows: CSV data rows
            headers: Column headers
            max_rows_per_chunk: Maximum rows per chunk
            include_row_numbers: Whether to include row numbers

        Returns:
            Formatted text representation
        """
        if not rows:
            return "Empty CSV file\n"

        text_parts = []

        # Process rows in chunks to avoid memory issues with large files
        for i in range(0, len(rows), max_rows_per_chunk):
            chunk_rows = rows[i:i + max_rows_per_chunk]

            for row_idx, row in enumerate(chunk_rows):
                global_row_idx = i + row_idx + 1  # 1-based row numbering

                if include_row_numbers:
                    text_parts.append(f"Row {global_row_idx}:\n")
                else:
                    text_parts.append("Row:\n")

                # Ensure row has same number of columns as headers
                while len(row) < len(headers):
                    row.append("")
                row = row[:len(headers)]  # Truncate if too many columns

                # Format each column
                for col_idx, (header, value) in enumerate(zip(headers, row)):
                    text_parts.append(f"  {header}: {value}\n")

                text_parts.append("\n")  # Blank line between rows

        return "".join(text_parts)


# Register the CSV loader
def _register_csv_loader():
    """Register CSV loader with the registry."""
    try:
        from .registry import register_loader
        register_loader(CSVLoader, ['csv'])
        logger.debug("CSV loader registered successfully")
    except ImportError:
        logger.debug("Registry not available during import")


# Auto-register when module is imported
_register_csv_loader()
