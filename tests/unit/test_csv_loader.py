#!/usr/bin/env python3
"""
Unit tests for CSV document loader.
"""

import pytest
import asyncio
import csv
import io
from pathlib import Path

from mcp_memory_service.ingestion.csv_loader import CSVLoader
from mcp_memory_service.ingestion.base import DocumentChunk


class TestCSVLoader:
    """Test suite for CSVLoader class."""

    def test_initialization(self):
        """Test basic initialization of CSVLoader."""
        loader = CSVLoader(chunk_size=500, chunk_overlap=50)

        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        assert 'csv' in loader.supported_extensions

    def test_can_handle_file(self):
        """Test file format detection."""
        loader = CSVLoader()

        # Create temporary test files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_file.touch()

            txt_file = Path(tmpdir) / "test.txt"
            txt_file.touch()

            # Test supported formats
            assert loader.can_handle(csv_file) is True

            # Test unsupported formats
            assert loader.can_handle(txt_file) is False

    @pytest.mark.asyncio
    async def test_extract_chunks_simple_csv(self):
        """Test extraction from simple CSV file."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """name,age,city
John,25,New York
Jane,30,San Francisco"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            # Verify chunks were created
            assert len(chunks) > 0

            # Verify chunk structure
            first_chunk = chunks[0]
            assert isinstance(first_chunk, DocumentChunk)
            assert isinstance(first_chunk.content, str)
            assert first_chunk.source_file == csv_file

            # Verify content contains formatted rows
            content = first_chunk.content
            assert "name: John" in content
            assert "age: 25" in content
            assert "city: New York" in content
            assert "name: Jane" in content
            assert "age: 30" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_csv_with_headers(self):
        """Test extraction from CSV with header detection."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file with headers
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """product,price,category
Widget,19.99,Electronics
Gadget,29.99,Electronics
Book,12.99,Media"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            content = chunks[0].content
            assert "product: Widget" in content
            assert "price: 19.99" in content
            assert "category: Electronics" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_csv_no_headers(self):
        """Test extraction from CSV without headers."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file without headers
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """John,25,New York
Jane,30,San Francisco"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file, has_header=False):
                chunks.append(chunk)

            content = chunks[0].content
            # Should use col_1, col_2, col_3 as headers
            assert "col_1: John" in content
            assert "col_2: 25" in content
            assert "col_3: New York" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_different_delimiters(self):
        """Test extraction with different CSV delimiters."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Test semicolon delimiter
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = "name;age;city\nJohn;25;New York\nJane;30;San Francisco"
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file, delimiter=';'):
                chunks.append(chunk)

            content = chunks[0].content
            assert "name: John" in content
            assert "age: 25" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_row_numbers(self):
        """Test extraction with row numbers."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """name,age
John,25
Jane,30"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file, include_row_numbers=True):
                chunks.append(chunk)

            content = chunks[0].content
            assert "Row 1:" in content
            assert "Row 2:" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_no_row_numbers(self):
        """Test extraction without row numbers."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """name,age
John,25"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file, include_row_numbers=False):
                chunks.append(chunk)

            content = chunks[0].content
            assert "Row:" in content
            assert "Row 1:" not in content

    @pytest.mark.asyncio
    async def test_extract_chunks_large_file_chunking(self):
        """Test that large CSV files are processed correctly."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create CSV with many rows
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "large.csv"
            rows = ["name,value"] + [f"item{i},{i}" for i in range(10)]
            csv_content = "\n".join(rows)
            csv_file.write_text(csv_content)

            # Process the file
            chunks = []
            async for chunk in loader.extract_chunks(csv_file, max_rows_per_chunk=50):
                chunks.append(chunk)

            # Should create at least one chunk
            assert len(chunks) >= 1

            # Verify all content is included
            all_content = "".join(chunk.content for chunk in chunks)
            assert "item0" in all_content
            assert "item9" in all_content
            assert "name: item0" in all_content
            assert "value: 0" in all_content

    @pytest.mark.asyncio
    async def test_extract_chunks_empty_file(self):
        """Test handling of empty CSV files."""
        loader = CSVLoader()

        # Create empty CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "empty.csv"
            csv_file.write_text("")

            # Should not raise error but return no chunks
            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_extract_chunks_malformed_csv(self):
        """Test handling of malformed CSV files."""
        loader = CSVLoader()

        # Create malformed CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "malformed.csv"
            # CSV with inconsistent columns - should still work
            csv_content = """name,age,city
John,25
Jane,30,San Francisco,Extra"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            # Should handle gracefully
            assert len(chunks) > 0
            content = chunks[0].content
            assert "name: John" in content
            assert "name: Jane" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_encoding_detection(self):
        """Test automatic encoding detection."""
        loader = CSVLoader()

        # Create CSV file with UTF-8 content
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "utf8.csv"
            csv_content = """name,city
José,São Paulo
François,Montréal"""
            csv_file.write_text(csv_content, encoding='utf-8')

            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            content = chunks[0].content
            assert "José" in content
            assert "São Paulo" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_metadata(self):
        """Test that metadata is properly included."""
        loader = CSVLoader(chunk_size=1000, chunk_overlap=200)

        # Create test CSV file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_content = """name,age
John,25
Jane,30"""
            csv_file.write_text(csv_content)

            chunks = []
            async for chunk in loader.extract_chunks(csv_file):
                chunks.append(chunk)

            first_chunk = chunks[0]
            assert first_chunk.metadata['content_type'] == 'csv'
            assert first_chunk.metadata['has_header'] is True
            assert first_chunk.metadata['column_count'] == 2
            assert first_chunk.metadata['row_count'] == 2
            assert first_chunk.metadata['headers'] == ['name', 'age']
            assert 'file_size' in first_chunk.metadata
            assert first_chunk.metadata['loader_type'] == 'CSVLoader'


class TestCSVLoaderRegistry:
    """Test CSV loader registration."""

    def test_loader_registration(self):
        """Test that CSV loader is registered."""
        from mcp_memory_service.ingestion.registry import get_loader_for_file

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV file
            csv_file = Path(tmpdir) / "test.csv"
            csv_file.write_text("name,value\nJohn,25")

            loader = get_loader_for_file(csv_file)

            # Should get CSVLoader
            assert loader is not None
            assert isinstance(loader, CSVLoader)


class TestCSVDelimiterDetection:
    """Test CSV delimiter detection."""

    def test_detect_delimiter_comma(self):
        """Test comma delimiter detection."""
        loader = CSVLoader()
        content = "name,age,city\nJohn,25,New York\nJane,30,San Francisco"
        delimiter = loader._detect_delimiter(content)
        assert delimiter == ','

    def test_detect_delimiter_semicolon(self):
        """Test semicolon delimiter detection."""
        loader = CSVLoader()
        content = "name;age;city\nJohn;25;New York\nJane;30;San Francisco"
        delimiter = loader._detect_delimiter(content)
        assert delimiter == ';'

    def test_detect_delimiter_tab(self):
        """Test tab delimiter detection."""
        loader = CSVLoader()
        content = "name\tage\tcity\nJohn\t25\tNew York\nJane\t30\tSan Francisco"
        delimiter = loader._detect_delimiter(content)
        assert delimiter == '\t'

    def test_detect_delimiter_pipe(self):
        """Test pipe delimiter detection."""
        loader = CSVLoader()
        content = "name|age|city\nJohn|25|New York\nJane|30|San Francisco"
        delimiter = loader._detect_delimiter(content)
        assert delimiter == '|'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
