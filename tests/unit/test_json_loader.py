#!/usr/bin/env python3
"""
Unit tests for JSON document loader.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from mcp_memory_service.ingestion.json_loader import JSONLoader
from mcp_memory_service.ingestion.base import DocumentChunk


class TestJSONLoader:
    """Test suite for JSONLoader class."""

    def test_initialization(self):
        """Test basic initialization of JSONLoader."""
        loader = JSONLoader(chunk_size=500, chunk_overlap=50)

        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        assert 'json' in loader.supported_extensions

    def test_can_handle_file(self):
        """Test file format detection."""
        loader = JSONLoader()

        # Create temporary test files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            json_file.touch()

            txt_file = Path(tmpdir) / "test.txt"
            txt_file.touch()

            # Test supported formats
            assert loader.can_handle(json_file) is True

            # Test unsupported formats
            assert loader.can_handle(txt_file) is False

    @pytest.mark.asyncio
    async def test_extract_chunks_simple_json(self):
        """Test extraction from simple JSON file."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create test JSON file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            test_data = {
                "name": "John Doe",
                "age": 30,
                "city": "New York"
            }
            json_file.write_text(json.dumps(test_data, indent=2))

            chunks = []
            async for chunk in loader.extract_chunks(json_file):
                chunks.append(chunk)

            # Verify chunks were created
            assert len(chunks) > 0

            # Verify chunk structure
            first_chunk = chunks[0]
            assert isinstance(first_chunk, DocumentChunk)
            assert isinstance(first_chunk.content, str)
            assert first_chunk.source_file == json_file

            # Verify content contains flattened JSON
            content = first_chunk.content
            assert "name: John Doe" in content
            assert "age: 30" in content
            assert "city: New York" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_nested_json(self):
        """Test extraction from nested JSON file."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create test JSON file with nested structure
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            test_data = {
                "config": {
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                },
                "servers": [
                    {"name": "web", "port": 8080},
                    {"name": "api", "port": 3000}
                ]
            }
            json_file.write_text(json.dumps(test_data, indent=2))

            chunks = []
            async for chunk in loader.extract_chunks(json_file):
                chunks.append(chunk)

            # Verify chunks were created
            assert len(chunks) > 0

            # Verify content contains flattened nested structure
            content = chunks[0].content
            assert "config.database.host: localhost" in content
            assert "config.database.port: 5432" in content
            assert "servers[0].name: web" in content
            assert "servers[1].port: 3000" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_with_options(self):
        """Test extraction with various options."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create test JSON file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            test_data = {
                "user": {
                    "name": "John",
                    "details": {
                        "age": 25
                    }
                }
            }
            json_file.write_text(json.dumps(test_data, indent=2))

            # Test with bracket notation
            chunks = []
            async for chunk in loader.extract_chunks(
                json_file,
                flatten_strategy='bracket_notation'
            ):
                chunks.append(chunk)

            content = chunks[0].content
            assert "user[name]: John" in content
            assert "user[details][age]: 25" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_invalid_json(self):
        """Test handling of invalid JSON files."""
        loader = JSONLoader()

        # Create invalid JSON file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "invalid.json"
            json_file.write_text("{ invalid json content }")

            with pytest.raises(ValueError, match="Invalid JSON format"):
                async for chunk in loader.extract_chunks(json_file):
                    pass

    @pytest.mark.asyncio
    async def test_extract_chunks_empty_file(self):
        """Test handling of empty JSON files."""
        loader = JSONLoader()

        # Create empty file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "empty.json"
            json_file.write_text("")

            with pytest.raises(ValueError, match="Invalid JSON format"):
                async for chunk in loader.extract_chunks(json_file):
                    pass

    @pytest.mark.asyncio
    async def test_extract_chunks_large_nested_structure(self):
        """Test extraction from deeply nested JSON."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create deeply nested JSON
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "nested.json"
            test_data = {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "value": "deep"
                            }
                        }
                    }
                }
            }
            json_file.write_text(json.dumps(test_data, indent=2))

            chunks = []
            async for chunk in loader.extract_chunks(json_file):
                chunks.append(chunk)

            content = chunks[0].content
            assert "level1.level2.level3.level4.value: deep" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_with_arrays(self):
        """Test extraction with different array handling strategies."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create JSON with arrays
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "arrays.json"
            test_data = {
                "items": ["apple", "banana", "cherry"],
                "numbers": [1, 2, 3]
            }
            json_file.write_text(json.dumps(test_data, indent=2))

            # Test expand strategy (default)
            chunks = []
            async for chunk in loader.extract_chunks(
                json_file,
                array_handling='expand'
            ):
                chunks.append(chunk)

            content = chunks[0].content
            assert "items[0]: apple" in content
            assert "items[1]: banana" in content
            assert "numbers[0]: 1" in content

    @pytest.mark.asyncio
    async def test_extract_chunks_metadata(self):
        """Test that metadata is properly included."""
        loader = JSONLoader(chunk_size=1000, chunk_overlap=200)

        # Create test JSON file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            test_data = {"key": "value"}
            json_file.write_text(json.dumps(test_data))

            chunks = []
            async for chunk in loader.extract_chunks(json_file):
                chunks.append(chunk)

            first_chunk = chunks[0]
            assert first_chunk.metadata['content_type'] == 'json'
            assert first_chunk.metadata['encoding'] in ['utf-8', 'utf-16', 'utf-32', 'latin-1', 'cp1252']
            assert 'file_size' in first_chunk.metadata
            assert first_chunk.metadata['loader_type'] == 'JSONLoader'


class TestJSONLoaderRegistry:
    """Test JSON loader registration."""

    def test_loader_registration(self):
        """Test that JSON loader is registered."""
        from mcp_memory_service.ingestion.registry import get_loader_for_file

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON file
            json_file = Path(tmpdir) / "test.json"
            json_file.write_text('{"test": "data"}')

            loader = get_loader_for_file(json_file)

            # Should get JSONLoader
            assert loader is not None
            assert isinstance(loader, JSONLoader)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
