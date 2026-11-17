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
Tests for directory ingestion parameter propagation (Issue #2040).

Verifies that memory_type and chunk_overlap are properly accepted and
forwarded to document loaders and memory creation during directory ingestion.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import mcp_memory_service.ingestion
from mcp_memory_service.utils.directory_ingestion import FileIngestionProcessor
from mcp_memory_service.server.handlers.documents import handle_ingest_directory


class TestFileIngestionProcessorParams:
    """Test parameter handling in FileIngestionProcessor."""

    def test_default_parameters(self):
        """Verify default values for chunk_overlap and memory_type."""
        storage = MagicMock()
        processor = FileIngestionProcessor(
            storage=storage,
            chunk_size=1000,
            base_tags=["tag1"],
        )
        assert processor.chunk_overlap == 200
        assert processor.memory_type == "document"
        assert processor.store == "default"

    def test_custom_parameters(self):
        """Verify custom chunk_overlap and memory_type are stored."""
        storage = MagicMock()
        processor = FileIngestionProcessor(
            storage=storage,
            chunk_size=500,
            base_tags=["tag1"],
            chunk_overlap=50,
            memory_type="learning",
            store="custom_store",
        )
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.memory_type == "learning"
        assert processor.store == "custom_store"

    @pytest.mark.asyncio
    async def test_process_file_configures_loader_and_memory_type(self):
        """Verify process_file sets loader.chunk_overlap and forwards memory_type."""
        mock_storage = AsyncMock()
        processor = FileIngestionProcessor(
            storage=mock_storage,
            chunk_size=600,
            base_tags=["base"],
            chunk_overlap=75,
            memory_type="reference",
            store="notes",
        )

        mock_loader = MagicMock()
        mock_chunk = MagicMock(content="chunk text", chunk_index=0, metadata={})

        async def _extract(_path):
            yield mock_chunk

        mock_loader.extract_chunks = _extract

        with patch("mcp_memory_service.ingestion.get_loader_for_file", return_value=mock_loader) as mock_get_loader, \
             patch("mcp_memory_service.utils.document_processing._process_and_store_chunk", new_callable=AsyncMock) as mock_store:
            mock_store.return_value = (True, None)

            await processor.process_file(
                file_path=Path("sample.md"),
                file_index=1,
                total_files=1,
                directory_name="test_dir",
            )

            mock_get_loader.assert_called_once_with(
                Path("sample.md"),
                chunk_size=600,
                chunk_overlap=75,
            )

            mock_store.assert_called_once()
            _, kwargs = mock_store.call_args
            assert kwargs.get("memory_type") == "reference"
            assert kwargs.get("store") == "notes"

    def test_get_loader_for_file_constructs_with_custom_overlap(self, tmp_path):
        """Verify get_loader_for_file passes chunk_size and chunk_overlap to loader constructor."""
        from mcp_memory_service.ingestion import get_loader_for_file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello", encoding="utf-8")

        loader = get_loader_for_file(test_file, chunk_size=350, chunk_overlap=60)
        assert loader is not None
        assert loader.chunk_size == 350
        assert loader.chunk_overlap == 60
        assert loader.chunker.strategy.chunk_size == 350
        assert loader.chunker.strategy.chunk_overlap == 60

    @pytest.mark.asyncio
    async def test_process_file_with_real_loader_applies_chunk_overlap(self, tmp_path):
        """Verify process_file uses a real loader and actually applies custom chunk_overlap to chunks."""
        mock_storage = AsyncMock()
        processor = FileIngestionProcessor(
            storage=mock_storage,
            chunk_size=400,
            base_tags=["base"],
            chunk_overlap=80,
            memory_type="reference",
            store="notes",
        )

        paragraphs = [
            f"Paragraph {i}: " + "Detailed discussion of ingestion topic with sufficient content. " * 4
            for i in range(8)
        ]
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("\n\n".join(paragraphs), encoding="utf-8")

        stored_chunks = []

        async def _capture_store(chunk, *args, **kwargs):
            stored_chunks.append((chunk, kwargs))
            return (True, None)

        with patch("mcp_memory_service.utils.document_processing._process_and_store_chunk", side_effect=_capture_store):
            await processor.process_file(
                file_path=sample_file,
                file_index=1,
                total_files=1,
                directory_name="test_dir",
            )

        assert len(stored_chunks) > 1

        for chunk, kwargs in stored_chunks:
            assert kwargs.get("memory_type") == "reference"
            assert kwargs.get("store") == "notes"

        # Verify consecutive chunks overlap
        chunk0_content = stored_chunks[0][0].content
        chunk1_content = stored_chunks[1][0].content
        overlap_snippet = chunk0_content[-50:].strip()
        assert any(word in chunk1_content for word in overlap_snippet.split() if len(word) > 4)

    def test_text_chunker_zero_overlap_returns_empty_string(self):
        """Verify _get_overlap_text returns empty string when chunk_overlap is 0."""
        from mcp_memory_service.ingestion.chunker import TextChunker, ChunkingStrategy
        strategy = ChunkingStrategy(chunk_size=300, chunk_overlap=0)
        chunker = TextChunker(strategy)
        text = "This is a paragraph with several sentences. Another sentence follows. Final sentence here."
        assert chunker._get_overlap_text(text) == ""

    @pytest.mark.asyncio
    async def test_process_file_with_zero_overlap_has_no_repeated_paragraphs(self, tmp_path):
        """Verify chunk_overlap=0 produces distinct chunks without prepending earlier paragraphs."""
        mock_storage = AsyncMock()
        processor = FileIngestionProcessor(
            storage=mock_storage,
            chunk_size=300,
            base_tags=["base"],
            chunk_overlap=0,
            memory_type="reference",
            store="notes",
        )

        p0 = "Alpha Section: " + "First distinct paragraph details. " * 5
        p1 = "Beta Section: " + "Second distinct paragraph details. " * 5
        p2 = "Gamma Section: " + "Third distinct paragraph details. " * 5
        sample_file = tmp_path / "distinct.txt"
        sample_file.write_text(f"{p0}\n\n{p1}\n\n{p2}", encoding="utf-8")

        stored_chunks = []

        async def _capture_store(chunk, *args, **kwargs):
            stored_chunks.append(chunk.content)
            return (True, None)

        with patch("mcp_memory_service.utils.document_processing._process_and_store_chunk", side_effect=_capture_store):
            await processor.process_file(
                file_path=sample_file,
                file_index=1,
                total_files=1,
                directory_name="test_dir",
            )

        assert len(stored_chunks) >= 2
        # Verify chunk 1 does not repeat any content from paragraph 0 (Alpha)
        assert "Alpha Section" in stored_chunks[0]
        assert "Alpha Section" not in stored_chunks[1]
        if len(stored_chunks) > 2:
            assert "Beta Section" not in stored_chunks[2]


class TestHandleIngestDirectoryParams:
    """Test parameter parsing in handle_ingest_directory."""

    @pytest.mark.asyncio
    async def test_handle_ingest_directory_forwards_memory_type_and_overlap(self, tmp_path):
        """Verify handle_ingest_directory extracts chunk_overlap and memory_type."""
        server = MagicMock()
        server._ensure_storage_initialized = AsyncMock(return_value=AsyncMock())

        doc = tmp_path / "notes.txt"
        doc.write_text("sample content")

        captured = []
        original_init = FileIngestionProcessor.__init__

        def _capture_init(self, *args, **kwargs):
            captured.append(kwargs)
            original_init(self, *args, **kwargs)

        with patch.object(FileIngestionProcessor, "__init__", _capture_init), \
             patch.object(FileIngestionProcessor, "process_file", new_callable=AsyncMock):
            arguments = {
                "directory_path": str(tmp_path),
                "chunk_size": 400,
                "chunk_overlap": 80,
                "memory_type": "decision",
                "store": "project",
                "recursive": False,
            }

            res = await handle_ingest_directory(server, arguments)

            assert len(captured) == 1
            params = captured[0]
            assert params["chunk_size"] == 400
            assert params["chunk_overlap"] == 80
            assert params["memory_type"] == "decision"
            assert params["store"] == "project"

    @pytest.mark.asyncio
    async def test_handle_ingest_directory_default_params(self, tmp_path):
        """Verify handle_ingest_directory uses defaults when parameters are omitted."""
        server = MagicMock()
        server._ensure_storage_initialized = AsyncMock(return_value=AsyncMock())

        doc = tmp_path / "readme.md"
        doc.write_text("readme")

        captured = []
        original_init = FileIngestionProcessor.__init__

        def _capture_init(self, *args, **kwargs):
            captured.append(kwargs)
            original_init(self, *args, **kwargs)

        with patch.object(FileIngestionProcessor, "__init__", _capture_init), \
             patch.object(FileIngestionProcessor, "process_file", new_callable=AsyncMock):
            arguments = {
                "directory_path": str(tmp_path),
                "recursive": False,
            }

            res = await handle_ingest_directory(server, arguments)

            assert len(captured) == 1
            params = captured[0]
            assert params["chunk_size"] == 1000
            assert params["chunk_overlap"] == 200
            assert params["memory_type"] == "document"
            assert params["store"] == "default"
