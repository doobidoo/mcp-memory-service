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

        with patch("mcp_memory_service.ingestion.get_loader_for_file", return_value=mock_loader), \
             patch("mcp_memory_service.utils.document_processing._process_and_store_chunk", new_callable=AsyncMock) as mock_store:
            mock_store.return_value = (True, None)

            await processor.process_file(
                file_path=Path("sample.md"),
                file_index=1,
                total_files=1,
                directory_name="test_dir",
            )

            assert mock_loader.chunk_size == 600
            assert mock_loader.chunk_overlap == 75

            mock_store.assert_called_once()
            _, kwargs = mock_store.call_args
            assert kwargs.get("memory_type") == "reference"
            assert kwargs.get("store") == "notes"


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
