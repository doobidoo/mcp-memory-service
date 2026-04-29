#!/usr/bin/env python3
"""
Unit tests for MarkItDown document loader.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mcp_memory_service.ingestion import markitdown_loader as md_module
from mcp_memory_service.ingestion.markitdown_loader import MarkItDownLoader
from mcp_memory_service.ingestion.base import DocumentChunk


class TestMarkItDownLoader:
    """Test suite for MarkItDownLoader class."""

    def test_initialization(self):
        """Test basic initialization of MarkItDownLoader."""
        loader = MarkItDownLoader(chunk_size=500, chunk_overlap=50)

        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        # PDFs and the office formats are all expected to flow through markitdown
        assert 'pdf' in loader.supported_extensions
        assert 'docx' in loader.supported_extensions
        assert 'doc' in loader.supported_extensions
        assert 'pptx' in loader.supported_extensions
        assert 'ppt' in loader.supported_extensions
        assert 'xlsx' in loader.supported_extensions
        assert 'xls' in loader.supported_extensions

    def test_can_handle_returns_false_when_markitdown_unavailable(self, tmp_path):
        """can_handle must return False when markitdown is not importable,
        even for otherwise-supported extensions, so the registry can fall
        back to PDFLoader / semtools."""
        loader = MarkItDownLoader()
        loader._available = False

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        docx_file = tmp_path / "test.docx"
        docx_file.touch()

        assert loader.can_handle(pdf_file) is False
        assert loader.can_handle(docx_file) is False

    def test_can_handle_supported_extensions(self, tmp_path):
        """can_handle returns True for every supported extension when
        markitdown is available."""
        loader = MarkItDownLoader()
        loader._available = True

        for ext in ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls']:
            f = tmp_path / f"test.{ext}"
            f.touch()
            assert loader.can_handle(f) is True, f"expected True for .{ext}"

    def test_can_handle_rejects_unsupported_extensions(self, tmp_path):
        """can_handle returns False for extensions outside the supported set."""
        loader = MarkItDownLoader()
        loader._available = True

        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        csv_file = tmp_path / "test.csv"
        csv_file.touch()

        assert loader.can_handle(txt_file) is False
        assert loader.can_handle(csv_file) is False

    def test_can_handle_rejects_missing_files(self, tmp_path):
        """can_handle returns False when the file doesn't exist on disk."""
        loader = MarkItDownLoader()
        loader._available = True

        ghost = tmp_path / "does_not_exist.pdf"
        assert loader.can_handle(ghost) is False

    @pytest.mark.asyncio
    async def test_extract_chunks_raises_when_unavailable(self, tmp_path):
        """extract_chunks refuses to run when markitdown is unavailable.
        validate_file's can_handle check fires first ('not supported'),
        which is the production path we want to confirm."""
        loader = MarkItDownLoader()
        loader._available = False

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")

        with pytest.raises(ValueError, match="not supported|not installed"):
            async for _ in loader.extract_chunks(pdf_file):
                pass

    @pytest.mark.asyncio
    async def test_extract_chunks_success_yields_chunks(self, tmp_path):
        """Happy path: markitdown returns markdown, loader chunks it and
        yields DocumentChunk objects with the expected metadata."""
        loader = MarkItDownLoader(chunk_size=200, chunk_overlap=50)
        loader._available = True

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")

        # Enough content to produce at least one chunk.
        markdown_text = "# Doc Title\n\n" + ("Sentence with enough text to chunk. " * 20)

        with patch.object(loader, "_convert_with_markitdown", return_value=markdown_text):
            chunks = []
            async for chunk in loader.extract_chunks(pdf_file):
                chunks.append(chunk)

        assert len(chunks) > 0
        first = chunks[0]
        assert isinstance(first, DocumentChunk)
        assert isinstance(first.content, str)
        assert first.source_file == pdf_file
        assert first.metadata['extraction_method'] == 'markitdown'
        assert first.metadata['parser_backend'] == 'microsoft-markitdown'
        assert first.metadata['content_type'] == 'markdown'

    @pytest.mark.asyncio
    async def test_extract_chunks_emits_usage_log(self, tmp_path, caplog):
        """A summary [ingestion] log line is emitted at INFO level after
        a successful extract — this is the line we deploy-time grep for."""
        loader = MarkItDownLoader(chunk_size=200, chunk_overlap=50)
        loader._available = True

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_text("dummy")

        markdown_text = "# Title\n\n" + ("Body text repeated. " * 30)

        with patch.object(loader, "_convert_with_markitdown", return_value=markdown_text):
            with caplog.at_level(logging.INFO, logger="mcp_memory_service.ingestion.markitdown_loader"):
                async for _ in loader.extract_chunks(pdf_file):
                    pass

        usage_lines = [r.getMessage() for r in caplog.records
                       if "[ingestion] markitdown processed" in r.getMessage()]
        assert len(usage_lines) == 1, f"expected exactly one usage line, got: {usage_lines}"
        line = usage_lines[0]
        assert "file=report.pdf" in line
        assert "ext=pdf" in line
        assert "chars=" in line
        assert "chunks=" in line
        assert "took=" in line

    @pytest.mark.asyncio
    async def test_extract_chunks_propagates_conversion_error(self, tmp_path):
        """If markitdown raises during conversion, the loader surfaces a
        ValueError so the upload pipeline can record the failure."""
        loader = MarkItDownLoader()
        loader._available = True

        pdf_file = tmp_path / "broken.pdf"
        pdf_file.write_text("dummy")

        async def _raise(_path):
            raise ValueError("Failed to convert document with markitdown: boom")

        with patch.object(loader, "_convert_with_markitdown", side_effect=_raise):
            with pytest.raises(ValueError, match="markitdown"):
                async for _ in loader.extract_chunks(pdf_file):
                    pass

    @pytest.mark.asyncio
    async def test_extract_chunks_timeout_surfaces_runtime_error(self, tmp_path):
        """A markitdown call that exceeds the timeout is reported as a
        RuntimeError so callers can distinguish 'too slow' from 'malformed'."""
        loader = MarkItDownLoader()
        loader._available = True

        pdf_file = tmp_path / "slow.pdf"
        pdf_file.write_text("dummy")

        # Force the real conversion path to raise TimeoutError without
        # actually sleeping — patch asyncio.wait_for as it's looked up
        # inside the loader module.
        with patch.object(md_module.asyncio, "wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(RuntimeError, match="timed out"):
                async for _ in loader.extract_chunks(pdf_file):
                    pass

    @pytest.mark.asyncio
    async def test_extract_chunks_empty_content_raises(self, tmp_path):
        """markitdown returning an empty string is treated as a failure
        (otherwise we would silently store zero-chunk uploads)."""
        loader = MarkItDownLoader()
        loader._available = True

        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_text("dummy")

        # Stub MarkItDown to return empty content directly through the real
        # _convert_with_markitdown path.
        fake_md = MagicMock()
        fake_md.convert.return_value = MagicMock(text_content="")

        with patch.object(md_module, "MarkItDown", return_value=fake_md):
            with pytest.raises((RuntimeError, ValueError), match="empty content|Failed to convert"):
                async for _ in loader.extract_chunks(pdf_file):
                    pass


class TestMarkItDownLoaderRegistry:
    """Test markitdown loader registration behavior."""

    def test_register_only_when_available(self):
        """_register_markitdown_loader must be a no-op when markitdown is
        not importable, so PDFLoader / semtools remain registered."""
        from mcp_memory_service.ingestion import registry

        # Snapshot + clear so we observe registration in isolation.
        original = dict(registry._LOADER_REGISTRY)
        registry._LOADER_REGISTRY.clear()
        try:
            saved_flag = md_module._MARKITDOWN_AVAILABLE
            md_module._MARKITDOWN_AVAILABLE = False
            try:
                md_module._register_markitdown_loader()
            finally:
                md_module._MARKITDOWN_AVAILABLE = saved_flag

            assert registry._LOADER_REGISTRY == {}, (
                "no extensions should be registered when markitdown unavailable"
            )
        finally:
            registry._LOADER_REGISTRY.clear()
            registry._LOADER_REGISTRY.update(original)

    def test_register_claims_pdf_and_office_extensions(self):
        """When available, markitdown registers itself for pdf + the office
        formats. Run in isolation (snapshot registry) so we don't depend on
        whatever was registered during package import."""
        from mcp_memory_service.ingestion import registry

        original = dict(registry._LOADER_REGISTRY)
        registry._LOADER_REGISTRY.clear()
        try:
            saved_flag = md_module._MARKITDOWN_AVAILABLE
            md_module._MARKITDOWN_AVAILABLE = True
            try:
                md_module._register_markitdown_loader()
            finally:
                md_module._MARKITDOWN_AVAILABLE = saved_flag

            for ext in ('pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls'):
                assert ext in registry._LOADER_REGISTRY, f".{ext} should be registered"
                assert registry._LOADER_REGISTRY[ext] is MarkItDownLoader
        finally:
            registry._LOADER_REGISTRY.clear()
            registry._LOADER_REGISTRY.update(original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
