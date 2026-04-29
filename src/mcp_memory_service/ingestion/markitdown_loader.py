## Copyright 2024 Heinrich Krupp
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0

"""
MarkItDown document loader — converts Office formats (docx, pptx, xlsx) and
PDFs to markdown using Microsoft's `markitdown` library, then chunks the
result.

Pure-Python alternative to `semtools_loader` for teams that don't want a
LlamaParse API key. When `markitdown` is installed, it registers as the
primary loader for docx/doc/pptx/xlsx AND pdf. When not installed, it
skips registration entirely so PDFLoader / semtools remain the fallback.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)

try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    _MARKITDOWN_AVAILABLE = False


class MarkItDownLoader(DocumentLoader):
    """Loader that uses Microsoft markitdown to convert Office docs to markdown."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls']
        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=True,
        ))
        self._available = _MARKITDOWN_AVAILABLE
        if not self._available:
            logger.debug(
                "markitdown not installed — install with: pip install 'markitdown[docx,pptx,xlsx]'"
            )

    def can_handle(self, file_path: Path) -> bool:
        if not self._available:
            return False
        ext = file_path.suffix.lower().lstrip('.')
        return (
            ext in self.supported_extensions
            and file_path.exists()
            and file_path.is_file()
        )

    async def extract_chunks(
        self, file_path: Path, **kwargs
    ) -> AsyncGenerator[DocumentChunk, None]:
        await self.validate_file(file_path)

        if not self._available:
            raise ValueError(
                "markitdown is not installed. "
                "Install with: pip install 'markitdown[docx,pptx,xlsx]'"
            )

        ext = file_path.suffix.lower().lstrip('.')
        logger.info(f"Extracting chunks from {file_path} using markitdown")

        start = time.monotonic()
        markdown_content = await self._convert_with_markitdown(file_path)

        base_metadata = self.get_base_metadata(file_path)
        base_metadata.update({
            'extraction_method': 'markitdown',
            'parser_backend': 'microsoft-markitdown',
            'content_type': 'markdown',
        })

        chunks = self.chunker.chunk_text(markdown_content, base_metadata)
        chunk_count = 0
        for idx, (chunk_text, chunk_metadata) in enumerate(chunks):
            chunk_count += 1
            yield DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=idx,
                source_file=file_path,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[ingestion] markitdown processed file=%s ext=%s chars=%d chunks=%d took=%dms",
            file_path.name,
            ext,
            len(markdown_content),
            chunk_count,
            elapsed_ms,
        )

    async def _convert_with_markitdown(self, file_path: Path) -> str:
        """Run markitdown in a thread to avoid blocking the event loop."""
        def _convert() -> str:
            md = MarkItDown(enable_plugins=False)
            result = md.convert(str(file_path))
            return result.text_content or ""

        try:
            content = await asyncio.wait_for(
                asyncio.to_thread(_convert),
                timeout=300,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"markitdown conversion timed out for {file_path}") from exc
        except Exception as exc:
            logger.error(f"markitdown conversion failed for {file_path}: {exc}")
            raise ValueError(f"Failed to convert document with markitdown: {exc}") from exc

        if not content.strip():
            raise RuntimeError(f"markitdown returned empty content for {file_path}")

        logger.debug(f"markitdown extracted {len(content)} chars from {file_path}")
        return content


def _register_markitdown_loader() -> None:
    if not _MARKITDOWN_AVAILABLE:
        logger.debug(
            "markitdown not installed — skipping registration; "
            "install with: pip install 'markitdown[docx,pptx,xlsx]'"
        )
        return
    try:
        from .registry import register_loader
        register_loader(
            MarkItDownLoader,
            ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls'],
        )
        logger.info(
            "[ingestion] markitdown registered for: pdf, docx, doc, pptx, ppt, xlsx, xls"
        )
    except ImportError:
        logger.debug("Registry not available during import")


_register_markitdown_loader()
