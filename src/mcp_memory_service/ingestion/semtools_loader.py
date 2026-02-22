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
Semtools document loader for enhanced text extraction using Rust-based parser.

Uses semtools CLI (https://github.com/run-llama/semtools) for superior document
parsing with LlamaParse API integration. Supports PDF, DOCX, PPTX and other formats.
"""

import logging
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator
import shutil

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


class SemtoolsLoader(DocumentLoader):
    """
    Document loader using semtools for superior text extraction.

    Leverages semtools' Rust-based parser with LlamaParse API for:
    - Advanced OCR capabilities
    - Table extraction
    - Multi-format support (PDF, DOCX, PPTX, etc.)

    Falls back gracefully when semtools is not available.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize Semtools loader.

        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['pdf', 'docx', 'doc', 'pptx', 'xlsx']
        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=True
        ))

        # Check semtools availability
        self._semtools_available = self._check_semtools_availability()

        # Get API key from environment
        self.api_key = os.getenv('LLAMAPARSE_API_KEY')
        if self._semtools_available and not self.api_key:
            logger.warning(
                "Semtools is available but LLAMAPARSE_API_KEY not set. "
                "Document parsing quality may be limited."
            )

    def _check_semtools_availability(self) -> bool:
        """
        Check if semtools is installed and available.

        Returns:
            True if semtools CLI is available
        """
        semtools_path = shutil.which('semtools')
        if semtools_path:
            logger.info(f"Semtools found at: {semtools_path}")
            return True
        else:
            logger.debug(
                "Semtools not available. Install with: npm i -g @llamaindex/semtools "
                "or cargo install semtools"
            )
            return False

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if semtools is available and file format is supported
        """
        if not self._semtools_available:
            return False

        return (file_path.suffix.lower().lstrip('.') in self.supported_extensions and
                file_path.exists() and
                file_path.is_file())

    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a document using semtools.

        Args:
            file_path: Path to the document file
            **kwargs: Additional options (currently unused)

        Yields:
            DocumentChunk objects containing parsed content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If semtools is not available or parsing fails
        """
        await self.validate_file(file_path)

        if not self._semtools_available:
            raise ValueError(
                "Semtools is not available. Install with: npm i -g @llamaindex/semtools"
            )

        logger.info(f"Extracting chunks from {file_path} using semtools")

        try:
            # Parse document to markdown using semtools
            markdown_content = await self._parse_with_semtools(file_path)

            # Get base metadata
            base_metadata = self.get_base_metadata(file_path)
            base_metadata.update({
                'extraction_method': 'semtools',
                'parser_backend': 'llamaparse',
                'content_type': 'markdown',
                'has_api_key': bool(self.api_key)
            })

            # Chunk the markdown content
            chunks = self.chunker.chunk_text(markdown_content, base_metadata)

            chunk_index = 0
            for chunk_text, chunk_metadata in chunks:
                yield DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    source_file=file_path
                )
                chunk_index += 1

        except Exception as e:
            logger.error(f"Error processing {file_path} with semtools: {e}")
            raise ValueError(f"Failed to parse document: {str(e)}") from e

    async def _parse_with_semtools(self, file_path: Path) -> str:
        """
        Parse document using semtools CLI.

        Args:
            file_path: Path to document to parse

        Returns:
            Markdown content extracted from document

        Raises:
            RuntimeError: If semtools command fails
        """
        # Prepare semtools command
        cmd = ['semtools', 'parse', str(file_path)]

        # Set up environment with API key if available
        env = os.environ.copy()
        if self.api_key:
            env['LLAMAPARSE_API_KEY'] = self.api_key

        try:
            # Run semtools parse command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minute timeout for large documents
            )

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace')
                logger.error(f"Semtools parsing failed: {error_msg}")
                raise RuntimeError(f"Semtools returned error: {error_msg}")

            # Parse markdown output
            markdown_content = stdout.decode('utf-8', errors='replace')

            if not markdown_content.strip():
                logger.warning(f"Semtools returned empty content for {file_path}")
                raise RuntimeError("Semtools returned empty content")

            logger.debug(f"Successfully parsed {file_path}, extracted {len(markdown_content)} characters")
            return markdown_content

        except asyncio.TimeoutError:
            logger.error(f"Semtools parsing timed out for {file_path}")
            raise RuntimeError("Document parsing timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Error running semtools: {e}")
            raise


# Register the semtools loader
def _register_semtools_loader():
    """Register semtools loader with the registry."""
    try:
        from .registry import register_loader
        register_loader(SemtoolsLoader, ['pdf', 'docx', 'doc', 'pptx', 'xlsx'])
        logger.debug("Semtools loader registered successfully")
    except ImportError:
        logger.debug("Registry not available during import")


# Auto-register when module is imported
_register_semtools_loader()
