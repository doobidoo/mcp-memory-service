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
Text document loader for plain text and Markdown files.
"""

import logging
import re
import chardet
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional
import asyncio

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


class TextLoader(DocumentLoader):
    """
    Document loader for plain text and Markdown files.
    
    Features:
    - Automatic encoding detection
    - Markdown structure preservation
    - Section-aware chunking
    - Code block handling
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text loader.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['txt', 'md', 'markdown', 'rst', 'text']
        
        # Markdown patterns
        self.md_header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.md_code_block_pattern = re.compile(r'^```[\s\S]*?^```', re.MULTILINE)
        self.md_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=True,
            respect_sentence_boundaries=True
        ))
    
    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given text file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this loader can process the text file
        """
        if not file_path.exists() or not file_path.is_file():
            return False
        
        extension = file_path.suffix.lower().lstrip('.')
        return extension in self.supported_extensions
    
    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a text file.
        
        Args:
            file_path: Path to the text file
            **kwargs: Additional options:
                - encoding: Text encoding to use (auto-detected if not specified)
                - preserve_structure: Whether to preserve Markdown structure (default: True)
                - extract_links: Whether to extract and preserve links (default: False)
            
        Yields:
            DocumentChunk objects containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If the text file doesn't exist
            ValueError: If the text file can't be read or processed
        """
        await self.validate_file(file_path)
        
        encoding = kwargs.get('encoding', None)
        preserve_structure = kwargs.get('preserve_structure', True)
        extract_links = kwargs.get('extract_links', False)
        
        logger.info(f"Extracting chunks from text file: {file_path}")
        
        try:
            # Read file content
            content, detected_encoding = await self._read_file_content(file_path, encoding)
            
            # Determine file type
            is_markdown = file_path.suffix.lower() in ['.md', '.markdown']
            
            # Process content based on type
            if is_markdown and preserve_structure:
                async for chunk in self._extract_markdown_chunks(
                    file_path, content, detected_encoding, extract_links
                ):
                    yield chunk
            else:
                async for chunk in self._extract_text_chunks(
                    file_path, content, detected_encoding
                ):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error extracting from text file {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract text content: {str(e)}") from e
    
    async def _read_file_content(self, file_path: Path, encoding: Optional[str]) -> tuple:
        """
        Read file content with encoding detection.
        
        Args:
            file_path: Path to the file
            encoding: Specific encoding to use, or None for auto-detection
            
        Returns:
            Tuple of (content, encoding_used)
        """
        def _read_sync():
            # Auto-detect encoding if not specified
            if encoding is None:
                # For markdown files, default to UTF-8 as it's the standard
                if file_path.suffix.lower() in ['.md', '.markdown']:
                    file_encoding = 'utf-8'
                else:
                    with open(file_path, 'rb') as file:
                        raw_data = file.read()
                    detected = chardet.detect(raw_data)
                    file_encoding = detected['encoding'] or 'utf-8'
            else:
                file_encoding = encoding
            
            # Read with detected/specified encoding
            try:
                with open(file_path, 'r', encoding=file_encoding) as file:
                    content = file.read()
                return content, file_encoding
            except UnicodeDecodeError:
                # Fallback to UTF-8 with error handling
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                return content, 'utf-8'
        
        # Run file reading in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_sync)
    
    async def _extract_text_chunks(
        self, 
        file_path: Path, 
        content: str, 
        encoding: str
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract chunks from plain text.
        
        Args:
            file_path: Path to the source file
            content: File content
            encoding: Encoding used to read the file
            
        Yields:
            DocumentChunk objects
        """
        base_metadata = self.get_base_metadata(file_path)
        base_metadata.update({
            'encoding': encoding,
            'content_type': 'plain_text',
            'total_characters': len(content),
            'total_lines': content.count('\n') + 1
        })
        
        # Chunk the content
        chunks = self.chunker.chunk_text(content, base_metadata)
        
        for i, (chunk_text, chunk_metadata) in enumerate(chunks):
            yield DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
                source_file=file_path
            )
    
    async def _extract_markdown_chunks(
        self, 
        file_path: Path, 
        content: str, 
        encoding: str,
        extract_links: bool
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract chunks from Markdown with structure preservation.
        
        Args:
            file_path: Path to the source file
            content: File content
            encoding: Encoding used to read the file
            extract_links: Whether to extract and preserve links
            
        Yields:
            DocumentChunk objects
        """
        base_metadata = self.get_base_metadata(file_path)
        base_metadata.update({
            'encoding': encoding,
            'content_type': 'markdown',
            'total_characters': len(content),
            'total_lines': content.count('\n') + 1
        })
        
        # Extract Markdown structure
        headers = self._extract_headers(content)
        code_blocks = self._extract_code_blocks(content)
        links = self._extract_links(content) if extract_links else []
        
        # Add structural metadata
        base_metadata.update({
            'header_count': len(headers),
            'code_block_count': len(code_blocks),
            'link_count': len(links)
        })
        
        # Use section-aware chunking for Markdown
        chunks = self.chunker.chunk_by_sections(content, base_metadata)
        
        for i, (chunk_text, chunk_metadata) in enumerate(chunks):
            # Add Markdown-specific metadata to each chunk
            chunk_headers = self._get_chunk_headers(chunk_text, headers)
            chunk_metadata.update({
                'markdown_headers': chunk_headers,
                'has_code_blocks': bool(self.md_code_block_pattern.search(chunk_text)),
                'chunk_links': self._get_chunk_links(chunk_text) if extract_links else []
            })
            
            yield DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
                source_file=file_path
            )
    
    def _extract_headers(self, content: str) -> list:
        """
        Extract Markdown headers from content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of header dictionaries with level, text, and position
        """
        headers = []
        for match in self.md_header_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            position = match.start()
            headers.append({
                'level': level,
                'text': text,
                'position': position
            })
        return headers
    
    def _extract_code_blocks(self, content: str) -> list:
        """
        Extract code blocks from Markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of code block dictionaries
        """
        code_blocks = []
        for match in self.md_code_block_pattern.finditer(content):
            block = match.group(0)
            # Extract language if specified
            first_line = block.split('\n')[0]
            language = first_line[3:].strip() if len(first_line) > 3 else ''
            
            code_blocks.append({
                'language': language,
                'content': block,
                'position': match.start(),
                'length': len(block)
            })
        return code_blocks
    
    def _extract_links(self, content: str) -> list:
        """
        Extract links from Markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of link dictionaries
        """
        links = []
        for match in self.md_link_pattern.finditer(content):
            text = match.group(1)
            url = match.group(2)
            position = match.start()
            links.append({
                'text': text,
                'url': url,
                'position': position
            })
        return links
    
    def _get_chunk_headers(self, chunk_text: str, all_headers: list) -> list:
        """
        Get headers that appear in a specific chunk.
        
        Args:
            chunk_text: The text chunk to analyze
            all_headers: All headers from the document
            
        Returns:
            List of headers found in this chunk
        """
        chunk_headers = []
        for header in all_headers:
            if header['text'] in chunk_text:
                chunk_headers.append({
                    'level': header['level'],
                    'text': header['text']
                })
        return chunk_headers
    
    def _get_chunk_links(self, chunk_text: str) -> list:
        """
        Get links that appear in a specific chunk.
        
        Args:
            chunk_text: The text chunk to analyze
            
        Returns:
            List of links found in this chunk
        """
        links = []
        for match in self.md_link_pattern.finditer(chunk_text):
            text = match.group(1)
            url = match.group(2)
            links.append({
                'text': text,
                'url': url
            })
        return links


# Register the text loader
def _register_text_loader():
    """Register text loader with the registry."""
    try:
        from .registry import register_loader
        register_loader(TextLoader, ['txt', 'md', 'markdown', 'rst', 'text'])
        logger.debug("Text loader registered successfully")
    except ImportError:
        logger.debug("Registry not available during import")


# Auto-register when module is imported
_register_text_loader()