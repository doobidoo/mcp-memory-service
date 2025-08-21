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
PDF document loader for extracting text content from PDF files.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional
import asyncio

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)

# Try to import PDF processing library
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    logger.warning("PyPDF2 not available. PDF support will be limited.")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.debug("pdfplumber not available, falling back to PyPDF2")


class PDFLoader(DocumentLoader):
    """
    Document loader for PDF files.
    
    Supports multiple PDF processing backends:
    - pdfplumber (preferred): Better text extraction, table support
    - PyPDF2 (fallback): Basic text extraction
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF loader.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['pdf']
        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=True
        ))
        
        # Check which PDF backend is available
        if HAS_PDFPLUMBER:
            self.backend = 'pdfplumber'
        elif HAS_PYPDF2:
            self.backend = 'pypdf2'
        else:
            self.backend = None
            logger.error("No PDF processing library available. Install pypdf2 or pdfplumber.")
    
    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given PDF file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this loader can process the PDF file
        """
        if self.backend is None:
            return False
        
        return (file_path.suffix.lower() == '.pdf' and 
                file_path.exists() and 
                file_path.is_file())
    
    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            **kwargs: Additional options:
                - extract_images: Whether to extract image descriptions (default: False)
                - extract_tables: Whether to extract table content (default: False)
                - page_range: Tuple of (start, end) pages to extract (1-indexed)
            
        Yields:
            DocumentChunk objects containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF file is invalid or can't be processed
        """
        await self.validate_file(file_path)
        
        if self.backend is None:
            raise ValueError("No PDF processing backend available")
        
        extract_images = kwargs.get('extract_images', False)
        extract_tables = kwargs.get('extract_tables', False)
        page_range = kwargs.get('page_range', None)
        
        logger.info(f"Extracting chunks from PDF: {file_path} using {self.backend}")
        
        try:
            if self.backend == 'pdfplumber':
                async for chunk in self._extract_with_pdfplumber(
                    file_path, extract_images, extract_tables, page_range
                ):
                    yield chunk
            else:
                async for chunk in self._extract_with_pypdf2(
                    file_path, page_range
                ):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error extracting from PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract PDF content: {str(e)}") from e
    
    async def _extract_with_pdfplumber(
        self, 
        file_path: Path, 
        extract_images: bool,
        extract_tables: bool,
        page_range: Optional[tuple]
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text using pdfplumber backend.
        
        Args:
            file_path: Path to PDF file
            extract_images: Whether to extract image descriptions
            extract_tables: Whether to extract table content
            page_range: Optional page range to extract
            
        Yields:
            DocumentChunk objects
        """
        def _extract_sync():
            """Synchronous extraction function to run in thread pool."""
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                start_page = page_range[0] - 1 if page_range else 0
                end_page = min(page_range[1], total_pages) if page_range else total_pages
                
                for page_num in range(start_page, end_page):
                    page = pdf.pages[page_num]
                    
                    # Extract main text
                    text = page.extract_text() or ""
                    
                    # Extract table content if requested
                    if extract_tables:
                        tables = page.extract_tables()
                        for table in tables or []:
                            table_text = self._format_table(table)
                            text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]"
                    
                    # Note: Image extraction would require additional processing
                    if extract_images:
                        # Placeholder for image extraction
                        images = page.images
                        if images:
                            text += f"\n\n[IMAGES: {len(images)} images found on this page]"
                    
                    if text.strip():
                        yield (page_num + 1, text.strip())
        
        # Run extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        page_generator = await loop.run_in_executor(None, lambda: list(_extract_sync()))
        
        base_metadata = self.get_base_metadata(file_path)
        chunk_index = 0
        
        for page_num, page_text in page_generator:
            page_metadata = base_metadata.copy()
            page_metadata.update({
                'page_number': page_num,
                'extraction_method': 'pdfplumber',
                'content_type': 'pdf_page'
            })
            
            # Chunk the page text
            chunks = self.chunker.chunk_text(page_text, page_metadata)
            
            for chunk_text, chunk_metadata in chunks:
                yield DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    source_file=file_path
                )
                chunk_index += 1
    
    async def _extract_with_pypdf2(
        self, 
        file_path: Path,
        page_range: Optional[tuple]
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text using PyPDF2 backend.
        
        Args:
            file_path: Path to PDF file
            page_range: Optional page range to extract
            
        Yields:
            DocumentChunk objects
        """
        def _extract_sync():
            """Synchronous extraction function."""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                start_page = page_range[0] - 1 if page_range else 0
                end_page = min(page_range[1], total_pages) if page_range else total_pages
                
                for page_num in range(start_page, end_page):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        yield (page_num + 1, text.strip())
        
        # Run extraction in thread pool
        loop = asyncio.get_event_loop()
        page_generator = await loop.run_in_executor(None, lambda: list(_extract_sync()))
        
        base_metadata = self.get_base_metadata(file_path)
        chunk_index = 0
        
        for page_num, page_text in page_generator:
            page_metadata = base_metadata.copy()
            page_metadata.update({
                'page_number': page_num,
                'extraction_method': 'pypdf2',
                'content_type': 'pdf_page'
            })
            
            # Chunk the page text
            chunks = self.chunker.chunk_text(page_text, page_metadata)
            
            for chunk_text, chunk_metadata in chunks:
                yield DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    source_file=file_path
                )
                chunk_index += 1
    
    def _format_table(self, table: list) -> str:
        """
        Format extracted table data as text.
        
        Args:
            table: Table data as list of rows
            
        Returns:
            Formatted table text
        """
        if not table:
            return ""
        
        # Simple table formatting
        formatted_rows = []
        for row in table:
            if row:  # Skip empty rows
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                formatted_rows.append(" | ".join(cleaned_row))
        
        return "\n".join(formatted_rows)


# Register the PDF loader
def _register_pdf_loader():
    """Register PDF loader with the registry."""
    try:
        from .registry import register_loader
        register_loader(PDFLoader, ['pdf'])
        logger.debug("PDF loader registered successfully")
    except ImportError:
        logger.debug("Registry not available during import")


# Auto-register when module is imported
_register_pdf_loader()