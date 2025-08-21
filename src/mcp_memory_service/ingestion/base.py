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
Base classes and interfaces for document ingestion.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text extracted from a document.
    
    Attributes:
        content: The extracted text content
        metadata: Additional metadata about the chunk
        chunk_index: Position of this chunk in the document
        source_file: Original file path
    """
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    source_file: Path
    
    def __post_init__(self):
        """Add default metadata after initialization."""
        if 'source' not in self.metadata:
            self.metadata['source'] = str(self.source_file)
        if 'chunk_index' not in self.metadata:
            self.metadata['chunk_index'] = self.chunk_index
        if 'extracted_at' not in self.metadata:
            self.metadata['extracted_at'] = datetime.now().isoformat()


@dataclass
class IngestionResult:
    """
    Result of document ingestion operation.
    
    Attributes:
        success: Whether ingestion was successful
        chunks_processed: Number of chunks created
        chunks_stored: Number of chunks successfully stored
        errors: List of error messages encountered
        source_file: Original file that was processed
        processing_time: Time taken to process in seconds
    """
    success: bool
    chunks_processed: int
    chunks_stored: int
    errors: List[str]
    source_file: Path
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.chunks_processed == 0:
            return 0.0
        return (self.chunks_stored / self.chunks_processed) * 100


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    
    Each document format (PDF, text, etc.) should implement this interface
    to provide consistent document processing capabilities.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions: List[str] = []
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this loader can process the file
        """
        pass
    
    @abstractmethod
    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a document.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional options specific to the loader
            
        Yields:
            DocumentChunk objects containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            Exception: Other processing errors
        """
        pass
    
    async def validate_file(self, file_path: Path) -> None:
        """
        Validate that a file can be processed.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not supported or invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not self.can_handle(file_path):
            raise ValueError(f"File format not supported: {file_path.suffix}")
    
    def get_base_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Get base metadata common to all document types.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of base metadata
        """
        stat = file_path.stat()
        return {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'loader_type': self.__class__.__name__
        }