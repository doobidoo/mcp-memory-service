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
Intelligent text chunking strategies for document ingestion.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Configuration for text chunking behavior."""
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Characters to overlap between chunks
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    min_chunk_size: int = 100  # Minimum characters for a valid chunk


class TextChunker:
    """
    Intelligent text chunking that respects document structure.
    
    Provides multiple chunking strategies:
    - Sentence-aware chunking
    - Paragraph-aware chunking  
    - Token-based chunking
    - Custom delimiter chunking
    """
    
    def __init__(self, strategy: ChunkingStrategy = None):
        """
        Initialize text chunker.
        
        Args:
            strategy: Chunking configuration to use
        """
        self.strategy = strategy or ChunkingStrategy()
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_separator = re.compile(r'\n\s*\n')
        
        # Common section headers (for structured documents)
        self.section_headers = re.compile(
            r'^(#{1,6}\s+|Chapter\s+\d+|Section\s+\d+|Part\s+\d+|\d+\.\s+)',
            re.MULTILINE | re.IGNORECASE
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks using the configured strategy.
        
        Args:
            text: Text content to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        if not text or len(text.strip()) < self.strategy.min_chunk_size:
            return []
        
        metadata = metadata or {}
        
        # Try different chunking strategies in order of preference
        if self.strategy.respect_paragraph_boundaries:
            chunks = self._chunk_by_paragraphs(text)
        elif self.strategy.respect_sentence_boundaries:
            chunks = self._chunk_by_sentences(text)
        else:
            chunks = self._chunk_by_characters(text)
        
        # Add metadata to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_length': len(chunk_text),
                'total_chunks': len(chunks),
                'chunking_strategy': self._get_strategy_name()
            })
            result.append((chunk_text, chunk_metadata))
        
        logger.debug(f"Created {len(result)} chunks from {len(text)} characters")
        return result
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraph boundaries, respecting size limits.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        paragraphs = self.paragraph_separator.split(text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if (len(current_chunk) + len(paragraph) + 2 > self.strategy.chunk_size 
                and len(current_chunk) > 0):
                
                # Finalize current chunk
                if len(current_chunk.strip()) >= self.strategy.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk)
                current_chunk = overlap + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining text
        if len(current_chunk.strip()) >= self.strategy.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentence boundaries, respecting size limits.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        sentences = self.sentence_endings.split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if (len(current_chunk) + len(sentence) + 1 > self.strategy.chunk_size 
                and len(current_chunk) > 0):
                
                # Finalize current chunk
                if len(current_chunk.strip()) >= self.strategy.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk)
                current_chunk = overlap + sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add remaining text
        if len(current_chunk.strip()) >= self.strategy.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Chunk text by character count with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.strategy.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.strategy.chunk_size
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for space to avoid breaking words
                for i in range(end, max(start + self.strategy.min_chunk_size, end - 100), -1):
                    if text[i].isspace():
                        end = i
                        break
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.strategy.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.strategy.chunk_overlap)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            text: Text to extract overlap from
            
        Returns:
            Overlap text to include in next chunk
        """
        if len(text) <= self.strategy.chunk_overlap:
            return text + " "
        
        overlap = text[-self.strategy.chunk_overlap:]
        
        # Try to start overlap at a sentence boundary
        sentences = self.sentence_endings.split(overlap)
        if len(sentences) > 1:
            overlap = " ".join(sentences[1:])
        
        return overlap + " " if overlap else ""
    
    def _get_strategy_name(self) -> str:
        """Get human-readable name for current chunking strategy."""
        if self.strategy.respect_paragraph_boundaries:
            return "paragraph_aware"
        elif self.strategy.respect_sentence_boundaries:
            return "sentence_aware"
        else:
            return "character_based"
    
    def chunk_by_sections(self, text: str, metadata: Dict[str, Any] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunk text by document sections (headers, chapters, etc.).
        
        Args:
            text: Text content to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        metadata = metadata or {}
        
        # Find section boundaries
        section_matches = list(self.section_headers.finditer(text))
        if not section_matches:
            # No sections found, use regular chunking
            return self.chunk_text(text, metadata)
        
        chunks = []
        section_start = 0
        
        for i, match in enumerate(section_matches):
            section_end = match.start()
            
            # Extract previous section if it exists
            if section_start < section_end:
                section_text = text[section_start:section_end].strip()
                if len(section_text) >= self.strategy.min_chunk_size:
                    section_metadata = metadata.copy()
                    section_metadata.update({
                        'section_index': i,
                        'is_section': True,
                        'section_start': section_start,
                        'section_end': section_end
                    })
                    
                    # If section is too large, sub-chunk it
                    if len(section_text) > self.strategy.chunk_size * 2:
                        sub_chunks = self.chunk_text(section_text, section_metadata)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append((section_text, section_metadata))
            
            section_start = match.start()
        
        # Handle final section
        if section_start < len(text):
            final_text = text[section_start:].strip()
            if len(final_text) >= self.strategy.min_chunk_size:
                final_metadata = metadata.copy()
                final_metadata.update({
                    'section_index': len(section_matches),
                    'is_section': True,
                    'section_start': section_start,
                    'section_end': len(text)
                })
                
                if len(final_text) > self.strategy.chunk_size * 2:
                    sub_chunks = self.chunk_text(final_text, final_metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append((final_text, final_metadata))
        
        return chunks