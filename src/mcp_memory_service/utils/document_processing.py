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
Utilities for processing document chunks into memories.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from ..models.memory import Memory
from .hashing import generate_content_hash

logger = logging.getLogger(__name__)


def create_memory_from_chunk(
    chunk: Any,
    base_tags: List[str],
    memory_type: str = "document",
    context_tags: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Memory:
    """
    Create a Memory object from a document chunk with tag and metadata processing.

    Args:
        chunk: Document chunk object with content, metadata, and chunk_index
        base_tags: Base tags to apply to the memory
        memory_type: Type of memory (default: "document")
        context_tags: Additional context-specific tags as key-value pairs
                     (e.g., {"source_dir": "docs", "file_type": "pdf"})
        extra_metadata: Additional metadata to merge into chunk metadata

    Returns:
        Memory object ready for storage

    Example:
        >>> memory = create_memory_from_chunk(
        ...     chunk,
        ...     base_tags=["documentation"],
        ...     context_tags={"source_dir": "docs", "file_type": "pdf"},
        ...     extra_metadata={"upload_id": "batch123"}
        ... )
    """
    # Build tag list
    all_tags = list(base_tags)

    # Add context-specific tags
    if context_tags:
        for key, value in context_tags.items():
            all_tags.append(f"{key}:{value}")

    # Handle chunk metadata tags (can be string or list)
    if chunk.metadata and chunk.metadata.get('tags'):
        chunk_tags = chunk.metadata['tags']
        if isinstance(chunk_tags, str):
            # Split comma-separated string into list
            chunk_tags = [tag.strip() for tag in chunk_tags.split(',') if tag.strip()]
        all_tags.extend(chunk_tags)

    # Prepare metadata
    chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
    if extra_metadata:
        chunk_metadata.update(extra_metadata)

    # Create and return memory object
    return Memory(
        content=chunk.content,
        content_hash=generate_content_hash(chunk.content, chunk_metadata),
        tags=list(set(all_tags)),  # Remove duplicates
        memory_type=memory_type,
        metadata=chunk_metadata
    )


async def _process_and_store_chunk(
    chunk: Any,
    storage: Any,
    file_name: str,
    base_tags: List[str],
    context_tags: Dict[str, str],
    memory_type: str = "document",
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Process a document chunk and store it as a memory.

    This consolidates the common pattern of creating a memory from a chunk
    and storing it to the database across multiple ingestion entry points.

    Args:
        chunk: Document chunk with content and metadata
        storage: Storage backend instance
        file_name: Name of the source file (for error messages)
        base_tags: Base tags to apply to the memory
        context_tags: Context-specific tags (e.g., source_dir, file_type)
        memory_type: Type of memory (default: "document")
        extra_metadata: Additional metadata to merge into chunk metadata

    Returns:
        Tuple of (success: bool, error: Optional[str])
            - (True, None) if stored successfully
            - (False, error_message) if storage failed
    """
    try:
        # Create memory from chunk with context
        memory = create_memory_from_chunk(
            chunk,
            base_tags=base_tags,
            memory_type=memory_type,
            context_tags=context_tags,
            extra_metadata=extra_metadata
        )

        # Store the memory
        success, error = await storage.store(memory)
        if not success:
            return False, f"{file_name} chunk {chunk.chunk_index}: {error}"
        return True, None

    except Exception as e:
        return False, f"{file_name} chunk {chunk.chunk_index}: {str(e)}"
