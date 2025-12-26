"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation and ensuring consistent behavior.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, TypedDict
try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10
from datetime import datetime

from ..config import (
    INCLUDE_HOSTNAME,
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT
)
from ..storage.base import MemoryStorage
from ..models.memory import Memory
from ..utils.content_splitter import split_content
from ..utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)


def normalize_tags(tags: Union[str, List[str], None]) -> List[str]:
    """
    Normalize tags to a consistent list format.

    Handles all input formats:
    - None → []
    - "tag1,tag2,tag3" → ["tag1", "tag2", "tag3"]
    - "single-tag" → ["single-tag"]
    - ["tag1", "tag2"] → ["tag1", "tag2"]

    Args:
        tags: Tags in any supported format (None, string, comma-separated string, or list)

    Returns:
        List of tag strings, empty list if None or empty string
    """
    if tags is None:
        return []

    if isinstance(tags, str):
        # Empty string returns empty list
        if not tags.strip():
            return []
        # Split by comma if present, otherwise single tag
        if ',' in tags:
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        return [tags.strip()]

    # Already a list - return as-is
    return tags


class MemoryResult(TypedDict):
    """Type definition for memory operation results."""
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    created_at_iso: str
    updated_at_iso: str


# Store Memory Return Types
class StoreMemorySingleSuccess(TypedDict):
    """Return type for successful single memory storage."""
    success: bool
    memory: MemoryResult


class StoreMemoryChunkedSuccess(TypedDict):
    """Return type for successful chunked memory storage."""
    success: bool
    memories: List[MemoryResult]
    total_chunks: int
    original_hash: str
    failed_chunks: NotRequired[int]  # Number of chunks that failed to store


class StoreMemoryFailure(TypedDict):
    """Return type for failed memory storage."""
    success: bool
    error: str


# List Memories Return Types
class ListMemoriesSuccess(TypedDict):
    """Return type for successful memory listing."""
    memories: List[MemoryResult]
    page: int
    page_size: int
    total: int
    has_more: bool


class ListMemoriesError(TypedDict):
    """Return type for failed memory listing."""
    success: bool
    error: str
    memories: List[MemoryResult]
    page: int
    page_size: int


# Retrieve Memories Return Types
class RetrieveMemoriesSuccess(TypedDict):
    """Return type for successful memory retrieval."""
    memories: List[MemoryResult]
    query: str
    count: int


class RetrieveMemoriesError(TypedDict):
    """Return type for failed memory retrieval."""
    memories: List[MemoryResult]
    query: str
    error: str


# Search by Tag Return Types
class SearchByTagSuccess(TypedDict):
    """Return type for successful tag search."""
    memories: List[MemoryResult]
    tags: List[str]
    match_type: str
    count: int


class SearchByTagError(TypedDict):
    """Return type for failed tag search."""
    memories: List[MemoryResult]
    tags: List[str]
    error: str


# Delete Memory Return Types
class DeleteMemorySuccess(TypedDict):
    """Return type for successful memory deletion."""
    success: bool
    content_hash: str


class DeleteMemoryFailure(TypedDict):
    """Return type for failed memory deletion."""
    success: bool
    content_hash: str
    error: str


# Health Check Return Types
class HealthCheckSuccess(TypedDict, total=False):
    """Return type for successful health check."""
    healthy: bool
    storage_type: str
    total_memories: int
    last_updated: str
    # Additional fields from storage stats (marked as not required via total=False)


class HealthCheckFailure(TypedDict):
    """Return type for failed health check."""
    healthy: bool
    error: str


class MemoryService:
    """
    Shared service for memory operations with consistent business logic.

    This service centralizes all memory-related business logic to ensure
    consistent behavior across API endpoints and MCP tools, eliminating
    code duplication and potential inconsistencies.
    """

    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def list_memories(
        self,
        page: int = 1,
        page_size: int = 10,
        tag: Optional[str] = None,
        memory_type: Optional[str] = None
    ) -> Union[ListMemoriesSuccess, ListMemoriesError]:
        """
        List memories with pagination and optional filtering.

        This method provides database-level filtering for optimal performance,
        avoiding the common anti-pattern of loading all records into memory.

        Args:
            page: Page number (1-based)
            page_size: Number of memories per page
            tag: Filter by specific tag
            memory_type: Filter by memory type

        Returns:
            Dictionary with memories and pagination info
        """
        try:
            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Use database-level filtering for optimal performance
            tags_list = [tag] if tag else None
            memories = await self.storage.get_all_memories(
                limit=page_size,
                offset=offset,
                memory_type=memory_type,
                tags=tags_list
            )

            # Get accurate total count for pagination
            total = await self.storage.count_all_memories(
                memory_type=memory_type,
                tags=tags_list
            )

            # Format results for API response
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            return {
                "memories": results,
                "page": page,
                "page_size": page_size,
                "total": total,
                "has_more": offset + page_size < total
            }

        except Exception as e:
            logger.exception(f"Unexpected error listing memories: {e}")
            return {
                "success": False,
                "error": f"Failed to list memories: {str(e)}",
                "memories": [],
                "page": page,
                "page_size": page_size
            }

    async def store_memory(
        self,
        content: str,
        tags: Union[str, List[str], None] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client_hostname: Optional[str] = None
    ) -> Union[StoreMemorySingleSuccess, StoreMemoryChunkedSuccess, StoreMemoryFailure]:
        """
        Store a new memory with validation and content processing.

        Accepts tags in multiple formats for maximum flexibility:
        - None → []
        - "tag1,tag2,tag3" → ["tag1", "tag2", "tag3"]
        - "single-tag" → ["single-tag"]
        - ["tag1", "tag2"] → ["tag1", "tag2"]

        Args:
            content: The memory content
            tags: Optional tags for the memory (string, comma-separated string, or list)
            memory_type: Optional memory type classification
            metadata: Optional additional metadata (can also contain tags)
            client_hostname: Optional client hostname for source tagging

        Returns:
            Dictionary with operation result
        """
        try:
            # Normalize tags from parameter (handles all formats)
            final_tags = normalize_tags(tags)

            # Extract and normalize metadata.tags if present
            final_metadata = metadata or {}
            if metadata and "tags" in metadata:
                metadata_tags = normalize_tags(metadata.get("tags"))
                # Merge with parameter tags and remove duplicates
                final_tags = list(set(final_tags + metadata_tags))

            # Apply hostname tagging if provided (for consistent source tracking)
            if client_hostname:
                source_tag = f"source:{client_hostname}"
                if source_tag not in final_tags:
                    final_tags.append(source_tag)
                final_metadata["hostname"] = client_hostname

            # Generate content hash for deduplication
            content_hash = generate_content_hash(content)

            # Process content if auto-splitting is enabled and content exceeds max length
            max_length = self.storage.max_content_length
            if ENABLE_AUTO_SPLIT and max_length and len(content) > max_length:
                # Split content into chunks
                chunks = split_content(
                    content,
                    max_length=max_length,
                    preserve_boundaries=CONTENT_PRESERVE_BOUNDARIES,
                    overlap=CONTENT_SPLIT_OVERLAP
                )
                stored_memories = []
                failed_chunks = []

                for i, chunk in enumerate(chunks):
                    chunk_hash = generate_content_hash(chunk)
                    chunk_metadata = final_metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["original_hash"] = content_hash

                    memory = Memory(
                        content=chunk,
                        content_hash=chunk_hash,
                        tags=final_tags,
                        memory_type=memory_type,
                        metadata=chunk_metadata
                    )

                    success, message = await self.storage.store(memory)
                    if success:
                        stored_memories.append(self._format_memory_response(memory))
                    else:
                        failed_chunks.append({"index": i, "reason": message})

                # If NO chunks were stored, return failure
                if not stored_memories:
                    reasons = ", ".join(set(fc["reason"] for fc in failed_chunks))
                    return {
                        "success": False,
                        "error": f"Failed to store all {len(chunks)} chunks: {reasons}"
                    }

                # If SOME chunks were stored, return partial success
                return {
                    "success": True,
                    "memories": stored_memories,
                    "total_chunks": len(chunks),
                    "original_hash": content_hash,
                    "failed_chunks": len(failed_chunks)
                }
            else:
                # Store as single memory
                memory = Memory(
                    content=content,
                    content_hash=content_hash,
                    tags=final_tags,
                    memory_type=memory_type,
                    metadata=final_metadata
                )

                success, message = await self.storage.store(memory)

                if success:
                    return {
                        "success": True,
                        "memory": self._format_memory_response(memory)
                    }
                else:
                    return {
                        "success": False,
                        "error": message
                    }

        except ValueError as e:
            # Handle validation errors specifically
            logger.warning(f"Validation error storing memory: {e}")
            return {
                "success": False,
                "error": f"Invalid memory data: {str(e)}"
            }
        except ConnectionError as e:
            # Handle storage connectivity issues
            logger.error(f"Storage connection error: {e}")
            return {
                "success": False,
                "error": f"Storage connection failed: {str(e)}"
            }
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error storing memory: {e}")
            return {
                "success": False,
                "error": f"Failed to store memory: {str(e)}"
            }

    async def retrieve_memories(
        self,
        query: str,
        n_results: int = 10,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None
    ) -> Union[RetrieveMemoriesSuccess, RetrieveMemoriesError]:
        """
        Retrieve memories by semantic search with optional filtering.

        Args:
            query: Search query string
            n_results: Maximum number of results
            tags: Optional tag filtering
            memory_type: Optional memory type filtering

        Returns:
            Dictionary with search results
        """
        try:
            # Retrieve memories using semantic search
            # Note: storage.retrieve() only supports query and n_results
            # We'll filter by tags/type after retrieval if needed
            memories = await self.storage.retrieve(
                query=query,
                n_results=n_results
            )

            # Apply optional post-filtering
            filtered_memories = memories
            if tags or memory_type:
                filtered_memories = []
                for query_result in memories:
                    # Filter by tags if specified
                    if tags:
                        memory_tags = query_result.memory.metadata.get('tags', [])
                        if not any(tag in memory_tags for tag in tags):
                            continue

                    # Filter by memory_type if specified
                    if memory_type:
                        mem_type = query_result.memory.metadata.get('memory_type', '')
                        if mem_type != memory_type:
                            continue

                    filtered_memories.append(query_result)

            results = []
            for result in filtered_memories:
                # Extract Memory object from MemoryQueryResult and add similarity score
                memory_dict = self._format_memory_response(result.memory)
                memory_dict['similarity_score'] = result.relevance_score
                results.append(memory_dict)

            return {
                "memories": results,
                "query": query,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {
                "memories": [],
                "query": query,
                "error": f"Failed to retrieve memories: {str(e)}"
            }

    async def search_by_tag(
        self,
        tags: Union[str, List[str]],
        match_all: bool = False
    ) -> Union[SearchByTagSuccess, SearchByTagError]:
        """
        Search memories by tags with flexible matching options.

        Args:
            tags: Tag or list of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag

        Returns:
            Dictionary with matching memories
        """
        try:
            # Normalize tags to list (handles all formats including comma-separated)
            tags = normalize_tags(tags)

            # Search using database-level filtering
            # Note: Using search_by_tag from base class (singular)
            memories = await self.storage.search_by_tag(tags=tags)

            # Format results
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            # Determine match type description
            match_type = "ALL" if match_all else "ANY"

            return {
                "memories": results,
                "tags": tags,
                "match_type": match_type,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return {
                "memories": [],
                "tags": tags if isinstance(tags, list) else [tags],
                "error": f"Failed to search by tags: {str(e)}"
            }

    async def get_memory_by_hash(self, content_hash: str) -> Dict[str, Any]:
        """
        Retrieve a specific memory by its content hash using O(1) direct lookup.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Dictionary with memory data or error
        """
        try:
            # Use direct O(1) lookup via storage.get_by_hash()
            memory = await self.storage.get_by_hash(content_hash)

            if memory:
                return {
                    "memory": self._format_memory_response(memory),
                    "found": True
                }
            else:
                return {
                    "found": False,
                    "content_hash": content_hash
                }

        except Exception as e:
            logger.error(f"Error getting memory by hash: {e}")
            return {
                "found": False,
                "content_hash": content_hash,
                "error": f"Failed to get memory: {str(e)}"
            }

    async def delete_memory(self, content_hash: str) -> Union[DeleteMemorySuccess, DeleteMemoryFailure]:
        """
        Delete a memory by its content hash.

        Args:
            content_hash: The content hash of the memory to delete

        Returns:
            Dictionary with operation result
        """
        try:
            success, message = await self.storage.delete(content_hash)
            if success:
                return {
                    "success": True,
                    "content_hash": content_hash
                }
            else:
                return {
                    "success": False,
                    "content_hash": content_hash,
                    "error": message
                }

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {
                "success": False,
                "content_hash": content_hash,
                "error": f"Failed to delete memory: {str(e)}"
            }

    async def health_check(self) -> Union[HealthCheckSuccess, HealthCheckFailure]:
        """
        Perform a health check on the memory storage system.

        Returns:
            Dictionary with health status and statistics
        """
        try:
            stats = await self.storage.get_stats()
            return {
                "healthy": True,
                "storage_type": stats.get("backend", "unknown"),
                "total_memories": stats.get("total_memories", 0),
                "last_updated": datetime.now().isoformat(),
                **stats
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": f"Health check failed: {str(e)}"
            }

    def _format_memory_response(self, memory: Memory) -> MemoryResult:
        """
        Format a memory object for API response.

        Args:
            memory: The memory object to format

        Returns:
            Formatted memory dictionary
        """
        return {
            "content": memory.content,
            "content_hash": memory.content_hash,
            "tags": memory.tags,
            "memory_type": memory.memory_type,
            "metadata": memory.metadata,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "created_at_iso": memory.created_at_iso,
            "updated_at_iso": memory.updated_at_iso
        }
