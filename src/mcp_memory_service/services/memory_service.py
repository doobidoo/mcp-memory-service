"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation.
"""

import logging
import socket
from typing import Dict, List, Optional, Any, Union, Tuple, TypedDict
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


class MemoryResult(TypedDict):
    """Type definition for memory operation results."""
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    created_at: str
    similarity_score: Optional[float]


class OperationResult(TypedDict):
    """Type definition for operation results."""
    success: bool
    message: str
    content_hash: Optional[str]
    chunks_created: Optional[int]
    chunk_hashes: Optional[List[str]]


class HealthStats(TypedDict):
    """Type definition for health statistics."""
    status: str
    backend: str
    total_memories: int
    total_tags: int
    storage_size: str
    last_backup: str
    timestamp: str


class MemoryService:
    """
    Shared business logic for memory operations.

    This service encapsulates all the business logic that was previously
    duplicated between the MCP server and HTTP server implementations.
    """

    def __init__(self, storage: MemoryStorage):
        """
        Initialize the MemoryService with a storage backend.

        Args:
            storage: The storage backend to use for persistence
        """
        self.storage = storage

    async def store_memory(
        self,
        content: str,
        tags: Union[str, List[str], None] = None,
        memory_type: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
        client_hostname: Optional[str] = None
    ) -> OperationResult:
        """
        Store a new memory with content and optional metadata.

        This method contains the shared business logic for storing memories
        that was previously duplicated between servers.

        Args:
            content: The content to store as memory
            tags: Optional tags (accepts array or comma-separated string)
            memory_type: Type of memory (note, decision, task, reference)
            metadata: Additional metadata for the memory
            client_hostname: Client machine hostname for source tracking

        Returns:
            Dictionary with operation results
        """
        try:
            # Normalize tags to list (accept both string and array formats)
            if isinstance(tags, str):
                # Split comma-separated string into array
                final_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif isinstance(tags, list):
                final_tags = tags
            else:
                final_tags = []

            final_metadata = metadata or {}

            # Add hostname tracking if enabled
            if INCLUDE_HOSTNAME:
                # Prioritize client-provided hostname, then fallback to server
                if client_hostname:
                    hostname = client_hostname
                else:
                    hostname = socket.gethostname()

                source_tag = f"source:{hostname}"
                if source_tag not in final_tags:
                    final_tags.append(source_tag)
                final_metadata["hostname"] = hostname

            # Check if content needs splitting
            max_length = self.storage.max_content_length
            if max_length and len(content) > max_length:
                if not ENABLE_AUTO_SPLIT:
                    logger.warning(f"Content length {len(content)} exceeds limit {max_length}, and auto-split is disabled.")
                    return {
                        "success": False,
                        "message": f"Content length {len(content)} exceeds backend limit of {max_length}. Auto-splitting is disabled.",
                    }

                # Content exceeds limit - split into chunks
                logger.info(f"Content length {len(content)} exceeds backend limit {max_length}, splitting...")

                chunks = split_content(
                    content,
                    max_length,
                    preserve_boundaries=CONTENT_PRESERVE_BOUNDARIES,
                    overlap=CONTENT_SPLIT_OVERLAP
                )
                total_chunks = len(chunks)
                chunk_memories = []

                # Create all chunk memories
                for i, chunk in enumerate(chunks):
                    # Add chunk metadata
                    chunk_metadata = final_metadata.copy()
                    chunk_metadata.update({
                        "is_chunk": True,
                        "chunk_index": i + 1,
                        "total_chunks": total_chunks,
                        "original_length": len(content)
                    })

                    # Add chunk indicator to tags
                    chunk_tags = final_tags.copy()
                    chunk_tags.append(f"chunk:{i+1}/{total_chunks}")

                    # Create chunk memory object
                    chunk_memory = Memory(
                        content=chunk,
                        content_hash=generate_content_hash(chunk, chunk_metadata),
                        tags=chunk_tags,
                        memory_type=memory_type,
                        metadata=chunk_metadata
                    )
                    chunk_memories.append(chunk_memory)

                # Store all chunks in a single batch operation
                results = await self.storage.store_batch(chunk_memories)

                successful_chunks = [mem for mem, (success, _) in zip(chunk_memories, results) if success]
                failed_count = len(chunk_memories) - len(successful_chunks)

                if failed_count == 0:
                    chunk_hashes = [mem.content_hash for mem in successful_chunks]
                    return {
                        "success": True,
                        "message": f"Content split into {total_chunks} chunks and stored successfully",
                        "chunks_created": total_chunks,
                        "chunk_hashes": chunk_hashes
                    }
                else:
                    error_messages = [msg for success, msg in results if not success]
                    logger.error(f"Failed to store {failed_count} chunks: {error_messages}")
                    return {
                        "success": False,
                        "message": f"Failed to store {failed_count}/{total_chunks} chunks. Errors: {error_messages}",
                        "chunks_created": len(successful_chunks),
                        "chunk_hashes": [mem.content_hash for mem in successful_chunks]
                    }

            else:
                # Content within limit - store as single memory
                memory = Memory(
                    content=content,
                    content_hash=generate_content_hash(content, final_metadata),
                    tags=final_tags,
                    memory_type=memory_type,
                    metadata=final_metadata
                )

                # Store memory
                success, message = await self.storage.store(memory)

                return {
                    "success": success,
                    "message": message,
                    "content_hash": memory.content_hash
                }

        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to store memory: {str(e)}"
            }

    async def retrieve_memory(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.0
    ) -> Dict[str, Union[List[MemoryResult], str]]:
        """
        Retrieve memories based on semantic similarity to a query.

        Args:
            query: Search query for semantic similarity
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity score threshold

        Returns:
            Dictionary with retrieved memories and metadata
        """
        try:
            # Search for memories
            results = await self.storage.search(
                query=query,
                n_results=n_results,
                min_similarity=min_similarity
            )

            # Format results
            memories = []
            for result in results:
                memories.append({
                    "content": result.memory.content,
                    "content_hash": result.memory.content_hash,
                    "tags": result.memory.metadata.tags,
                    "memory_type": result.memory.metadata.memory_type,
                    "created_at": result.memory.metadata.created_at_iso,
                    "similarity_score": result.similarity_score
                })

            return {
                "memories": memories,
                "query": query,
                "total_results": len(memories)
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
    ) -> Dict[str, Union[List[MemoryResult], str, bool, int]]:
        """
        Search memories by tags.

        Args:
            tags: Tag or list of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag

        Returns:
            Dictionary with matching memories
        """
        try:
            # Normalize tags to list
            if isinstance(tags, str):
                tags = [tags]

            # Search by tags
            memories = await self.storage.search_by_tags(
                tags=tags,
                match_all=match_all
            )

            # Format results
            results = []
            for memory in memories:
                results.append({
                    "content": memory.content,
                    "content_hash": memory.content_hash,
                    "tags": memory.metadata.tags,
                    "memory_type": memory.metadata.memory_type,
                    "created_at": memory.metadata.created_at_iso
                })

            return {
                "memories": results,
                "search_tags": tags,
                "match_all": match_all,
                "total_results": len(results)
            }

        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return {
                "memories": [],
                "search_tags": tags,
                "error": f"Failed to search by tags: {str(e)}"
            }

    async def delete_memory(self, content_hash: str) -> Dict[str, Union[bool, str]]:
        """
        Delete a specific memory by its content hash.

        Args:
            content_hash: Hash of the memory content to delete

        Returns:
            Dictionary with success status and message
        """
        try:
            # Delete memory
            success, message = await self.storage.delete(content_hash)

            return {
                "success": success,
                "message": message,
                "content_hash": content_hash
            }

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {
                "success": False,
                "message": f"Failed to delete memory: {str(e)}",
                "content_hash": content_hash
            }

    async def check_database_health(self) -> Dict[str, Union[str, HealthStats]]:
        """
        Check the health and status of the memory database.

        Returns:
            Dictionary with health status and statistics
        """
        try:
            # Get health status and statistics
            stats = await self.storage.get_stats()

            return {
                "status": "healthy",
                "backend": self.storage.__class__.__name__,
                "statistics": {
                    "total_memories": stats.get("total_memories", 0),
                    "total_tags": stats.get("total_tags", 0),
                    "storage_size": stats.get("storage_size", "unknown"),
                    "last_backup": stats.get("last_backup", "never")
                },
                "timestamp": stats.get("timestamp", "unknown")
            }

        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            return {
                "status": "error",
                "backend": "unknown",
                "error": f"Health check failed: {str(e)}"
            }

    async def list_memories(
        self,
        page: int = 1,
        page_size: int = 10,
        tag: Optional[str] = None,
        memory_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List memories with pagination and optional filtering.

        Args:
            page: Page number (1-based)
            page_size: Number of memories per page
            tag: Filter by specific tag
            memory_type: Filter by memory type

        Returns:
            Dictionary with memories and pagination info
        """
        try:
            # Calculate offset
            offset = (page - 1) * page_size

            # Use database-level filtering for better performance
            tags_list = [tag] if tag else None
            memories = await self.storage.get_all_memories(
                limit=page_size,
                offset=offset,
                memory_type=memory_type,
                tags=tags_list
            )

            # Format results
            results = []
            for memory in memories:
                results.append({
                    "content": memory.content,
                    "content_hash": memory.content_hash,
                    "tags": memory.tags,
                    "memory_type": memory.memory_type,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at_iso,
                    "updated_at": memory.updated_at_iso
                })

            return {
                "memories": results,
                "page": page,
                "page_size": page_size,
                "total_found": len(results)
            }

        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return {
                "memories": [],
                "page": page,
                "page_size": page_size,
                "error": f"Failed to list memories: {str(e)}"
            }
