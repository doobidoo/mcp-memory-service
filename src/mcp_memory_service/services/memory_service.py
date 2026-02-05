"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation and ensuring consistent behavior.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, TypedDict

from ..config import (
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT,
    settings,
)
from ..models.memory import Memory
from ..storage.base import MemoryStorage
from ..utils.content_splitter import split_content
from ..utils.hashing import generate_content_hash
from ..utils.hybrid_search import (
    apply_recency_decay,
    combine_results_rrf,
    extract_query_keywords,
    get_adaptive_alpha,
)

logger = logging.getLogger(__name__)


class MemoryResult(TypedDict):
    """Type definition for memory operation results."""

    content: str
    content_hash: str
    tags: list[str]
    memory_type: str | None
    metadata: dict[str, Any] | None
    created_at: str
    updated_at: str
    created_at_iso: str
    updated_at_iso: str


class MemoryService:
    """
    Shared service for memory operations with consistent business logic.

    This service centralizes all memory-related business logic to ensure
    consistent behavior across API endpoints and MCP tools, eliminating
    code duplication and potential inconsistencies.
    """

    # Tag cache TTL in seconds
    _TAG_CACHE_TTL = 60

    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self._tag_cache: tuple[float, set[str]] | None = None

    async def _get_cached_tags(self) -> set[str]:
        """Get all tags with 60-second TTL caching for performance."""
        now = time.time()
        if self._tag_cache is not None:
            cache_time, cached_tags = self._tag_cache
            if now - cache_time < self._TAG_CACHE_TTL:
                return cached_tags

        # Cache miss - fetch from storage
        all_tags = await self.storage.get_all_tags()
        self._tag_cache = (now, set(all_tags))
        return self._tag_cache[1]

    async def _retrieve_vector_only(
        self,
        query: str,
        page: int,
        page_size: int,
        tags: list[str] | None,
        memory_type: str | None,
        min_similarity: float | None,
    ) -> dict[str, Any]:
        """Fallback to pure vector search (original behavior)."""
        offset = (page - 1) * page_size

        # Use a reasonable limit for count to avoid sqlite-vec k limit (4096)
        try:
            total = await self.storage.count_semantic_search(
                query=query, tags=tags, memory_type=memory_type, min_similarity=min_similarity
            )
        except Exception:
            # Fallback: estimate based on page_size if count fails
            total = page_size * 10  # Reasonable estimate

        memories = await self.storage.retrieve(
            query=query,
            n_results=page_size,
            tags=tags,
            memory_type=memory_type,
            min_similarity=min_similarity,
            offset=offset,
        )
        results = []
        for item in memories:
            if hasattr(item, "memory"):
                memory_dict = self._format_memory_response(item.memory)
                memory_dict["similarity_score"] = item.similarity_score
                results.append(memory_dict)
            else:
                results.append(self._format_memory_response(item))
        return {
            "memories": results,
            "query": query,
            "hybrid_enabled": False,
            **self._build_pagination_metadata(total, page, page_size),
        }

    def _build_pagination_metadata(self, total: int, page: int, page_size: int) -> dict[str, Any]:
        """
        Build consistent pagination metadata for all endpoints.

        DRY principle: Single source of truth for pagination structure.

        Args:
            total: Total number of matching records across all pages
            page: Current page number (1-indexed)
            page_size: Number of results per page

        Returns:
            Dictionary with pagination metadata
        """
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (page * page_size) < total,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 1,
        }

    async def list_memories(
        self, page: int = 1, page_size: int = 10, tag: str | None = None, memory_type: str | None = None
    ) -> dict[str, Any]:
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
                limit=page_size, offset=offset, memory_type=memory_type, tags=tags_list
            )

            # Get accurate total count for pagination
            total = await self.storage.count_all_memories(memory_type=memory_type, tags=tags_list)

            # Format results for API response
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            return {"memories": results, **self._build_pagination_metadata(total, page, page_size)}

        except Exception as e:
            logger.exception(f"Unexpected error listing memories: {e}")
            return {
                "success": False,
                "error": f"Failed to list memories: {str(e)}",
                "memories": [],
                "page": page,
                "page_size": page_size,
            }

    async def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        client_hostname: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a new memory with validation and content processing.

        Args:
            content: The memory content
            tags: Optional tags for the memory
            memory_type: Optional memory type classification
            metadata: Optional additional metadata
            client_hostname: Optional client hostname for source tagging

        Returns:
            Dictionary with operation result
        """
        try:
            # Prepare tags and metadata with optional hostname tagging
            final_tags = tags or []
            final_metadata = metadata or {}

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
                    overlap=CONTENT_SPLIT_OVERLAP,
                )
                stored_memories = []

                for i, chunk in enumerate(chunks):
                    chunk_hash = generate_content_hash(chunk)
                    chunk_metadata = final_metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["original_hash"] = content_hash

                    memory = Memory(
                        content=chunk, content_hash=chunk_hash, tags=final_tags, memory_type=memory_type, metadata=chunk_metadata
                    )

                    success, message = await self.storage.store(memory)
                    if success:
                        stored_memories.append(self._format_memory_response(memory))

                return {"success": True, "memories": stored_memories, "total_chunks": len(chunks), "original_hash": content_hash}
            else:
                # Store as single memory
                memory = Memory(
                    content=content, content_hash=content_hash, tags=final_tags, memory_type=memory_type, metadata=final_metadata
                )

                success, message = await self.storage.store(memory)

                if success:
                    return {"success": True, "memory": self._format_memory_response(memory)}
                else:
                    return {"success": False, "error": message}

        except ValueError as e:
            # Handle validation errors specifically
            logger.warning(f"Validation error storing memory: {e}")
            return {"success": False, "error": f"Invalid memory data: {str(e)}"}
        except ConnectionError as e:
            # Handle storage connectivity issues
            logger.error(f"Storage connection error: {e}")
            return {"success": False, "error": f"Storage connection failed: {str(e)}"}
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error storing memory: {e}")
            return {"success": False, "error": f"Failed to store memory: {str(e)}"}

    async def retrieve_memories(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve memories using hybrid search (semantic + tag matching).

        Combines vector similarity with automatic tag extraction for improved retrieval.
        When query terms match existing tags, those memories receive a score boost.
        This solves the "rathole problem" where project-specific queries return
        semantically similar but categorically unrelated results.

        Hybrid search is enabled by default. To opt-out to pure vector search:
        - Set environment variable MCP_MEMORY_HYBRID_ALPHA=1.0

        Args:
            query: Search query string (tags extracted automatically)
            page: Page number (1-indexed)
            page_size: Number of results per page
            tags: Optional explicit tag filtering (bypasses hybrid, uses vector only)
            memory_type: Optional memory type filtering
            min_similarity: Optional minimum similarity threshold (0.0 to 1.0)

        Returns:
            Dictionary with search results and pagination metadata
        """
        try:
            config = settings.hybrid_search

            # If tags explicitly provided (even empty list), skip hybrid and use pure vector search
            # Distinguishes "no tags" (None) from "explicit empty tags" ([])
            if tags is not None:
                return await self._retrieve_vector_only(query, page, page_size, tags, memory_type, min_similarity)

            # Get cached tags for keyword extraction
            existing_tags = await self._get_cached_tags()

            # Extract potential tag keywords from query
            keywords = extract_query_keywords(query, existing_tags)

            # If no keywords match existing tags, fall back to vector-only
            if not keywords:
                return await self._retrieve_vector_only(query, page, page_size, None, memory_type, min_similarity)

            # Determine alpha (explicit > env > adaptive)
            corpus_size = await self.storage.count()
            alpha = get_adaptive_alpha(corpus_size, len(keywords), config)

            # If alpha is 1.0, pure vector search (opt-out)
            if alpha >= 1.0:
                return await self._retrieve_vector_only(query, page, page_size, None, memory_type, min_similarity)

            # Fetch larger result set for RRF combination
            # Must cover offset + page_size to support pagination beyond page 1
            offset = (page - 1) * page_size
            fetch_size = min(max(page_size * 3, offset + page_size), 100)

            # Parallel fetch: vector results + tag-matching memories
            vector_task = self.storage.retrieve(
                query=query,
                n_results=fetch_size,
                tags=None,
                memory_type=memory_type,
                min_similarity=min_similarity,
                offset=0,
            )
            tag_task = self.storage.search_by_tags(
                tags=keywords,
                match_all=False,  # ANY tag matches
                limit=fetch_size,
            )

            vector_results, tag_matches = await asyncio.gather(vector_task, tag_task)

            # Combine using RRF
            combined = combine_results_rrf(vector_results, tag_matches, alpha)

            # Apply recency decay
            if config.recency_decay > 0:
                combined = apply_recency_decay(combined, config.recency_decay)

            # Apply pagination to combined results (offset calculated above for fetch_size)
            total = len(combined)
            paginated = combined[offset : offset + page_size]

            # Format results
            results = []
            for memory, score, debug_info in paginated:
                memory_dict = self._format_memory_response(memory)
                memory_dict["similarity_score"] = score
                memory_dict["hybrid_debug"] = debug_info
                results.append(memory_dict)

            return {
                "memories": results,
                "query": query,
                "hybrid_enabled": True,
                "alpha_used": alpha,
                "keywords_extracted": keywords,
                **self._build_pagination_metadata(total, page, page_size),
            }

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"memories": [], "query": query, "error": f"Failed to retrieve memories: {str(e)}"}

    async def search_by_tag(
        self,
        tags: str | list[str],
        match_all: bool = False,
        page: int = 1,
        page_size: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, list[MemoryResult] | str | bool | int]:
        """
        Search memories by tags with flexible matching options, pagination, and optional date filtering.

        Args:
            tags: Tag or list of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag
            page: Page number (1-indexed)
            page_size: Number of results per page
            start_date: Filter memories from this date (YYYY-MM-DD format)
            end_date: Filter memories until this date (YYYY-MM-DD format)

        Returns:
            Dictionary with matching memories and pagination metadata
        """
        try:
            # Normalize tags to list
            if isinstance(tags, str):
                tags = [tags]

            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Convert date strings to timestamps if provided
            from datetime import datetime

            start_timestamp = None
            end_timestamp = None

            if start_date:
                dt = datetime.fromisoformat(start_date)
                start_timestamp = datetime(dt.year, dt.month, dt.day).timestamp()

            if end_date:
                dt = datetime.fromisoformat(end_date)
                end_timestamp = datetime(dt.year, dt.month, dt.day, 23, 59, 59).timestamp()

            # Get total count for pagination
            total = await self.storage.count_tag_search(
                tags=tags, match_all=match_all, start_timestamp=start_timestamp, end_timestamp=end_timestamp
            )

            # Search using database-level filtering
            memories = await self.storage.search_by_tag(
                tags=tags,
                limit=page_size,
                offset=offset,
                match_all=match_all,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

            # Format results
            results = []
            for item in memories:
                # Handle both Memory and MemoryQueryResult objects
                if hasattr(item, "memory"):
                    results.append(self._format_memory_response(item.memory))
                else:
                    results.append(self._format_memory_response(item))

            # Determine match type description
            match_type = "ALL" if match_all else "ANY"

            return {
                "memories": results,
                "tags": tags,
                "match_type": match_type,
                **self._build_pagination_metadata(total, page, page_size),
            }

        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return {
                "memories": [],
                "tags": tags if isinstance(tags, list) else [tags],
                "error": f"Failed to search by tags: {str(e)}",
            }

    async def get_memory_by_hash(self, content_hash: str) -> dict[str, Any]:
        """
        Retrieve a specific memory by its content hash.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Dictionary with memory data or error
        """
        try:
            # Use efficient database-level hash lookup
            memory = await self.storage.get_memory_by_hash(content_hash)

            if memory:
                return {"memory": self._format_memory_response(memory), "found": True}
            else:
                return {"found": False, "content_hash": content_hash}

        except Exception as e:
            logger.error(f"Error getting memory by hash: {e}")
            return {"found": False, "content_hash": content_hash, "error": f"Failed to get memory: {str(e)}"}

    async def delete_memory(self, content_hash: str) -> dict[str, Any]:
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
                return {"success": True, "content_hash": content_hash}
            else:
                return {"success": False, "content_hash": content_hash, "error": message}

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "content_hash": content_hash, "error": f"Failed to delete memory: {str(e)}"}

    async def check_database_health(self) -> dict[str, Any]:
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
                **stats,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": f"Health check failed: {str(e)}"}

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
            "updated_at_iso": memory.updated_at_iso,
        }
