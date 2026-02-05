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
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..models.memory import Memory, MemoryQueryResult


class MemoryStorage(ABC):
    """Abstract base class for memory storage implementations."""

    @property
    @abstractmethod
    def max_content_length(self) -> int | None:
        """
        Maximum content length supported by this storage backend.

        Returns:
            Maximum number of characters allowed in memory content, or None for unlimited.
            This limit is based on the underlying embedding model's token limits.
        """
        pass

    @property
    @abstractmethod
    def supports_chunking(self) -> bool:
        """
        Whether this backend supports automatic content chunking.

        Returns:
            True if the backend can store chunked memories with linking metadata.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def store(self, memory: Memory) -> tuple[bool, str]:
        """Store a memory. Returns (success, message)."""
        pass

    async def store_batch(self, memories: list[Memory]) -> list[tuple[bool, str]]:
        """
        Store multiple memories in a single operation.

        Default implementation calls store() for each memory concurrently using asyncio.gather.
        Override this method in concrete storage backends to provide true batch operations
        for improved performance (e.g., single database transaction, bulk network request).

        Args:
            memories: List of Memory objects to store

        Returns:
            A list of (success, message) tuples, one for each memory in the batch.
        """
        if not memories:
            return []

        results = await asyncio.gather(*(self.store(memory) for memory in memories), return_exceptions=True)

        # Process results to handle potential exceptions from gather
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                # If a store operation failed with an exception, record it as a failure
                final_results.append((False, f"Failed to store memory: {res}"))
            else:
                final_results.append(res)
        return final_results

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """
        Retrieve memories by semantic search with optional filtering and pagination.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            tags: Optional list of tags to filter by (matches ANY tag)
            memory_type: Optional memory type filter
            min_similarity: Optional minimum similarity threshold
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            List of MemoryQueryResult objects, filtered and sorted by relevance
        """
        pass

    @abstractmethod
    async def search_by_tag(
        self,
        tags: list[str],
        limit: int = 10,
        offset: int = 0,
        match_all: bool = False,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> list[Memory]:
        """
        Search memories by tags with optional date filtering.

        Args:
            tags: List of tags to search for
            limit: Maximum number of results to return (default: 10)
            offset: Number of results to skip for pagination (default: 0)
            match_all: If True, memory must have ALL tags; if False, ANY tag (default: False)
            start_timestamp: Filter memories from this timestamp (inclusive)
            end_timestamp: Filter memories until this timestamp (inclusive)

        Returns:
            List of Memory objects
        """
        pass

    async def search_by_tag_chronological(self, tags: list[str], limit: int = None, offset: int = 0) -> list[Memory]:
        """
        Search memories by tags with chronological ordering (newest first).

        Args:
            tags: List of tags to search for
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        # Use search_by_tag with pagination (most backends handle ordering)
        # If limit is None, use a high default for backward compatibility
        effective_limit = limit if limit is not None else 10000
        memories = await self.search_by_tag(tags, limit=effective_limit, offset=offset)

        # Ensure chronological ordering (newest first)
        memories.sort(key=lambda m: m.created_at or 0, reverse=True)

        return memories

    @abstractmethod
    async def get_memory_by_hash(self, content_hash: str) -> Memory | None:
        """
        Retrieve a specific memory by its content hash.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Memory object if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, content_hash: str) -> tuple[bool, str]:
        """Delete a memory by its hash."""
        pass

    @abstractmethod
    async def delete_by_tag(self, tag: str) -> tuple[int, str]:
        """Delete memories by tag. Returns (count_deleted, message)."""
        pass

    async def delete_by_tags(self, tags: list[str]) -> tuple[int, str]:
        """
        Delete memories matching ANY of the given tags.

        Default implementation calls delete_by_tag for each tag sequentially.
        Override in concrete implementations for better performance (e.g., single query with OR).

        Args:
            tags: List of tags - memories matching ANY tag will be deleted

        Returns:
            Tuple of (total_count_deleted, message)
        """
        if not tags:
            return 0, "No tags provided"

        total_count = 0
        errors = []

        for tag in tags:
            try:
                count, message = await self.delete_by_tag(tag)
                total_count += count
                if "error" in message.lower() or "failed" in message.lower():
                    errors.append(f"{tag}: {message}")
            except Exception as e:
                errors.append(f"{tag}: {str(e)}")

        if errors:
            error_summary = "; ".join(errors[:3])  # Limit error details
            if len(errors) > 3:
                error_summary += f" (+{len(errors) - 3} more errors)"
            return total_count, f"Deleted {total_count} memories with partial failures: {error_summary}"

        return total_count, f"Deleted {total_count} memories across {len(tags)} tag(s)"

    @abstractmethod
    async def delete_by_all_tags(self, tags: list[str]) -> tuple[int, str]:
        """
        Delete memories matching ALL of the given tags (AND logic).

        Args:
            tags: List of tags - only memories containing ALL tags will be deleted

        Returns:
            Tuple of (count_deleted, message)
        """
        pass

    @abstractmethod
    async def cleanup_duplicates(self) -> tuple[int, str]:
        """Remove duplicate memories. Returns (count_removed, message)."""
        pass

    @abstractmethod
    async def update_memory_metadata(
        self, content_hash: str, updates: dict[str, Any], preserve_timestamps: bool = True
    ) -> tuple[bool, str]:
        """
        Update memory metadata without recreating the entire memory entry.

        Args:
            content_hash: Hash of the memory to update
            updates: Dictionary of metadata fields to update
            preserve_timestamps: Whether to preserve original created_at timestamp

        Returns:
            Tuple of (success, message)

        Note:
            - Only metadata, tags, and memory_type can be updated
            - Content and content_hash cannot be modified
            - updated_at timestamp is always refreshed
            - created_at is preserved unless preserve_timestamps=False
        """
        pass

    async def update_memory(self, memory: Memory) -> bool:
        """
        Update an existing memory with new metadata, tags, and memory_type.

        Args:
            memory: Memory object with updated fields

        Returns:
            True if update was successful, False otherwise
        """
        updates = {"tags": memory.tags, "metadata": memory.metadata, "memory_type": memory.memory_type}
        success, _ = await self.update_memory_metadata(memory.content_hash, updates, preserve_timestamps=True)
        return success

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics. Override for specific implementations."""
        return {"total_memories": 0, "storage_backend": self.__class__.__name__, "status": "operational"}

    async def get_all_tags(self) -> list[str]:
        """Get all unique tags in the storage. Override for specific implementations."""
        return []

    async def get_recent_memories(self, n: int = 10) -> list[Memory]:
        """Get n most recent memories. Override for specific implementations."""
        return []

    async def recall_memory(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Recall memories based on natural language time expression. Override for specific implementations."""
        # Default implementation just uses regular search
        results = await self.retrieve(query, n_results, tags, memory_type, min_similarity, offset)
        return [r.memory for r in results]

    async def search(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """Search memories. Default implementation uses retrieve."""
        return await self.retrieve(query, n_results, tags, memory_type, min_similarity, offset)

    async def get_all_memories(
        self, limit: int = None, offset: int = 0, memory_type: str | None = None, tags: list[str] | None = None
    ) -> list[Memory]:
        """
        Get all memories in storage ordered by creation time (newest first).

        Args:
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC, optionally filtered by type and tags
        """
        return []

    async def count_all_memories(self, memory_type: str | None = None, tags: list[str] | None = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (memories matching ANY of the tags)

        Returns:
            Total number of memories, optionally filtered by type and/or tags
        """
        return 0

    async def count_memories_by_tag(self, tags: list[str]) -> int:
        """
        Count memories that match any of the given tags.

        Args:
            tags: List of tags to search for

        Returns:
            Number of memories matching any tag
        """
        # Default implementation: search then count
        memories = await self.search_by_tag(tags)
        return len(memories)

    @abstractmethod
    async def count_semantic_search(
        self, query: str, tags: list[str] | None = None, memory_type: str | None = None, min_similarity: float | None = None
    ) -> int:
        """
        Count memories matching semantic search criteria.

        This enables pagination for semantic search without loading all results.
        Should perform same filtering as retrieve() but return count only.

        Args:
            query: Search query text
            tags: Optional list of tags to filter by (matches ANY tag)
            memory_type: Optional memory type filter
            min_similarity: Optional minimum similarity threshold

        Returns:
            Total number of memories matching the criteria
        """
        pass

    @abstractmethod
    async def count_tag_search(
        self, tags: list[str], match_all: bool = False, start_timestamp: float | None = None, end_timestamp: float | None = None
    ) -> int:
        """
        Count memories matching tag search with optional date filtering.

        Should use same filters as search_by_tag() for consistency.

        Args:
            tags: List of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag (default)
            start_timestamp: Filter memories from this timestamp (inclusive)
            end_timestamp: Filter memories until this timestamp (inclusive)

        Returns:
            Total number of memories matching the criteria
        """
        pass

    @abstractmethod
    async def count_time_range(
        self,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> int:
        """
        Count memories within time range with optional filters.

        Args:
            start_timestamp: Filter from this time (inclusive)
            end_timestamp: Filter until this time (inclusive)
            tags: Optional tag filter (ANY match)
            memory_type: Optional type filter

        Returns:
            Total number of memories matching the criteria
        """
        pass

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> list[Memory]:
        """Get memories within a time range. Override for specific implementations."""
        return []

    async def get_memory_connections(self) -> dict[str, int]:
        """Get memory connection statistics. Override for specific implementations."""
        return {}

    async def get_access_patterns(self) -> dict[str, datetime]:
        """Get memory access pattern statistics. Override for specific implementations."""
        return {}
