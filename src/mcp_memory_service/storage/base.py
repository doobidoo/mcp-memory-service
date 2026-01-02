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
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta, date
from ..models.memory import Memory, MemoryQueryResult

class MemoryStorage(ABC):
    """Abstract base class for memory storage implementations."""

    @property
    @abstractmethod
    def max_content_length(self) -> Optional[int]:
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
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory. Returns (success, message)."""
        pass

    async def store_batch(self, memories: List[Memory]) -> List[Tuple[bool, str]]:
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

        results = await asyncio.gather(
            *(self.store(memory) for memory in memories),
            return_exceptions=True
        )

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
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories by semantic search."""
        pass

    async def retrieve_with_quality_boost(
        self,
        query: str,
        n_results: int = 10,
        quality_boost: Optional[bool] = None,
        quality_weight: Optional[float] = None
    ) -> List[MemoryQueryResult]:
        """
        Retrieve memories with optional quality-based reranking.

        This method enables quality-aware search that prioritizes high-quality memories
        in the results. It over-fetches candidates (3x) then reranks by a composite score
        combining semantic similarity and quality scores.

        Args:
            query: Search query
            n_results: Number of results to return
            quality_boost: Enable quality reranking (default from config)
            quality_weight: Weight for quality score 0.0-1.0 (default 0.3, meaning 30% quality, 70% semantic)

        Returns:
            List of MemoryQueryResult, reranked by quality if enabled

        Example:
            # Standard search (semantic similarity only)
            results = await storage.retrieve("python async", n_results=10)

            # Quality-boosted search (70% semantic + 30% quality)
            results = await storage.retrieve_with_quality_boost(
                "python async",
                n_results=10,
                quality_boost=True,
                quality_weight=0.3
            )
        """
        from ..config import MCP_QUALITY_BOOST_ENABLED, MCP_QUALITY_BOOST_WEIGHT

        # Get config defaults if not specified
        if quality_boost is None:
            quality_boost = MCP_QUALITY_BOOST_ENABLED
        if quality_weight is None:
            quality_weight = MCP_QUALITY_BOOST_WEIGHT

        # Validate quality_weight
        if not 0.0 <= quality_weight <= 1.0:
            raise ValueError(f"quality_weight must be 0.0-1.0, got {quality_weight}")

        if not quality_boost:
            # Standard retrieval, no reranking
            return await self.retrieve(query, n_results)

        # Quality-boosted retrieval
        # Step 1: Over-fetch (3x) to have pool for reranking
        oversample_factor = 3
        candidates = await self.retrieve(query, n_results * oversample_factor)

        if not candidates:
            return []

        # Step 2: Rerank by composite score
        semantic_weight = 1.0 - quality_weight

        for result in candidates:
            semantic_score = result.relevance_score  # Original similarity
            quality_score = result.memory.quality_score

            # Composite score
            result.relevance_score = (
                semantic_weight * semantic_score +
                quality_weight * quality_score
            )

            # Store components in debug_info for transparency
            if result.debug_info is None:
                result.debug_info = {}
            result.debug_info.update({
                'original_semantic_score': semantic_score,
                'quality_score': quality_score,
                'quality_weight': quality_weight,
                'semantic_weight': semantic_weight,
                'reranked': True
            })

        # Step 3: Resort by new composite score
        candidates.sort(key=lambda r: r.relevance_score, reverse=True)

        # Step 4: Return top N
        return candidates[:n_results]

    @abstractmethod
    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """Search memories by tags with optional time filtering.

        Args:
            tags: List of tags to search for
            time_start: Optional Unix timestamp (in seconds) to filter memories created after this time

        Returns:
            List of Memory objects matching the tag criteria and time filter
        """
        pass

    @abstractmethod
    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> List[Memory]:
        """Search memories by tags with AND/OR semantics and time range filtering.

        Args:
            tags: List of tag names to search for
            operation: "AND" (all tags must match) or "OR" (any tag matches)
            time_start: Optional Unix timestamp for inclusive range start
            time_end: Optional Unix timestamp for inclusive range end

        Returns:
            List of Memory objects matching the criteria
        """
        pass

    async def search_by_tag_chronological(self, tags: List[str], limit: int = None, offset: int = 0) -> List[Memory]:
        """
        Search memories by tags with chronological ordering (newest first).

        Args:
            tags: List of tags to search for
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        # Default implementation: use search_by_tag then sort
        memories = await self.search_by_tag(tags)
        memories.sort(key=lambda m: m.created_at or 0, reverse=True)

        # Apply pagination
        if offset > 0:
            memories = memories[offset:]
        if limit is not None:
            memories = memories[:limit]

        return memories
    
    @abstractmethod
    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its hash."""
        pass

    async def is_deleted(self, content_hash: str) -> bool:
        """
        Check if a memory has been soft-deleted (tombstone exists).

        Used by hybrid sync to prevent re-syncing memories that were
        intentionally deleted on this device.

        Args:
            content_hash: The content hash of the memory to check

        Returns:
            True if the memory was soft-deleted, False otherwise

        Note:
            Default implementation returns False (no soft-delete support).
            Override in backends that support tombstones (e.g., sqlite_vec).
        """
        return False

    async def purge_deleted(self, older_than_days: int = 30) -> int:
        """
        Permanently delete tombstones older than specified days.

        This should be called periodically to clean up old soft-deleted records
        after they have been synced to all devices.

        Args:
            older_than_days: Delete tombstones older than this many days (default 30)

        Returns:
            Number of tombstones permanently deleted

        Note:
            Default implementation returns 0 (no soft-delete support).
            Override in backends that support tombstones (e.g., sqlite_vec).
        """
        return 0

    async def delete_by_timeframe(self, start_date: date, end_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """
        Delete memories within a specific date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            tag: Optional tag filter

        Returns:
            Tuple of (count, message)
        """
        raise NotImplementedError("Subclasses must implement delete_by_timeframe")

    async def delete_before_date(self, before_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """
        Delete memories created before a specific date.

        Args:
            before_date: Date threshold (exclusive - memories before this date are deleted)
            tag: Optional tag filter

        Returns:
            Tuple of (count, message)
        """
        raise NotImplementedError("Subclasses must implement delete_before_date")

    @abstractmethod
    async def get_by_exact_content(self, content: str) -> List[Memory]:
        """
        Retrieve memories by exact content match.

        Args:
            content: Exact content string to match

        Returns:
            List of Memory objects with matching content
        """
        raise NotImplementedError("Subclasses must implement get_by_exact_content")

    @abstractmethod
    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """
        Get a memory by its content hash using direct O(1) lookup.

        Args:
            content_hash: The content hash of the memory to retrieve

        Returns:
            Memory object if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag. Returns (count_deleted, message)."""
        pass

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str]:
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
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate memories. Returns (count_removed, message)."""
        pass
    
    @abstractmethod
    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
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
        updates = {
            'tags': memory.tags,
            'metadata': memory.metadata,
            'memory_type': memory.memory_type
        }
        success, _ = await self.update_memory_metadata(
            memory.content_hash,
            updates,
            preserve_timestamps=True
        )
        return success

    async def update_memories_batch(self, memories: List[Memory]) -> List[bool]:
        """
        Update multiple memories in a batch operation.

        Default implementation calls update_memory() for each memory concurrently using asyncio.gather.
        Override this method in concrete storage backends to provide true batch operations
        for improved performance (e.g., single database transaction with multiple UPDATEs).

        Args:
            memories: List of Memory objects with updated fields

        Returns:
            List of success booleans, one for each memory in the batch
        """
        if not memories:
            return []

        results = await asyncio.gather(
            *(self.update_memory(memory) for memory in memories),
            return_exceptions=True
        )

        # Process results to handle potential exceptions from gather
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                final_results.append(False)
            else:
                final_results.append(res)
        return final_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics. Override for specific implementations."""
        return {
            "total_memories": 0,
            "storage_backend": self.__class__.__name__,
            "status": "operational"
        }
    
    async def get_all_tags(self) -> List[str]:
        """Get all unique tags in the storage. Override for specific implementations."""
        return []
    
    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """Get n most recent memories. Override for specific implementations."""
        return []
    
    async def recall_memory(self, query: str, n_results: int = 5) -> List[Memory]:
        """Recall memories based on natural language time expression. Override for specific implementations."""
        # Default implementation just uses regular search
        results = await self.retrieve(query, n_results)
        return [r.memory for r in results]
    
    async def search(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Search memories. Default implementation uses retrieve."""
        return await self.retrieve(query, n_results)
    
    async def get_all_memories(self, limit: int = None, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
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
    
    async def count_all_memories(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (memories matching ANY of the tags)

        Returns:
            Total number of memories, optionally filtered by type and/or tags
        """
        return 0

    async def count_memories_by_tag(self, tags: List[str]) -> int:
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

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """Get memories within a time range. Override for specific implementations."""
        return []
    
    async def get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection statistics. Override for specific implementations."""
        return {}

    async def get_access_patterns(self) -> Dict[str, datetime]:
        """Get memory access pattern statistics. Override for specific implementations."""
        return {}

    async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
        """
        Get memory creation timestamps only, without loading full memory objects.

        This is an optimized method for analytics that only needs timestamps,
        avoiding the overhead of loading full memory content and embeddings.

        Args:
            days: Optional filter to only get memories from last N days

        Returns:
            List of Unix timestamps (float) in descending order (newest first)
        """
        # Default implementation falls back to get_recent_memories
        # Concrete backends should override with optimized SQL queries
        n = 5000 if days is None else days * 100  # Rough estimate
        memories = await self.get_recent_memories(n=n)
        timestamps = [m.created_at for m in memories if m.created_at]

        # Filter by days if specified
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_timestamp = cutoff.timestamp()
            timestamps = [ts for ts in timestamps if ts >= cutoff_timestamp]

        return sorted(timestamps, reverse=True)
