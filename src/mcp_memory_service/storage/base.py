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
import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta, date
from ..models.memory import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)

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

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str, List[str]]:
        """
        Delete memories matching ANY of the given tags.

        Default implementation calls delete_by_tag for each tag sequentially.
        Override in concrete implementations for better performance (e.g., single query with OR).

        Args:
            tags: List of tags - memories matching ANY tag will be deleted

        Returns:
            Tuple of (total_count_deleted, message, deleted_hashes)
            Note: Base implementation returns empty list for deleted_hashes since
            delete_by_tag doesn't track individual hashes. Override in concrete
            implementations to provide hash tracking.
        """
        if not tags:
            return 0, "No tags provided", []

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
            return total_count, f"Deleted {total_count} memories with partial failures: {error_summary}", []

        return total_count, f"Deleted {total_count} memories across {len(tags)} tag(s)", []

    async def delete_memories(
        self,
        content_hash: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        before: Optional[str] = None,
        after: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Unified memory deletion with flexible filtering.

        This method consolidates all delete operations into a single flexible interface.
        Combines filters with AND logic for precise targeting.

        Args:
            content_hash: Specific memory hash to delete (ignores other filters if provided)
            tags: List of tags to filter by
            tag_match: "any" to match ANY tag, "all" to match ALL tags (default: "any")
            before: Delete memories created before this ISO date (YYYY-MM-DD)
            after: Delete memories created after this ISO date (YYYY-MM-DD)
            dry_run: If True, preview deletions without executing (default: False)

        Returns:
            Dictionary with:
                - success: bool
                - deleted_count: int
                - deleted_hashes: List[str] (if dry_run, shows what would be deleted)
                - dry_run: bool
                - message: str

        Filter Logic:
            - If content_hash: Delete single memory (ignores other filters)
            - If tags: Filter by tags using tag_match mode
            - If before/after: Filter by time range
            - Multiple filters combine with AND logic
            - dry_run: Preview deletions without executing

        Examples:
            # Delete single memory
            await storage.delete_memories(content_hash="abc123")

            # Delete by tags (ANY match)
            await storage.delete_memories(tags=["temporary", "draft"])

            # Delete by tags (ALL match)
            await storage.delete_memories(tags=["archived", "old"], tag_match="all")

            # Delete by time range
            await storage.delete_memories(before="2024-01-01")
            await storage.delete_memories(after="2024-06-01", before="2024-12-31")

            # Combined filters
            await storage.delete_memories(tags=["cleanup"], before="2024-01-01")

            # Dry run preview
            await storage.delete_memories(tags=["test"], dry_run=True)
        """
        try:
            # Validate tag_match parameter
            if tag_match not in ("any", "all"):
                return {
                    "success": False,
                    "deleted_count": 0,
                    "deleted_hashes": [],
                    "dry_run": dry_run,
                    "error": f"Invalid tag_match: {tag_match}. Must be 'any' or 'all'"
                }

            # Case 1: Delete single memory by hash
            if content_hash:
                if dry_run:
                    # Check if memory exists
                    memory = await self.get_by_hash(content_hash)
                    if memory:
                        return {
                            "success": True,
                            "deleted_count": 1,
                            "deleted_hashes": [content_hash],
                            "dry_run": True,
                            "message": f"Would delete 1 memory with hash: {content_hash}"
                        }
                    else:
                        return {
                            "success": False,
                            "deleted_count": 0,
                            "deleted_hashes": [],
                            "dry_run": True,
                            "error": f"Memory not found: {content_hash}"
                        }

                # Execute deletion
                success, message = await self.delete(content_hash)
                return {
                    "success": success,
                    "deleted_count": 1 if success else 0,
                    "deleted_hashes": [content_hash] if success else [],
                    "dry_run": False,
                    "message": message if success else message,
                    "error": message if not success else None
                }

            # Case 2: No filters provided - error to prevent accidental mass deletion
            if not tags and not before and not after:
                return {
                    "success": False,
                    "deleted_count": 0,
                    "deleted_hashes": [],
                    "dry_run": dry_run,
                    "error": "At least one filter required (content_hash, tags, before, or after)"
                }

            # Case 3: Filter-based deletion (tags and/or time range)
            # Try optimized SQL-level filtering first (Issue #374 optimization)
            use_optimized = False
            filtered_memories = []

            # Check for optimized backend methods - SQL filtering is 30-280x faster
            if tags and not before and not after and tag_match == "any":
                # Optimized path: tag-only deletion with ANY match
                if hasattr(self, 'delete_by_tags') and not dry_run:
                    use_optimized = True
                    count, message, deleted_hashes = await self.delete_by_tags(tags)
                    return {
                        "success": count > 0,
                        "deleted_count": count,
                        "deleted_hashes": deleted_hashes,
                        "dry_run": False,
                        "message": message
                    }
                elif hasattr(self, 'get_memories_by_tags'):
                    # Use optimized query for dry_run
                    use_optimized = True
                    filtered_memories = await self.get_memories_by_tags(tags, match_mode="any")

            elif (before or after) and not tags:
                # Optimized path: time-only filtering (no tags)
                if hasattr(self, 'get_memories_by_time_range'):
                    use_optimized = True
                    try:
                        # Convert date strings to timestamps
                        if after:
                            after_date = datetime.fromisoformat(after)
                            start_time = after_date.timestamp()
                        else:
                            start_time = 0.0

                        if before:
                            before_date = datetime.fromisoformat(before)
                            end_time = before_date.timestamp()
                        else:
                            end_time = datetime.now().timestamp()

                        filtered_memories = await self.get_memories_by_time_range(
                            start_time=start_time,
                            end_time=end_time
                        )
                    except ValueError as e:
                        return {
                            "success": False,
                            "deleted_count": 0,
                            "deleted_hashes": [],
                            "dry_run": dry_run,
                            "error": f"Invalid date format: {e}. Use YYYY-MM-DD"
                        }

            # Fallback: Load all memories and filter in Python (slower but always works)
            if not use_optimized:
                all_memories = await self.get_all_memories()

                for memory in all_memories:
                    # Tag filter
                    if tags:
                        if tag_match == "any":
                            # Match ANY tag
                            if not any(tag in memory.tags for tag in tags):
                                continue
                        else:  # tag_match == "all"
                            # Match ALL tags
                            if not all(tag in memory.tags for tag in tags):
                                continue

                    # Time filters
                    if before or after:
                        # Parse memory creation time
                        try:
                            memory_date = datetime.fromisoformat(memory.created_at_iso.replace('Z', '+00:00')).date()
                        except Exception:
                            # Skip memories with invalid timestamps
                            continue

                        if before:
                            try:
                                before_date = datetime.fromisoformat(before).date()
                                if memory_date >= before_date:
                                    continue
                            except ValueError:
                                return {
                                    "success": False,
                                    "deleted_count": 0,
                                    "deleted_hashes": [],
                                    "dry_run": dry_run,
                                    "error": f"Invalid before date format: {before}. Use YYYY-MM-DD"
                                }

                        if after:
                            try:
                                after_date = datetime.fromisoformat(after).date()
                                if memory_date <= after_date:
                                    continue
                            except ValueError:
                                return {
                                    "success": False,
                                    "deleted_count": 0,
                                    "deleted_hashes": [],
                                    "dry_run": dry_run,
                                    "error": f"Invalid after date format: {after}. Use YYYY-MM-DD"
                                }

                    # Memory passed all filters
                    filtered_memories.append(memory)

            # Dry run: return what would be deleted
            if dry_run:
                deleted_hashes = [m.content_hash for m in filtered_memories]
                return {
                    "success": True,
                    "deleted_count": len(filtered_memories),
                    "deleted_hashes": deleted_hashes,
                    "dry_run": True,
                    "message": f"Would delete {len(filtered_memories)} memories"
                }

            # Execute deletions
            deleted_count = 0
            deleted_hashes = []
            errors = []

            for memory in filtered_memories:
                try:
                    success, msg = await self.delete(memory.content_hash)
                    if success:
                        deleted_count += 1
                        deleted_hashes.append(memory.content_hash)
                    else:
                        errors.append(f"{memory.content_hash}: {msg}")
                except Exception as e:
                    errors.append(f"{memory.content_hash}: {str(e)}")

            # Build response
            if errors:
                error_summary = "; ".join(errors[:3])
                if len(errors) > 3:
                    error_summary += f" (+{len(errors) - 3} more errors)"
                return {
                    "success": deleted_count > 0,
                    "deleted_count": deleted_count,
                    "deleted_hashes": deleted_hashes,
                    "dry_run": False,
                    "message": f"Deleted {deleted_count} memories with {len(errors)} failures",
                    "error": error_summary
                }

            return {
                "success": True,
                "deleted_count": deleted_count,
                "deleted_hashes": deleted_hashes,
                "dry_run": False,
                "message": f"Successfully deleted {deleted_count} memories"
            }

        except Exception as e:
            return {
                "success": False,
                "deleted_count": 0,
                "deleted_hashes": [],
                "dry_run": dry_run,
                "error": f"Delete operation failed: {str(e)}"
            }

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

    async def get_relationship_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of relationship types in the knowledge graph.

        Returns:
            Dictionary mapping relationship type names to counts.
            Example: {"causes": 45, "fixes": 23, "related": 102, ...}
        """
        return {}

    async def get_graph_visualization_data(
        self,
        limit: int = 100,
        min_connections: int = 1
    ) -> Dict[str, Any]:
        """
        Get graph data for visualization in D3.js-compatible format.

        Fetches the most connected memories and their relationships for
        interactive force-directed graph rendering.

        Args:
            limit: Maximum number of nodes to include
            min_connections: Minimum number of connections a memory must have to be included

        Returns:
            Dictionary with "nodes" and "edges" keys in D3.js format:
            {
                "nodes": [
                    {
                        "id": "hash123",
                        "type": "observation",
                        "content": "preview text",
                        "connections": 5
                    },
                    ...
                ],
                "edges": [
                    {
                        "source": "hash123",
                        "target": "hash456",
                        "relationship_type": "causes",
                        "similarity": 0.85
                    },
                    ...
                ]
            }
        """
        return {"nodes": [], "edges": []}

    async def search_memories(
        self,
        query: Optional[str] = None,
        mode: str = "semantic",
        time_expr: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        tags: Optional[List[str]] = None,
        quality_boost: float = 0.0,
        limit: int = 10,
        include_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Unified memory search with flexible modes and filters.

        This method consolidates all search/retrieve operations into a single flexible interface.
        Supports semantic search, exact matching, time-based filtering, tag filtering, and
        quality-based reranking.

        Args:
            query: Search query string (required for semantic/exact modes, optional for time-only)
            mode: Search mode - "semantic" (default), "exact", or "hybrid"
            time_expr: Natural language time filter (e.g., "last week", "yesterday", "2 days ago")
            after: Return memories created after this ISO date (YYYY-MM-DD)
            before: Return memories created before this ISO date (YYYY-MM-DD)
            tags: Filter to memories with any of these tags
            quality_boost: Quality weight for reranking 0.0-1.0 (0.0=pure semantic, 1.0=pure quality)
            limit: Maximum number of results to return (default: 10)
            include_debug: Include debug information in response (default: False)

        Returns:
            Dictionary with:
                - memories: List[Memory] - Matching memories
                - total: int - Number of results
                - query: str - The query used
                - mode: str - The search mode used
                - debug: Dict (if include_debug=True) - Debug information

        Modes:
            - semantic: Vector similarity search (default) - finds conceptually similar content
            - exact: Exact string match in content - finds memories containing the exact query
            - hybrid: Semantic search with quality-based reranking

        Time Filters:
            - time_expr: Natural language like "yesterday", "last week", "2 days ago", "last month"
            - after/before: Explicit ISO dates (YYYY-MM-DD)
            - If both time_expr and after/before provided, time_expr takes precedence

        Quality Boost (for semantic/hybrid modes):
            - 0.0: Pure semantic ranking (default)
            - 0.3: 30% quality weight, 70% semantic (recommended for important lookups)
            - 1.0: Pure quality ranking

        Examples:
            # Semantic search
            await storage.search_memories(query="python async patterns")

            # Exact match
            await storage.search_memories(query="async def", mode="exact")

            # Time-based
            await storage.search_memories(time_expr="last week", limit=20)

            # Combined filters
            await storage.search_memories(
                query="database config",
                time_expr="yesterday",
                tags=["important"]
            )

            # Quality-boosted search
            await storage.search_memories(
                query="architecture decisions",
                tags=["important"],
                quality_boost=0.3,
                mode="hybrid"
            )

            # Time range search
            await storage.search_memories(
                after="2024-01-01",
                before="2024-06-30",
                limit=50
            )

            # Debug mode
            await storage.search_memories(
                query="error handling",
                include_debug=True
            )
        """
        try:
            # Validate mode
            if mode not in ("semantic", "exact", "hybrid"):
                return {
                    "memories": [],
                    "total": 0,
                    "query": query,
                    "mode": mode,
                    "error": f"Invalid mode: {mode}. Must be 'semantic', 'exact', or 'hybrid'"
                }

            # Validate quality_boost
            if not 0.0 <= quality_boost <= 1.0:
                return {
                    "memories": [],
                    "total": 0,
                    "query": query,
                    "mode": mode,
                    "error": f"Invalid quality_boost: {quality_boost}. Must be 0.0-1.0"
                }

            pre_filter_count = 0
            start_time = None
            end_time = None

            # Parse time expression if provided
            if time_expr:
                try:
                    from ..utils.time_parser import parse_time_expression
                    # Parse time range from natural language
                    start_timestamp, end_timestamp = parse_time_expression(time_expr)
                    if start_timestamp is not None:
                        start_time = start_timestamp
                    if end_timestamp is not None:
                        end_time = end_timestamp
                except Exception as e:
                    # Continue without time filter rather than failing
                    pass

            # Use explicit after/before if no time_expr
            if not time_expr:
                if after:
                    try:
                        after_date = datetime.fromisoformat(after)
                        start_time = after_date.timestamp()
                    except ValueError:
                        return {
                            "memories": [],
                            "total": 0,
                            "query": query,
                            "mode": mode,
                            "error": f"Invalid after date format: {after}. Use YYYY-MM-DD"
                        }

                if before:
                    try:
                        before_date = datetime.fromisoformat(before)
                        end_time = before_date.timestamp()
                    except ValueError:
                        return {
                            "memories": [],
                            "total": 0,
                            "query": query,
                            "mode": mode,
                            "error": f"Invalid before date format: {before}. Use YYYY-MM-DD"
                        }

            # Perform search based on mode
            results = []

            if mode == "exact":
                # Exact string match
                if not query:
                    return {
                        "memories": [],
                        "total": 0,
                        "query": query,
                        "mode": mode,
                        "error": "query required for exact mode"
                    }

                # Get memories with exact content match
                matched_memories = await self.get_by_exact_content(query)
                # Convert to MemoryQueryResult format for consistency
                results = [
                    MemoryQueryResult(memory=m, relevance_score=1.0, debug_info=None)
                    for m in matched_memories
                ]
                pre_filter_count = len(results)

            elif mode in ("semantic", "hybrid"):
                # Vector similarity search
                if not query and not start_time and not end_time and not tags:
                    return {
                        "memories": [],
                        "total": 0,
                        "query": query,
                        "mode": mode,
                        "error": "At least one filter required (query, time_expr, after, before, or tags)"
                    }

                if query:
                    # Determine fetch limit (over-fetch if quality boost enabled)
                    fetch_limit = limit * 3 if quality_boost > 0 and mode == "hybrid" else limit

                    # Choose search method based on mode and available features
                    if mode == "hybrid" and hasattr(self, 'retrieve_hybrid'):
                        # Use hybrid search (BM25 + Vector)
                        from ..config import MCP_HYBRID_SEARCH_ENABLED
                        if MCP_HYBRID_SEARCH_ENABLED:
                            results = await self.retrieve_hybrid(
                                query,
                                n_results=fetch_limit
                            )
                        else:
                            # Fall back to semantic if hybrid disabled in config
                            logger.warning("Hybrid search requested but disabled in config, using semantic")
                            if quality_boost > 0:
                                results = await self.retrieve_with_quality_boost(
                                    query,
                                    n_results=fetch_limit,
                                    quality_boost=True,
                                    quality_weight=quality_boost
                                )
                            else:
                                results = await self.retrieve(query, n_results=fetch_limit)
                    elif quality_boost > 0:
                        # Use quality-boosted retrieval
                        results = await self.retrieve_with_quality_boost(
                            query,
                            n_results=fetch_limit,
                            quality_boost=True,
                            quality_weight=quality_boost
                        )
                    else:
                        # Standard semantic search
                        results = await self.retrieve(query, n_results=fetch_limit)

                    pre_filter_count = len(results)
                else:
                    # Time-only or tag-only search - try optimized path first (Issue #374)
                    use_optimized_search = False

                    # Optimized: time-range only query
                    if (start_time is not None or end_time is not None) and not tags:
                        if hasattr(self, 'get_memories_by_time_range'):
                            use_optimized_search = True
                            st = start_time if start_time is not None else 0.0
                            et = end_time if end_time is not None else datetime.now().timestamp()
                            memories = await self.get_memories_by_time_range(st, et)
                            results = [
                                MemoryQueryResult(memory=m, relevance_score=0.5, debug_info=None)
                                for m in memories
                            ]
                            pre_filter_count = len(results)

                    # Fallback: load all memories then filter
                    if not use_optimized_search:
                        all_memories = await self.get_all_memories()
                        results = [
                            MemoryQueryResult(memory=m, relevance_score=0.5, debug_info=None)
                            for m in all_memories
                        ]
                        pre_filter_count = len(results)

            # Apply time filters
            if start_time is not None or end_time is not None:
                filtered_results = []
                for result in results:
                    memory_timestamp = result.memory.created_at

                    # Skip if no timestamp
                    if not memory_timestamp:
                        continue

                    # Check time range
                    if start_time is not None and memory_timestamp < start_time:
                        continue
                    if end_time is not None and memory_timestamp > end_time:
                        continue

                    filtered_results.append(result)

                results = filtered_results

            # Apply tag filters
            if tags:
                filtered_results = []
                for result in results:
                    # Match ANY tag
                    if any(tag in result.memory.tags for tag in tags):
                        filtered_results.append(result)
                results = filtered_results

            # Limit results
            results = results[:limit]

            # Extract memories from results
            # Convert results to flat dictionary format with similarity_score
            memories = []
            for r in results:
                mem_dict = r.memory.to_dict()
                mem_dict["similarity_score"] = r.relevance_score
                if r.debug_info:
                    mem_dict["debug_info"] = r.debug_info
                memories.append(mem_dict)

            # Build response
            response = {
                "memories": memories,
                "total": len(memories),
                "query": query,
                "mode": mode
            }

            # Add debug info if requested
            if include_debug:
                response["debug"] = {
                    "time_filter": {
                        "time_expr": time_expr,
                        "after": after,
                        "before": before,
                        "start_timestamp": start_time,
                        "end_timestamp": end_time
                    },
                    "tag_filter": tags,
                    "quality_boost": quality_boost,
                    "pre_filter_count": pre_filter_count,
                    "post_filter_count": len(memories),
                    "limit": limit
                }

            return response

        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return {
                "memories": [],
                "total": 0,
                "query": query,
                "mode": mode,
                "error": f"Search operation failed: {str(e)}"
            }
