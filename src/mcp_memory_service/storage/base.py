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
from datetime import datetime
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
    
    @abstractmethod
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags."""
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
    
    @abstractmethod
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag. Returns (count_deleted, message)."""
        pass
    
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
    
    async def count_all_memories(self, memory_type: Optional[str] = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type

        Returns:
            Total number of memories, optionally filtered by type
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