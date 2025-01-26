"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from ..models.memory import Memory, MemoryQueryResult

class MemoryStorage(ABC):
    """Abstract base class for memory storage implementations."""
    
    @abstractmethod
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory. Returns (success, message)."""
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories by semantic search."""
        pass

    @abstractmethod
    async def recall(
        self,
        query: Optional[str] = None,
        n_results: int = 5,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> List[MemoryQueryResult]:
        """
        Retrieve memories using semantic search and/or time filtering.
        
        Args:
            query: Optional semantic search query. If None, returns most recent memories
            n_results: Maximum number of results to return
            start_timestamp: Optional start time for filtering (inclusive)
            end_timestamp: Optional end time for filtering (inclusive)
            
        Returns:
            List of MemoryQueryResult objects sorted by relevance (if query provided) 
            or timestamp (if no query)
            
        Raises:
            ValueError: If invalid timestamp range is provided
        """
        pass

    @abstractmethod
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags."""
        pass
    
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