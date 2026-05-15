"""Abstract base adapter for benchmark comparisons."""

from abc import ABC, abstractmethod
from typing import Any


class MemoryAdapter(ABC):
    """Interface that all memory tool adapters must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for reports."""
        ...

    @abstractmethod
    async def setup(self) -> None:
        """Initialize connection, create collections, etc."""
        ...

    @abstractmethod
    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        """Store a memory. Returns content hash/ID."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {content, score, metadata}."""
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup resources, delete temp collections."""
        ...
