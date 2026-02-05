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
Storage backends for MCP Memory Service.

Provides:
- BaseStorage: Protocol defining storage interface
- MemoryStorage: ABC for backward compatibility
- SqliteVecMemoryStorage: SQLite-vec backend (development)
- QdrantStorage: Qdrant backend (production)
"""

from typing import Any, Protocol, runtime_checkable

from ..models.memory import Memory, MemoryQueryResult


@runtime_checkable
class BaseStorage(Protocol):
    """
    Protocol defining the core storage interface.

    Implementations must provide these methods for compatibility.
    Uses structural subtyping - no explicit inheritance required.
    """

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        ...

    async def store(self, memory: Memory) -> tuple[bool, str]:
        """Store a memory. Returns (success, message)."""
        ...

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """Retrieve memories by semantic search."""
        ...

    async def search_by_tag(
        self,
        tags: list[str],
        limit: int = 10,
        offset: int = 0,
        match_all: bool = False,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> list[Memory]:
        """Search memories by tags."""
        ...

    async def get_all_memories(
        self, limit: int | None = None, offset: int = 0, memory_type: str | None = None, tags: list[str] | None = None
    ) -> list[Memory]:
        """List all memories with optional filters."""
        ...

    async def delete(self, content_hash: str) -> tuple[bool, str]:
        """Delete a memory by hash."""
        ...

    async def count_all_memories(self, memory_type: str | None = None, tags: list[str] | None = None) -> int:
        """Count total memories."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get storage health and statistics."""
        ...

    async def close(self) -> None:
        """Close storage connections."""
        ...


# Export ABC for backward compatibility
from .base import MemoryStorage  # noqa: E402

# Export factory function
from .factory import create_storage_instance, get_storage_backend_class  # noqa: E402

__all__ = [
    "BaseStorage",
    "MemoryStorage",
    "create_storage_instance",
    "get_storage_backend_class",
]

# Conditional imports for backends
try:
    from .sqlite_vec import SqliteVecMemoryStorage

    __all__.append("SqliteVecMemoryStorage")
except ImportError:
    SqliteVecMemoryStorage = None

try:
    from .qdrant_storage import QdrantStorage

    __all__.append("QdrantStorage")
except ImportError:
    QdrantStorage = None
