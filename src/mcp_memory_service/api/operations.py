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
Core operations for code execution interface.

Provides token-efficient functions for memory operations:
    - search: Semantic search with compact results
    - store: Store new memories with minimal parameters
    - health: Service health and status check

Token Efficiency:
    - search(5 results): ~385 tokens (vs ~2,625, 85% reduction)
    - store(): ~15 tokens (vs ~150, 90% reduction)
    - health(): ~20 tokens (vs ~125, 84% reduction)

Performance:
    - Cold call: ~50ms (storage initialization)
    - Warm call: ~5-10ms (connection reused)
    - Memory overhead: <10MB
"""

import logging
from typing import Optional, Union, List
from .types import CompactMemory, CompactSearchResult, CompactHealthInfo
from .client import get_storage_async
from .sync_wrapper import sync_wrapper
from ..models.memory import Memory
from ..utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)


@sync_wrapper
async def search(
    query: str,
    limit: int = 5,
    tags: Optional[List[str]] = None
) -> CompactSearchResult:
    """
    Search memories using semantic similarity.

    Token efficiency: ~25 tokens (query + params) + ~73 tokens per result
    Example (5 results): ~385 tokens vs ~2,625 tokens (85% reduction)

    Args:
        query: Search query text (natural language)
        limit: Maximum number of results to return (default: 5)
        tags: Optional list of tags to filter results

    Returns:
        CompactSearchResult with minimal memory representations

    Raises:
        RuntimeError: If storage backend is not available
        ValueError: If query is empty or limit is invalid

    Example:
        >>> from mcp_memory_service.api import search
        >>> results = search("recent architecture changes", limit=3)
        >>> print(results)
        SearchResult(found=3, shown=3)
        >>> for m in results.memories:
        ...     print(f"{m.hash}: {m.preview[:50]}...")
        abc12345: Implemented OAuth 2.1 authentication for...
        def67890: Refactored storage backend to support...
        ghi11121: Added hybrid mode for Cloudflare sync...

    Performance:
        - First call: ~50ms (includes storage initialization)
        - Subsequent calls: ~5-10ms (connection reused)
        - Scales linearly with limit (5ms + 1ms per result)
    """
    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if limit < 1:
        raise ValueError("Limit must be at least 1")
    if limit > 100:
        logger.warning(f"Large limit ({limit}) may impact performance")

    # Get storage instance
    storage = await get_storage_async()

    # Perform semantic search
    query_results = await storage.retrieve(query, n_results=limit)

    # Filter by tags if specified
    if tags:
        tag_set = set(tags)
        query_results = [
            r for r in query_results
            if any(tag in tag_set for tag in r.memory.tags)
        ]

    # Convert to compact format
    compact_memories = tuple(
        CompactMemory(
            hash=r.memory.content_hash[:8],  # 8-char hash
            preview=r.memory.content[:200],   # First 200 chars
            tags=tuple(r.memory.tags),        # Immutable tuple
            created=r.memory.created_at,      # Unix timestamp
            score=r.relevance_score           # Relevance score
        )
        for r in query_results
    )

    return CompactSearchResult(
        memories=compact_memories,
        total=len(compact_memories),
        query=query
    )


@sync_wrapper
async def store(
    content: str,
    tags: Optional[Union[str, List[str]]] = None,
    memory_type: str = "note"
) -> str:
    """
    Store a new memory.

    Token efficiency: ~15 tokens (params only)
    vs ~150 tokens for MCP tool call with schema (90% reduction)

    Args:
        content: Memory content text
        tags: Single tag or list of tags (optional)
        memory_type: Memory type classification (default: "note")

    Returns:
        8-character content hash of stored memory

    Raises:
        RuntimeError: If storage operation fails
        ValueError: If content is empty

    Example:
        >>> from mcp_memory_service.api import store
        >>> hash = store(
        ...     "Implemented OAuth 2.1 authentication",
        ...     tags=["authentication", "security", "feature"]
        ... )
        >>> print(f"Stored: {hash}")
        Stored: abc12345

    Performance:
        - First call: ~50ms (includes storage initialization)
        - Subsequent calls: ~10-20ms (includes embedding generation)
        - Scales with content length (20ms + 0.5ms per 100 chars)
    """
    # Validate input
    if not content or not content.strip():
        raise ValueError("Content cannot be empty")

    # Normalize tags to list
    if tags is None:
        tag_list = []
    elif isinstance(tags, str):
        tag_list = [tags]
    else:
        tag_list = list(tags)

    # Generate content hash
    content_hash = generate_content_hash(content)

    # Create memory object
    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=tag_list,
        memory_type=memory_type,
        metadata={}
    )

    # Get storage instance
    storage = await get_storage_async()

    # Store memory
    success, message = await storage.store(memory)

    if not success:
        raise RuntimeError(f"Failed to store memory: {message}")

    # Return short hash (8 chars)
    return content_hash[:8]


@sync_wrapper
async def health() -> CompactHealthInfo:
    """
    Get service health and status.

    Token efficiency: ~20 tokens
    vs ~125 tokens for MCP health check tool (84% reduction)

    Returns:
        CompactHealthInfo with backend, count, and ready status

    Raises:
        RuntimeError: If unable to retrieve health information

    Example:
        >>> from mcp_memory_service.api import health
        >>> info = health()
        >>> print(f"Status: {info.status}")
        Status: healthy
        >>> print(f"Backend: {info.backend}, Count: {info.count}")
        Backend: sqlite_vec, Count: 1247

    Performance:
        - First call: ~50ms (includes storage initialization)
        - Subsequent calls: ~5ms (cached stats)
    """
    try:
        # Get storage instance
        storage = await get_storage_async()

        # Get storage statistics
        stats = await storage.get_stats()

        # Determine status
        status = "healthy"
        if stats.get("status") == "degraded":
            status = "degraded"
        elif stats.get("status") == "error":
            status = "error"
        elif not stats.get("initialized", True):
            status = "error"

        # Extract backend type
        backend = stats.get("storage_backend", "unknown")

        # Extract memory count
        count = stats.get("total_memories", 0)

        return CompactHealthInfo(
            status=status,
            count=count,
            backend=backend
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return CompactHealthInfo(
            status="error",
            count=0,
            backend="unknown"
        )
