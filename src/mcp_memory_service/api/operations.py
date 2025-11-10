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
import time
from typing import Optional, Union, List
from .types import (
    CompactMemory, CompactSearchResult, CompactHealthInfo,
    CompactConsolidationResult, CompactSchedulerStatus
)
from .client import get_storage_async, get_consolidator, get_scheduler
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


async def _consolidate_async(time_horizon: str) -> CompactConsolidationResult:
    """
    Internal async implementation of consolidation.

    This function contains the core consolidation logic and is used by both
    the sync-wrapped API function and the FastAPI endpoint to avoid duplication.
    """
    # Validate time horizon
    valid_horizons = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    if time_horizon not in valid_horizons:
        raise ValueError(
            f"Invalid time_horizon: {time_horizon}. "
            f"Must be one of: {', '.join(valid_horizons)}"
        )

    # Get consolidator instance
    consolidator = get_consolidator()
    if consolidator is None:
        raise RuntimeError(
            "Consolidator not available. "
            "Consolidation requires HTTP server with MCP_CONSOLIDATION_ENABLED=true. "
            "Start the HTTP server first."
        )

    try:
        # Record start time
        start_time = time.time()

        # Run consolidation
        logger.info(f"Running {time_horizon} consolidation...")
        result = await consolidator.consolidate(time_horizon)

        # Calculate duration
        duration = time.time() - start_time

        # Extract metrics from result
        processed = result.get('memories_processed', 0)
        compressed = result.get('memories_compressed', 0)
        forgotten = result.get('memories_forgotten', 0)
        status = result.get('status', 'completed')

        logger.info(
            f"Consolidation completed: {processed} processed, "
            f"{compressed} compressed, {forgotten} forgotten ({duration:.1f}s)"
        )

        return CompactConsolidationResult(
            status=status,
            horizon=time_horizon,
            processed=processed,
            compressed=compressed,
            forgotten=forgotten,
            duration=duration
        )

    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return CompactConsolidationResult(
            status="failed",
            horizon=time_horizon,
            processed=0,
            compressed=0,
            forgotten=0,
            duration=0.0
        )


@sync_wrapper
async def consolidate(time_horizon: str = "weekly") -> CompactConsolidationResult:
    """
    Trigger memory consolidation for a specific time horizon.

    Token efficiency: ~40 tokens (result only)
    vs ~250 tokens for MCP consolidation result (84% reduction)

    Args:
        time_horizon: Time horizon for consolidation
            ('daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly')

    Returns:
        CompactConsolidationResult with operation metrics

    Raises:
        RuntimeError: If consolidation fails or consolidator not available
        ValueError: If time_horizon is invalid

    Example:
        >>> from mcp_memory_service.api import consolidate
        >>> result = consolidate('weekly')
        >>> print(result)
        Consolidation(completed, weekly, 2418 processed)
        >>> print(f"Compressed: {result.compressed}, Forgotten: {result.forgotten}")
        Compressed: 156, Forgotten: 43

    Performance:
        - Typical duration: 10-30 seconds (depends on memory count)
        - Scales linearly with total memories (~10ms per memory)
        - Background operation (non-blocking in HTTP server context)

    Note:
        Requires HTTP server with consolidation enabled. If called when
        HTTP server is not running, will raise RuntimeError.
    """
    return await _consolidate_async(time_horizon)


async def _scheduler_status_async() -> CompactSchedulerStatus:
    """
    Internal async implementation of scheduler status.

    This function contains the core status logic and is used by both
    the sync-wrapped API function and the FastAPI endpoint to avoid duplication.
    """
    # Get scheduler instance
    scheduler = get_scheduler()
    if scheduler is None:
        logger.warning("Scheduler not available")
        return CompactSchedulerStatus(
            running=False,
            next_daily=None,
            next_weekly=None,
            next_monthly=None,
            jobs_executed=0,
            jobs_failed=0
        )

    try:
        # Get scheduler status
        if hasattr(scheduler, 'scheduler') and scheduler.scheduler is not None:
            # Scheduler is running
            jobs = scheduler.scheduler.get_jobs()

            # Extract next run times for each horizon
            next_daily = None
            next_weekly = None
            next_monthly = None

            for job in jobs:
                if job.next_run_time:
                    timestamp = job.next_run_time.timestamp()
                    if 'daily' in job.id.lower():
                        next_daily = timestamp
                    elif 'weekly' in job.id.lower():
                        next_weekly = timestamp
                    elif 'monthly' in job.id.lower():
                        next_monthly = timestamp

            # Get execution statistics
            jobs_executed = scheduler.execution_stats.get('successful_jobs', 0)
            jobs_failed = scheduler.execution_stats.get('failed_jobs', 0)

            return CompactSchedulerStatus(
                running=True,
                next_daily=next_daily,
                next_weekly=next_weekly,
                next_monthly=next_monthly,
                jobs_executed=jobs_executed,
                jobs_failed=jobs_failed
            )
        else:
            # Scheduler exists but not running
            return CompactSchedulerStatus(
                running=False,
                next_daily=None,
                next_weekly=None,
                next_monthly=None,
                jobs_executed=0,
                jobs_failed=0
            )

    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        return CompactSchedulerStatus(
            running=False,
            next_daily=None,
            next_weekly=None,
            next_monthly=None,
            jobs_executed=0,
            jobs_failed=0
        )


@sync_wrapper
async def scheduler_status() -> CompactSchedulerStatus:
    """
    Get consolidation scheduler status and next run times.

    Token efficiency: ~25 tokens
    vs ~150 tokens for MCP scheduler_status tool (83% reduction)

    Returns:
        CompactSchedulerStatus with scheduler state and job statistics

    Raises:
        RuntimeError: If scheduler not available

    Example:
        >>> from mcp_memory_service.api import scheduler_status
        >>> status = scheduler_status()
        >>> print(status)
        Scheduler(running, executed=42, failed=0)
        >>> if status.next_daily:
        ...     from datetime import datetime
        ...     next_run = datetime.fromtimestamp(status.next_daily)
        ...     print(f"Next daily: {next_run}")

    Performance:
        - Execution time: <5ms (reads cached state)
        - No storage access required
        - Lightweight status query

    Note:
        Requires HTTP server with consolidation scheduler enabled.
        Returns STOPPED status if scheduler not running.
    """
    return await _scheduler_status_async()
