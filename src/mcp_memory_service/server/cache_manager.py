"""
Cache management module for MCP Memory Service.

Copyright (c) 2024 mcp-memory-service
Licensed under the MIT License. See LICENSE file in the project root.

GLOBAL CACHING FOR MCP SERVER PERFORMANCE OPTIMIZATION
========================================================
Module-level caches to persist storage/service instances across stateless HTTP calls.
This reduces initialization overhead from ~1,810ms to <400ms on cache hits.

Cache Keys:
- Storage: "{backend_type}:{db_path}" (e.g., "sqlite_vec:/path/to/db")
- MemoryService: storage instance ID (id(storage))

Thread Safety:
- Uses asyncio.Lock to prevent race conditions during concurrent access

Lifecycle:
- Cached instances persist for the lifetime of the Python process
- NOT cleared between stateless HTTP calls (intentional for performance)
- Cleaned up on process shutdown via lifespan context manager
"""

import asyncio
import time
from typing import Dict, Any, Optional

from .logging_config import logger


# =============================================================================
# GLOBAL CACHE STORAGE
# =============================================================================

_STORAGE_CACHE: Dict[str, Any] = {}  # Storage instances keyed by "{backend}:{path}"
_MEMORY_SERVICE_CACHE: Dict[int, Any] = {}  # MemoryService instances keyed by storage ID
_CACHE_LOCK: Optional[asyncio.Lock] = None  # Initialized on first use to avoid event loop issues
_CACHE_STATS = {
    "storage_hits": 0,
    "storage_misses": 0,
    "service_hits": 0,
    "service_misses": 0,
    "total_calls": 0,
    "initialization_times": []  # Track initialization durations for cache misses
}


# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def _get_cache_lock() -> asyncio.Lock:
    """Get or create the global cache lock (lazy initialization to avoid event loop issues)."""
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    return _CACHE_LOCK


def _get_or_create_memory_service(storage: Any) -> Any:
    """
    Get cached MemoryService or create new one.

    Args:
        storage: Storage instance to use as cache key

    Returns:
        MemoryService instance (cached or newly created)
    """
    from ..services.memory_service import MemoryService

    storage_id = id(storage)
    if storage_id in _MEMORY_SERVICE_CACHE:
        memory_service = _MEMORY_SERVICE_CACHE[storage_id]
        _CACHE_STATS["service_hits"] += 1
        logger.info("✅ MemoryService Cache HIT - Reusing service instance (storage_id: %d)", storage_id)
    else:
        _CACHE_STATS["service_misses"] += 1
        logger.info("❌ MemoryService Cache MISS - Creating new service instance...")

        # Initialize memory service with shared business logic
        memory_service = MemoryService(storage)

        # Cache the memory service instance
        _MEMORY_SERVICE_CACHE[storage_id] = memory_service
        logger.info("💾 Cached MemoryService instance (storage_id: %d)", storage_id)

    return memory_service


def _log_cache_performance(start_time: float) -> None:
    """
    Log comprehensive cache performance statistics.

    Args:
        start_time: Timer start time to calculate total elapsed time
    """
    total_time = (time.time() - start_time) * 1000
    cache_hit_rate = (
        (_CACHE_STATS["storage_hits"] + _CACHE_STATS["service_hits"]) /
        (_CACHE_STATS["total_calls"] * 2)  # 2 caches per call
    ) * 100

    logger.info(
        "📊 Cache Stats - "
        "Hit Rate: %.1f%% | "
        "Storage: %dH/%dM | "
        "Service: %dH/%dM | "
        "Total Time: %.1fms",
        cache_hit_rate,
        _CACHE_STATS["storage_hits"],
        _CACHE_STATS["storage_misses"],
        _CACHE_STATS["service_hits"],
        _CACHE_STATS["service_misses"],
        total_time,
    )


# =============================================================================
# CACHE CLEANUP FUNCTIONS (v8.71.0 - Memory Management, Discussion #331)
# =============================================================================

def clear_all_caches() -> Dict[str, int]:
    """
    Clear all global caches to free memory.

    This function is used during graceful shutdown or when memory pressure
    is detected. It clears both storage and service caches.

    Returns:
        Dict with counts of cleared items:
        - storage_instances_cleared: Number of storage instances removed
        - service_instances_cleared: Number of service instances removed
    """
    global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE

    storage_count = len(_STORAGE_CACHE)
    service_count = len(_MEMORY_SERVICE_CACHE)

    _STORAGE_CACHE.clear()
    _MEMORY_SERVICE_CACHE.clear()

    logger.info(
        "🧹 Caches cleared - "
        "Storage: %d instances, "
        "Service: %d instances",
        storage_count,
        service_count,
    )

    return {
        "storage_instances_cleared": storage_count,
        "service_instances_cleared": service_count
    }


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics for the process.

    Returns:
        Dict with memory metrics:
        - rss_mb: Resident Set Size in MB (physical memory)
        - vms_mb: Virtual Memory Size in MB
        - cached_storage_count: Number of cached storage instances
        - cached_service_count: Number of cached service instances
        - cache_stats: Hit/miss statistics
    """
    try:
        import psutil  # inline import
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
    except ImportError:
        # psutil not available, return placeholder values
        rss_mb = -1
        vms_mb = -1
        logger.warning("psutil not available, memory metrics unavailable")

    return {
        "rss_mb": round(rss_mb, 2),
        "vms_mb": round(vms_mb, 2),
        "cached_storage_count": len(_STORAGE_CACHE),
        "cached_service_count": len(_MEMORY_SERVICE_CACHE),
        "cache_stats": {
            "storage_hits": _CACHE_STATS["storage_hits"],
            "storage_misses": _CACHE_STATS["storage_misses"],
            "service_hits": _CACHE_STATS["service_hits"],
            "service_misses": _CACHE_STATS["service_misses"],
            "total_calls": _CACHE_STATS["total_calls"]
        }
    }


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics including hit rates.

    Returns:
        Dict with cache performance metrics
    """
    total_storage_ops = _CACHE_STATS["storage_hits"] + _CACHE_STATS["storage_misses"]
    total_service_ops = _CACHE_STATS["service_hits"] + _CACHE_STATS["service_misses"]

    storage_hit_rate = (
        (_CACHE_STATS["storage_hits"] / total_storage_ops * 100)
        if total_storage_ops > 0 else 0
    )
    service_hit_rate = (
        (_CACHE_STATS["service_hits"] / total_service_ops * 100)
        if total_service_ops > 0 else 0
    )

    return {
        "storage": {
            "hits": _CACHE_STATS["storage_hits"],
            "misses": _CACHE_STATS["storage_misses"],
            "hit_rate_percent": round(storage_hit_rate, 1),
            "cached_count": len(_STORAGE_CACHE)
        },
        "service": {
            "hits": _CACHE_STATS["service_hits"],
            "misses": _CACHE_STATS["service_misses"],
            "hit_rate_percent": round(service_hit_rate, 1),
            "cached_count": len(_MEMORY_SERVICE_CACHE)
        },
        "total_calls": _CACHE_STATS["total_calls"]
    }
