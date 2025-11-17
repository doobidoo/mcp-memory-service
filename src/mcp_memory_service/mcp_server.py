#!/usr/bin/env python3
"""
FastAPI MCP Server for Memory Service

This module implements a native MCP server using the FastAPI MCP framework,
replacing the Node.js HTTP-to-MCP bridge to resolve SSL connectivity issues
and provide direct MCP protocol support.

Features:
- Native MCP protocol implementation using FastMCP
- Direct integration with existing memory storage backends
- Streamable HTTP transport for remote access
- All 22 core memory operations (excluding dashboard tools)
- SSL/HTTPS support with proper certificate handling
"""

import asyncio
import logging
import os
import socket
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TypedDict
try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent

# Import existing memory service components
from .config import (
    STORAGE_BACKEND,
    CONSOLIDATION_ENABLED, EMBEDDING_MODEL_NAME, INCLUDE_HOSTNAME,
    SQLITE_VEC_PATH,
    CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX,
    CLOUDFLARE_D1_DATABASE_ID, CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES, CLOUDFLARE_BASE_DELAY,
    HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE, HYBRID_MAX_QUEUE_SIZE,
    HYBRID_SYNC_ON_STARTUP, HYBRID_FALLBACK_TO_PRIMARY,
    CONTENT_PRESERVE_BOUNDARIES, CONTENT_SPLIT_OVERLAP, ENABLE_AUTO_SPLIT
)
from .storage.base import MemoryStorage
from .services.memory_service import MemoryService

# Configure logging
logging.basicConfig(level=logging.INFO)  # Default to INFO level
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL CACHING FOR MCP SERVER PERFORMANCE OPTIMIZATION
# =============================================================================
# Module-level caches to persist storage/service instances across stateless HTTP calls.
# This reduces initialization overhead from ~1,810ms to <400ms on cache hits.
#
# Cache Keys:
# - Storage: "{backend_type}:{db_path}" (e.g., "sqlite_vec:/path/to/db")
# - MemoryService: storage instance ID (id(storage))
#
# Thread Safety:
# - Uses asyncio.Lock to prevent race conditions during concurrent access
#
# Lifecycle:
# - Cached instances persist for the lifetime of the Python process
# - NOT cleared between stateless HTTP calls (intentional for performance)
# - Cleaned up on process shutdown via lifespan context manager

_STORAGE_CACHE: Dict[str, MemoryStorage] = {}
_MEMORY_SERVICE_CACHE: Dict[int, MemoryService] = {}
_CACHE_LOCK: Optional[asyncio.Lock] = None  # Initialized on first use
_CACHE_STATS = {
    "storage_hits": 0,
    "storage_misses": 0,
    "service_hits": 0,
    "service_misses": 0,
    "total_calls": 0,
    "initialization_times": []  # Track initialization durations for cache misses
}

def _get_cache_lock() -> asyncio.Lock:
    """Get or create the global cache lock (lazy initialization to avoid event loop issues)."""
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    return _CACHE_LOCK

@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""
    storage: MemoryStorage
    memory_service: MemoryService

@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """
    Manage MCP server lifecycle with global caching for performance optimization.

    Performance Impact:
    - Cache HIT: ~200-400ms (reuses existing instances)
    - Cache MISS: ~1,810ms (initializes new instances)

    Caching Strategy:
    - Storage instances cached by "{backend}:{path}" key
    - MemoryService instances cached by storage ID
    - Thread-safe with asyncio.Lock
    - Persists across stateless HTTP calls (by design)
    """
    global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE, _CACHE_STATS

    # Track call statistics
    _CACHE_STATS["total_calls"] += 1
    start_time = time.time()

    logger.info(f"🔄 MCP Server Call #{_CACHE_STATS['total_calls']} - Checking global cache...")

    # Acquire lock for thread-safe cache access
    cache_lock = _get_cache_lock()
    async with cache_lock:
        # Generate cache key for storage backend
        cache_key = f"{STORAGE_BACKEND}:{SQLITE_VEC_PATH}"

        # Check storage cache
        if cache_key in _STORAGE_CACHE:
            storage = _STORAGE_CACHE[cache_key]
            _CACHE_STATS["storage_hits"] += 1
            logger.info(f"✅ Storage Cache HIT - Reusing {STORAGE_BACKEND} instance (key: {cache_key})")
        else:
            _CACHE_STATS["storage_misses"] += 1
            logger.info(f"❌ Storage Cache MISS - Initializing {STORAGE_BACKEND} instance...")

            # Initialize storage backend using shared factory
            from .storage.factory import create_storage_instance
            storage = await create_storage_instance(SQLITE_VEC_PATH, server_type="mcp")

            # Cache the storage instance
            _STORAGE_CACHE[cache_key] = storage
            init_time = (time.time() - start_time) * 1000  # Convert to ms
            _CACHE_STATS["initialization_times"].append(init_time)
            logger.info(f"💾 Cached storage instance (key: {cache_key}, init_time: {init_time:.1f}ms)")

        # Check memory service cache
        storage_id = id(storage)
        if storage_id in _MEMORY_SERVICE_CACHE:
            memory_service = _MEMORY_SERVICE_CACHE[storage_id]
            _CACHE_STATS["service_hits"] += 1
            logger.info(f"✅ MemoryService Cache HIT - Reusing service instance (storage_id: {storage_id})")
        else:
            _CACHE_STATS["service_misses"] += 1
            logger.info(f"❌ MemoryService Cache MISS - Creating new service instance...")

            # Initialize memory service with shared business logic
            memory_service = MemoryService(storage)

            # Cache the memory service instance
            _MEMORY_SERVICE_CACHE[storage_id] = memory_service
            logger.info(f"💾 Cached MemoryService instance (storage_id: {storage_id})")

        # Log overall cache performance
        total_time = (time.time() - start_time) * 1000
        cache_hit_rate = (
            (_CACHE_STATS["storage_hits"] + _CACHE_STATS["service_hits"]) /
            (_CACHE_STATS["total_calls"] * 2)  # 2 caches per call
        ) * 100

        logger.info(
            f"📊 Cache Stats - "
            f"Hit Rate: {cache_hit_rate:.1f}% | "
            f"Storage: {_CACHE_STATS['storage_hits']}H/{_CACHE_STATS['storage_misses']}M | "
            f"Service: {_CACHE_STATS['service_hits']}H/{_CACHE_STATS['service_misses']}M | "
            f"Total Time: {total_time:.1f}ms | "
            f"Cache Size: {len(_STORAGE_CACHE)} storage + {len(_MEMORY_SERVICE_CACHE)} services"
        )

    try:
        yield MCPServerContext(
            storage=storage,
            memory_service=memory_service
        )
    finally:
        # IMPORTANT: Do NOT close cached storage instances here!
        # They are intentionally kept alive across stateless HTTP calls for performance.
        # Cleanup only happens on process shutdown (handled by FastMCP framework).
        logger.info(f"✅ MCP Server Call #{_CACHE_STATS['total_calls']} complete - Cached instances preserved")

# Create FastMCP server instance
mcp = FastMCP(
    name="MCP Memory Service", 
    host="0.0.0.0",  # Listen on all interfaces for remote access
    port=8000,       # Default port
    lifespan=mcp_server_lifespan,
    stateless_http=True  # Enable stateless HTTP for Claude Code compatibility
)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class StoreMemorySuccess(TypedDict):
    """Return type for successful single memory storage."""
    success: bool
    message: str
    content_hash: str

class StoreMemorySplitSuccess(TypedDict):
    """Return type for successful chunked memory storage."""
    success: bool
    message: str
    chunks_created: int
    chunk_hashes: List[str]

class StoreMemoryFailure(TypedDict):
    """Return type for failed memory storage."""
    success: bool
    message: str
    chunks_created: NotRequired[int]
    chunk_hashes: NotRequired[List[str]]

# =============================================================================
# CORE MEMORY OPERATIONS
# =============================================================================

@mcp.tool()
async def store_memory(
    content: str,
    ctx: Context,
    tags: Union[str, List[str], None] = None,
    memory_type: str = "note",
    metadata: Optional[Dict[str, Any]] = None,
    client_hostname: Optional[str] = None
) -> Union[StoreMemorySuccess, StoreMemorySplitSuccess, StoreMemoryFailure]:
    """
    Store a new memory with content and optional metadata.

    **IMPORTANT - Content Length Limits:**
    - Cloudflare backend: 800 characters max (BGE model 512 token limit)
    - SQLite-vec backend: No limit (local storage)
    - Hybrid backend: 800 characters max (constrained by Cloudflare sync)

    If content exceeds the backend's limit, it will be automatically split into
    multiple linked memory chunks with preserved context (50-char overlap).
    The splitting respects natural boundaries: paragraphs → sentences → words.

    Args:
        content: The content to store as memory
        tags: Optional tags to categorize the memory (accepts array or comma-separated string)
        memory_type: Type of memory (note, decision, task, reference)
        metadata: Additional metadata for the memory
        client_hostname: Client machine hostname for source tracking

    **Tag Formats - All Formats Supported:**
    Both the tags parameter AND metadata.tags accept ALL formats:
    - ✅ Array format: tags=["tag1", "tag2", "tag3"]
    - ✅ Comma-separated string: tags="tag1,tag2,tag3"
    - ✅ Single string: tags="single-tag"
    - ✅ In metadata: metadata={"tags": "tag1,tag2", "type": "note"}
    - ✅ In metadata (array): metadata={"tags": ["tag1", "tag2"], "type": "note"}

    All formats are automatically normalized internally. If tags are provided in both
    the tags parameter and metadata.tags, they will be merged (duplicates removed).

    Returns:
        Dictionary with:
        - success: Boolean indicating if storage succeeded
        - message: Status message
        - content_hash: Hash of original content (for single memory)
        - chunks_created: Number of chunks (if content was split)
        - chunk_hashes: List of content hashes (if content was split)
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.store_memory(
        content=content,
        tags=tags,
        memory_type=memory_type,
        metadata=metadata,
        client_hostname=client_hostname
    )

    # Transform MemoryService response to MCP tool format
    if not result.get("success"):
        return StoreMemoryFailure(
            success=False,
            message=result.get("error", "Failed to store memory")
        )

    # Handle chunked response (multiple memories)
    if "memories" in result:
        chunk_hashes = [mem["content_hash"] for mem in result["memories"]]
        return StoreMemorySplitSuccess(
            success=True,
            message=f"Successfully stored {len(result['memories'])} memory chunks",
            chunks_created=result["total_chunks"],
            chunk_hashes=chunk_hashes
        )

    # Handle single memory response
    memory_data = result["memory"]
    return StoreMemorySuccess(
        success=True,
        message="Memory stored successfully",
        content_hash=memory_data["content_hash"]
    )

@mcp.tool()
async def retrieve_memory(
    query: str,
    ctx: Context,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Retrieve memories based on semantic similarity to a query.

    Args:
        query: Search query for semantic similarity
        n_results: Maximum number of results to return

    Returns:
        Dictionary with retrieved memories and metadata
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.retrieve_memories(
        query=query,
        n_results=n_results
    )

@mcp.tool()
async def search_by_tag(
    tags: Union[str, List[str]],
    ctx: Context,
    match_all: bool = False
) -> Dict[str, Any]:
    """
    Search memories by tags.

    Args:
        tags: Tag or list of tags to search for
        match_all: If True, memory must have ALL tags; if False, ANY tag

    Returns:
        Dictionary with matching memories
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.search_by_tag(
        tags=tags,
        match_all=match_all
    )

@mcp.tool()
async def delete_memory(
    content_hash: str,
    ctx: Context
) -> Dict[str, Union[bool, str]]:
    """
    Delete a specific memory by its content hash.

    Args:
        content_hash: Hash of the memory content to delete

    Returns:
        Dictionary with success status and message
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.delete_memory(content_hash)

@mcp.tool()
async def check_database_health(ctx: Context) -> Dict[str, Any]:
    """
    Check the health and status of the memory database.

    Returns:
        Dictionary with health status and statistics
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.check_database_health()

@mcp.tool()
async def list_memories(
    ctx: Context,
    page: int = 1,
    page_size: int = 10,
    tag: Optional[str] = None,
    memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    List memories with pagination and optional filtering.

    Args:
        page: Page number (1-based)
        page_size: Number of memories per page
        tag: Filter by specific tag
        memory_type: Filter by memory type

    Returns:
        Dictionary with memories and pagination info
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.list_memories(
        page=page,
        page_size=page_size,
        tag=tag,
        memory_type=memory_type
    )

@mcp.tool()
async def get_cache_stats(ctx: Context) -> Dict[str, Any]:
    """
    Get MCP server global cache statistics for performance monitoring.

    Returns detailed metrics about storage and memory service caching,
    including hit rates, initialization times, and cache sizes.

    This tool is useful for:
    - Monitoring cache effectiveness
    - Debugging performance issues
    - Verifying cache persistence across stateless HTTP calls

    Returns:
        Dictionary with cache statistics:
        - total_calls: Total MCP server invocations
        - hit_rate: Overall cache hit rate percentage
        - storage_cache: Storage cache metrics (hits/misses/size)
        - service_cache: MemoryService cache metrics (hits/misses/size)
        - performance: Initialization time statistics (avg/min/max)
        - backend_info: Current storage backend configuration
    """
    global _CACHE_STATS, _STORAGE_CACHE, _MEMORY_SERVICE_CACHE

    # Import shared stats calculation utility
    from mcp_memory_service.utils.cache_manager import CacheStats, calculate_cache_stats_dict

    # Convert global dict to CacheStats dataclass
    stats = CacheStats(
        total_calls=_CACHE_STATS["total_calls"],
        storage_hits=_CACHE_STATS["storage_hits"],
        storage_misses=_CACHE_STATS["storage_misses"],
        service_hits=_CACHE_STATS["service_hits"],
        service_misses=_CACHE_STATS["service_misses"],
        initialization_times=_CACHE_STATS["initialization_times"]
    )

    # Calculate statistics using shared utility
    cache_sizes = (len(_STORAGE_CACHE), len(_MEMORY_SERVICE_CACHE))
    result = calculate_cache_stats_dict(stats, cache_sizes)

    # Add server-specific details
    result["storage_cache"]["keys"] = list(_STORAGE_CACHE.keys())
    result["backend_info"]["embedding_model"] = EMBEDDING_MODEL_NAME

    return result



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the FastAPI MCP server."""
    # Configure for Claude Code integration
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting MCP Memory Service FastAPI server on {host}:{port}")
    logger.info(f"Storage backend: {STORAGE_BACKEND}")
    
    # Run server with streamable HTTP transport
    mcp.run("streamable-http")

if __name__ == "__main__":
    main()