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

# FastMCP is not available in current MCP library version
# This module is kept for future compatibility
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("FastMCP not available in mcp library - mcp_server module cannot be used")
    
    # Create dummy objects for graceful degradation
    class _DummyFastMCP:
        def tool(self):
            """Dummy decorator that does nothing."""
            def decorator(func):
                return func
            return decorator
    
    FastMCP = _DummyFastMCP  # type: ignore
    Context = None  # type: ignore

from mcp.types import TextContent, ToolAnnotations

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

def _get_or_create_memory_service(storage: MemoryStorage) -> MemoryService:
    """
    Get cached MemoryService or create new one.

    Args:
        storage: Storage instance to use as cache key

    Returns:
        MemoryService instance (cached or newly created)
    """
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
        f"📊 Cache Stats - "
        f"Hit Rate: {cache_hit_rate:.1f}% | "
        f"Storage: {_CACHE_STATS['storage_hits']}H/{_CACHE_STATS['storage_misses']}M | "
        f"Service: {_CACHE_STATS['service_hits']}H/{_CACHE_STATS['service_misses']}M | "
        f"Total Time: {total_time:.1f}ms | "
        f"Cache Size: {len(_STORAGE_CACHE)} storage + {len(_MEMORY_SERVICE_CACHE)} services"
    )

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

        # Check memory service cache and log performance
        memory_service = _get_or_create_memory_service(storage)
        _log_cache_performance(start_time)

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
try:
    mcp = FastMCP(
        name="MCP Memory Service", 
        host="0.0.0.0",  # Listen on all interfaces for remote access
        port=8000,       # Default port
        lifespan=mcp_server_lifespan,
        stateless_http=True  # Enable stateless HTTP for Claude Code compatibility
    )
except TypeError:
    # FastMCP not available - create dummy instance
    mcp = _DummyFastMCP()  # type: ignore

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

@mcp.tool(
    annotations=ToolAnnotations(
        title="Store Memory",
        destructiveHint=False,
    ),
)
async def store_memory(
    content: str,
    ctx: Context,
    tags: Union[str, List[str], None] = None,
    memory_type: str = "note",
    metadata: Optional[Dict[str, Any]] = None,
    client_hostname: Optional[str] = None
) -> Union[StoreMemorySuccess, StoreMemorySplitSuccess, StoreMemoryFailure]:
    """Store new information in persistent memory with semantic search capabilities and optional categorization.

USE THIS WHEN:
- User provides information to remember for future sessions (decisions, preferences, facts, code snippets)
- Capturing important context from current conversation ("remember this for later")
- User explicitly says "remember", "save", "store", "keep this", "note that"
- Documenting technical decisions, API patterns, project architecture, user preferences
- Creating knowledge base entries, documentation snippets, troubleshooting notes

THIS IS THE PRIMARY STORAGE TOOL - use it whenever information should persist beyond the current session.

DO NOT USE FOR:
- Temporary conversation context (use native conversation history instead)
- Information already stored (check first with retrieve_memory to avoid duplicates)
- Streaming or real-time data that changes frequently

CONTENT LENGTH LIMITS:
- Cloudflare/Hybrid backends: 800 characters max (auto-splits into chunks if exceeded)
- SQLite-vec backend: No limit
- Auto-chunking preserves context with 50-character overlap at natural boundaries

TAG FORMATS (all supported):
- Array: ["tag1", "tag2"]
- String: "tag1,tag2"
- Single: "single-tag"
- Both tags parameter AND metadata.tags are merged automatically

RETURNS:
- success: Boolean indicating storage status
- message: Status message
- content_hash: Unique identifier for retrieval/deletion (single memory)
- chunks_created: Number of chunks (if content was split)
- chunk_hashes: Array of hashes (if content was split)

Examples:
{
    "content": "User prefers async/await over callbacks in Python projects",
    "metadata": {
        "tags": ["coding-style", "python", "preferences"],
        "type": "preference"
    }
}

{
    "content": "API endpoint /api/v1/users requires JWT token in Authorization header",
    "metadata": {
        "tags": "api-documentation,authentication",
        "type": "reference"
    }
}
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

@mcp.tool(
    annotations=ToolAnnotations(
        title="Retrieve Memory",
        readOnlyHint=True,
    ),
)
async def retrieve_memory(
    query: str,
    ctx: Context,
    n_results: int = 5
) -> Dict[str, Any]:
    """Search stored memories using semantic similarity - finds conceptually related content even if exact words differ.

USE THIS WHEN:
- User asks "what do you remember about X", "do we have info on Y", "recall Z"
- Looking for past decisions, preferences, or context from previous sessions
- Need to retrieve related information without exact wording (semantic search)
- General memory lookup where time frame is NOT specified
- User references "last time we discussed", "you should know", "I told you before"

THIS IS THE PRIMARY SEARCH TOOL - use it for most memory lookups.

DO NOT USE FOR:
- Time-based queries ("yesterday", "last week") - use recall_memory instead
- Exact content matching - use exact_match_retrieve instead
- Tag-based filtering - use search_by_tag instead
- Browsing all memories - use list_memories instead (if available in mcp_server.py)

HOW IT WORKS:
- Converts query to vector embedding using the same model as stored memories
- Finds top N most similar memories using cosine similarity
- Returns ranked by relevance score (0.0-1.0, higher is more similar)
- Works across sessions - retrieves memories from any time period

RETURNS:
- Array of matching memories with:
  - content: The stored text
  - content_hash: Unique identifier
  - similarity_score: Relevance score (0.0-1.0)
  - metadata: Tags, type, timestamp, etc.
  - created_at: When memory was stored

Examples:
{
    "query": "python async patterns we discussed",
    "n_results": 5
}

{
    "query": "database connection settings",
    "n_results": 10
}

{
    "query": "user authentication workflow preferences",
    "n_results": 3
}
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.retrieve_memories(
        query=query,
        n_results=n_results
    )

@mcp.tool(
    annotations=ToolAnnotations(
        title="Search by Tag",
        readOnlyHint=True,
    ),
)
async def search_by_tag(
    tags: Union[str, List[str]],
    ctx: Context,
    match_all: bool = False
) -> Dict[str, Any]:
    """Search memories by exact tag matching - retrieves all memories categorized with specific tags (OR logic by default).

USE THIS WHEN:
- User asks to filter by category ("show me all 'api-docs' memories", "find 'important' notes")
- Need to retrieve memories of a specific type without semantic search
- User wants to browse a category ("what do we have tagged 'python'")
- Looking for all memories with a particular classification
- User says "show me everything about X" where X is a known tag

DO NOT USE FOR:
- Semantic search - use retrieve_memory instead
- Time-based queries - use recall_memory instead
- Finding specific content - use exact_match_retrieve instead

HOW IT WORKS:
- Exact string matching on memory tags (case-sensitive)
- Returns memories matching ANY of the specified tags (OR logic)
- No semantic search - purely categorical filtering
- No similarity scoring - all results are equally relevant

TAG FORMATS (all supported):
- Array: ["tag1", "tag2"]
- String: "tag1,tag2"

RETURNS:
- Array of all memories with matching tags:
  - content: The stored text
  - tags: Array of tags (will include at least one from search)
  - content_hash: Unique identifier
  - metadata: Additional memory metadata
  - No similarity score (categorical match, not semantic)

Examples:
{
    "tags": ["important", "reference"]
}

{
    "tags": "python,async,best-practices"
}

{
    "tags": ["api-documentation"]
}
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.search_by_tag(
        tags=tags,
        match_all=match_all
    )

@mcp.tool(
    annotations=ToolAnnotations(
        title="Delete Memory",
        destructiveHint=True,
    ),
)
async def delete_memory(
    content_hash: str,
    ctx: Context
) -> Dict[str, Union[bool, str]]:
    """Delete a specific memory by its unique content hash identifier - permanent removal of a single memory entry.

USE THIS WHEN:
- User explicitly requests deletion of a specific memory ("delete that", "remove the memory about X")
- After showing user a memory and they want it removed
- Correcting mistakenly stored information
- User says "forget about X", "delete the note about Y", "remove that memory"
- Have the content_hash from a previous retrieve/search operation

DO NOT USE FOR:
- Deleting multiple memories - use delete_by_tag, delete_by_tags, or delete_by_all_tags instead
- Deleting by content without hash - search first with retrieve_memory to get the hash
- Bulk cleanup - use cleanup_duplicates or delete_by_tag instead
- Time-based deletion - use delete_by_timeframe or delete_before_date instead

IMPORTANT:
- This is a PERMANENT operation - memory cannot be recovered after deletion
- You must have the exact content_hash (obtained from search/retrieve operations)
- Only deletes the single memory matching the hash

HOW TO GET content_hash:
1. First search for the memory using retrieve_memory, recall_memory, or search_by_tag
2. Memory results include "content_hash" field
3. Use that hash in this delete operation

RETURNS:
- success: Boolean indicating if deletion succeeded
- content_hash: The hash of the deleted memory
- error: Error message (only present if success is False)

Examples:
# Step 1: Find the memory
retrieve_memory(query: "outdated API documentation")
# Returns: [{content_hash: "a1b2c3d4e5f6...", content: "...", ...}]

# Step 2: Delete it
{
    "content_hash": "a1b2c3d4e5f6..."
}
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.delete_memory(content_hash)

@mcp.tool(
    annotations=ToolAnnotations(
        title="Check Database Health",
        readOnlyHint=True,
    ),
)
async def check_database_health(ctx: Context) -> Dict[str, Any]:
    """Check database health, storage backend status, and retrieve comprehensive memory service statistics.

USE THIS WHEN:
- User asks "how many memories are stored", "is the database working", "memory service status"
- Diagnosing performance issues or connection problems
- User wants to know storage backend configuration (SQLite/Cloudflare/Hybrid)
- Checking if memory service is functioning correctly
- Need to verify successful initialization or troubleshoot errors
- User asks "what storage backend are we using"

DO NOT USE FOR:
- Searching or retrieving specific memories - use retrieve_memory instead
- Getting cache performance stats - use get_cache_stats instead (if available)
- Listing actual memory content - this only returns counts and status

WHAT IT CHECKS:
- Database connectivity and responsiveness
- Storage backend type (sqlite_vec, cloudflare, hybrid)
- Total memory count in database
- Database file size and location (for SQLite backends)
- Sync status (for hybrid backend)
- Configuration details (embedding model, index names, etc.)

RETURNS:
- status: "healthy" or error status
- backend: Storage backend type (sqlite_vec/cloudflare/hybrid)
- total_memories: Count of stored memories
- database_info: Path, size, configuration details
- timestamp: When health check was performed
- Any error messages or warnings

Examples:
No parameters required - just call it:
{}

Common use cases:
- User: "How many memories do I have?" → check_database_health()
- User: "Is the memory service working?" → check_database_health()
- User: "What backend are we using?" → check_database_health()
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.health_check()

@mcp.tool(
    annotations=ToolAnnotations(
        title="List Memories",
        readOnlyHint=True,
    ),
)
async def list_memories(
    ctx: Context,
    page: int = 1,
    page_size: int = 10,
    tag: Optional[str] = None,
    memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """List stored memories with pagination and optional filtering - browse all memories in pages rather than searching.

USE THIS WHEN:
- User wants to browse/explore all memories ("show me my memories", "list everything")
- Need to paginate through large result sets
- Filtering by tag OR memory type for categorical browsing
- User asks "what do I have stored", "show me all notes", "browse my memories"
- Want to see memories without searching for specific content

DO NOT USE FOR:
- Searching for specific content - use retrieve_memory instead
- Time-based queries - use recall_memory instead
- Finding exact text - use exact_match_retrieve instead

HOW IT WORKS:
- Returns memories in pages (default 10 per page)
- Optional filtering by single tag or memory type
- Sorted by creation time (newest first)
- Supports pagination through large datasets

PAGINATION:
- page: 1-based page number (default 1)
- page_size: Number of results per page (default 10, max usually 100)
- Returns total count and page info for navigation

RETURNS:
- memories: Array of memory objects for current page
- total: Total count of matching memories
- page: Current page number
- page_size: Results per page
- total_pages: Total pages available

Examples:
{
    "page": 1,
    "page_size": 10
}

{
    "page": 2,
    "page_size": 20,
    "tag": "python"
}

{
    "page": 1,
    "page_size": 50,
    "memory_type": "decision"
}
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.list_memories(
        page=page,
        page_size=page_size,
        tag=tag,
        memory_type=memory_type
    )

@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Cache Stats",
        readOnlyHint=True,
    ),
)
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