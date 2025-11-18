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
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, TypedDict

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10
import os
import sys
import socket
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent

# Import existing memory service components
from .config import (
    STORAGE_BACKEND,
    CONSOLIDATION_ENABLED,
    EMBEDDING_MODEL_NAME,
    INCLUDE_HOSTNAME,
    SQLITE_VEC_PATH,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_VECTORIZE_INDEX,
    CLOUDFLARE_D1_DATABASE_ID,
    CLOUDFLARE_R2_BUCKET,
    CLOUDFLARE_EMBEDDING_MODEL,
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
    CLOUDFLARE_MAX_RETRIES,
    CLOUDFLARE_BASE_DELAY,
    HYBRID_SYNC_INTERVAL,
    HYBRID_BATCH_SIZE,
    HYBRID_MAX_QUEUE_SIZE,
    HYBRID_SYNC_ON_STARTUP,
    HYBRID_FALLBACK_TO_PRIMARY,
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT,
)
from .storage.base import MemoryStorage
from .services.memory_service import MemoryService

# Configure logging
logging.basicConfig(level=logging.INFO)  # Default to INFO level
logger = logging.getLogger(__name__)


@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""

    storage: MemoryStorage
    memory_service: MemoryService


@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """Manage MCP server lifecycle with proper resource initialization and cleanup."""
    logger.info("Initializing MCP Memory Service components...")

    # Initialize storage backend using shared factory
    from .storage.factory import create_storage_instance

    storage = await create_storage_instance(SQLITE_VEC_PATH)

    # Initialize memory service with shared business logic
    memory_service = MemoryService(storage)

    try:
        yield MCPServerContext(storage=storage, memory_service=memory_service)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down MCP Memory Service components...")
        if hasattr(storage, "close"):
            await storage.close()


# Create FastMCP server instance
mcp = FastMCP(
    name="MCP Memory Service",
    host="0.0.0.0",  # Listen on all interfaces for remote access
    port=8000,  # Default port
    lifespan=mcp_server_lifespan,
    stateless_http=True,  # Enable stateless HTTP for Claude Code compatibility
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
    client_hostname: Optional[str] = None,
) -> Union[StoreMemorySuccess, StoreMemorySplitSuccess, StoreMemoryFailure]:
    """
    Store a new memory for future semantic retrieval.

    Persists content with optional categorization (tags, type, metadata) for later
    retrieval via semantic search or tag filtering. Content is automatically vectorized
    for similarity matching.

    Args:
        content: The text content to store (will be embedded for semantic search)
        tags: Categorization labels (accepts ["tag1", "tag2"] or "tag1,tag2")
        memory_type: Classification - "note", "decision", "task", or "reference"
        metadata: Additional structured data to attach
        client_hostname: Source machine identifier (optional)

    Content Length Handling:
        - SQLite-vec/Qdrant: No limit
        - Cloudflare/Hybrid: 800 chars max (auto-splits if exceeded)
        - Auto-splitting preserves context with 50-char overlap
        - Respects natural boundaries: paragraphs → sentences → words

    Tag Formats (both supported):
        - Array: tags=["python", "bug-fix", "urgent"]
        - String: tags="python,bug-fix,urgent"

    Returns:
        Single memory:
            - success: True/False
            - message: Status description
            - content_hash: Unique identifier for retrieval/deletion

        Split memory (>800 chars on Cloudflare/Hybrid):
            - success: True/False
            - message: Status description
            - chunks_created: Number of linked chunks
            - chunk_hashes: List of content hashes

    Use this for: Capturing information for later retrieval, building knowledge base,
    recording decisions, storing context across conversations.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.store_memory(
        content=content,
        tags=tags,
        memory_type=memory_type,
        metadata=metadata,
        client_hostname=client_hostname,
    )

    # Transform MemoryService response to MCP schema
    if result["success"]:
        if "memory" in result:
            # Single memory case
            return StoreMemorySuccess(
                success=True,
                message="Memory stored successfully",
                content_hash=result["memory"]["content_hash"]
            )
        elif "memories" in result:
            # Chunked memory case
            chunk_hashes = [m["content_hash"] for m in result["memories"]]
            return StoreMemorySplitSuccess(
                success=True,
                message=f"Memory stored as {result['total_chunks']} chunks",
                chunks_created=result["total_chunks"],
                chunk_hashes=chunk_hashes
            )

    # Failure case
    return StoreMemoryFailure(
        success=False,
        message=result.get("error", "Unknown error occurred")
    )


@mcp.tool()
async def retrieve_memory(
    query: str,
    ctx: Context,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float = 0.6
) -> Dict[str, Any]:
    """
    Retrieve memories using semantic similarity search with automatic quality filtering.

    Uses vector embeddings to find memories with similar meaning to the query.
    Default min_similarity of 0.6 filters out low-quality matches.

    Args:
        query: Natural language search query
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        min_similarity: Quality threshold (0.0-1.0, default: 0.6)
            - 0.6-0.7: Good matches (recommended)
            - 0.7-0.9: Very similar matches
            - 0.9+: Nearly identical
            - Lower for exploratory search, higher for precision

    Returns:
        Dictionary with memories and pagination metadata:
        - memories: List of matching memory objects
        - total: Total matching memories across all pages
        - page/page_size: Current pagination state
        - has_more: Whether additional pages exist
        - total_pages: Total pages available

    Use this for: Finding relevant context, answering questions, discovering related information.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.retrieve_memories(
        query=query, page=page, page_size=page_size, min_similarity=min_similarity
    )


@mcp.tool()
async def search_by_tag(
    tags: Union[str, List[str]],
    ctx: Context,
    match_all: bool = False,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Search memories by exact tag matches with flexible filtering.

    Finds memories tagged with specific labels. Use for categorical retrieval
    when you know the exact tags (e.g., "python", "bug-fix", "customer-support").

    Args:
        tags: Single tag string or list of tags to search for
        match_all: Matching mode (default: False = ANY)
            - False (ANY): Returns memories with at least one matching tag
              Use for: Broad category search, exploration
            - True (ALL): Returns only memories with every specified tag
              Use for: Precise filtering, intersection of categories
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)

    Returns:
        Dictionary with:
        - memories: List of tagged memory objects
        - tags: Tags that were searched
        - match_type: "ANY" or "ALL"
        - total/page/page_size/has_more/total_pages: Pagination metadata

    Examples:
        - ["python", "api"] with match_all=False → memories tagged python OR api
        - ["python", "api"] with match_all=True → memories tagged python AND api
        - "bug-fix" → all memories tagged bug-fix

    Use this for: Tag-based filtering, categorical search, known classification retrieval.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.search_by_tag(
        tags=tags, match_all=match_all, page=page, page_size=page_size
    )


@mcp.tool()
async def delete_memory(content_hash: str, ctx: Context) -> Dict[str, Union[bool, str]]:
    """
    Permanently delete a specific memory by its unique identifier.

    Removes a memory from the database. This operation is irreversible.
    The content_hash is returned when storing memories or can be found in
    search/retrieve results.

    Args:
        content_hash: Unique identifier returned from store_memory or found in search results

    Returns:
        Dictionary with:
        - success: True if deleted, False if not found or error
        - message: Confirmation or error description

    Use this for: Removing outdated information, cleaning up test data, deleting
    sensitive content, managing storage space.

    Warning: Deletion is permanent. Verify the content_hash before deleting.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.delete_memory(content_hash)


@mcp.tool()
async def check_database_health(ctx: Context) -> Dict[str, Any]:
    """
    Check memory database health and get storage statistics.

    Verifies database connectivity and returns operational metrics including
    total memory count, storage backend type, and system status.

    Returns:
        Dictionary with:
        - status: "healthy" or error state
        - backend: Storage backend in use (sqlite_vec, cloudflare, hybrid, qdrant)
        - total_memories: Total count of stored memories
        - storage_info: Backend-specific statistics
        - version: Service version

    Use this for: Debugging connection issues, monitoring storage usage,
    verifying service status, troubleshooting performance problems.
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
    memory_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List memories in chronological order with optional filtering.

    Returns memories ordered by creation time (newest first), without semantic
    ranking. Use this for browsing recent memories or getting a chronological view.
    For finding relevant content, use retrieve_memory (semantic) or search_by_tag instead.

    Args:
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        tag: Optional - return only memories with this specific tag
        memory_type: Optional - filter by type (note, decision, task, reference)

    Returns:
        Dictionary with:
        - memories: List ordered by creation time (newest first)
        - total/page/page_size/has_more/total_pages: Pagination metadata

    Differences from other search tools:
        - retrieve_memory: Semantic similarity ranking (finds meaning)
        - search_by_tag: Exact tag matching with AND/OR logic
        - list_memories: Chronological order (this tool)

    Use this for: Browsing recent activity, reviewing what was stored,
    chronological exploration, getting latest entries.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.list_memories(
        page=page, page_size=page_size, tag=tag, memory_type=memory_type
    )


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

    # Check transport mode from environment
    transport_mode = os.getenv("MCP_TRANSPORT_MODE", "streamable-http")

    if transport_mode == "stdio":
        # Run server with stdio transport
        mcp.run("stdio")
    else:
        # Run server with streamable HTTP transport (default)
        mcp.run("streamable-http")


if __name__ == "__main__":
    main()
