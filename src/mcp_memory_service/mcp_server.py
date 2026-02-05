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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10
import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from fastmcp import Context, FastMCP  # noqa: E402

# Import existing memory service components
from .config import SQLITE_VEC_PATH, STORAGE_BACKEND  # noqa: E402
from .formatters.toon import format_search_results_as_toon  # noqa: E402
from .resources.toon_documentation import TOON_FORMAT_DOCUMENTATION  # noqa: E402
from .services.memory_service import MemoryService  # noqa: E402
from .storage.base import MemoryStorage  # noqa: E402

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

    # Check if shared storage is already initialized (by unified_server)
    from .shared_storage import get_shared_storage, is_storage_initialized

    if is_storage_initialized():
        logger.info("Using pre-initialized shared storage instance")
        storage = await get_shared_storage()
    else:
        # Fallback to creating storage if running standalone
        logger.info("No shared storage found, initializing new instance (standalone mode)")
        from .storage.factory import create_storage_instance

        storage = await create_storage_instance(SQLITE_VEC_PATH)

    # Initialize memory service with shared business logic
    memory_service = MemoryService(storage)

    try:
        yield MCPServerContext(storage=storage, memory_service=memory_service)
    finally:
        # Only close storage if we created it (standalone mode)
        # Shared storage is managed by unified_server
        if not is_storage_initialized():
            logger.info("Shutting down MCP Memory Service components...")
            if hasattr(storage, "close"):
                await storage.close()


# Create FastMCP server instance
mcp = FastMCP("MCP Memory Service", lifespan=mcp_server_lifespan)


# =============================================================================
# RESOURCES
# =============================================================================


@mcp.resource("toon://format/documentation")
def toon_format_docs() -> str:
    """
    Return comprehensive TOON format specification for LLM consumption.

    This resource provides the complete TOON (Terser Object Notation) format
    specification, including structure, field types, parsing strategies, and examples.
    Used by LLMs to understand the compact pipe-delimited format returned by memory tools.
    """
    return TOON_FORMAT_DOCUMENTATION


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
    chunk_hashes: list[str]


class StoreMemoryFailure(TypedDict):
    """Return type for failed memory storage."""

    success: bool
    message: str
    chunks_created: NotRequired[int]
    chunk_hashes: NotRequired[list[str]]


# =============================================================================
# CORE MEMORY OPERATIONS
# =============================================================================


@mcp.tool()
async def store_memory(
    content: str,
    ctx: Context,
    tags: str | list[str] | None = None,
    memory_type: str = "note",
    metadata: dict[str, Any] | None = None,
    client_hostname: str | None = None,
) -> StoreMemorySuccess | StoreMemorySplitSuccess | StoreMemoryFailure:
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
                success=True, message="Memory stored successfully", content_hash=result["memory"]["content_hash"]
            )
        elif "memories" in result:
            # Chunked memory case
            chunk_hashes = [m["content_hash"] for m in result["memories"]]
            return StoreMemorySplitSuccess(
                success=True,
                message=f"Memory stored as {result['total_chunks']} chunks",
                chunks_created=result["total_chunks"],
                chunk_hashes=chunk_hashes,
            )

    # Failure case
    return StoreMemoryFailure(success=False, message=result.get("error", "Unknown error occurred"))


@mcp.tool()
async def retrieve_memory(query: str, ctx: Context, page: int = 1, page_size: int = 10, min_similarity: float = 0.6) -> str:
    """
    Retrieve memories using hybrid search (semantic + tag matching).

    Combines vector similarity with automatic tag extraction for improved retrieval.
    When query terms match existing tags, those memories receive a score boost.
    This solves the "rathole problem" where project-specific queries return
    semantically similar but categorically unrelated results.

    Hybrid search is enabled by default. To opt-out to pure vector search:
    - Set environment variable MCP_MEMORY_HYBRID_ALPHA=1.0

    Args:
        query: Natural language search query (tags extracted automatically)
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        min_similarity: Quality threshold (0.0-1.0, default: 0.6)
            - 0.6-0.7: Good matches (recommended)
            - 0.7-0.9: Very similar matches
            - 0.9+: Nearly identical
            - Lower for exploratory search, higher for precision

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash|similarity_score

        For complete TOON specification, see resource: toon://format/documentation

    Use this for: Finding relevant context, answering questions, discovering related information.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.retrieve_memories(query=query, page=page, page_size=page_size, min_similarity=min_similarity)

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


@mcp.tool()
async def search_by_tag(tags: str | list[str], ctx: Context, match_all: bool = False, page: int = 1, page_size: int = 10) -> str:
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

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash

        For complete TOON specification, see resource: toon://format/documentation

    Examples:
        - ["python", "api"] with match_all=False → memories tagged python OR api
        - ["python", "api"] with match_all=True → memories tagged python AND api
        - "bug-fix" → all memories tagged bug-fix

    Use this for: Tag-based filtering, categorical search, known classification retrieval.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.search_by_tag(tags=tags, match_all=match_all, page=page, page_size=page_size)

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


@mcp.tool()
async def delete_memory(content_hash: str, ctx: Context) -> dict[str, bool | str]:
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
async def check_database_health(ctx: Context) -> dict[str, Any]:
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
    tag: str | None = None,
    memory_type: str | None = None,
) -> str:
    """
    List memories in chronological order with optional filtering.

    Returns memories ordered by creation time (newest first), without semantic
    ranking. Use this for browsing recent memories or getting a chronological view.

    Args:
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        tag: Optional - return only memories with this specific tag
        memory_type: Optional - filter by type (note, decision, task, reference)

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash

        For complete TOON specification, see resource: toon://format/documentation

    Differences from other search tools:
        - retrieve_memory: Semantic similarity ranking (finds meaning)
        - search_by_tag: Exact tag matching with AND/OR logic
        - list_memories: Chronological order (this tool)

    Use this for: Browsing recent activity, reviewing what was stored,
    chronological exploration, getting latest entries.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.list_memories(page=page, page_size=page_size, tag=tag, memory_type=memory_type)

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


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
    transport_mode = os.getenv("MCP_TRANSPORT_MODE", "http")

    if transport_mode == "stdio":
        # Run server with stdio transport
        mcp.run(transport="stdio")
    else:
        # Run server with HTTP transport (FastMCP v2.0 uses 'http' instead of 'streamable-http')
        mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
