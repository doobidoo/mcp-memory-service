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
        yield MCPServerContext(
            storage=storage,
            memory_service=memory_service
        )
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down MCP Memory Service components...")
        if hasattr(storage, 'close'):
            await storage.close()

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

    **IMPORTANT - Tag Formats (Both Supported):**
    The tags parameter accepts BOTH formats:
    - ✅ Array format: tags=["tag1", "tag2", "tag3"]
    - ✅ String format: tags="tag1,tag2,tag3"

    Both will be normalized to an array internally.

    **IMPORTANT - Metadata Tag Format:**
    When providing tags in the metadata parameter, they MUST be an array:
    - ✅ CORRECT: metadata={"tags": ["tag1", "tag2"], "type": "note"}
    - ❌ WRONG: metadata={"tags": "tag1,tag2", "type": "note"}

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
    return await memory_service.store_memory(
        content=content,
        tags=tags,
        memory_type=memory_type,
        metadata=metadata,
        client_hostname=client_hostname
    )

@mcp.tool()
async def retrieve_memory(
    query: str,
    ctx: Context,
    n_results: int = 5,
    min_similarity: float = 0.0
) -> Dict[str, Any]:
    """
    Retrieve memories based on semantic similarity to a query.

    Args:
        query: Search query for semantic similarity
        n_results: Maximum number of results to return
        min_similarity: Minimum similarity score threshold

    Returns:
        Dictionary with retrieved memories and metadata
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.retrieve_memory(
        query=query,
        n_results=n_results,
        min_similarity=min_similarity
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