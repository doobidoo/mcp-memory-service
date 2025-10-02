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
from typing import Dict, List, Optional, Any, Union
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
    CHROMA_PATH, COLLECTION_METADATA, STORAGE_BACKEND,
    CONSOLIDATION_ENABLED, EMBEDDING_MODEL_NAME, INCLUDE_HOSTNAME,
    SQLITE_VEC_PATH,
    CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX,
    CLOUDFLARE_D1_DATABASE_ID, CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES, CLOUDFLARE_BASE_DELAY,
    HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE, HYBRID_MAX_QUEUE_SIZE,
    HYBRID_SYNC_ON_STARTUP, HYBRID_FALLBACK_TO_PRIMARY
)
from .storage.base import MemoryStorage

from .models.memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)  # Default to INFO level
logger = logging.getLogger(__name__)

@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""
    storage: MemoryStorage

@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """Manage MCP server lifecycle with proper resource initialization and cleanup."""
    logger.info("Initializing MCP Memory Service components...")
    
    # Initialize storage backend using shared factory
    from .storage.factory import create_storage_instance
    storage = await create_storage_instance(SQLITE_VEC_PATH)
    
    try:
        yield MCPServerContext(
            storage=storage
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
# CORE MEMORY OPERATIONS
# =============================================================================

@mcp.tool()
async def store_memory(
    content: str,
    ctx: Context,
    tags: Optional[List[str]] = None,
    memory_type: str = "note",
    metadata: Optional[Dict[str, Any]] = None,
    client_hostname: Optional[str] = None
) -> Dict[str, Union[bool, str]]:
    """
    Store a new memory with content and optional metadata.
    
    Args:
        content: The content to store as memory
        tags: Optional tags to categorize the memory
        memory_type: Type of memory (note, decision, task, reference)
        metadata: Additional metadata for the memory
        client_hostname: Client machine hostname for source tracking
    
    Returns:
        Dictionary with success status and message
    """
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Prepare tags and metadata with optional hostname
        final_tags = tags or []
        final_metadata = metadata or {}
        
        if INCLUDE_HOSTNAME:
            # Prioritize client-provided hostname, then fallback to server
            if client_hostname:
                hostname = client_hostname
            else:
                hostname = socket.gethostname()
                
            source_tag = f"source:{hostname}"
            if source_tag not in final_tags:
                final_tags.append(source_tag)
            final_metadata["hostname"] = hostname
        
        # Create memory object
        memory = Memory(
            content=content,
            tags=final_tags,
            memory_type=memory_type,
            metadata=final_metadata
        )
        
        # Store memory
        success, message = await storage.store(memory)
        
        return {
            "success": success,
            "message": message,
            "content_hash": memory.content_hash
        }
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return {
            "success": False,
            "message": f"Failed to store memory: {str(e)}"
        }

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
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Search for memories
        results = await storage.search(
            query=query,
            n_results=n_results,
            min_similarity=min_similarity
        )
        
        # Format results
        memories = []
        for result in results:
            memories.append({
                "content": result.memory.content,
                "content_hash": result.memory.content_hash,
                "tags": result.memory.metadata.tags,
                "memory_type": result.memory.metadata.memory_type,
                "created_at": result.memory.metadata.created_at_iso,
                "similarity_score": result.similarity_score
            })
        
        return {
            "memories": memories,
            "query": query,
            "total_results": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return {
            "memories": [],
            "query": query,
            "error": f"Failed to retrieve memories: {str(e)}"
        }

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
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Normalize tags to list
        if isinstance(tags, str):
            tags = [tags]
        
        # Search by tags
        memories = await storage.search_by_tags(
            tags=tags,
            match_all=match_all
        )
        
        # Format results
        results = []
        for memory in memories:
            results.append({
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.metadata.tags,
                "memory_type": memory.metadata.memory_type,
                "created_at": memory.metadata.created_at_iso
            })
        
        return {
            "memories": results,
            "search_tags": tags,
            "match_all": match_all,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching by tags: {e}")
        return {
            "memories": [],
            "search_tags": tags,
            "error": f"Failed to search by tags: {str(e)}"
        }

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
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Delete memory
        success, message = await storage.delete(content_hash)
        
        return {
            "success": success,
            "message": message,
            "content_hash": content_hash
        }
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return {
            "success": False,
            "message": f"Failed to delete memory: {str(e)}",
            "content_hash": content_hash
        }

@mcp.tool()
async def check_database_health(ctx: Context) -> Dict[str, Any]:
    """
    Check the health and status of the memory database.
    
    Returns:
        Dictionary with health status and statistics
    """
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Get health status and statistics
        stats = await storage.get_stats()
        
        return {
            "status": "healthy",
            "backend": storage.__class__.__name__,
            "statistics": {
                "total_memories": stats.get("total_memories", 0),
                "total_tags": stats.get("total_tags", 0),
                "storage_size": stats.get("storage_size", "unknown"),
                "last_backup": stats.get("last_backup", "never")
            },
            "timestamp": stats.get("timestamp", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error checking database health: {e}")
        return {
            "status": "error",
            "backend": "unknown",
            "error": f"Health check failed: {str(e)}"
        }

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
    try:
        storage = ctx.request_context.lifespan_context.storage
        
        # Calculate offset
        offset = (page - 1) * page_size

        # Use database-level filtering for better performance
        tags_list = [tag] if tag else None
        memories = await storage.get_all_memories(
            limit=page_size,
            offset=offset,
            memory_type=memory_type,
            tags=tags_list
        )
        
        # Format results
        results = []
        for memory in memories:
            results.append({
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.tags,
                "memory_type": memory.memory_type,
                "metadata": memory.metadata,
                "created_at": memory.created_at_iso,
                "updated_at": memory.updated_at_iso
            })
        
        return {
            "memories": results,
            "page": page,
            "page_size": page_size,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        return {
            "memories": [],
            "page": page,
            "page_size": page_size,
            "error": f"Failed to list memories: {str(e)}"
        }



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
    logger.info(f"Data path: {CHROMA_PATH}")
    
    # Run server with streamable HTTP transport
    mcp.run("streamable-http")

if __name__ == "__main__":
    main()