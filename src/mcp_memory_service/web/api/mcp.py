"""
MCP (Model Context Protocol) endpoints for Claude Code integration.

This module provides MCP protocol endpoints that allow Claude Code clients
to directly access memory operations using the MCP standard.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ..dependencies import get_storage
from ...utils.hashing import generate_content_hash
from ...config import OAUTH_ENABLED

# Import OAuth dependencies only when needed
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None
    require_write_access = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    """MCP protocol request structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP protocol response structure.

    Note: JSON-RPC 2.0 spec requires that successful responses EXCLUDE the 'error'
    field entirely (not include it as null), and error responses EXCLUDE 'result'.
    The exclude_none config ensures proper compliance.
    """
    model_config = ConfigDict(exclude_none=True)

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


# Define MCP tools available
MCP_TOOLS = [
    MCPTool(
        name="store_memory",
        description="Store a new memory with optional tags, metadata, and client information",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for the memory"},
                "memory_type": {"type": "string", "description": "Optional memory type (e.g., 'note', 'reminder', 'fact')"},
                "metadata": {"type": "object", "description": "Additional metadata for the memory"},
                "client_hostname": {"type": "string", "description": "Client machine hostname for source tracking"}
            },
            "required": ["content"]
        }
    ),
    MCPTool(
        name="retrieve_memory", 
        description="Search and retrieve memories using semantic similarity",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for finding relevant memories"},
                "limit": {"type": "integer", "description": "Maximum number of memories to return", "default": 10},
                "similarity_threshold": {"type": "number", "description": "Minimum similarity score threshold (0.0-1.0)", "default": 0.7, "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="recall_memory",
        description="Retrieve memories using natural language time expressions and optional semantic search",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query specifying the time frame or content to recall"},
                "n_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="search_by_tag",
        description="Search memories by specific tags",
        inputSchema={
            "type": "object", 
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to search for"},
                "operation": {"type": "string", "enum": ["AND", "OR"], "description": "Tag search operation", "default": "AND"}
            },
            "required": ["tags"]
        }
    ),
    MCPTool(
        name="delete_memory",
        description="Delete a specific memory by content hash",
        inputSchema={
            "type": "object",
            "properties": {
                "content_hash": {"type": "string", "description": "Hash of the memory to delete"}
            },
            "required": ["content_hash"]
        }
    ),
    MCPTool(
        name="check_database_health",
        description="Check the health and status of the memory database",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    MCPTool(
        name="list_memories",
        description="List memories with pagination and optional filtering",
        inputSchema={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "Page number (1-based)", "default": 1, "minimum": 1},
                "page_size": {"type": "integer", "description": "Number of memories per page", "default": 10, "minimum": 1, "maximum": 100},
                "tag": {"type": "string", "description": "Filter by specific tag"},
                "memory_type": {"type": "string", "description": "Filter by memory type"}
            }
        }
    ),
]


@router.post("/")
@router.post("")
async def mcp_endpoint(
    request: MCPRequest,
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """Main MCP protocol endpoint for processing MCP requests."""
    try:
        storage = get_storage()

        if request.method == "initialize":
            response = MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mcp-memory-service",
                        "version": "4.1.1"
                    }
                }
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        elif request.method == "tools/list":
            response = MCPResponse(
                id=request.id,
                result={
                    "tools": [tool.model_dump() for tool in MCP_TOOLS]
                }
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        elif request.method == "tools/call":
            tool_name = request.params.get("name") if request.params else None
            arguments = request.params.get("arguments", {}) if request.params else {}

            result = await handle_tool_call(storage, tool_name, arguments)

            response = MCPResponse(
                id=request.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result)
                        }
                    ]
                }
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        else:
            response = MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        response = MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))


async def handle_tool_call(storage, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool calls and route to appropriate memory operations."""
    
    if tool_name == "store_memory":
        from mcp_memory_service.models.memory import Memory
        
        content = arguments.get("content")
        tags = arguments.get("tags", [])
        memory_type = arguments.get("memory_type")
        metadata = arguments.get("metadata", {})
        client_hostname = arguments.get("client_hostname")
        
        # Ensure metadata is a dict
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}
        
        # Add client_hostname to metadata if provided
        if client_hostname:
            metadata["client_hostname"] = client_hostname
        
        content_hash = generate_content_hash(content, metadata)
        
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=tags,
            memory_type=memory_type,
            metadata=metadata
        )
        
        success, message = await storage.store(memory)
        
        return {
            "success": success,
            "message": message,
            "content_hash": memory.content_hash if success else None
        }
    
    elif tool_name == "retrieve_memory":
        query = arguments.get("query")
        limit = arguments.get("limit", 10)
        similarity_threshold = arguments.get("similarity_threshold", 0.0)
        
        # Get results from storage (no similarity filtering at storage level)
        results = await storage.retrieve(query=query, n_results=limit)
        
        # Apply similarity threshold filtering (same as API implementation)
        if similarity_threshold is not None:
            results = [
                result for result in results
                if result.relevance_score and result.relevance_score >= similarity_threshold
            ]
        
        return {
            "results": [
                {
                    "content": r.memory.content,
                    "content_hash": r.memory.content_hash,
                    "tags": r.memory.tags,
                    "similarity_score": r.relevance_score,
                    "created_at": r.memory.created_at_iso
                }
                for r in results
            ],
            "total_found": len(results)
        }

    elif tool_name == "recall_memory":
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)

        # Use storage recall_memory method which handles time expressions
        memories = await storage.recall_memory(query=query, n_results=n_results)

        return {
            "results": [
                {
                    "content": m.content,
                    "content_hash": m.content_hash,
                    "tags": m.tags,
                    "created_at": m.created_at_iso
                }
                for m in memories
            ],
            "total_found": len(memories)
        }

    elif tool_name == "search_by_tag":
        tags = arguments.get("tags")
        operation = arguments.get("operation", "AND")
        
        results = await storage.search_by_tags(tags=tags, operation=operation)
        
        return {
            "results": [
                {
                    "content": memory.content,
                    "content_hash": memory.content_hash,
                    "tags": memory.tags,
                    "created_at": memory.created_at_iso
                }
                for memory in results
            ],
            "total_found": len(results)
        }
    
    elif tool_name == "delete_memory":
        content_hash = arguments.get("content_hash")
        
        success, message = await storage.delete(content_hash)
        
        return {
            "success": success,
            "message": message
        }
    
    elif tool_name == "check_database_health":
        stats = await storage.get_stats()

        return {
            "status": "healthy",
            "statistics": stats
        }
    
    elif tool_name == "list_memories":
        page = arguments.get("page", 1)
        page_size = arguments.get("page_size", 10)
        tag = arguments.get("tag")
        memory_type = arguments.get("memory_type")
        
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
        
        return {
            "memories": [
                {
                    "content": memory.content,
                    "content_hash": memory.content_hash,
                    "tags": memory.tags,
                    "memory_type": memory.memory_type,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at_iso,
                    "updated_at": memory.updated_at_iso
                }
                for memory in memories
            ],
            "page": page,
            "page_size": page_size,
            "total_found": len(memories)
        }
    
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@router.get("/tools")
async def list_mcp_tools(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """List available MCP tools for discovery."""
    return {
        "tools": [tool.dict() for tool in MCP_TOOLS],
        "protocol": "mcp",
        "version": "1.0"
    }


@router.get("/health")
async def mcp_health():
    """MCP-specific health check."""
    storage = get_storage()
    stats = await storage.get_stats()

    return {
        "status": "healthy",
        "protocol": "mcp",
        "tools_available": len(MCP_TOOLS),
        "storage_backend": "sqlite-vec",
        "statistics": stats
    }