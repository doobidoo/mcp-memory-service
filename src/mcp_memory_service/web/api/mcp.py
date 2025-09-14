"""
MCP (Model Context Protocol) endpoints for Claude Code integration.

This module provides MCP protocol endpoints that allow Claude Code clients
to directly access memory operations using the MCP standard.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..dependencies import get_storage
from ...utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    """MCP protocol request structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP protocol response structure."""
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
    MCPTool(
        name="search_by_time",
        description="Search memories by time-based queries using natural language",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language time query (e.g., 'last week', 'yesterday', 'this month')"},
                "n_results": {"type": "integer", "description": "Maximum number of memories to return", "default": 10, "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="search_similar",
        description="Search for memories similar to a given memory by content hash",
        inputSchema={
            "type": "object",
            "properties": {
                "content_hash": {"type": "string", "description": "Content hash of the memory to find similar ones for"},
                "limit": {"type": "integer", "description": "Maximum number of similar memories to return", "default": 5}
            },
            "required": ["content_hash"]
        }
    )
]


@router.post("/")
@router.post("")
async def mcp_endpoint(request: MCPRequest):
    """Main MCP protocol endpoint for processing MCP requests."""
    try:
        storage = get_storage()
        
        if request.method == "initialize":
            return MCPResponse(
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

        elif request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [tool.dict() for tool in MCP_TOOLS]
                }
            )
        
        elif request.method == "tools/call":
            tool_name = request.params.get("name") if request.params else None
            arguments = request.params.get("arguments", {}) if request.params else {}
            
            result = await handle_tool_call(storage, tool_name, arguments)
            
            return MCPResponse(
                id=request.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            )
        
        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )
            
    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )


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
                import json
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
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        
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
        stats = storage.get_stats()
        
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
        
        # Get memories with pagination
        memories = await storage.get_all_memories(
            limit=page_size,
            offset=offset
        )
        
        # Apply tag and memory_type filtering if specified
        if tag or memory_type:
            filtered_memories = []
            for memory in memories:
                if tag and tag not in memory.tags:
                    continue
                if memory_type and memory.memory_type != memory_type:
                    continue
                filtered_memories.append(memory)
            memories = filtered_memories
        
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
    
    elif tool_name == "search_by_time":
        query = arguments.get("query")
        n_results = arguments.get("n_results", 10)
        
        # Use the same time parsing logic as the API
        from datetime import datetime
        from .search import parse_time_query, is_within_time_range
        try:
            time_filter = parse_time_query(query)
            
            if not time_filter:
                return {
                    "success": False,
                    "message": f"Could not parse time query: '{query}'. Try 'yesterday', 'last week', 'this month', etc."
                }
            
            # Get memories and filter by time (same as API implementation)
            query_results = await storage.retrieve("", n_results=1000)  # Get many results to filter
            
            # Filter by time (exactly like API - convert float to datetime)
            filtered_memories = []
            for result in query_results:
                memory_time = None
                if result.memory.created_at:
                    memory_time = datetime.fromtimestamp(result.memory.created_at)
                
                if memory_time and is_within_time_range(memory_time, time_filter):
                    filtered_memories.append(result)
            
            # Limit results
            results = filtered_memories[:n_results]
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Time search failed: {str(e)}"
            }
        
        return {
            "results": [
                {
                    "content": result.memory.content,
                    "content_hash": result.memory.content_hash,
                    "tags": result.memory.tags,
                    "memory_type": result.memory.memory_type,
                    "created_at": result.memory.created_at_iso,
                    "updated_at": result.memory.updated_at_iso
                }
                for result in results
            ],
            "total_found": len(results)
        }
    
    elif tool_name == "search_similar":
        content_hash = arguments.get("content_hash")
        limit = arguments.get("limit", 5)
        
        # Get the target memory first
        target_memory = await storage.get_by_hash(content_hash)
        if not target_memory:
            return {
                "success": False,
                "message": "Memory not found"
            }
        
        # Find similar memories using the target memory's content
        similar_results = await storage.retrieve(
            query=target_memory.content,
            n_results=limit + 1  # +1 because the original will be included
        )
        
        # Filter out the original memory
        results = [
            r for r in similar_results 
            if r.memory.content_hash != content_hash
        ][:limit]
        
        return {
            "success": True,
            "target_memory": {
                "content": target_memory.content,
                "content_hash": target_memory.content_hash,
                "tags": target_memory.tags
            },
            "similar_memories": [
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
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@router.get("/tools")
async def list_mcp_tools():
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
    stats = storage.get_stats()
    
    return {
        "status": "healthy",
        "protocol": "mcp",
        "tools_available": len(MCP_TOOLS),
        "storage_backend": "sqlite-vec",
        "statistics": stats
    }