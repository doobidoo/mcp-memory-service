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
Memory CRUD endpoints for the HTTP interface.
"""

import logging
import socket
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...models.memory import Memory
from ...services.memory_service import MemoryService
from ...config import INCLUDE_HOSTNAME
# OAuth config no longer needed - auth is always enabled
from ..dependencies import get_storage, get_memory_service
from ..sse import sse_manager, create_memory_stored_event, create_memory_deleted_event

# OAuth authentication imports
from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory."""
    content: str = Field(..., description="The memory content to store")
    tags: List[str] = Field(default=[], description="Tags to categorize the memory")
    memory_type: Optional[str] = Field(None, description="Type of memory. Validated against the built-in ontology (base types: observation, decision, learning, error, pattern, planning, ceremony, milestone, stakeholder, meeting, research, communication). Unknown types are coerced to 'observation' and a warning is included in the response. Register additional types via MCP_CUSTOM_MEMORY_TYPES env var.")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the memory")
    client_hostname: Optional[str] = Field(None, description="Client machine hostname for source tracking")
    conversation_id: Optional[str] = Field(None, description="Optional conversation identifier. When provided, semantic deduplication is skipped, allowing multiple incremental memories from the same conversation to be stored even if their content is topically similar.")


class MemoryUpdateRequest(BaseModel):
    """Request model for updating memory metadata (tags, type, metadata only)."""
    tags: Optional[List[str]] = Field(None, description="Updated tags to categorize the memory")
    memory_type: Optional[str] = Field(None, description="Updated memory type (e.g., 'note', 'reminder', 'fact')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata for the memory")


class MemoryResponse(BaseModel):
    """Response model for memory data."""
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    metadata: Dict[str, Any]
    created_at: Optional[float]
    created_at_iso: Optional[str]
    updated_at: Optional[float]  
    updated_at_iso: Optional[str]


class MemoryListResponse(BaseModel):
    """Response model for paginated memory list."""
    memories: List[MemoryResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class MemoryCreateResponse(BaseModel):
    """Response model for memory creation."""
    success: bool
    message: str
    content_hash: Optional[str] = None
    memory: Optional[MemoryResponse] = None


class MemoryDeleteResponse(BaseModel):
    """Response model for memory deletion."""
    success: bool
    message: str
    content_hash: str


class MemoryUpdateResponse(BaseModel):
    """Response model for memory update."""
    success: bool
    message: str
    content_hash: str
    memory: Optional[MemoryResponse] = None


class TagResponse(BaseModel):
    """Response model for a single tag with its count."""
    tag: str
    count: int


class TagListResponse(BaseModel):
    """Response model for tags list."""
    tags: List[TagResponse]


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Convert Memory model to response format."""
    return MemoryResponse(
        content=memory.content,
        content_hash=memory.content_hash,
        tags=memory.tags,
        memory_type=memory.memory_type,
        metadata=memory.metadata,
        created_at=memory.created_at,
        created_at_iso=memory.created_at_iso,
        updated_at=memory.updated_at,
        updated_at_iso=memory.updated_at_iso
    )


@router.post("/memories", response_model=MemoryCreateResponse, tags=["memories"])
async def store_memory(
    request: MemoryCreateRequest,
    http_request: Request,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Store a new memory.

    Uses the MemoryService for consistent business logic including content processing,
    hostname tagging, and metadata enrichment.
    """
    try:
        # Resolve hostname for consistent tagging (logic stays in API layer, tagging in service)
        client_hostname = None
        if INCLUDE_HOSTNAME:
            # Prioritize client-provided hostname, then header, then fallback to server
            # 1. Check if client provided hostname in request body
            if request.client_hostname:
                client_hostname = request.client_hostname
            # 2. Check for X-Client-Hostname header
            elif http_request.headers.get('X-Client-Hostname'):
                client_hostname = http_request.headers.get('X-Client-Hostname')
            # 3. Fallback to server hostname (original behavior)
            else:
                client_hostname = socket.gethostname()

        # Auto-tag with agent identity if provided via header
        tags = list(request.tags)
        agent_id = http_request.headers.get('X-Agent-ID')
        if agent_id:
            agent_tag = f"agent:{agent_id}"
            if agent_tag not in tags:
                tags.append(agent_tag)

        # Use injected MemoryService for consistent business logic (hostname tagging handled internally)
        result = await memory_service.store_memory(
            content=request.content,
            tags=tags,
            memory_type=request.memory_type,
            metadata=request.metadata,
            client_hostname=client_hostname,
            conversation_id=request.conversation_id,
        )

        if result["success"]:
            # Broadcast SSE event for successful memory storage
            try:
                # Handle both single memory and chunked responses
                if "memory" in result:
                    memory_data = {
                        "content_hash": result["memory"]["content_hash"],
                        "content": result["memory"]["content"],
                        "tags": result["memory"]["tags"],
                        "memory_type": result["memory"]["memory_type"]
                    }
                else:
                    # For chunked responses, use the first chunk's data
                    first_memory = result["memories"][0]
                    memory_data = {
                        "content_hash": first_memory["content_hash"],
                        "content": first_memory["content"],
                        "tags": first_memory["tags"],
                        "memory_type": first_memory["memory_type"]
                    }

                event = create_memory_stored_event(memory_data)
                await sse_manager.broadcast_event(event)
            except Exception as e:
                # Don't fail the request if SSE broadcasting fails
                logger.warning(f"Failed to broadcast memory_stored event: {e}")

            # Surface ontology coercion: when the requested memory_type is not
            # in the ontology, Memory.__post_init__ silently rewrites it to
            # "observation". Without an explicit warning the caller sees
            # "success" while their type filter silently breaks.
            def _coercion_warning(effective: Optional[str]) -> str:
                requested = request.memory_type
                # Use `is not None` (not truthy) so an explicit empty string —
                # also coerced to 'observation' — still surfaces the warning.
                # Mirrors the MCP handler's presence-based check.
                if requested is not None and effective and requested != effective:
                    return (
                        f" Warning: requested memory_type '{requested}' is not in "
                        f"the ontology — stored as '{effective}'. Register custom "
                        f"types via MCP_CUSTOM_MEMORY_TYPES, "
                        f"e.g. '{{\"{requested}\": []}}'."
                    )
                return ""

            # Return appropriate response based on MemoryService result
            if "memory" in result:
                # Single memory response
                base_msg = "Memory stored successfully"
                return MemoryCreateResponse(
                    success=True,
                    message=base_msg + _coercion_warning(result["memory"].get("memory_type")),
                    content_hash=result["memory"]["content_hash"],
                    memory=result["memory"]
                )
            else:
                # Chunked memory response
                first_memory = result["memories"][0]
                base_msg = f"Memory stored as {result['total_chunks']} chunks"
                return MemoryCreateResponse(
                    success=True,
                    message=base_msg + _coercion_warning(first_memory.get("memory_type")),
                    content_hash=first_memory["content_hash"],
                    memory=first_memory
                )
        else:
            return MemoryCreateResponse(
                success=False,
                message=result.get("error", "Failed to store memory"),
                content_hash=None
            )
            
    except Exception as e:
        logger.error(f"Failed to store memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store memory. Please try again.")


@router.get("/memories", response_model=MemoryListResponse, tags=["memories"])
async def list_memories(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of memories per page"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    List memories with pagination and optional filtering.

    Uses the MemoryService for consistent business logic and optimal database-level filtering.
    """
    try:
        # Use the injected service for consistent, performant memory listing
        result = await memory_service.list_memories(
            page=page,
            page_size=page_size,
            tag=tag,
            memory_type=memory_type
        )

        return MemoryListResponse(
            memories=result["memories"],
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            has_more=result["has_more"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


@router.get("/memories/{content_hash}", response_model=MemoryResponse, tags=["memories"])
async def get_memory(
    content_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Get a specific memory by its content hash.
    
    Retrieves a single memory entry using its unique content hash identifier.
    """
    try:
        # Use the new get_by_hash method for direct hash lookup
        memory = await storage.get_by_hash(content_hash)
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory_to_response(memory)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")


@router.delete("/memories/{content_hash}", response_model=MemoryDeleteResponse, tags=["memories"])
async def delete_memory(
    content_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Delete a memory by its content hash.
    
    Permanently removes a memory entry from the storage.
    """
    try:
        success, message = await storage.delete(content_hash)
        
        # Broadcast SSE event for memory deletion
        try:
            event = create_memory_deleted_event(content_hash, success)
            await sse_manager.broadcast_event(event)
        except Exception as e:
            # Don't fail the request if SSE broadcasting fails
            logger.warning(f"Failed to broadcast memory_deleted event: {e}")
        
        return MemoryDeleteResponse(
            success=success,
            message=message,
            content_hash=content_hash
        )

    except Exception as e:
        logger.error(f"Failed to delete memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete memory. Please try again.")


@router.put("/memories/{content_hash}", response_model=MemoryUpdateResponse, tags=["memories"])
async def update_memory(
    content_hash: str,
    request: MemoryUpdateRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Update memory metadata (tags, type, metadata) without changing content or timestamps.

    This endpoint allows updating only the metadata aspects of a memory while preserving
    the original content and creation timestamp. Only provided fields will be updated.
    """
    try:
        # First, check if the memory exists
        existing_memory = await storage.get_by_hash(content_hash)
        if not existing_memory:
            raise HTTPException(status_code=404, detail=f"Memory with hash {content_hash} not found")

        # Build the updates dictionary with only provided fields
        updates = {}
        if request.tags is not None:
            updates['tags'] = request.tags
        if request.memory_type is not None:
            updates['memory_type'] = request.memory_type
        if request.metadata is not None:
            updates['metadata'] = request.metadata

        # If no updates provided, return current memory
        if not updates:
            return MemoryUpdateResponse(
                success=True,
                message="No updates provided - memory unchanged",
                content_hash=content_hash,
                memory=memory_to_response(existing_memory)
            )

        # Perform the update
        success, message = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates=updates,
            preserve_timestamps=True
        )

        if success:
            # Get the updated memory
            updated_memory = await storage.get_by_hash(content_hash)

            return MemoryUpdateResponse(
                success=True,
                message=message,
                content_hash=content_hash,
                memory=memory_to_response(updated_memory) if updated_memory else None
            )
        else:
            return MemoryUpdateResponse(
                success=False,
                message=message,
                content_hash=content_hash
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")


@router.get("/tags", response_model=TagListResponse, tags=["tags"])
async def get_tags(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Get all tags with their usage counts.

    Returns a list of all unique tags along with how many memories use each tag,
    sorted by count in descending order.
    """
    try:
        # Get tags with counts from storage
        tag_data = await storage.get_all_tags_with_counts()

        # Convert to response format
        tags = [TagResponse(tag=item["tag"], count=item["count"]) for item in tag_data]

        return TagListResponse(tags=tags)

    except AttributeError as e:
        # Handle case where storage backend doesn't implement get_all_tags_with_counts
        raise HTTPException(status_code=501, detail=f"Tags endpoint not supported by current storage backend: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


# ── Session ingestion ──────────────────────────────────────────────────────────

class SessionTurn(BaseModel):
    role: str = Field(..., description="Speaker role, e.g. 'user' or 'assistant'")
    content: str = Field(..., description="Turn content")


class SessionCreateRequest(BaseModel):
    turns: List[SessionTurn] = Field(..., min_length=1, description="Ordered conversation turns (at least one required)")
    session_id: Optional[str] = Field(None, description="Stable session identifier; auto-generated UUID if omitted")
    tags: List[str] = Field(default=[], description="Additional tags. 'session:<id>' is always added automatically.")
    metadata: Dict[str, Any] = Field(default={}, description="Optional extra metadata")


class SessionCreateResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    content_hash: Optional[str] = None
    turn_count: int


@router.post("/sessions", response_model=SessionCreateResponse, tags=["memories"])
async def store_session(
    request: SessionCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access),
):
    """Store a conversation session as a single memory unit.

    All turns are concatenated into '[role] content' lines and stored as
    memory_type='session' with a 'session:<id>' tag for reliable session-level retrieval.
    """
    import uuid as _uuid

    session_id = request.session_id or str(_uuid.uuid4())
    lines = [f"[{t.role}] {t.content.strip()}" for t in request.turns if t.content.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="All turns have empty content")

    content = "\n".join(lines)
    tags = [f"session:{session_id}"] + request.tags

    try:
        result = await memory_service.store_memory(
            content=content,
            tags=tags,
            memory_type="session",
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Storage error"))

    raw_memory = result.get("memory") or {}
    return SessionCreateResponse(
        success=True,
        message="Session stored successfully",
        session_id=session_id,
        content_hash=raw_memory.get("content_hash"),
        turn_count=len(lines),
    )