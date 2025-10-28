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
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...models.memory import Memory
from ...utils.hashing import generate_content_hash
from ...config import INCLUDE_HOSTNAME, OAUTH_ENABLED
from ..dependencies import get_storage
from ..sse import sse_manager, create_memory_stored_event, create_memory_deleted_event

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None
    require_write_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory."""
    content: str = Field(..., description="The memory content to store")
    tags: List[str] = Field(default=[], description="Tags to categorize the memory")
    memory_type: Optional[str] = Field(None, description="Type of memory (e.g., 'note', 'reminder', 'fact')")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the memory")
    client_hostname: Optional[str] = Field(None, description="Client machine hostname for source tracking")


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
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Store a new memory.
    
    Creates a new memory entry with the provided content, tags, and metadata.
    The system automatically generates a unique hash for the content.
    """
    try:
        # Generate content hash
        content_hash = generate_content_hash(request.content)
        
        # Prepare tags and metadata with optional hostname
        final_tags = request.tags or []
        final_metadata = request.metadata or {}
        
        if INCLUDE_HOSTNAME:
            # Prioritize client-provided hostname, then header, then fallback to server
            hostname = None
            
            # 1. Check if client provided hostname in request body
            if request.client_hostname:
                hostname = request.client_hostname
                
            # 2. Check for X-Client-Hostname header
            elif http_request.headers.get('X-Client-Hostname'):
                hostname = http_request.headers.get('X-Client-Hostname')
                
            # 3. Fallback to server hostname (original behavior)
            else:
                hostname = socket.gethostname()
            
            source_tag = f"source:{hostname}"
            if source_tag not in final_tags:
                final_tags.append(source_tag)
            final_metadata["hostname"] = hostname
        
        # Create memory object
        memory = Memory(
            content=request.content,
            content_hash=content_hash,
            tags=final_tags,
            memory_type=request.memory_type,
            metadata=final_metadata
        )
        
        # Store the memory
        success, message = await storage.store(memory)
        
        if success:
            # Broadcast SSE event for successful memory storage
            try:
                memory_data = {
                    "content_hash": content_hash,
                    "content": memory.content,
                    "tags": memory.tags,
                    "memory_type": memory.memory_type
                }
                event = create_memory_stored_event(memory_data)
                await sse_manager.broadcast_event(event)
            except Exception as e:
                # Don't fail the request if SSE broadcasting fails
                logger.warning(f"Failed to broadcast memory_stored event: {e}")
            
            return MemoryCreateResponse(
                success=True,
                message=message,
                content_hash=content_hash,
                memory=memory_to_response(memory)
            )
        else:
            return MemoryCreateResponse(
                success=False,
                message=message,
                content_hash=content_hash
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
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    List memories with pagination.
    
    Retrieves memories with optional filtering by tag or memory type.
    Results are paginated for better performance.
    """
    try:
        # Calculate offset for pagination
        offset = (page - 1) * page_size
        
        if tag:
            # Filter by tag with proper chronological ordering and pagination
            if memory_type:
                # When filtering by both tag and memory_type, we need to get all matching
                # tag memories, filter by type, then paginate (suboptimal but correct)
                all_tag_memories = await storage.search_by_tag_chronological([tag])
                filtered_memories = [m for m in all_tag_memories if m.memory_type == memory_type]

                total = len(filtered_memories)
                page_memories = filtered_memories[offset:offset + page_size]
                has_more = offset + page_size < total
            else:
                # Tag-only filtering with server-side pagination
                page_memories = await storage.search_by_tag_chronological([tag], limit=page_size, offset=offset)
                total = await storage.count_memories_by_tag([tag])
                has_more = offset + page_size < total
        else:
            if memory_type:
                # Memory type filtering without tag - now efficiently handled at storage layer
                total = await storage.count_all_memories(memory_type=memory_type)
                page_memories = await storage.get_all_memories(limit=page_size, offset=offset, memory_type=memory_type)
                has_more = offset + page_size < total
            else:
                # No filtering - use efficient server-side pagination
                total = await storage.count_all_memories()
                page_memories = await storage.get_all_memories(limit=page_size, offset=offset)
                has_more = offset + page_size < total
        
        return MemoryListResponse(
            memories=[memory_to_response(m) for m in page_memories],
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


@router.get("/memories/{content_hash}", response_model=MemoryResponse, tags=["memories"])
async def get_memory(
    content_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
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
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
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
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
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
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
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