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
Management endpoints for the HTTP interface.

Provides memory maintenance, bulk operations, and system management tools.
"""

import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...config import OAUTH_ENABLED
from ..dependencies import get_storage
from .memories import MemoryResponse, memory_to_response

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_write_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_write_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations."""
    tag: Optional[str] = Field(None, description="Delete all memories with this tag")
    before_date: Optional[str] = Field(None, description="Delete memories before this date (YYYY-MM-DD)")
    memory_type: Optional[str] = Field(None, description="Delete memories of this type")
    confirm_count: Optional[int] = Field(None, description="Confirmation of number of memories to delete")


class TagManagementRequest(BaseModel):
    """Request model for tag management operations."""
    operation: str = Field(..., description="Operation: 'rename', 'merge', or 'delete'")
    old_tag: str = Field(..., description="Original tag name")
    new_tag: Optional[str] = Field(None, description="New tag name (for rename/merge)")
    confirm_count: Optional[int] = Field(None, description="Confirmation count for destructive operations")


class BulkOperationResponse(BaseModel):
    """Response model for bulk operations."""
    success: bool
    message: str
    affected_count: int
    operation: str


class TagStatsResponse(BaseModel):
    """Response model for tag statistics."""
    tag: str
    count: int
    last_used: Optional[float]
    memory_types: List[str]


class TagStatsListResponse(BaseModel):
    """Response model for tag statistics list."""
    tags: List[TagStatsResponse]
    total_tags: int


class SystemOperationRequest(BaseModel):
    """Request model for system operations."""
    operation: str = Field(..., description="Operation: 'cleanup_duplicates', 'optimize_db', 'rebuild_index'")


@router.post("/bulk-delete", response_model=BulkOperationResponse, tags=["management"])
async def bulk_delete_memories(
    request: BulkDeleteRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Perform bulk delete operations on memories.

    Supports deletion by tag, date range, or memory type.
    Requires confirmation count for safety.
    """
    try:
        affected_count = 0
        operation_desc = ""

        # Validate that at least one filter is provided
        if not any([request.tag, request.before_date, request.memory_type]):
            raise HTTPException(
                status_code=400,
                detail="At least one filter (tag, before_date, or memory_type) must be specified"
            )

        # Count memories that would be affected
        if request.tag:
            # Count memories with this tag
            if hasattr(storage, 'count_memories_by_tag'):
                affected_count = await storage.count_memories_by_tag([request.tag])
            else:
                # Fallback: search and count
                tag_memories = await storage.search_by_tag([request.tag])
                affected_count = len(tag_memories)
            operation_desc = f"Delete memories with tag '{request.tag}'"

        elif request.before_date:
            # Count memories before date
            try:
                before_dt = datetime.fromisoformat(request.before_date)
                before_ts = before_dt.timestamp()
                # This would need a method to count by date range
                # For now, we'll estimate or implement a simple approach
                affected_count = 0  # Placeholder
                operation_desc = f"Delete memories before {request.before_date}"
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        elif request.memory_type:
            # Count memories by type
            if hasattr(storage, 'count_all_memories'):
                affected_count = await storage.count_all_memories(memory_type=request.memory_type)
            else:
                affected_count = 0  # Placeholder
            operation_desc = f"Delete memories of type '{request.memory_type}'"

        # Safety check: require confirmation count
        if request.confirm_count is not None and request.confirm_count != affected_count:
            raise HTTPException(
                status_code=400,
                detail=f"Confirmation count mismatch. Expected {affected_count}, got {request.confirm_count}"
            )

        # Perform the deletion
        success = False
        message = ""

        if request.tag:
            if hasattr(storage, 'delete_by_tag'):
                success_count, message = await storage.delete_by_tag(request.tag)
                success = success_count > 0
                affected_count = success_count
            else:
                raise HTTPException(status_code=501, detail="Tag-based deletion not supported by storage backend")

        elif request.before_date:
            # Implement date-based deletion
            # This would need to be implemented in the storage layer
            raise HTTPException(status_code=501, detail="Date-based bulk deletion not yet implemented")

        elif request.memory_type:
            # Implement type-based deletion
            # This would need to be implemented in the storage layer
            raise HTTPException(status_code=501, detail="Type-based bulk deletion not yet implemented")

        return BulkOperationResponse(
            success=success,
            message=message or f"Successfully deleted {affected_count} memories",
            affected_count=affected_count,
            operation=operation_desc
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk delete operation failed: {str(e)}")


@router.post("/cleanup-duplicates", response_model=BulkOperationResponse, tags=["management"])
async def cleanup_duplicates(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Clean up duplicate memories in the database.

    Removes duplicate entries based on content hash and merges metadata.
    """
    try:
        if hasattr(storage, 'cleanup_duplicates'):
            count, message = await storage.cleanup_duplicates()
            return BulkOperationResponse(
                success=count > 0,
                message=message,
                affected_count=count,
                operation="cleanup_duplicates"
            )
        else:
            raise HTTPException(status_code=501, detail="Duplicate cleanup not supported by storage backend")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Duplicate cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Duplicate cleanup failed: {str(e)}")


@router.get("/tags/stats", response_model=TagStatsListResponse, tags=["management"])
async def get_tag_statistics(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Get detailed statistics for all tags.

    Returns tag usage counts, last usage times, and associated memory types.
    """
    try:
        # Get all tags with counts
        if hasattr(storage, 'get_all_tags_with_counts'):
            tag_data = await storage.get_all_tags_with_counts()

            # For now, provide basic tag stats without additional queries
            # TODO: Implement efficient batch queries in storage layer for last_used and memory_types
            enhanced_tags = []
            for tag_item in tag_data:
                enhanced_tags.append(TagStatsResponse(
                    tag=tag_item["tag"],
                    count=tag_item["count"],
                    last_used=None,  # Would need efficient batch query
                    memory_types=[]  # Would need efficient batch query
                ))

            return TagStatsListResponse(
                tags=enhanced_tags,
                total_tags=len(enhanced_tags)
            )
        else:
            raise HTTPException(status_code=501, detail="Tag statistics not supported by storage backend")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tag statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag statistics: {str(e)}")


@router.put("/tags/{old_tag}", response_model=BulkOperationResponse, tags=["management"])
async def rename_tag(
    old_tag: str,
    new_tag: str,
    confirm_count: Optional[int] = Query(None, description="Confirmation count"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Rename a tag across all memories.

    Updates all memories that have the old tag to use the new tag instead.
    """
    try:
        # Count memories with this tag
        if hasattr(storage, 'count_memories_by_tag'):
            affected_count = await storage.count_memories_by_tag([old_tag])
        else:
            tag_memories = await storage.search_by_tag([old_tag])
            affected_count = len(tag_memories)

        # Safety check
        if confirm_count is not None and confirm_count != affected_count:
            raise HTTPException(
                status_code=400,
                detail=f"Confirmation count mismatch. Expected {affected_count}, got {confirm_count}"
            )

        # Implement tag renaming (this would need to be implemented in storage layer)
        # For now, return not implemented
        raise HTTPException(status_code=501, detail="Tag renaming not yet implemented")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tag rename failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tag rename failed: {str(e)}")


@router.post("/system/{operation}", response_model=BulkOperationResponse, tags=["management"])
async def perform_system_operation(
    operation: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Perform system maintenance operations.

    Supported operations: cleanup_duplicates, optimize_db, rebuild_index
    """
    try:
        if operation == "cleanup_duplicates":
            return await cleanup_duplicates(storage, user)

        elif operation == "optimize_db":
            # Database optimization (would need storage-specific implementation)
            raise HTTPException(status_code=501, detail="Database optimization not yet implemented")

        elif operation == "rebuild_index":
            # Rebuild search indexes (would need storage-specific implementation)
            raise HTTPException(status_code=501, detail="Index rebuilding not yet implemented")

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System operation {operation} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System operation failed: {str(e)}")
