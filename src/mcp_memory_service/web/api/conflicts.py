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
REST API endpoints for memory conflict detection and resolution.

Provides endpoints to list unresolved conflicts between semantically similar
but textually divergent memories, and to resolve them by picking a winner.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ..dependencies import get_storage
from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult

router = APIRouter()
logger = logging.getLogger(__name__)


# Response Models
class ConflictResponse(BaseModel):
    """A single unresolved conflict between two memories."""
    hash_a: str = Field(..., description="Content hash of first memory")
    hash_b: str = Field(..., description="Content hash of second memory")
    content_a: str = Field(..., description="Content of first memory")
    content_b: str = Field(..., description="Content of second memory")
    similarity: float = Field(..., description="Cosine similarity between the two memories")
    divergence: Optional[float] = Field(None, description="Text divergence (1 - SequenceMatcher ratio)")
    detected_at: Optional[float] = Field(None, description="Unix timestamp when conflict was detected")


class ConflictListResponse(BaseModel):
    """Response for listing conflicts."""
    conflicts: List[ConflictResponse]
    count: int = Field(..., description="Number of unresolved conflicts")


# Request Models
class ResolveRequest(BaseModel):
    """Request to resolve a conflict by choosing a winner."""
    winner_hash: str = Field(..., description="Content hash of the memory to keep (winner)")
    loser_hash: str = Field(..., description="Content hash of the memory to supersede (loser)")


class ResolveResponse(BaseModel):
    """Response after resolving a conflict."""
    success: bool
    message: str


@router.get("/conflicts", response_model=ConflictListResponse, tags=["conflicts"])
async def list_conflicts(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access),
):
    """
    List all unresolved memory conflicts.

    Returns pairs of memories that are semantically similar (cosine > 0.95)
    but textually divergent (divergence > 20%), indicating potential contradictions.
    Only active, non-superseded memories are included.
    """
    try:
        conflicts = await storage.get_conflicts()
        return ConflictListResponse(
            conflicts=[ConflictResponse(**c) for c in conflicts],
            count=len(conflicts),
        )
    except Exception:
        logger.error("Failed to list conflicts", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve conflicts")


@router.post("/conflicts/resolve", response_model=ResolveResponse, tags=["conflicts"])
async def resolve_conflict(
    request: ResolveRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access),
):
    """
    Resolve a conflict by choosing a winner.

    The winner memory gets its confidence boosted to 1.0. The loser memory
    is marked as superseded by the winner. The conflict:unresolved tag is
    removed from both memories.
    """
    try:
        ok, msg = await storage.resolve_conflict(request.winner_hash, request.loser_hash)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return ResolveResponse(success=ok, message=msg)
    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to resolve conflict", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resolve conflict")
