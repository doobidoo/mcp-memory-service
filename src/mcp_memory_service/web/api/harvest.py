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
Session Harvest API endpoint.

Extracts learnings (decisions, bugs, conventions, learnings, context) from
Claude Code session transcripts. Wraps :class:`SessionHarvester` to mirror
the ``memory_harvest`` MCP tool over HTTP.
"""

import asyncio
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...harvest.harvester import SessionHarvester
from ...harvest.models import HARVEST_TYPES, HarvestConfig, MAX_CANDIDATE_PREVIEW_LENGTH
from ..oauth.middleware import AuthenticationResult, require_write_access

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/harvest", tags=["harvest"])


class HarvestRequest(BaseModel):
    """Request model for session harvest."""
    sessions: int = Field(default=1, ge=1, description="Number of recent sessions to harvest")
    session_ids: Optional[List[str]] = Field(default=None, description="Specific session IDs to harvest")
    use_llm: bool = Field(default=False, description="Enable LLM-based classification (Phase 2)")
    dry_run: bool = Field(default=True, description="Preview candidates without storing")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    types: List[str] = Field(
        default_factory=lambda: list(HARVEST_TYPES),
        description="Candidate memory types to include",
    )
    project_path: Optional[str] = Field(default=None, description="Override project directory")


class HarvestCandidateModel(BaseModel):
    """A single extracted candidate in the response."""
    type: str
    content: str
    confidence: float
    tags: List[str]


class HarvestSessionResult(BaseModel):
    """Per-session harvest result."""
    session_id: str
    total_messages: int
    found: int
    stored: int
    by_type: Dict[str, int]
    candidates: List[HarvestCandidateModel]


class HarvestResponse(BaseModel):
    """Response model mirroring the ``memory_harvest`` MCP tool output."""
    dry_run: bool
    results: List[HarvestSessionResult]


_INVALID_PROJECT_PATH = (
    "project_path must be a directory name under ~/.claude/projects/ "
    "(no '..', no absolute paths, no path separators beyond nested project dirs)"
)


def _resolve_project_path(override: Optional[str]) -> Path:
    """Resolve the Claude project directory.

    Unlike the MCP tool (which can infer the project from CWD because it
    runs inside the Claude Code process), the HTTP endpoint runs in a
    long-lived web server whose CWD is typically the repo root or ``/``.
    CWD-based inference there would silently point at the wrong directory,
    so ``project_path`` is required.

    ``project_path`` is treated as a *relative name* under
    ``~/.claude/projects/``. Absolute paths, ``..`` components, and
    escape via symlinks are rejected with HTTP 400 (CodeQL #383/#384).
    """
    if not override:
        raise HTTPException(
            status_code=400,
            detail="project_path is required for HTTP harvest (CWD inference only works in MCP context)",
        )
    claude_projects = (Path.home() / ".claude" / "projects").resolve()
    relative = override.strip()
    pure = PurePosixPath(relative.replace(os.sep, "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise HTTPException(status_code=400, detail=_INVALID_PROJECT_PATH)
    candidate = (claude_projects / Path(*pure.parts)).resolve()
    if not candidate.is_relative_to(claude_projects):
        raise HTTPException(status_code=400, detail=_INVALID_PROJECT_PATH)
    return candidate


@router.post("", response_model=HarvestResponse)
async def harvest_sessions(
    request: HarvestRequest,
    user: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    """
    Extract learnings from Claude Code session transcripts.

    When ``dry_run`` is true (default), returns candidates without storing.
    When false, stores high-confidence candidates as memories (evolves existing
    similar memories where possible).

    Raises:
        HTTPException: 400 for invalid input, 404 if project directory missing,
            500 for harvest failures.
    """
    invalid_types = [t for t in request.types if t not in HARVEST_TYPES]
    if invalid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid types: {invalid_types}. Must be subset of {HARVEST_TYPES}",
        )

    project_path = _resolve_project_path(request.project_path)
    if not project_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Project directory not found: {project_path}",
        )

    config = HarvestConfig(
        sessions=request.sessions,
        session_ids=request.session_ids,
        types=request.types,
        min_confidence=request.min_confidence,
        dry_run=request.dry_run,
        project_path=str(project_path),
        use_llm=request.use_llm,
    )

    memory_service = None
    if not config.dry_run:
        try:
            from ...services.memory_service import MemoryService
            from ..dependencies import get_storage
            storage = get_storage()
            memory_service = MemoryService(storage)
        except Exception:
            logger.error("Failed to initialize memory service for harvest", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to initialize storage for harvest")

    harvester = SessionHarvester(project_dir=project_path, memory_service=memory_service)

    try:
        if config.dry_run:
            # harvest() does synchronous file I/O — offload to avoid blocking the event loop.
            results = await asyncio.to_thread(harvester.harvest, config)
        else:
            results = await harvester.harvest_and_store(config)
    except Exception:
        logger.error("Session harvest failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Harvest failed")

    return {
        "dry_run": config.dry_run,
        "results": [
            {
                "session_id": r.session_id,
                "total_messages": r.total_messages,
                "found": r.found,
                "stored": r.stored,
                "by_type": r.by_type,
                "candidates": [
                    {
                        "type": c.memory_type,
                        "content": c.content[:MAX_CANDIDATE_PREVIEW_LENGTH],
                        "confidence": round(c.confidence, 2),
                        "tags": c.tags,
                    }
                    for c in r.candidates
                ],
            }
            for r in results
        ],
    }
