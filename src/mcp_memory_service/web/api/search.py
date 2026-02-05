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
Search endpoints for the HTTP interface.

Provides semantic search, tag-based search, and time-based recall functionality.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...config import OAUTH_ENABLED
from ...models.memory import Memory, MemoryQueryResult
from ...services.memory_service import MemoryService
from ...storage.base import MemoryStorage
from ..dependencies import get_memory_service, get_storage
from ..sse import create_search_completed_event, sse_manager
from .memories import MemoryResponse, memory_to_response

# Constants
_TIME_SEARCH_CANDIDATE_POOL_SIZE = 100  # Number of candidates to retrieve for time filtering (reduced for performance)

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import AuthenticationResult, require_read_access
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Pagination Models
class PaginationParams(BaseModel):
    """Reusable pagination request parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of results per page")


class PaginationMetadata(BaseModel):
    """Standard pagination response metadata."""

    total: int = Field(..., description="Total number of matching records across all pages")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of results per page")
    has_more: bool = Field(..., description="Whether more pages exist")
    total_pages: int = Field(..., description="Total number of pages available")


# Request Models
class SemanticSearchRequest(BaseModel):
    """Request model for semantic similarity search."""

    query: str = Field(..., description="The search query for semantic similarity")
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of results per page")
    similarity_threshold: float | None = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum similarity score (default: 0.6 for quality filtering)"
    )


class TagSearchRequest(BaseModel):
    """Request model for tag-based search."""

    tags: list[str] = Field(..., description="List of tags to search for")
    match_all: bool = Field(default=False, description="If true, memory must have ALL tags; if false, ANY tag")
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of results per page")


class TimeSearchRequest(BaseModel):
    """Request model for time-based search."""

    query: str = Field(..., description="Natural language time query (e.g., 'last week', 'yesterday')")
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of results per page")
    semantic_query: str | None = Field(None, description="Optional semantic query for relevance filtering within time range")


# Response Models
class SearchResult(BaseModel):
    """Individual search result with similarity score."""

    memory: MemoryResponse
    similarity_score: float | None = Field(None, description="Similarity score (0-1, higher is more similar)")
    relevance_reason: str | None = Field(None, description="Why this result was included")


class SearchResponse(BaseModel):
    """Response model for search operations."""

    results: list[SearchResult]
    query: str
    search_type: str
    pagination: PaginationMetadata
    processing_time_ms: float | None = None


def memory_query_result_to_search_result(query_result: MemoryQueryResult) -> SearchResult:
    """Convert MemoryQueryResult to SearchResult format."""
    return SearchResult(
        memory=memory_to_response(query_result.memory),
        similarity_score=query_result.relevance_score,
        relevance_reason=f"Semantic similarity: {query_result.relevance_score:.3f}" if query_result.relevance_score else None,
    )


def memory_to_search_result(memory: Memory, reason: str = None) -> SearchResult:
    """Convert Memory to SearchResult format."""
    return SearchResult(memory=memory_to_response(memory), similarity_score=None, relevance_reason=reason)


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(
    request: SemanticSearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Perform semantic similarity search on memory content with pagination.

    Uses vector embeddings to find memories with similar meaning to the query,
    even if they don't share exact keywords.
    """
    import time

    start_time = time.time()

    try:
        # Perform semantic search using the memory service
        result = await memory_service.retrieve_memories(
            query=request.query, page=request.page, page_size=request.page_size, min_similarity=request.similarity_threshold
        )

        # Convert memories to search results
        search_results = []
        for memory_dict in result.get("memories", []):
            # Extract similarity score if present
            similarity_score = memory_dict.pop("similarity_score", None)

            # Convert to MemoryResponse format and create SearchResult
            search_result = SearchResult(
                memory=MemoryResponse(**memory_dict),
                similarity_score=similarity_score,
                relevance_reason=f"Semantic similarity: {similarity_score:.3f}" if similarity_score else None,
            )
            search_results.append(search_result)

        processing_time = (time.time() - start_time) * 1000

        # Build pagination metadata
        pagination = PaginationMetadata(
            total=result.get("total", 0),
            page=result.get("page", request.page),
            page_size=result.get("page_size", request.page_size),
            has_more=result.get("has_more", False),
            total_pages=result.get("total_pages", 1),
        )

        # Broadcast SSE event for search completion
        try:
            event = create_search_completed_event(
                query=request.query,
                search_type="semantic",
                results_count=len(search_results),
                processing_time_ms=processing_time,
            )
            await sse_manager.broadcast_event(event)
        except Exception as e:
            logger.warning(f"Failed to broadcast search_completed event: {e}")

        return SearchResponse(
            results=search_results,
            query=request.query,
            search_type="semantic",
            pagination=pagination,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search operation failed. Please try again.") from e


@router.post("/search/by-tag", response_model=SearchResponse, tags=["search"])
async def tag_search(
    request: TagSearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Search memories by tags with pagination.

    Finds memories that contain any of the specified tags (OR search) or
    all of the specified tags (AND search) based on the match_all parameter.
    """
    import time

    start_time = time.time()

    try:
        if not request.tags:
            raise HTTPException(status_code=400, detail="At least one tag must be specified")

        # Use the memory service's tag search with pagination
        result = await memory_service.search_by_tag(
            tags=request.tags, match_all=request.match_all, page=request.page, page_size=request.page_size
        )

        # Convert memories to search results
        match_type = result.get("match_type", "ALL" if request.match_all else "ANY")
        search_results = []
        for memory_dict in result.get("memories", []):
            # Get tags for this memory to show which ones matched
            memory_tags = memory_dict.get("tags", [])
            matched_tags = set(memory_tags) & set(request.tags)

            search_result = SearchResult(
                memory=MemoryResponse(**memory_dict),
                similarity_score=None,
                relevance_reason=f"Tags match ({match_type}): {', '.join(matched_tags)}" if matched_tags else None,
            )
            search_results.append(search_result)

        processing_time = (time.time() - start_time) * 1000

        query_string = f"Tags: {', '.join(request.tags)} ({match_type})"

        # Build pagination metadata
        pagination = PaginationMetadata(
            total=result.get("total", 0),
            page=result.get("page", request.page),
            page_size=result.get("page_size", request.page_size),
            has_more=result.get("has_more", False),
            total_pages=result.get("total_pages", 1),
        )

        # Broadcast SSE event for search completion
        try:
            event = create_search_completed_event(
                query=query_string, search_type="tag", results_count=len(search_results), processing_time_ms=processing_time
            )
            await sse_manager.broadcast_event(event)
        except Exception as e:
            logger.warning(f"Failed to broadcast search_completed event: {e}")

        return SearchResponse(
            results=search_results,
            query=query_string,
            search_type="tag",
            pagination=pagination,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tag search failed: {str(e)}") from e


@router.post("/search/by-time", response_model=SearchResponse, tags=["search"])
async def time_search(
    request: TimeSearchRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Search memories by time-based queries with pagination.

    Supports natural language time expressions like 'yesterday', 'last week',
    'this month', etc. Currently implements basic time filtering - full natural
    language parsing can be enhanced later.
    """
    import time

    start_time = time.time()

    try:
        # Parse time query (basic implementation)
        time_filter = parse_time_query(request.query)

        if not time_filter:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse time query: '{request.query}'. Try 'yesterday', 'last week', 'this month', etc.",
            )

        # Get time range timestamps
        start_dt = time_filter.get("start")
        end_dt = time_filter.get("end")
        start_ts = start_dt.timestamp() if start_dt else None
        end_ts = end_dt.timestamp() if end_dt else None

        # Calculate offset for pagination
        offset = (request.page - 1) * request.page_size

        # Get total count for pagination metadata
        total = await storage.count_time_range(start_timestamp=start_ts, end_timestamp=end_ts, tags=None, memory_type=None)

        # Retrieve memories within time range with pagination
        query_results = await storage.recall(
            query=request.semantic_query.strip() if request.semantic_query and request.semantic_query.strip() else None,
            n_results=request.page_size,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            offset=offset,
        )

        # If no semantic query was provided, sort by recency (newest first)
        # Note: This is only for display ordering within the page
        if not (request.semantic_query and request.semantic_query.strip()):
            query_results.sort(key=lambda r: r.memory.created_at or 0.0, reverse=True)

        # Convert to search results
        search_results = [memory_query_result_to_search_result(result) for result in query_results]

        # Update relevance reason for time-based results
        for result in search_results:
            result.relevance_reason = f"Time match: {request.query}"

        processing_time = (time.time() - start_time) * 1000

        # Build pagination metadata
        pagination = PaginationMetadata(
            total=total,
            page=request.page,
            page_size=request.page_size,
            has_more=(request.page * request.page_size) < total,
            total_pages=(total + request.page_size - 1) // request.page_size if request.page_size > 0 else 1,
        )

        return SearchResponse(
            results=search_results,
            query=request.query,
            search_type="time",
            pagination=pagination,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time search failed: {str(e)}") from e


@router.get("/search/similar/{content_hash}", response_model=SearchResponse, tags=["search"])
async def find_similar(
    content_hash: str,
    n_results: int = Query(default=10, ge=1, le=100, description="Number of similar memories to find"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Find memories similar to a specific memory identified by its content hash.

    Uses the content of the specified memory as a search query to find
    semantically similar memories.
    """
    import time

    start_time = time.time()

    try:
        # First, get the target memory by searching with its hash
        # This is inefficient but works with current storage interface
        target_results = await storage.retrieve(content_hash, n_results=1)

        if not target_results or target_results[0].memory.content_hash != content_hash:
            raise HTTPException(status_code=404, detail="Memory not found")

        target_memory = target_results[0].memory

        # Use the target memory's content to find similar memories
        similar_results = await storage.retrieve(
            query=target_memory.content,
            n_results=n_results + 1,  # +1 because the original will be included
        )

        # Filter out the original memory
        filtered_results = [result for result in similar_results if result.memory.content_hash != content_hash][:n_results]

        # Convert to search results
        search_results = [memory_query_result_to_search_result(result) for result in filtered_results]

        processing_time = (time.time() - start_time) * 1000

        # Build pagination metadata (single page, no actual pagination for this endpoint)
        total = len(search_results)
        pagination = PaginationMetadata(total=total, page=1, page_size=n_results, has_more=False, total_pages=1)

        return SearchResponse(
            results=search_results,
            query=f"Similar to: {target_memory.content[:50]}...",
            search_type="similar",
            pagination=pagination,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}") from e


# Helper functions for time parsing
def parse_time_query(query: str) -> dict[str, Any] | None:
    """
    Parse natural language time queries into time ranges.

    This is a basic implementation - can be enhanced with more sophisticated
    natural language processing later.
    """
    query_lower = query.lower().strip()
    now = datetime.now(timezone.utc)

    # Define time mappings
    if query_lower in ["yesterday"]:
        start = now - timedelta(days=1)
        return {"start": start.replace(hour=0, minute=0, second=0), "end": start.replace(hour=23, minute=59, second=59)}

    elif query_lower in ["today"]:
        return {"start": now.replace(hour=0, minute=0, second=0), "end": now}

    elif query_lower in ["last week", "past week"]:
        start = now - timedelta(weeks=1)
        return {"start": start, "end": now}

    elif query_lower in ["last month", "past month"]:
        start = now - timedelta(days=30)
        return {"start": start, "end": now}

    elif query_lower in ["this week"]:
        # Start of current week (Monday)
        days_since_monday = now.weekday()
        start = now - timedelta(days=days_since_monday)
        return {"start": start.replace(hour=0, minute=0, second=0), "end": now}

    elif query_lower in ["this month"]:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return {"start": start, "end": now}

    elif query_lower in ["last 2 weeks", "past 2 weeks", "last-2-weeks"]:
        start = now - timedelta(weeks=2)
        return {"start": start, "end": now}

    # Add more time expressions as needed
    return None


def is_within_time_range(memory_time: datetime, time_filter: dict[str, Any]) -> bool:
    """Check if a memory's timestamp falls within the specified time range."""
    start_time = time_filter.get("start")
    end_time = time_filter.get("end")

    if start_time and end_time:
        return start_time <= memory_time <= end_time
    elif start_time:
        return memory_time >= start_time
    elif end_time:
        return memory_time <= end_time

    return True
