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
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...models.memory import Memory, MemoryQueryResult
from ...config import OAUTH_ENABLED
from ...utils.time_parser import parse_time_expression
from ..dependencies import get_storage
from .memories import MemoryResponse, memory_to_response
from ..sse import sse_manager, create_search_completed_event

# Constants
_TIME_SEARCH_CANDIDATE_POOL_SIZE = 100  # Number of candidates to retrieve for time filtering (reduced for performance)

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Request Models
class SemanticSearchRequest(BaseModel):
    """Request model for semantic similarity search."""
    query: str = Field(..., description="The search query for semantic similarity")
    n_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class TagSearchRequest(BaseModel):
    """Request model for tag-based search."""
    tags: List[str] = Field(..., description="List of tags to search for (ANY match)")
    match_all: bool = Field(default=False, description="If true, memory must have ALL tags; if false, ANY tag")
    time_filter: Optional[str] = Field(None, description="Optional natural language time filter (e.g., 'last week', 'yesterday')")


class TimeSearchRequest(BaseModel):
    """Request model for time-based search."""
    query: str = Field(..., description="Natural language time query (e.g., 'last week', 'yesterday')")
    n_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    semantic_query: Optional[str] = Field(None, description="Optional semantic query for relevance filtering within time range")


# Response Models
class SearchResult(BaseModel):
    """Individual search result with similarity score."""
    memory: MemoryResponse
    similarity_score: Optional[float] = Field(None, description="Similarity score (0-1, higher is more similar)")
    relevance_reason: Optional[str] = Field(None, description="Why this result was included")


class SearchResponse(BaseModel):
    """Response model for search operations."""
    results: List[SearchResult]
    total_found: int
    query: str
    search_type: str
    processing_time_ms: Optional[float] = None


def memory_query_result_to_search_result(query_result: MemoryQueryResult) -> SearchResult:
    """Convert MemoryQueryResult to SearchResult format."""
    return SearchResult(
        memory=memory_to_response(query_result.memory),
        similarity_score=query_result.relevance_score,
        relevance_reason=f"Semantic similarity: {query_result.relevance_score:.3f}" if query_result.relevance_score else None
    )


def memory_to_search_result(memory: Memory, reason: str = None) -> SearchResult:
    """Convert Memory to SearchResult format."""
    return SearchResult(
        memory=memory_to_response(memory),
        similarity_score=None,
        relevance_reason=reason
    )


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(
    request: SemanticSearchRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Perform semantic similarity search on memory content.
    
    Uses vector embeddings to find memories with similar meaning to the query,
    even if they don't share exact keywords.
    """
    import time
    start_time = time.time()
    
    try:
        # Perform semantic search using the storage layer
        query_results = await storage.retrieve(
            query=request.query,
            n_results=request.n_results
        )
        
        # Filter by similarity threshold if specified
        if request.similarity_threshold is not None:
            query_results = [
                result for result in query_results
                if result.relevance_score and result.relevance_score >= request.similarity_threshold
            ]
        
        # Convert to search results
        search_results = [
            memory_query_result_to_search_result(result)
            for result in query_results
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        # Broadcast SSE event for search completion
        try:
            event = create_search_completed_event(
                query=request.query,
                search_type="semantic",
                results_count=len(search_results),
                processing_time_ms=processing_time
            )
            await sse_manager.broadcast_event(event)
        except Exception as e:
            logger.warning(f"Failed to broadcast search_completed event: {e}")
        
        return SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=request.query,
            search_type="semantic",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search operation failed. Please try again.")


@router.post("/search/by-tag", response_model=SearchResponse, tags=["search"])
async def tag_search(
    request: TagSearchRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Search memories by tags with optional time filtering.

    Finds memories that contain any of the specified tags (OR search) or
    all of the specified tags (AND search) based on the match_all parameter.

    Optionally filters by time range using natural language expressions like
    'last week', 'yesterday', 'this month', etc.
    """
    import time
    start_time = time.time()

    try:
        if not request.tags:
            raise HTTPException(status_code=400, detail="At least one tag must be specified")

        # Parse time filter if provided
        time_start = None
        if request.time_filter:
            start_ts, _ = parse_time_expression(request.time_filter)
            time_start = start_ts if start_ts else None

        # Use the storage layer's tag search with optional time filtering
        memories = await storage.search_by_tag(request.tags, time_start=time_start)

        # If match_all is True, filter to only memories that have ALL tags
        if request.match_all and len(request.tags) > 1:
            tag_set = set(request.tags)
            memories = [
                memory for memory in memories
                if tag_set.issubset(set(memory.tags))
            ]

        # Convert to search results
        match_type = "ALL" if request.match_all else "ANY"
        search_results = [
            memory_to_search_result(
                memory,
                reason=f"Tags match ({match_type}): {', '.join(set(memory.tags) & set(request.tags))}"
            )
            for memory in memories
        ]

        processing_time = (time.time() - start_time) * 1000

        # Build query string with time filter info if present
        query_string = f"Tags: {', '.join(request.tags)} ({match_type})"
        if request.time_filter:
            query_string += f" | Time: {request.time_filter}"

        # Broadcast SSE event for search completion
        try:
            event = create_search_completed_event(
                query=query_string,
                search_type="tag",
                results_count=len(search_results),
                processing_time_ms=processing_time
            )
            await sse_manager.broadcast_event(event)
        except Exception as e:
            logger.warning(f"Failed to broadcast search_completed event: {e}")

        return SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=query_string,
            search_type="tag",
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tag search failed: {str(e)}")


@router.post("/search/by-time", response_model=SearchResponse, tags=["search"])
async def time_search(
    request: TimeSearchRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Search memories by time-based queries.
    
    Supports natural language time expressions like 'yesterday', 'last week',
    'this month', etc. Currently implements basic time filtering - full natural
    language parsing can be enhanced later.
    """
    import time
    start_time = time.time()
    
    try:
        # Parse time query using robust time_parser
        start_ts, end_ts = parse_time_expression(request.query)

        if start_ts is None and end_ts is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse time query: '{request.query}'. Try 'yesterday', 'last week', 'this month', etc."
            )

        # Retrieve memories within time range (with larger candidate pool if semantic query provided)
        candidate_pool_size = _TIME_SEARCH_CANDIDATE_POOL_SIZE if request.semantic_query else request.n_results
        query_results = await storage.recall(
            query=request.semantic_query.strip() if request.semantic_query and request.semantic_query.strip() else None,
            n_results=candidate_pool_size,
            start_timestamp=start_ts,
            end_timestamp=end_ts
        )

        # If semantic query was provided, results are already ranked by relevance
        # Otherwise, sort by recency (newest first)
        if not (request.semantic_query and request.semantic_query.strip()):
            query_results.sort(key=lambda r: r.memory.created_at or 0.0, reverse=True)

        # Limit results
        filtered_memories = query_results[:request.n_results]
        
        # Convert to search results
        search_results = [
            memory_query_result_to_search_result(result)
            for result in filtered_memories
        ]
        
        # Update relevance reason for time-based results
        for result in search_results:
            result.relevance_reason = f"Time match: {request.query}"
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=request.query,
            search_type="time",
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time search failed: {str(e)}")


@router.get("/search/similar/{content_hash}", response_model=SearchResponse, tags=["search"])
async def find_similar(
    content_hash: str,
    n_results: int = Query(default=10, ge=1, le=100, description="Number of similar memories to find"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
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
            n_results=n_results + 1  # +1 because the original will be included
        )
        
        # Filter out the original memory
        filtered_results = [
            result for result in similar_results
            if result.memory.content_hash != content_hash
        ][:n_results]
        
        # Convert to search results
        search_results = [
            memory_query_result_to_search_result(result)
            for result in filtered_results
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=f"Similar to: {target_memory.content[:50]}...",
            search_type="similar",
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")
