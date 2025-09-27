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
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...storage.sqlite_vec import SqliteVecMemoryStorage
from ...models.memory import Memory, MemoryQueryResult
from ...config import OAUTH_ENABLED
from ..dependencies import get_storage
from .memories import MemoryResponse, memory_to_response
from ..sse import sse_manager, create_search_completed_event

# OAuth authentication imports (conditional)
if OAUTH_ENABLED:
    from ..oauth.middleware import require_read_access, AuthenticationResult

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


class TimeSearchRequest(BaseModel):
    """Request model for time-based search."""
    query: str = Field(..., description="Natural language time query (e.g., 'last week', 'yesterday')")
    n_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")


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
    storage: SqliteVecMemoryStorage = Depends(get_storage),
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
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.post("/search/by-tag", response_model=SearchResponse, tags=["search"])
async def tag_search(
    request: TagSearchRequest,
    storage: SqliteVecMemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Search memories by tags.
    
    Finds memories that contain any of the specified tags (OR search) or
    all of the specified tags (AND search) based on the match_all parameter.
    """
    import time
    start_time = time.time()
    
    try:
        if not request.tags:
            raise HTTPException(status_code=400, detail="At least one tag must be specified")
        
        # Use the storage layer's tag search
        memories = await storage.search_by_tag(request.tags)
        
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
        
        query_string = f"Tags: {', '.join(request.tags)} ({match_type})"
        
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
    storage: SqliteVecMemoryStorage = Depends(get_storage),
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
        # Parse time query (basic implementation)
        time_filter = parse_time_query(request.query)
        
        if not time_filter:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not parse time query: '{request.query}'. Try 'yesterday', 'last week', 'this month', etc."
            )
        
        # For now, we'll do a broad search and then filter by time
        # TODO: Implement proper time-based search in storage layer
        query_results = await storage.retrieve("", n_results=1000)  # Get many results to filter
        
        # Filter by time
        filtered_memories = []
        for result in query_results:
            memory_time = None
            if result.memory.created_at:
                memory_time = datetime.fromtimestamp(result.memory.created_at)
            
            if memory_time and is_within_time_range(memory_time, time_filter):
                filtered_memories.append(result)
        
        # Limit results
        filtered_memories = filtered_memories[:request.n_results]
        
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
    storage: SqliteVecMemoryStorage = Depends(get_storage),
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


# Helper functions for time parsing
def parse_time_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Parse natural language time queries into time ranges.
    
    This is a basic implementation - can be enhanced with more sophisticated
    natural language processing later.
    """
    query_lower = query.lower().strip()
    now = datetime.now()
    
    # Define time mappings
    if query_lower in ['yesterday']:
        start = now - timedelta(days=1)
        end = now
        return {'start': start.replace(hour=0, minute=0, second=0), 'end': start.replace(hour=23, minute=59, second=59)}
    
    elif query_lower in ['today']:
        return {'start': now.replace(hour=0, minute=0, second=0), 'end': now}
    
    elif query_lower in ['last week', 'past week']:
        start = now - timedelta(weeks=1)
        return {'start': start, 'end': now}
    
    elif query_lower in ['last month', 'past month']:
        start = now - timedelta(days=30)
        return {'start': start, 'end': now}
    
    elif query_lower in ['this week']:
        # Start of current week (Monday)
        days_since_monday = now.weekday()
        start = now - timedelta(days=days_since_monday)
        return {'start': start.replace(hour=0, minute=0, second=0), 'end': now}
    
    elif query_lower in ['this month']:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return {'start': start, 'end': now}
    
    # Add more time expressions as needed
    return None


def is_within_time_range(memory_time: datetime, time_filter: Dict[str, Any]) -> bool:
    """Check if a memory's timestamp falls within the specified time range."""
    start_time = time_filter.get('start')
    end_time = time_filter.get('end')
    
    if start_time and end_time:
        return start_time <= memory_time <= end_time
    elif start_time:
        return memory_time >= start_time
    elif end_time:
        return memory_time <= end_time
    
    return True