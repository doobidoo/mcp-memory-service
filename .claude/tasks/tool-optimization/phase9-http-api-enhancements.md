# Phase 9: HTTP API Enhancements (Optional)

## Ziel

Die HTTP REST API mit den neuen unified Konzepten erweitern. Das Dashboard profitiert von flexibleren Endpunkten.

**Status:** OPTIONAL - Das Dashboard funktioniert auch ohne diese Änderungen.

---

## Aktuelle HTTP API Struktur

```
/api/
├── memories/           # CRUD
│   ├── POST           # Create memory
│   ├── GET            # List memories
│   ├── GET /:hash     # Get single memory
│   ├── PUT /:hash     # Update metadata
│   └── DELETE /:hash  # Delete single memory
├── search/            # Search
│   ├── POST           # Semantic search
│   ├── POST /by-tag   # Tag search
│   ├── POST /by-time  # Time search
│   └── GET /similar/:hash
├── consolidation/     # Consolidation
│   ├── POST /trigger
│   ├── GET /status
│   └── GET /recommendations/:horizon
└── ...
```

---

## Vorgeschlagene Erweiterungen

### 9.1: Unified Search Endpoint

Erweitere `POST /api/search` um alle Such-Modi:

**Datei:** `src/mcp_memory_service/web/api/search.py`

```python
class UnifiedSearchRequest(BaseModel):
    """Request model for unified search."""
    query: Optional[str] = Field(None, description="Search query")
    mode: Literal["semantic", "exact", "hybrid"] = Field(
        default="semantic", 
        description="Search mode"
    )
    time_expr: Optional[str] = Field(
        None, 
        description="Natural language time filter (e.g., 'last week')"
    )
    after: Optional[str] = Field(None, description="ISO date - results after")
    before: Optional[str] = Field(None, description="ISO date - results before")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    quality_boost: float = Field(
        default=0.0, 
        ge=0.0, le=1.0,
        description="Quality weight for reranking"
    )
    n_results: int = Field(default=10, ge=1, le=100)
    include_debug: bool = Field(default=False)


@router.post("/search/unified", response_model=SearchResponse, tags=["search"])
async def unified_search(
    request: UnifiedSearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Unified search endpoint combining semantic, exact, time-based and tag filtering.
    
    This endpoint provides a single interface for all search operations:
    - Semantic similarity search (default)
    - Exact string matching (mode="exact")
    - Time-based filtering (time_expr or after/before)
    - Tag filtering
    - Quality-boosted reranking
    
    Filters can be combined for precise queries.
    """
    import time
    start_time = time.time()
    
    try:
        result = await memory_service.search_memories(
            query=request.query,
            mode=request.mode,
            time_expr=request.time_expr,
            after=request.after,
            before=request.before,
            tags=request.tags,
            quality_boost=request.quality_boost,
            limit=request.n_results,
            include_debug=request.include_debug
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to response format
        search_results = []
        for memory_data in result.get("memories", []):
            search_results.append(SearchResult(
                memory=memory_data,
                similarity_score=memory_data.get("similarity_score"),
                relevance_reason=f"Mode: {request.mode}"
            ))
        
        return SearchResponse(
            results=search_results,
            total_found=result.get("total", len(search_results)),
            query=request.query or f"Time: {request.time_expr}",
            search_type=f"unified_{request.mode}",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Unified search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")
```

---

### 9.2: Bulk Delete Endpoint

Erweitere `DELETE /api/memories` für flexible Bulk-Löschung:

**Datei:** `src/mcp_memory_service/web/api/memories.py`

```python
class BulkDeleteRequest(BaseModel):
    """Request model for bulk deletion."""
    content_hash: Optional[str] = Field(None, description="Delete single by hash")
    tags: Optional[List[str]] = Field(None, description="Delete by tags")
    tag_match: Literal["any", "all"] = Field(default="any")
    after: Optional[str] = Field(None, description="Delete after date")
    before: Optional[str] = Field(None, description="Delete before date")
    dry_run: bool = Field(default=False, description="Preview without deleting")


class BulkDeleteResponse(BaseModel):
    """Response model for bulk deletion."""
    success: bool
    deleted_count: int
    deleted_hashes: List[str] = []
    dry_run: bool


@router.post("/memories/bulk-delete", response_model=BulkDeleteResponse, tags=["memories"])
async def bulk_delete_memories(
    request: BulkDeleteRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Bulk delete memories with flexible filtering.
    
    Supports deletion by:
    - Single content hash
    - Tags (any or all match)
    - Time range
    - Combinations of the above
    
    Use dry_run=true to preview what would be deleted.
    """
    try:
        result = await memory_service.delete_memories(
            content_hash=request.content_hash,
            tags=request.tags,
            tag_match=request.tag_match,
            after=request.after,
            before=request.before,
            dry_run=request.dry_run
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return BulkDeleteResponse(
            success=result.get("success", True),
            deleted_count=result.get("deleted_count", 0),
            deleted_hashes=result.get("deleted_hashes", []),
            dry_run=request.dry_run
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Delete operation failed")
```

---

### 9.3: Unified Consolidation Endpoint

**Datei:** `src/mcp_memory_service/web/api/consolidation.py`

```python
class UnifiedConsolidationRequest(BaseModel):
    """Request model for unified consolidation operations."""
    action: Literal["run", "status", "recommend", "scheduler", "pause", "resume"]
    time_horizon: Optional[Literal["daily", "weekly", "monthly", "quarterly", "yearly"]] = None
    immediate: bool = Field(default=True)


@router.post("/consolidation/manage", tags=["consolidation"])
async def manage_consolidation(
    request: UnifiedConsolidationRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Unified consolidation management endpoint.
    
    Actions:
    - run: Execute consolidation (requires time_horizon)
    - status: Get consolidation system status
    - recommend: Get recommendations for time_horizon
    - scheduler: View scheduled jobs
    - pause: Pause jobs (optional time_horizon)
    - resume: Resume jobs (optional time_horizon)
    """
    try:
        result = await memory_service.manage_consolidation(
            action=request.action,
            time_horizon=request.time_horizon,
            immediate=request.immediate
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consolidation operation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Operation failed")
```

---

## Dashboard Integration (Optional)

Das Dashboard könnte die neuen Endpoints nutzen:

### app.js Erweiterungen

```javascript
// Unified search with all modes
async unifiedSearch(params) {
    const response = await this.apiCall('/search/unified', {
        method: 'POST',
        body: JSON.stringify({
            query: params.query,
            mode: params.mode || 'semantic',
            time_expr: params.timeExpr,
            tags: params.tags,
            quality_boost: params.qualityBoost || 0,
            n_results: params.limit || 10
        })
    });
    return response;
}

// Bulk delete with preview
async bulkDelete(filters, dryRun = false) {
    const response = await this.apiCall('/memories/bulk-delete', {
        method: 'POST',
        body: JSON.stringify({
            ...filters,
            dry_run: dryRun
        })
    });
    return response;
}
```

---

## Checkliste

- [ ] `UnifiedSearchRequest` Model erstellt
- [ ] `POST /api/search/unified` Endpoint implementiert
- [ ] `BulkDeleteRequest` Model erstellt
- [ ] `POST /api/memories/bulk-delete` Endpoint implementiert
- [ ] `UnifiedConsolidationRequest` Model erstellt
- [ ] `POST /api/consolidation/manage` Endpoint implementiert
- [ ] OpenAPI Schema aktualisiert
- [ ] Dashboard app.js erweitert (optional)
- [ ] API Tests hinzugefügt

---

## Hinweis

Diese Phase ist **optional**. Die bestehende HTTP API bleibt funktionsfähig.
Die neuen Endpoints sind Erweiterungen, keine Ersetzungen.

Das Dashboard funktioniert ohne diese Änderungen, da es die bestehenden REST-Endpunkte nutzt.
