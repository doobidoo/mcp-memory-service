---
name: h-fix-search-filtering
branch: fix/h-fix-search-filtering
status: completed
created: 2025-10-30
completed: 2025-11-01
---

# Fix Vector Search Filtering Architecture

## Problem/Goal
The current implementation performs **post-hoc filtering in Python** after vector search, which defeats semantic search optimization and returns incorrect results.

**Current broken behavior:**
1. User requests 5 memories with tag "important"
2. System does vector search for top 5 semantically similar memories
3. Then filters those 5 for tag "important" in Python
4. Returns 0-5 results (depending on how many have the tag)

**Expected behavior:**
1. Filter to memories with tag "important"
2. THEN do vector search within that filtered set
3. Return top 5 semantic matches from filtered results

**Root cause:** Storage backend `retrieve()` method only accepts `query` and `n_results` parameters. The `tags`, `memory_type`, and `min_similarity` parameters are accepted by `memory_service.py` but not passed to storage backends, forcing post-hoc filtering in Python.

**Evidence:**
- `memory_service.py:255-278` - Post-hoc Python filtering
- `base.py:95` - Abstract method lacks filter parameters
- `sqlite_vec.py:786` - Implementation lacks WHERE clauses
- `cloudflare.py:448` - Implementation lacks filter parameter

## Success Criteria

**Phase 1: Investigation & Validation**
- [ ] Determine if post-hoc filtering is intentional design or actual bug
- [ ] Check git history to understand who added the filtering and why
- [ ] Test actual behavior with filters to validate the problem exists
- [ ] Document findings: Is this a bug, design trade-off, or working as intended?

**Phase 2: Fix Implementation (IF bug confirmed)**
- [ ] Vector search returns exactly `n_results` memories when filters are applied (or fewer if total matching is less)
- [ ] Tag filtering happens at database/vector layer (pre-filtering), not in Python
- [ ] Memory type filtering happens at database/vector layer
- [ ] Similarity threshold filtering happens at database/vector layer

**Phase 2: Backend Implementation (IF bug confirmed)**
- [ ] `MemoryStorage.retrieve()` signature updated to accept `tags`, `memory_type`, `min_similarity` parameters
- [ ] SQLite-vec backend implements filtering via SQL WHERE clauses
- [ ] Cloudflare backend implements filtering via Vectorize API filter parameter (if supported)
- [ ] Hybrid backend inherits filtering from primary storage
- [ ] HTTP client backend supports filtering parameters
- [ ] Post-hoc Python filtering code removed from `memory_service.py:256-285`

**Phase 2: Testing (IF bug confirmed)**
- [ ] Tests verify correct result count with filters applied
- [ ] Tests verify filtering happens before vector search (semantic ordering preserved)
- [ ] Tests cover all backends: SQLite-vec, Cloudflare, Hybrid

## Context Manifest

### How Vector Search with Filtering Currently Works

**Entry Point: User Request → Memory Service → Storage Backend**

When a user requests memories using semantic search with filters (e.g., "Find 5 memories about Python with tag 'important'"), the request flows through these layers:

1. **MCP Tool Handler** (`mcp_server.py:184-207`): The `retrieve_memory()` tool accepts user parameters including `query`, `n_results`, and `min_similarity`. However, it does NOT accept `tags` or `memory_type` parameters - these are missing from the MCP tool signature.

2. **Memory Service Layer** (`memory_service.py:234-311`): The `retrieve_memories()` method accepts all filter parameters (`query`, `n_results`, `tags`, `memory_type`, `min_similarity`) BUT does not pass them to the storage backend. Instead:
   - Lines 256-261: Calls `storage.retrieve()` with ONLY `query` and `n_results`
   - Lines 263-285: Performs **post-hoc Python filtering** on the returned results
   - This means if you request 5 results with a tag filter, the system:
     1. Gets top 5 semantically similar memories (ignoring tags)
     2. Filters those 5 for the tag in Python
     3. Returns 0-5 results depending on how many match

3. **Storage Backend Interface** (`base.py:95-97`): The abstract `retrieve()` method signature is:
   ```python
   async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
   ```
   It **does not include** `tags`, `memory_type`, or `min_similarity` parameters, so backends cannot perform pre-filtering.

4. **Concrete Storage Implementations**:
   - **SQLite-vec** (`sqlite_vec.py:786-887`): Retrieves via vector similarity search using `vec0` virtual table with cosine distance. The SQL query joins `memories` and `memory_embeddings` tables but has NO `WHERE` clauses for tag/type filtering.
   - **Cloudflare** (`cloudflare.py:448-486`): Uses Vectorize API for semantic search with `topK` parameter but does not utilize Vectorize's metadata filtering capabilities.
   - **Hybrid** (`hybrid.py:810-816`): Delegates to primary (SQLite-vec) storage, inheriting the same limitation.

**Why This Architecture Breaks Semantic Search**

Vector similarity search finds the N most semantically similar items from the ENTIRE dataset. When you filter AFTER retrieval:
- If you have 10,000 memories total
- And request "5 memories with tag 'important'"
- The system returns the 5 most similar memories from all 10,000
- Then filters those 5 for 'important' tag
- You might get 0 results even though 100 'important' memories exist

The correct approach is:
1. Filter to memories with tag 'important' (e.g., 100 memories)
2. Run vector search on ONLY those 100 memories
3. Return top 5 semantic matches from the filtered set

### Historical Context: When and Why This Was Implemented

**Git Blame Analysis** reveals this is NOT yet committed work (lines 256-260 show "Not Committed Yet 2025-10-30"):
- The comment "Storage backends only accept query and n_results" was added during current development
- The original MemoryService refactor (commit `36e98453` on 2025-10-28) intended to pass filters through to storage
- The post-hoc filtering appears to be a **temporary workaround** because the storage backends don't yet support filter parameters

**Related Commits**:
- `ec81ef57`: DRY refactor that created MemoryService (2025-10-26)
- `36e98453`: MemoryService architecture with tests (2025-10-28)
- `f935c56`: Added `tags` parameter to `count_all_memories` (suggesting awareness of filtering needs)

This suggests the post-hoc filtering is a **known issue**, not intentional design, awaiting proper backend implementation.

### How Database-Level Filtering Should Work

**SQLite-vec Example** (based on existing patterns in the codebase):

The codebase already demonstrates correct filtering patterns in other methods:

1. **Tag Filtering** (`sqlite_vec.py:899-940`):
   ```python
   tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
   tag_params = [f"%{tag}%" for tag in tags]

   cursor = self.conn.execute(f'''
       SELECT ... FROM memories
       WHERE {tag_conditions}
       ORDER BY created_at DESC
   ''', tag_params)
   ```

2. **Type + Tag Filtering** (`sqlite_vec.py:1801-1866` in `get_all_memories()`):
   ```python
   where_conditions = []
   if memory_type is not None:
       where_conditions.append('m.memory_type = ?')
   if tags:
       tag_conditions = " OR ".join(["m.tags LIKE ?" for _ in tags])
       where_conditions.append(f"({tag_conditions})")

   if where_conditions:
       query += ' WHERE ' + ' AND '.join(where_conditions)
   ```

3. **Vector Search Pattern** (`sqlite_vec.py:812-840`):
   ```python
   cursor = self.conn.execute('''
       SELECT m.content_hash, m.content, m.tags, m.memory_type, ...
       FROM memories m
       INNER JOIN (
           SELECT rowid, distance
           FROM memory_embeddings
           WHERE content_embedding MATCH ?
           ORDER BY distance
           LIMIT ?
       ) e ON m.id = e.rowid
       ORDER BY e.distance
   ''', (serialize_float32(query_embedding), n_results))
   ```

**The Fix Pattern**: Combine these approaches - add WHERE clause to filter memories BEFORE the vector search join:

```python
# Proposed fix for sqlite_vec.py retrieve() method
def build_filter_clause(tags, memory_type):
    conditions = []
    params = []
    if memory_type:
        conditions.append('m.memory_type = ?')
        params.append(memory_type)
    if tags:
        tag_conditions = " OR ".join(["m.tags LIKE ?" for _ in tags])
        conditions.append(f"({tag_conditions})")
        params.extend([f"%{tag}%" for tag in tags])
    return conditions, params

# In retrieve() method:
filter_conditions, filter_params = build_filter_clause(tags, memory_type)

base_query = '''
    SELECT m.content_hash, m.content, m.tags, m.memory_type, ...
    FROM memories m
    INNER JOIN (...) e ON m.id = e.rowid
'''
if filter_conditions:
    base_query = base_query.replace(
        'FROM memories m',
        f'FROM memories m WHERE {" AND ".join(filter_conditions)}'
    )

cursor = self.conn.execute(base_query, filter_params + [embedding, n_results])
```

**Cloudflare Backend**: Vectorize API supports metadata filtering via the `filter` parameter in queries. From Cloudflare docs, the query payload should include:

```python
search_payload = {
    "vector": query_embedding,
    "topK": n_results,
    "returnMetadata": "all",
    "returnValues": False,
    "filter": {  # Add this
        "memory_type": {"$eq": memory_type},  # If specified
        "tags": {"$in": tags}  # If specified
    }
}
```

### Test Coverage Analysis

**Existing Tests for Retrieve/Search** (`tests/unit/test_memory_service.py:289-340`):

```python
@pytest.mark.asyncio
async def test_retrieve_memories_basic(memory_service, mock_storage, sample_memories):
    """Test basic semantic search retrieval."""
    mock_storage.retrieve.return_value = sample_memories[:3]

    result = await memory_service.retrieve_memories(query="test query", n_results=3)

    mock_storage.retrieve.assert_called_once_with(
        query="test query",
        n_results=3,
        tags=None,
        memory_type=None
    )
```

**The test EXPECTS filters to be passed to storage!** This confirms the current implementation doesn't match the intended design. The test passes `tags` and `memory_type` to `storage.retrieve()`, but the actual implementation (memory_service.py:258-260) does NOT.

**Missing Test Coverage**:
- No tests verify that filtering happens at database level (pre-filtering)
- No tests check result count when filters reduce available memories
- No tests verify semantic ordering is preserved within filtered set
- No integration tests for backend filter implementation

### API Surface Exposed to Users

**MCP Tool Interface** (`mcp_server.py:184-207`):

```python
@mcp.tool()
async def retrieve_memory(
    query: str,
    ctx: Context,
    n_results: int = 5,
    min_similarity: float = 0.0
) -> Dict[str, Any]:
```

**Critical Issue**: The MCP tool does NOT expose `tags` or `memory_type` parameters to users! Even if we fix the backend filtering, users cannot currently request filtered searches via MCP tools.

**HTTP API Interface** (from `server.py` - not shown but referenced in memory_service tests):
The HTTP endpoints likely expose full filtering via query parameters/request body, but the MCP tool interface is missing these.

### Technical Reference Details

#### Storage Backend Method Signatures

**Current (Broken)**:
```python
# base.py:95
async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]
```

**Proposed (Fixed)**:
```python
# base.py:95
async def retrieve(
    self,
    query: str,
    n_results: int = 5,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    min_similarity: Optional[float] = None
) -> List[MemoryQueryResult]
```

#### Data Model Structures

**MemoryQueryResult** (`models/memory.py`):
```python
@dataclass
class MemoryQueryResult:
    memory: Memory
    relevance_score: float
    debug_info: Optional[Dict[str, Any]] = None
```

**Memory** (`models/memory.py`):
```python
@dataclass
class Memory:
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: float
    updated_at: float
    created_at_iso: str
    updated_at_iso: str
```

#### Database Schema (SQLite-vec)

**Tables**:
- `memories`: Stores metadata (content_hash, content, tags, memory_type, created_at, etc.)
- `memory_embeddings`: Virtual vec0 table (rowid, content_embedding FLOAT[384])
- Join condition: `memories.id = memory_embeddings.rowid`

**Tags Storage Format**: Comma-separated string (e.g., "important,python,development")
**Filtering Pattern**: Uses `LIKE` with wildcards (e.g., `tags LIKE '%important%'`)

#### File Locations

**Implementation Files**:
- Base interface: `src/mcp_memory_service/storage/base.py` (line 95)
- Memory service: `src/mcp_memory_service/services/memory_service.py` (lines 234-311)
- SQLite backend: `src/mcp_memory_service/storage/sqlite_vec.py` (lines 786-887)
- Cloudflare backend: `src/mcp_memory_service/storage/cloudflare.py` (lines 448-486)
- Hybrid backend: `src/mcp_memory_service/storage/hybrid.py` (lines 810-816)

**MCP Tool Handlers**:
- MCP server: `src/mcp_memory_service/mcp_server.py` (lines 184-207)

**Test Files**:
- Memory service tests: `tests/unit/test_memory_service.py` (lines 289-340)
- SQLite tests: `tests/test_sqlite_vec_storage.py`
- Integration tests: `tests/test_memory_ops.py`

**Configuration**:
- Storage config: `src/mcp_memory_service/config.py`

## User Notes
<!-- Any specific notes or requirements from the developer -->

## Work Log
<!-- Updated as work progresses -->

### 2025-10-30 - Phase 1: Validation Complete ✅

**Bug Confirmed:** Post-hoc filtering is NOT intentional design - it's a temporary workaround.

**Evidence:**

1. **Staged Changes Analysis** (`memory_service.py`):
   - BEFORE: Code passed `tags` and `memory_type` to `storage.retrieve()`
   - AFTER: Added comment "Storage backends only accept query and n_results"
   - AFTER: Implemented post-hoc Python filtering (lines 256-285)
   - **Conclusion**: Someone started proper implementation, then changed to workaround

2. **Unit Test Expectations** (`test_memory_service.py:302-327`):
   - Tests EXPECT `storage.retrieve()` to be called with `tags` and `memory_type` parameters
   - Test assertions explicitly check: `mock_storage.retrieve.assert_called_once_with(query="test", n_results=5, tags=["tag1"], memory_type="note")`
   - **Conclusion**: Current WIP code breaks these tests

3. **Storage Backend Reality** (`base.py:95`):
   - Abstract method signature: `async def retrieve(self, query: str, n_results: int = 5)`
   - NO parameters for `tags`, `memory_type`, or `min_similarity`
   - **Conclusion**: Backend interface doesn't support filtering yet

**Root Cause:**
The architecture was designed to support database-level filtering (tests prove this), but storage backend implementations were never updated to accept filter parameters. The post-hoc filtering is a stopgap that defeats semantic search.

**Why This Matters:**
- **Correctness**: Requesting "5 memories with tag X" might return 0-5 results instead of exactly 5
- **Performance**: Fetches unnecessary data from database/network
- **Semantics**: Filters after vector search instead of searching within filtered set

**Decision:** Proceed to Phase 2 - Fix the architecture

### 2025-11-01 - Task Completed ✅

**Database-Level Filtering Implemented** (Commit: ee1cac5)

**What Was Achieved:**
- Database-level filtering successfully implemented in all storage backends
- Python post-filtering removed from MemoryService
- All 6 affected methods updated with exact tag matching
- Comprehensive test coverage added (47/47 tests passing)
- Cloudflare type mismatch fixed (string → array format)

**Technical Implementation:**
- SQLite-vec: WHERE clauses with comma-boundary matching `',' || tags || ',' LIKE '%,python,%'`
- Cloudflare: Vectorize metadata filter with `$in` operator on array tags
- Hybrid: Inherits filtering from primary storage
- HTTP client: Filter parameters passed to API

**Known Limitations (Deferred to h-fix-tag-filtering-performance-migration.md):**
- Index not actually used (expression + leading wildcard prevents index usage)
- Still O(n) table scans despite O(log n) claim
- Cloudflare migration missing (breaking change for existing data)
- Test coverage gaps (tests check index exists, not if it's used)

**Outcome:**
- **Functionally correct**: Filtering happens at database level, not Python
- **Not production-optimized**: Performance issues require normalized tag storage (next task)
- **Tests passing**: All success criteria met for database-level filtering goal
