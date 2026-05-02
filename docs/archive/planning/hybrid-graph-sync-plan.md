# Hybrid Backend Graph Sync Plan (v9.3.0)

**Status:** Planning Phase
**Priority:** Medium
**Complexity:** High
**Estimated Effort:** 3-5 days

## Overview

Add `memory_graph` table synchronization to the hybrid backend, enabling:
- Cloud backup of graph relationships
- Multi-device sync of knowledge graph
- Redundancy for graph data
- Future: Real-time collaborative knowledge graphs

## Current Architecture

### What Works Now
- ✅ SQLite GraphStorage with typed relationships (6 types)
- ✅ Local graph visualization with 2,730 relationships
- ✅ Hybrid backend syncs `memories` table only
- ✅ Cloudflare D1 migration script ready (commit eccd272)

### What's Missing
- ❌ `memory_graph` table doesn't exist in Cloudflare D1
- ❌ No sync mechanism for graph relationships
- ❌ Graph changes not propagated to cloud
- ❌ Multi-device users can't sync their knowledge graphs

## Technical Analysis

### Current Sync Architecture (Memories Only)

**File:** `src/mcp_memory_service/storage/hybrid.py`

**Key Components:**
1. **SyncOperation** - Represents pending sync operations
2. **_sync_loop()** - Background task processing sync queue
3. **force_sync()** - Manual sync trigger with batch processing
4. **_periodic_sync()** - Periodic cloud sync
5. **_detect_and_sync_drift()** - Drift detection for convergence

**Sync Flow:**
```
SQLite (Primary) → Queue → Background Loop → Cloudflare (Secondary)
   ↓
Fast Reads (5ms)
```

### Graph Storage Architecture

**File:** `src/mcp_memory_service/storage/graph.py`

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS memory_graph (
    source_hash TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    similarity REAL NOT NULL,
    connection_types TEXT NOT NULL,  -- JSON array
    metadata TEXT,                    -- JSON object
    created_at REAL NOT NULL,
    relationship_type TEXT DEFAULT 'related',  -- NEW in v9.0.0
    PRIMARY KEY (source_hash, target_hash)
)
```

**Key Methods:**
- `store_association()` - Stores relationship with symmetric/asymmetric logic
- `get_associations()` - Retrieves all relationships for a memory
- `find_connected()` - Multi-hop BFS traversal
- `find_shortest_path()` - Shortest path between two memories
- `get_memory_subgraph()` - Subgraph extraction for visualization

**Relationship Types:**
- **Symmetric (bidirectional):** related, contradicts
- **Asymmetric (directed):** causes, fixes, supports, follows

## Implementation Plan

### Phase 1: Cloudflare D1 Schema Setup

**Goal:** Create `memory_graph` table in Cloudflare D1

**Tasks:**
1. Update `cloudflare.py` to include `memory_graph` schema in `_initialize_d1_schema()`
2. Add `relationship_type` column with DEFAULT 'related'
3. Create indexes for performance:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_memory_graph_source ON memory_graph(source_hash)
   CREATE INDEX IF NOT EXISTS idx_memory_graph_target ON memory_graph(target_hash)
   CREATE INDEX IF NOT EXISTS idx_memory_graph_type ON memory_graph(relationship_type)
   ```
4. Run Cloudflare D1 migration script (already created in commit eccd272)

**Affected Files:**
- `src/mcp_memory_service/storage/cloudflare.py` (add schema)
- `scripts/migration/add_relationship_type_column_cloudflare.py` (fix and run)

**Estimated Time:** 1 day

### Phase 2: Graph Sync Operations

**Goal:** Add graph-specific sync operations to HybridStorage

**Tasks:**
1. Extend `SyncOperation` dataclass with graph operation types:
   ```python
   @dataclass
   class GraphSyncOperation:
       operation: str  # "store_association" | "delete_association"
       source_hash: str
       target_hash: str
       similarity: float
       connection_types: List[str]
       metadata: Optional[Dict[str, Any]]
       created_at: float
       relationship_type: str
       timestamp: float
       retry_count: int = 0
   ```

2. Add graph sync queue: `self._graph_sync_queue: deque[GraphSyncOperation] = deque()`

3. Implement graph sync methods in `hybrid.py`:
   ```python
   async def _sync_graph_operation(self, operation: GraphSyncOperation) -> bool:
       """Sync a single graph operation to Cloudflare D1."""

   async def _sync_graph_batch(self, batch: List[GraphSyncOperation]) -> Dict[str, Any]:
       """Batch sync graph operations to Cloudflare D1."""

   async def force_graph_sync(self) -> Dict[str, Any]:
       """Force sync all pending graph operations."""
   ```

4. Integrate graph sync into existing `_sync_loop()`:
   - Check both `_sync_queue` and `_graph_sync_queue`
   - Process in batches (100 operations per batch)
   - Handle errors with exponential backoff

**Affected Files:**
- `src/mcp_memory_service/storage/hybrid.py` (+300 lines)
- `src/mcp_memory_service/storage/cloudflare.py` (+150 lines for graph operations)

**Estimated Time:** 2 days

### Phase 3: GraphStorage Integration

**Goal:** Make GraphStorage aware of hybrid backend for automatic queuing

**Options:**

**Option A: Wrapper Pattern (Recommended)**
- Create `HybridGraphStorage` that wraps `GraphStorage`
- Intercepts `store_association()` and `delete_association()` calls
- Queues operations to `HybridStorage._graph_sync_queue`
- Delegates actual storage to SQLite `GraphStorage`

**Option B: Direct Integration**
- Modify `GraphStorage` to accept optional `sync_callback`
- `HybridStorage` passes callback during initialization
- GraphStorage calls callback after local operations

**Option C: Event-Based**
- GraphStorage emits events after operations
- HybridStorage listens and queues sync operations
- Loosely coupled but more complex

**Recommended: Option A - Wrapper Pattern**

```python
# src/mcp_memory_service/storage/hybrid_graph.py (NEW)

class HybridGraphStorage:
    """Hybrid graph storage with automatic cloud sync."""

    def __init__(self, local_graph: GraphStorage, hybrid_storage: HybridStorage):
        self.local = local_graph
        self.hybrid = hybrid_storage

    async def store_association(
        self,
        source_hash: str,
        target_hash: str,
        similarity: float,
        connection_types: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        relationship_type: str = "related"
    ) -> bool:
        # Store locally first (fast)
        success = await self.local.store_association(...)

        if success:
            # Queue for cloud sync (background)
            await self.hybrid.queue_graph_sync(
                GraphSyncOperation(...)
            )

        return success

    # Delegate all read operations directly to local
    async def get_associations(self, *args, **kwargs):
        return await self.local.get_associations(*args, **kwargs)

    async def find_connected(self, *args, **kwargs):
        return await self.local.find_connected(*args, **kwargs)

    # ... other methods
```

**Affected Files:**
- `src/mcp_memory_service/storage/hybrid_graph.py` (NEW, ~400 lines)
- `src/mcp_memory_service/storage/hybrid.py` (+50 lines for queue_graph_sync)
- `src/mcp_memory_service/server/handlers/graph.py` (update to use HybridGraphStorage)

**Estimated Time:** 1.5 days

### Phase 4: Initial Graph Sync

**Goal:** Sync existing 2,730 relationships from SQLite to Cloudflare D1

**Tasks:**
1. Add `_perform_initial_graph_sync()` method to `HybridStorage`
2. Batch read from SQLite `memory_graph` table (1000 rows per batch)
3. Batch write to Cloudflare D1 with progress tracking
4. Handle errors and resume from last successful batch
5. Add progress monitoring via SSE events

**Implementation:**
```python
async def _perform_initial_graph_sync(self) -> Dict[str, Any]:
    """One-time sync of all graph relationships from SQLite to Cloudflare D1."""

    # 1. Count total relationships
    graph = GraphStorage(self.primary.db_path)
    total = await graph.count_total_associations()

    # 2. Batch process (1000 per batch to avoid D1 limits)
    batch_size = 1000
    synced = 0

    for offset in range(0, total, batch_size):
        batch = await graph.get_all_associations(limit=batch_size, offset=offset)

        # 3. Bulk insert to Cloudflare D1
        await self._sync_graph_batch_to_cloudflare(batch)

        synced += len(batch)

        # 4. Progress event via SSE
        if SSE_AVAILABLE:
            await sse_manager.broadcast(
                create_graph_sync_progress_event(synced, total)
            )

    return {"synced": synced, "total": total}
```

**Affected Files:**
- `src/mcp_memory_service/storage/hybrid.py` (+150 lines)
- `src/mcp_memory_service/storage/graph.py` (+50 lines for batch retrieval)

**Estimated Time:** 0.5 days

### Phase 5: Conflict Resolution

**Goal:** Handle conflicts when same relationship is modified on multiple devices

**Strategy: Last-Write-Wins (LWW) with `created_at` timestamp**

**Conflict Scenarios:**
1. **Same relationship, different metadata** → Use latest `created_at`
2. **Symmetric relationship added from both ends** → Already handled (A→B and B→A are equivalent)
3. **Relationship deleted on one device, modified on another** → Deletion wins (safer)

**Implementation:**
```python
async def _resolve_graph_conflict(
    self,
    local: GraphAssociation,
    remote: GraphAssociation
) -> GraphAssociation:
    """Resolve conflict using Last-Write-Wins strategy."""

    if local.created_at > remote.created_at:
        # Local is newer, sync to cloud
        await self._sync_graph_to_cloudflare(local)
        return local
    else:
        # Remote is newer, update local
        await self._sync_graph_from_cloudflare(remote)
        return remote
```

**Affected Files:**
- `src/mcp_memory_service/storage/hybrid.py` (+100 lines)

**Estimated Time:** 0.5 days

### Phase 6: Testing & Validation

**Goal:** Comprehensive test coverage for graph sync

**Test Categories:**

1. **Unit Tests** (`tests/storage/test_hybrid_graph_sync.py`)
   - Queue operations correctly
   - Batch processing works
   - Error handling with retries
   - Conflict resolution

2. **Integration Tests** (`tests/integration/test_hybrid_graph_e2e.py`)
   - Store relationship → appears in Cloudflare
   - Multi-device sync scenario
   - Large graph sync (1000+ relationships)
   - Drift detection and convergence

3. **Performance Tests** (`tests/performance/test_graph_sync_performance.py`)
   - Batch size optimization
   - Sync latency measurement
   - Memory usage during initial sync

**Affected Files:**
- `tests/storage/test_hybrid_graph_sync.py` (NEW, ~500 lines)
- `tests/integration/test_hybrid_graph_e2e.py` (NEW, ~300 lines)
- `tests/performance/test_graph_sync_performance.py` (NEW, ~200 lines)

**Estimated Time:** 1 day

### Phase 7: Documentation & Migration Guide

**Goal:** Complete documentation for users upgrading to v9.3.0

**Tasks:**
1. Update `README.md` with hybrid graph sync features
2. Create migration guide: `docs/migration/v9.2-to-v9.3.md`
3. Update `CHANGELOG.md` with breaking changes (if any)
4. Add configuration options to `.env.example`:
   ```bash
   # Graph Sync Configuration (v9.3.0+)
   HYBRID_GRAPH_SYNC_ENABLED=true
   HYBRID_GRAPH_SYNC_BATCH_SIZE=1000
   HYBRID_GRAPH_SYNC_ON_STARTUP=true
   ```
5. Create troubleshooting guide: `docs/troubleshooting/graph-sync.md`

**Affected Files:**
- `README.md` (+50 lines)
- `docs/migration/v9.2-to-v9.3.md` (NEW, ~200 lines)
- `CHANGELOG.md` (+30 lines)
- `.env.example` (+10 lines)
- `docs/troubleshooting/graph-sync.md` (NEW, ~150 lines)

**Estimated Time:** 0.5 days

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Memory Service                      │
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  GraphStorage    │         │ HybridGraphStorage│        │
│  │  (SQLite-only)   │◄────────│   (Wrapper)       │        │
│  │                  │         │                   │        │
│  │  - store_assoc() │         │  - Delegates reads│        │
│  │  - get_assoc()   │         │  - Queues writes  │        │
│  │  - find_conn()   │         └────────┬──────────┘        │
│  └──────────────────┘                  │                   │
│           │                             │                   │
│           │                             ▼                   │
│           │                  ┌──────────────────┐           │
│           ▼                  │ HybridStorage    │           │
│  ┌──────────────────┐        │                  │           │
│  │ SQLite Database  │        │ - _graph_sync_   │           │
│  │                  │        │   queue          │           │
│  │ - memories       │        │ - _sync_loop()   │           │
│  │ - memory_graph   │◄───────│ - force_graph_   │           │
│  │                  │        │   sync()         │           │
│  └──────────────────┘        └────────┬─────────┘           │
│                                       │                     │
│                                       │ Background Sync     │
│                                       ▼                     │
│                            ┌──────────────────┐             │
│                            │ Cloudflare D1    │             │
│                            │                  │             │
│                            │ - memories       │             │
│                            │ - memory_graph   │             │
│                            │                  │             │
│                            └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘

Sync Flow:
1. User creates relationship → HybridGraphStorage
2. Store locally (SQLite) - FAST (5ms)
3. Queue to HybridStorage._graph_sync_queue
4. Background loop processes queue → Cloudflare D1
5. Retry on failure with exponential backoff
```

## Risk Assessment

### High Risk
- **Data Consistency:** Relationship added locally but fails to sync to cloud
  - **Mitigation:** Persistent queue, exponential backoff, manual force_graph_sync()

- **D1 Limits:** Cloudflare D1 has 10GB limit, could hit with large graphs
  - **Mitigation:** Monitor storage usage, add warnings at 80%, block at 95%

### Medium Risk
- **Initial Sync Time:** 2,730 relationships × 2 (symmetric) = 5,460 rows to sync
  - **Mitigation:** Batch processing (1000 per batch), progress tracking via SSE

- **Symmetric Relationship Storage:** A→B and B→A stored as separate rows
  - **Mitigation:** De-duplicate on read, or store only A→B where A < B lexicographically

### Low Risk
- **Backward Compatibility:** Existing users without Cloudflare credentials
  - **Mitigation:** Graph sync only enabled if Cloudflare credentials present

## Configuration Options

**New Environment Variables (v9.3.0):**

```bash
# Enable/disable graph sync (default: true if Cloudflare credentials present)
HYBRID_GRAPH_SYNC_ENABLED=true

# Batch size for initial sync (default: 1000)
HYBRID_GRAPH_SYNC_BATCH_SIZE=1000

# Sync existing relationships on startup (default: false to avoid blocking)
HYBRID_GRAPH_SYNC_ON_STARTUP=false

# Max retry attempts for failed sync operations (default: 5)
HYBRID_GRAPH_SYNC_MAX_RETRIES=5

# Retry backoff multiplier (default: 2.0)
HYBRID_GRAPH_SYNC_BACKOFF_MULTIPLIER=2.0
```

## Success Criteria

1. ✅ `memory_graph` table created in Cloudflare D1 with correct schema
2. ✅ New graph relationships automatically sync to cloud (< 5 seconds)
3. ✅ Existing 2,730 relationships sync via initial sync command
4. ✅ Multi-device scenario: Add relationship on Device A → appears on Device B
5. ✅ Error recovery: Network failure → queue persists → auto-retry on reconnect
6. ✅ Performance: Graph operations remain fast (< 10ms local, background cloud sync)
7. ✅ 100% test coverage for all graph sync operations
8. ✅ Documentation complete with migration guide and troubleshooting

## Alternative Approaches Considered

### A. Event-Based Sync (Observer Pattern)
**Pros:** Loosely coupled, easy to extend
**Cons:** More complex, harder to debug, event bus overhead
**Decision:** Rejected - Wrapper pattern simpler and more maintainable

### B. Direct GraphStorage Modification
**Pros:** No wrapper, single class
**Cons:** Tightly coupled, breaks separation of concerns
**Decision:** Rejected - Violates single responsibility principle

### C. Polling-Based Sync
**Pros:** Simple implementation
**Cons:** Inefficient, high latency, resource wasteful
**Decision:** Rejected - Queue-based approach is superior

## Timeline

**Total Estimated Time:** 7 days

| Phase | Days | Status |
|-------|------|--------|
| 1. D1 Schema Setup | 1.0 | ⏳ Pending |
| 2. Graph Sync Operations | 2.0 | ⏳ Pending |
| 3. GraphStorage Integration | 1.5 | ⏳ Pending |
| 4. Initial Graph Sync | 0.5 | ⏳ Pending |
| 5. Conflict Resolution | 0.5 | ⏳ Pending |
| 6. Testing & Validation | 1.0 | ⏳ Pending |
| 7. Documentation | 0.5 | ⏳ Pending |

## Next Steps

1. **Review this plan** with stakeholders
2. **Create GitHub issue** for tracking (v9.3.0 milestone)
3. **Set up feature branch:** `feature/hybrid-graph-sync`
4. **Begin Phase 1** - Cloudflare D1 schema setup
5. **Incremental commits** with TCR workflow

## Questions for Discussion

1. Should initial graph sync run automatically on startup or require manual trigger?
   - **Recommendation:** Manual trigger via `/api/sync/graph/initial` to avoid blocking startup

2. How to handle very large graphs (100k+ relationships)?
   - **Recommendation:** Pagination with resume capability, show progress in dashboard

3. Should we sync all 6 relationship types or start with subset?
   - **Recommendation:** Sync all types - schema already supports it

4. Future: Should we support pull sync (Cloudflare → SQLite)?
   - **Recommendation:** Yes, but Phase 2 of v9.4.0 - bidirectional sync

5. Should symmetric relationships be deduplicated in storage?
   - **Recommendation:** No, keep current approach (A→B and B→A) for simplicity

---

**Document Version:** 1.0
**Last Updated:** 2026-01-18
**Author:** Claude Sonnet 4.5
**Status:** Ready for Review
