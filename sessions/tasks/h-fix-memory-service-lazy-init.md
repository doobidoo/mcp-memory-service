---
name: h-fix-memory-service-lazy-init
branch: fix/memory-service-lazy-init
status: pending
created: 2025-11-05
---

# Fix Missing MemoryService Initialization in Lazy Loading Path

## Problem/Goal

Critical bug: When storage backend initialization happens via lazy loading (after eager init timeout/failure), the `_ensure_storage_initialized()` method successfully creates `self.storage` but fails to create `self.memory_service`. This causes all memory operations to fail with:

```
Error storing memory: 'NoneType' object has no attribute 'store_memory'
```

**Root Cause**: Lines 689-695 in `server.py` initialize storage but never call `self.memory_service = MemoryService(self.storage)`, unlike the eager init path at line 540.

**Impact**: Any user whose MCP server falls back to lazy loading (common when embedding model loading exceeds timeout) experiences complete memory system failure.

## Success Criteria
- [ ] Add `self.memory_service = MemoryService(self.storage)` initialization after line 692 in lazy loading path
- [ ] Verify eager init path remains unchanged and functional
- [ ] Confirm error handling paths still properly set both `storage` and `memory_service` to None
- [ ] Test that `handle_store_memory` successfully calls `self.memory_service.store_memory()` after lazy init
- [ ] Verify no regression in existing functionality (all tests pass)

## Context Manifest

### How Server Initialization Currently Works

The MCP Memory Service has a two-phase initialization strategy designed to prevent the server from hanging during startup due to expensive embedding model loading. Understanding this flow is critical to fixing the bug.

#### Phase 1: Eager Initialization with Timeout (Preferred Path)

When the server starts, the `initialize()` method at line 707 attempts **eager initialization** with a timeout. This is the "happy path" that tries to load everything during startup:

1. **Server Creation** (`__init__`, lines 292-375): Creates the `MemoryServer` instance and **defers storage initialization** to prevent hanging. At lines 346-348, both `self.storage` and `self.memory_service` are set to `None`, with `self._storage_initialized = False`.

2. **Async Initialize** (`initialize()`, line 707): Called after server creation, attempts eager initialization with a configurable timeout (default determined by `get_recommended_timeout()`).

3. **Eager Init** (`_initialize_storage_with_timeout()`, lines 422-555): This is the **complete, working initialization path**:
   - Lines 429-529: Creates the appropriate storage backend instance based on `STORAGE_BACKEND`:
     - `sqlite_vec`: Creates `SqliteVecMemoryStorage` with coordination mode detection for multi-client scenarios
     - `cloudflare`: Creates `CloudflareStorage` with full Cloudflare credentials
     - `hybrid`: Creates `HybridMemoryStorage` with both SQLite and Cloudflare config
   - Line 533: Calls `await self.storage.initialize()` to initialize the backend
   - Line 536: Sets `self._storage_initialized = True`
   - **Line 540: CRITICAL - Creates `self.memory_service = MemoryService(self.storage)`**
   - Line 541: Logs successful MemoryService initialization
   - Lines 548-549: Initializes consolidation system if enabled

4. **Success or Timeout** (lines 732-756): If eager init succeeds or times out:
   - Success: Server is fully operational with both `storage` and `memory_service` initialized
   - Timeout/Failure: Lines 748, 755, 764 reset both `self.storage = None` and `self.memory_service = None`, falling back to lazy loading

#### Phase 2: Lazy Initialization (Fallback Path - THE BUG LOCATION)

When eager initialization fails or times out, the server defers initialization until the first memory operation. This happens when any handler calls `await self._ensure_storage_initialized()`.

**The Lazy Loading Flow** (`_ensure_storage_initialized()`, lines 557-705):

1. **Check Flag** (line 559): If `self._storage_initialized` is False, proceed with initialization
2. **Backend Creation** (lines 574-674): **Identical storage backend creation logic to eager init**:
   - SQLite-vec: Lines 574-611 (with coordination mode detection)
   - Cloudflare: Lines 612-628 (with full Cloudflare config)
   - Hybrid: Lines 629-660 (with SQLite + Cloudflare config)
3. **Storage Initialize** (line 678): Calls `await self.storage.initialize()`
4. **Verification** (lines 682-687): Checks if storage is properly initialized
5. **Set Flag** (lines 689-692):
   - Line 689: Sets `self._storage_initialized = True`
   - Line 690: Gets storage type for logging
   - Lines 691-692: Logs success
   - **MISSING: No `self.memory_service = MemoryService(self.storage)` call!**
6. **Consolidation** (line 695): Initializes consolidation system if enabled
7. **Error Handling** (lines 697-704): On failure, sets `self.storage = None` and `self._storage_initialized = False` at line 702, **but never touches `self.memory_service`**

**The Critical Bug**: After line 692, the lazy init path never creates the `MemoryService` instance, unlike the eager init path at line 540. This means `self.memory_service` remains `None`, causing all memory operations to fail with `AttributeError: 'NoneType' object has no attribute 'store_memory'`.

### Why This Causes Complete Memory System Failure

#### The MemoryService Dependency Chain

The `MemoryService` class (defined at `src/mcp_memory_service/services/memory_service.py`, line 40) is the **centralized business logic layer** introduced in v8.12.0 to eliminate code duplication. It wraps the storage backend and provides consistent behavior across all memory operations:

```python
class MemoryService:
    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def store_memory(self, content, tags, memory_type, metadata, client_hostname):
        # Handles content splitting, deduplication, validation, etc.
        ...

    async def retrieve_memories(self, query, n_results):
        # Handles search, scoring, formatting
        ...
```

**Every MCP tool handler depends on MemoryService**, not directly on storage:

- `handle_store_memory()` (line 1683): Calls `await self.memory_service.store_memory()` at line 1700
- `handle_retrieve_memory()` (line 1723): Calls `await self.memory_service.retrieve_memories()` at line 1738
- `handle_search_by_tag()` (line 1793): Calls `await self.memory_service.search_by_tag()` at line 1806
- Other handlers follow the same pattern

**When MemoryService is None**, attempting to call `self.memory_service.store_memory()` raises:
```
AttributeError: 'NoneType' object has no attribute 'store_memory'
```

This affects **every user whose server falls back to lazy loading** - typically when:
- Embedding model loading exceeds the timeout (common on first run or slow systems)
- Network issues delay Cloudflare backend initialization
- System is under heavy load during startup

### The Fix: Mirror Eager Init's MemoryService Creation

The fix is straightforward - add the same MemoryService initialization that exists in the eager path (line 540) to the lazy path after line 692.

**Eager Init Pattern** (lines 539-541):
```python
# Initialize MemoryService with shared business logic
self.memory_service = MemoryService(self.storage)
logger.info(f"✅ EAGER INIT: MemoryService initialized with {STORAGE_BACKEND} storage")
```

**Required Change in Lazy Init** (after line 692):
```python
# Initialize MemoryService with shared business logic
self.memory_service = MemoryService(self.storage)
logger.info(f"✅ LAZY INIT: MemoryService initialized with {STORAGE_BACKEND} storage")
```

This ensures that regardless of which initialization path is taken (eager or lazy), both `self.storage` and `self.memory_service` are properly initialized before any memory operations occur.

### Error Handling Consistency

Both initialization paths follow the same error handling pattern - setting both `storage` and `memory_service` to `None` on failure:

**Eager Init Error Paths**:
- Line 346-347: Initial state in `__init__`
- Line 355-356: Error during `__init__`
- Line 748: Eager init failure fallback
- Line 755: Eager init timeout fallback
- Line 764: Eager init exception fallback

**Lazy Init Error Path**:
- Line 702: Sets `self.storage = None` on failure
- **Should also set**: `self.memory_service = None` (but currently doesn't touch it)

After the fix, the lazy init error path at line 702 should mirror line 355-356:
```python
self.storage = None
self.memory_service = None
self._storage_initialized = False
```

### Import Location and Dependencies

The `MemoryService` class is already imported at line 214:
```python
from .services.memory_service import MemoryService
```

No additional imports are needed. The class is available and used correctly in the eager init path (line 540), just missing from the lazy init path.

### Testing Strategy

To verify the fix works:

1. **Force Lazy Loading**: Set a very short timeout to ensure eager init always fails:
   ```bash
   export MCP_MEMORY_EAGER_TIMEOUT=0.001  # Force immediate timeout
   ```

2. **Trigger Memory Operation**: Use any MCP tool that requires storage:
   ```bash
   /memory-store "test content" --tags test
   ```

3. **Expected Behavior**:
   - **Before Fix**: `AttributeError: 'NoneType' object has no attribute 'store_memory'`
   - **After Fix**: Memory stored successfully via lazy-initialized MemoryService

4. **Verify Logs**: Check that lazy init logs show:
   ```
   ✅ LAZY INIT: MemoryService initialized with [backend] storage
   ```

### Implementation Checklist

The fix requires changes at three specific locations in `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/server.py`:

1. **After Line 692** (lazy init success path):
   ```python
   # Initialize MemoryService with shared business logic
   self.memory_service = MemoryService(self.storage)
   logger.info(f"✅ LAZY INIT: MemoryService initialized with {STORAGE_BACKEND} storage")
   ```

2. **Line 702** (lazy init error path) - Add after `self.storage = None`:
   ```python
   self.memory_service = None
   ```

3. **Verification**: Ensure no other code paths set `self._storage_initialized = True` without also setting `self.memory_service`

### Technical Reference

#### MemoryService Constructor Signature
```python
class MemoryService:
    def __init__(self, storage: MemoryStorage):
        """
        Initialize MemoryService with a storage backend.

        Args:
            storage: Any object implementing the MemoryStorage abstract base class
                    (SqliteVecMemoryStorage, CloudflareStorage, HybridMemoryStorage, etc.)
        """
        self.storage = storage
```

#### Storage Backend Interface
All storage backends implement the `MemoryStorage` abstract base class (`src/mcp_memory_service/storage/base.py`):
- `async def initialize()` - Called during server initialization
- `async def store(memory: Memory)` - Store a single memory
- `async def retrieve(query, n_results, tags, memory_type)` - Search memories
- Properties: `max_content_length`, `supports_chunking`

#### Handler Pattern
Every MCP tool handler that uses memory operations follows this pattern:
```python
async def handle_<operation>(self, arguments: dict) -> List[types.TextContent]:
    # 1. Extract arguments
    content = arguments.get("content")

    # 2. Ensure storage is initialized (triggers lazy loading if needed)
    await self._ensure_storage_initialized()

    # 3. Call MemoryService method (NOT storage directly)
    result = await self.memory_service.<operation>(...)

    # 4. Format and return result
    return [types.TextContent(type="text", text=result)]
```

This pattern is why MemoryService initialization is mandatory - handlers don't have fallback logic to call storage directly.

### File Locations

- **Bug Location**: `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/server.py`
  - Lines 557-705: `_ensure_storage_initialized()` method
  - Line 692: Where MemoryService creation should be added
  - Line 702: Error path that needs `self.memory_service = None`

- **Working Reference**: Same file, line 540 in `_initialize_storage_with_timeout()`

- **MemoryService Source**: `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/services/memory_service.py`

- **Import Statement**: Line 214 of `server.py` (already present)

- **Handler Examples**: Lines 1683-1721 (`handle_store_memory`), 1723+ (other handlers)

## User Notes
<!-- Any specific notes or requirements from the developer -->

## Work Log
<!-- Updated as work progresses -->
