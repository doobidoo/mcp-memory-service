# Changelog

**Recent releases for MCP Memory Service (v8.70.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

For older releases (v8.69.0 and earlier), see [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [10.6.0] - 2026-02-07

### Added
- **Server Management Dashboard** (PR #421): Complete server administration from Dashboard Settings
  - **REST API**: 4 new endpoints at `/api/server/*` (admin-protected):
    - `GET /api/server/status` - Real-time server status (version, uptime, platform info)
    - `GET /api/server/version/check` - Git-based update detection (commits behind origin)
    - `POST /api/server/update` - One-click git pull + pip install workflow
    - `POST /api/server/restart` - Safe server restart with 3-second delay
  - **Dashboard UI**: Server management section in Settings modal
  - **Security**: Admin-only access, explicit confirmation required, full audit logging

## [10.5.1] - 2026-02-06

### Added
- **Test Environment Safety Scripts** (Issue #419, PR #418): 4 critical scripts (524 lines) to prevent production database testing
  - **backup-before-test.sh**: Mandatory backup before testing with date-stamped archives and environment isolation
  - **setup-test-environment.sh**: Isolated test environment setup (port 8001) with separate data directory
  - **cleanup-test-environment.sh**: Safe test data cleanup with environment verification
  - **scripts/test/README.md**: Comprehensive testing workflow documentation (250 lines)
  - **Testing Workflow**: Complete development-to-deployment test lifecycle
  - **Safety Guarantees**: Environment validation prevents production database corruption
  - **Developer Experience**: Clear instructions for safe testing practices
  - **CRITICAL**: Prevents data loss incidents from running tests against production databases
  - Note: Security improvements (command injection fixes, atomic backups) tracked in Issue #419

## [10.5.0] - 2026-02-06

### Added
- **Dashboard Authentication UI** (Issue #414, Issue #410, PR #416): Comprehensive authentication detection and graceful user experience
  - **Authentication Detection**: Automatically detects authentication state on dashboard load (HTTP 401/403 responses)
  - **User-Friendly Modal**: Authentication modal with clear instructions for API key and OAuth flows
  - **API Key Authentication**: Secure input field with autocomplete=off, session storage, and automatic page reload after auth
  - **OAuth Flow**: Prominent "Sign in with OAuth" button for team collaboration
  - **Security Improvements**: HTTPS warning for production deployments, credential cleanup on state changes
  - **Dark Mode Compatible**: Fully styled authentication UI with dark mode support
  - **State Management**: Robust authentication state tracking across page loads
  - **Error Handling**: Clear error messages for authentication failures with retry guidance
  - **~400 lines** across 3 files: app.js (state management), index.html (modal UI), style.css (dark mode styling)
  - Resolves user confusion when accessing HTTP dashboard without authentication (Issue #410)
  - Provides discoverable authentication flow replacing raw 403 errors (Issue #414)

## [10.4.6] - 2026-02-06

### Changed
- **Documentation Enhancement** (Issue #410, Issue #414, PR #415): Clarified HTTP dashboard authentication requirements in README.md
  - Added authentication setup example to Document Ingestion section with `MCP_ALLOW_ANONYMOUS_ACCESS=true` for local development
  - Added prominent warning callout explaining authentication requirement by default
  - Documented all three authentication options in Configuration section:
    - **Option 1**: API Key authentication (`MCP_API_KEY`) - recommended for production
    - **Option 2**: Anonymous access (`MCP_ALLOW_ANONYMOUS_ACCESS=true`) - local development only
    - **Option 3**: OAuth team collaboration (`MCP_OAUTH_ENABLED=true`)
  - Improves first-time user experience by clarifying why dashboard returns 403 errors without authentication
  - Addresses user confusion when accessing HTTP dashboard with default secure configuration

### Fixed
- **CHANGELOG Cleanup** (PR f1de0ca): Removed duplicate v10.4.3 release entry from Previous Releases section

## [10.4.5] - 2026-02-05

### Added
- **Unified CLI Interface** (Issue #410, PR #261d324): `memory server --http` flag for starting HTTP REST API server
  - **Before**: Required manual script invocation: `python scripts/server/run_http_server.py`
  - **After**: Simple unified command: `memory server --http`
  - Direct uvicorn.run() integration (no subprocess overhead)
  - Respects `MCP_HTTP_PORT` and `MCP_HTTP_HOST` environment variables
  - Clear user feedback with URLs and graceful error handling
  - Better UX: Single command interface, easier to discover in --help

### Changed
- **Documentation Update** (PR #301ba2d): Updated README.md to use `memory server --http` instead of script invocation
  - Simplifies user workflow with unified command interface
  - Removes confusion about separate MCP vs HTTP server commands

## [10.4.4] - 2026-02-05

### Security
- **CRITICAL: Timing Attack Vulnerability** (PR #411): Fixed CWE-208 timing attack in API key comparison
  - Replaced direct string comparison with `secrets.compare_digest()` for constant-time comparison
  - Prevents attackers from determining correct API key character-by-character via timing analysis
  - All API key authentication methods now use secure comparison (X-API-Key header, query parameter, Bearer token)
  - Severity: CRITICAL - Recommend immediate upgrade for all deployments using API key authentication

### Fixed
- **API Key Authentication without OAuth** (Issue #407, PR #411): API key auth now works independently of OAuth configuration
  - Root cause: API routes had conditional OAuth dependencies that prevented API key authentication when `MCP_OAUTH_ENABLED=false`
  - Solution: Removed OAuth conditionals from ALL 44 API route endpoints
  - Authentication middleware now handles all auth methods unconditionally: OAuth, API key, or anonymous
  - Enables simple single-user deployments without OAuth overhead

### Added
- **X-API-Key Header Authentication** (PR #411): Recommended method for API key authentication
  - Usage: `curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/memories`
  - More secure than query parameters (not logged in server logs)
  - Works with all API endpoints
- **Query Parameter Authentication** (PR #411): Convenient fallback for scripts/browsers
  - Usage: `curl "http://localhost:8000/api/memories?api_key=your-secret-key"`
  - Warning: Logs API keys in server logs - use X-API-Key header in production
- **Bearer Token Compatibility** (PR #411): Backward compatible with existing Bearer token auth
  - Usage: `curl -H "Authorization: Bearer your-secret-key" http://localhost:8000/api/memories`

### Changed
- **OAuth Import Behavior** (PR #411): OAuth authentication modules now always imported regardless of configuration
  - Previous behavior caused FastAPI AssertionError when `OAUTH_ENABLED=false`
  - Middleware decides auth method at runtime based on configuration
  - Allows server startup with OAuth disabled while maintaining proper authentication

## [10.4.3] - 2026-02-04

### Fixed
- **Consolidation Logger** (PR #404): Fixed NameError in consolidator.py where logger.warning() was called but no module-level logger existed
  - Added module-level logger import to prevent crashes in `_handle_compression_results` and `_apply_forgetting_results` methods
  - Ensures proper error logging during memory consolidation operations
- **Windows Task Scheduler HTTP Dashboard** (PR #402): Fixed 6 bugs preventing scheduled task from starting HTTP dashboard on Windows
  - Fixed Task Scheduler PATH issues by using full path resolution via $env:SystemRoot for powershell.exe
  - Added robust executable discovery (Find-Executable) that probes known uv/python install locations (`$env:USERPROFILE\.local\bin`, `$env:LOCALAPPDATA\uv`, `$env:USERPROFILE\.cargo\bin`)
  - Fixed $using:LogFile silent failure in .NET event handlers by changing to $script:LogFile for proper log capture
  - Corrected health check URLs from https:// to http:// (server defaults to HTTP)
  - Fixed display URLs in status output to show correct http:// protocol
  - Added executable path logging for easier Task Scheduler debugging
  - Improved Python version discovery with py.exe launcher and descending version sort (prefers latest)

## [10.4.2] - 2026-02-01

### Fixed
- **Docker Container Startup** (Issue #400): Fixed ModuleNotFoundError for aiosqlite and core dependencies
  - Docker container failed to start with "ModuleNotFoundError: No module named 'aiosqlite'"
  - Root cause: `uv pip install -e .${INSTALL_EXTRA}` was not properly installing core dependencies
  - Solution: Split Docker package installation into three clear steps:
    1. Install CPU-only PyTorch if needed (conditional)
    2. Always install core dependencies with `python -m uv pip install -e .`
    3. Install optional dependencies with `python -m uv pip install -e ".${INSTALL_EXTRA}"`
  - This ensures all core dependencies (aiosqlite, fastapi, sqlite-vec, etc.) are always installed
  - Preserved backward compatibility with all build argument combinations
  - Verified with testing: core dependencies properly installed in all configurations

## [10.4.1] - 2026-01-29

### Fixed
- **Time Expression Parsing** (Issue #396): Fixed `time_expr` parameter to correctly parse natural language time expressions
  - Changed from `extract_time_expression()` to `parse_time_expression()` in `search_memories()`
  - Now correctly handles: "last week", "3 days ago", "last 5 days", "1 week ago"
  - `extract_time_expression()` was designed for extracting time expressions from larger text queries
  - `parse_time_expression()` is the correct function for isolated time expressions
  - Added comprehensive regression tests covering reported failures and edge cases
  - ISO date workaround (`after`/`before` parameters) continues to work as before

## [10.4.0] - 2026-01-29

### Added
- **Semantic Deduplication** (Issues #390, #391): Prevents storing semantically similar content within configurable time window
  - New `_check_semantic_duplicate()` method in SQLiteVecStorage using KNN cosine similarity
  - Configurable via environment variables:
    - `MCP_SEMANTIC_DEDUP_ENABLED` (default: true) - Enable/disable feature
    - `MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS` (default: 24) - Time window for duplicate detection
    - `MCP_SEMANTIC_DEDUP_THRESHOLD` (default: 0.85) - Similarity threshold (0.0-1.0)
  - Catches cross-hook duplicates (e.g., PostToolUse + SessionEnd reformulations)
  - Returns descriptive error messages: "Duplicate content detected (semantically similar to {hash}...)"
  - Efficient KNN search using sqlite-vec's `vec_distance_cosine()` function
  - Comprehensive test suite with 6 new tests covering time windows, configuration, and edge cases

- **Memory Budget Optimization** (Issue #390): Increased memory retrieval capacity and reserved slots for curated memories
  - Increased `maxMemoriesPerSession` from 8 to 14 slots (75% increase)
  - New `reservedTagSlots` configuration (default: 3) - Guarantees minimum slots for tag-based retrieval
  - Smart slot allocation across retrieval phases:
    - Phase 0 (Git): Up to 3 slots (adaptive)
    - Phase 1 (Recent): ~60% of remaining slots
    - Phase 2 (Tags): At least `reservedTagSlots`, more if available
  - Prevents semantic search from crowding out curated memories
  - Configuration documentation added to session-start.js

- **Enhanced Content Truncation** (Issue #392): Multi-delimiter sentence boundary detection
  - Expanded from 4 to 9-10 delimiter types: `. ` `! ` `? ` `.\n` `!\n` `?\n` `.\t` `;\n` `\n\n`
  - Improved break point algorithm preserves natural sentence boundaries
  - Lowered threshold from 80% to 70% for more flexibility
  - Applied consistently across auto-capture-patterns.js and context-formatter.js
  - Eliminates mid-sentence cuts at colons/commas when better delimiters present

### Changed
- **Tag Case-Normalization** (Issue #391): All tags stored in lowercase with case-insensitive deduplication
  - Updated `normalize_tags()` in memory_service.py to convert all tags to lowercase
  - Eliminates duplicate tags like `["Tag", "tag", "TAG"]` → `["tag"]`
  - Applied across all tag sources: parameter tags, metadata tags, hook-generated tags
  - JavaScript hooks updated for consistent case-normalization:
    - auto-capture-patterns.js: generateTags() normalizes all tags
    - session-end.js: Project name, language, topics, frameworks all lowercase
  - Comprehensive test suite with 11 new tests for unit and integration scenarios
  - Backward compatible: Existing mixed-case tags remain unchanged, searches already case-insensitive

- **Hook Deduplication Threshold** (Issue #391): Improved Jaccard similarity detection
  - Lowered threshold from 80% to 65% in context-formatter.js
  - Catches more cross-hook reformulations (55-70% similarity range)
  - Maintains balance between duplicate detection and legitimate variations

### Fixed
- **Test Environment Isolation**: Disabled semantic deduplication during tests
  - Prevents interference with existing test expectations
  - Tests can now create similar content without triggering dedup
  - Added clear documentation in conftest.py

### Documentation
- Added comprehensive implementation plan in fix-plan-issues-390-391-392.md
- Updated .env.example with semantic deduplication configuration
- Added 17 new tests with detailed documentation
- Created TEST_ADDITIONS_SUMMARY.md documenting all test scenarios

### Performance
- Semantic dedup adds <100ms overhead per storage operation
- KNN search leverages sqlite-vec's optimized cosine distance calculations
- No impact on retrieval performance
- Hook execution time unchanged (<10s total)

## [10.3.0] - 2026-01-29

### Added
- **SQL-Level Filtering Optimization** (#374): Dramatic performance improvements for large datasets
  - Optimized `delete_memories` path using SQL filtering instead of Python-level filtering
  - New `delete_by_tags` method for Cloudflare backend for efficient bulk deletion
  - New `get_memories_by_time_range` support for time-based filtering
  - Performance benchmarks demonstrating significant improvements:
    - **115x speedup** for tag filtering (1000 memories: 116ms → 1ms)
    - **74x speedup** for time range filtering (1000 memories: 36ms → 0.49ms)
    - **98% memory reduction** (10,000 memories: 147MB → 2.5MB)
  - Contributor: @isiahw1

### Fixed
- **API Consistency** (#393): Standardized `delete_by_tags` signature across all backends
  - All backends now return 3-tuple `(count, message, deleted_hashes)`
  - Prevents unpacking errors and provides audit trail for deleted content
  - Enhanced tracking for sync operations in hybrid backend
- **Enhanced Exception Handling**: Improved error handling in external embedding API
  - Specific JSONDecodeError catch for better error messages
  - Duplicate index detection and validation
  - Missing 'index' field validation in API responses
- **Test Fixes**: Corrected test class name typo in `test_external_embeddings.py`

## [10.2.1] - 2026-01-28

### Fixed
- **Integer Enum Incompatibility** (#387): Fixed OpenCode with Gemini model failure
  - Changed `memory_quality` tool `rating` parameter from integer enum to string enum
  - Added backwards-compatible conversion logic in quality handler
  - Resolves MCP client compatibility issues with integer enum values
  - Files: `server_impl.py`, `server/handlers/quality.py`
- **Wrong Method Name in delete_with_filters** (#389): Fixed delete operations with tag/time filters
  - Replaced non-existent `list_memories()` calls with correct `get_all_memories()` method name
  - Affected delete operations using tag or timeframe filters
  - Files: `storage/base.py` (2 locations)
  - All 8 delete tests now passing in `test_unified_tools.py`

## [10.2.0] - 2026-01-28

### Added
- **External Embedding API Support** (#386): Use external OpenAI-compatible embedding APIs (vLLM, Ollama, TEI, OpenAI) instead of local models
  - Configure via `MCP_EXTERNAL_EMBEDDING_URL`, `MCP_EXTERNAL_EMBEDDING_MODEL`, `MCP_EXTERNAL_EMBEDDING_API_KEY`
  - **Important**: Only supported with `sqlite_vec` backend (not compatible with `hybrid` or `cloudflare` backends)
  - Graceful fallback to local models if external API unavailable
  - Supports any OpenAI-compatible `/v1/embeddings` endpoint
  - Automatic dimension detection from API responses
  - Backend validation ensures correct configuration
  - 10/10 core tests passing (3 integration tests require refactoring)
  - See `docs/deployment/external-embeddings.md` for setup guide
  - Contributor: @isiahw1

## [10.1.2] - 2026-01-27

### Fixed
- **Windows PowerShell 7+ compatibility**: Fixed SSL certificate validation in `manage_service.ps1`
  - Extends previous fix from `update_and_restart.ps1` to `manage_service.ps1`
  - PowerShell 5.1: Uses `ICertificatePolicy` (.NET Framework)
  - PowerShell 7+: Uses `-SkipCertificateCheck` parameter on `Invoke-WebRequest`
  - Error was: `CS0246: The type or namespace name 'ICertificatePolicy' was not found`

## [10.1.1] - 2026-01-27

### Fixed
- **Missing `requests` dependency**: Added `requests>=2.28.0` to pyproject.toml (Fixes #378)
  - `sentence-transformers` requires `requests` but doesn't declare it as a dependency
  - Caused `ModuleNotFoundError: No module named 'requests'` during embedding initialization
  - Affects fresh installations without `requests` pre-installed
- **Windows PowerShell 7+ compatibility**: Fixed `update_and_restart.ps1` SSL certificate validation
  - PowerShell 5.1: Uses `ICertificatePolicy` (.NET Framework)
  - PowerShell 7+: Uses `-SkipCertificateCheck` parameter on `Invoke-RestMethod`
  - Script now detects PowerShell version and uses appropriate method
  - Error was: `CS0246: The type or namespace name 'ICertificatePolicy' was not found`

### Changed
- **Relationship Inference Threshold Tuning**: Improved relationship type diversity in memory graph analytics
  - Relaxed default minimum confidence threshold recommendation from 0.6 to 0.4 for existing graphs
  - Enables discovery of more nuanced relationship types (causes, fixes, contradicts, supports, follows)
  - Production results: Improved from 2 to 3 relationship types with 0.4 threshold
  - Distribution on 2,392 edges: related (75.52%), contradicts (18.09%), follows (6.39%)
  - Script usage: `python scripts/maintenance/update_graph_relationship_types.py --min-confidence=0.4`
  - Balances precision and recall - lower values (0.3) may introduce false positives
  - Note: Script default remains 0.6 for conservative new deployments

## [10.1.0] - 2026-01-25

### Added
- **Python 3.14 Support**: Extended Python compatibility to 3.10-3.14 (Fixes #376)
  - Upgraded tokenizers dependency from ==0.20.3 to >=0.22.2
  - Resolves PyO3 compatibility issues preventing installation on Python 3.14
  - Fixed tokenizers API change: `encode((query, text))` → `encode(query, pair=text)` in ONNX ranker
  - No breaking changes - maintains full backward compatibility
  - All 1005 tests passing across all supported Python versions
  - Enables adoption by projects requiring Python 3.14

## [10.0.3] - 2026-01-25

### Fixed
- **Backup scheduler critical bugs** (Fixes #375)
  - **Scheduler never started**: `BackupScheduler.start()` was never called in FastAPI lifespan
    - Added `backup_scheduler` global variable to `app.py`
    - Integrated scheduler startup into FastAPI lifespan context manager
    - Added graceful shutdown handling in lifespan cleanup
    - Automatic backups now work as intended
  - **Past dates in "Next Scheduled"**: `_calculate_next_backup_time()` returned past dates when server was offline longer than backup interval
    - Rewrote calculation logic with while loop to advance to future time
    - Now correctly handles multi-interval downtime (e.g., server down for 5 days with daily backups)
    - "Next Scheduled" field always shows a future timestamp
  - **Comprehensive testing**: Added 8 new tests covering hourly/daily/weekly intervals, past-due scenarios (5h, 10d, 4w overdue), and edge cases
  - Bug existed since commit 8a19ba8 (PR #233, Nov 2025)
  - Commit: 94f3c3a

## [10.0.2] - 2026-01-23

### Fixed
- **Tool list now shows only 12 unified tools**
  - Removed 20 deprecated tool definitions from MCP tool list advertisement
  - Deprecated tools still work via backwards compatibility routing in `compat.py`
  - Achieves the promised "64% tool reduction" (34→12 visible tools)
  - Claude Desktop and other MCP clients now see clean, focused tool list
  - No breaking changes - all old tool names continue working with deprecation warnings
  - Tools removed from advertisement: `recall_memory`, `retrieve_memory`, `retrieve_with_quality_boost`, `debug_retrieve`, `exact_match_retrieve`, `recall_by_timeframe`, `delete_memory`, `delete_by_tag`, `delete_by_tags`, `delete_by_all_tags`, `delete_by_timeframe`, `delete_before_date`, `get_raw_embedding`, `consolidate_memories`, `consolidation_status`, `consolidation_recommendations`, `scheduler_status`, `trigger_consolidation`, `pause_consolidation`, `resume_consolidation`

## [10.0.1] - 2026-01-23

### Fixed
- **CRITICAL: MCP tools not loading in Claude Desktop**
  - Fixed NameError in `handle_list_tools()` caused by JavaScript-style booleans (`false`/`true`) instead of Python booleans (`False`/`True`)
  - Affected locations: `src/mcp_memory_service/server_impl.py` lines 1446, 1662, 2182, 2380
  - Impact: v10.0.0 prevented ALL MCP tools from loading in Claude Desktop
  - Error message: "name 'false' is not defined"
  - Resolution: All tool schemas now use proper Python boolean literals
  - Tools now load correctly in Claude Desktop
  - Commit: 2958aef

## [10.0.0] - 2026-01-23

### Major Changes

**MCP Tool Consolidation: 34 → 12 Tools (64% API Simplification)**

The most significant API redesign in MCP Memory Service history - consolidating 34 tools into 12 unified tools for better usability, maintainability, and MCP best practices compliance. While technically maintaining 100% backwards compatibility, this represents a new generation of the API architecture warranting a major version bump.

**Key Achievements:**
- **64% Tool Reduction**: 34 tools → 12 tools with enhanced capabilities
- **100% Backwards Compatibility**: All 33 deprecated tools continue working with deprecation warnings
- **Zero Breaking Changes**: Existing integrations work unchanged until v11.0.0
- **Enhanced Capabilities**: New unified tools offer combined functionality (e.g., filter by tags + time range)
- **Comprehensive Testing**: 62 new tests added (100% pass rate, 968 total tests)
- **Migration Guide**: Complete documentation in `docs/MIGRATION.md`

### Tool Consolidation Details

**Delete Operations (6 → 1):**
- `delete_memory`, `delete_by_tag`, `delete_by_tags`, `delete_by_all_tags`, `delete_by_timeframe`, `delete_before_date`
- **Unified as:** `memory_delete` with combined filters (tags + time range + dry_run mode)

**Search Operations (6 → 1):**
- `retrieve_memory`, `recall_memory`, `recall_by_timeframe`, `retrieve_with_quality_boost`, `exact_match_retrieve`, `debug_retrieve`
- **Unified as:** `memory_search` with modes (semantic/exact/hybrid), quality boost, and time filtering

**Consolidation Operations (7 → 1):**
- `consolidate_memories`, `consolidation_status`, `consolidation_recommendations`, `scheduler_status`, `trigger_consolidation`, `pause_consolidation`, `resume_consolidation`
- **Unified as:** `memory_consolidate` with action parameter (run/status/recommend/scheduler/pause/resume)

**Ingestion Operations (2 → 1):**
- `ingest_document`, `ingest_directory`
- **Unified as:** `memory_ingest` with automatic directory detection

**Quality Operations (3 → 1):**
- `rate_memory`, `get_memory_quality`, `analyze_quality_distribution`
- **Unified as:** `memory_quality` with action parameter (rate/get/analyze)

**Graph Operations (3 → 1):**
- `find_connected_memories`, `find_shortest_path`, `get_memory_subgraph`
- **Unified as:** `memory_graph` with action parameter (connected/path/subgraph)

**Simple Renames (5 tools):**
- `store_memory` → `memory_store`
- `check_database_health` → `memory_health`
- `get_cache_stats` → `memory_stats`
- `cleanup_duplicates` → `memory_cleanup`
- `update_memory_metadata` → `memory_update`

**New Tool:**
- `memory_list` - Browse memories with pagination (replaces `search_by_tag` with enhanced features)

### Deprecation Architecture

**New Compatibility Layer:** `src/mcp_memory_service/server/compat.py` (318 lines)
- Centralized `DEPRECATED_TOOLS` mapping with migration hints
- Automatic warning emission for deprecated tool usage
- Clean delegation to new unified handlers
- Zero performance overhead for new tools

**Deprecation Timeline:**
- **v10.0+**: All old tools work with warnings (current release)
- **v11.0**: Old tools removed (breaking change)

### Technical Improvements

**Code Quality:**
- Reduced API surface area by 64%
- Eliminated duplicate validation logic across 33 handlers
- Improved maintainability with unified error handling
- Better parameter naming consistency (e.g., `n_results` → `limit`, `content_hash` standardized)

**Testing:**
- 62 new comprehensive tests covering all tool migrations
- 100% test pass rate maintained (968 total tests)
- Validation of deprecation warnings and parameter transformations
- Integration tests for unified handler flows

**Documentation:**
- Complete migration guide with side-by-side examples
- Deprecation warnings with actionable migration hints
- Updated MCP schema with new tool definitions
- CLAUDE.md updated with new tool reference

### Migration Guide

**For Existing Users:**
1. **No Action Required**: Old tool names continue working
2. **Optional**: Update to new tool names to eliminate deprecation warnings
3. **Follow Migration Guide**: `docs/MIGRATION.md` provides mapping for all tools
4. **Timeline**: Update before v11.0.0 (removal date TBD)

**Example Migration:**
```python
# Old (v9.x) - Still works in v10.0 with warning
{"tool": "retrieve_memory", "query": "python", "n_results": 5}

# New (v10.0+) - No warnings
{"tool": "memory_search", "query": "python", "limit": 5, "mode": "semantic"}
```

**Enhanced Capabilities:**
```python
# Old (v9.x) - Required multiple tool calls
delete_by_tags(tags=["temp"])  # First call
delete_by_timeframe(start="2024-01-01")  # Second call

# New (v10.0+) - Single unified call
memory_delete(tags=["temp"], after="2024-01-01", tag_match="any")
```

### Related Issues

- Closes #372 - MCP Tool Optimization
- Related #374 - Follow-up performance optimizations

### Contributors

Special thanks to the community for feedback on API design and testing the deprecation layer during development.

---

## [9.3.1] - 2026-01-20

### Fixed
- **Fatal Python error during shutdown** (#368)
  - Fixed "database is locked" crash when Claude Desktop sends shutdown signal (SIGTERM/SIGINT)
  - Changed signal handler to use `os._exit(0)` instead of `sys.exit(0)` to avoid buffered I/O lock deadlock
  - Prevents "Fatal Python error: _enter_buffered_busy: could not acquire lock" during interpreter shutdown
  - Server now shuts down cleanly without "Server disconnected" errors in Claude Desktop
  - Root cause: `sys.exit(0)` in signal handler tried to flush buffered stdio streams while locks were held
  - Solution: Use `os._exit(0)` to bypass I/O cleanup after `_cleanup_on_shutdown()` completes
  - Resolved issue #368

## [9.3.0] - 2026-01-19

### Added
- **Relationship Inference Engine** (Commit 40db489)
  - Intelligent association typing for knowledge graph relationships
  - Multi-factor analysis: memory type combinations, content semantics, temporal patterns, contradictions
  - Pattern-based classification: causes, fixes, contradicts, supports, follows, related
  - Confidence scoring (0.0-1.0) with configurable minimum threshold (default: 0.6)
  - **Implementation**:
    - New module: `src/mcp_memory_service/consolidation/relationship_inference.py` (433 lines)
    - `RelationshipInferenceEngine` class with async relationship type inference
    - 4 analysis methods: type combination, content semantics, temporal, contradictions
  - **Maintenance Scripts**:
    - `scripts/maintenance/improve_memory_ontology.py` - Batch re-classify memory types and standardize tags
    - `scripts/maintenance/update_graph_relationship_types.py` - Batch update existing graph relationships
    - Enhanced `scripts/maintenance/cleanup_memories.py` with HTTP/HTTPS auto-detection
  - **Benefits**:
    - Rich knowledge graphs with meaningful relationship types beyond "related"
    - Automatic relationship type inference during consolidation
    - Retroactive classification of existing relationships

### Fixed
- **Invalid memory type 'knowledge' in Web UI** (#364)
  - Removed invalid "knowledge" option from document upload form
  - Added missing memory types to ontology: `document`, `note`, `reference`
  - Fixed Knowledge Graph display issues for uploaded documents
  - All 34 ontology tests passing
  - No more validation warnings in server logs

- **wandb dependency conflict causing embedding failures** (#311)
  - Fixed incompatibility between wandb 0.15.2 and protobuf 5.x
  - Set `WANDB_DISABLED=true` environment variable to prevent wandb loading
  - Updated dependency constraint to `wandb>=0.18.0` in pyproject.toml
  - Restored semantic search functionality with proper SentenceTransformer embeddings
  - Fixed 3 failing tests related to embedding model initialization
  - Added regression test to prevent future embedding model fallback issues

## [9.2.1] - 2026-01-19

### Fixed
- **CRITICAL: Knowledge Graph table initialization bug** (#362)
  - **Root Cause**: Migration files existed but were never executed during storage initialization
  - **Impact**: All Knowledge Graph operations failed with "no such table: memory_graph" errors
  - **Solution**: Created MigrationRunner utility to execute SQL migrations during SqliteVecMemoryStorage.initialize()
  - **Migrations Applied**: Graph table migrations (008, 009, 010) now run automatically
  - **Test Results**: Knowledge Graph tests improved from 14/51 to 44/51 passing (37 fixes)
  - **Full Test Suite**: 910 tests passing, 0 regressions introduced
  - **Idempotency**: Migration runner handles duplicate column errors gracefully (non-fatal warnings)
  - **Technical Details**:
    - New `MigrationRunner` class in `storage/migration_runner.py` (190 lines)
    - Integrated into `SqliteVecMemoryStorage.initialize()` with 48 new lines
    - 340 lines of comprehensive unit tests (10 test cases)
    - Supports both sync and async migration execution
    - Non-fatal error handling (logs warnings, doesn't crash)
  - **Remaining Issues**: 7 Knowledge Graph test failures are pre-existing bugs (deprecated API usage, not related to this fix)

### Technical Details
- Migration runner executes migrations in both new database and existing database initialization paths
- Skips already-applied migrations automatically (e.g., duplicate columns)
- Ensures memory_graph table is always available for graph operations
- No user action required - migrations run automatically on next storage initialization

## [9.2.0] - 2026-01-18

### Added
- **Knowledge Graph Dashboard with D3.js v7.9.0** - Interactive force-directed graph visualization
  - **Interactive Force-Directed Graph**: Zoom, pan, drag nodes to explore memory relationships
  - **Relationship Type Distribution Chart**: Bar chart showing breakdown of 6 relationship types (Chart.js)
  - **6 Typed Relationships**: causes, fixes, contradicts, supports, follows, related
  - **2 New API Endpoints**:
    - `/api/analytics/relationship-types` - Get distribution of relationship types
    - `/api/analytics/graph-visualization` - Get graph data for visualization
  - **Interactive Controls**:
    - Zoom with mouse wheel or touch
    - Pan by dragging background
    - Drag nodes to explore connections
    - Hover tooltips with memory details
  - **Multi-Language Support**: Full UI localization for 7 languages (en, zh, de, es, fr, ja, ko)
  - **154 New i18n Translation Keys**: 22 translation keys per language for graph UI
  - **Dark Mode Integration**: Seamless theme switching with existing dashboard
  - **Performance**: Successfully tested with 2,730 relationships

### Changed
- **Extended Storage Interface**: Added graph analytics methods
  - `get_relationship_type_distribution()` - Query relationship type counts
  - `get_graph_visualization_data()` - Fetch graph data optimized for D3.js rendering

### Database
- **Migration Required**: Added `relationship_type` column to `memory_graph` table
  - **SQLite Migration**: `python scripts/migration/add_relationship_type_column.py`
  - **Cloudflare D1**: Not required (graph is SQLite-only in current version)
  - **Default Value**: All existing relationships set to 'related' type
  - **Backward Compatible**: Migration is idempotent and safe to re-run

### Documentation
- **New Guide**: Complete Knowledge Graph Dashboard documentation in `docs/features/knowledge-graph-dashboard.md`
- **Updated README**: Added Knowledge Graph Dashboard section with features and screenshots

## [9.0.6] - 2026-01-18

### Added
- **OAuth Persistent Storage Backend** (#360): SQLite-based OAuth storage for multi-worker deployments
  - New `MCP_OAUTH_STORAGE_BACKEND` environment variable (memory|sqlite)
  - New `MCP_OAUTH_SQLITE_PATH` configuration for database location
  - WAL mode enabled for multi-process safety
  - Atomic one-time authorization code consumption (prevents replay attacks)
  - Performance: <10ms token operations
  - Backward compatible (defaults to memory backend)
  - Comprehensive test suite (30 tests, parametrized across backends)
  - Documentation: [docs/oauth-storage-backends.md](docs/oauth-storage-backends.md)

### Fixed
- **uvx HTTP Test Failures** (#361): Lazy initialization of asyncio.Lock in api/client.py
  - Fixed module-level Lock creation that caused event loop context issues
  - All HTTP endpoint tests now pass in uvx CI environment
  - No impact on existing functionality

### Technical Details
- OAuth storage now supports pluggable backends via abstract base class
- SQLite backend uses atomic UPDATE WHERE for race-safe code consumption
- Factory pattern for backend selection (create_oauth_storage)
- All backends tested with identical test suite (parametrized fixtures)

## [9.0.5] - 2026-01-18

🚨 **CRITICAL HOTFIX** - Fixes OAuth 2.1 token endpoint routing bug

### Fixed
- **CRITICAL: OAuth 2.1 token endpoint routing fixed** (Issue #358)
  - **Root Cause**: `@router.post("/token")` decorator was on wrong function (`_handle_authorization_code_grant` instead of `token`)
  - **Impact**: Token endpoint completely non-functional for all standard OAuth clients (Claude Desktop, MCPJam, etc.)
  - **Symptoms**: HTTP 422 Unprocessable Entity errors when attempting to exchange authorization codes for access tokens
  - **Incident**: OAuth clients could not complete authorization code flow - authorization succeeded but token exchange failed
  - **Fix**: Moved `@router.post("/token")` decorator from internal helper function to correct public endpoint handler
  - **Files Changed**:
    - `src/mcp_memory_service/web/oauth/authorization.py:220` - Removed decorator from `_handle_authorization_code_grant`
    - `src/mcp_memory_service/web/oauth/authorization.py:348` - Added decorator to `token` function
  - **OAuth 2.1 Compliance**: Token endpoint now correctly implements RFC 6749 token exchange flow
  - **Backward Compatibility**: Fixes broken functionality introduced in recent OAuth refactoring
  - **Recommendation**: All users attempting to use OAuth authentication should upgrade to v9.0.5 immediately
- **Script reliability improvements for update and server management**
  - **Network error handling**: Added retry logic (up to 3 attempts) to `update_and_restart.sh` for transient DNS/network failures
    - Network connectivity check before pip install (ping 8.8.8.8, 1.1.1.1)
    - 2-second delay between retries, 5-second wait when network check fails
    - Clear error messages with helpful suggestions on failure
    - Fixes DNS errors: "nodename nor servname provided, or not known" (Errno 8)
  - **Server startup timeout**: Increased HTTP server initialization timeout from 10s to 20s
    - Hybrid storage + embedding model loading takes 12-15 seconds
    - Previous 10s timeout caused false "failed to start" errors
    - Updated `http_server_manager.sh` to allow adequate initialization time
  - **User Impact**: More reliable updates on unstable networks, no more premature timeout failures
  - **Files Changed**:
    - `scripts/update_and_restart.sh:349-398` - Network retry logic
    - `scripts/service/http_server_manager.sh:182-185` - Timeout increase
  - **Testing**: Server restart verified successful (6-13s startup, well within 20s limit)

## [9.0.4] - 2026-01-17

🚨 **CRITICAL HOTFIX** - Fixes OAuth validation blocking server startup

### Fixed
- **CRITICAL: OAuth validation preventing server startup** (discovered in v9.0.3)
  - **Root Cause**: `OAUTH_ENABLED` defaulted to `True`, causing OAuth validation to run on every import
  - **Impact**: Server startup failed with `ValueError: Invalid OAuth configuration: JWT configuration error: No JWT signing key available`
  - **Symptoms**: `update_and_restart.sh` fails during dependency installation, config.py import fails
  - **Fix**: Changed `OAUTH_ENABLED` default from `True` to `False` (opt-in, not opt-out)
  - **Fix**: Made OAuth validation non-fatal (logs errors but doesn't raise exception)
  - **Files Changed**:
    - `src/mcp_memory_service/config.py:690` - Changed default to `False`
    - `src/mcp_memory_service/config.py:890-895` - Made validation non-fatal
  - **Backward Compatibility**: No impact (OAuth was broken by default, now works correctly)
  - **Recommendation**: All users on v9.0.3 should upgrade to v9.0.4 immediately

## [9.0.3] - 2026-01-17

🚨 **CRITICAL HOTFIX** - Fixes Cloudflare D1 schema migration bug causing container reboot loop

### Fixed
- **CRITICAL: Add automatic schema migration for Cloudflare D1 backend** (Issue #354)
  - **Root Cause**: v8.72.0 added `tags` and `deleted_at` columns but Cloudflare backend had no migration logic
  - **Impact**: Users upgrading from v8.69.0 → v8.72.0+ experienced container reboot loop due to missing columns
  - **Symptoms**: `400 Bad Request` errors from D1 when trying to use missing columns
  - **Fix**: Added `_migrate_d1_schema()` method with automatic column detection and migration
  - **Fix**: Added retry logic with exponential backoff to handle D1 metadata sync issues
  - **Fix**: Added clear error messages with manual SQL workaround if automated migration fails
  - **Files Changed**:
    - `src/mcp_memory_service/storage/cloudflare.py:290-495` - Added migration methods
  - **Migration Process**:
    1. Check existing schema using `PRAGMA table_info(memories)`
    2. Add missing columns: `tags TEXT`, `deleted_at REAL DEFAULT NULL`
    3. Create index: `idx_memories_deleted_at`
    4. Verify columns are usable (handles D1 metadata sync issues)
  - **Backward Compatibility**: Safe for all versions (idempotent, no data loss)
  - **Manual Workaround** (if automated migration fails):
    ```sql
    ALTER TABLE memories ADD COLUMN tags TEXT;
    ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL;
    CREATE INDEX IF NOT EXISTS idx_memories_deleted_at ON memories(deleted_at);
    ```
  - **Recommendation**: All Cloudflare users on v8.72.0+ should upgrade to v9.0.3 immediately

## [9.0.2] - 2026-01-17

🚨 **CRITICAL HOTFIX** - Actually includes the code fix from v9.0.1

### Fixed
- **CRITICAL: Include actual code changes for mass deletion bug fix**
  - **Issue**: v9.0.1 was tagged and released WITHOUT the actual code changes to `manage.py`
  - **Root Cause**: Code changes were committed AFTER the v9.0.1 tag was created
  - **Impact**: PyPI and Docker images for v9.0.1 do NOT contain the fix
  - **This Release**: v9.0.2 includes the actual code changes from commit 9c5ed87
  - **Files Fixed**:
    - `src/mcp_memory_service/web/api/manage.py:254` - `confirm_count` now REQUIRED
    - Added comprehensive security documentation and error messages
  - **Recommendation**: All users should upgrade to v9.0.2 (not v9.0.1)

## [9.0.1] - 2026-01-17

⚠️ **WARNING**: This release was tagged incorrectly and does NOT include the actual code fix. Please upgrade to v9.0.2 instead.

🚨 **CRITICAL HOTFIX** - Fixes accidental mass deletion bug in v9.0.0

### Fixed
- **CRITICAL: Fix accidental mass deletion via /delete-untagged endpoint** (Hotfix)
  - **Root Cause**: `confirm_count` parameter was optional in `/api/manage/delete-untagged` endpoint
  - **Impact**: If endpoint called without `confirm_count`, ALL untagged memories were deleted without confirmation
  - **Incident**: On 2026-01-17 at 10:59:20, 6733 memories (87% of database) were accidentally soft-deleted
  - **Fix**: Made `confirm_count` parameter REQUIRED (not optional)
  - **Fix**: Enhanced safety check to always validate confirm_count matches actual count
  - **Fix**: Improved error message to guide users to use GET /api/manage/count-untagged first
  - **Security**: Added comprehensive documentation about the security implications
  - **Recovery**: All affected memories can be restored by setting `deleted_at = NULL`
  - **File**: `src/mcp_memory_service/web/api/manage.py:254`
  - **Breaking Change**: API now requires `confirm_count` parameter (previously optional)
  - **Recommendation**: All v9.0.0 users should upgrade immediately to v9.0.1

## [9.0.0] - 2026-01-17

⚠️ **MAJOR RELEASE** - Contains breaking changes. See Migration Guide in README.md.

### Fixed
- **Fix 33 API/HTTP test failures with package import error** (Issue #351, PR #352)
  - Changed relative imports to absolute imports in `web/api/backup.py` for pytest compatibility
  - Fixed `ModuleNotFoundError: 'mcp_memory_service' is not a package` during test collection
  - Resolved pytest `--import-mode=prepend` confusion with triple-dot relative imports
  - **Test Results**: 829/914 tests passing (up from 818), all API/HTTP integration tests pass
  - **Impact**: Single-line fix, no API changes, no breaking changes

- **Fix validation tests and legacy type migration** (PR #350)
  - Fixed all 5 validation tests (editable install detection, version matching, runtime warnings)
  - Fixed `check_dev_setup.py` to import `_version.py` directly instead of text parsing
  - Fixed timestamp test import error (namespace collision in `models` import)
  - Migrated 38 legacy memory types to new ontology (task→observation, note→observation, standard→observation)
  - Pinned `numpy<2` for pandas compatibility (prevents binary incompatibility errors)
  - Extracted offline mode setup to standalone `offline_mode.py` module (cleaner package structure)
  - Restored core imports in `__init__.py` for pytest package recognition
  - Updated consolidation retention periods for new ontology types
  - **Test Results**: 818/851 tests passing (96%), all validation tests pass

- **Fix bidirectional storage for asymmetric relationships** ⚠️ **BREAKING CHANGE** (Issue #348, PR #349)
  - Asymmetric relationships (causes, fixes, supports, follows) now store only directed edges
  - Symmetric relationships (related, contradicts) continue storing bidirectional edges
  - Database migration (010) removes incorrect reverse edges from existing data
  - Query infrastructure with `direction` parameter now works correctly with asymmetric storage
  - SemanticReasoner methods (find_causes, find_fixes) validated with new storage model
  - New `is_symmetric_relationship()` function in ontology.py for relationship classification
  - Updated `store_association()` to conditionally store edges based on relationship symmetry
  - Updated `find_connected()` direction="both" to use CASE expression for asymmetric edges
  - **Breaking Change**: Code expecting bidirectional asymmetric edges needs `direction="both"` parameter

### Added
- **Phase 0 Ontology Foundation - Core Implementation** ⚠️ **BREAKING CHANGE** (PR #347)
  - **Memory Type Ontology**: Formal classification system with 5 base types and 21 subtypes
    - Base types: observation, decision, learning, error, pattern
    - Hierarchical taxonomy with parent-child relationships
    - Soft validation: defaults to 'observation' for invalid types (backward compatible)
  - **Tag Taxonomy**: Structured namespace system with 6 predefined namespaces
    - Namespaces: `sys:`, `q:`, `proj:`, `topic:`, `t:`, `user:`
    - Backward compatible with legacy tags (no namespace)
    - O(1) namespace validation via exposed `VALID_NAMESPACES` class attribute
  - **Typed Relationships**: Semantic relationship system for knowledge graph
    - 6 relationship types: causes, fixes, contradicts, supports, follows, related
    - Database migration adds `relationship_type` column to memory_graph table
    - `TypedAssociation` dataclass for explicit relationship semantics
    - GraphStorage extended with relationship type filtering in queries
  - **Lightweight Reasoning Engine**: Foundation for causal inference
    - `SemanticReasoner` class with contradiction detection
    - Causal chain analysis: `find_fixes()`, `find_causes()`
    - Placeholder methods for future reasoning capabilities
  - **Performance Optimizations**: Caching and validation improvements
    - `get_all_types()`: 97.5x speedup via module-level caching
    - `get_parent_type()`: 35.9x speedup via cached reverse lookup map
    - Tag validation: 47.3% speedup (eliminated double parsing)
  - **Security Improvements**: Template-based SQL query building
  - **Testing**: 80 comprehensive tests with 100% backward compatibility
  - **Breaking Change**: Legacy types (task, note, standard) deprecated, automatically migrated via soft-validation (defaults to 'observation')

- **Response Size Limiter** (PR #344)
  - Added `max_response_chars` parameter to 5 memory retrieval tools to prevent context window overflow
  - Tools updated: `retrieve_memory`, `recall_memory`, `retrieve_with_quality_boost`, `search_by_tag`, `recall_by_timeframe`
  - Environment variable: `MCP_MAX_RESPONSE_CHARS` (default: 0 = unlimited)
  - Truncates at memory boundaries (never mid-content) to preserve data integrity
  - Always returns at least one memory even if it exceeds the character limit
  - Displays truncation warning with shown/total counts when limit is applied
  - New utility module: `src/mcp_memory_service/server/utils/response_limiter.py` (260 lines)
  - Comprehensive test coverage: 29 tests (415 lines) covering edge cases
  - Recommended values: 30000-50000 characters for typical LLM context windows
  - Backward compatible: existing code continues to work with unlimited responses

## [8.76.0] - 2026-01-12

### Added
- **Official Lite Distribution Support** (PR #341)
  - **New Package**: `mcp-memory-service-lite` - Official lightweight distribution for ONNX-only installations
    - 90% installation size reduction: 7.7GB → 805MB
    - Same nvidia-quality-classifier-deberta ONNX model, just lighter dependency chain
    - Faster installation time: <2 minutes vs 10-15 minutes
  - **Dual Package Publishing**: Automated CI/CD workflow (`publish-dual.yml`) publishes both packages to PyPI
    - Full package: `mcp-memory-service` (includes transformers, torch, sentence-transformers)
    - Lite package: `mcp-memory-service-lite` (ONNX-only, tokenizers-based embeddings)
    - Both packages share the same codebase via pyproject-lite.toml
  - **Conditional Dependency Loading**: Transformers becomes truly optional
    - Quality scoring works with tokenizers-only (lite) or full transformers (full package)
    - Graceful fallback: embedding service detects available packages and loads accordingly
    - No runtime performance impact - same quality scoring performance
  - **Implementation Details**:
    - Created `pyproject-lite.toml` with minimal dependencies (no transformers/torch)
    - Updated `onnx_ranker.py` to use tokenizers directly instead of transformers
    - Fixed quality_provider metadata access bug in async_scorer.py
    - 15 comprehensive integration tests (`tests/test_lightweight_onnx.py`, 487 lines)
  - **Documentation**:
    - Complete setup guide: `docs/LIGHTWEIGHT_ONNX_SETUP.md`
    - Setup script: `scripts/setup-lightweight.sh`
  - **Use Cases**:
    - CI/CD pipelines (faster builds, lower disk usage)
    - Resource-constrained environments (VPS, containers)
    - Quick local development setup
    - Users who only need quality scoring (not full ML features)
- **Production Refactor Command**: Added `/refactor-function-prod` command for production-ready code refactoring
  - Enhanced version of the refactor-function PoC with production features
- **Refactoring Metrics Documentation**: Comprehensive Issue #340 refactoring documentation in `.metrics/`
  - Baseline complexity measurements
  - Complexity comparison showing 75.2% average reduction
  - Tracking tables and completion reports

### Fixed
- **Multi-Protocol Port Detection and Cross-Platform Fallback** (`scripts/service/http_server_manager.sh`)
  - **Problem**: Update script failed with port conflict on Linux systems without lsof installed
  - **Root Causes**:
    - Script used lsof exclusively for port detection, silently failed on systems without it (common on Arch/Manjaro)
    - Health checks only tried HTTP protocol, failed when server used HTTPS
    - Led to "server not running" false positive → attempted restart → port conflict error
  - **Solution**:
    - Implemented 4-level port detection fallback chain: lsof → ss → netstat → ps
    - Health check now tries both HTTP and HTTPS protocols with automatic fallback
    - Explicit error messages when all detection methods fail
    - Cross-platform compatibility validated on Arch Linux with ss-only environment
  - **Testing**: Manual testing on Arch Linux without lsof, confirmed fallback to ss works correctly
  - **Impact**: Resolves installation failures on minimal Linux distributions, improves HTTPS deployment reliability
- **MCP HTTP Transport**: Fix KeyError 'backend_info' in get_cache_stats tool (Issue #342, PR #343)
  - **Problem**: `get_cache_stats` tool crashed with `KeyError: 'backend_info'` when called via HTTP transport
  - **Root Cause**: Code tried to set `result["backend_info"]["embedding_model"]` without creating the dict first
  - **Solution**: Create complete `backend_info` dict with all required fields (storage_backend, sqlite_path, embedding_model)
  - **Impact**: HIGH severity (tool completely broken in HTTP transport), LOW risk fix
  - **Testing**: Added regression test validating backend_info structure
  - Thanks to @Sundeepg98 for reporting with clear reproduction steps!
- **Hook Installer**: Auto-register PreToolUse hook in settings.json (Issue #335)
  - **Problem**: `permission-request.js` was copied but never registered in `settings.json`, so the hook never executed
  - **Solution**: Installer now auto-adds PreToolUse hook configuration for MCP permission management
  - Added 7 new safe patterns: `store`, `remember`, `ingest`, `rate`, `proactive`, `context`, `summary`, `recommendations`
  - Hook now correctly auto-approves additive operations (e.g., `store_memory`)
- **Memory Hooks**: Fix cluster memory categorization showing incorrect dates
  - **Problem**: Consolidated cluster memories (from consolidation system) showed "🕒 today" in "Recent Work" section because hook used `created_at` (when cluster was created) instead of `temporal_span` (time period the cluster represents)
  - **Solution**: Added new "📦 Consolidated Memories" section with proper temporal span display (e.g., "📅 180d span")
  - **Changes**: Updated `claude-hooks/utilities/context-formatter.js` to detect `compressed_cluster` memory type and display `metadata.temporal_span.span_days`
  - **Impact**: Prevents confusion between recent development and historical memory summaries

### Changed
- **Hook Installer**: Refactored MCP configuration detection functions for improved maintainability (Issue #340)
  - Reduced complexity by 45.5% across 3 core functions
  - Extracted 6 well-structured helper functions (avg complexity 3.83)
  - Fixed validation bug: Added 'detected' server type support (PR #339 follow-up)
  - Improved code grade distribution: 58% A-grade functions (up from 53%)
- **Memory Consolidation**: Improved cluster concept quality with intelligent deduplication
  - **Problem**: Cluster summaries showed redundant concepts (e.g., "memories, Memories, MEMORY, memory") and noise (SQL keywords like "BETWEEN")
  - **Solution**: Added case-insensitive deduplication and filtering of SQL keywords/meta-concepts
  - **Changes**: Enhanced `_extract_key_concepts()` in `consolidation/compression.py` to deduplicate case variants and filter 10 SQL keywords + 6 meta-concepts
  - **Impact**: Cluster summaries now show meaningful thematic concepts instead of technical noise

## [8.75.1] - 2026-01-10

### Fixed
- **Hook Installer**: Support flexible MCP server naming conventions (PR #339)
  - **Problem**: Installer required exact server name `memory`, causing installation failures for users with custom MCP configurations (e.g., `mcp-memory-service`, `memory-service`)
  - **Solution**:
    - Installer now detects servers matching patterns: `memory`, `mcp-memory`, `*memory*service*`, `*memory*server*`
    - Backward compatible with existing `memory` server name
    - Provides clear error messages when no matching server found
    - Improved user experience for custom MCP configurations
  - **Testing**: Manual testing with various server name configurations
  - **Contributors**: Thanks to @timkjr for reporting and testing the fix!

## [8.75.0] - 2026-01-09

### Added
- **Lightweight ONNX Quality Scoring without Transformers Dependency** (PR #337)
  - **Problem**: transformers package adds 6.9GB of dependencies (torch, tensorflow, etc.), making installation bloated for users who only need quality scoring
  - **Solution**: Use tokenizers package directly instead of transformers
    - 90% disk space reduction: 7.7GB → 805MB total installation
    - Same ONNX model (nvidia-quality-classifier-deberta), just lighter dependency chain
    - Conditional dependency loading - only install what you use
  - **Implementation**:
    - Modified `src/mcp_memory_service/quality/onnx_ranker.py` to use tokenizers directly
    - Added tokenizers as optional dependency in pyproject.toml
    - Updated embedding service to handle both tokenizers and transformers (graceful fallback)
    - Fixed quality_provider metadata access bug in async_scorer.py
  - **Testing**: 15 comprehensive integration tests (`tests/test_lightweight_onnx.py`, 487 lines)
  - **Documentation**:
    - Complete setup guide: `docs/LIGHTWEIGHT_ONNX_SETUP.md`
    - Setup script: `scripts/setup-lightweight.sh`
  - **Benefits**:
    - Faster installation (<2 min vs 10-15 min)
    - Lower disk usage (805MB vs 7.7GB)
    - Same quality scoring performance
    - No runtime performance impact

### Fixed
- **Multi-Protocol and Cross-Platform Port Detection** (`scripts/service/http_server_manager.sh`)
  - **Problem**: Update script failed with port conflict on Linux systems without lsof installed
  - **Root Cause**:
    - Script used lsof exclusively, silently failed on systems without it (common on Arch/Manjaro)
    - Health checks only tried HTTP, failed when server used HTTPS
    - Led to "server not running" false positive → port conflict
  - **Solution**:
    - Port detection fallback chain: lsof → ss → netstat → ps
    - Health check supports both HTTP and HTTPS protocols with automatic fallback
    - Tested on Arch Linux with ss-only environment
  - **Fixes**: Issue #341

## [8.74.0] - 2026-01-09

### Added
- **Cross-Platform Wrapper Scripts for Orphan Process Cleanup** (`scripts/run/`)
  - **Python Wrapper** (`memory_wrapper_cleanup.py`): Universal cross-platform solution (recommended)
  - **Bash Wrapper** (`memory_wrapper_cleanup.sh`): Native macOS/Linux implementation
  - **PowerShell Wrapper** (`memory_wrapper_cleanup.ps1`): Native Windows implementation
  - **Automatic Orphan Detection**: Identifies orphaned MCP memory processes when Claude Desktop/Code crashes
    - Unix: Detects processes adopted by init/launchd (ppid == 1)
    - Windows: Detects processes whose parent no longer exists
  - **Database Lock Prevention**: Cleans up orphaned processes before starting new server instance
  - **Safe Cleanup**: Only terminates actual orphans, preserves active sessions
  - **Comprehensive Documentation** (`README_CLEANUP_WRAPPER.md`): Installation guide, troubleshooting, and technical details
  - **Fixes**: SQLite "database is locked" errors when running multiple Claude instances after crashes

## [8.73.0] - 2026-01-09

### Added
- **Universal Permission Request Hook v1.0** (`claude-hooks/core/permission-request.js`)
  - **Automatic MCP Tool Permission Management**: Eliminates repetitive permission prompts for safe operations
  - **Safe-by-Default Design**: Auto-approves read-only operations (get, list, retrieve, search, query, etc.)
  - **Destructive Operation Protection**: Requires confirmation for write/delete operations (delete, update, execute, etc.)
  - **Universal MCP Server Support**: Works across all MCP servers (memory, browser, context7, playwright, etc.)
  - **Zero Configuration**: Works out of the box with sensible defaults
  - **Extensible Pattern Matching**: Configure custom patterns via `claude-hooks/config.json`
  - **Comprehensive Documentation**: Installation guide, configuration reference, and troubleshooting
  - **Installer Integration**: Added to core hooks installation via `install_hooks.py`
- **Sync Status Maintenance Script** (`scripts/maintenance/sync_status.py`)
  - Compares local SQLite database with Cloudflare D1
  - Shows memory counts (total, active, tombstones) for both databases
  - Identifies sync discrepancies: local-only, D1-only, pending deletions
  - Cross-platform support (Windows, macOS, Linux)
  - Handles schema differences gracefully

### Fixed
- **Cloudflare D1 Schema Missing Columns** (`src/mcp_memory_service/storage/cloudflare.py`)
  - **Problem**: D1 schema was missing `deleted_at` and `tags` columns, causing soft-delete operations to fail silently
  - **Impact**: Tombstone-based sync between devices was broken - deleted memories on one device were not synced to others
  - **Fix**: Added `deleted_at REAL DEFAULT NULL` and `tags TEXT` columns to D1 schema
  - **Index**: Added `idx_memories_deleted_at` index for query performance

## [8.72.0] - 2026-01-08

### Added
- **Graph Traversal MCP Tools** (PR #332)
  - **Feature**: Three new MCP tools for querying memory association graph directly
  - **Tools**:
    - `find_connected_memories` - Multi-hop connection discovery (5ms, 30x faster than tag search)
    - `find_shortest_path` - BFS pathfinding between two memories (15ms)
    - `get_memory_subgraph` - Subgraph extraction for visualization (25ms)
  - **Implementation**:
    - New handler module: `src/mcp_memory_service/server/handlers/graph.py` (380 lines)
    - Integration: `server_impl.py` tool registration and routing (134 lines)
    - Graceful fallback if `memory_graph` table unavailable
  - **Testing**: 10 comprehensive tests with 100% handler coverage (`tests/test_graph_traversal.py`, 328 lines, 58 assertions)
  - **Validation**: `scripts/validation/validate_graph_tools.py` for MCP schema and query correctness
  - **Quality Metrics**:
    - Health Score: 85/100 (EXCELLENT)
    - Complexity: Average 4.5 (Grade A)
    - Security: 0 vulnerabilities
  - **Benefits**:
    - 30x performance improvement over tag-based connected memory search
    - Enables Claude Code to traverse memory associations directly via MCP
    - Supports exploration of knowledge relationships and context chains

## [8.71.0] - 2026-01-06

### Added
- **Memory Management APIs and Graceful Shutdown** (PR #331)
  - **Problem**: Orphaned MCP sessions consuming excessive memory (reported in Discussion #331)
  - **Cache Cleanup Functions** (`src/mcp_memory_service/services/cache_manager.py`):
    - `clear_all_caches()` - Clear storage and service caches
    - `get_memory_usage()` - Get process memory statistics (RSS, VMS, available memory)
    - `get_cache_stats()` - Get cache hit/miss statistics
  - **Model Cache Cleanup** (`src/mcp_memory_service/storage/sqlite_vec.py`):
    - `clear_model_caches()` - Clear embedding model caches
    - `get_model_cache_stats()` - Get model cache statistics
  - **Graceful Shutdown** (`src/mcp_memory_service/server/server_impl.py`):
    - `_cleanup_on_shutdown()` - Cleanup function for signal handlers
    - Updated `shutdown()` method with proper cache and connection cleanup
    - Added `atexit` handler for normal process exit
    - SIGTERM and SIGINT handlers now call cleanup before exit
  - **Memory Management API Endpoints** (`src/mcp_memory_service/api/routes/health.py`):
    - `GET /api/memory-stats` - Get detailed memory usage (process memory + cache stats + model cache stats)
    - `POST /api/clear-caches` - Clear all caches manually (returns memory freed)
  - **Documentation**:
    - New file: `docs/troubleshooting/memory-management.md` - Comprehensive guide for monitoring and managing memory
  - **Benefits**:
    - Prevents memory leaks from orphaned sessions
    - Enables proactive memory monitoring and cleanup
    - Graceful resource release on server shutdown
    - Production-ready memory management for long-running deployments

## [8.70.0] - 2026-01-05

### Added
- **User Override Commands for Memory Hooks** (`~/.claude/hooks/utilities/user-override-detector.js`)
  - **Feature**: New `#skip` and `#remember` commands give users manual control over automatic memory triggers
  - **Implementation**:
    - Shared detection module for consistent behavior across all hooks
    - `#skip` - Bypasses memory retrieval in session-start hook
    - `#remember` - Forces mid-conversation analysis (bypasses cooldown) or session-end consolidation (bypasses thresholds)
    - Applied to: `session-start.js`, `mid-conversation.js`, `session-end.js`
  - **Configuration**: New `applyTo` array in `config.json` defines which hooks are active
  - **Documentation**: Updated `README.md` with "User Overrides" section and `README-AUTO-CAPTURE.md` with comprehensive supported hooks table
  - **Use Cases**:
    - Skip retrieval when starting fresh conversation (`#skip`)
    - Force memory capture of important mid-session insights (`#remember`)
    - Override 100-character threshold for critical session notes (`#remember`)

### Added (from v8.69.1-dev)
- **Automatic Test Memory Cleanup System** (`tests/conftest.py`, `tests/api/test_operations.py`)
  - **Problem**: Test memories polluted production database (106+ orphaned test memories found)
  - **Solution**: Tag-based cleanup system
    - Reserved `__test__` tag for all test-created memories
    - New `test_store` fixture auto-tags memories for cleanup
    - `pytest_sessionfinish` hook automatically deletes tagged memories at end of test session
    - New `delete_by_tag()` function in Code Execution API
  - **Result**: 92 test memories automatically cleaned after test run
  - **Migration**: Tests using `store()` should migrate to `test_store()` fixture

### Fixed (from v8.69.1-dev)
- **PowerShell Update Script Git Stderr Handling** (`update_and_restart.ps1`)
  - **Problem**: Script failed with `NativeCommandError` during `git pull --rebase`
  - **Root Cause**: Git writes informational messages to stderr even on success (e.g., "From github.com:..."), and `$ErrorActionPreference = "Stop"` treats any stderr as error
  - **Solution**: Temporarily set `ErrorActionPreference = "Continue"` during git pull, then check `$LASTEXITCODE` for actual errors

---

**For older releases (v8.69.0 and earlier)**, see [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).
