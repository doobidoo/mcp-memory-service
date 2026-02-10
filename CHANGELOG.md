# Changelog

**Recent releases for MCP Memory Service (v10.0.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

For older releases (v9.3.1 and earlier), see [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [10.10.6] - 2026-02-10

### Fixed
- **Test Infrastructure (#317):** Fixed TypedDict import for Python 3.11 compatibility - resolved Pydantic v2.12 error blocking test collection
  - Changed `from typing import TypedDict` to `from typing_extensions import TypedDict` for Python 3.11 support
  - **Result:** All tests now run successfully on Python 3.11+
- **Performance Tests (#318):** Re-enabled pytest-benchmark performance tests with dev dependencies
  - Added pytest-benchmark to dev dependencies in pyproject.toml
  - **Result:** Performance benchmarking now available for development

### Documentation
- **Issue Triage (#91, #261):** Updated status documentation for ontology and quality system milestones
  - #91: Phase 0 ontology improvements 97% complete (6,542 memories processed)
  - #261: Phase 1 quality system complete, Phase 2 decision needed on Rust vs Python
- **Coverage Baseline (#317):** Established 60.05% coverage baseline with 4-phase improvement plan
  - Phase 1: Core storage/embeddings to 75%+
  - Phase 2: Services to 70%+
  - Phase 3: Server/API to 65%+
  - Phase 4: Utils/edge cases to 60%+

## [10.10.5] - 2026-02-10

### Fixed
- **Embedding Dimension Cache (#412):** Fixed embedding dimension not being restored from model cache, causing dimension mismatches (384 vs 768) across multiple storage instances
  - Added `_DIMENSION_CACHE` dictionary alongside `_MODEL_CACHE` to track embedding dimensions
  - Store dimension when caching models (external API, ONNX, SentenceTransformer)
  - Restore dimension when retrieving cached models
  - **Result:** Consistent embedding dimensions across all storage instances

## [10.10.4] - 2026-02-10

### Fixed
- **CLI Batch Ingestion (#447):** Fixed async bug in `ingest-directory` command causing "NoneType object can't be awaited" errors
  - Made `sqlite_vec.close()` async to match other storage backend interfaces
  - **Result:** CLI batch ingestion now works at 100% success rate (was 9.4% before fix)
- **Test Infrastructure (#451):** Fixed graph visualization validation and test authentication setup
  - **Graph Visualization:** Tightened limit validation from le=1000 to le=500 for consistency with test expectations
  - **Test Authentication:** Fixed module import order causing 15 tests to fail with 401 errors
  - **Result:** All 15 tests now pass (previously 14 failed, 1 xpassed)

## [10.10.3] - 2026-02-10

### Fixed
- **Test Infrastructure (#451):** Fixed test_analytics_graph.py failures causing CI pipeline failures
  - **Graph Visualization Validation:** Tightened limit validation from le=1000 to le=500 for consistency with test expectations and other API limits
  - **Test Authentication:** Fixed module import order issue where config loaded before test environment variables, causing all 15 tests to fail with 401 errors
  - **Result:** All 15 tests now pass (previously 14 failed, 1 xpassed)
- **Memory Scoring (#450):** Fixed score inflation in memory-scorer.js - capped finalScore to 1.0 before applying 0.5x penalty to prevent bonus inflation while preserving cross-project technology sharing

## [10.10.2] - 2026-02-10

### Fixed
- **Memory Injection Filtering (#449):** Fixed two critical bugs preventing proper memory filtering for empty/new projects
  - **minRelevanceScore Enforcement:** Applied configured relevance threshold (default 0.3) in memory scoring filter - was loaded but never enforced, allowing low-relevance cross-project memories (scored ~12% after 85% penalty) to pass through
  - **Project-Affinity Filter:** Added Phase 2 tag-based search filter to prevent cross-project memory pollution - tag searches now require project tag presence or project name mention in content
  - Generic tags (architecture, key-decisions, claude-code-reference) previously returned memories from ALL projects due to OR logic in `/api/search/by-tag` endpoint

### Security
- **Command Injection Prevention (#449):** Replaced `execSync` with `execFileSync` in memory service queries to prevent command injection via project names
- **Log Sanitization (#449):** Added `sanitizeForLog()` function to strip ANSI/control characters from logged project names
- **Null Guards (#449):** Added defensive null/empty checks for `projectTag` in affinity filter

## [10.10.1] - 2026-02-09

### Fixed
- **Search Handler (#444, #446):** Fixed AttributeError in memory_search - handle dict results correctly when storage backend returns dictionaries instead of SearchResult objects
- **Import Error (#443):** Fixed response_limiter import path in server/handlers/memory.py (max_response_chars feature now works)
- **Security (#441):** Added allowlist validation to maintenance scripts (SQL injection prevention in soft_delete_test_memories.py and migration scripts)

### Changed
- **Exact Search Mode (#445):** Changed to case-insensitive substring matching (LIKE) instead of full-content equality for more intuitive search behavior
  - **BREAKING CHANGE:** Users relying on exact full-match behavior may need to adjust queries
  - Previous: `mode=exact` matched only if entire content was identical
  - New: `mode=exact` performs case-insensitive substring search using SQL LIKE operator

### Known Issues
- **CLI Async (#447):** ingest-directory async handling under investigation - deferred to separate PR

## [10.10.0] - 2026-02-08

### Added
- **Environment Configuration Viewer**: New Settings Panel tab for comprehensive environment configuration visibility
  - **New API Endpoint**: `GET /api/config/env` returns 11 categorized parameter groups
  - **11 Configuration Categories**: Storage Backend, Database Settings, HTTP Server, Security, Quality System, Consolidation, OAuth, Embeddings, Logging, Graph Storage, Advanced Settings
  - **Security Features**: Sensitive value masking for API tokens, keys, and credentials
  - **User Experience**: Copy-to-clipboard functionality, dark mode optimization, organized accordion layout
  - **1405+ Lines**: 5 files modified (config_env.py, config.html, config.css, config.js, analytics.py)
  - **Use Cases**: Configuration troubleshooting, team onboarding, environment verification
- **Changelog Archival Agent**: New `changelog-archival` agent automates archival of older CHANGELOG entries when main file exceeds ~1000 lines
  - Automated version boundary detection and safe file splitting
  - Preserves all content in `docs/archive/CHANGELOG-HISTORIC.md`
  - Triggered after major version milestones or on explicit request
  - Maintains lean CHANGELOG focused on recent releases

### Enhanced
- **Graph Visualization Enrichment**: Added quality scores, updated timestamps, and metadata to graph nodes
  - Parse `metadata` JSON field for enriched node information (quality_score extraction)
  - Include `updated_at` timestamp for tracking node freshness
  - Increased max node limit from 500 to 1000 for larger graph visualization
  - Improved data completeness for analytics and visualization rendering

### Fixed
- **Installation Script Errors** (PR #439): Fixed three critical bugs preventing successful installation
  - Fixed NameError: `install_package` function now properly defined
  - Fixed ModuleNotFoundError: GPU detection uses `importlib.util` for direct file loading to avoid package __init__.py dependency issues during installation
  - Fixed Cloudflare API 401 error: Updated token verification to use account-specific endpoint with proper fallback to user-level endpoint
  - Added safety validations: spec/loader existence checks, account_id validation with `.strip()`
  - **Contributors:** @sykuang

## [10.9.0] - 2026-02-08

### Added
- **Batched Inference for Consolidation Pipeline** (PR #432): 4-16x performance improvement with GPU support
  - **GPU Performance**: 4-16x speedup on RTX 5050 Blackwell (1.4ms/item for batch=16, 0.7ms/item for batch=32 vs 5.2ms/item sequential)
  - **CPU Performance**: 2.3-2.5x speedup with batched ONNX inference
  - **Adaptive GPU Dispatch**: Automatically falls back to sequential processing for small batches (<16 items) to avoid padding overhead
  - **Configuration**:
    - `MCP_QUALITY_BATCH_SIZE` (default: 32) - controls batch size for quality scoring
    - `MCP_QUALITY_MIN_GPU_BATCH` (default: 16) - minimum batch size to use GPU (below threshold uses CPU sequential)
  - **New Methods**:
    - `ONNXRankerModel.score_quality_batch()` - batched DeBERTa classifier and MS-MARCO cross-encoder scoring
    - `QualityEvaluator.evaluate_quality_batch()` - two-pass batch evaluation with fallback strategy
    - `SqliteVecMemoryStorage.store_batch()` - batched embedding generation with atomic transaction handling
    - `SemanticCompressionEngine` - parallel cluster compression via `asyncio.gather`
  - **Backward Compatibility**: 100% backward compatible, set `MCP_QUALITY_BATCH_SIZE=1` for instant rollback
  - **Test Coverage**: 442 lines of new tests (17 tests in `test_batch_inference.py`)
  - **Contributors**: @rodboev

### Fixed
- **Token Truncation**: Fixed character-based [:512] truncation to proper tokenizer truncation (PR #432)
  - **Root Cause**: Classifier scoring truncated at 512 characters before tokenization, discarding ~75% of DeBERTa's 512-token context window (~2000 chars)
  - **Fix**: Enable `truncation(max_length=512)` at tokenizer initialization for all paths (classifier, cross-encoder, sequential, batch)
  - **Impact**: Better quality scores by utilizing full model context window
- **Embedding Orphan Prevention**: Fixed orphaned embeddings in `store()` and `store_batch()` methods (PR #432)
  - **Root Cause**: When embedding INSERT failed, fallback inserted without rowid, breaking `memories.id <-> memory_embeddings.rowid` JOIN
  - **Fix**: Wrap both memory INSERT and embedding INSERT in SAVEPOINT for atomicity - both succeed or both roll back
  - **Impact**: All memories now guaranteed to be searchable via semantic search
- **ONNX Float32 GPU Compatibility**: Cast model to float32 before ONNX export (PR #437)
  - **Root Cause**: DeBERTa v3 stores some weights in float16, producing mixed-precision ONNX graph that ONNX Runtime rejects
  - **Error Message**: "Type parameter (T) of Optype (MatMul) bound to different types (tensor(float16) and tensor(float))"
  - **Fix**: Add `model.float()` after `model.eval()` to cast all parameters to float32 before export
  - **Validation**: Tested on RTX 5050 Blackwell with PyTorch 2.10.0+cu128, ONNX Runtime 1.24.1 (CUDAExecutionProvider)
  - **Contributors**: @rodboev
- **Concurrent Write Stability**: Increased retry budget for WAL mode write contention (PR #435)
  - **Root Cause**: 3 retries with 0.1s initial delay (~0.7s total backoff) insufficient for multiple connections contending for SQLite RESERVED write lock
  - **Fix**: Bumped to 5 retries with 0.2s initial delay (~6.2s total backoff: 0.2+0.4+0.8+1.6+3.2)
  - **Impact**: `test_two_clients_concurrent_write` now passes consistently under WAL mode
  - **Contributors**: @rodboev

## [10.8.0] - 2026-02-08

### Added
- **Hybrid BM25 + Vector Search** (Issue #175, PR #436): Combines keyword matching with semantic search for improved exact match scoring
  - FTS5-based BM25 keyword search with trigram tokenizer for multilingual support
  - Parallel execution of BM25 and vector searches (<15ms typical latency)
  - Configurable score fusion weights (default: 30% keyword, 70% semantic)
  - Automatic FTS5 index synchronization via database triggers
  - Backward compatible: existing `mode="semantic"` searches unchanged
  - Available via unified search interface: `mode="hybrid"`
  - Configuration options:
    - `MCP_HYBRID_SEARCH_ENABLED`: Enable hybrid search (default: true)
    - `MCP_HYBRID_KEYWORD_WEIGHT`: BM25 weight (default: 0.3)
    - `MCP_HYBRID_SEMANTIC_WEIGHT`: Vector weight (default: 0.7)
  - Comprehensive test suite: 12 tests covering unit, integration, performance
  - Performance: <50ms average latency for 100 memories
  - Reference implementation: AgentKits Memory (sub-10ms latency, 70% token efficiency gains)

### Fixed
- **search_memories() Response Format** (PR #436): Corrected pre-existing bug where `search_memories()` returned Memory objects instead of dictionaries with `similarity_score`
  - Now returns flat dictionaries with all memory fields plus `similarity_score`
  - Updated affected tests (test_issue_396, hybrid search tests) to use dictionary access
  - Maintains API specification compliance
- **Test Safety: Prevent accidental Cloudflare data deletion** (Commit d3d8425): Tests now force `sqlite_vec` backend to prevent soft-deleting production memories in Cloudflare D1
  - Automatically overrides `MCP_MEMORY_STORAGE_BACKEND` to `sqlite_vec` for all test runs
  - Prints warning when overriding a cloud backend setting
  - Allows explicit cloud testing via `MCP_TEST_ALLOW_CLOUD_BACKEND=true`
- **Test Safety: Comprehensive safeguards to prevent production database deletion** (PR #438): Implemented triple safety system after incident on 2026-02-08 where test cleanup deleted 8,663 production memories
  - **Forced Test Database Path**: Creates isolated temp directory with `mcp-test-` prefix at module import time, forces `MCP_MEMORY_SQLITE_PATH` to test database
  - **Pre-Test Verification**: `pytest_sessionstart` hook verifies database is in temp directory, **aborts test run** if production path detected
  - **Triple-Check Cleanup**: `pytest_sessionfinish` validates (1) temp directory location, (2) no production indicators, (3) test markers present
  - **Critical Change**: Removed `allow_production=True` bypass flag - now relies on `delete_by_tag`'s own safety checks
  - **Additional Safeguards**: Backend forced to `sqlite_vec` unless explicitly allowed, verbose logging, explicit error handling
  - **Impact**: Defense in depth with 4 layers - environment override, pre-test abort, triple-check cleanup, backend-level safety
  - **Files Modified**: `tests/conftest.py` (+130 lines of safety code)
- **Dashboard Version Display**: Updated `src/mcp_memory_service/_version.py` to v10.8.0 (was showing v10.7.2 in dashboard)

### Added
- **Hybrid BM25 + Vector Search** (Issue #175, PR #436): Combines keyword matching with semantic search for improved exact match scoring
  - FTS5-based BM25 keyword search with trigram tokenizer for multilingual support
  - Parallel execution of BM25 and vector searches (<15ms typical latency)
  - Configurable score fusion weights (default: 30% keyword, 70% semantic)
  - Automatic FTS5 index synchronization via database triggers
  - Backward compatible: existing `mode="semantic"` searches unchanged
  - Available via unified search interface: `mode="hybrid"`
  - Configuration options:
    - `MCP_HYBRID_SEARCH_ENABLED`: Enable hybrid search (default: true)
    - `MCP_HYBRID_KEYWORD_WEIGHT`: BM25 weight (default: 0.3)
    - `MCP_HYBRID_SEMANTIC_WEIGHT`: Vector weight (default: 0.7)
  - Comprehensive test suite: 12 tests covering unit, integration, performance
  - Performance: <50ms average latency for 100 memories
  - Reference implementation: AgentKits Memory (sub-10ms latency, 70% token efficiency gains)

### Fixed
- **search_memories() Response Format** (PR #436): Corrected pre-existing bug where `search_memories()` returned Memory objects instead of dictionaries with `similarity_score`
  - Now returns flat dictionaries with all memory fields plus `similarity_score`
  - Updated affected tests (test_issue_396, hybrid search tests) to use dictionary access
  - Maintains API specification compliance
- **Test Safety: Prevent accidental Cloudflare data deletion** (Commit d3d8425): Tests now force `sqlite_vec` backend to prevent soft-deleting production memories in Cloudflare D1
  - Automatically overrides `MCP_MEMORY_STORAGE_BACKEND` to `sqlite_vec` for all test runs
  - Prints warning when overriding a cloud backend setting
  - Allows explicit cloud testing via `MCP_TEST_ALLOW_CLOUD_BACKEND=true`

## [10.7.2] - 2026-02-07

### Fixed
- **Server Management buttons cause page reload** (PR #429): Added `type="button"` to Check for Updates, Update & Restart, and Restart Server buttons in Settings modal. Without explicit type, buttons inside `<form>` default to `type="submit"`, causing unintended form submission and page reload.

## [10.7.1] - 2026-02-07

### Fixed
- **Test Script Security Hardening** (Issue #419, PR #427): Hardened test backup scripts with several security and robustness fixes:
  - **Command Injection (HIGH)**: Replaced unquoted heredoc with quoted `<< 'EOF'` + `printf '%q'` for safe path escaping
  - **Argument Injection**: Added `--` separators to `sqlite3` and `rm` commands
  - **Database Corruption Risk**: Replaced `cp` with `sqlite3 .backup` for atomic, WAL-safe backups
  - **Overly Broad pkill**: Narrowed pattern from `"memory server"` to `"mcp_memory_service"`
  - **Cross-Platform Documentation**: Added platform-specific path table (macOS/Linux/Windows) to README
  - **Test Fix**: Added auth dependency overrides to quality HTTP endpoint tests
- **Dashboard Authentication for API Calls** (Commit 5bf4834): Added authentication headers to all Dashboard API endpoints
  - **Frontend Fixes**: Replaced 19 direct `fetch()` calls with `this.apiCall()` for proper auth header handling
  - **Affected Tabs**: Manage, Analytics, Quality tabs now properly authenticate
  - **FormData Uploads**: Added auth headers to 2 document upload fetch calls
  - **API Middleware**: Added auth middleware to consolidation API (3 endpoints: trigger, status, recommendations)
  - **API Middleware**: Added auth middleware to quality API (5 endpoints: rate, evaluate, get, distribution, trends)
  - **API Overview Page**: Fixed `/api-overview` page to pass auth headers to `/api/health/detailed` fetch calls
  - **Root Cause**: Direct fetch() calls bypassed the apiCall() auth layer, causing 401 errors when API key authentication was enabled
  - **Files Changed**: 5 files modified (app.js, index.html, consolidation.py, quality.py, app.py)
  - **Impact**: All Dashboard features now work correctly with API key authentication enabled

## [10.7.0] - 2026-02-07

### Added
- **Backup UI Enhancements** (Issue #375, PR #375): Complete backup management interface with View Backups modal
  - **View Backups Modal**: Interactive backup history showing filename, size, date, and age
  - **Backup Directory Display**: Shows backup directory path in summary for easy file access
  - **API Enhancement**: Added `backup_directory` field to BackupStatusResponse API
  - **User Experience**: Cleaner, more informative backup management interface

### Fixed
- **Backup Form Controls**: Fixed backup buttons using `type="button"` to prevent unintended form submission
- **Event Binding**: Switched to inline onclick handlers for reliable event binding in settings modal
- **Toast Notifications**: Fixed toast pointer-events to ensure notifications appear over modals
- **Cache Busting**: Updated cache-busters for static assets to ensure proper browser refresh

## [10.6.1] - 2026-02-07

### Fixed
- **Dashboard SSE Authentication** (Issue #420, PR #423): Fixed EventSource API authentication for Server-Sent Events
  - **Root Cause**: Browser `EventSource` API does not support custom headers, causing 401 errors on `/api/events` endpoint
  - **Solution**: Pass authentication credentials as URL query parameters (`api_key=` and `token=` for OAuth)
  - **Implementation**: Used `URL` and `URLSearchParams` APIs for clean query parameter handling
  - **OAuth Support**: Added `token` query parameter support in auth middleware for SSE connections
  - **Security**: Added `<meta name="referrer" content="no-referrer">` to prevent API key leakage via HTTP Referer headers
  - **Impact**: Dashboard now maintains real-time SSE connection when authentication is enabled

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

