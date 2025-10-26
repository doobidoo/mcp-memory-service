# Changelog

**Recent releases for MCP Memory Service (v6.16.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

For older releases, see [CHANGELOG-HISTORIC.md](./CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [8.7.1] - 2025-10-26

### Fixed
- **Cloudflare Vectorize Deletion** - Fixed vector deletion endpoint bug (src/mcp_memory_service/storage/cloudflare.py:671)
  - **Problem**: Used incorrect endpoint `/delete-by-ids` (hyphens) causing 404 Not Found errors, preventing vector deletion
  - **Solution**:
    - Changed to correct Cloudflare API endpoint `/delete_by_ids` (underscores)
    - Fixed payload format from `[vector_id]` to `{"ids": [vector_id]}`
    - Created working cleanup script: `scripts/maintenance/delete_orphaned_vectors_fixed.py`
    - Removed obsolete broken script: `scripts/maintenance/delete_orphaned_vectors.py`
  - **Impact**: Successfully deleted 646 orphaned vectors from Vectorize in 7 batches
  - **Testing**: Verified with production data (646 vectors, 100/batch, all mutations successful)
  - **Discovery**: Found via web research of official Cloudflare Vectorize API documentation

## [8.7.0] - 2025-10-26

### Fixed
- **Cosine Similarity Migration** - Fixed 0% similarity scores in search results (src/mcp_memory_service/storage/sqlite_vec.py:187)
  - **Problem**: L2 distance metric gave 0% similarity for all searches due to score calculation `max(0, 1-distance)` returning 0 for distances >1.0
  - **Solution**:
    - Migrated embeddings table from L2 to cosine distance metric
    - Updated score calculation to `1.0 - (distance/2.0)` for cosine range [0,2]
    - Added automatic migration logic with database locking retry (exponential backoff)
    - Implemented `_initialized` flag to prevent multiple initialization
    - Created metadata table for storage configuration persistence
  - **Performance**: Search scores improved from 0% to 70-79%, exact match accuracy 79.2% (was 61%)
  - **Impact**: 2605 embeddings regenerated successfully

- **Dashboard Search Improvements** - Enhanced search threshold handling (src/mcp_memory_service/web/static/app.js:283)
  - Fixed search threshold always being sent even when not explicitly set
  - Improved document filtering to properly handle memory object structure
  - Only send `similarity_threshold` parameter when user explicitly sets it
  - Better handling of `memory.memory_type` and `memory.tags` for document results

### Added
- **Maintenance Scripts** - Comprehensive database maintenance tooling (scripts/maintenance/)
  - **regenerate_embeddings.py** - Regenerate all embeddings after migrations (~5min for 2600 memories)
  - **fast_cleanup_duplicates.sh** - 1800x faster duplicate removal using direct SQL (<5s for 100+ duplicates vs 2.5 hours via API)
  - **find_all_duplicates.py** - Fast duplicate detection with timestamp normalization (<2s for 2000 memories)
  - **README.md** - Complete documentation with performance benchmarks, best practices, and troubleshooting

### Technical Details
- **Migration Approach**: Drop-and-recreate embeddings table to change distance metric (vec0 limitation)
- **Retry Logic**: Exponential backoff for database locking (1s → 2s → 4s delays)
- **Performance Benchmark**: Direct SQL vs API operations show 1800x speedup for bulk deletions
- **Duplicate Detection**: Content normalization removes timestamps for semantic comparison using MD5 hashing

## [8.6.0] - 2025-10-25

### Added
- **Document Ingestion System** - Complete document upload and management through web UI (#147, #164)
  - **Single and Batch Upload**: Drag-and-drop or file browser support for PDF, TXT, MD, JSON documents
  - **Background Processing**: Async document processing with progress tracking and status updates
  - **Document Management UI**: New Documents tab in web dashboard with full CRUD operations
  - **Upload History**: Track all document ingestion with status, chunk counts, and file sizes
  - **Document Viewer**: Modal displaying all memory chunks from uploaded documents (up to 1000 chunks)
  - **Document Removal**: Delete documents and their associated memory chunks with confirmation
  - **Search Ingested Content**: Semantic search within uploaded documents to verify indexing
  - **Claude Commands**: `/memory-ingest` and `/memory-ingest-dir` for CLI document upload
  - **API Endpoints**:
    - `POST /api/documents/upload` - Single document upload
    - `POST /api/documents/batch-upload` - Multiple document upload
    - `GET /api/documents/history` - Upload history
    - `GET /api/documents/status/{upload_id}` - Upload status
    - `GET /api/documents/search-content/{upload_id}` - View document chunks
    - `DELETE /api/documents/remove/{upload_id}` - Remove document
    - `DELETE /api/documents/remove-by-tags` - Bulk remove by tags
  - **Files Created**:
    - `src/mcp_memory_service/web/api/documents.py` (779 lines) - Document API
    - `claude_commands/memory-ingest.md` - Single document ingestion command
    - `claude_commands/memory-ingest-dir.md` - Directory ingestion command
    - `docs/development/dashboard-workflow.md` - Development workflow documentation

- **Chunking Configuration Help** - Interactive UI guidance for document chunking parameters
  - Inline help panels with collapsible sections for chunk size and overlap settings
  - Visual diagram showing how overlap works between consecutive chunks
  - Pre-configured recommendations (Default: 1000/200, Smaller: 500/100, Larger: 2000/400)
  - Rule-of-thumb guidelines (15-25% overlap of chunk size)
  - Full dark mode support for all help elements

- **Tag Length Validation** - Server-side validation to prevent data corruption (#174)
  - Maximum tag length enforced at 100 characters
  - Validation on both single and batch upload endpoints
  - Clear error messages showing first invalid tag
  - Frontend filtering to hide malformed tags in display
  - Prevents bloated tags from accidental file path pasting

### Fixed
- **Security Vulnerabilities** - Multiple critical security fixes addressed
  - Path traversal vulnerability in file uploads (use `tempfile.NamedTemporaryFile()`)
  - XSS prevention in tag display and event handlers (escape all user-provided filenames)
  - CSP compliance by removing inline `onclick` handlers, using `addEventListener` instead
  - Proper input validation and sanitization throughout upload flow

- **Document Viewer Critical Bugs** - Comprehensive fixes for document management
  - **Chunk Limit**: Increased from 10 to 1000 chunks (was only showing first 10 of 430 chunks)
  - **Upload Session Persistence**: Documents now viewable after server restart (session optional, uses `upload_id` tag search)
  - **Filename Retrieval**: Get filename from memory metadata when session unavailable
  - **Batch File Size**: Calculate and display total file size for batch uploads (was showing 0.0 KB)
  - **Multiple Confirmation Dialogs**: Fixed duplicate event listeners causing N dialogs for N uploads
  - **Event Listener Deduplication**: Added `documentsListenersSetup` flag to prevent duplicate setup

- **Storage Backend Enhancements** - `delete_by_tags` implementation for document deletion
  - Added `delete_by_tags()` method to `MemoryStorage` base class with error aggregation
  - Optimized `SqliteVecMemoryStorage.delete_by_tags()` with single SQL query using OR conditions
  - Added `HybridMemoryStorage.delete_by_tags()` with sync queue support for cloud backends
  - Fixed return value handling (tuple unpacking instead of dict access)

- **UI/UX Improvements** - Enhanced user experience across document management
  - Added scrolling to Recent Memories section (max-height: 600px) to prevent infinite expansion
  - Document chunk modal now scrollable (max-height: 400px) for long content
  - Modal visibility fixed with proper `active` class pattern and CSS transitions
  - Dark mode support for all document UI components (chunk items, modals, previews)
  - Event handlers for View/Remove buttons in document preview cards
  - Responsive design with mobile breakpoints (768px, 1024px)

- **Resource Management** - Proper cleanup and error handling
  - Temp file cleanup moved to `finally` blocks to prevent orphaned files
  - File extension validation fixed (strip leading dot for consistent checking)
  - Session cleanup timing bug fixed (use `total_seconds()` instead of `.seconds`)
  - Loader registration order corrected (PDFLoader takes precedence as fallback)

- **MCP Server Tag Format Support** - Accept both string and array formats
  - MCP tools now accept `"tag1,tag2"` (string) and `["tag1", "tag2"]` (array)
  - Consistent tag handling between API and MCP endpoints
  - Fixes validation errors from schema mismatches

### Changed
- **API Response Improvements** - Better error messages and status handling
  - Float timestamp handling in document search (convert via `datetime.fromtimestamp()`)
  - Partial success handling for bulk operations with clear error reporting
  - Progress tracking for background tasks with status updates

### Technical Details
- **Testing**: 19 Gemini Code Assist reviews addressed with comprehensive fixes
- **Performance**: Document viewer handles 430+ chunks efficiently
- **Compatibility**: Cross-platform temp file handling (Windows, macOS, Linux)
- **Code Quality**: Removed dead code, duplicate docstrings, and unused Pydantic models

### Migration Notes
- No breaking changes - fully backward compatible
- Existing installations will automatically gain document ingestion capabilities
- Tag validation only affects new uploads (existing tags unchanged)

## [8.5.14] - 2025-10-23

### Added
- **Memory Hooks: Expanded Git Keyword Extraction** - Dramatically improved memory retrieval by capturing more relevant technical terms from git commits
  - **Problem**: Limited keyword extraction (only 12 terms) missed important development context
    - Git analyzer captured only generic terms: `fix, memory, chore, feat, refactor`
    - Recent work on timestamp parsing, dashboard, analytics not reflected in queries
    - Version numbers (v8.5.12, v8.5.13) not extracted
    - Memory hooks couldn't match against specific technical work
  - **Solution**: Expanded keyword extraction in `git-analyzer.js`
    - **Technical Terms**: Increased from 12 to 38 terms including:
      - Time/Date: `timestamp, parsing, sort, sorting, date, age`
      - Dashboard: `dashboard, analytics, footer, layout, grid, css, stats, display`
      - Development: `async, sync, bugfix, release, version`
      - Features: `embedding, consolidation, memory, retrieval, scoring`
      - Infrastructure: `api, endpoint, server, http, mcp, client, protocol`
    - **Version Extraction**: Added regex to capture version numbers (v8.5.12, v8.5.13, etc.)
    - **Changelog Terms**: Expanded from 12 to 23 terms with same additions
    - **Keyword Limits**: Increased capacity
      - keywords: 15 → 20 terms
      - themes: 10 → 12 entries
      - filePatterns: 10 → 12 entries
  - **Impact**:
    - **Before**: 5 generic terms → limited semantic matching
    - **After**: 20 specific development terms → precise context retrieval
    - Example: `feat, git, memory, retrieval, fix, timestamp, age, v8.5.8, chore, version, v8.5.13, sort, date, dashboard, analytics, stats, display, footer, layout, v8.5.12`
  - **Result**: Memory hooks now capture and retrieve memories about specific technical work (releases, features, bugfixes)
  - **Files Modified**:
    - `claude-hooks/utilities/git-analyzer.js` - Expanded `extractDevelopmentKeywords()` function (commit 4a02c1a)
  - **Testing**: Verified improved extraction with test run showing 20 relevant keywords vs previous 5 generic terms

## [8.5.13] - 2025-10-23

### Fixed
- **Memory Hooks: Unix Timestamp Parsing in Date Sorting** - Fixed critical bug where memories were not sorting chronologically in Claude Code session start
  - **Root Cause**: JavaScript `Date()` constructor expects milliseconds but API returns Unix timestamps in seconds
  - **Impact**: Memory hooks showed old memories (Oct 11-21) before recent ones (Oct 23) despite `sortByCreationDate: true` configuration
  - **Technical Details**:
    - API returns `created_at` as Unix timestamp in seconds (e.g., 1729700000)
    - JavaScript `new Date(1729700000)` interprets this as milliseconds → January 21, 1970
    - All dates appeared as 1970-01-01, breaking chronological sort
    - Relevance scores then determined order, causing old high-scoring memories to rank first
  - **Fix**:
    - Created `getTimestamp()` helper function in `session-start.js` (lines 907-928)
    - Converts `created_at` (seconds) to milliseconds by multiplying by 1000
    - Falls back to `created_at_iso` string parsing if available
    - Proper date comparison ensures newest memories sort first
  - **Result**: Memory hooks now correctly show most recent project memories at session start
  - **Files Modified**:
    - `claude-hooks/core/session-start.js` - Added Unix timestamp conversion helper (commit 71606e5)

## [8.5.12] - 2025-10-23

### Fixed
- **Dashboard: Analytics Stats Display** - Fixed analytics tab showing 0/N/A for key metrics
  - **Root Cause**: Async/sync mismatch in `get_stats()` method implementations
  - **Impact**: Analytics dashboard displayed only "this week" count; total memories, unique tags, and database size showed 0 or N/A
  - **Fix**:
    - Made `SqliteVecMemoryStorage.get_stats()` async (line 1242)
    - Updated `HybridMemoryStorage.get_stats()` to properly await primary storage call (line 878)
    - Added `database_size_bytes` and `database_size_mb` to hybrid stats response
    - Fixed all callers in `health.py` and `mcp.py` to await `get_stats()`
  - **Result**: All metrics now display correctly (1778 memories, 2549 tags, 7.74MB)
  - **Files Modified**:
    - `src/mcp_memory_service/storage/sqlite_vec.py` - Made get_stats() async
    - `src/mcp_memory_service/storage/hybrid.py` - Added await and database size fields
    - `src/mcp_memory_service/web/api/health.py` - Simplified async handling
    - `src/mcp_memory_service/web/api/mcp.py` - Added await calls

- **Dashboard: Footer Layout** - Fixed footer appearing between header and content instead of at bottom
  - **Root Cause**: Footer not included in CSS grid layout template
  - **Impact**: Broken visual layout with footer misplaced in page flow
  - **Fix**:
    - Updated `.app-container` grid to include 5th row with "footer" area
    - Assigned `grid-area: footer` to `.app-footer` class
  - **Result**: Footer now correctly positioned at bottom of page
  - **Files Modified**:
    - `src/mcp_memory_service/web/static/style.css` - Updated grid layout (lines 101-110, 1899)

- **HTTP Server: Runtime Warnings** - Eliminated "coroutine was never awaited" warnings in logs
  - **Root Cause**: Legacy sync/async detection code after all backends became async
  - **Impact**: Runtime warnings cluttering server logs
  - **Fix**: Removed hybrid backend detection logic, all `get_stats()` calls now consistently await
  - **Result**: Clean server logs with no warnings

## [8.5.11] - 2025-10-23

### Fixed
- **Consolidation System: Embedding Retrieval in get_all_memories()** - Fixed SQLite-vec backend to actually retrieve embeddings (PR #171, fixes #169)
  - **Root Cause**: `get_all_memories()` methods only queried `memories` table without joining `memory_embeddings` virtual table
  - **Impact**: Consolidation system received 0 embeddings despite 1773 memories in database, preventing association discovery and semantic clustering
  - **Discovery**: PR #170 claimed to fix this but only modified debug tools; actual fix required changes to `sqlite_vec.py`
  - **Fix**:
    - Added `deserialize_embedding()` helper function using numpy.frombuffer() (sqlite-vec only provides serialize, not deserialize)
    - Updated both `get_all_memories()` methods (lines 1468 and 1681) with LEFT JOIN to `memory_embeddings` table
    - Modified `_row_to_memory()` helper to handle 10-column rows with embeddings
    - Applied Gemini Code Assist improvement to simplify row unpacking logic
  - **Test Results** (1773 memories):
    - Embeddings retrieved: 1773/1773 (100%)
    - Associations discovered: 90-91 (0.3-0.7 similarity range)
    - Semantic clusters created: 3 (DBSCAN grouping)
    - Performance: 1249-1414 memories/second
    - Duration: 1.25-1.42 seconds
  - **Consolidation Status**: ✅ **FULLY FUNCTIONAL** (all three blockers fixed: PR #166, #168, #171)
  - **Files Modified**:
    - `src/mcp_memory_service/storage/sqlite_vec.py` - Added embedding retrieval to all memory fetch operations

## [8.5.10] - 2025-10-23

### Fixed
- **Debug Tools: Embedding Retrieval Functionality** - Fixed debug MCP tools for SQLite-vec backend (PR #170, addresses #169)
  - **Root Cause**: `debug_retrieve_memory` function was written for ChromaDB but codebase now uses SQLite-vec storage
  - **Impact**: Debug tools (`debug_retrieve`) were broken, preventing debugging of embedding retrieval operations
  - **Fix**: Updated debug utilities to work with current SQLite-vec storage backend
  - **Changes**:
    - Fixed `debug_retrieve_memory` in `src/mcp_memory_service/utils/debug.py` to use storage's `retrieve()` method
    - Enhanced debug output with similarity scores, backend information, query details, and raw distance values
    - Added proper filtering by similarity threshold
  - **Files Modified**:
    - `src/mcp_memory_service/utils/debug.py` - Updated for SQLite-vec compatibility
    - `src/mcp_memory_service/server.py` - Enhanced debug output formatting

### Added
- **Debug Tool: get_raw_embedding MCP Tool** - New debugging capability for embedding inspection (PR #170)
  - **Purpose**: Direct debugging of embedding generation process
  - **Features**:
    - Shows raw embedding vectors with configurable display (first 10 and last 10 values for readability)
    - Displays embedding dimensions
    - Shows generation status and error messages
  - **Use Case**: Troubleshooting embedding-related issues in consolidation and semantic search
  - **Files Modified**:
    - `src/mcp_memory_service/server.py` - Added `get_raw_embedding` tool and handler

## [8.5.9] - 2025-10-22

### Fixed
- **Consolidation System: Missing update_memory() Method** - Added `update_memory()` method to all storage backends (PR #166, fixes #165)
  - **Root Cause**: Storage backends only implemented `update_memory_metadata()`, but consolidation system's `StorageProtocol` required `update_memory()` for saving consolidated results
  - **Impact**: Prevented consolidation system from saving associations, clusters, compressions, and archived memories
  - **Fix**: Added `update_memory()` method to base `MemoryStorage` class, delegating to `update_memory_metadata()` for proper implementation
  - **Affected Backends**: CloudflareStorage, SqliteVecMemoryStorage, HybridMemoryStorage
  - **Test Results**:
    - Verified on SQLite-vec backend with 1773 memories
    - Performance: 5011 memories/second (local SQLite-vec) vs 2.5 mem/s (Cloudflare)
    - Method successfully executes without AttributeError
  - **Files Modified**:
    - `src/mcp_memory_service/storage/base.py` - Added `update_memory()` to base class
    - `src/mcp_memory_service/storage/http_client.py` - Updated HTTP client call
    - `src/mcp_memory_service/storage/hybrid.py` - Fixed method reference

- **Consolidation System: Datetime Timezone Mismatch** - Fixed timezone handling in decay calculator (PR #168, fixes #167)
  - **Root Cause**: Mixed offset-naive and offset-aware datetime objects causing `TypeError` when calculating time differences
  - **Location**: `src/mcp_memory_service/consolidation/decay.py:191` in `_calculate_access_boost()`
  - **Impact**: Blocked decay calculator from completing, preventing associations, clustering, compression, and archival
  - **Fix**: Added timezone normalization to ensure both `current_time` and `last_accessed` are timezone-aware (UTC) before subtraction
  - **Implementation**:
    - Check if datetime is timezone-naive and convert to UTC if needed
    - Ensures consistent timezone handling across all datetime operations
  - **Files Modified**:
    - `src/mcp_memory_service/consolidation/decay.py` - Added timezone normalization logic

### Added
- **Consolidation Documentation** - Comprehensive setup and testing guides
  - `CONSOLIDATION_SETUP.md` - Complete configuration guide for dream-inspired memory consolidation
  - `CONSOLIDATION_TEST_RESULTS.md` - Expected results and troubleshooting guide
  - Documentation covers all 7 consolidation engines and 7 MCP tools

## [8.5.8] - 2025-10-22

### Fixed
- **Critical: Memory Age Calculation in Hooks** - Fixed Unix timestamp handling that caused memories to appear 20,363 days old (55 years) when they were actually recent
  - **Root Cause**: JavaScript's `Date()` constructor expects milliseconds, but SQLite database stores Unix timestamps in seconds. Three functions incorrectly treated seconds as milliseconds: `calculateTimeDecay()`, `calculateRecencyBonus()`, and `analyzeMemoryAgeDistribution()`
  - **Symptoms**:
    - Memory Age Analyzer showed `avgAge: 20363` days instead of actual age
    - Stale memory detection incorrectly triggered (`isStale: true`)
    - Recent memory percentage showed 0% when should be 100%
    - Time decay scores incorrect (1% instead of 100% for today's memories)
    - Recency bonus not applied (0% instead of +15%)
  - **Fix**: Added type checking to convert Unix timestamps properly - multiply by 1000 when timestamp is a number (seconds), pass through when it's an ISO string
  - **Impact**: Memory age calculations now accurate, stale detection works correctly, recency bonuses applied properly
  - **Files Modified**:
    - `claude-hooks/utilities/memory-scorer.js` (lines 11-17, 237-243, 524-534)
  - **Test Results**: Memories now show correct ages (0.4 days vs 20,363 days before fix)
  - **Platform**: All platforms (macOS, Linux, Windows)

### Changed
- **Installer Enhancement**: Added automatic statusLine configuration for v8.5.7 features
  - Installer now copies `statusline.sh` to `~/.claude/hooks/`
  - Checks for `jq` dependency (required for statusLine parsing)
  - Automatically adds `statusLine` configuration to `settings.json`
  - Enhanced documentation for statusLine setup and requirements

### Documentation
- Added `jq` as required dependency for statusLine feature
- Documented statusLine configuration in README.md installation section
- Clarified Unix timestamp handling in memory-scorer.js code comments

## [8.5.7] - 2025-10-21

### Added
- **SessionStart Hook Visibility Features** - Three complementary methods to view session memory context
  - **Visible Summary Output**: Clean bordered console display showing project, storage, memory count with recent indicator, and git context
  - **Detailed Log File**: Complete session context written to `~/.claude/last-session-context.txt` including project details, storage backend, memory statistics, git analysis, and top loaded memories
  - **Status Line Display**: Always-visible status bar at bottom of Claude Code terminal showing `🧠 8 (5 recent) | 📊 10 commits`
  - **Files Modified**:
    - `~/.claude/hooks/core/session-start.js` - Added summary output, log file generation, and cache file write logic
    - `~/.claude/settings.json` - Added statusLine configuration
  - **Files Created**:
    - `~/.claude/statusline.sh` - Bash script for status line display (requires `jq`)
    - `~/.claude/last-session-context.txt` - Auto-generated detailed log file
    - `~/.claude/hooks/utilities/session-cache.json` - Status line data cache
  - **Platform**: Linux/macOS (Windows SessionStart hook still broken - issue #160)

### Changed
- SessionStart hook output now provides visible feedback instead of being hidden in system-reminder tags
- Status line updates every 300ms with latest session memory context
- Log file automatically updates on each SessionStart hook execution

### Documentation
- Clarified difference between macOS and Linux hook output behavior (both use system-reminder tags since v2.2.0)
- Documented that `<session-start-hook>` wrapper tags were intentionally removed in v2.2.0 for cleaner output
- Added troubleshooting guide for status line visibility features

## [8.5.6] - 2025-10-16

### Fixed
- **Critical: Memory Hooks HTTPS SSL Certificate Validation** - Fixed hooks failing to connect to HTTPS server with self-signed certificates
  - **Root Cause**: Node.js HTTPS requests were rejecting self-signed SSL certificates silently, causing "No active connection available" errors
  - **Symptoms**:
    - Hooks showed "Failed to connect using any available protocol"
    - No memories retrieved despite server being healthy
    - HTTP server running but hooks couldn't establish connection
  - **Fix**: Added `rejectUnauthorized: false` to both health check and API POST request options in memory-client.js
  - **Impact**: Hooks now successfully connect via HTTPS to servers with self-signed certificates
  - **Files Modified**:
    - `claude-hooks/utilities/memory-client.js` (lines 174, 257)
    - `~/.claude/hooks/utilities/memory-client.js` (deployed)
  - **Test Results**: ✅ 7 memories retrieved from 1558 total, all phases working correctly
  - **Platform**: All platforms (macOS, Linux, Windows)

### Changed
- Memory hooks now support HTTPS endpoints with self-signed certificates without manual certificate trust configuration

## [8.5.5] - 2025-10-14

### Fixed
- **Critical: Claude Code Hooks Configuration** - Fixed session-start hook hanging/unresponsiveness on Windows
  - **Root Cause**: Missing forced process exit in session-start.js caused Node.js event loop to remain active with unclosed connections
  - **Fix 1**: Added `.finally()` block with 100ms delayed `process.exit(0)` to ensure clean termination
  - **Fix 2**: Corrected port mismatch in `~/.claude/hooks/config.json` (8889 → 8000) to match HTTP server
  - **Impact**: Hooks now complete in <15 seconds without hanging, Claude Code remains responsive
  - **Files Modified**:
    - `~/.claude/hooks/core/session-start.js` (lines 1010-1013)
    - `~/.claude/hooks/config.json` (line 7)
  - **Platform**: Windows (also applies to macOS/Linux)

### Changed
- **Documentation**: Added critical warning section to CLAUDE.md about hook configuration synchronization
  - Documents port mismatch symptoms (hanging hooks, unresponsive Claude Code, connection timeouts)
  - Lists all configuration files to check (`config.json`, HTTP server port, dashboard port)
  - Provides verification commands for Windows/Linux/macOS
  - Explains common mistakes (using dashboard port 8888/8443 instead of API port 8000)

## [8.5.4] - 2025-10-13

### Fixed
- **MCP Server**: Added explicit documentation to `store_memory` tool clarifying that `metadata.tags` must be an array, not a comma-separated string
  - Prevents validation error: `Input validation error: '...' is not of type 'array'`
  - Includes clear examples showing correct (array) vs incorrect (string) format
  - Documentation-only change - no code logic modified

### Changed
- Improved `store_memory` tool docstring with metadata format validation examples in `src/mcp_memory_service/mcp_server.py`

## [Unreleased]

### ✨ **Added**

#### **Linux systemd Service Support**
Added comprehensive systemd user service support for automatic HTTP server management on Linux systems.

**New Files:**
- `scripts/service/mcp-memory-http.service` - systemd user service definition
- `scripts/service/install_http_service.sh` - Interactive installation script
- `docs/deployment/systemd-service.md` - Detailed systemd setup guide

**Features:**
- ✅ **Automatic startup** on user login
- ✅ **Persistent operation** with loginctl linger support
- ✅ **Automatic restarts** on failure (RestartSec=10)
- ✅ **Centralized logging** via journald
- ✅ **Easy management** via systemctl commands
- ✅ **Environment loading** from .env file
- ✅ **Security hardening** (NoNewPrivileges, PrivateTmp)

**Usage:**
```bash
bash scripts/service/install_http_service.sh  # Install
systemctl --user start mcp-memory-http.service
systemctl --user enable mcp-memory-http.service
loginctl enable-linger $USER
```

### 📚 **Documentation**

#### **Enhanced HTTP Server Management**
- Updated `docs/http-server-management.md` with systemd section
- Added troubleshooting for port mismatch issues (8000 vs 8889)
- Documented hooks endpoint configuration requirements

#### **CLAUDE.md Updates**
- Added systemd commands to Essential Commands section
- Added "Troubleshooting Hooks Not Retrieving Memories" section
- Cross-referenced detailed documentation guides

### 🐛 **Fixed**

#### **Hooks Configuration Troubleshooting**
- Documented common port mismatch issue between hooks config and HTTP server
- Added diagnostic commands for verifying HTTP server status
- Clarified that HTTP server is **required** for hooks (stdio MCP cannot be used)

**Root Cause:** Many installations had hooks configured for port 8889 while HTTP server runs on port 8000 (default in .env). This caused silent failures where hooks couldn't connect.

**Solution:**
1. Update hooks config endpoint to `http://127.0.0.1:8000`
2. Verify with: `systemctl --user status mcp-memory-http.service`
3. Test with: `curl http://127.0.0.1:8000/api/health`

### 🔧 **Changed**

- Service files now use `WantedBy=default.target` for user services (not multi-user.target)
- Removed `User=` and `Group=` directives from user service (causes GROUP error)
- Enhanced error messages and troubleshooting documentation

### 🎨 **Improved**

#### **Claude Code Hooks UI Enhancements**
Significantly improved visual formatting and readability of memory context injection in Claude Code hooks.

**Enhanced Features:**
- ✅ **Intelligent Text Wrapping** - New `wrapText()` function preserves word boundaries and indentation
- ✅ **Unicode Box Drawing** - Professional visual formatting with ╭╮╯╰ characters for better structure
- ✅ **Recency-Based Display** - Recent memories (< 7 days) stay prominent, older ones are dimmed
- ✅ **Simplified Date Formatting** - Cleaner date display with recency indicators (today, yesterday, day name, date)
- ✅ **Enhanced Memory Categorization** - Better visual hierarchy for different memory types

**Files Modified:**
- `claude-hooks/utilities/context-formatter.js` - Major refactoring with wrapText function and enhanced formatMemoryForCLI
- `claude-hooks/core/session-start.js` - Minor display improvements for project detection
- `.claude/settings.local.json` - Platform-specific configuration updates (Windows→Linux path migration)

**Performance:**
- No performance impact - Lightweight formatting enhancements
- Better readability improves development efficiency
- Maintains all existing functionality while improving presentation

---

## [8.5.3] - 2025-10-12

### 🐛 **Fixed**

#### **Critical Memory Hooks Bug Fixes (Claude Code Integration)**
Fixed critical bugs preventing memory retrieval in Claude Code session-start hooks. Memory awareness now works correctly with both semantic and time-based searches.

**Problem**: Memory hooks showed "No relevant memories found" despite 1,419 memories in database. Retrieved memories were unrelated (from wrong projects) and sorted incorrectly (oldest first).

**Root Causes Fixed:**

1. **Empty Semantic Query Bug** (search.py:264-272)
   - **Issue**: `storage.retrieve("")` with empty string returned no results
   - **Fix**: Now uses `get_recent_memories()` when `semantic_query` is empty
   - **Impact**: Time-based searches without semantic filtering now work correctly

2. **Missing Time Expression** (search.py:401-403)
   - **Issue**: Hook sends `'last-2-weeks'` but parser didn't recognize it
   - **Fix**: Added support for 'last 2 weeks', 'past 2 weeks', 'last-2-weeks'
   - **Impact**: Phase 2 fallback queries now work properly

3. **Performance Optimization** (search.py:36)
   - **Change**: Reduced candidate pool from 1000 to 100 for time filtering
   - **Rationale**: Prevents timeout on large databases, improves response time
   - **Impact**: Search completes in <100ms vs timing out

4. **CRITICAL: Missing `await` Keywords** (hybrid.py:912, 916, 935, 947)
   - **Issue**: 4 async methods returned unawaited coroutines, causing server hangs
   - **Methods Fixed**:
     - Line 912: `get_all_tags()`
     - Line 916: `get_recent_memories(n)` ⭐ **THE CRITICAL ONE**
     - Line 935: `recall_memory(query, n_results)`
     - Line 947: `get_memories_by_time_range(start_time, end_time)`
   - **Impact**: Hybrid backend now works perfectly (11ms response time!)

5. **JavaScript Refactoring** (memory-client.js:213-293)
   - **Issue**: ~100 lines of duplicated HTTP request code
   - **Fix**: Created `_performApiPost()` helper to eliminate duplication
   - **Impact**: Improved maintainability, DRY compliance

6. **Port Consistency** (memory-client.js:166)
   - **Issue**: Health checks used standard web ports (443/80) while API calls used dev ports (8443/8889)
   - **Fix**: Made both use development ports consistently
   - **Impact**: Prevents connection failures with development endpoints

**Testing & Verification:**
```bash
# Before fix: Timeout
curl -s -m 5 "http://127.0.0.1:8889/api/search/by-time" -H "Content-Type: application/json" -d '{"query":"last-2-weeks","n_results":10}'
# (hangs indefinitely)

# After fix: Success
curl -s -m 5 "http://127.0.0.1:8889/api/search/by-time" -H "Content-Type: application/json" -d '{"query":"last-2-weeks","n_results":10}'
# Status: 200 (11ms)
# Results: 5 memories retrieved
```

**Files Modified:**
- `src/mcp_memory_service/web/api/search.py` - Empty query fix, time expression, pool size, UTC timezone
- `src/mcp_memory_service/storage/hybrid.py` - Fixed 4 missing await keywords
- `claude-hooks/utilities/memory-client.js` - Refactored HTTP helpers, port consistency, API contract
- `claude-hooks/core/session-start.js` - Updated hardcoded endpoint fallbacks
- `claude-hooks/config.json` - HTTP endpoint configuration

**Code Review**: All fixes reviewed and approved by Gemini Code Assist with PEP 8 compliance, timezone-aware datetimes, list comprehensions, and proper error handling.

**PR Reference**: [#156](https://github.com/doobidoo/mcp-memory-service/pull/156)

## [8.5.2] - 2025-10-11

### 🐛 **Fixed**

#### **v8.5.0 Implementation Missing (Code Completion)**
Complete implementation of Hybrid Backend Sync Dashboard feature that was documented in v8.5.0 CHANGELOG but code was never committed.

**Context**: v8.5.0 release (c241292) included CHANGELOG documentation and version bump but the actual implementation files were accidentally not staged/committed. This release completes the v8.5.0 feature by committing the missing implementation code.

**Files Added:**
- `src/mcp_memory_service/web/api/sync.py` - Sync API endpoints (GET /api/sync/status, POST /api/sync/force)
- `start_http_server.sh` - Cross-platform HTTP server management script

**Files Modified:**
- `src/mcp_memory_service/web/app.py` - Integrated sync router
- `src/mcp_memory_service/web/static/app.js` - Sync status UI with polling
- `src/mcp_memory_service/web/static/index.html` - Sync status bar markup
- `src/mcp_memory_service/web/static/style.css` - Sync bar styling + grid layout
- `src/mcp_memory_service/storage/hybrid.py` - Added `get_sync_status()` method
- `src/mcp_memory_service/web/api/health.py` - Health check enhancements
- `src/mcp_memory_service/storage/sqlite_vec.py` - Database path fixes

**Additional Improvements:**
- `claude-hooks/utilities/context-formatter.js` - Tree text wrapping improvements for better CLI output

**Impact**: Users can now access the complete Hybrid Backend Sync Dashboard feature including manual sync triggers and real-time status monitoring as originally intended in v8.5.0.

## [8.5.1] - 2025-10-11

### 🎯 **New Features**

#### **Dynamic Memory Weight Adjustment (Claude Code Hooks)**
Intelligent auto-calibration prevents stale memories from dominating session context when recent development exists.

**Problem Solved:**
Users reported "Current Development" section showing outdated memories (24-57 days old) instead of recent work from the last 7 days. Root cause: static configuration couldn't adapt to mismatches between git activity and memory age.

**Solution - Memory Age Distribution Analyzer:**
- **Auto-Detection**: Analyzes memory age percentiles (median, p75, p90, avg)
- **Staleness Detection**: Triggers when median > 30 days or < 20% recent memories
- **Smart Calibration**: Automatically adjusts weights:
  - `timeDecay`: 0.25 → 0.50 (+100% boost for recent memories)
  - `tagRelevance`: 0.35 → 0.20 (-43% reduce old tag matches)
- **Impact**: Stale memory sets automatically prioritize any recent memories

**Solution - Adaptive Git Context Weight:**
- **Scenario 1**: Recent commits (< 7d) + Stale memories (median > 30d)
  - Reduces git weight by 30%: `1.8x → 1.3x`
  - Prevents old git-related memories from dominating
- **Scenario 2**: Recent commits + Recent memories (both < 14d)
  - Keeps configured weight: `1.8x → 1.8x`
  - Git context is relevant and aligned
- **Scenario 3**: Old commits (> 14d) + Some recent memories
  - Reduces git weight by 15%: `1.8x → 1.5x`
  - Lets recent non-git memories surface

**Configuration Options:**
```json
{
  "memoryScoring": {
    "autoCalibrate": true  // Enable/disable auto-calibration
  },
  "gitAnalysis": {
    "adaptiveGitWeight": true  // Enable/disable adaptive git weight
  }
}
```

**Transparency Output:**
```
🎯 Auto-Calibration → Stale memory set detected (median: 54d old, 0% recent)
   Adjusted Weights → timeDecay: 0.50, tagRelevance: 0.20
⚙️  Adaptive Git Weight → Recent commits (1d ago) but stale memories - reducing git boost: 1.8 → 1.3
```

**Files Added:**
- `claude-hooks/test-adaptive-weights.js` - Comprehensive test scenarios

**Files Modified:**
- `claude-hooks/utilities/memory-scorer.js` (+162 lines):
  - `analyzeMemoryAgeDistribution()` - Detects staleness and recommends adjustments
  - `calculateAdaptiveGitWeight()` - Dynamically adjusts git boost based on context alignment
- `claude-hooks/core/session-start.js` (+60 lines):
  - Integrated age analysis before scoring
  - Auto-calibration logic with config check
  - Adaptive git weight calculation with transparency output
- `claude-hooks/config.json` (+2 options):
  - Added `memoryScoring.autoCalibrate: true` (default enabled)
  - Added `gitAnalysis.adaptiveGitWeight: true` (default enabled)

**Benefits:**
- ✅ **Automatic Detection**: No manual config changes when memories become stale
- ✅ **Context-Aware**: Git boost only applies when it enhances (not harms) relevance
- ✅ **Transparent**: Shows reasoning for adjustments in session output
- ✅ **Opt-Out Available**: Users can disable via config if desired
- ✅ **Backward Compatible**: Defaults preserve existing behavior when memories are recent

**Test Results:**
- Scenario 1 (Stale): Automatically calibrated weights and reduced git boost 1.8x → 1.3x
- Scenario 2 (Recent): No calibration needed, preserved git weight at 1.8x
- Both scenarios working as expected, preventing outdated context issues

## [8.5.0] - 2025-10-11

### 🎉 **New Features**

#### **Hybrid Backend Sync Dashboard**
Manual sync management UI with real-time status monitoring for hybrid storage backend.

**Features:**
- **Sync Status Bar** - Color-coded visual indicator between navigation and main content
  - 🔄 Syncing (blue gradient) - Active synchronization in progress
  - ⏱️ Pending (yellow gradient) - Operations queued, shows ETA and count
  - ✅ Synced (green gradient) - All operations synchronized, shows last sync time
  - ⚠️ Error (red gradient) - Sync failures detected, shows failed operation count
- **"Sync Now" Button** - Manual trigger for immediate Cloudflare ↔ SQLite synchronization
- **Real-time Monitoring** - 10-second polling for live sync status updates
- **REST API Endpoints:**
  - `GET /api/sync/status` - Current sync state, pending operations, last sync time
  - `POST /api/sync/force` - Manually trigger immediate sync

**Technical Implementation:**
- New sync API router: `src/mcp_memory_service/web/api/sync.py` (complete CRUD endpoints)
- Frontend integration: `src/mcp_memory_service/web/static/app.js:379-485` (status monitoring + manual sync)
- CSS styling: `src/mcp_memory_service/web/static/style.css:292-403` (grid layout + animations)
- HTML structure: `src/mcp_memory_service/web/static/index.html:125-138` (sync bar markup)
- Backend method: `src/mcp_memory_service/storage/hybrid.py:982-994` (added `get_sync_status()`)

### 🐛 **Fixed**

- **CSS Grid Layout Bug** - Sync status bar invisible despite JavaScript detecting hybrid mode
  - **Root Cause**: `.app-container` grid layout defined `"header" "nav" "main"` but sync bar wasn't assigned a grid area
  - **Fix**: Added `grid-area: sync` to `.sync-status-bar` and expanded grid to include sync row
  - **Files**: `style.css:101-109` (grid layout), `style.css:293` (sync bar grid area)

- **Sync Status Logic Error** - "Sync Now" button incorrectly disabled when background service running
  - **Root Cause**: Confused `is_running` (service alive) with `actively_syncing` (active sync operation)
  - **Fix**: Changed status determination to check `actively_syncing` field instead of `is_running`
  - **Impact**: Button now correctly enabled when 0 pending operations
  - **File**: `src/mcp_memory_service/web/api/sync.py:106-118`

- **Database Path Mismatch** - HTTP server using different SQLite database than Claude Code MCP
  - **Root Cause**: Missing `MCP_MEMORY_SQLITE_PATH` environment variable in HTTP server startup
  - **Fix**: Added explicit database path to match Claude Desktop config
  - **File**: `start_http_server.sh:4` (added `MCP_MEMORY_SQLITE_PATH` export)

- **Backend Configuration Inconsistency** - Claude Desktop using `cloudflare` backend while HTTP server using `hybrid`
  - **Root Cause**: Mismatched storage backend configurations preventing data synchronization
  - **Fix**: Unified both to use `hybrid` backend with same SQLite database
  - **File**: `claude_desktop_config.json:70` (changed `"cloudflare"` → `"hybrid"`)
  - **Impact**: Dashboard now shows same 1413 memories as Claude Code

### 🔧 **Improvements**

- **Enhanced Health Check** - `/api/health/detailed` now includes sync status for hybrid backend
  - Shows sync service state, pending operations, last sync time, failed operations
  - File: `src/mcp_memory_service/web/api/health.py:141-154`

- **Cleaned Database Files** - Removed obsolete SQLite databases to prevent confusion
  - Deleted: `memory_http.db` (701 memories), `backup_sqlite_vec.db`, `sqlite_vec_backup_20250822_230643.db`

- **Updated Startup Script** - `start_http_server.sh` now includes all required environment variables
  - Added: `MCP_MEMORY_SQLITE_PATH`, `MCP_HTTP_ENABLED`
  - Ensures HTTP server uses same database as Claude Code

### 📊 **Impact**

- **User Experience**: Dashboard now provides complete visibility and control over hybrid backend synchronization
- **Data Consistency**: Unified backend configuration ensures Claude Code and HTTP dashboard always show same data
- **Performance**: Manual sync trigger allows immediate synchronization instead of waiting 5 minutes
- **Reliability**: Fixed grid layout bug ensures sync status bar always visible when in hybrid mode

## [8.4.3] - 2025-10-11

### 🐛 Fixed
- **Sync Script Import Path:** Fixed `scripts/sync/sync_memory_backends.py` module import path to work correctly from scripts directory
  - Changed `sys.path.insert(0, str(Path(__file__).parent.parent))` → `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`
  - Resolves `ModuleNotFoundError: No module named 'src'` when using manual sync commands
  - Fixes: `python scripts/sync/claude_sync_commands.py backup/restore/sync` commands

### 📊 Impact
- Users can now successfully run manual sync utilities for hybrid backend
- Manual Cloudflare ↔ SQLite synchronization commands now functional

## [8.4.2] - 2025-10-11

### 🎯 **Performance & Optimization**

#### **Additional MCP Context Optimization: Debug Tools Removal**
- **Problem**: Continuing context optimization efforts from v8.4.1, identified 2 additional low-value debug tools
- **Solution**: Removed debug/maintenance MCP tools with zero test dependencies

**Tools Removed:**
- `get_embedding` (606 tokens) - Returns raw embedding vectors; low-level debugging only
- `check_embedding_model` (553 tokens) - Checks if embedding model loaded; errors surface naturally

**Rationale:** These were specialized debugging tools rarely needed in practice. Embedding errors are caught during normal retrieval operations, and raw embedding inspection is a niche development task not required for AI assistant integration.

**Impact:**
- ✅ **MCP tools**: 26.8k → 25.6k tokens (4.5% additional reduction, -1.2k tokens)
- ✅ **Total optimization since v8.4.0**: 31.4k → 25.6k tokens (18.5% reduction, -5.8k tokens saved)
- ✅ **Zero breaking changes**: No test coverage for these tools
- ✅ **Conservative approach**: Removed only tools with no dependencies

**Files Modified:**
- `src/mcp_memory_service/server.py`: Removed 2 tool definitions, handlers, and implementations (~61 lines)

**Note:** Further optimization possible with MODERATE approach (debug_retrieve, exact_match_retrieve, cleanup_duplicates) if additional context savings needed.

## [8.4.1] - 2025-10-11

### 🎯 **Performance & Optimization**

#### **MCP Context Optimization: Dashboard Tools Removal**
- **Problem**: MCP tools consuming 31.4k tokens (15.7% of context budget) with redundant dashboard variants that duplicated web UI functionality
- **Solution**: Removed 8 dashboard-specific MCP tools that were unnecessary for Claude Code integration

**Tools Removed:**
- `dashboard_check_health`, `dashboard_recall_memory`, `dashboard_retrieve_memory`
- `dashboard_search_by_tag`, `dashboard_get_stats`, `dashboard_optimize_db`
- `dashboard_create_backup`, `dashboard_delete_memory`

**Rationale:** Web dashboard uses REST API endpoints (`/api/*`), not MCP tools. These were legacy wrappers created during early dashboard development that bloated context without providing value for AI assistant integration.

**Impact:**
- ✅ **MCP tools**: 31.4k → 26.8k tokens (15% reduction, -4.6k tokens saved)
- ✅ **Zero functional impact**: Core memory tools preserved (`check_database_health`, `recall_memory`, etc.)
- ✅ **Cleaner separation**: MCP protocol for Claude Code integration, HTTP REST API for web dashboard

**Files Modified:**
- `src/mcp_memory_service/server.py`: Removed 8 tool definitions, call_tool handlers, and method implementations (~506 lines)
- `docs/api/tag-standardization.md`: Updated to use `check_database_health()` instead of `dashboard_get_stats()`
- `docs/maintenance/memory-maintenance.md`: Removed redundant dashboard tool reference
- `docs/guides/mcp-enhancements.md`: Removed `dashboard_optimize_db` progress tracking example
- `docs/assets/images/project-infographic.svg`: Removed `dashboard_*_ops` visual reference

**Note:** Web dashboard at `https://localhost:8443` continues working normally via REST API. No user-facing changes.

## [8.4.0] - 2025-10-08

### ✨ **Features & Improvements**

#### **Claude Code Memory Hooks Recency Optimization**
- **Problem Solved**: Memory hooks were surfacing 60+ day old memories instead of recent development work (Oct v8.0-v8.3), causing critical development context to be missing despite being stored in the database
- **Core Enhancement**: Comprehensive recency optimization with rebalanced scoring algorithm to prioritize recent memories over well-tagged old content

##### **Scoring Algorithm Improvements**
- **Weight Rebalancing** (`config.json`):
  - `timeDecay`: 0.25 → 0.40 (+60% influence on recency)
  - `tagRelevance`: 0.35 → 0.25 (-29% to reduce tag dominance)
  - `contentQuality`: 0.25 → 0.20 (-20% to balance with time)
- **Gentler Time Decay**:
  - `timeDecayRate`: 0.1 → 0.05 (30-day memories: 0.22 vs 0.05 score - preserves older memories)
- **Stronger Git Context**:
  - `gitContextWeight`: 1.2 → 1.8 (80% boost vs 20% for git-derived memories)
  - Implemented multiplication in `session-start.js` after scoring
- **Expanded Time Windows**:
  - `recentTimeWindow`: "last-week" → "last-month" (broader recent search)
  - `fallbackTimeWindow`: "last-month" → "last-3-months" (wider fallback range)
- **Higher Quality Bar**: `minRelevanceScore`: 0.3 → 0.4 (filters generic old content)

##### **New Recency Bonus System** (`memory-scorer.js`)
- **Tiered Additive Bonuses**:
  - < 7 days: +0.15 bonus (strong boost for last week)
  - < 14 days: +0.10 bonus (moderate boost for last 2 weeks)
  - < 30 days: +0.05 bonus (small boost for last month)
- **Implementation**: Configurable tier-based system using `RECENCY_TIERS` array
- **Impact**: Ensures recent memories always get advantage regardless of tag relevance

##### **Documentation & Testing**
- **Added**: Comprehensive `CONFIGURATION.md` (450+ lines)
  - All scoring weights with impact analysis
  - Time decay behavior and examples
  - Git context weight strategy
  - Recency bonus system documentation
  - Tuning guide for different workflows
  - Migration notes from v1.0
- **Added**: Validation test suite (`test-recency-scoring.js`)
  - Tests scoring algorithm with memories of different ages
  - Validates time decay and recency bonus calculations
  - Confirms recent memories rank higher (success criteria: 2 of top 3 < 7 days old)

##### **Results**
- **Before**: Top 3 memories averaged 45+ days old (July-Sept content)
- **After**: All top 3 memories < 7 days old ✅
- **Validation**: 80% higher likelihood of surfacing recent work

**Impact**: ✅ Memory hooks now reliably surface recent development context, significantly improving Claude Code session awareness for active projects

##### **Technical Details**
- **PR**: [#155](https://github.com/doobidoo/mcp-memory-service/pull/155) - Memory hooks recency optimization
- **Files Modified**:
  - Configuration: `claude-hooks/config.json` (scoring weights, time windows, git context)
  - Scoring: `claude-hooks/utilities/memory-scorer.js` (recency bonus, configurable decay)
  - Session: `claude-hooks/core/session-start.js` (git context weight implementation)
  - Tests: `claude-hooks/test-recency-scoring.js` (validation suite)
  - Documentation: `claude-hooks/CONFIGURATION.md` (comprehensive guide)
- **Code Review**: 7 rounds of Gemini Code Assist review completed
  - **CRITICAL bugs fixed**: Config values not being used (round 5), gitContextWeight not implemented (round 6)
  - **Security fixes**: TLS certificate validation, future timestamp handling
  - **Maintainability**: DRY refactoring, tier-based configuration, comprehensive docs
- **Test Results**: ✅ All validation checks passed - recent memories consistently prioritized

## [8.3.1] - 2025-10-07

### ✨ **Features & Improvements**

#### **HTTP Server Management Tools**
- **Added**: Cross-platform HTTP server management utilities for Claude Code Natural Memory Triggers
- **New Scripts**:
  - `scripts/server/check_http_server.py`: Health check utility for HTTP server status verification
    - Supports both HTTP and HTTPS endpoints via environment variables
    - Verbose output by default, `-q` flag for quiet mode (exit codes only)
    - Detects MCP_HTTPS_ENABLED, MCP_HTTP_PORT, MCP_HTTPS_PORT configuration
  - `scripts/server/start_http_server.sh`: Auto-start script for Unix/macOS
    - Intelligent server detection with 5-second polling loop
    - Background process management via nohup
    - Logs to `/tmp/mcp-http-server.log`
  - `scripts/server/start_http_server.bat`: Auto-start script for Windows
    - 5-second polling loop for reliable startup detection
    - Starts server in new window for easy monitoring
    - Handles already-running servers gracefully
- **Documentation**: Comprehensive `docs/http-server-management.md` guide
  - Why HTTP server is required for Natural Memory Triggers
  - Quick health check commands
  - Manual and auto-start procedures
  - Troubleshooting guide with common issues
  - Integration with Claude Code hooks
  - Automation examples (launchd, Task Scheduler, shell aliases)
- **Use Case**: Essential for Claude Code hooks to inject relevant memories at session start without MCP conflicts
- **Wiki**: User-specific setup examples moved to wiki as reference guides
  - Windows-Hybrid-Backend-Setup-Example.md
  - Windows-Setup-Summary-Example.md

**Impact**: ✅ Streamlined HTTP server management, improved Natural Memory Triggers reliability, better cross-platform support

##### **Technical Details**
- **PR**: [#154](https://github.com/doobidoo/mcp-memory-service/pull/154) - HTTP server management tools
- **Files Added**: 4 new files
  - Scripts: `check_http_server.py`, `start_http_server.sh`, `start_http_server.bat`
  - Documentation: `http-server-management.md`
- **Gemini Reviews**: 3 rounds of code review and refinement
  - Security: Removed hardcoded credentials (replaced with placeholders)
  - Robustness: Improved exception handling, added polling loops
  - CLI Usability: Simplified argument parsing (removed redundant `-v` flag)
- **Cross-platform**: Fully tested on Unix/macOS and Windows environments
- **Integration**: Works seamlessly with existing `run_http_server.py` script

## [8.3.0] - 2025-10-07

### 🧹 **Refactoring & Code Cleanup**

#### **Complete ChromaDB Backend Removal**
- **Removed**: ~300-500 lines of ChromaDB dead code following v8.0.0 deprecation
- **Scope**: Complete cleanup across 18 files including configuration, CLI, server, storage, utilities, and web interface
- **Changes**:
  - **Configuration** (`config.py`): Removed CHROMA_PATH, CHROMA_SETTINGS, COLLECTION_METADATA, CHROMADB_MAX_CONTENT_LENGTH
  - **CLI** (`cli/main.py`, `cli/ingestion.py`): Removed `--chroma-path` option, removed 'chromadb' from storage backend choices
  - **Server** (`server.py`): Removed ChromaDB initialization (~60 lines), stats fallback (~40 lines), backup handler, validation logic
  - **Utilities** (`utils/db_utils.py`): Removed ChromaDB validation, stats, and repair functions (~140 lines)
  - **Storage** (`storage/cloudflare.py`, `storage/sqlite_vec.py`): Updated docstrings to be backend-agnostic
  - **Web** (`web/app.py`): Removed 'chromadb' from backend display name mapping
  - **Documentation**: Updated all error messages to suggest Cloudflare instead of ChromaDB
- **Impact**: ✅ Cleaner codebase, reduced technical debt, no misleading ChromaDB references
- **SUPPORTED_BACKENDS**: Now correctly shows `['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']`

#### **CLI Backend Consistency Enhancement**
- **Added**: 'sqlite-vec' hyphenated alias to all CLI storage backend choices
- **Affected commands**: `server`, `status`, `ingest_document`, `ingest_directory`
- **Rationale**: Ensures CLI behavior matches SUPPORTED_BACKENDS configuration
- **Impact**: ✅ Improved user experience with consistent backend naming across CLI and configuration

### 🐛 **Bug Fixes**

#### **Dashboard System Information Display (Issue #151)**
- **Fixed**: Dashboard showing "N/A" for embedding model, embedding dimensions, and database size on non-hybrid backends
- **Root cause**: JavaScript expected hybrid-backend-specific nested paths (`storage.primary_stats.*`)
- **Solution**: Added fallback paths in `app.js` SYSTEM_INFO_CONFIG:
  - `settingsEmbeddingModel`: Falls back to `storage.embedding_model`
  - `settingsEmbeddingDim`: Falls back to `storage.embedding_dimension`
  - `settingsDbSize`: Falls back to `storage.database_size_mb`
- **Impact**: ✅ Dashboard now correctly displays system information for sqlite-vec, cloudflare, and hybrid backends

##### **Technical Details**
- **PR**: [#153](https://github.com/doobidoo/mcp-memory-service/pull/153) - ChromaDB dead code removal + Issue #151 fix
- **Files Modified**: 18 files
  - Core cleanup: `config.py`, `server.py`, `mcp_server.py`, `utils/db_utils.py`
  - CLI: `cli/main.py`, `cli/ingestion.py`, `cli/utils.py`
  - Storage: `storage/factory.py`, `storage/cloudflare.py`, `storage/sqlite_vec.py`
  - Web: `web/app.py`, `web/api/health.py`, `web/static/app.js`
  - Utilities: `utils/debug.py`, `embeddings/onnx_embeddings.py`
  - Package: `__init__.py`, `dependency_check.py`
- **Code Review**: Approved by Gemini Code Assist with high-quality feedback
- **Testing**: ✅ SQLite-vec backend initialization, ✅ SUPPORTED_BACKENDS verification, ✅ Service startup

## [8.2.4] - 2025-10-06

### 🐛 **Bug Fixes**

#### **Critical: Memory Hooks JSON Parsing Failure**
- **Fixed**: Memory awareness hooks completely broken - unable to retrieve memories due to JSON parsing errors
- **Root cause**: Naive string replacement in HTTP client destroyed valid JSON
  - `replace(/'/g, '"')` broke apostrophes in content (e.g., "it's" → "it"s")
  - Replaced Python-style values (True/False/None) in already-valid JSON
  - Used `/mcp` MCP-over-HTTP bridge instead of direct REST API
- **Solution**:
  - Removed destructive string replacements
  - Updated to use direct REST API endpoints (`/api/search`, `/api/search/by-time`)
  - Parse JSON responses directly without conversion
- **Impact**: ✅ Memory hooks now successfully retrieve context-relevant memories at session start

#### **HTTP Server Backend Configuration Override**
- **Fixed**: HTTP server ignored `.env` configuration, forcing `sqlite_vec` instead of configured `hybrid` backend
- **Root cause**: `run_http_server.py` used `os.environ.setdefault()` after `.env` loading, overriding user config
- **Solution**: Commented out the backend override line to respect `.env` settings
- **Impact**: ✅ Hybrid backend now works correctly via HTTP server

##### **Technical Details**
- **Files**:
  - `C:\Users\heinrich.krupp\.claude\hooks\utilities\memory-client.js` - Fixed `queryMemoriesHTTP()` method
  - `scripts/server/run_http_server.py` - Removed backend configuration override (line 148)
- **Affected**: All users using memory hooks with HTTP protocol (automatic session awareness)

## [8.2.3] - 2025-10-05

### ✨ **Enhancements**

#### **Dashboard Footer Navigation**
- **Added**: Comprehensive footer to dashboard with three sections
  - **Documentation**: Links to Wiki Home, Troubleshooting Guide, Backend Configuration Issues
  - **Resources**: GitHub Repository (with icon), Portfolio (doobidoo.github.io), API Documentation
  - **About**: Project description, Apache 2.0 license link, copyright notice
- **Features**: Security attributes (target="_blank", rel="noopener"), responsive design (mobile breakpoint 768px)
- **Impact**: ✅ Improved discoverability of documentation and resources from dashboard

### 🐛 **Bug Fixes**

#### **Dark Mode Footer Styling**
- **Critical fix**: Footer appearing bright/light in dark mode instead of dark
- **Root cause**: Incorrect CSS variable usage - using wrong end of inverted color scale
  - Background used `var(--neutral-900)` (#f9fafb - light) instead of `var(--neutral-100)` (#1f2937 - dark)
  - Headings used `var(--neutral-100)` (dark text) instead of `var(--neutral-900)` (light text)
- **Solution**: Corrected CSS variables to match dashboard card pattern with !important flags
- **Impact**: ✅ Footer now properly displays with dark background and light text in dark mode

##### **Technical Details**
- **Files**:
  - `src/mcp_memory_service/web/static/index.html` - Footer HTML structure (lines 463-517)
  - `src/mcp_memory_service/web/static/style.css` - Footer styling and dark mode overrides (lines 1757-1893)

## [8.2.2] - 2025-10-05

### ✨ **Enhancements**

#### **HTTP-MCP Bridge: recall_memory Tool Support**
- **Added**: `recall_memory` tool to MCP HTTP bridge API
- **Functionality**: Natural language time-based memory retrieval (e.g., "last week", "yesterday")
- **Integration**: Seamlessly maps to storage backend's `recall_memory` method
- **API**: Accepts `query` (natural language) and optional `n_results` parameter
- **Use Case**: Enables time-aware memory recall through HTTP/MCP bridge interface

##### **Technical Details**
- **File**: `src/mcp_memory_service/web/api/mcp.py`
  - Added `recall_memory` tool definition to `MCP_TOOLS` array
  - Implemented handler in `handle_tool_call()` function
  - Returns standardized format: content, content_hash, tags, created_at

## [8.2.1] - 2025-10-05

### 🐛 **Bug Fixes**

#### **Critical: Missing Core Dependencies**
- **Fixed**: `sentence-transformers` and `torch` moved from optional `[ml]` extras to base dependencies
- **Root cause**: v8.2.0 removed ChromaDB but accidentally made semantic search dependencies optional
- **Impact**: Service failed to start with `ImportError: sentence-transformers is not available`
- **Resolution**: These are core dependencies required for semantic memory functionality
- **Breaking**: Users upgrading from v8.2.0 must run `uv sync` to install corrected dependencies

##### **Technical Details**
- **File**: `pyproject.toml`
  - Moved `sentence-transformers>=2.2.2` from `[ml]` to `dependencies`
  - Moved `torch>=2.0.0` from `[ml]` to `dependencies`
  - Semantic search is core functionality, not optional

## [8.2.0] - 2025-10-05

### ✨ **Dashboard UX Improvements**

#### **Dark Mode Polish**
- **Fixed**: Connection status indicator now properly displays in dark mode
- **Implementation**: Added dark mode CSS override for `.connection-status` component
- **Impact**: ✅ All dashboard elements now fully support dark mode without visual glitches

#### **Browse Tab User Experience**
- **Enhancement**: Automatic smooth scroll to results when clicking a tag
- **Implementation**: Added `scrollIntoView()` with smooth behavior to `filterByTag()` method
- **User Benefit**: No more manual scrolling needed - tag selection immediately shows filtered memories
- **Impact**: ✅ Significantly improved discoverability and flow in Browse by Tags view

##### **Technical Details**
- **File**: `src/mcp_memory_service/web/static/style.css`
  - Added dark mode override for connection status background, border, and text colors
  - Uses CSS variables for consistency with theme system
- **File**: `src/mcp_memory_service/web/static/app.js`
  - Added smooth scroll animation when displaying tag-filtered results
  - Scrolls results section into view with `block: 'start'` positioning

## [8.1.2] - 2025-10-05

### 🐛 **Bug Fixes**

#### **Dashboard Statistics Display**
- **Critical fix**: Dashboard showing 0 for "This Week" and "Tags" statistics on Hybrid and Cloudflare backends
- **Root cause**: Statistics fields not exposed at top level of storage health response

##### **Hybrid Backend Fix** (`src/mcp_memory_service/storage/hybrid.py`)
- Extract `unique_tags` from `primary_stats` to top-level stats dictionary
- Extract `memories_this_week` from `primary_stats` to top-level stats dictionary
- Maintains consistency with SQLite-vec standalone backend behavior

##### **Cloudflare Backend Fix** (`src/mcp_memory_service/storage/cloudflare.py`)
- Added SQL subquery to calculate `unique_tags` from tags table
- Added SQL subquery to calculate `memories_this_week` (last 7 days)
- Now returns both statistics in `get_stats()` response

##### **Impact**
- ✅ Dashboard now correctly displays weekly memory count for all backends
- ✅ Dashboard now correctly displays unique tags count for all backends
- ✅ SQLite-vec standalone backend already had these fields (no change needed)
- ✅ Fixes issue where hybrid/cloudflare users saw "0" despite having memories and tags

## [8.1.1] - 2025-10-05

### 🐛 **Bug Fixes**

#### **Dark Mode Text Contrast Regression**
- **Critical fix**: Memory card text barely visible in dark mode due to hardcoded white backgrounds
- **Root cause**: CSS variable redefinition made text colors too faint when applied to white backgrounds
- **Solution**: Override all major containers with dark backgrounds (`#1f2937`) and force bright text colors

##### **Fixed Components**
- Memory cards: Now use dark card backgrounds with bright white text (`#f9fafb`)
- Memory metadata: Labels bright white (`#f9fafb`), values light gray (`#d1d5db`)
- Action cards: Dark backgrounds for proper contrast
- All containers: App header, welcome card, search filters, modals now properly dark

##### **Technical Details**
- Added `!important` overrides for 11 container backgrounds
- Memory content text: `var(--neutral-900) !important` → `#f9fafb`
- Memory meta labels: `var(--neutral-900) !important` → `#f9fafb`
- Memory meta values: `var(--neutral-600) !important` → `#d1d5db`
- Cache-busting comments to force browser reload

##### **Impact**
- ✅ Dark mode now fully readable across all dashboard views
- ✅ Proper contrast ratios for accessibility
- ✅ No visual regression from v8.1.0 light mode

## [8.1.0] - 2025-10-04

### ✨ **Dashboard Dark Mode & UX Enhancements**

Production-ready dashboard improvements with comprehensive dark mode support, settings management, and optimized CSS architecture.

#### 🎨 **New Features**

##### **Dark Mode Toggle**
- **Clean theme switching** with sun/moon icon toggle in header
- **Persistent preference** via localStorage - theme survives page reloads
- **Smooth transitions** between light and dark themes
- **Full coverage** across all dashboard views (Dashboard, Search, Browse)
- **Performance**: Instant theme switching with CSS class toggle

##### **Settings Modal**
- **Centralized preferences** accessible via cogwheel button
- **User preferences**:
  - Theme selection (Light/Dark)
  - View density (Comfortable/Compact)
  - Memory preview lines (1-10)
- **System information display**:
  - Application version
  - Storage backend configuration (Hybrid/SQLite/Cloudflare)
  - Primary/secondary backend details
  - Embedding model and dimensions
  - Database size
  - Total memories count
  - Server uptime (human-readable format)
- **Robust data loading**: Promise.allSettled() for graceful error handling
- **User feedback**: Toast notifications for save failures

#### 🏗️ **Architecture & Performance**

##### **CSS Optimization - Variable Redefinition Approach**
- **Massive code reduction**: 2116 → 1708 lines (**-408 lines, -19% smaller**)
- **Clean implementation**: Redefine CSS variables in `body.dark-mode` instead of 200+ hardcoded overrides
- **Maintainability**: Single source of truth for dark mode colors
- **Automatic theming**: All components using CSS variables get dark mode support
- **No !important abuse**: Eliminated all !important tags except `.hidden` utility class

##### **JavaScript Improvements**
- **Data-driven configuration**: System info fields defined in static config object
- **Static class properties**: Constants defined once per class, not per instance
- **Robust error handling**: Promise.allSettled() prevents partial failures
- **Zero value handling**: Proper `!= null` checks (displays 0 MB, 0 memories correctly)
- **Smart field updates**: Targeted element updates using config keys

##### **HTML Optimization**
- **SVG icon deduplication**: Info icon defined once in `<defs>`, reused via `<use>`
- **File size reduction**: 4 inline SVG instances → 1 reusable symbol
- **Accessibility**: Proper `aria-hidden` and semantic structure
- **No inline styles**: All styling moved to CSS for better separation of concerns

#### 📊 **Performance Metrics**

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Page Load | <2s | 25ms | ✅ EXCELLENT |
| Memory Operations | <1s | 26ms | ✅ EXCELLENT |
| Tag Search | <500ms | <100ms | ✅ EXCELLENT |
| Theme Toggle | Instant | <1ms | ✅ EXCELLENT |
| CSS File Size | Smaller | -19% | ✅ EXCELLENT |

#### 🔍 **Code Quality**

##### **Gemini Code Assist Review**
- **8 review iterations** - All feedback addressed
- **Final verdict**: "Solid enhancement to the dashboard's user experience"
- **Key improvements**:
  - Variable redefinition pattern for dark mode
  - Removed redundant arrays (derive from Object.keys)
  - SVG icon deduplication
  - Better error messages for users
  - Static method optimization

##### **Files Changed**
- `src/mcp_memory_service/web/static/style.css`: -408 lines (major refactoring)
- `src/mcp_memory_service/web/static/app.js`: +255 lines (settings, theme management)
- `src/mcp_memory_service/web/static/index.html`: +134 lines (modal, icons, SVG defs)
- **Net change**: -19 lines (improved functionality with less code)

#### 🎯 **User Experience**

- **Visual comfort**: Dark mode reduces eye strain for long sessions
- **Personalization**: User-controlled theme and display preferences
- **Transparency**: System information visible in settings modal
- **Feedback**: Error notifications for localStorage failures
- **Consistency**: Dark mode styling matches across all views
- **Accessibility**: High contrast, semantic HTML, keyboard navigation

#### 📝 **Technical Details**

- **Conservative approach**: Original light mode design preserved pixel-perfect
- **Additive CSS**: Dark mode styles never modify existing rules
- **Browser compatibility**: CSS variables, localStorage, SSE all widely supported
- **Mobile responsive**: Works on all screen sizes (tested 768px, 1024px breakpoints)
- **XSS protection**: All user inputs properly escaped via `escapeHtml()`

**PR**: #150 (16 commits, 543 additions, 23 deletions)

---

## [8.0.0] - 2025-10-04

### 💥 **BREAKING CHANGE: ChromaDB Backend Removed**

**This is a major breaking change release**. The ChromaDB backend has been completely removed from the codebase after being deprecated since v5.x.

#### ❌ **Removed**

##### **ChromaDB Backend Complete Removal**
- **Deleted 2,841 lines** of ChromaDB-related code from the codebase
- **Core files removed**:
  - `src/mcp_memory_service/storage/chroma.py` (1,501 lines)
  - `src/mcp_memory_service/storage/chroma_enhanced.py` (176 lines)
  - `tests/unit/test_chroma.py`
  - `tests/chromadb/test_chromadb_types.py`
- **Dependencies removed**:
  - `chromadb` optional dependency group from `pyproject.toml`
  - ~2GB PyTorch + sentence-transformers dependency burden eliminated
- **Factory updates**:
  - Removed ChromaDB backend case from storage factory
  - Removed ChromaStorage initialization logic
  - Added clear error messages directing to migration guide

#### 📦 **Migration & Legacy Support**

##### **ChromaDB Legacy Branch**
- **Branch**: [`chromadb-legacy`](https://github.com/doobidoo/mcp-memory-service/tree/chromadb-legacy)
- **Tag**: `chromadb-legacy-final` - Final ChromaDB code snapshot before removal
- **Status**: Frozen/Archived - No active maintenance
- **Purpose**: Historical reference and migration support

##### **Migration Script Preserved**
- **Location**: `scripts/migration/legacy/migrate_chroma_to_sqlite.py`
- **Status**: Moved to legacy folder, still functional for migrations
- **Alternative**: Check chromadb-legacy branch for additional migration tools

##### **Migration Guide**
See **Issue #148** for comprehensive ChromaDB to Hybrid/SQLite-vec/Cloudflare migration instructions:
- Step-by-step migration procedures
- Data backup and validation steps
- Recommended migration path: **ChromaDB → Hybrid Backend**

#### ✅ **Supported Storage Backends (v8.0.0+)**

| Backend | Status | Use Case | Performance |
|---------|--------|----------|-------------|
| **Hybrid** | ⭐ RECOMMENDED | Production, multi-device | 5ms (SQLite) + cloud sync |
| **SQLite-vec** | ✅ Supported | Development, single-device | 5ms read/write |
| **Cloudflare** | ✅ Supported | Cloud-native, serverless | Network dependent |
| **HTTP Client** | ✅ Supported | Distributed, multi-client | Network dependent |
| **ChromaDB** | ❌ REMOVED | N/A - See legacy branch | N/A |

#### 📊 **Impact & Rationale**

**Why Remove ChromaDB?**
- **Performance**: ChromaDB 15ms vs SQLite-vec 5ms (3x slower)
- **Dependencies**: ~2GB PyTorch download eliminated
- **Maintenance**: 2,841 lines of code removed reduces complexity
- **Better Alternatives**: Hybrid backend provides superior performance with cloud sync

**For Existing ChromaDB Users:**
- **No immediate action required** - Can continue using v7.x releases
- **Upgrade path available** - Migration guide in Issue #148
- **Legacy branch available** - Full code preserved for reference
- **Support timeline**: v7.x will remain available, but no new features

#### 🔧 **Technical Changes**

**Code Removed:**
- ChromaDB storage backend implementations
- ChromaDB-specific tests and fixtures
- ChromaDB configuration handling in factory
- ChromaDB deprecation warnings in server.py

**Error Handling:**
- Attempting to use `MCP_MEMORY_STORAGE_BACKEND=chroma` now raises clear `ValueError`
- Error message includes link to migration guide and legacy branch
- Fallback logic removed - only valid backends accepted

**Dependencies:**
- Removed `chromadb>=0.5.0` from optional dependencies
- Updated `full` dependency group to exclude chromadb
- No impact on core dependencies - only optional dependency cleanup

#### 🚀 **Upgrade Instructions**

**For ChromaDB Users (REQUIRED MIGRATION):**
1. **Backup your data**:
   ```bash
   # Use legacy migration script
   git checkout chromadb-legacy
   python scripts/migration/migrate_chroma_to_sqlite.py
   ```

2. **Switch backend**:
   ```bash
   # Recommended: Hybrid backend (best of both worlds)
   export MCP_MEMORY_STORAGE_BACKEND=hybrid

   # Or: SQLite-vec (local-only)
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

   # Or: Cloudflare (cloud-only)
   export MCP_MEMORY_STORAGE_BACKEND=cloudflare
   ```

3. **Update to v8.0.0**:
   ```bash
   git checkout main
   git pull origin main
   python install.py --storage-backend hybrid
   ```

4. **Validate migration**:
   ```bash
   python scripts/validation/validate_configuration_complete.py
   ```

**For Non-ChromaDB Users (No Action Required):**
- Upgrade seamlessly - no breaking changes for SQLite-vec, Cloudflare, or Hybrid users
- Enjoy reduced dependency footprint and simplified codebase

#### 📚 **Documentation Updates**
- Updated architecture diagrams to show ChromaDB as deprecated/removed
- Updated storage backend comparison tables
- Added migration guide in Issue #148
- Legacy branch README updated with archive notice

#### 🔗 **References**
- **Issue**: #148 - Plan ChromaDB Backend Deprecation and Removal (→ v8.0.0)
- **Legacy Branch**: https://github.com/doobidoo/mcp-memory-service/tree/chromadb-legacy
- **Migration Guide**: See Issue #148 for detailed migration instructions

---

## [7.6.0] - 2025-10-04

### ✨ **Enhanced Document Ingestion with Semtools Support**

#### 🆕 **Core Features**
- **Semtools loader integration** - Optional Rust-based document parser with LlamaParse API for superior extraction quality
- **New format support** - DOCX, DOC, PPTX, XLSX (requires semtools installation)
- **Intelligent chunking** - Respects paragraph and sentence boundaries for better semantic coherence
- **Graceful fallback** - Auto-detects semtools availability, uses native parsers (PyPDF2/pdfplumber) if unavailable
- **Configuration options** - Environment variables for LLAMAPARSE_API_KEY, MCP_DOCUMENT_CHUNK_SIZE, MCP_DOCUMENT_CHUNK_OVERLAP
- **Zero breaking changes** - Fully backward compatible, existing document ingestion unchanged

#### 📄 **Supported Document Formats**
| Format | Native Parser | With Semtools | Quality |
|--------|--------------|---------------|---------|
| PDF | PyPDF2/pdfplumber | ✅ LlamaParse | Excellent (OCR, tables) |
| DOCX/DOC | ❌ Not supported | ✅ LlamaParse | Excellent |
| PPTX | ❌ Not supported | ✅ LlamaParse | Excellent |
| XLSX | ❌ Not supported | ✅ LlamaParse | Excellent |
| TXT/MD | ✅ Built-in | N/A | Perfect |

#### 🔧 **Technical Implementation**
- **New file**: `src/mcp_memory_service/ingestion/semtools_loader.py` (220 lines)
  - SemtoolsLoader class implementing DocumentLoader interface
  - Async subprocess execution with 5-minute timeout for large documents
  - Automatic semtools availability detection via `shutil.which()`
  - LlamaParse API key support via LLAMAPARSE_API_KEY environment variable
  - Comprehensive error handling with detailed logging
- **Modified**: `src/mcp_memory_service/config.py` - Added document processing configuration section (lines 564-586)
- **Modified**: `src/mcp_memory_service/ingestion/registry.py` - Registered new formats (DOCX, PPTX, XLSX)
- **Modified**: `src/mcp_memory_service/ingestion/__init__.py` - Auto-registration of semtools loader
- **Modified**: `CLAUDE.md` - Added comprehensive "Document Ingestion (v7.6.0+)" section with usage examples
- **Tests**: `tests/unit/test_semtools_loader.py` - 12 comprehensive unit tests, all passing ✅

#### 📦 **Installation & Configuration**
```bash
# Optional - install semtools for enhanced parsing
npm i -g @llamaindex/semtools
# or
cargo install semtools

# Optional - configure LlamaParse API for best quality
export LLAMAPARSE_API_KEY="llx-..."

# Document chunking configuration
export MCP_DOCUMENT_CHUNK_SIZE=1000          # Characters per chunk (default: 1000)
export MCP_DOCUMENT_CHUNK_OVERLAP=200        # Overlap between chunks (default: 200)
```

#### 🎯 **Usage Example**
```python
from pathlib import Path
from mcp_memory_service.ingestion import get_loader_for_file

# Automatic format detection and loader selection
loader = get_loader_for_file(Path("document.pdf"))
async for chunk in loader.extract_chunks(Path("document.pdf")):
    await store_memory(chunk.content, tags=["documentation"])
```

#### ✅ **Benefits**
- **Superior PDF parsing** - OCR capabilities and table extraction via LlamaParse
- **Microsoft Office support** - DOCX, PPTX formats now supported (previously unavailable)
- **Production-ready** - Comprehensive error handling, timeout protection, detailed logging
- **Flexible deployment** - Optional enhancement, works perfectly without semtools
- **Automatic detection** - No configuration needed, auto-selects best available parser
- **Minimal overhead** - Only ~5ms initialization cost when semtools not installed

#### 🔗 **Related Issues**
- Closes #94 - Integrate Semtools for Enhanced Document Processing
- Future work tracked in #147 - CLI commands, batch processing, progress reporting, benchmarks

#### 📊 **Test Coverage**
- 12/12 unit tests passing
- Tests cover: initialization, availability checking, file handling, successful extraction, API key usage, error scenarios, timeout handling, empty content, registry integration
- Comprehensive mocking of subprocess execution for reliable CI/CD

## [7.5.5] - 2025-10-04

### 🐛 **Bug Fixes - HybridMemoryStorage Critical Issues**

#### Fixed - Health Check Support (PR #145)
- **HybridMemoryStorage recognition in health checks** - Resolved "Unknown storage type: HybridMemoryStorage" error
- **Dashboard statistics for hybrid backend** - Added comprehensive stats collection from SQLite-vec primary storage
- **Health validation for hybrid storage** - Implemented proper validation logic for hybrid backend
- **Cloudflare sync status visibility** - Display sync service status (not_configured/configured/syncing)

#### Fixed - Missing recall() Method (PR #146)
- **AttributeError on time-based queries** - Added missing `recall()` method to HybridMemoryStorage
- **Server.py compatibility** - Resolves errors when server calls `storage.recall()` with time filtering
- **Consistent API** - Matches method signature of SqliteVecMemoryStorage and CloudflareStorage
- **Delegation to primary** - Properly delegates to SQLite-vec primary storage for recall operations

#### Technical Details
- Added `HybridMemoryStorage` case to `dashboard_get_stats()` endpoint (server.py:2503)
- Added `HybridMemoryStorage` case to `check_database_health()` endpoint (server.py:3705)
- Added `recall()` method to HybridMemoryStorage (hybrid.py:916)
- Method signature: `async def recall(query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]`
- Query primary storage (SQLite-vec) for memory counts, tags, database info
- Fixed code quality issues from Gemini Code Assist review (removed duplicate imports, refactored getattr usage)

#### Impact
- ✅ HTTP dashboard now properly displays hybrid backend statistics
- ✅ MCP health check tool correctly validates hybrid storage
- ✅ Time-based recall queries now work correctly with hybrid backend
- ✅ No more "Unknown storage type" or AttributeError exceptions
- ✅ HybridMemoryStorage fully compatible with all server.py operations

## [7.5.4] - 2025-10-04

### ✨ **Configurable Hybrid Sync Break Conditions**

#### 🔄 **Enhanced Synchronization Control**
- **Configurable early break conditions** - Made hybrid sync termination thresholds configurable via environment variables
  - `MCP_HYBRID_MAX_EMPTY_BATCHES` - Stop after N consecutive batches without new syncs (default: 20, was hardcoded 5)
  - `MCP_HYBRID_MIN_CHECK_COUNT` - Minimum memories to check before early stop (default: 1000, was hardcoded 200)
- **Increased default thresholds** - Quadrupled default values (5→20 batches, 200→1000 memories) to ensure complete synchronization
- **Enhanced logging** - Added detailed sync progress logging every 100 memories with consecutive empty batch tracking
- **Threshold visibility** - Break condition log messages now display threshold values for better diagnostics

#### 🐛 **Bug Fix - Incomplete Synchronization**
- **Resolved incomplete sync issue** - Dashboard was showing only 1040 memories instead of 1200+ from Cloudflare
- **Root cause** - Hardcoded early break conditions triggered prematurely causing missing memories
- **Impact** - Missing memories distributed throughout Cloudflare dataset were never synced to local SQLite

#### ⚙️ **Configuration**
```bash
# Environment variables for tuning sync behavior
export MCP_HYBRID_MAX_EMPTY_BATCHES=20     # Stop after N empty batches (min: 1)
export MCP_HYBRID_MIN_CHECK_COUNT=1000     # Min memories to check before early stop (min: 1)
```

#### 🔧 **Code Quality Improvements**
- **Added input validation** - `min_value=1` constraint prevents zero values that would break sync
- **Fixed progress logging** - Prevents misleading initial log message at `processed_count=0`
- **Eliminated duplicate defaults** - Refactored to use `getattr` pattern for config imports
- **Improved maintainability** - Centralized default values in config.py

#### ✅ **Benefits**
- Complete synchronization of all Cloudflare memories to SQLite
- Configurable per deployment needs without code changes
- Better diagnostics for troubleshooting sync issues
- Maintains protection against infinite loops (early break still active)
- Preserves Cloudflare API protection through configurable limits
- No behavior change for deployments with small datasets

#### 🔗 **References**
- Closes issue: Incomplete hybrid sync (1040/1200+ memories)
- PR #142: Configurable hybrid sync break conditions
- All Gemini Code Assist feedback addressed

## [7.5.3] - 2025-10-04

### 🏗️ **Repository Organization**

#### 📁 **Litestream Sync System Reorganization**
- **Consolidated Litestream scripts** → `scripts/sync/litestream/`
  - Moved 9 shell scripts from `/sync/` directory (git-like staging workflow)
  - Relocated 4 root-level setup scripts (`enhanced_memory_store.sh`, `setup_local_litestream.sh`, etc.)
  - Moved macOS launchd service (`io.litestream.replication.plist`)
  - Moved staging database schema (`staging_db_init.sql`)
- **Created comprehensive documentation** - `scripts/sync/litestream/README.md`
  - Local network HTTP API sync architecture
  - Git-like staging workflow guide
  - Setup and configuration instructions
  - Comparison with Cloudflare hybrid sync

#### 📂 **Deployment Files Consolidation**
- **Moved systemd service** → `scripts/service/mcp-memory.service`
- **Archived unused configs** → `archive/deployment-configs/`
  - `smithery.yaml`
  - `empty_config.yml`
- **Removed empty `/deployment/` directory**

#### 🛠️ **Debug/Investigation Files Organization**
- **Moved to `scripts/development/`**:
  - `debug_server_initialization.py` - Cloudflare backend debugger
  - `verify_hybrid_sync.py` - Hybrid storage verification
- **Archived documentation** → `archive/`
  - `MACOS_HOOKS_INVESTIGATION.md` → `archive/investigations/`
  - `release-notes-v7.1.4.md` → `archive/release-notes/`

#### 📚 **Documentation Updates**
- **Enhanced `scripts/README.md`** with dual sync system documentation
  - Cloudflare Hybrid Sync (cloud backend) section
  - Litestream Sync (local network HTTP API) section
  - Clear distinction between the two systems

### 🎯 **Key Clarifications**
- **Litestream sync**: Multi-device synchronization via central SQLite-vec HTTP API (local network)
  - Use case: Privacy-focused, data stays on local network
  - Architecture: Git-like staging workflow with conflict detection
- **Cloudflare sync**: Cloud-based hybrid backend (internet)
  - Use case: Global access, automatic cloud backup
  - Architecture: Direct sync queue with background operations

### 📦 **Files Affected**
- 27 files changed, 594 insertions(+), 3 deletions(-)
- 13 files renamed/relocated
- 3 new documentation files
- 3 new archive directories

### ⚠️ **Breaking Changes**
None - Purely organizational changes with no functional impact

### 🔄 **Migration Notes**
If using Litestream sync scripts:
- Update script paths: `/sync/memory_sync.sh` → `scripts/sync/litestream/memory_sync.sh`
- Launchd plist location: `/deployment/io.litestream.replication.plist` → `scripts/sync/litestream/io.litestream.replication.plist`
- All scripts remain functionally identical

## [7.5.2] - 2025-10-03

### 🐛 **Bug Fixes**

#### 🔧 **MCP HTTP Endpoint Fixes**
- **Fixed JSON serialization** - Changed `str(result)` to `json.dumps(result)` for proper client parsing
  - MCP endpoint was returning Python dict string representation (`{'key': 'value'}`) instead of valid JSON (`{"key": "value"}`)
  - Caused hook clients to fail parsing responses with "Expected ',' or '}'" errors
- **Fixed similarity threshold** - Changed default from `0.7` to `0.0` to return all relevant memories
  - 70% similarity threshold was too restrictive, filtering out memories with scores 0.2-0.5
  - Now returns all results, allowing client-side scoring to determine relevance

#### 🔌 **Memory Hooks HTTP/HTTPS Protocol Detection**
- **Fixed protocol detection** in `claude-hooks/utilities/memory-client.js`
  - Added `http` module import alongside existing `https` module
  - Implemented dynamic protocol selection: `const protocol = url.protocol === 'https:' ? https : http`
  - Previously hardcoded `https.request()` failed for `http://` endpoints

### 🎯 **Impact**
- ✅ Session-start hooks now properly inject memory context on Claude Code startup
- ✅ HTTP memory server (port 8888) connectivity fully restored
- ✅ Relevant memories (score 0.2-0.5) no longer filtered out by overly restrictive threshold
- ✅ JSON parsing errors resolved for all memory retrieval operations

## [7.5.1] - 2025-10-03

### 🛠️ **Linux Enhancements**

#### 🔄 **Manual Sync Utilities for Hybrid Storage**
- **`sync_now.py` script** - Manual on-demand synchronization for hybrid storage on Linux
  - Type-safe data structures with `TypedDict` (SyncResult, SyncStatus)
  - Comprehensive logging with configurable levels
  - Verbose mode (`--verbose`) for detailed error tracebacks
  - Robust status validation prevents misleading success reports
  - Proper error handling with specific exception types
- **Systemd integration** - Automated hourly background synchronization
  - `mcp-memory-sync.service` - Systemd service for executing sync operations
  - `mcp-memory-sync.timer` - Systemd timer triggering hourly syncs (5min after boot, persistent across reboots)
- **Security improvement** - API key moved to separate environment file in systemd service template

### 🔧 **Code Quality**
- Enhanced error handling throughout sync utilities
- Improved type safety with typed dictionaries for API results
- Better logging practices using `logger.exception()` for verbose errors
- Modular import structure following Python best practices

## [7.5.0] - 2025-10-03

### ✨ **New Features**

#### 🎯 **Backend-Specific Content Length Limits with Auto-Splitting**
- **Intelligent content length management** - Prevents embedding failures by enforcing backend-specific limits
- **Automatic content splitting** - Long content automatically splits into linked chunks with preserved context
- **Backend-aware limits**:
  - Cloudflare: 800 characters (BGE-base-en-v1.5 model 512 token limit)
  - ChromaDB: 1500 characters (all-MiniLM-L6-v2 model 384 token limit)
  - SQLite-vec: Unlimited (local storage)
  - Hybrid: 800 characters (constrained by Cloudflare secondary storage)
- **Smart boundary preservation** - Splits respect natural boundaries (paragraphs → sentences → words)
- **Context preservation** - 50-character overlap between chunks maintains semantic continuity
- **LLM-friendly tool descriptions** - MCP tool docstrings inform LLMs about limits upfront

### 🔧 **Infrastructure Enhancements**

#### 📦 **New Content Splitter Utility**
- **`content_splitter.py` module** - Comprehensive content chunking with boundary-aware splitting
- **Priority-based split points**:
  1. Double newlines (paragraph breaks)
  2. Single newlines
  3. Sentence endings (. ! ? followed by space)
  4. Spaces (word boundaries)
  5. Character position (last resort)
- **Configurable overlap** - Default 50 chars, customizable via `MCP_CONTENT_SPLIT_OVERLAP`
- **Validation helpers** - `estimate_chunks_needed()`, `validate_chunk_lengths()` utilities

#### 🏗️ **Storage Backend Updates**
- **Abstract base class properties** - Added `max_content_length` and `supports_chunking` to `MemoryStorage`
- **Backend implementations**:
  - `CloudflareStorage`: 800 char limit, chunking supported
  - `ChromaMemoryStorage`: 1500 char limit, chunking supported
  - `SqliteVecMemoryStorage`: No limit (None), chunking supported
  - `HybridMemoryStorage`: 800 char limit (follows Cloudflare), chunking supported

#### ⚙️ **Configuration System**
- **New config constants** in `config.py`:
  - `CLOUDFLARE_MAX_CONTENT_LENGTH` (default: 800)
  - `CHROMADB_MAX_CONTENT_LENGTH` (default: 1500)
  - `SQLITEVEC_MAX_CONTENT_LENGTH` (default: None/unlimited)
  - `HYBRID_MAX_CONTENT_LENGTH` (default: 800)
  - `ENABLE_AUTO_SPLIT` (default: True)
  - `CONTENT_SPLIT_OVERLAP` (default: 50)
  - `CONTENT_PRESERVE_BOUNDARIES` (default: True)
- **Environment variable support** - All limits configurable via environment variables
- **Validation and logging** - Safe parsing with min/max bounds and startup logging

### 🛠️ **MCP Server Tool Enhancements**

#### 💾 **Enhanced `store_memory` Tool**
- **Automatic content splitting** - Transparently handles content exceeding backend limits
- **Chunk metadata tracking**:
  - `is_chunk`: Boolean flag identifying chunked memories
  - `chunk_index`: Current chunk number (1-based)
  - `total_chunks`: Total number of chunks
  - `original_length`: Original content length before splitting
- **Chunk tags** - Automatic `chunk:N/M` tags for easy retrieval
- **Enhanced return values**:
  - Single memory: `content_hash`
  - Split content: `chunks_created`, `chunk_hashes` array
- **Updated docstring** - Clear backend limits documentation visible to LLMs

### 🧪 **Testing & Validation**

#### ✅ **Comprehensive Test Suite**
- **`test_content_splitting.py`** - 20+ test cases covering:
  - Basic splitting functionality (short/long content, empty strings)
  - Boundary preservation (paragraphs, sentences, words, code blocks)
  - Overlap validation and chunk estimation
  - Backend limit verification (all 4 backends)
  - Configuration constant validation
- **Edge case coverage** - Empty content, exact lengths, overlaps
- **Integration testing** - Ready for all storage backends

### 📝 **Technical Implementation Details**

#### 🔍 **Design Decisions**
- **Conservative limits** - Buffer below actual token limits to account for tokenization variance
- **Cloudflare priority** - Hybrid backend follows Cloudflare's stricter limit for sync compatibility
- **Opt-out capable** - Set `MCP_ENABLE_AUTO_SPLIT=false` to disable auto-splitting
- **Backward compatible** - No breaking changes to existing functionality

#### ⚡ **Performance Considerations**
- **Minimal overhead** - Content length checks are O(1) property access
- **Efficient chunking** - Single-pass splitting with smart boundary detection
- **No unnecessary splitting** - Content within limits passes through unchanged
- **Batch operations** - All chunks stored in single transaction when possible

### 🔗 **References**
- Addresses issue: First memory store attempt (1,570 chars) exceeded Cloudflare's BGE model limit
- Solution: Backend-specific limits with automatic intelligent content splitting
- Feature branch: `feat/content-length-limits-with-splitting`

## [7.4.1] - 2025-10-03

### 🐛 **Bug Fixes**

#### 🧪 **Claude Hooks Integration Tests**
- **Fixed dual-protocol config compatibility** - Tests now support both legacy (direct endpoint) and new (dual-protocol) configuration structures
- **Improved CI/CD compatibility** - Tests gracefully handle scenarios when memory service is not running
- **Enhanced error handling** - Better detection and handling of connection failures and missing dependencies
- **Achieved 100% test pass rate** - Improved from 78.6% to 100% success rate across all 14 integration tests

### 🔧 **Technical Improvements**
- Updated configuration loading test to detect both `config.memoryService.endpoint` and `config.memoryService.http.endpoint`
- Enhanced connectivity test to treat service unavailability as expected behavior in test environments
- Improved mock session start hook to handle `memoryClient` reference errors gracefully

## [7.4.0] - 2025-10-03

### ✨ **Enhanced Search Tab UX**

#### 🔍 **Advanced Search Functionality**
- **Enhanced date filter options** - Added "Yesterday" and "This quarter" options to improve time-based search granularity
- **Live search mode with toggle** - Implemented intelligent live/manual search modes with debounced input (300ms) to prevent API overload
- **Independent semantic search** - Semantic search now works independently from tag filtering for more flexible query combinations
- **Improved filter behavior** - Fixed confusing filter interactions and enhanced user experience with clear mode indicators

#### 🎨 **UI/UX Improvements**
- **Resolved toggle visibility issues** - Fixed Live Search toggle contrast and visibility problems on white backgrounds
- **Eliminated layout shifts** - Moved toggle to header to prevent dynamic position changes due to text length variations
- **Enhanced tooltips** - Increased tooltip widths (desktop: 300px, mobile: 250px) for better readability
- **Accessible design patterns** - Implemented standard toggle design with proper contrast ratios and always-visible controls

#### ⚡ **Performance Optimization**
- **Debounced search input** - 300ms delay prevents overwhelming API with rapid keystrokes during tag searches
- **Smart search triggering** - Live search mode provides immediate results while manual mode offers user control
- **Efficient event handling** - Optimized DOM manipulation and event listener management

### 🔧 **Code Quality Enhancement**

#### 📚 **DRY Principles Implementation**
- **Eliminated code duplication** - Refactored diagnostic script `test_cloudflare_token()` function following Gemini Code Assist feedback
- **Extracted reusable helper** - Created `_verify_token_endpoint()` function reducing ~60 lines of duplicated token verification logic
- **Enhanced consistency** - Both account-scoped and user endpoint tests now display identical token information fields
- **Improved maintainability** - Centralized error handling and output formatting for easier future extensions

### 🔗 **References**
- Addresses user feedback on search tab UX requiring "further attention" with comprehensive improvements
- Implements Gemini Code Assist code review recommendations from PR #139
- Enhances overall dashboard usability with systematic testing of filter combinations

## [7.3.2] - 2025-10-03

### 🐛 **Critical Bug Fixes**

#### 🔧 **HybridMemoryStorage Import Missing**
- **Fixed critical import error** - Added missing `HybridMemoryStorage` import in `storage/__init__.py` after v7.3.0 update
- **Symptom resolved** - "Unknown storage type: HybridMemoryStorage" error no longer occurs
- **Health check restored** - HTTP dashboard now properly displays hybrid backend status
- **Backwards compatibility** - Import follows same conditional pattern as other storage backends

#### 🛡️ **Enhanced Cloudflare Token Authentication**
- **Resolved token endpoint confusion** - Clear guidance on using account-scoped vs generic verification endpoints
- **Documentation improvements** - Comprehensive `.env.example` with correct curl examples and warnings
- **Enhanced diagnostics** - `diagnose_backend_config.py` now tests both token verification endpoints
- **Developer experience** - New troubleshooting guide prevents common authentication mistakes

### 📚 **Documentation Enhancements**

#### 🔍 **Comprehensive Troubleshooting Guide**
- **New guide:** `docs/troubleshooting/cloudflare-authentication.md` with complete Cloudflare setup guidance
- **Token verification clarity** - Explains difference between account-scoped and generic API endpoints
- **Common errors documented** - Solutions for "Invalid API Token" and related authentication failures
- **Step-by-step checklist** - Systematic approach to diagnosing token and authentication issues

#### ⚙️ **Enhanced Configuration Examples**
- **Improved .env.example** - Combines comprehensive v7.3.1 configuration with token verification guidance
- **Clear warnings** - Explicit guidance on which endpoints to use and avoid
- **Security best practices** - Token handling and verification recommendations

### 🔗 **References**
- Closes critical post-v7.3.0 hybrid storage import issue
- Addresses developer confusion around Cloudflare token verification endpoints
- PR #139: Fix HybridMemoryStorage import + Add comprehensive Cloudflare token verification guide

## [7.3.1] - 2025-10-03

### 🐛 **Bug Fixes**

#### 🔧 **HTTP Dashboard Backend Selection**
- **Fixed HTTP dashboard backend selection** - Dashboard now properly respects `MCP_MEMORY_STORAGE_BACKEND` configuration
- **Universal backend support** - Web interface works with all backends: SQLite-vec, Cloudflare, ChromaDB, and Hybrid
- **Tags functionality restored** - Fixed broken browse by tags feature for all storage backends
- **Shared factory pattern** - Eliminated code duplication between MCP server and web interface initialization

#### 🛠️ **Code Quality Improvements**
- **Extracted fallback logic** - Centralized SQLite-vec fallback handling for better maintainability
- **Enhanced type safety** - Improved type hints throughout web interface components
- **Gemini Code Assistant feedback** - Addressed all code review suggestions for better robustness

### 🔗 **References**
- Closes #136: HTTP Dashboard doesn't use Cloudflare backend despite configuration
- PR #138: Complete universal storage backend support for HTTP dashboard

## [7.3.0] - 2025-10-02

### 🎉 **API Documentation Restoration**

**Successfully restored comprehensive API documentation with interactive dashboard integration following PR #121.**

### ✅ **Key Features**

#### 🔍 **Dual Interface Solution**
- **Dedicated `/api-overview` route** - Standalone comprehensive API documentation page
- **API Documentation tab** - Integrated dashboard tab for seamless user experience
- **Unified navigation** - Consistent access to API information across both interfaces

#### ⚡ **Dynamic Content Loading**
- **Real-time version display** - Dynamic version loading via `/api/health/detailed` endpoint
- **Backend status integration** - Live backend information display
- **Enhanced user awareness** - Always shows current system state

#### 📱 **Enhanced User Experience**
- **Responsive design** - Organized endpoint sections with mobile compatibility
- **Performance optimized** - CSS transitions optimized for better performance
- **Consistent navigation** - Fixed naming conflicts for seamless tab switching

### 🛠️ **Technical Improvements**

#### 🔧 **API Consistency**
- **Fixed endpoint path documentation** - Updated from `{hash}` to `{content_hash}` for accuracy
- **Comprehensive endpoint coverage** - All API endpoints properly documented
- **Organized by functionality** - Logical grouping of endpoints for easy navigation

#### 🎨 **Performance Optimization**
- **CSS performance** - Replaced `transition: all` with specific `border-color` and `box-shadow` transitions
- **Load time maintained** - 25ms page load performance preserved
- **Memory operation speed** - 26ms operation performance maintained

### 📊 **Restored Functionality**

| Feature | Status | Notes |
|---------|--------|-------|
| API Overview Page | ✅ RESTORED | `/api-overview` route with full documentation |
| Dashboard Integration | ✅ NEW | API docs tab in interactive dashboard |
| Dynamic Content | ✅ ENHANCED | Real-time version and backend display |
| Mobile Responsive | ✅ MAINTAINED | CSS breakpoints preserved |
| Performance | ✅ OPTIMIZED | Enhanced CSS transitions |

### 🔄 **Architecture**

#### **Dual Interface Implementation**
- **FastAPI Integration** - `get_api_overview_html()` function with embedded JavaScript
- **Dashboard Enhancement** - Additional navigation tab with organized content sections
- **Unified Styling** - Consistent CSS styling across both interfaces
- **Protocol Independence** - Works with both HTTP and MCP protocols

### 🎯 **User Impact**

**Addresses critical missing functionality:**
- Restores API documentation that was missing after v7.2.2 interactive dashboard
- Provides both standalone and integrated access to API information
- Maintains excellent performance benchmarks while adding functionality
- Enhances developer experience with comprehensive endpoint documentation

**This release ensures users have complete access to API documentation through multiple interfaces while preserving the performance excellence of the interactive dashboard.**

## [7.2.2] - 2025-09-30

### 🎉 **Interactive Dashboard Validation Complete**

**Successfully completed comprehensive testing and validation of the Interactive Dashboard (PR #125).**

### ✅ **Validation Results**
- **Performance Excellence**: Page load 25ms (target: <2s), Memory operations 26ms (target: <1s)
- **Search Functionality**: Semantic search, tag-based search, and time-based search all working perfectly
- **Real-time Updates**: Server-Sent Events (SSE) with heartbeat and connection management validated
- **Security**: XSS protection via escapeHtml function properly implemented throughout frontend
- **OAuth Compatibility**: Both enabled and disabled OAuth modes tested and working
- **Mobile Responsive**: CSS breakpoints for mobile (768px) and tablet (1024px) verified
- **Large Dataset Performance**: Excellent performance tested with 994+ memories
- **Claude Desktop Integration**: MCP protocol compatibility confirmed

### 🚀 **Production Ready**
The Interactive Dashboard is now **fully validated and ready for production use**, providing:
- Complete memory CRUD operations
- Advanced search and filtering capabilities
- Real-time updates via Server-Sent Events
- Mobile-responsive design
- Security best practices
- Excellent performance with large datasets

### 📊 **Testing Metrics**
| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Page Load | <2s | 25ms | ✅ EXCELLENT |
| Memory Ops | <1s | 26ms | ✅ EXCELLENT |
| Tag Search | <500ms | <100ms | ✅ EXCELLENT |
| Large Dataset | 1000+ | 994+ tested | ✅ EXCELLENT |

**Issue #123 closed as completed. Dashboard provides immediate user value and solid foundation for future features.**

## [7.2.0] - 2025-09-30

### 🚀 **Major Performance: ChromaDB Optional Docker Optimization**

**⚠️ BREAKING CHANGE**: ChromaDB is no longer installed by default to dramatically improve Docker build performance and reduce image sizes.

### 🎯 **Key Benefits**
- **70-80% faster Docker build times** (from ~10-15 min to ~2-3 min)
- **1-2GB smaller Docker images** (~2.5GB → ~800MB standard, ~400MB slim)
- **Lower memory footprint** in production deployments
- **Maintained backward compatibility** with clear opt-in mechanism

### 🔧 **Installation Changes**
```bash
# Default installation (lightweight, sqlite_vec only)
python scripts/installation/install.py

# With ChromaDB support (heavy dependencies)
python scripts/installation/install.py --with-chromadb

# Docker builds automatically use optimized sqlite_vec backend
docker build -f tools/docker/Dockerfile -t mcp-memory-service:latest .
```

### 📋 **What Changed**
- **pyproject.toml**: Added `full` optional dependency group, moved ChromaDB to optional
- **server.py**: Added conditional ChromaDB imports with graceful error handling
- **mcp_server.py**: Enhanced ChromaDB import error messages and fallback logic
- **install.py**: Added `--with-chromadb` flag for opt-in ChromaDB installation
- **README.md**: Updated storage backend documentation with ChromaDB optional notes
- **NEW**: `docs/docker-optimized-build.md` - Comprehensive Docker optimization guide

### 🛡️ **Migration Guide**
**For users who need ChromaDB:**
1. Run: `python scripts/installation/install.py --with-chromadb`
2. Or install manually: `pip install mcp-memory-service[chromadb]`

**For Docker users:**
- No action needed - automatically get performance improvements
- Docker builds now default to optimized sqlite_vec backend

### 🧪 **Error Handling**
- Clear error messages when ChromaDB backend selected but not installed
- Graceful fallback to sqlite_vec when ChromaDB unavailable
- Helpful guidance on how to install ChromaDB if needed

### 📊 **Performance Comparison**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Docker build | ~10-15 min | ~2-3 min | **80% faster** |
| Image size | ~2.5GB | ~800MB | **68% smaller** |
| Memory usage | High | Low | **Significantly reduced** |

## [7.1.5] - 2025-09-29

### 🔧 **Improvements**

- **Enhanced timestamp consistency across memory retrieval methods** - All memory retrieval endpoints now display consistent timestamp information:
  - `retrieve_memory` now shows timestamps in "YYYY-MM-DD HH:MM:SS" format matching `recall_memory`
  - `search_by_tag` now shows timestamps in same consistent format
  - Improved code quality using `getattr` pattern instead of `hasattr` checks
  - Resolves timestamp metadata inconsistency reported in issue #126

- **Enhanced CLI hybrid backend support** - CLI commands now fully support hybrid storage backend:
  - Added 'hybrid' option to `--storage-backend` choices for both `server` and `status` commands
  - Completes hybrid backend integration across all system components
  - Enables seamless CLI usage with hybrid SQLite-vec + Cloudflare architecture

- **Hybrid storage backend server integration** - Server.py now fully supports hybrid backend operations:
  - Added `sanitized` method to hybrid storage for tag handling compatibility
  - Enhanced initialization and health check support for hybrid backend
  - Maintains performance optimization with Cloudflare synchronization

### 🛡️ **Security Fixes**

- **Credential exposure prevention** - Enhanced security measures to prevent accidental credential exposure:
  - Improved handling of environment variables in logging and error messages
  - Additional safeguards against sensitive configuration leakage
  - Follows security best practices for credential management

- **Resource leak fixes** - Memory and resource management improvements:
  - Enhanced connection cleanup in storage backends
  - Improved async resource handling to prevent leaks
  - Better error recovery and cleanup procedures

### 🎯 **Code Quality**

- **Implemented Gemini Code Assistant improvements** - Enhanced code maintainability and safety:
  - Replaced `hasattr` + direct attribute access with safer `getattr(obj, "attr", None)` pattern
  - Cleaner, more readable code with consistent error handling
  - Improved null safety and defensive programming practices

## [7.1.4] - 2025-09-28

### 🚀 **Major Feature: Unified Cross-Platform Hook Installer**

- **NEW: Single Python installer replaces 4+ platform-specific scripts**
  - Consolidated `install.sh`, `install-natural-triggers.sh`, `install_claude_hooks_windows.bat` into unified `install_hooks.py`
  - Full cross-platform compatibility (Windows, macOS, Linux)
  - Intelligent JSON configuration merging preserves existing Claude Code hooks
  - Dynamic path resolution eliminates hardcoded developer paths
  - Atomic installations with automatic rollback on failure

- **Enhanced Safety & User Experience**
  - Smart settings.json merging prevents configuration loss
  - Comprehensive backup system with timestamped restore points
  - Empty directory cleanup for proper uninstall process
  - Dry-run support for safe testing before installation
  - Enhanced error handling with detailed user feedback

- **Natural Memory Triggers v7.1.3 Integration**
  - Advanced trigger detection with 85%+ accuracy
  - Multi-tier performance optimization (50ms/150ms/500ms)
  - Mid-conversation memory injection
  - CLI management tools for real-time configuration
  - Git-aware context and repository integration

### 🔧 **Installation Commands Updated**
```bash
# New unified installation (replaces all previous methods)
cd claude-hooks
python install_hooks.py --natural-triggers  # Recommended
python install_hooks.py --basic             # Basic hooks only
python install_hooks.py --all              # Everything

# Integrated with main installer
python scripts/installation/install.py --install-natural-triggers
```

### 📋 **Migration & Documentation**
- Added comprehensive `claude-hooks/MIGRATION.md` with transition guide
- Updated README.md installation instructions
- Legacy shell scripts removed (eliminates security and compatibility issues)
- Clear upgrade path for existing users

### 🛠 **Technical Improvements**
- Addressed all Gemini Code Assist review feedback
- Enhanced cross-platform path handling with proper quoting
- Improved integration between main installer and hook installer
- Professional CLI interface with consistent options across platforms

### ⚠️ **Breaking Changes**
- Legacy shell installers (`install.sh`, `install-natural-triggers.sh`) removed
- Installation commands updated - see `claude-hooks/MIGRATION.md` for details
- Users must switch to unified Python installer for future installations

## [7.1.3] - 2025-09-28

### 🚨 **SECURITY FIX**

- **CRITICAL: Removed sensitive configuration files from repository** - Immediate security remediation:
  - **Removed `.claude/settings.local.json*` files from git tracking and complete history**
  - **Used `git filter-branch` to purge all sensitive data from repository history**
  - **Force-pushed rewritten history to remove exposed API tokens and secrets**
  - Added comprehensive `.gitignore` patterns for future protection
  - **BREAKING: Repository history rewritten - force pull required for existing clones**
  - **ACTION REQUIRED: Rotate any exposed Cloudflare API tokens immediately**
  - Addresses critical security vulnerability from issues #118 and personal config exposure

### ⚠️ **Post-Security Actions Required**
1. **Immediately rotate any Cloudflare API tokens** that were in the exposed files
2. **Force pull** or re-clone repository: `git fetch origin && git reset --hard origin/develop`
3. **Review local `.claude/settings.local.json`** files for any other sensitive data
4. **Verify no sensitive data** remains in your local configurations

## [7.1.2] - 2025-09-28

### 🔧 **Improvements**

- **Stop tracking personal Claude settings to prevent merge conflicts** - Added `.claude/settings.local.json*` patterns to `.gitignore`:
  - Prevents future tracking of personal configuration files
  - Uses `--skip-worktree` to ignore local changes to existing tracked files
  - Protects user privacy and eliminates merge conflicts
  - Preserves existing user configurations while fixing repository hygiene (Fixes #118)

## [7.1.1] - 2025-09-28

### 🐛 **Bug Fixes**

- **Fixed misleading error message in document ingestion** - The `ingest_document` tool now provides accurate error messages:
  - Shows "File not found" with full resolved path when files don't exist
  - Only shows "Unsupported file format" for truly unsupported formats
  - Includes list of supported formats (.md, .txt, .pdf, .json, .csv) in format errors
  - Resolves issue where Markdown files were incorrectly reported as unsupported (Fixes #122)

## [7.1.0] - 2025-09-27

### 🧠 **Natural Memory Triggers for Claude Code**

This release introduces **Natural Memory Triggers v7.1.0** - an intelligent memory awareness system that automatically detects when Claude should retrieve relevant memories from your development history.

#### ✨ **New Features**

##### 🎯 **Intelligent Trigger Detection**
- **✅ Semantic Analysis** - Advanced natural language processing to understand memory-seeking patterns
  - **Pattern Recognition**: Detects phrases like "What did we decide...", "How did we implement..."
  - **Question Classification**: Identifies when user is seeking information from past work
  - **Context Understanding**: Analyzes conversation flow and topic shifts
- **✅ Git-Aware Context** - Repository integration for enhanced relevance
  - **Commit Analysis**: Extracts development themes from recent commit history
  - **Changelog Integration**: Parses project changelogs for version-specific context
  - **Development Keywords**: Builds search queries from git history and file patterns

##### ⚡ **Performance-Optimized Architecture**
- **✅ Multi-Tier Processing** - Three-tier performance system
  - **Instant Tier** (< 50ms): Pattern matching and cache checks
  - **Fast Tier** (< 150ms): Lightweight semantic analysis
  - **Intensive Tier** (< 500ms): Deep semantic understanding
- **✅ Adaptive Performance Profiles**
  - **Speed Focused**: Minimal latency, basic memory awareness
  - **Balanced**: Optimal speed/context balance (recommended)
  - **Memory Aware**: Maximum context awareness
  - **Adaptive**: Machine learning-based optimization

##### 🎮 **CLI Management System**
- **✅ Memory Mode Controller** - Comprehensive command-line interface
  - **Profile Switching**: `node memory-mode-controller.js profile balanced`
  - **Sensitivity Control**: `node memory-mode-controller.js sensitivity 0.7`
  - **Status Monitoring**: Real-time performance metrics and configuration display
  - **System Management**: Enable/disable triggers, reset to defaults

#### 🔧 **Technical Implementation**

##### **Core Components**
- **`claude-hooks/core/mid-conversation.js`** - Main hook implementation with stateful management
- **`claude-hooks/utilities/tiered-conversation-monitor.js`** - Multi-tier semantic analysis engine
- **`claude-hooks/utilities/performance-manager.js`** - Performance monitoring and adaptive optimization
- **`claude-hooks/utilities/git-analyzer.js`** - Git repository context analysis
- **`claude-hooks/memory-mode-controller.js`** - CLI controller for system management

##### **Smart Memory Scoring**
- **✅ Multi-Factor Relevance** - Sophisticated scoring algorithm
  - **Content Relevance** (15%): Semantic similarity to current context
  - **Tag Relevance** (35%): Project and topic-specific weighting
  - **Time Decay** (25%): Recent memories weighted higher
  - **Content Quality** (25%): Filters out low-value memories
- **✅ Conversation Context** - Session-aware analysis
  - **Topic Tracking**: Maintains context window for semantic analysis
  - **Pattern Detection**: Learns user preferences and conversation patterns
  - **Confidence Thresholds**: Only triggers when confidence meets user-defined threshold

#### 🧪 **Quality Assurance**

##### **Comprehensive Testing**
- **✅ Test Suite** - 18 automated tests covering all functionality
  - **Configuration Management**: Nested JSON handling and validation
  - **Performance Profiling**: Latency measurement and optimization
  - **Semantic Analysis**: Pattern detection and confidence scoring
  - **CLI Integration**: Command processing and state management
- **✅ Gemini Code Assist Integration** - AI-powered code review
  - **Static Analysis**: Identified and fixed 21 code quality issues
  - **Performance Optimization**: Division-by-zero prevention, cache management
  - **Configuration Validation**: Duplicate key detection and consolidation

#### 🔄 **Installation & Compatibility**

##### **Seamless Integration**
- **✅ Zero-Restart Installation** - Dynamic hook loading during Claude Code sessions
- **✅ Backward Compatibility** - Works alongside existing memory service functionality
- **✅ Configuration Preservation** - Maintains existing settings while adding new features
- **✅ Platform Support** - macOS, Windows, and Linux compatibility

#### 📊 **Performance Metrics**

##### **Benchmarks**
- **Instant Analysis**: < 50ms response time for pattern matching
- **Fast Analysis**: < 150ms for lightweight semantic processing
- **Cache Performance**: < 5ms for cached results with LRU management
- **Memory Efficiency**: Automatic cleanup prevents memory bloat
- **Trigger Accuracy**: 85%+ confidence for memory-seeking pattern detection

#### 🎯 **Usage Examples**

Natural Memory Triggers automatically activate for phrases like:
- "What approach did we use for authentication?"
- "How did we handle error handling in this project?"
- "What were the main architectural decisions we made?"
- "Similar to what we implemented before..."
- "Remember when we discussed..."

#### 📚 **Documentation**

- **✅ Complete User Guide** - Comprehensive documentation at `claude-hooks/README-NATURAL-TRIGGERS.md`
- **✅ CLI Reference** - Detailed command documentation and usage examples
- **✅ Configuration Guide** - Performance profile explanations and optimization tips
- **✅ Troubleshooting** - Common issues and resolution steps

---

## [7.0.0] - 2025-09-27

### 🎉 **Major Release - OAuth 2.1 Dynamic Client Registration**

This major release introduces comprehensive **OAuth 2.1 Dynamic Client Registration**, enabling **Claude Code HTTP transport** and **enterprise-grade authentication** while maintaining full backward compatibility with existing API key workflows.

#### ✨ **New Features**

##### 🔐 **OAuth 2.1 Implementation**
- **✅ Dynamic Client Registration** - Complete RFC 7591 compliant implementation
  - **Auto-Discovery**: `.well-known/oauth-authorization-server/mcp` endpoint for client auto-configuration
  - **Runtime Registration**: Clients can register dynamically without manual setup
  - **Standards Compliance**: Full OAuth 2.1 and RFC 8414 authorization server metadata
  - **Security Best Practices**: HTTPS enforcement, secure redirect URI validation

- **✅ JWT Authentication** - Modern token-based authentication
  - **RS256 Signing**: RSA key pairs for enhanced security (with HS256 fallback)
  - **Scope-Based Authorization**: Granular permissions (`read`, `write`, `admin`)
  - **Token Validation**: Comprehensive JWT verification with proper error handling
  - **Configurable Expiration**: Customizable token and authorization code lifetimes

##### 🚀 **Claude Code Integration**
- **✅ HTTP Transport Support** - Direct integration with Claude Code
  - **Automatic Setup**: Claude Code discovers and registers OAuth client automatically
  - **Team Collaboration**: Enables Claude Code team features via HTTP transport
  - **Seamless Authentication**: JWT tokens handled transparently by client

##### 🛡️ **Enhanced Security Architecture**
- **✅ Multi-Method Authentication** - Flexible authentication options
  - **OAuth Bearer Tokens**: Primary authentication method for modern clients
  - **API Key Fallback**: Existing API key authentication preserved for backward compatibility
  - **Anonymous Access**: Optional anonymous access with explicit opt-in (`MCP_ALLOW_ANONYMOUS_ACCESS`)

- **✅ Production Security Features**
  - **Thread-Safe Operations**: Async/await with proper locking mechanisms
  - **Background Token Cleanup**: Automatic expiration and cleanup of tokens/codes
  - **Security Validation**: Comprehensive startup validation with production warnings
  - **Configuration Hardening**: HTTP transport warnings, key strength validation

#### 🔧 **Technical Implementation**

##### **New OAuth Endpoints**
- **`/.well-known/oauth-authorization-server/mcp`** - OAuth server metadata discovery
- **`/.well-known/openid-configuration/mcp`** - OpenID Connect compatibility endpoint
- **`/oauth/register`** - Dynamic client registration endpoint
- **`/oauth/authorize`** - Authorization code flow endpoint
- **`/oauth/token`** - Token exchange endpoint (supports both `authorization_code` and `client_credentials` flows)

##### **Authentication Middleware**
- **✅ Unified Auth Handling**: Single middleware protecting all API endpoints
- **✅ Scope Validation**: Automatic scope checking for protected resources
- **✅ Graceful Fallback**: OAuth → API key → Anonymous (if enabled)
- **✅ Enhanced Error Messages**: Context-aware authentication error responses

##### **Configuration System**
- **✅ Environment Variables**: Comprehensive OAuth configuration options
  ```bash
  MCP_OAUTH_ENABLED=true                    # Enable/disable OAuth (default: true)
  MCP_OAUTH_SECRET_KEY=<secure-key>         # JWT signing key (auto-generated if not set)
  MCP_OAUTH_ISSUER=<issuer-url>            # OAuth issuer URL (auto-detected)
  MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES=60  # Token expiration (default: 60 minutes)
  MCP_ALLOW_ANONYMOUS_ACCESS=false         # Anonymous access (default: false)
  ```

#### 🔄 **Backward Compatibility**
- **✅ Zero Breaking Changes**: All existing API key workflows continue to work unchanged
- **✅ Optional OAuth**: OAuth can be completely disabled with `MCP_OAUTH_ENABLED=false`
- **✅ Graceful Coexistence**: API key and OAuth authentication work side-by-side
- **✅ Migration Path**: Existing users can adopt OAuth gradually or continue with API keys

#### 📊 **Development & Quality Metrics**
- **✅ 17 Comprehensive Review Cycles** with Gemini Code Assist feedback integration
- **✅ All Security Issues Resolved** (critical, high, medium severity vulnerabilities addressed)
- **✅ Extensive Testing Suite**: New integration tests for OAuth flows and security scenarios
- **✅ Production Readiness**: Comprehensive validation, monitoring, and health checks

#### 🚀 **Impact & Benefits**

##### **For Existing Users**
- **No Changes Required**: Continue using API key authentication without modification
- **Enhanced Security**: Option to upgrade to industry-standard OAuth when ready
- **Future-Proof**: Foundation for additional enterprise features

##### **For Claude Code Users**
- **Team Collaboration**: HTTP transport enables Claude Code team features
- **Automatic Setup**: Zero-configuration OAuth setup and token management
- **Enterprise Ready**: Standards-compliant authentication for organizational use

##### **For Enterprise Environments**
- **Standards Compliance**: Full OAuth 2.1 and RFC compliance for security audits
- **Centralized Auth**: Foundation for integration with existing identity providers
- **Audit Trail**: Comprehensive logging and token lifecycle management

#### 🔜 **Future Enhancements**
This release provides the foundation for additional OAuth features:
- **Persistent Storage**: Production-ready client and token storage backends
- **PKCE Support**: Enhanced security for public clients
- **Refresh Tokens**: Long-lived authentication sessions
- **User Consent UI**: Interactive authorization flows
- **Identity Provider Integration**: SAML, OIDC, and enterprise SSO support

#### 📚 **Documentation**
- **✅ Complete Setup Guide**: Step-by-step OAuth configuration documentation (`docs/oauth-setup.md`)
- **✅ API Reference**: Comprehensive endpoint documentation with examples
- **✅ Security Guide**: Production deployment best practices and security considerations
- **✅ Migration Guide**: Smooth transition path for existing users

---

**This major release transforms the MCP Memory Service from a simple memory tool into an enterprise-ready service with standards-compliant authentication, enabling new use cases while preserving the simplicity that makes it valuable.**

## [6.23.0] - 2025-09-27

### 🎉 **Major Feature Release - Memory Management Enhancement**

This release combines three major improvements: comprehensive memory management tools, enhanced documentation, and dependency standardization. All changes have been reviewed and approved by Gemini Code Assist with very positive feedback.

#### ✨ **New Features**
- **🛠️ New `list_memories` MCP Tool** - Added paginated memory browsing with filtering capabilities
  - ✅ **Pagination Support**: Page-based navigation (1-based indexing) with configurable page sizes (1-100)
  - ✅ **Database-Level Filtering**: Filter by memory type and tags using efficient SQL queries
  - ✅ **Performance Optimized**: Direct database filtering instead of Python-level post-processing
  - ✅ **Consistent API**: Available in both MCP server and HTTP/REST endpoints

#### 🚀 **Performance Improvements**
- **⚡ Database-Level Filtering** - Replaced inefficient Python-level filtering with SQL WHERE clauses
  - ❌ **Previous**: Fetch all records → filter in Python → paginate (slow, memory-intensive)
  - ✅ **Now**: Filter + paginate in database → return results (5ms response time)
  - ✅ **Benefits**: Dramatically reduced memory usage and improved response times for large datasets
  - ✅ **Backends**: Implemented across SQLite-vec, ChromaDB, Cloudflare, and Hybrid storage

- **🔧 Enhanced Storage Interface** - Extended `get_all_memories()` with tags parameter
  - ✅ **Tag Filtering**: Support for OR-based tag matching at database level
  - ✅ **Backward Compatible**: All existing code continues to work unchanged
  - ✅ **Consistent**: Same interface across all storage backends

#### 🛡️ **Security Enhancements**
- **🔒 Eliminated Security Vulnerabilities** - Removed dangerous runtime dependency installation
  - ❌ **Removed**: Automatic `pip install` execution in Docker containers
  - ✅ **Security**: Prevents potential code injection and supply chain attacks
  - ✅ **Reliability**: Dependencies now properly managed through container build process

- **🔑 Fixed Hardcoded Credentials** - Replaced hardcoded API keys with environment variables
  - ❌ **Previous**: API keys stored in plain text in debug scripts
  - ✅ **Fixed**: All credentials now sourced from secure environment variables
  - ✅ **Security**: Follows security best practices for credential management

#### 📚 **Documentation Improvements**
- **📖 Comprehensive Documentation Suite** - Added professional documentation in `docs/mastery/`
  - ✅ **API Reference**: Complete API documentation with examples
  - ✅ **Architecture Overview**: Detailed system architecture documentation
  - ✅ **Configuration Guide**: Comprehensive configuration management guide
  - ✅ **Setup Instructions**: Step-by-step local setup and run guide
  - ✅ **Testing Guide**: Testing strategies and debugging instructions
  - ✅ **Troubleshooting**: Common issues and solutions

- **🔧 Enhanced Development Resources** - Added advanced search and refactoring documentation
  - ✅ **Search Enhancement Guide**: Advanced search capabilities and examples
  - ✅ **Refactoring Summary**: Complete analysis of architectural changes
  - ✅ **Integration Examples**: Multi-client setup for various AI platforms

#### 🔧 **Infrastructure Improvements**
- **🐳 Docker Optimization** - Enhanced Docker configuration for production deployments
  - ✅ **Security Updates**: Updated base images and security patches
  - ✅ **Performance**: Optimized container size and startup time
  - ✅ **Flexibility**: Better support for different deployment scenarios

- **📦 Dependency Management** - Standardized and improved dependency handling
  - ✅ **ChromaDB Compatibility**: Restored ChromaDB as optional dependency for backward compatibility
  - ✅ **Updated Dependencies**: Updated PyPDF2 → pypdf2 for better maintenance
  - ✅ **Optional Dependencies**: Clean separation of core vs optional features

#### 🪟 **Platform Support**
- **💻 Enhanced Windows Support** - Added comprehensive Windows debugging capabilities
  - ✅ **Debug Script**: New `start_http_debug.bat` for Windows HTTP mode testing
  - ✅ **103 Lines Added**: Comprehensive Windows debugging and troubleshooting support
  - ✅ **Environment Variables**: Proper Windows environment variable handling

#### 🧹 **Code Quality**
- **♻️ Major Refactoring** - Removed redundant functionality while maintaining compatibility
  - ✅ **317 Lines Removed**: Eliminated duplicate `search_by_time` and `search_similar` tools
  - ✅ **Functional Redundancy**: Removed tools that exactly duplicated existing functionality
  - ✅ **API Consolidation**: Streamlined API surface while preserving all capabilities
  - ✅ **Performance**: Reduced codebase complexity without losing features

#### 🤖 **AI Code Review Integration**
- **✅ Gemini Code Assist Approved** - All changes reviewed and approved with very positive feedback
  - ✅ **Architecture Review**: Praised database-level filtering implementation
  - ✅ **Security Review**: Confirmed elimination of security vulnerabilities
  - ✅ **Performance Review**: Validated performance optimization approach
  - ✅ **Code Quality**: Approved refactoring and redundancy removal

#### 📋 **Migration Notes**
- **🔄 Backward Compatibility**: All existing integrations continue to work unchanged
- **📦 Optional Dependencies**: ChromaDB users should install with `pip install mcp-memory-service[chromadb]`
- **🛠️ New Tools**: The `list_memories` tool is automatically available to all MCP clients
- **⚠️ Removed Tools**: `search_by_time` and `search_similar` tools have been removed (functionality available through existing tools)

#### 💡 **Usage Examples**
```python
# New list_memories tool with filtering
await list_memories(page=1, page_size=20, tag="important", memory_type="note")

# Database-level tag filtering (improved performance)
memories = await storage.get_all_memories(limit=50, tags=["work", "project"])

# Enhanced pagination with type filtering
memories = await storage.get_all_memories(
    limit=10, offset=20, memory_type="decision", tags=["urgent"]
)
```

---

## [6.22.1] - 2025-09-26

### 🔧 **Dashboard Statistics Fix**

#### Bug Fixes
- **🎯 Backend-Agnostic Dashboard Stats** - Fixed `dashboard_get_stats` to use configured storage backend instead of hardcoded ChromaDB
  - ❌ **Previous Issue**: Dashboard always showed ChromaDB stats (often 0 memories) regardless of actual backend
  - ✅ **Fixed**: Now properly detects and uses SQLite-vec, Cloudflare, or ChromaDB based on configuration
  - ✅ **Consistency**: Uses same pattern as `handle_check_database_health` for reliable backend detection
  - ✅ **Accuracy**: Dashboard now shows correct memory counts and backend information

#### Technical Improvements
- **Backend Detection**: Dynamic storage type detection via `storage.__class__.__name__`
- **Error Handling**: Proper async/await handling and graceful error reporting
- **Code Consistency**: Unified approach with existing health check functionality

---

**Resolves**: GitHub Issue where dashboard stats were incorrectly hardcoded to ChromaDB
**Credit**: Thanks to @MichaelPaulukonis for identifying and fixing this backend detection issue

---

## [6.22.0] - 2024-09-25

### 🎯 **Chronological Ordering & Performance Improvements**

#### Major API Enhancements
- **🌟 Chronological Memory Ordering** - `/api/memories` endpoint now returns memories in chronological order (newest first)
  - ✅ **Improved User Experience**: More intuitive memory browsing with recent memories prioritized
  - ✅ **Consistent Across All Backends**: SQLite-vec, ChromaDB, Cloudflare D1, and Hybrid
  - ✅ **Proper Pagination Support**: Server-side sorting with efficient limit/offset handling
  - ✅ **Backward Compatible**: Same API interface with enhanced ordering

#### Critical Performance Fixes 🚀
- **⚡ Storage-Layer Memory Type Filtering** - Addressed critical performance bottleneck
  - ❌ **Previous Issue**: API loaded ALL memories into application memory when filtering by `memory_type`
  - ✅ **Fixed**: Efficient storage-layer filtering with SQL WHERE clauses
  - ✅ **Performance Impact**: 16.5% improvement in filtering operations
  - ✅ **Scalability**: Prevents service instability with large datasets (1000+ memories)
- **Enhanced Storage Interface**
  - Added `memory_type` parameter to `get_all_memories()` and `count_all_memories()` methods
  - Implemented across all backends: SQLite-vec, ChromaDB, Cloudflare D1, Hybrid
  - Maintains chronological ordering while applying efficient filters

#### Code Quality Improvements
- **🔧 ChromaDB Code Refactoring** - Eliminated code duplication
  - Created `_create_memory_from_results()` helper method
  - Consolidated 5 duplicate Memory object creation patterns
  - Enhanced maintainability and consistency across ChromaDB operations
- **Comprehensive Test Suite**
  - Added 10 new test cases specifically for chronological ordering
  - Covers edge cases: empty storage, large offsets, mixed timestamps
  - Validates API endpoint behavior and storage backend compatibility

#### Backend-Specific Optimizations
- **SQLite-vec**: Efficient `ORDER BY created_at DESC` with parameterized WHERE clauses
- **ChromaDB**: Client-side sorting with performance warnings for large datasets (>1000 memories)
- **Cloudflare D1**: Server-side SQL sorting and filtering for optimal performance
- **Hybrid**: Delegates to primary storage (SQLite-vec) for consistent performance

#### Developer Experience
- Enhanced error handling and logging for filtering operations
- Improved API response consistency across all storage backends
- Better performance monitoring and debugging capabilities

---

**Resolves**: GitHub Issue #79 - Implement chronological ordering for /api/memories endpoint
**Addresses**: Gemini Code Assist performance and maintainability feedback

---

## [6.21.0] - 2024-09-25

### 🚀 **Hybrid Storage Backend - Performance Revolution**

#### Major New Features
- **🌟 Revolutionary Hybrid Storage Backend** - Combines the best of both worlds:
  - ✅ **SQLite-vec Performance**: ~5ms reads/writes (10-100x faster than Cloudflare-only)
  - ✅ **Cloudflare Persistence**: Multi-device synchronization and cloud backup
  - ✅ **Zero User-Facing Latency**: All operations hit SQLite-vec first, background sync to cloud
  - ✅ **Intelligent Write-Through Cache**: Instant response with async cloud synchronization

#### Enhanced Architecture & Performance
- **Background Synchronization Service**
  - Async queue with intelligent retry logic and exponential backoff
  - Concurrent sync operations with configurable batch processing
  - Real-time health monitoring and capacity tracking
  - Graceful degradation when cloud services are unavailable
- **Advanced Error Handling**
  - Intelligent error categorization (temporary vs permanent vs limit errors)
  - Automatic retry for network/temporary issues
  - No-retry policy for hard limits (prevents infinite loops)
  - Comprehensive logging with error classification

#### Cloudflare Limit Protection & Monitoring 🛡️
- **Pre-Sync Validation**
  - Metadata size validation (10KB limit per vector)
  - Vector count monitoring (5M vector limit)
  - Automatic capacity checks before sync operations
- **Real-Time Capacity Monitoring**
  - Usage percentage tracking with warning thresholds
  - Critical alerts at 95% capacity, warnings at 80%
  - Proactive limit detection and graceful handling
- **Enhanced Limit Error Handling**
  - Detection of 413, 507, and quota exceeded responses
  - Automatic capacity status updates on limit errors
  - Permanent failure classification for hard limits

#### Configuration & Deployment
- **Simple Setup**: Just set `MCP_MEMORY_STORAGE_BACKEND=hybrid` + Cloudflare credentials
- **Advanced Tuning Options**:
  - `MCP_HYBRID_SYNC_INTERVAL`: Background sync frequency (default: 300s)
  - `MCP_HYBRID_BATCH_SIZE`: Sync batch size (default: 50)
  - `MCP_HYBRID_MAX_QUEUE_SIZE`: Queue capacity (default: 1000)
  - Health check intervals and retry configurations

#### Benefits
- **For Users**:
  - Instant memory operations (no more waiting for cloud responses)
  - Reliable offline functionality with automatic sync when online
  - Seamless multi-device access to memories
- **For Production**:
  - Handles Cloudflare's strict limits intelligently
  - Robust error recovery and monitoring
  - Scales from single-user to enterprise deployments

### 🧪 **Comprehensive Testing & Validation**
- **347 lines of Cloudflare limit testing** (`tests/test_hybrid_cloudflare_limits.py`)
- **Performance characteristic validation**
- **Background sync verification scripts**
- **Live testing utilities for production validation**

### 📖 **Documentation & Setup**
- **CLAUDE.md**: Hybrid marked as **RECOMMENDED** default for new installations
- **Installation Script Updates**: Interactive hybrid backend selection
- **Configuration Validation**: Enhanced diagnostic tools for setup verification

**🎯 Recommendation**: This should become the **default backend for all new installations** due to its superior performance and reliability characteristics.

## [6.20.1] - 2024-09-24

### 🐛 **Critical Bug Fixes**

#### SQLite-vec Backend Regression Fix
- **Fixed MCP Server Initialization**: Corrected critical regression that prevented sqlite_vec backend from working
  - ✅ Fixed class name mismatch: `SqliteVecStorage` → `SqliteVecMemoryStorage`
  - ✅ Fixed constructor parameters: Updated to use correct `db_path` and `embedding_model` parameters
  - ✅ Fixed database path: Use `SQLITE_VEC_PATH` instead of incorrect ChromaDB path
  - ✅ Added missing imports: `SQLITE_VEC_PATH` and `EMBEDDING_MODEL_NAME` from config
  - ✅ Code quality improvements: Added `_get_sqlite_vec_storage()` helper function to reduce duplication

#### Impact
- **Restores Default Backend**: sqlite_vec backend (default) now works correctly with MCP server
- **Fixes Memory Operations**: Resolves "No embedding model available" errors during memory operations
- **Claude Desktop Integration**: Enables proper memory storage and retrieval functionality
- **Embedding Support**: Ensures embedding model loads and generates embeddings successfully

Thanks to @ergut for identifying and fixing this critical regression!

## [6.20.0] - 2024-09-24

### 🚀 **Claude Code Dual Protocol Memory Hooks**

#### Major New Features
- **Dual Protocol Memory Hook Support** - Revolutionary enhancement to Claude Code memory hooks
  - ✅ **HTTP Protocol Support**: Full compatibility with web-based memory services at `https://localhost:8443`
  - ✅ **MCP Protocol Support**: Direct integration with MCP server processes via `uv run memory server`
  - ✅ **Smart Auto-Detection**: Automatically selects best available protocol (MCP preferred, HTTP fallback)
  - ✅ **Graceful Fallback Chain**: MCP → HTTP → Environment-based storage detection
  - ✅ **Protocol Flexibility**: Choose specific protocols (`http`, `mcp`) or auto-selection (`auto`)

#### Enhanced Architecture
- **Unified MemoryClient Class** (`claude-hooks/utilities/memory-client.js`)
  - Transparent protocol switching with single interface
  - Connection pooling and error recovery
  - Protocol-specific optimizations (MCP direct communication, HTTP REST API)
  - Comprehensive error handling and timeout management
- **Enhanced Configuration System** (`claude-hooks/config.json`)
  - Protocol-specific settings (HTTP endpoint/API keys, MCP server commands)
  - Configurable fallback behavior and connection timeouts
  - Backward compatibility with existing configurations

#### Reliability Improvements
- **Multi-Protocol Resilience**: Hooks work across diverse deployment scenarios
  - Local development (MCP direct), production servers (HTTP), hybrid setups
  - Network connectivity issues gracefully handled
  - Service unavailability doesn't break git analysis or project detection
- **Enhanced Error Handling**: Clear protocol-specific error messages and fallback reporting
- **Connection Management**: Proper cleanup and resource management for both protocols

#### Developer Experience
- **Comprehensive Testing Suite** (`claude-hooks/test-dual-protocol-hook.js`)
  - Tests all protocol combinations: auto-MCP-preferred, auto-HTTP-preferred, MCP-only, HTTP-only
  - Validates protocol detection, fallback behavior, and error handling
  - Demonstrates graceful degradation capabilities
- **Backward Compatibility**: Existing hook configurations continue working unchanged
- **Enhanced Debugging**: Protocol selection and connection status clearly reported

#### Technical Implementation
- **Protocol Abstraction Layer**: Single interface for memory operations regardless of protocol
- **Smart Connection Logic**: Connection attempts with timeouts, fallback sequencing
- **Memory Query Unification**: Semantic search, time-based queries work identically across protocols
- **Storage Backend Detection**: Enhanced parsing for both HTTP JSON responses and MCP tool output

#### Benefits for Different Use Cases
- **Claude Desktop Users**: Better reliability with HTTP fallback when MCP struggles
- **VS Code Extension Users**: Optimized for HTTP-based deployments
- **CI/CD Systems**: More robust memory operations in automated environments
- **Development Workflows**: Local MCP for speed, HTTP for production consistency

## [6.19.0] - 2024-09-24

### 🔧 **Configuration Validation Scripts Consolidation**

#### Improvements
- **Consolidated validation scripts** - Merged `validate_config.py` and `validate_configuration.py` into comprehensive `validate_configuration_complete.py`
  - ✅ Multi-platform support (Windows/macOS/Linux)
  - ✅ All configuration sources validation (.env, Claude Desktop, Claude Code)
  - ✅ Cross-configuration consistency checking
  - ✅ Enhanced API token validation with known invalid token detection
  - ✅ Improved error reporting and recommendations
  - ✅ Windows console compatibility (no Unicode issues)

#### Removed
- ❌ **Deprecated scripts**: `validate_config.py` and `validate_configuration.py` (redundant)

#### Fixed
- **Cloudflare Backend Critical Issue**: Implemented missing `recall` method in CloudflareStorage class
  - ✅ Dual search strategy (semantic + time-based)
  - ✅ Graceful fallback mechanism
  - ✅ Comprehensive error handling
  - ✅ Time filtering support

#### Documentation Updates
- **Updated all documentation references** to use new consolidated validation script
- **Created comprehensive API token setup guide** (`docs/troubleshooting/cloudflare-api-token-setup.md`)

## [6.18.0] - 2025-09-23

### 🚀 **Cloudflare Dual-Environment Configuration Suite**

#### New Diagnostic Tools
- **Added comprehensive backend configuration diagnostic script** (`scripts/validation/diagnose_backend_config.py`)
  - Environment file validation with masked sensitive data display
  - Environment variable loading verification with dotenv support
  - Configuration module import testing with clear error reporting
  - Storage backend creation testing with full traceback on failures
  - Status indicators with clear success/warning/error messaging
- **Enhanced troubleshooting workflow** with step-by-step validation process

#### Documentation Improvements
- **Created streamlined 5-minute setup guide** (`docs/quick-setup-cloudflare-dual-environment.md`)
  - Comprehensive dual-environment configuration for Claude Desktop + Claude Code
  - Configuration templates with explicit environment variable examples
  - Validation commands with expected health check results
  - Troubleshooting section for common configuration issues
  - Migration guide from SQLite-vec to Cloudflare backend
- **Fixed incorrect CLAUDE.md documentation** that suggested SQLite-vec as "expected behavior"
- **Added configuration management best practices** with environment variable precedence
- **Enhanced troubleshooting sections** with specific solutions for environment variable loading issues

#### Configuration Enhancements
- **Improved environment variable loading reliability** with explicit MCP server configuration
- **Added execution context guidance** for different environments (Claude Desktop vs Claude Code)
- **Enhanced working directory awareness** for proper .env file loading
- **Better configuration validation** with clear error messages for missing required variables

#### Technical Improvements
- **Unified diagnostic approach** for both Cloudflare and SQLite-vec backends
- **Enhanced error reporting** with masked sensitive data for security
- **Improved configuration precedence handling** between global and project settings
- **Better cross-platform path handling** for Windows environments

#### Benefits for Users
- **Eliminates configuration confusion** between different execution environments
- **Provides clear validation tools** to quickly identify and resolve setup issues
- **Ensures consistent backend usage** across Claude Desktop and Claude Code
- **Streamlines Cloudflare backend adoption** with comprehensive setup guidance
- **Reduces setup time** from complex debugging to 5-minute guided process

## [6.17.2] - 2025-09-23

### 🔧 **Development Environment Stability Fix**

#### Module Isolation Improvements
- **Enhanced script module loading** in `scripts/server/run_memory_server.py` to prevent version conflicts
- **Added module cache clearing** to remove conflicting cached imports before loading local development code
- **Improved path prioritization** to ensure local `src/` directory takes precedence over installed packages
- **Better logging** shows exactly which modules are being cleared and paths being added for debugging

#### Technical Improvements
- **Prevents import conflicts** between development code and installed package versions
- **Ensures consistent behavior** when switching between development and production environments
- **Fixes version mismatch issues** that could cause `ImportError` for missing attributes like `INCLUDE_HOSTNAME`
- **More robust script execution** with conditional path management based on environment

#### Benefits for Developers
- **Reliable development environment** - Local changes always take precedence
- **Easier debugging** - Clear logging of module loading process
- **Consistent Cloudflare backend** - No more fallback to ChromaDB due to version conflicts
- **Zero breaking changes** - Maintains compatibility with all existing configurations

## [6.17.1] - 2025-09-23

### 🔧 **Script Reorganization Compatibility Hotfix**

#### Backward Compatibility Added
- **Added compatibility stub** at `scripts/run_memory_server.py` that redirects to new location with helpful migration notices
- **Updated configuration templates** to use Python module approach as primary method for maximum stability
- **Added comprehensive migration documentation** for users updating from pre-v6.17.0 versions
- **Zero disruption approach**: Existing configurations continue working immediately

#### Recommended Launch Methods (in order of stability)
1. **`python -m mcp_memory_service.server`** - Most stable, no path dependencies, works across all reorganizations
2. **`uv run memory server`** - Integrated with UV tooling, already documented as preferred
3. **`scripts/server/run_memory_server.py`** - Direct script execution at new location
4. **`scripts/run_memory_server.py`** - Legacy location with backward compatibility (shows migration notice)

#### Documentation Improvements
- **Enhanced README**: Clear migration notice with multiple working options
- **Updated examples**: Python module approach as primary recommendation
- **Migration guide**: Created comprehensive GitHub issue ([#108](https://github.com/doobidoo/mcp-memory-service/issues/108)) with all approaches
- **Template updates**: Configuration templates now show most stable approaches first

#### Why This Approach
- **Immediate relief**: No users are blocked during v6.17.0 update
- **Multiple pathways**: Users can choose the approach that fits their setup
- **Future-proof**: Python module approach survives any future directory changes
- **Clear migration path**: Informational notices guide users to better practices without forcing changes

## [6.17.0] - 2025-09-22

### 🚀 **Enhanced Installer with Cloudflare Backend Support**

#### Major Installer Improvements
- **Added Cloudflare backend to installer**: Full support for cloud-first installation workflow
  - **Interactive credential setup**: Guided collection of API token, Account ID, D1 database, and Vectorize index
  - **Automatic .env generation**: Securely saves credentials to project environment file
  - **Connection testing**: Validates Cloudflare API during installation process
  - **Graceful fallbacks**: Falls back to local backends if cloud setup fails
- **Enhanced backend selection logic**: Usage-based recommendations for optimal backend choice
  - **Production scenarios**: Cloudflare for shared access and cloud storage
  - **Development scenarios**: SQLite-vec for single-user, lightweight setup
  - **Team scenarios**: ChromaDB for multi-client local collaboration
- **Improved CLI options**: Updated `--storage-backend` with clear use case descriptions
  - **New choices**: `cloudflare` (production), `sqlite_vec` (development), `chromadb` (team), `auto_detect`
  - **Better help text**: Explains when to use each backend option

#### User Experience Enhancements
- **Interactive backend selection**: Guided setup with compatibility analysis and recommendations
- **Clear usage guidance**: Backend selection now includes use case scenarios and performance characteristics
- **Enhanced auto-detection**: Prioritizes most reliable backends for the detected system
- **Comprehensive documentation**: Updated installation commands and backend comparison table

#### Technical Improvements
- **Robust error handling**: Comprehensive fallback mechanisms for failed setups
- **Modular design**: Separate functions for credential collection, validation, and environment setup
- **Connection validation**: Real-time API testing during Cloudflare backend configuration
- **Environment file management**: Smart .env file handling that preserves existing settings

#### Benefits for Users
- **Seamless production setup**: Single command path from installation to Cloudflare backend
- **Reduced configuration errors**: Automated credential setup eliminates manual .env file creation
- **Better backend choice**: Clear guidance helps users select optimal storage for their use case
- **Improved reliability**: Fallback mechanisms ensure installation succeeds even with setup issues

## [6.16.1] - 2025-09-22

### 🔧 **Docker Build Hotfix**

#### Infrastructure Fix
- **Fixed Docker build failure**: Updated Dockerfile script path after v6.15.0 scripts reorganization
  - **Issue**: Docker build failing due to `scripts/install_uv.py` not found
  - **Solution**: Updated path to `scripts/installation/install_uv.py`
  - **Impact**: Restores automated Docker publishing workflows
- **No functional changes**: Pure infrastructure fix for CI/CD

## [6.16.0] - 2025-09-22

### 🔧 **Configuration Management & Backend Selection Fixes**

#### Critical Configuration Issues Resolved
- **Fixed Cloudflare backend fallback issue**: Resolved service falling back to SQLite-vec despite correct Cloudflare configuration
  - **Root cause**: Configuration module wasn't loading `.env` file automatically
  - **CLI override issue**: CLI default parameter was overriding environment variables
  - **Solution**: Added automatic `.env` loading and fixed CLI parameter precedence
- **Enhanced environment loading**: Added `load_dotenv()` to configuration initialization
  - **Automatic detection**: Config module now automatically loads `.env` file when present
  - **Backward compatibility**: Graceful fallback if python-dotenv not available
  - **Logging**: Added confirmation logging when environment file is loaded
- **Fixed CLI parameter precedence**: Changed CLI defaults to respect environment configuration
  - **Server command**: Changed `--storage-backend` default from `'sqlite_vec'` to `None`
  - **Environment priority**: Environment variables now take precedence over CLI defaults
  - **Explicit overrides**: CLI parameters only override when explicitly provided

#### Content Size Management Improvements
- **Added Cloudflare content limits to context provider**: Enhanced memory management guidance
  - **Content size warnings**: Added ~1500 character limit documentation
  - **Embedding model constraints**: Documented `@cf/baai/bge-base-en-v1.5` strict input limits
  - **Best practices**: Guidance for chunking large content and using document ingestion
  - **Error recognition**: Help identifying "Failed to store vector" errors from size issues
- **Enhanced troubleshooting**: Better error messages and debugging capabilities for configuration issues

#### Technical Improvements
- **Configuration validation**: Improved environment variable loading and validation
- **Error handling**: Better error messages when storage backend initialization fails
- **Documentation**: Updated context provider with Cloudflare-specific constraints and best practices

#### Benefits for Users
- **Seamless backend switching**: Cloudflare configuration now works reliably out of the box
- **Fewer configuration errors**: Automatic environment loading reduces setup friction
- **Better error diagnosis**: Clear guidance on content size limits and chunking strategies
- **Improved reliability**: Configuration precedence issues eliminated


---

## Historic Releases

For older releases (v6.15.1 and earlier), see [CHANGELOG-HISTORIC.md](./CHANGELOG-HISTORIC.md).

**Historic Version Range**: v0.1.0 through v6.15.1 (2025-07-XX through 2025-09-22)
