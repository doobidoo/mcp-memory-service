# Changelog

**Recent releases for MCP Memory Service (v6.16.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

For older releases, see [CHANGELOG-HISTORIC.md](./CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
