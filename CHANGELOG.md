# Changelog

**Recent releases for MCP Memory Service (v8.0.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

For older releases, see [CHANGELOG-HISTORIC.md](./CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [8.20.0] - 2025-11-08

### Added
- **PR Automation Workflow** - Comprehensive Gemini Code Assist integration
  - `auto_review.sh` - Automated iterative PR review cycles (saves 10-30 min/PR)
  - `quality_gate.sh` - Pre-PR quality checks (complexity, security, tests, breaking changes)
  - `generate_tests.sh` - Auto-generate pytest tests for new code
  - `detect_breaking_changes.sh` - API diff analysis with severity classification
- **Code Quality Guard Agent** - Fast automated code quality analysis
  - Pre-commit hook for complexity scoring (blocks >8, warns >7)
  - Security pattern detection (SQL injection, XSS, command injection)
  - TODO prioritization with impact analysis (Critical/High/Medium/Low)
  - Integrated with Gemini CLI for fast analysis
- **Gemini PR Automator Agent** - Eliminates manual review wait times
  - Automated "Fix → Comment → /gemini review → Wait → Repeat" workflow
  - Intelligent fix classification (safe vs unsafe)
  - Comprehensive test generation for all new code
  - Breaking change detection with severity levels
- **Groq Bridge Integration** - 10x faster alternative to Gemini CLI
  - Python CLI (`groq_agent_bridge.py`) for non-interactive LLM calls
  - Support for llama-3.3-70b-versatile, Kimi K2 (256K context), and other Groq models
  - Drop-in replacement for Gemini CLI with same interface
  - Model comparison documentation with performance benchmarks
- **Pre-commit Hook Infrastructure** - Quality gates before commit
  - Symlinked hook: `.git/hooks/pre-commit` → `scripts/hooks/pre-commit`
  - Automated complexity checks on staged Python files
  - Security vulnerability scanning (blocks commit if found)
  - Graceful degradation if Gemini CLI not installed

### Documentation
- Added `docs/integrations/groq-bridge.md` - Setup and usage guide for Groq integration
- Added `docs/integrations/groq-integration-summary.md` - Integration overview
- Added `docs/integrations/groq-model-comparison.md` - Performance benchmarks and model selection
- Added `docs/amp-cli-bridge.md` - Amp CLI integration guide
- Added `docs/development/todo-tracker.md` - Automated TODO tracking output
- Added `docs/troubleshooting/hooks-quick-reference.md` - Hook debugging and troubleshooting
- Added `docs/document-ingestion.md` - Document parsing guide
- Added `docs/legacy/dual-protocol-hooks.md` - Historical hook configuration
- Added `scripts/maintenance/memory-types.md` - Memory type taxonomy documentation

### Changed
- Updated `.claude/agents/github-release-manager.md` - Enhanced with latest release workflow
- Updated `.env.example` - Removed deprecated ChromaDB backend references
- Updated API specification - Removed `chroma` from backend enum
- Updated example configs - Modernized to use Python module approach

### Fixed
- Removed all ChromaDB artifacts from active code (deprecated in v8.0.0)
  - Fixed broken test imports (`CHROMADB_MAX_CONTENT_LENGTH` removed)
  - Updated integration tests to remove `--chroma-path` CLI assertions
  - Cleaned up example configurations
  - Added clear deprecation warnings in `install.py`

### Performance
- Groq bridge provides 10x faster inference vs Gemini CLI
  - llama-3.3-70b: ~300ms response time
  - Kimi K2: ~200ms response time (fastest)
  - llama-3.1-8b-instant: ~300ms response time
- Pre-commit hooks minimize developer wait time with complexity checks

## [8.19.1] - 2025-11-07

### Fixed
- **Critical MCP Tool Regressions** (Issue #211) - Two core MCP tools broken in v8.19.0
  - **retrieve_memory**: Fixed parameter error where unsupported 'tags' parameter was passed to storage.retrieve()
    - Removed unsupported parameters from storage call
    - Implemented post-retrieval filtering in MemoryService
    - File: `src/mcp_memory_service/services/memory_service.py`
  - **search_by_tag**: Fixed type error where code assumed created_at was always string
    - Added type checking for both float (timestamp) and string (ISO format)
    - Uses `datetime.fromtimestamp()` for floats, `datetime.fromisoformat()` for strings
    - Files: `src/mcp_memory_service/server.py` (handle_search_by_tag, handle_retrieve_memory)
  - **Impact**: Both tools now work correctly with hybrid storage backend
  - **Root Cause**: v8.19.0 refactoring introduced incompatibilities with hybrid storage

## [8.19.0] - 2025-11-07

### Added
- **Code Execution Interface API** 🚀 - Revolutionary token efficiency for MCP Memory Service (Issue #206)
  - **Core API Module** (`src/mcp_memory_service/api/`):
    - Compact NamedTuple types (91% size reduction vs MCP tool responses)
    - Core operations: `search()`, `store()`, `health()`
    - Sync wrapper utilities (hide asyncio complexity)
    - Storage client with connection reuse
  - **Session Hook Migration** (`claude-hooks/core/session-start.js`):
    - Code execution wrapper for token-efficient memory retrieval
    - Automatic MCP fallback (100% reliability)
    - Performance metrics tracking
    - Zero breaking changes
  - **Migration Guide** (`docs/migration/code-execution-api-quick-start.md`):
    - 5-minute migration workflow
    - Cost savings calculator
    - Comprehensive troubleshooting
    - Platform-specific instructions
  - **Comprehensive Documentation**:
    - API reference guide
    - Phase 1 and Phase 2 implementation summaries
    - Research documents (10,000+ words)
    - Performance benchmarks

### Changed
- **Installer Default** - Code execution now enabled by default for new installations
  - Auto-detects Python path (Windows: `python`, Unix: `python3`)
  - Validates Python version (warns if < 3.10)
  - Shows token reduction benefits during installation
  - Graceful upgrade path for existing users

### Performance
- **Token Reduction (Validated)**:
  - Session hooks: **75% reduction** (3,600 → 900 tokens)
  - Search operations: **85% reduction** (2,625 → 385 tokens)
  - Store operations: **90% reduction** (150 → 15 tokens)
  - Health checks: **84% reduction** (125 → 20 tokens)
- **Execution Performance**:
  - Cold start: 61.1ms (target: <100ms)
  - Warm calls: 1.3ms avg (target: <10ms)
  - Memory overhead: <10MB
- **Annual Savings Potential**:
  - 10 users: **$23.82/year** (158.8M tokens saved)
  - 100 users: **$238.20/year** (1.59B tokens saved)
  - 1000 users: **$2,382/year** (15.9B tokens saved)

### Fixed
- **Cross-Platform Compatibility** - Replaced hardcoded macOS paths with `get_base_directory()`
- **Async/Await Pattern** - Fixed async `health()` to properly await storage operations
- **Resource Management** - Added explicit `close()` and `close_async()` methods
- **Documentation** - Fixed Unicode symbols and absolute local paths for better compatibility

### Technical Details
- **Files Added**: 23 files, 6,517 lines (production + tests + docs)
- **Test Coverage**: 52 tests (88% passing)
- **Backward Compatibility**: 100% (zero breaking changes)
- **Platform Support**: macOS, Linux, Windows
- **API Exports**: `search`, `store`, `health`, `close`, `close_async`

### Removed
- **Obsolete ChromaDB Test Infrastructure** - Removed all test files referencing deprecated ChromaDB storage backend
  - **Deleted Files**:
    - `tests/performance/test_caching.py` - ChromaDB-specific performance tests
    - `tests/test_timestamp_recall.py` - Timestamp tests using ChromaDB APIs
    - `tests/test_tag_storage.py` - Tag storage tests for ChromaDB
    - `tests/integration/test_storage.py` - ChromaDB storage diagnostic script
    - `tests/unit/test_tags.py` - Tag deletion tests using deprecated CHROMA_PATH
    - `tests/test_content_splitting.py::test_chromadb_limit()` - ChromaDB content limit test
  - **Updated**: `tests/conftest.py` - Generic database testing fixture (removed ChromaDB reference)
  - **Context**: ChromaDB backend was fully replaced by SQLite-vec, Cloudflare, and Hybrid backends
  - **Impact**: Test suite no longer has broken imports or references to removed storage layer

## [8.18.2] - 2025-11-06

### Fixed
- **MCP Tool Handler Method Name** - Fixed critical bug where MCP tool handlers called non-existent `retrieve_memory()` method
  - **Root Cause**: Method name mismatch introduced in commit 36e9845 during MemoryService refactoring (Oct 28, 2025)
  - **Symptom**: `'MemoryService' object has no attribute 'retrieve_memory'` error when using MCP retrieve_memory tool
  - **Fix**: Updated handlers to call correct `retrieve_memories()` method (plural)
  - **Files Modified**: `src/mcp_memory_service/server.py` (line 2153), `src/mcp_memory_service/mcp_server.py` (line 227)
  - **Additional**: Removed unsupported `min_similarity` parameter from MCP tool definition
  - **Impact**: MCP retrieve_memory tool now functions correctly
  - **Issue**: [#207](https://github.com/doobidoo/mcp-memory-service/issues/207)

## [8.18.1] - 2025-11-05

### Fixed
- **Test Suite Import Errors** - Resolved critical import failures blocking all test collection
  - **MCP Client Import**: Updated from deprecated `mcp` module to `mcp.client.session` (v1.1.2 API)
  - **Storage Path Import**: Removed obsolete `CHROMA_PATH` constant that no longer exists in current architecture
  - **Impact**: Restored ability to collect and run 190 test cases across unit, integration, and E2E test suites
  - **PR**: [#205](https://github.com/doobidoo/mcp-memory-service/pull/205)
  - **Issue**: [#204](https://github.com/doobidoo/mcp-memory-service/issues/204)

### Technical Details
- **Test Files Updated**: `tests/unit/test_import.py`, `tests/integration/test_store_memory.py`
- **Import Fix**: `from mcp import ClientSession, StdioServerParameters` → `from mcp.client.session import ClientSession`
- **Cleanup**: Removed dependency on deprecated storage constants

## [8.18.0] - 2025-11-05

### Performance
- **Analytics Dashboard Optimization** - 90% performance improvement for dashboard analytics
  - **New Method**: `get_memory_timestamps()` added to all storage backends (SQLite-Vec, Cloudflare, Hybrid)
  - **Optimization**: Single optimized SQL query instead of N individual queries for timestamp retrieval
  - **Impact**: Dashboard analytics load significantly faster, especially with large memory datasets
  - **PR**: [#203](https://github.com/doobidoo/mcp-memory-service/pull/203)
  - **Issue**: [#186](https://github.com/doobidoo/mcp-memory-service/issues/186)

### Added
- **Type Safety Enhancements** - Pydantic models for analytics data structures
  - **Models**: `LargestMemory` (content hash, size, created timestamp) and `GrowthTrendPoint` (date, cumulative count)
  - **Benefit**: Enhanced type checking and validation for analytics endpoints
  - **File**: `src/mcp_memory_service/web/api/analytics.py`

### Fixed
- **Weekly Activity Year Handling** - Fixed incorrect aggregation of same week numbers across different years
  - **Issue**: Week 1 of 2024 and Week 1 of 2025 were being combined in weekly activity charts
  - **Solution**: Year included in grouping logic for proper temporal separation
  - **Impact**: Weekly activity charts now accurately reflect year boundaries

### Technical Details
- **Storage Interface**: `get_memory_timestamps()` method added to base `MemoryStorage` class
- **Backend Implementation**: Optimized SQL queries for SQLite-Vec, Cloudflare D1, and Hybrid storage
- **Code Quality**: Import organization and naming consistency improvements per Gemini Code Assist review

## [8.17.1] - 2025-11-05

### Documentation
- **Workflow Documentation Harvest** - Comprehensive project management templates adapted from PR #199
  - **GitHub Issue Templates**: Structured bug/feature/performance reports with environment validation
    - `bug_report.yml`: OS, Python version, storage backend, reproduction steps
    - `feature_request.yml`: Use case, alternatives, impact assessment
    - `performance_issue.yml`: Metrics, database size, profiling data
    - `config.yml`: Template selector with Wiki/Discussions links
  - **Regression Test Scenarios**: 10 structured test procedures (Setup → Execute → Evidence → Pass/Fail)
    - Database locking tests (concurrent MCP servers, concurrent operations)
    - Storage backend integrity (hybrid sync, backend switching)
    - Dashboard performance (page load <2s, operations <1s)
    - Tag filtering correctness (exact matching, index usage)
    - MCP protocol compliance (schema validation, tool execution)
  - **PR Review Guide**: Standardized code review with Gemini integration
    - Template compliance checklist, code quality standards (type hints, async patterns, security)
    - Testing verification (>80% coverage), merge criteria, effective feedback templates
  - **Issue Management Guide**: Triage, closure, and post-release workflows
    - 48h triage response, professional closure templates, GitHub CLI automation
    - Issue→PR→Release tracking, post-release systematic review
  - **Release Checklist Enhancements**: Version management and emergency procedures
    - 3-file version bump procedure (`__init__.py` → `pyproject.toml` → `uv lock`)
    - CHANGELOG quality gates, GitHub workflow verification
    - Emergency rollback procedure (Docker/PyPI/Git steps, recovery timeline)
  - **Impact**: 2,400+ lines of actionable documentation (7 new files, 1 enhanced)
  - **Files**: `.github/ISSUE_TEMPLATE/`, `docs/testing/regression-tests.md`, `docs/development/pr-review-guide.md`, `docs/development/issue-management.md`, `docs/development/release-checklist.md`
  - **Source**: Adapted from [PR #199](https://github.com/doobidoo/mcp-memory-service/pull/199) by Ray Walker (27Bslash6)
  - **Commit**: [ca5ccf3](https://github.com/doobidoo/mcp-memory-service/commit/ca5ccf3f460feca06d3c9232303a6e528ca2c76f)

### Fixed
- **CRITICAL: Dashboard Analytics Accuracy** - Fixed analytics endpoint showing incorrect memory count
  - **Issue**: Dashboard displayed 1,000 memories (sampling) instead of actual 2,222 total
  - **Root Cause**: Analytics endpoint used `get_recent_memories(n=1000)` sampling approach instead of direct SQL query
  - **Solution**: Direct SQL query via `storage.primary.conn` for `HybridMemoryStorage` backend
  - **File**: `src/mcp_memory_service/web/api/analytics.py` (line 386)
  - **Impact**: Dashboard now shows accurate memory totals for all storage backends
  - **Additional Fix**: Corrected attribute check from `storage.sqlite` to `storage.primary` for hybrid backend detection

### Added
- **Malformed Tags Repair Utility** - Intelligent repair tool for JSON serialization artifacts in tags
  - **Script**: `scripts/maintenance/repair_malformed_tags.py`
  - **Repaired**: 1,870 malformed tags across 369 memories
  - **Patterns Detected**: JSON quotes (`\"tag\"`), brackets (`[tag]`), mixed artifacts
  - **Features**: Multi-tier parser, dry-run mode, automatic backups, progress tracking
  - **Safety**: Creates backup before modifications, validates repairs

- **Intelligent Memory Type Assignment** - Automated type inference for untyped memories
  - **Script**: `scripts/maintenance/assign_memory_types.py`
  - **Processed**: 119 untyped memories with multi-tier inference algorithm
  - **Inference Methods**:
    - Tag-based mapping (80+ tag-to-type associations)
    - Pattern-based detection (40+ content patterns)
    - Metadata analysis (activity indicators)
    - Fallback heuristics (default to "note")
  - **Confidence Scoring**: High/medium/low confidence levels for inference quality
  - **Features**: Dry-run mode, automatic backups, detailed statistics

### Technical Details
- **Analytics Fix**: Removed "Using sampling approach" warning from logs
- **Database Backups**: Both maintenance scripts create timestamped backups before modifications
- **Type Taxonomy**: Follows 24 core types from consolidated memory type system
- **Hybrid Backend**: Scripts properly detect and handle `HybridMemoryStorage` architecture

## [8.17.0] - 2025-11-04

### Added
- **Platform-Aware Consolidation Tool** - Cross-platform database path detection for memory type consolidation
  - **Auto-detection**: macOS (`~/Library/Application Support`), Windows (`%LOCALAPPDATA%`), Linux (`~/.local/share`)
  - **Script**: `scripts/maintenance/consolidate_memory_types.py` enhanced with `platform.system()` detection
  - **Benefit**: Works seamlessly across all platforms without manual path configuration
  - **PR**: [#201](https://github.com/doobidoo/mcp-memory-service/pull/201)

- **External JSON Configuration** - Editable consolidation mappings for flexible type management
  - **File**: `scripts/maintenance/consolidation_mappings.json` (294 mappings, v1.1.0)
  - **Schema**: JSON Schema validation with taxonomy documentation
  - **Extensibility**: Add custom type mappings without code changes
  - **Taxonomy**: 24 core types organized into 5 categories
  - **PR**: [#201](https://github.com/doobidoo/mcp-memory-service/pull/201)

- **Agent System Documentation** - Comprehensive agent guidelines for development workflows
  - **File**: `AGENTS.md` - Central documentation for available agents
  - **Agent**: `amp-bridge` - Amp CLI integration for research without credit consumption
  - **Integration**: Semi-automated file-based workflow with Amp's `@file` reference syntax
  - **CLAUDE.md**: Updated with Amp CLI Bridge architecture and quick start guide
  - **PR**: [#201](https://github.com/doobidoo/mcp-memory-service/pull/201)

### Fixed
- **Dashboard Analytics Chart Layout** - Resolved rendering and proportionality issues
  - **Fixed**: Chart bars rendering outside containers
  - **Fixed**: Uniform bar sizes despite different values
  - **Fixed**: Memory type distribution showing incorrect proportions
  - **Enhancement**: Switched to 200px pixel scale for proper visualization
  - **Enhancement**: CSS container constraints with `overflow-x: auto` and flexbox improvements
  - **Files**: `web/static/app.js`, `web/static/index.html`, `web/static/style.css`
  - **PR**: [#200](https://github.com/doobidoo/mcp-memory-service/pull/200)

### Technical Details
- **Platform Detection**: Uses `platform.system()` for "Darwin" (macOS), "Windows", and Linux
- **Backward Compatibility**: Existing scripts continue to work with new platform-aware paths
- **Agent Workflow**: Claude Code creates prompts → User runs `amp @prompt-file` → Amp writes response → Agent presents results
- **Chart Rendering**: Dashboard now properly visualizes memory distribution with accurate proportions

## [8.16.2] - 2025-11-03

### Fixed
- **Critical Bug**: Fixed bidirectional sync script crash due to incorrect Memory attribute access
  - **File**: `scripts/sync/sync_memory_backends.py`
  - **Root Cause**: Script accessed non-existent `memory.id` and `memory.hash` attributes instead of correct `memory.content_hash`
  - **Impact**: Bidirectional sync between SQLite-vec and Cloudflare backends completely broken
  - **Lines Fixed**: 81, 121, 137, 146, 149, 151, 153, 155, 158
  - **Testing**: Dry-run validated with 1,978 SQLite memories and 1,958 Cloudflare memories
  - **Result**: Successfully identified 338 memories to sync, 1,640 duplicates correctly skipped

### Technical Details
- **Error Type**: `AttributeError: 'Memory' object has no attribute 'id'/'hash'`
- **Correct Attribute**: `content_hash` - SHA-256 based unique identifier for memory content
- **Affected Methods**: `get_all_memories_from_backend()`, `_sync_between_backends()`
- **Backward Compatibility**: No breaking changes, only fixes broken functionality

## [8.16.1] - 2025-11-02

### Fixed
- **Critical Bug**: Fixed `KeyError: 'message'` in MCP server handler (`server.py:2118`)
  - **Issue**: [#198](https://github.com/doobidoo/mcp-memory-service/issues/198)
  - **Root Cause**: `handle_store_memory()` attempted to access non-existent `result["message"]` key
  - **Impact**: All memory store operations via MCP `server.py` handler failed completely
  - **Fix**: Properly handle `MemoryService.store_memory()` response format:
    - Success (single): `{"success": True, "memory": {...}}`
    - Success (chunked): `{"success": True, "memories": [...], "total_chunks": N}`
    - Failure: `{"success": False, "error": "..."}`
  - **Response Messages**: Now include truncated content hash for verification
  - **Related**: This was part of issue #197 where async/await bug was fixed in v8.16.0, but this response format bug was missed

### Added
- **Integration Tests**: New test suite for MCP handler methods (`tests/integration/test_mcp_handlers.py`)
  - **Coverage**: 11 test cases for `handle_store_memory()`, `handle_retrieve_memory()`, `handle_search_by_tag()`
  - **Regression Tests**: Specific tests for issue #198 to prevent future KeyError bugs
  - **Test Scenarios**: Success, chunked response, error handling, edge cases
  - **Purpose**: Prevent similar bugs in future releases

### Technical Details
- **Affected Handler**: Only `handle_store_memory()` was affected by this bug
- **Fixed Code Pattern**: Matches the correct pattern used in `mcp_server.py:182-205`
- **Backward Compatibility**: No breaking changes, only fixes broken functionality

## [8.16.0] - 2025-11-01

### Added
- **Memory Type Consolidation Tool** 🆕 - Professional-grade database maintenance for type taxonomy cleanup
  - **Script**: `scripts/maintenance/consolidate_memory_types.py` (v1.0.0)
  - **Configuration**: `scripts/maintenance/consolidation_mappings.json` (168 predefined mappings)
  - **Performance**: ~5 seconds for 1,000 memory updates
  - **Safety Features**:
    - ✅ Automatic timestamped backups before execution
    - ✅ Dry-run mode for safe preview
    - ✅ Transaction safety (atomic with rollback)
    - ✅ Database lock detection
    - ✅ HTTP server status warnings
    - ✅ Disk space verification
    - ✅ Backup integrity validation
  - **Consolidates**: 341 fragmented types → 24 core taxonomy types
  - **Real-world test**: 1,049 memories updated in 5s (59% of database)
  - **Type reduction**: 342 → 128 unique types (63% reduction)
  - **Zero data loss**: Only type reassignments, preserves all content

- **Standardized Memory Type Taxonomy** - 24 core types organized into 5 categories
  - **Content Types** (4): note, reference, document, guide
  - **Activity Types** (5): session, implementation, analysis, troubleshooting, test
  - **Artifact Types** (4): fix, feature, release, deployment
  - **Progress Types** (2): milestone, status
  - **Infrastructure Types** (5): configuration, infrastructure, process, security, architecture
  - **Other Types** (4): documentation, solution, achievement, technical
  - **Purpose**: Prevents future type fragmentation
  - **Benefits**: Improved query efficiency, consistent naming, better semantic grouping

### Changed
- **CLAUDE.md** - Added Memory Type Taxonomy section to Development Guidelines
  - Documents 24 core types with clear usage guidelines
  - Provides examples of what to avoid (bug-fix vs fix, technical-* prefixes)
  - Added consolidation commands to Essential Commands section
  - Includes best practices for maintaining type consistency

### Documentation
- **Comprehensive Maintenance Documentation**
  - Updated `scripts/maintenance/README.md` with consolidation tool guide
  - Added to Quick Reference table with performance metrics
  - Detailed usage instructions with safety prerequisites
  - Recovery procedures for backup restoration
  - Maintenance schedule recommendations (monthly dry-run checks)
  - **Real-world example**: Production consolidation results from Nov 2025

### Technical Details
- **Consolidation Mappings**:
  - NULL/empty → `note` (609 memories in real test)
  - Milestone/completion variants → `milestone` (21 source types → 60 memories)
  - Session variants → `session` (8 source types → 37 memories)
  - Technical-* prefix removal → base types (62 memories)
  - Project-* prefix removal → base types (67 memories)
  - Fix/bug variants → `fix` (8 source types → 28 memories)
  - See `consolidation_mappings.json` for complete mapping list (168 rules)

### Notes
- **Customizable**: Edit `consolidation_mappings.json` to customize behavior
- **Idempotent**: Safe to run multiple times with same mappings
- **Platform support**: Linux, macOS, Windows (disk space check requires statvfs)
- **Recommended schedule**: Run --dry-run monthly, execute when types exceed 150

## [8.15.1] - 2025-10-31

### Fixed
- **Critical Python Syntax Error in Hook Installer** - Fixed IndentationError in `claude-hooks/install_hooks.py` (line 790)
  - **Issue**: Extra closing braces in SessionEnd hook configuration caused installation to fail
  - **Symptom**: `IndentationError: unexpected indent` when running `python install_hooks.py`
  - **Root Cause**: Git merge conflict resolution left two extra `}` characters (lines 790-791)
  - **Impact**: Users could not install or update hooks after pulling v8.15.0
  - **Fix**: Removed extra closing braces, corrected indentation
  - **Files Modified**: `claude-hooks/install_hooks.py`
  - **Testing**: Verified successful installation on macOS after fix

### Technical Details
- **Line Numbers**: 788-791 in install_hooks.py
- **Error Type**: IndentationError (Python syntax)
- **Detection Method**: Manual testing during hook reinstallation
- **Resolution Time**: Immediate (same-day patch)

## [8.15.0] - 2025-10-31

### Added
- **`/session-start` Slash Command** - Manual session initialization for all platforms
  - Provides same functionality as automatic SessionStart hook
  - Displays project context, git history, relevant memories
  - **Platform**: Works on Windows, macOS, Linux
  - **Use Case**: Primary workaround for Windows SessionStart hook bug (#160)
  - **Location**: `claude_commands/session-start.md`
  - **Benefits**:
    - ✅ Safe manual alternative to automatic hooks
    - ✅ No configuration changes needed
    - ✅ Full memory awareness functionality
    - ✅ Prevents Claude Code hangs on Windows

### Changed
- **Windows-Aware Installer** - Platform detection and automatic configuration
  - Detects Windows platform during hook installation
  - Automatically skips SessionStart hook configuration on Windows
  - Shows clear warning about SessionStart limitation
  - Suggests `/session-start` slash command as alternative
  - Also skips statusLine configuration on Windows (requires bash)
  - **Files Modified**: `claude-hooks/install_hooks.py` (lines 749-817)
  - **User Impact**: Prevents Windows users from accidentally breaking Claude Code

### Documentation
- **Enhanced Windows Support Documentation**
  - Updated `claude_commands/README.md` with `/session-start` command details
  - Added Windows workaround section to `claude-hooks/README.md`
  - Updated `CLAUDE.md` with `/session-start` as #1 recommended workaround
  - Added comprehensive troubleshooting guidance
  - Updated GitHub issue #160 with new workaround instructions

### Fixed
- **Windows Installation Experience** - Prevents SessionStart hook deadlock (Issue #160)
  - **Previous Behavior**: Windows users install hooks → Claude Code hangs → frustration
  - **New Behavior**: Windows users see warning → skip SessionStart → use `/session-start` → works
  - **Breaking Change**: None - fully backward compatible
  - **Upstream Issue**: Awaiting fix from Anthropic Claude Code team (claude-code#9542)

### Technical Details
- **Files Created**: 1 new slash command
  - `claude_commands/session-start.md` - Full command documentation
- **Files Modified**: 4 files
  - `claude-hooks/install_hooks.py` - Windows platform detection and warnings
  - `claude_commands/README.md` - Added `/session-start` to available commands
  - `claude-hooks/README.md` - Added slash command workaround reference
  - `CLAUDE.md` - Updated workaround prioritization

- **Platform Compatibility**:
  - ✅ Windows: `/session-start` command works, automatic hooks skipped
  - ✅ macOS: All features work (automatic hooks + slash command)
  - ✅ Linux: All features work (automatic hooks + slash command)

## [8.14.2] - 2025-10-31

### Performance
- **Optimized MemoryService.get_memory_by_hash()** - O(1) direct lookup replaces O(n) scan (Issue #196)
  - **Previous Implementation**: Loaded ALL memories via `storage.get_all_memories()`, then filtered by hash
  - **New Implementation**: Direct O(1) database lookup via `storage.get_by_hash()`
  - **Performance Impact**:
    - Small datasets (10-100 memories): ~2x faster
    - Medium datasets (100-1000 memories): ~10-50x faster
    - Large datasets (1000+ memories): ~100x+ faster
  - **Memory Usage**: Single memory object vs loading entire dataset into memory

### Added
- **Abstract method `get_by_hash()` in MemoryStorage base class** (storage/base.py)
  - Enforces O(1) direct hash lookup across all storage backends
  - Required implementation for all storage backends
  - Returns `Optional[Memory]` (None if not found)

- **Implemented `get_by_hash()` in CloudflareStorage** (storage/cloudflare.py)
  - Direct D1 SQL query: `SELECT * FROM memories WHERE content_hash = ?`
  - Handles R2 content loading if needed
  - Loads tags separately
  - Follows same pattern as SqliteVecMemoryStorage implementation

### Changed
- **MemoryService.get_memory_by_hash()** now uses direct storage lookup
  - Removed inefficient `get_all_memories()` + filter approach
  - Simplified implementation (5 lines vs 10 lines)
  - Updated docstring to reflect O(1) lookup

### Testing
- **Updated unit tests** (tests/unit/test_memory_service.py)
  - Mocks now use `storage.get_by_hash()` instead of `storage.get_all_memories()`
  - Added assertions to verify method calls
  - All 3 test cases pass (found, not found, error handling)

- **Updated integration tests** (tests/integration/test_api_with_memory_service.py)
  - Mock updated for proper method delegation
  - Real storage backends (SqliteVecMemoryStorage, HybridMemoryStorage) work correctly

### Technical Details
- **Files Modified**: 5 files
  - `src/mcp_memory_service/storage/base.py`: Added abstract `get_by_hash()` method
  - `src/mcp_memory_service/storage/cloudflare.py`: Implemented `get_by_hash()` (40 lines)
  - `src/mcp_memory_service/services/memory_service.py`: Optimized `get_memory_by_hash()`
  - `tests/unit/test_memory_service.py`: Updated mocks
  - `tests/integration/test_api_with_memory_service.py`: Updated mocks

- **Breaking Changes**: None - fully backward compatible
- **All Storage Backends Now Support get_by_hash()**:
  - ✅ SqliteVecMemoryStorage (line 1153)
  - ✅ CloudflareStorage (line 666)
  - ✅ HybridMemoryStorage (line 974, delegates to primary)

## [8.14.1] - 2025-10-31

### Added
- **Type Safety Improvements** - Comprehensive TypedDict definitions for all MemoryService return types
  - **Problem**: All MemoryService methods returned loose `Dict[str, Any]` types, providing no IDE support or compile-time validation
  - **Solution**: Created 14 specific TypedDict classes for structured return types
    - Store operations: `StoreMemorySingleSuccess`, `StoreMemoryChunkedSuccess`, `StoreMemoryFailure`
    - List operations: `ListMemoriesSuccess`, `ListMemoriesError`
    - Retrieve operations: `RetrieveMemoriesSuccess`, `RetrieveMemoriesError`
    - Search operations: `SearchByTagSuccess`, `SearchByTagError`
    - Delete operations: `DeleteMemorySuccess`, `DeleteMemoryFailure`
    - Health operations: `HealthCheckSuccess`, `HealthCheckFailure`
  - **Benefits**:
    - ✅ IDE autocomplete for all return values (type `result["` to see available keys)
    - ✅ Compile-time type checking catches typos (e.g., `result["succes"]` → type error)
    - ✅ Self-documenting API - clear contracts for all methods
    - ✅ Refactoring safety - renaming keys shows all affected code
    - ✅ 100% backward compatible - no runtime changes
    - ✅ Zero performance impact - pure static typing

### Fixed
- **Missing HybridMemoryStorage.get_by_hash()** - Added delegation method to HybridMemoryStorage
  - Fixed `AttributeError: 'HybridMemoryStorage' object has no attribute 'get_by_hash'`
  - All storage backends now implement `get_by_hash()`: SqliteVecMemoryStorage, CloudflareMemoryStorage, HybridMemoryStorage
  - Enables direct memory retrieval by content hash without loading all memories
  - See issue #196 for planned optimization to use this method in MemoryService

### Technical Details
- **Files Modified**:
  - `src/mcp_memory_service/services/memory_service.py`: Added 14 TypedDict classes, updated 6 method signatures
  - `src/mcp_memory_service/storage/hybrid.py`: Added `get_by_hash()` delegation method
- **Breaking Changes**: None - fully backward compatible (TypedDict is structural typing)
- **Testing**: All tests pass (15/15), comprehensive structure validation, HTTP API integration verified

## [8.14.0] - 2025-10-31

### Fixed
- **Comprehensive Tag Normalization** - DRY solution for all tag format handling
  - **Problem**: Inconsistent tag handling across different APIs caused validation errors
    - Top-level `tags` parameter accepted strings, but MemoryService expected arrays
    - `metadata.tags` field had no normalization, causing "is not of type 'array'" errors
    - Comma-separated strings like `"tag1,tag2,tag3"` were not split into arrays
    - Normalization logic was duplicated in some methods but missing in others
  - **Root Cause**:
    - MCP/HTTP adapters accepted `Union[str, List[str], None]` in signatures
    - But passed values to MemoryService without normalization
    - MemoryService expected `Optional[List[str]]`, causing type mismatch
    - `search_by_tag()` had normalization, but `store_memory()` did not (DRY violation)
  - **Solution** (DRY Principle Applied):
    - Created centralized `normalize_tags()` utility function (services/memory_service.py:27-56)
    - Handles ALL input formats:
      - `None` → `[]`
      - `"tag1,tag2,tag3"` → `["tag1", "tag2", "tag3"]`
      - `"single-tag"` → `["single-tag"]`
      - `["tag1", "tag2"]` → `["tag1", "tag2"]` (passthrough)
    - Updated `MemoryService.store_memory()` to:
      - Accept `Union[str, List[str], None]` type hint
      - Normalize both `tags` parameter and `metadata.tags` field
      - Merge tags from both sources with deduplication
    - Updated `MemoryService.search_by_tag()` to use utility (removed duplicate code)
  - **Files Modified**:
    - `src/mcp_memory_service/services/memory_service.py`: Added normalize_tags(), updated store_memory() and search_by_tag()
    - `src/mcp_memory_service/mcp_server.py`: Updated docstring to reflect all formats supported
  - **Benefits**:
    - ✅ Single source of truth for tag normalization (DRY)
    - ✅ All tag formats work everywhere (top-level, metadata, any protocol)
    - ✅ No more validation errors for comma-separated strings
    - ✅ Fully backward compatible
    - ✅ Consistent behavior across all endpoints
  - **User Impact**:
    - Can use any tag format anywhere without errors
    - No need to remember which parameter accepts which format
    - Improved developer experience and reduced friction

### Technical Details
- **Affected Components**: MemoryService (business logic layer), MCP server documentation
- **Breaking Changes**: None - fully backward compatible
- **Tag Merge Behavior**: When tags provided in both parameter and metadata, they are merged and deduplicated
- **Testing**: Verified with all format combinations (None, string, comma-separated, array, metadata.tags)

## [8.13.5] - 2025-10-31

### Fixed
- **Memory Hooks Display Polish** - Visual improvements for cleaner, more professional CLI output
  - **Issue**: Multiple visual inconsistencies in memory hooks tree structure display
  - **Problems Identified**:
    1. Redundant bottom frame (`╰────╯`) after tree naturally ended with `└─`
    2. Wrapped text continuation showing vertical lines (`│`) after last items
    3. Duplicate final summary message ("Context injected") when header already shows count
    4. Embedded formatting (✅, •, markdown) conflicting with tree structure
    5. Excessive content length despite adaptive truncation
  - **Fixes** (commits ed46d24, 998a39c):
    - **Content Sanitization**: Remove checkmarks, bullets, markdown formatting that conflicts with tree characters
    - **Smart Truncation**: Extract first 1-2 sentences for <400 char limits using sentence boundary detection
    - **Tree Continuation Logic**: Last items show clean indentation without vertical lines on wrapped text
    - **Remove Redundant Frame**: Tree ends naturally with `└─`, no separate closing box
    - **Remove Duplicate Message**: Header shows "X memories loaded", no redundant final summary
  - **Files Modified**:
    - `claude-hooks/utilities/context-formatter.js`: Content sanitization, smart truncation, tree continuation fixes
    - `claude-hooks/core/session-start.js`: Removed redundant success message
  - **Result**: Clean, consistent tree structure with proper visual hierarchy and no redundancy
  - **User Impact**: Professional CLI output, easier to scan, maintains continuous blue tree lines properly

### Technical Details
- **Affected Component**: Claude Code memory awareness hooks (SessionStart display)
- **Deployment**: Hooks loaded from repository automatically, no server restart needed
- **Testing**: Verified with mock execution and real Claude Code session

## [8.13.4] - 2025-10-30

### Fixed
- **Critical: Memory Hooks Showing Incorrect Ages** (#195) - Timestamp unit mismatch causing 20,371-day ages
  - **Error**: Memory hooks reporting `avgAge: 20371 days, medianAge: 20371 days` when actual age was 6 days
  - **User Impact**: Recent memories didn't surface, auto-calibration incorrectly triggered "stale memory" warnings
  - **Root Cause** (claude-hooks/utilities/memory-client.js:273): Timestamp unit mismatch
    - HTTP API returns Unix timestamps in **SECONDS**: `1758344479.729`
    - JavaScript `Date()` expects **MILLISECONDS**: Interpreted as `Jan 21, 1970` (55 years ago)
    - Age calculation: `(now - 1758344479ms) / 86400000 = 20371 days`
  - **Symptoms**:
    - `[Memory Age Analyzer] { avgAge: 20371, recentPercent: 0, isStale: true }`
    - Hooks showed "Stale memory set detected (median: 20371d old)"
    - Recent development work (< 7 days) never surfaced in context
  - **Fix** (claude-hooks/utilities/memory-client.js:273-294, commit 5c54894):
    - Convert API timestamps from seconds to milliseconds (`× 1000`)
    - Added year 2100 check (`< 4102444800`) to prevent double-conversion
    - Applied in `_performApiPost()` response handler for both `created_at` and `updated_at`
  - **Result**:
    - `avgAge: 6 days, medianAge: 5 days, recentPercent: 100%, isStale: false`
    - Recent memories surface correctly
    - Auto-calibration works properly
    - Git context weight adjusts based on actual ages
  - **Note**: Affects all users using HTTP protocol for memory hooks

### Technical Details
- **Affected Component**: Claude Code memory awareness hooks (HTTP protocol path)
- **File Changed**: `claude-hooks/utilities/memory-client.js` (lines 273-294)
- **Deployment**: Hooks loaded from repository automatically, no server restart needed
- **Issue**: https://github.com/doobidoo/mcp-memory-service/issues/195

## [8.13.3] - 2025-10-30

### Fixed
- **Critical: MCP Memory Tools Broken** - v8.12.0 regression preventing all MCP memory operations
  - **Error**: `KeyError: 'message'` when calling any MCP memory tool (store, retrieve, search, etc.)
  - **User Impact**: MCP tools completely non-functional - "Error storing memory: 'message'"
  - **Root Cause** (mcp_server.py:175): Return format mismatch between MemoryService and MCP tool expectations
    - MCP tool expects: `{success: bool, message: str, content_hash: str}`
    - MemoryService returns: `{success: bool, memory: {...}}`
    - MCP protocol tries to access missing 'message' field → KeyError
  - **Why It Persisted**: HTTP API doesn't require these specific fields, so integration tests passed
  - **Fix** (mcp_server.py:173-206): Transform MemoryService response to MCP TypedDict format
    - Capture result from MemoryService.store_memory()
    - Extract content_hash from nested memory object
    - Add descriptive "message" field
    - Handle 3 cases: failure (error message), chunked (multiple memories), single memory
  - **Result**: MCP tools now work correctly with proper error messages
  - **Note**: Requires MCP server restart (`/mcp` command in Claude Code) to load fix

### Technical Details
- **Introduced**: v8.12.0 MemoryService architecture refactoring (#176)
- **Affected Tools**: store_memory, all MCP protocol operations
- **HTTP API**: Unaffected (different response format requirements)
- **Test Gap**: No integration tests validating MCP tool response formats

## [8.13.2] - 2025-10-30

### Fixed
- **Memory Sync Script Broken** (#193): Fixed sync_memory_backends.py calling non-existent `store_memory()` method
  - **Error**: `AttributeError: 'CloudflareStorage' object has no attribute 'store_memory'`
  - **User Impact**: Sync script completely non-functional - couldn't sync memories between Cloudflare and SQLite backends
  - **Root Cause** (scripts/sync/sync_memory_backends.py:144-147): Script used old `store_memory()` API that no longer exists
  - **Fix** (#194, b69de83): Updated to use proper `Memory` object creation and `storage.store()` method
    - Create `Memory` object with `content`, `content_hash`, `tags`, `metadata`, `created_at`
    - Call `await target_storage.store(memory_obj)` instead of non-existent `store_memory()`
    - Added safe `.get('metadata', {})` to prevent KeyError on missing metadata
    - Fixed import order to comply with PEP 8 (config → models → storage)
  - **Result**: Sync script now successfully syncs memories between backends
  - **Credit**: Fix by AMP via PR #194, reviewed by Gemini

## [8.13.1] - 2025-10-30

### Fixed
- **Critical Concurrent Access Bug**: MCP server initialization failed with "database is locked" when HTTP server running
  - **Error**: `sqlite3.OperationalError: database is locked` during MCP tool initialization
  - **User Impact**: MCP memory tools completely non-functional while HTTP server running - "this worked before without any flaws"
  - **Root Cause #1** (sqlite_vec.py:329): Connection timeout set AFTER opening database instead of during connection
    - Original: `sqlite3.connect(path)` used default 5-second timeout, then applied `PRAGMA busy_timeout=15000`
    - SQLite only respects timeout parameter passed to `connect()`, not pragma applied afterward
    - MCP server timed out before it could set the higher timeout value
  - **Root Cause #2** (sqlite_vec.py:467-476): Both servers attempting DDL operations (CREATE TABLE, CREATE INDEX) simultaneously
    - Even with WAL mode, DDL operations require brief exclusive locks
    - No detection of already-initialized database before running DDL
  - **Fix #1** (sqlite_vec.py:291-326): Parse `busy_timeout` from `MCP_MEMORY_SQLITE_PRAGMAS` environment variable BEFORE opening connection
    - Convert from milliseconds to seconds (15000ms → 15.0s)
    - Pass timeout to `sqlite3.connect(path, timeout=15.0)` for immediate effect
    - Allows MCP server to wait up to 15 seconds for HTTP server's DDL operations
  - **Fix #2** (sqlite_vec.py:355-373): Detect already-initialized database and skip DDL operations
    - Check if `memories` and `memory_embeddings` tables exist after loading sqlite-vec extension
    - If tables exist, just load embedding model and mark as initialized
    - Prevents "database is locked" errors from concurrent CREATE TABLE/INDEX attempts
  - **Result**: MCP and HTTP servers now coexist without conflicts, maintaining pre-v8.9.0 concurrent access behavior

### Technical Details
- **Timeline**: Bug discovered during memory consolidation testing, fixed same day
- **Affected Versions**: v8.9.0 introduced database lock prevention pragmas but didn't fix concurrent initialization
- **Test Validation**: MCP health check returns healthy with 1857 memories while HTTP server running
- **Log Evidence**: "Database already initialized by another process, skipping DDL operations" confirms fix working

## [8.13.0] - 2025-10-29

### Added
- **HTTP Server Integration Tests** (#190): Comprehensive test suite with 32 tests prevents production bugs like v8.12.0
  - `tests/integration/test_http_server_startup.py`: 8 tests for server startup validation
  - `tests/unit/test_fastapi_dependencies.py`: 11 tests for dependency injection
  - `tests/unit/test_storage_interface_compatibility.py`: 13 tests for backend interface consistency
  - Extended `tests/integration/test_api_with_memory_service.py`: +11 HTTP API tests with TestClient
  - Tests would have caught all 3 v8.12.0 production bugs (import-time evaluation, syntax errors, interface mismatches)

- **Storage Method: get_largest_memories()** (#186): Efficient database queries for largest memories by content length
  - Added to all storage backends (SQLite, Cloudflare, Hybrid)
  - Uses `ORDER BY LENGTH(content) DESC LIMIT n` instead of loading 1000 memories and sorting in Python
  - Analytics dashboard now queries entire dataset for truly largest memories

### Fixed
- **Analytics Dashboard Timezone Bug** (#186): Fixed heatmap calendar showing wrong day-of-week near timezone boundaries
  - JavaScript `new Date('YYYY-MM-DD')` parsed as UTC midnight, but `getDay()` used local timezone
  - Changed to parse date components in local timezone: `new Date(year, month-1, day)`
  - Prevents calendar cells from shifting to previous/next day for users in UTC-12 to UTC+12 timezones

### Improved
- **Analytics Performance**: Reduced memory sample for average size calculation from 1000→100 memories
- **Test Coverage**: Zero HTTP integration tests → 32 comprehensive tests covering server startup, dependencies, and API endpoints

### Documentation
- **MCP Schema Caching** (#173): Closed with comprehensive documentation in CLAUDE.md and troubleshooting guides
  - Root cause: MCP protocol caches tool schemas client-side
  - Workaround: `/mcp` command reconnects server with fresh schema
  - Documented symptoms, diagnosis, and resolution steps

## [8.12.1] - 2025-10-28

### Fixed
- **Critical Production Bug #1** (ef2c64d): Import-time default parameter evaluation in `get_memory_service()`
  - **Error**: `HTTPException: 503: Storage not initialized` during module import
  - **Root Cause**: Python evaluates default parameters at function definition time, not call time
  - **Impact**: HTTP server couldn't start - module import failed immediately
  - **Fix**: Changed from `storage: MemoryStorage = get_storage()` to `storage: MemoryStorage = Depends(get_storage)`
  - **Technical**: FastAPI's `Depends()` defers evaluation until request time and integrates with dependency injection

- **Critical Production Bug #2** (77de4d2): Syntax error + missing FastAPI Depends import in `memories.py`
  - **Error**: `SyntaxError: expected an indented block after 'if' statement on line 152`
  - **Root Cause**: `if INCLUDE_HOSTNAME:` had no indented body, nested if-elif-else block not indented
  - **Impact**: SyntaxError prevented module import + FastAPI validation failure
  - **Fix**: Properly indented hostname resolution logic, added missing `Depends` import to dependencies.py

- **Critical Production Bug #3** (f935c56): Missing `tags` parameter in `count_all_memories()` across all storage backends
  - **Error**: `TypeError: count_all_memories() got an unexpected keyword argument 'tags'`
  - **User Report**: "failed to load dashboard data"
  - **Root Cause**: MemoryService called `count_all_memories(memory_type=type, tags=tags)` but base class and implementations didn't accept tags parameter
  - **Impact**: Dashboard completely broken - GET /api/memories returned 500 errors
  - **Fix**: Updated 4 files (base.py, hybrid.py, sqlite_vec.py, cloudflare.py) to add tags parameter with SQL LIKE filtering
  - **Why Tests Missed It**: AsyncMock accepts ANY parameters, never tested real storage backend implementations

- **Analytics Metrics Bug** (8beeb07): Analytics tab showed different metrics than Dashboard tab
  - **Problem**: Dashboard queried ALL memories, Analytics sampled only 1000 recent memories
  - **Impact**: "This Week" count was inaccurate when total memories > 1000
  - **Fix**: Changed Analytics endpoint to use `storage.get_stats()` instead of sampling
  - **Performance**: Eliminated O(n) memory loading for simple count operation, now uses efficient SQL COUNT

### Changed
- **Analytics Endpoint Performance** - Increased monthly sample from 2,000 to 5,000 memories
- **Code Quality** - Added TODO comments for moving monthly calculations to storage layer

### Technical Details
- **Timeline**: All 4 bugs discovered and fixed within 4 hours of v8.12.0 release (15:03 UTC → 22:03 UTC)
- **Post-Mortem**: Created Issue #190 for HTTP server integration tests to prevent future production bugs
- **Test Coverage Gap**: v8.12.0 had 55 tests but zero HTTP server integration tests
- **Lesson Learned**: Tests used mocked storage that never actually started the server or tested real FastAPI dependency injection

**Note**: This patch release resolves all production issues from v8.12.0 architectural changes. Comprehensive analysis stored in memory with tag `v8.12.0,post-release-bugs`.

## [8.12.0] - 2025-10-28

### Added
- **MemoryService Architecture** - Centralized business logic layer (Issue #188, PR #189)
  - Single source of truth for all memory operations
  - Consistent behavior across API endpoints and MCP tools
  - 80% code duplication reduction between API and MCP servers
  - Dependency injection pattern for clean architecture
  - **Comprehensive Test Coverage**:
    - 34 unit tests (100% pass rate)
    - 21 integration tests for API layer
    - End-to-end workflow tests with real storage
    - Performance validation for database-level filtering

### Fixed
- **Critical Bug #1**: `split_content()` missing required `max_length` parameter
  - Impact: Would crash immediately on any content chunking operation
  - Fix: Added proper parameter passing with storage backend max_length
- **Critical Bug #2**: `storage.delete_memory()` method does not exist in base class
  - Impact: Delete functionality completely broken
  - Fix: Changed to use `storage.delete(content_hash)` from base class
- **Critical Bug #3**: `storage.get_memory()` method does not exist in base class
  - Impact: Get by hash functionality completely broken
  - Fix: Implemented using `get_all_memories()` with client-side filtering
- **Critical Bug #4**: `storage.health_check()` method does not exist in base class
  - Impact: Health check functionality completely broken
  - Fix: Changed to use `storage.get_stats()` from base class
- **Critical Bug #5**: `storage.search_by_tags()` method mismatch (plural vs singular)
  - Impact: Tag search functionality completely broken
  - Fix: Changed to use `storage.search_by_tag()` (singular) from base class
- **Critical Bug #6**: Incorrect chunking logic comparing `len(content) > CONTENT_PRESERVE_BOUNDARIES`
  - Impact: ALL content >1 character would trigger chunking (CONTENT_PRESERVE_BOUNDARIES is boolean `True`)
  - Fix: Proper comparison using `storage.max_content_length` numeric value
- **Critical Bug #7**: Missing `store()` return value handling
  - Impact: Success/failure not properly tracked
  - Fix: Proper unpacking of `(success, message)` tuple from storage operations

### Changed
- **API Endpoints** - Refactored to use MemoryService dependency injection
  - `/api/memories` (list, create) uses MemoryService
  - `/api/search` endpoints use MemoryService
  - Consistent response formatting via service layer
- **Code Maintainability** - Removed 356 lines of duplicated code
  - Single place to modify business logic
  - Unified error handling
  - Consistent hostname tagging logic
- **Performance** - Database-level filtering prevents O(n) memory loading
  - Scalable pagination with offset/limit at storage layer
  - Efficient tag and type filtering

### Technical Details
- **Files Modified**: 6 files, 1469 additions, 356 deletions
- **Test Coverage**: 55 new tests (34 unit + 21 integration)
- **Bug Discovery**: Comprehensive testing revealed 7 critical bugs that would have made the release non-functional
- **Quality Process**: Test-driven debugging approach prevented broken release

**Note**: This release demonstrates the critical importance of comprehensive testing before merging architectural changes. All 7 bugs were caught through systematic unit and integration testing.

## [8.11.0] - 2025-10-28

### Added
- **JSON Document Loader** - Complete implementation of JSON file ingestion (Issue #181, PR #187)
  - **Nested Structure Flattening**: Converts nested JSON to searchable text with dot notation or bracket notation
  - **Configurable Strategies**: Choose flattening style, max depth, type inclusion
  - **Array Handling**: Multiple modes (expand, summarize, flatten) for different use cases
  - **Comprehensive Tests**: 15 unit tests covering all functionality
  - **Use Cases**: Knowledge base exports, API documentation, config files, structured metadata

- **CSV Document Loader** - Complete implementation of CSV file ingestion (Issue #181, PR #187)
  - **Auto-Detection**: Automatically detects delimiters (comma, semicolon, tab, pipe) and headers
  - **Row-Based Formatting**: Converts tabular data to searchable text with column context
  - **Encoding Support**: Auto-detects UTF-8, UTF-16, UTF-32, Latin-1, CP1252
  - **Large File Handling**: Efficient row-based chunking for scalability
  - **Comprehensive Tests**: 14 unit tests covering all functionality
  - **Use Cases**: Data dictionaries, reference tables, tabular documentation, log analysis

### Fixed
- **False Advertising** - Resolved issue where JSON and CSV were listed in `SUPPORTED_FORMATS` but had no loader implementations
  - Previous behavior: Upload would fail with "No loader available" error
  - New behavior: Full functional support with proper chunking and metadata

### Changed
- **Ingestion Module** - Updated to register new JSON and CSV loaders
- **Test Coverage** - Added 29 new unit tests (15 JSON + 14 CSV)

## [8.10.0] - 2025-10-28

### Added
- **Complete Analytics Dashboard Implementation** (Issue #182, PR #183)
  - Memory Types Breakdown (pie chart)
  - Activity Heatmap (GitHub-style calendar with 90d/6mo/1yr periods)
  - Top Tags Report (usage trends, co-occurrence patterns)
  - Recent Activity Report (hourly/daily/weekly breakdowns)
  - Storage Report (largest memories, efficiency metrics)
  - Streak Tracking (current and longest consecutive days)

### Fixed
- **Activity Streak Calculation** - Fixed current streak to include today check
- **Total Days Calculation** - Corrected date span vs active days count
- **Longest Streak Initialization** - Fixed from 0 to 1

### Changed
- **Analytics API** - Added 5 new endpoints with Pydantic models
- **Dashboard Documentation** - Updated wiki with complete analytics features

## [8.9.0] - 2025-10-27

### Fixed
- **Database Lock Prevention** - Resolved "database is locked" errors during concurrent HTTP + MCP server access (Issue discovered during performance troubleshooting)
  - **Root Cause**: Default `busy_timeout=5000ms` too short for concurrent writes from multiple MCP clients
  - **Solution**: Applied recommended SQLite pragmas (`busy_timeout=15000,cache_size=20000`)
  - **WAL Mode**: Already enabled by default, now properly configured for multi-client access
  - **Impact**: Zero database locks during testing with 5 concurrent write operations
  - **Documentation**: Updated multi-client architecture docs with pragma recommendations

### Added
- **Hybrid Backend Installer Support** - Full hybrid backend support in simplified installer (`scripts/installation/install.py`)
  - **Interactive Selection**: Hybrid now option 4 (recommended default) in installer menu
  - **Automatic Configuration**: SQLite pragmas set automatically for sqlite_vec and hybrid backends
  - **Cloudflare Setup**: Interactive credential configuration with connection testing
  - **Graceful Fallback**: Falls back to sqlite_vec if Cloudflare setup cancelled or fails
  - **Claude Desktop Integration**: Hybrid backend configuration includes:
    - SQLite pragmas for concurrent access (`MCP_MEMORY_SQLITE_PRAGMAS`)
    - Cloudflare credentials for background sync
    - Proper environment variable propagation
  - **Benefits**:
    - 5ms local reads (SQLite-vec)
    - Zero user-facing latency (background Cloudflare sync)
    - Multi-device synchronization
    - Concurrent access support

### Changed
- **Installer Defaults** - Hybrid backend now recommended for production use
  - Updated argparse choices to include `hybrid` option
  - Changed default selection from sqlite_vec to hybrid (option 4)
  - Enhanced compatibility detection with "recommended" status for hybrid
  - Improved final installation messages with backend-specific guidance
- **Environment Management** - Cloudflare credentials now set in current environment immediately
  - `save_credentials_to_env()` sets both .env file AND os.environ
  - Ensures credentials available for Claude Desktop config generation
  - Proper variable propagation for hybrid and cloudflare backends
- **Path Configuration** - Updated `configure_paths()` to handle all backends
  - SQLite database paths for: `sqlite_vec`, `hybrid`, `cloudflare`
  - Cloudflare credentials included when backend requires them
  - Backward compatible with existing installations

### Technical Details
- **Files Modified**:
  - `scripts/installation/install.py`: Lines 655-659 (compatibility), 758 (menu), 784-802 (selection), 970-1017 (hybrid install), 1123-1133 (env config), 1304 (path config), 1381-1401 (Claude Desktop config), 1808-1821 (final messages)
  - `src/mcp_memory_service/__init__.py`: Line 50 (version bump)
  - `pyproject.toml`: Line 7 (version bump)
- **Concurrent Access Testing**: 5/5 simultaneous writes succeeded without locks
- **HTTP Server Logs**: Confirmed background Cloudflare sync working (line 369: "Successfully stored memory")

## [8.8.2] - 2025-10-26

### Fixed
- **Document Upload Tag Validation** - Prevents bloated tags from space-separated file paths (Issue #174, PR #179)
  - **Enhanced Tag Parsing**: Split tags by comma OR space instead of comma only
  - **Robust file:// URI Handling**: Uses `urllib.parse` for proper URL decoding and path handling
    - Handles URL-encoded characters (e.g., `%20` for spaces)
    - Handles different path formats (e.g., `file:///C:/...`)
    - Properly handles Windows paths with leading slash from urlparse
  - **File Path Sanitization**: Remove `file://` prefixes, extract filenames only, clean path separators
  - **Explicit Tag Length Validation**: Tags exceeding 100 chars now raise explicit HTTPException instead of being silently dropped

### Added
- **Processing Mode Toggle** - UI enhancement for multiple file uploads (PR #179)
  - **Batch Processing**: All files processed together (faster, default)
  - **Individual Processing**: Each file processed separately with better error isolation
  - Toggle only appears when multiple files are selected
  - Comprehensive help section explaining both modes with pros/cons

### Changed
- **Code Quality Improvements** - Eliminated code duplication in document upload endpoints (PR #179)
  - Extracted `parse_and_validate_tags()` helper function to eliminate duplicate tag parsing logic
  - Removed 44 lines of duplicate code from `upload_document` and `batch_upload_documents`
  - Extracted magic number (500ms upload delay) to static constant `INDIVIDUAL_UPLOAD_DELAY`
  - Simplified toggle display logic with ternary operator
  - Created Issue #180 for remaining medium-priority code quality suggestions

## [8.8.1] - 2025-10-26

### Fixed
- **Error Handling Improvements** - Enhanced robustness in MemoryService and maintenance scripts (Issue #177)
  - **MemoryService.store_memory()**: Added specific exception handling for better error classification
    - `ValueError` → validation errors with "Validation error" messages
    - `httpx.NetworkError/TimeoutException/HTTPStatusError` → storage errors with "Storage error" messages
    - Generic `Exception` → unexpected errors with full logging and "unexpected error" messages
  - **Maintenance Scripts**: Added proper error handling to prevent crashes
    - `find_cloudflare_duplicates.py`: Wrapped `get_all_memories_bulk()` in try/except, graceful handling of empty results
    - `delete_orphaned_vectors_fixed.py`: Already used public API (no changes needed)

### Added
- **Encapsulation Methods** - New public APIs for Cloudflare storage operations (Issue #177)
  - `CloudflareStorage.delete_vectors_by_ids()` - Batch vector deletion with proper error handling
  - `CloudflareStorage.get_all_memories_bulk()` - Efficient bulk loading without N+1 tag queries
  - `CloudflareStorage._row_to_memory()` - Helper for converting D1 rows to Memory objects
  - **Performance**: Bulk operations avoid expensive individual tag lookups
  - **Maintainability**: Public APIs instead of direct access to private `_retry_request` method

### Changed
- **Dependency Management** - Added conditional typing-extensions for Python 3.10 (Issue #177)
  - Added `"typing-extensions>=4.0.0; python_version < '3.11'"` to pyproject.toml
  - Ensures `NotRequired` import works correctly on Python 3.10
  - No impact on Python 3.11+ installations

### Review
- **Gemini Code Assist**: "This pull request significantly improves the codebase by enhancing error handling and improving encapsulation... well-executed and contribute to better maintainability"
- **Feedback Addressed**: All review suggestions implemented, including enhanced exception handling

## [8.8.0] - 2025-10-26

### Changed
- **DRY Refactoring** - Eliminated code duplication between MCP and HTTP servers (PR #176, Issue #172)
  - **Problem**: MCP (`mcp_server.py`) and HTTP (`server.py`) servers had 364 lines of duplicated business logic
    - Bug fixes applied to one server were missed in the other (e.g., PR #162 tags validation)
    - Maintenance burden of keeping two implementations synchronized
    - Risk of behavioral inconsistencies between protocols
  - **Solution**:
    - Created `MemoryService` class (442 lines) as single source of truth for business logic
    - Refactored `mcp_server.py` to thin adapter (-338 lines, now ~50 lines per method)
    - Refactored `server.py` to use MemoryService (169 lines modified)
    - Both servers now delegate to shared business logic
  - **Benefits**:
    - **Single source of truth**: All memory operations (store, retrieve, search, delete) in one place
    - **Consistent behavior**: Both protocols guaranteed identical business logic
    - **Easier maintenance**: Bug fixes automatically apply to both servers
    - **Better testability**: Business logic isolated and independently testable
    - **Prevents future bugs**: Impossible to fix one server and forget the other
  - **Type Safety**: Added TypedDict classes (`MemoryResult`, `OperationResult`, `HealthStats`) for better type annotations
  - **Backward Compatibility**: No API changes, both servers remain fully compatible
  - **Testing**: All tests passing (15/15 Cloudflare storage tests)
  - **Review**: Gemini Code Assist: "significant and valuable refactoring... greatly improves maintainability and consistency"
  - **Follow-up**: Minor improvements tracked in Issue #177 (error handling, encapsulation)

### Fixed
- **Python 3.10 Compatibility** - Added `NotRequired` import fallback (src/mcp_memory_service/mcp_server.py:23-26)
  - Uses `typing.NotRequired` on Python 3.11+
  - Falls back to `typing_extensions.NotRequired` on Python 3.10
  - Ensures compatibility across Python versions

### Added
- **Maintenance Scripts** - Cloudflare cleanup utilities (from v8.7.1 work)
  - `scripts/maintenance/find_cloudflare_duplicates.py` - Detect duplicates in Cloudflare D1
  - `scripts/maintenance/delete_orphaned_vectors_fixed.py` - Clean orphaned Vectorize vectors
  - `scripts/maintenance/fast_cleanup_duplicates_with_tracking.sh` - Platform-aware SQLite cleanup
  - `scripts/maintenance/find_all_duplicates.py` - Platform detection (macOS/Linux paths)

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

## Historic Releases

For older releases (v7.21.0 and earlier), see [CHANGELOG-HISTORIC.md](./CHANGELOG-HISTORIC.md).

**Historic Version Range**: v0.1.0 through v7.21.0 (2025-07-XX through 2025-10-03)
