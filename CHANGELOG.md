# Changelog

All notable changes to the MCP Memory Service project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [6.15.1] - 2025-09-22

### 🔧 **Enhanced Cloudflare Backend Initialization & Diagnostics**

#### Cloudflare Backend Issue Resolution
- **Fixed silent fallback issue**: Resolved Cloudflare backend falling back to SQLite-vec during MCP startup
  - **Root cause identified**: Silent failures in eager initialization phase were not properly logged
  - **Enhanced error reporting**: Added comprehensive logging with visual indicators for all initialization phases
  - **Improved timeout handling**: Better error messages when Cloudflare initialization times out
- **Enhanced initialization logging**: Added detailed logging for both eager and lazy initialization paths
  - **Phase indicators**: 🚀 SERVER INIT, ☁️ Cloudflare-specific, ✅ Success markers, ❌ Error indicators
  - **Configuration validation**: Logs all Cloudflare environment variables for troubleshooting
  - **Storage type verification**: Confirms final storage backend type after initialization
- **Diagnostic improvements**: Created comprehensive diagnostic tools for Cloudflare backend issues
  - **Enhanced diagnostic script**: `debug_server_initialization.py` for testing initialization flows
  - **MCP environment testing**: `tests/integration/test_mcp_environment.py` for testing Claude Desktop integration
  - **Fixed test syntax errors**: Corrected f-string and async function issues in test files

#### Technical Improvements
- **Fixed `db_utils.py` async issue**: Changed `get_database_stats()` to async function to support Cloudflare storage
- **Enhanced error tracebacks**: Full exception details now logged for initialization failures
- **Better fallback documentation**: Clear distinction between eager vs lazy initialization behaviors

#### Troubleshooting Benefits
- **Faster issue diagnosis**: Clear visual indicators help identify where initialization fails
- **Better error messages**: Specific error details for common Cloudflare setup issues
- **Proactive debugging**: Enhanced logging appears in Claude Desktop MCP logs for easier troubleshooting

## [6.15.0] - 2025-09-22

### 🗂️ **Scripts Directory Reorganization & Professional Tooling**

#### Major Reorganization
- **Complete Scripts Restructuring**:
  - ✅ **Organized 62 loose scripts** into 12 logical categories for professional navigation
  - ✅ **Created systematic directory structure** with clear functional grouping
  - ✅ **Zero loose scripts remaining** in root directory - all properly categorized
  - ✅ **Maintained full backward compatibility** - all scripts work exactly as before
  - ✅ **Updated all documentation references** to reflect new paths

#### New Directory Structure
- **🔄 `sync/`** (4 scripts) - Backend synchronization utilities
- **🛠️ `service/`** (5 scripts) - Service management and deployment
- **✅ `validation/`** (7 scripts) - Configuration and system validation
- **🗄️ `database/`** (4 scripts) - Database analysis and health monitoring
- **🧹 `maintenance/`** (7 scripts) - Database cleanup and repair operations
- **💾 `backup/`** (4 scripts) - Backup and restore operations
- **🔄 `migration/`** (11 scripts) - Data migration and schema updates
- **🏠 `installation/`** (8 scripts) - Setup and installation scripts
- **🖥️ `server/`** (5 scripts) - Server runtime and operational scripts
- **🧪 `testing/`** (15 scripts) - Test scripts and validation
- **🔧 `utils/`** (7 scripts) - General utility scripts and wrappers
- **🛠️ `development/`** (6 scripts) - Development tools and debugging utilities

#### Enhanced Documentation
- **✅ Complete README.md rewrite** with comprehensive script index and usage examples
- **✅ Quick reference guide** for essential daily operations
- **✅ Detailed directory explanations** with purpose and key features
- **✅ Safety guidelines** and execution best practices
- **✅ Common use case workflows** for setup, operations, troubleshooting, and migration
- **✅ Integration documentation** linking to project wiki and guides

#### User Experience Improvements
- **🎯 Faster script discovery** - find tools by logical function instead of hunting through 62 files
- **📚 Professional documentation** with tables, examples, and clear categorization
- **🚀 Quick-start examples** for common operations and troubleshooting
- **🛡️ Safety-first approach** with dry-run recommendations and backup guidelines
- **🔗 Seamless integration** with existing CLAUDE.md, AGENTS.md, and documentation

#### Maintainability Enhancements
- **🏗️ Logical organization** makes adding new scripts intuitive
- **📝 Clear naming conventions** and directory purposes
- **🔄 Future-proof structure** that scales with project growth
- **✅ Consistent documentation patterns** across all categories
- **🧪 Verified functionality** - all critical scripts tested post-reorganization

#### Files Updated (4):
- **`scripts/README.md`** - COMPLETE REWRITE: Professional documentation with comprehensive index
- **`CLAUDE.md`** - UPDATED: All script paths updated to new locations
- **`AGENTS.md`** - UPDATED: Development workflow script references
- **`CHANGELOG.md`** - UPDATED: Historical script references to new paths

#### Impact
- 🎯 **Transforms user experience** from cluttered file hunting to professional navigation
- 🚀 **Enables faster development** with logical script organization
- 💻 **Simplifies maintenance** with clear categorization and documentation
- ✅ **Professional appearance** suitable for enterprise deployments
- 🔄 **Supports scalable growth** with extensible directory structure
- 🛡️ **Improves safety** with comprehensive usage guidelines and best practices

This release transforms the scripts directory from a disorganized collection into a professional, enterprise-ready toolkit that significantly improves developer and user experience while maintaining full functionality.

## [6.14.0] - 2025-09-22

### 🛠️ **Operational Utilities & Backend Synchronization**

#### New Backend Synchronization Capabilities
- **Bidirectional Sync Engine**:
  - ✅ `sync_memory_backends.py` - Core sync logic with intelligent deduplication
  - ✅ `claude_sync_commands.py` - User-friendly CLI wrapper for sync operations
  - ✅ Supports Cloudflare ↔ SQLite-vec synchronization with dry-run mode
  - ✅ Content-based hashing prevents duplicates across backends
  - ✅ Comprehensive status reporting and error handling
- **Service Management**:
  - ✅ `memory_service_manager.sh` - Linux service management for dual-backend deployments
  - ✅ State-based backend detection using `/tmp/memory-service-backend.state`
  - ✅ Environment file management for Cloudflare and SQLite configurations
  - ✅ Integrated sync operations and health monitoring
- **Configuration Validation**:
  - ✅ `validate_config.py` - Comprehensive configuration validator
  - ✅ Validates Claude Code global configuration (~/.claude.json)
  - ✅ Checks Cloudflare credentials and environment setup
  - ✅ Detects configuration conflicts and provides solutions
- **Documentation & Guides**:
  - ✅ Updated `scripts/README.md` with comprehensive utility documentation
  - ✅ Added backend sync references to main `README.md` and `CLAUDE.md`
  - ✅ Created `Backend-Synchronization-Guide.md` wiki page with complete setup guide

#### Code Quality Improvements (Gemini Code Assist Feedback)
- **Eliminated Code Duplication**:
  - ✅ Refactored `sync_memory_backends.py` to use shared `_sync_between_backends()` method
  - ✅ Reduced ~80 lines of duplicate code, improved maintainability
- **Cross-Platform Compatibility**:
  - ✅ Replaced hardcoded `/home/hkr/` paths with `$HOME` variable
  - ✅ Added missing final newlines to all utility scripts
- **Enhanced Validation & Robustness**:
  - ✅ Improved configuration validation with regex patterns instead of string matching
  - ✅ Added UTF-8 encoding to all file operations in `validate_config.py`
  - ✅ Fixed `.env.sqlite` conflict detection logic
- **Better Code Organization**:
  - ✅ Refactored command handling to dictionary-based dispatch pattern
  - ✅ Improved scalability for future command additions

#### Production Deployment Features
- **Hybrid Cloud/Local Deployments**: Enable Cloudflare primary with SQLite backup
- **Disaster Recovery**: Automated backup and restore capabilities
- **Multi-Machine Sync**: Consistent memory sharing across devices
- **Development/Production Workflows**: Seamless sync between environments
- **Troubleshooting Tools**: Configuration validation and service management

#### Impact
- 🎯 **Fills critical operational gaps** for production deployments
- 🚀 **Enables advanced deployment strategies** (hybrid cloud/local)
- 💻 **Simplifies troubleshooting** with validation and management tools
- ✅ **Professional code quality** meeting all review standards
- 🔄 **Supports complex workflows** for distributed teams

#### Files Added/Modified (8):
- `scripts/sync/sync_memory_backends.py` - NEW: Bidirectional sync engine
- `scripts/sync/claude_sync_commands.py` - NEW: CLI wrapper for sync operations
- `scripts/service/memory_service_manager.sh` - NEW: Linux service manager
- `scripts/validation/validate_config.py` - NEW: Configuration validator
- `scripts/README.md` - UPDATED: Comprehensive utility documentation
- `README.md` - UPDATED: Added troubleshooting references
- `CLAUDE.md` - UPDATED: Added sync and validation commands
- Wiki: `Backend-Synchronization-Guide.md` - NEW: Complete sync setup guide

## [6.13.8] - 2025-09-21

### 🔧 **Integration Completion**

#### Cloudflare Storage Backend Full Integration
- **Fixed critical integration gaps**: Cloudflare backend now has complete feature parity with sqlite_vec and chroma
- **Health Check Recognition**:
  - ✅ **Resolved "Unknown storage type: CloudflareStorage" errors**
  - ✅ Health check now properly validates Cloudflare storage and returns "healthy" status
  - ✅ Added Cloudflare support to `server.py` health validation (lines 3410-3450)
  - ✅ Added Cloudflare support to `db_utils.py` validation, stats, and repair functions
- **Complete CLI Integration**:
  - ✅ Added 'cloudflare' option to all CLI storage backend choices
  - ✅ Updated `cli/main.py`, `cli/ingestion.py` with cloudflare backend support
  - ✅ Added CloudflareStorage initialization logic to `cli/utils.py`
  - ✅ Updated configuration documentation to include cloudflare option
- **Documentation & Startup Guidance**:
  - ✅ **Critical**: Added warning about inappropriate use of `memory_offline.py` with Cloudflare backend
  - ✅ Cloudflare uses Workers AI for embeddings - offline mode breaks this functionality
  - ✅ Added proper startup script recommendations in `docs/cloudflare-setup.md`
  - ✅ Recommended: `uv run memory server`, `python scripts/run_memory_server.py`
- **Enhanced Test Coverage**:
  - ✅ Added comprehensive tests for tag sanitization method
  - ✅ Verified existing extensive test suite covers all major functionality
- **Impact**:
  - 🎯 **Cloudflare backend is now a first-class citizen** alongside sqlite_vec and chroma
  - 🚀 **Complete health check integration** - no more "Unknown storage type" errors
  - 💻 **Full CLI support** for cloudflare backend selection
  - 📚 **Clear startup guidance** prevents Workers AI compatibility issues
  - ✅ **Production ready** for distributed teams and cloud-native deployments

#### Files Modified (11):
- `src/mcp_memory_service/server.py` - Added Cloudflare health check case
- `src/mcp_memory_service/utils/db_utils.py` - Added validation, stats, repair support
- `src/mcp_memory_service/cli/main.py` - Added cloudflare CLI option
- `src/mcp_memory_service/cli/ingestion.py` - Added cloudflare CLI option
- `src/mcp_memory_service/cli/utils.py` - Added CloudflareStorage initialization
- `src/mcp_memory_service/config.py` - Updated backend documentation
- `docs/cloudflare-setup.md` - Added startup script guidance and warnings
- `tests/unit/test_cloudflare_storage.py` - Enhanced test coverage
- `src/mcp_memory_service/__init__.py` - Version bump to 6.13.8
- `pyproject.toml` - Version bump to 6.13.8

## [6.13.7] - 2025-09-20

### 🐛 **Critical Bug Fixes**

#### Cloudflare Vectorize ID Length Issue Fixed
- **Fixed critical bug**: Resolved Cloudflare Vectorize vector ID length limit error
  - **Root Cause**: CloudflareStorage was using `"mem_" + content_hash` format for vector IDs (68 characters)
  - **Cloudflare Limit**: Vectorize has a 64-byte maximum ID length restriction
  - **Solution**: Removed "mem_" prefix, now using `content_hash` directly (64 characters)
  - **Impact**: **Enables proper bidirectional sync** between multiple machines using Cloudflare backend
  - **Error Fixed**: `"id too long; max is 64 bytes, got 68 bytes"` when storing memories
- **Affects**: All users using Cloudflare backend for memory storage
- **Backward Compatibility**: ⚠️ **Breaking change** - existing Cloudflare vector IDs will have different format
- **Migration**: Users with existing Cloudflare deployments may need to re-import memories
- **Benefits**:
  - ✅ Cloudflare backend now works reliably for memory storage
  - ✅ Multi-machine sync scenarios (e.g., replacing failed servers) now supported
  - ✅ Consistent vector ID format aligns with Cloudflare specifications
- **Files Modified**: `src/mcp_memory_service/storage/cloudflare.py:295`

#### Multi-Machine Sync Capability
- **New capability**: Cloudflare can now serve as central memory hub for multiple machines
- **Use case**: Replacement for failed centralized servers (e.g., narrowbox failures)
- **Implementation**: Dual storage strategy with Cloudflare primary + local sqlite_vec backup
- **Tested**: Bidirectional sync verified between macOS machines using Cloudflare D1 + Vectorize

## [6.13.6] - 2025-09-16

### 📚 **Documentation**

#### Added AGENTS.md for AI Coding Agents
- **New standard format**: Added `AGENTS.md` following the industry-standard [agents.md](https://agents.md/) specification
  - **Purpose**: Provides AI coding agents with project-specific instructions and context
  - **Compatibility**: Works with GitHub Copilot, Cursor, VS Code, Continue, and other AI tools
  - **Complements CLAUDE.md**: Generic instructions for all AI agents vs Claude-specific in CLAUDE.md
- **Content includes**:
  - Setup commands and testing procedures
  - Project structure and key files overview
  - Code style guidelines and conventions
  - Common development tasks and patterns
  - Security guidelines and debugging tips
- **Benefits**:
  - Standardized location for AI agent instructions (becoming industry standard)
  - Improved developer experience when using AI coding assistants
  - Better onboarding for contributors using AI tools
- **Files Added**: `AGENTS.md`

#### Added CONTRIBUTING.md for Contributor Guidelines
- **Fixed dead link**: Resolved broken reference in README.md line 195
- **Comprehensive guidelines**: Added detailed contribution instructions including:
  - Development environment setup with platform-specific notes
  - Code style and Python conventions following PEP 8
  - Testing requirements with pytest examples
  - Semantic commit message format
  - Pull request process and review guidelines
- **Code of Conduct**: Included basic behavioral guidelines for inclusive community
- **Multiple contribution paths**: Documented various ways to contribute (code, docs, testing, support)
- **Integration**: Links to existing documentation (AGENTS.md, CLAUDE.md, Wiki)
- **Files Added**: `CONTRIBUTING.md`

## [6.13.5] - 2025-09-15

### 🐛 **Bug Fixes**

#### macOS Service Installation Script Fix (PR #101)
- **Fixed NameError**: Resolved `NameError: name 'paths' is not defined` in `install_macos_service.py` at line 238
  - **Root Cause**: Variable `paths` was referenced in `platform_info` dictionary without being initialized
  - **Solution**: Added `paths = get_service_paths()` call before usage, following existing code patterns
  - **Impact**: macOS service installation now works reliably without runtime errors
- **Credit**: Thanks to @hex for identifying and fixing this issue
- **Files Modified**: `scripts/install_macos_service.py`

## [6.13.4] - 2025-09-14

### 🐛 **Critical Bug Fixes**

#### Memory Search Timezone Inconsistency (Fixes Issue #99)
- **Fixed timezone handling bug**: Resolved critical inconsistency in timestamp validation between hook-generated and manual memories
  - **Root Cause**: Memory model was incorrectly interpreting ISO timestamps without 'Z' suffix as local time instead of UTC
  - **Solution**: Enhanced `iso_to_float()` function with explicit UTC handling using `calendar.timegm()`
  - **Impact**: Time-based memory searches (e.g., "yesterday", "last week") now consistently find all memories regardless of storage method
- **Improved timestamp validation**: Enhanced `_sync_timestamps()` method with better timezone mismatch detection
  - Added logic to detect timezone offset discrepancies between float and ISO timestamps
  - Automatic correction when timezone issues detected (prefers float timestamp as authoritative)
  - Better logging for timezone validation issues during memory creation
- **Comprehensive test coverage**: Added extensive test suite validating fix effectiveness
  - `test_hook_vs_manual_storage.py`: Validates consistency between hook and manual memory storage
  - `test_issue99_final_validation.py`: Confirms timezone fix resolves the original issue
  - `test_search_retrieval_inconsistency.py`: Root cause analysis and validation tests
  - `test_data_serialization_consistency.py`: Memory serialization consistency validation

### 🔧 **Technical Improvements**
- **Memory Model Enhancement**: Strengthened timestamp handling throughout the memory lifecycle
  - More robust ISO string parsing with fallback mechanisms
  - Better error handling for edge cases in timestamp conversion
  - Consistent UTC interpretation across all timestamp operations

### 📚 **Documentation Updates**
- **Issue Resolution**: GitHub Issue #99 documented and resolved with comprehensive technical analysis
- **Test Documentation**: Added detailed test suite documentation for memory storage consistency

## [6.13.3] - 2025-09-03

### 🐛 **Critical Bug Fixes**

#### macOS SQLite Extension Support (Fixes Issue #97)
- **Fixed AttributeError**: Resolved `'sqlite3.Connection' object has no attribute 'enable_load_extension'` error on macOS
  - Added comprehensive extension support detection before attempting to load sqlite-vec
  - Graceful error handling with clear, actionable error messages
  - Platform-specific guidance for macOS users (Homebrew Python, pyenv with extension support)
  - Interactive prompt to switch to ChromaDB backend when extensions unavailable
- **Enhanced sqlite_vec.py**: Added `_check_extension_support()` method with robust error handling
- **Improved install.py**: Added SQLite extension support detection and warnings during installation

### 📚 **Documentation Updates**

#### macOS Extension Loading Documentation
- **README Updates**: Added dedicated macOS SQLite extension support section with solutions
- **First-Time Setup Guide**: Comprehensive macOS extension issues section with verification commands
- **Troubleshooting Guide**: Detailed troubleshooting for `enable_load_extension` AttributeError
- **Clear Solutions**: Step-by-step instructions for Homebrew Python and pyenv with extension support

### 🔧 **Installation Improvements**
- **Proactive Detection**: Installer now checks SQLite extension support before attempting sqlite-vec installation
- **Interactive Fallback**: Option to automatically switch to ChromaDB when sqlite-vec cannot work
- **Better Error Messages**: Platform-specific solutions instead of cryptic errors
- **System Information**: Enhanced system info display includes SQLite extension support status

### 🏥 **Error Handling**
- **Runtime Protection**: sqlite-vec backend now fails gracefully with helpful error messages
- **Clear Guidance**: Detailed error messages explain why the error occurs and provide multiple solutions
- **Automatic Detection**: System automatically detects extension support capabilities

## [6.13.2] - 2025-09-03

### 🐛 **Bug Fixes**

#### Python 3.13 Compatibility (Fixes Issue #96)
- **Enhanced sqlite-vec Installation**: Added intelligent fallback strategies for Python 3.13 where pre-built wheels are not yet available
  - Automatic retry with multiple installation methods (uv pip, standard pip, source build, GitHub install)
  - Clear guidance for alternative solutions (Python 3.12, ChromaDB backend)
  - Interactive prompt to switch backends if sqlite-vec installation fails
- **Improved Error Messages**: Better error reporting with actionable manual installation options
- **uv Package Manager Support**: Prioritizes uv pip when available for better dependency resolution

### 📚 **Documentation Updates**

#### Python 3.13 Support Documentation
- **README Updates**: Added Python 3.13 compatibility note with recommended solutions
- **First-Time Setup Guide**: New section on Python 3.13 known issues and workarounds
- **Troubleshooting Guide**: Comprehensive sqlite-vec installation troubleshooting for Python 3.13
- **Clear Migration Path**: Step-by-step instructions for using Python 3.12 or switching to ChromaDB

### 🔧 **Installation Improvements**
- **Multi-Strategy Installation**: Installer now tries 5 different methods before failing
- **Source Build Fallback**: Attempts to build from source when wheels are unavailable
- **GitHub Direct Install**: Falls back to installing directly from sqlite-vec repository
- **Backend Switching**: Option to automatically switch to ChromaDB if sqlite-vec fails

## [6.13.1] - 2025-09-03

### 📚 **Documentation & User Experience**

#### First-Time Setup Documentation (Addresses Issue #95)
- **Added First-Time Setup Section**: New prominent section in README.md explaining expected warnings on initial run
- **Created Comprehensive Guide**: New `docs/first-time-setup.md` with detailed explanations of:
  - Normal warnings vs actual errors
  - Model download process (~25MB, 1-2 minutes)
  - Success indicators and verification steps
  - Ubuntu 24-specific installation instructions
- **Enhanced Troubleshooting**: Updated `docs/troubleshooting/general.md` with first-time warnings section
- **Improved Installer Messages**: Enhanced `install.py` with clear first-time setup expectations and progress indicators

### 🐛 **Bug Fixes**

#### User Experience Improvements
- **Fixed Issue #95**: Clarified that "No snapshots directory" warning is normal on first run
- **Addressed Confusion**: Distinguished between expected initialization warnings and actual errors
- **Ubuntu 24 Support**: Added specific documentation for Ubuntu 24 installation issues

### 📝 **Changes**
- Added clear messaging that warnings disappear after first successful run
- Included estimated download times and model sizes in documentation
- Improved installer output to set proper expectations for first-time users

## [6.13.0] - 2025-08-25

### ⚡ **Major Features**

#### Enhanced Storage Backend Visibility & Health Integration
- **Health Check Integration**: Claude Code hooks now use `/api/health/detailed` endpoint for **authoritative storage information**
  - **Real-time Backend Detection**: Queries live health data instead of environment variables
  - **Rich Storage Information**: Displays memory count, database size, connection status, and embedding model
  - **Comprehensive Database Stats**: Shows unique tags, file paths, platform info, and uptime
  - **Fallback Strategy**: Health check → Environment variables → Configuration inference

- **Enhanced Console Output**:
  - **Brief Mode**: `💾 Storage → 🪶 sqlite-vec (Connected) • 990 memories • 5.21MB`
  - **Detailed Mode**: Separate lines for storage type, location, and database statistics
  - **Icon-Only Mode**: Minimal display with backend type and memory count
  - **Path Display**: Shows actual database file locations from health data

- **Context Message Enhancement**:
  - **Storage Section**: Includes backend type, connection status, memory count, and database size
  - **Location Info**: Real file paths from health endpoint rather than configuration guesses  
  - **Model Information**: Displays active embedding model (e.g., all-MiniLM-L6-v2)
  - **Health Metadata**: Platform info and database accessibility status

### 🔧 **Configuration Enhancements**

#### New Health Check Settings
- **`memoryService.healthCheckEnabled`**: Enable/disable health endpoint queries (default: true)
- **`memoryService.healthCheckTimeout`**: Timeout for health requests in milliseconds (default: 3000)
- **`memoryService.useDetailedHealthCheck`**: Use `/api/health/detailed` vs `/api/health` (default: true)
- **Enhanced Display Modes**: `sourceDisplayMode` supports "brief", "detailed", "icon-only"

### 🐛 **Critical Bug Fixes**

#### Windows Path Resolution Issues
- **Git Analyzer**: Fixed Windows path handling in `execSync` calls using `path.resolve()`
- **Project Detector**: Resolved Windows path issues in git command execution
- **Path Normalization**: All working directory paths properly normalized for cross-platform compatibility

#### Memory Context Display Improvements  
- **Content Length Limits**: Increased from 300→500 characters to prevent truncation
- **CLI Formatting**: Enhanced from 180→400 chars for better context visibility
- **Categorized Display**: Improved from 160→350 chars for categorized memories
- **Deduplication Logging**: Fixed misleading "8 → 2" messages, now shows actual selection counts

#### Output Cleaning & Visual Improvements
- **Session Hook Tags**: Added cleaning logic to remove `<session-start-hook>` wrapper tags
- **Memory Categories**: Enhanced category titles (e.g., "Recent Work (Last 7 days)", "Bug Fixes & Issues")
- **Better Hierarchy**: Improved visual organization and color coding
- **Configurable Limits**: All content length limits now configurable via config.json

### 💡 **Smart Detection Improvements**

#### Backend Detection Hierarchy
1. **Health Endpoint** (Primary): Authoritative data from running service
2. **Environment Variables** (Secondary): MCP_MEMORY_STORAGE_BACKEND detection
3. **Configuration Inference** (Fallback): Endpoint-based local/remote classification

#### Supported Backend Types
- **SQLite-vec**: 🪶 Local database with file path and size information
- **ChromaDB**: 📦 Local or 🌐 Remote with endpoint details  
- **Cloudflare**: ☁️ Cloud service with account information
- **Generic Services**: 💾 Local or 🌐 Remote MCP services

### 🎯 **Performance & Reliability**

#### Health Check Implementation
- **Non-blocking**: Async health queries don't slow session start
- **Timeout Protection**: 3-second default timeout prevents hanging
- **Error Handling**: Graceful fallback to configuration-based detection
- **Caching**: Health info cached during session to avoid repeated calls

## [6.12.0] - 2025-08-25

### ⚡ **Major Features**

#### Git-Aware Memory Retrieval System
- **Repository Context Analysis**: Session hooks now analyze git commit history and changelog entries
  - **Git Context Analyzer**: New utility (`git-analyzer.js`) extracts development keywords from recent commits
  - **Commit Mining**: Analyzes last 14 days of commits to understand development patterns
  - **Changelog Integration**: Parses CHANGELOG.md to identify recent development themes
  - **Smart Query Generation**: Creates git-informed semantic queries for memory retrieval

- **Multi-Phase Memory Retrieval Enhancement**:
  - **Phase 0 (NEW)**: Git-aware memory search using repository context (highest priority)
  - **Phase 1**: Recent memories enhanced with git development keywords  
  - **Phase 2**: Important tagged memories with framework/language context
  - **Phase 3**: Fallback general context with extended time windows

- **Enhanced Context Categorization**:
  - **"Current Development" Category**: New category for git-context derived memories
  - **Repository-Aware Scoring**: Memories aligned with recent commits get higher relevance scores
  - **Development Timeline Integration**: Memory selection based on actual coding activity

### 🔧 **Configuration Enhancements**

#### New Git Analysis Settings
- **`gitAnalysis.enabled`**: Enable/disable git context analysis (default: true)
- **`gitAnalysis.commitLookback`**: Days of commit history to analyze (default: 14)
- **`gitAnalysis.maxCommits`**: Maximum commits to process (default: 20)
- **`gitAnalysis.includeChangelog`**: Parse changelog for context (default: true)
- **`gitAnalysis.maxGitMemories`**: Memory slots reserved for git context (default: 3)
- **`gitAnalysis.gitContextWeight`**: Relevance multiplier for git-derived memories (default: 1.2)

#### Enhanced Output Control
- **`output.showGitAnalysis`**: Display git analysis information (default: true)
- **`output.showPhaseDetails`**: Show detailed phase execution info (default: true)
- **Improved Memory Ratio Configuration**: `recentMemoryRatio`, `recentTimeWindow`, `fallbackTimeWindow`

### 💡 **Smart Query Improvements**

#### Development-Aware Semantic Queries
- **Commit Message Context**: Latest commit messages influence memory search
- **File Pattern Recognition**: Recently changed files inform query building  
- **Technical Keyword Extraction**: Action words (feat, fix, refactor) enhance search relevance
- **Branch-Aware Queries**: Git branch context integrated into semantic search

### 🎯 **Memory Relevance Enhancements**

#### Repository Activity Integration
- **Development Intensity Scoring**: High commit activity increases recent memory priority
- **File-Based Context**: Memories related to recently changed files get boosted relevance
- **Changelog Correlation**: Memory selection aligned with documented changes
- **Temporal Context Weighting**: Recent development activity influences memory scoring

### 🚀 **Performance & Reliability**

#### Git Integration Optimizations  
- **Lazy Git Analysis**: Only processes git context when repository detected
- **Performance Limits**: Configurable commit analysis limits to prevent slowdowns
- **Graceful Degradation**: Falls back to standard retrieval if git analysis fails
- **Async Processing**: Non-blocking git analysis for faster session startup

### 📊 **Enhanced Debugging & Monitoring**

#### Git Analysis Visibility
- **Git Context Reporting**: Shows analyzed commits, changelog entries, and extracted keywords
- **Phase Execution Details**: Clear indication of which phase found which memories
- **Query Source Tracking**: Memories tagged with their retrieval source (git-commits, git-files, etc.)
- **Development Keywords Display**: Shows extracted technical terms from recent activity

### 🔄 **Backward Compatibility**

#### Seamless Integration
- **Legacy Mode Support**: `recentFirstMode: false` preserves original behavior
- **Configuration Migration**: All existing configurations continue to work
- **Optional Git Features**: Git analysis can be completely disabled if needed
- **Incremental Adoption**: Features can be enabled progressively

---

## [6.11.1] - 2025-08-25

### 🐛 **Bug Fixes**

#### Windows Path Configuration Issues
- **Claude Code Hooks Installation**: Fixed Windows-specific path configuration issues
  - **Legacy Path Cleanup**: Removed outdated `.claude-code` references from installation scripts
  - **Windows Batch File**: Updated help text to use correct `.claude\hooks` directory structure
  - **PowerShell Script**: Cleaned up legacy path alternatives for better reliability
  - **Impact**: Windows users no longer encounter `.claude-code` vs `.claude` directory confusion

#### Enhanced Documentation
- **Windows Installation Guide**: Added comprehensive Windows-specific installation and troubleshooting section
  - **Path Format Guidance**: Clear instructions for JSON path formatting (forward slashes vs backslashes)
  - **Common Issues**: Solutions for `session-start-wrapper.bat` errors and legacy directory migration
  - **Settings Examples**: Proper Windows configuration examples with correct Node.js script paths
  - **Impact**: Windows users have clear guidance for resolving path configuration issues

#### Path Validation Utilities
- **Automated Detection**: Added path validation functions to detect and warn about configuration issues
  - **Legacy Path Detection**: Automatically identifies old `.claude-code` directories and provides migration steps
  - **Settings Validation**: Validates JSON settings files for Windows path issues and missing files
  - **Path Normalization**: Utility to convert Windows backslash paths to JSON-compatible format
  - **Validation Command**: New `python scripts/claude_commands_utils.py --validate` command for configuration checking
  - **Impact**: Proactive detection and resolution of Windows path issues during installation

## [6.10.1] - 2025-08-25

### 🐛 **Bug Fixes**

#### Fixed Claude Code Memory Commands
- **API Key Update**: Updated all Claude Code memory commands with current production API key
  - Fixed `/memory-context`, `/memory-store`, `/memory-recall`, `/memory-search`, `/memory-health` commands
  - Replaced outdated `test-key-123` with current production API key
  - **Impact**: All memory commands now work correctly with remote memory service

#### Enhanced `/memory-context` Command
- **Dynamic Session Capture**: Removed hardcoded session content, now captures real-time context
  - **Real-time Info**: Includes current timestamp, working directory, git branch, and recent commits  
  - **Dynamic Tags**: Automatically includes current project name as tag
  - **User Context**: Properly captures user-provided session description via `$ARGUMENTS`
  - **Impact**: `/memory-context` now stores actual session context instead of static placeholder text

## [6.10.0] - 2025-08-25

### ✨ **New Features**

#### Markdown-to-ANSI Conversion for Clean CLI Output
- **Automatic Markdown Processing**: Memory content with markdown formatting is now automatically converted to ANSI colors
  - **Headers**: `## Header` → Bold Cyan, `### Subheader` → Bold for visual hierarchy
  - **Emphasis**: `**bold**` → Bold, `*italic*` → Dim for text emphasis
  - **Code**: `` `inline code` `` → Gray, code blocks properly formatted with gray text
  - **Lists**: Markdown bullets (`-`, `*`) → Cyan bullet points (`•`), numbered lists → Cyan arrows (`›`)
  - **Links**: `[text](url)` → Cyan text without URL clutter
  - **Blockquotes**: `> quote` → Dimmed with visual indicator (`│`)
  - **Impact**: Raw markdown syntax no longer appears in CLI output, providing clean and professional display

#### Smart Content Processing
- **Environment-Aware**: Automatically detects CLI environment and applies appropriate formatting
- **Configuration Option**: Can be disabled via `CLAUDE_MARKDOWN_TO_ANSI=false` environment variable
- **Strip-Only Mode**: Option to remove markdown without adding ANSI colors for plain text output
- **Performance**: Minimal overhead (<5ms) with single-pass regex transformation

## [6.9.0] - 2025-08-25

### ✨ **New Features**

#### Enhanced Claude Code Hook Visual Output
- **Professional CLI Formatting**: Completely redesigned hook output with ANSI color coding and clean typography
  - **ANSI Color Support**: Full color coding with cyan, green, blue, yellow, gray, and red for different components
  - **Clean Typography**: Removed all markdown syntax (`**`, `#`, `##`) and HTML-like tags from CLI output
  - **Consistent Visual Pattern**: Standardized `icon component → description` format throughout all hook messages
  - **Unicode Box Drawing**: Enhanced use of `┌─`, `├─`, `└─`, and `│` characters for better visual structure
  - **Impact**: Hook output now looks professional and readable in Claude Code terminal sessions

#### Color-Coded Memory Content
- **Type-Based Coloring**: Different colors for memory types (decisions=yellow, insights=magenta, bugs=green, features=blue)
- **Visual Hierarchy**: Important information highlighted with bright colors, metadata dimmed with gray
- **Date Formatting**: Consistent gray coloring for dates and timestamps
- **Category Icons**: Enhanced category display with colored section headers

#### Improved Console Logging
- **Structured Messages**: All console output now follows consistent visual patterns
- **Progress Indicators**: Clear visual feedback for memory search, scoring, and processing steps
- **Error Handling**: Enhanced error messages with proper color coding and clear descriptions
- **Success Confirmation**: Prominent success indicators with checkmarks and summary information

#### Enhanced Project Detection
- **Confidence Visualization**: Color-coded confidence scores (green >80%, yellow >60%, gray <60%)
- **Clean Project Info**: Streamlined project detection output with proper visual hierarchy
- **Technology Stack Display**: Clear formatting for detected languages, frameworks, and tools

## [6.8.0] - 2025-08-25

### ✨ **New Features**

#### Enhanced Claude Code CLI Formatting
- **Beautiful CLI Output**: Claude Code hooks now feature enhanced visual formatting with Unicode box-drawing characters
  - **Visual Hierarchy**: Clean tree structure using `├─`, `└─`, and `│` characters for better readability
  - **Contextual Icons**: Memory context (🧠), project info (📂, 📝), and category-specific icons (🏗️, 🐛, ✨)
  - **Smart Environment Detection**: Automatically switches between Markdown (web) and enhanced CLI formatting
  - **Improved Categorization**: Visual grouping of memories by type with proper indentation and spacing
  - **Impact**: Dramatically improved readability of hook output in Claude Code terminal sessions

#### CLI Environment Detection
- **Automatic Detection**: Uses multiple detection methods including `CLAUDE_CODE_CLI` environment variable
- **Terminal Compatibility**: Enhanced formatting works seamlessly across different terminal emulators
- **Fallback Support**: Graceful degradation to standard formatting when enhanced features unavailable

#### Visual Design Improvements
- **Category Icons**: 🏗️ Architecture & Design, 🐛 Bug Fixes & Issues, ✨ Features & Implementation, 📝 Additional Context
- **Project Status Display**: Git branch (📂) and commit info (📝) with clean formatting
- **Memory Count Indicators**: Clear display of loaded memory count with 📚 icon
- **Empty State Handling**: Elegant display when no memories available

## [6.7.2] - 2025-08-25

### 🔧 **Enhancement**

#### Deduplication Script Configuration-Aware Refactoring
- **Enhanced API Integration**: Deduplication script now reads Claude Code hooks configuration for seamless integration
  - **Configuration Auto-Detection**: Automatically reads `~/.claude/hooks/config.json` for memory service endpoint and API key
  - **API-Based Analysis**: Supports remote memory service analysis via HTTPS API with self-signed certificate support
  - **Intelligent Pagination**: Handles large memory collections (972+ memories) with automatic page-by-page retrieval
  - **Backward Compatibility**: Maintains support for direct database access with `--db-path` parameter
  - **Impact**: Users can now run deduplication on remote memory services without manual configuration

#### New Command-Line Options
- **`--use-api`**: Force API-based analysis using configuration endpoint
- **Smart Fallback**: Automatically detects available options and suggests alternatives when paths missing
- **Enhanced Error Messages**: Clear guidance on configuration requirements and troubleshooting steps

#### Technical Improvements
- **SSL Context Configuration**: Proper handling of self-signed certificates for secure remote connections
- **Memory Format Normalization**: Converts API response format to internal analysis format seamlessly
- **Progress Reporting**: Real-time feedback during multi-page memory retrieval operations

#### Validation Results
- **✅ 972 Memories Analyzed**: Successfully processed complete memory collection via API
- **✅ No Duplicates Detected**: Confirms v6.7.0 content quality filters are preventing duplicate creation
- **✅ Configuration Integration**: Seamless integration with existing Claude Code hooks setup
- **✅ Performance Maintained**: Efficient pagination and processing of large memory collections

> **Usage**: Run `python scripts/find_duplicates.py --use-api` to analyze memories using your configured remote memory service

## [6.7.1] - 2025-08-25

### 🔧 **Critical Bug Fix**

#### Claude Code Hooks Installation Script Fix
- **Fixed Missing v6.7.0 Files in Installation**: Resolved critical installation script bug that prevented complete v6.7.0 setup
  - **Added Missing Core Hook**: `memory-retrieval.js` now properly copied during installation
  - **Added Missing Utility**: `context-shift-detector.js` now properly copied during installation
  - **Updated Install Messages**: Installation logs now accurately reflect all installed files
  - **Impact**: Eliminates "Cannot find module" errors on fresh v6.7.0 installations

#### Problem Resolved
- **Issue**: Installation script (`install.sh`) used hardcoded file list that was missing new v6.7.0 files
- **Symptoms**: Users experienced module import errors and incomplete feature sets after installation
- **Root Cause**: Script copied only `session-start.js`, `session-end.js` but missed `memory-retrieval.js` and `context-shift-detector.js`
- **Solution**: Added missing file copy commands and updated installation messaging

#### Validation
- **✅ Complete Installation**: All v6.7.0 files now properly installed
- **✅ Integration Tests**: 14/14 tests pass immediately after fresh installation 
- **✅ No Module Errors**: All dependencies resolved correctly
- **✅ Full Feature Set**: On-demand memory retrieval and smart timing work out-of-the-box

> **Note**: This is a critical patch for v6.7.0 users. If you installed v6.7.0 and experienced module errors, please reinstall using v6.7.1.

## [6.7.0] - 2025-08-25

### 🧠 **Claude Code Memory Awareness - Major Enhancement**

#### Smart Memory Context Presentation ([#Memory-Presentation-Fix](https://github.com/doobidoo/mcp-memory-service/issues/memory-presentation))
- **Eliminated Generic Content Fluff**: Complete overhaul of session-start hook memory presentation
  - **Smart Content Extraction**: Extracts meaningful content from session summaries (decisions, insights, code changes) instead of truncating at 200 chars
  - **Section-Aware Parsing**: Prioritizes showing actual decisions/insights over generic headers like "Topics Discussed - implementation..."
  - **Deduplication Logic**: Automatically filters out repetitive session summaries with 80% similarity threshold
  - **Impact**: Users now see actionable memory content instead of truncated generic summaries

#### Enhanced Memory Quality Scoring
- **Content Quality Factor**: New scoring dimension that heavily penalizes generic/empty content
  - **Meaningful Indicators**: Detects substantial content using decision words (decided, implemented, fixed, learned)
  - **Information Density**: Analyzes word diversity and content richness
  - **Generic Pattern Detection**: Automatically identifies and demotes "implementation..." session summaries
  - **Impact**: Quality memories now outrank recent-but-empty summaries

#### Smart Context Management
- **Post-Compacting Control**: Added `injectAfterCompacting: false` configuration option
  - **Problem Solved**: Stops disruptive mid-session memory injection after compacting events
  - **User-Controlled**: Can be enabled via `config.json` for users who prefer the old behavior
- **Context Shift Detection**: Intelligent timing for when to refresh memory context
  - **Project Change Detection**: Auto-refreshes when switching between projects/directories
  - **Topic Shift Analysis**: Detects significant conversation topic changes
  - **User Request Recognition**: Responds to explicit memory requests ("remind me", "what did we decide")
  - **Impact**: Memory context appears only when contextually appropriate, not disruptively

#### New Features
- **On-Demand Memory Retrieval**: New `memory-retrieval.js` hook for manual context requests
  - **User-Triggered**: Allows manual memory refresh when needed
  - **Query-Based**: Supports semantic search with user-provided queries
  - **Relevance Scoring**: Shows confidence scores for manual retrieval
- **Streamlined Presentation**: Cleaner formatting with reduced metadata clutter
  - **Smart Categorization**: Only groups memories when there's meaningful diverse content
  - **Relevant Tags Only**: Filters out machine-generated and generic tags
  - **Concise Dating**: More compact date formatting (Aug 25 vs full timestamps)

#### Technical Improvements
- **Enhanced Memory Scoring Weights**:
  ```json
  {
    "timeDecay": 0.25,        // Reduced from 0.30
    "tagRelevance": 0.35,     // Maintained
    "contentRelevance": 0.15,  // Reduced from 0.20
    "contentQuality": 0.25,    // NEW quality factor
    "conversationRelevance": 0.25 // Maintained
  }
  ```
- **New Utility Files**:
  - `context-shift-detector.js`: Intelligent context change detection
  - Enhanced `context-formatter.js` with smart content extraction
  - Quality-aware scoring in `memory-scorer.js`

#### Breaking Changes
- **Session-Start Hook**: Upgraded to v2.0.0 with smart timing and quality filtering
- **Configuration Schema**: Added new options for memory quality control and compacting behavior
- **Memory Filtering**: Generic session summaries are now automatically filtered out

#### Migration Guide
- **Automatic**: No user action required - existing configurations work with sensible defaults
- **Optional**: Users can enable `injectAfterCompacting: true` in config.json if they prefer old behavior
- **Benefit**: Immediate improvement in memory context quality and relevance

## [6.6.4] - 2025-08-25

### 🔧 **Installation Experience Improvements**

#### Universal Installer Bug Fixes ([#92](https://github.com/doobidoo/mcp-memory-service/issues/92))
- **Fixed "Installer Gets Stuck" Issues**: Resolved multiple causes of installation appearing frozen
  - Added prominent visual separators (`⚠️ USER INPUT REQUIRED`) around all input prompts
  - Extended system detection timeouts from 10 to 30 seconds (macOS `system_profiler`, Homebrew checks)
  - Added progress indicators for long-running operations (package installations, hardware detection)
  - **Impact**: Users now clearly understand when installer needs input vs. processing in background

#### New Non-Interactive Installation Mode
- **Added `--non-interactive` Flag**: Enables fully automated installations using sensible defaults
  - Automatically selects SQLite-vec backend (recommended)
  - Skips optional components (Claude Code commands, multi-client setup)
  - Perfect for CI/CD, Docker builds, and scripted deployments
  - **Usage**: `python install.py --non-interactive`

#### Enhanced User Experience
- **Better Progress Feedback**: Clear messaging during system detection and package installation phases
- **Improved Timeout Handling**: More resilient system detection with graceful fallbacks
- **Log File Visibility**: Prominently displays location of `installation.log` for troubleshooting

## [6.6.3] - 2025-08-24

### 🔧 **CI/CD Infrastructure Improvements**

#### GitHub Actions Workflow Fixes
- **Fixed npm dependency management**: Created proper `package.json` files for test directories
- **Simplified YAML structure**: Replaced complex multi-line scripts with focused single-step commands
- **Improved error reporting**: Split validation steps for clearer failure identification
- **Fixed false positives**: Updated status code validation to avoid flagging legitimate 200/201 handling
- **Enhanced reliability**: Used `working-directory` instead of embedded `cd` commands
- **Impact**: GitHub Actions CI/CD now properly installs dependencies and validates bridge fixes

#### Test Infrastructure Enhancements
- **Added dedicated test packages**: Separate `package.json` for bridge and integration tests
- **Improved module resolution**: Confirmed correct relative path handling in test files
- **Better validation coverage**: Enhanced API contract and smoke test verification

## [6.6.2] - 2025-08-24

### 🐛 **Critical Bug Fixes**

#### HTTP-MCP Bridge Complete Repair
- **Fixed Status Code Handling**: Corrected bridge expectation from HTTP 201 to HTTP 200 with success field check
  - Server returns HTTP 200 for both successful storage and duplicates
  - Bridge now properly checks `response.data.success` field to determine actual result
  - Supports both HTTP 200 and 201 for backward compatibility
  - **Impact**: Memory storage operations now work correctly instead of always failing
  
- **Fixed URL Construction Bug**: Resolved critical path construction issue in `makeRequestInternal`
  - `new URL(path, baseUrl)` was incorrectly replacing `/api` base path
  - Now properly concatenates base URL with API paths: `baseUrl + fullPath`
  - **Impact**: All API operations now reach correct endpoints instead of returning 404

#### Root Cause Analysis
- **Memory Storage**: Bridge expected HTTP 201 but server returns 200 → always interpreted as failure
- **All API Calls**: URL construction bug caused `/api` path to be lost → all requests returned 404
- **Combined Effect**: Made entire bridge non-functional despite server working correctly

### 🧪 **Major Enhancement: Comprehensive Test Infrastructure**

#### Test Suite Addition
- **Bridge Unit Tests** (`tests/bridge/test_http_mcp_bridge.js`): 20+ test cases covering:
  - URL construction with various base path scenarios
  - Status code handling for success, duplicates, and errors
  - MCP protocol compliance and error conditions
  - Authentication and retry logic
  
- **Integration Tests** (`tests/integration/test_bridge_integration.js`): End-to-end testing with:
  - Real server connectivity simulation
  - Complete MCP protocol flow validation
  - Authentication and error recovery testing
  - Critical bug scenario reproduction

- **Mock Response System** (`tests/bridge/mock_responses.js`): Accurate server behavior simulation
  - Matches actual API responses (HTTP 200 with success field)
  - Covers all edge cases and error conditions
  - Prevents future assumption-based bugs

#### CI/CD Pipeline Enhancement
- **Dedicated Bridge Testing** (`.github/workflows/bridge-tests.yml`):
  - Automated testing on every bridge-related change
  - Multiple Node.js version compatibility testing
  - Contract validation against API specification
  - Blocks merges if bridge tests fail

#### Documentation & Contracts
- **API Contract Specification** (`tests/contracts/api-specification.yml`):
  - Documents ACTUAL server behavior vs assumptions
  - Critical notes about HTTP 200 status codes and success fields
  - URL path requirements and authentication details

- **Release Process** (`RELEASE_CHECKLIST.md`):
  - 50+ item checklist preventing critical bugs
  - Manual and automated testing requirements
  - Post-release monitoring procedures

### 🔧 **Technical Details**

**Files Modified:**
- `examples/http-mcp-bridge.js` - Status code and URL construction fixes
- `src/mcp_memory_service/__init__.py` - Version bump
- `pyproject.toml` - Version bump

**Files Added:**
- Complete test infrastructure (6 new files)
- CI/CD pipeline configuration
- API contract documentation
- Release process documentation

**Backward Compatibility**: 
- All existing configurations continue working
- Bridge now accepts both HTTP 200 and 201 responses
- No breaking changes to client interfaces

### 🎯 **Impact**

**Before v6.6.2:**
- ❌ Health check: "unhealthy" → Fixed in v6.6.1
- ❌ Memory storage: "Not Found" errors 
- ❌ Memory retrieval: 404 failures
- ❌ All bridge operations non-functional

**After v6.6.2:**
- ✅ Health check: Returns "healthy" status
- ✅ Memory storage: Works correctly with proper success/duplicate handling
- ✅ Memory retrieval: Functions normally with semantic search
- ✅ All bridge operations fully functional

**Future Prevention:**
- ✅ Comprehensive test coverage prevents regression
- ✅ API contract validation catches assumption mismatches
- ✅ Automated CI/CD testing on every change
- ✅ Detailed release checklist ensures quality

This release completes the HTTP-MCP bridge repair, making it fully functional while establishing comprehensive testing infrastructure to prevent similar critical bugs in the future.

## [6.6.1] - 2025-08-24

### 🐛 **Bug Fixes**

#### HTTP-MCP Bridge Health Check Endpoint
- **Fixed Health Check URLs**: Corrected incorrect endpoint paths in `examples/http-mcp-bridge.js`
  - `testEndpoint()` method: Fixed `/health` → `/api/health` path (line 241)
  - `checkHealth()` method: Fixed `/health` → `/api/health` path (line 501)
  - **Impact**: Health checks now return "healthy" status instead of 404 errors
  - **Root Cause**: Bridge was requesting wrong endpoint causing false "unhealthy" status
  - **Verification**: Remote memory service connectivity confirmed working correctly

#### Technical Details
- Fixed endpoint inconsistencies that caused health checks to fail
- Remote memory service functionality was working correctly - issue was purely endpoint path mismatch
- All memory operations (store/retrieve) continue to work without changes
- Health check now properly reports service status to MCP clients

**Files Modified**: `examples/http-mcp-bridge.js`

## [6.6.0] - 2025-08-23

### 🔧 **Memory Awareness Hooks: Fully Functional Installation**

#### Major Improvements
- **Installation Scripts Fixed** - Memory Awareness Hooks now install and work end-to-end without manual intervention
- **Claude Code Integration** - Automatic `~/.claude/settings.json` configuration for seamless hook detection
- **JSON Parsing Fixed** - Resolved Python dict conversion errors that caused runtime failures
- **Enhanced Testing** - Added 4 new integration tests (total: 14 tests, 100% pass rate)

#### Added
- **Automated Claude Code Settings Integration** - Installation script now configures `~/.claude/settings.json` automatically
- **Comprehensive Troubleshooting Guide** - Complete wiki documentation with diagnosis commands and solutions
- **Installation Verification** - Post-install connectivity tests and hook detection validation
- **GitHub Wiki Documentation** - Detailed 500+ line guide for advanced users and developers

#### Fixed  
- **Installation Directory** - Changed from `~/.claude-code/hooks/` to correct `~/.claude/hooks/` location
- **JSON Parsing Errors** - Added Python dict to JSON conversion (single quotes, True/False/None handling)
- **Hook Detection** - Claude Code now properly detects installed hooks via settings configuration
- **Memory Service Connectivity** - Enhanced error handling and connection testing

#### Enhanced
- **Integration Tests** - Expanded test suite from 10 to 14 tests (40% increase):
  - Claude Code settings validation
  - Hook files location verification  
  - Claude Code CLI availability check
  - Memory service connectivity testing
- **Documentation Structure** - README streamlined (47% size reduction), detailed content moved to wiki
- **Error Messages** - Improved debugging output and user-friendly error reporting

#### Developer Experience
- **Quick Start** - Single command installation: `./install.sh`
- **Verification Commands** - Easy testing with `claude --debug hooks`
- **Troubleshooting** - Comprehensive guide covers all common issues
- **Custom Development** - Examples for extending hooks and memory integration

**Impact**: Memory Awareness Hooks are now publicly usable without the manual debugging sessions that were previously required. Users can run `./install.sh` and get fully functional automatic memory awareness in Claude Code.

## [6.5.0] - 2025-08-23

### 🗂️ **Repository Structure: Root Directory Cleanup**

#### Added
- **`deployment/` Directory** - Centralized location for service and configuration files
- **`sync/` Directory** - Organized synchronization scripts in dedicated folder
- **Dependency-Safe Reorganization** - All file references properly updated

#### Changed
- **Root Directory Cleanup** - Reduced clutter from 80+ files to ~65 files (19% reduction)
- **File Organization** - Moved deployment configs and sync scripts to logical subdirectories
- **Path References Updated** - All scripts and configurations point to new file locations
- **Claude Code Settings** - Updated paths in `.claude/settings.local.json` for moved scripts

#### Moved Files
**To `deployment/`:**
- `mcp-memory.service` - Systemd service configuration
- `smithery.yaml` - Smithery package configuration  
- `io.litestream.replication.plist` - macOS LaunchDaemon configuration
- `staging_db_init.sql` - Database initialization schema
- `empty_config.yml` - Template configuration file

**To `sync/`:**
- `manual_sync.sh`, `memory_sync.sh` - Core synchronization scripts
- `pull_remote_changes.sh`, `push_to_remote.sh` - Remote sync operations
- `sync_from_remote.sh`, `sync_from_remote_noconfig.sh` - Remote pull variants
- `apply_local_changes.sh`, `stash_local_changes.sh`, `resolve_conflicts.sh` - Local sync operations

#### Fixed
- **Broken References** - Updated all hardcoded paths in scripts and configurations
- **Claude Code Integration** - Fixed manual_sync.sh path reference
- **Installation Scripts** - Updated service file and SQL schema paths
- **Litestream Setup** - Fixed LaunchDaemon plist file reference

#### Repository Impact
This release significantly improves repository navigation and organization while maintaining full backward compatibility through proper reference updates. Users benefit from cleaner root directory structure, while developers gain logical file organization that reflects functional groupings.

## [6.4.0] - 2025-08-23

### 📚 **Documentation Revolution: Major UX Transformation**

#### Added
- **Comprehensive Installation Guide** in wiki consolidating 6+ previous scattered guides
- **Platform-Specific Setup Guide** with optimizations for Windows, macOS, and Linux
- **Complete Integration Guide** covering Claude Desktop, Claude Code, VS Code, and 13+ tools
- **Streamlined README.md** with clear quick start and wiki navigation (56KB → 8KB)
- **Safe Documentation Archive** preserving all removed files for reference

#### Changed
- **Major Documentation Restructuring** for improved user experience and maintainability
- **Single Source of Truth** established for installation, platform setup, and integrations
- **Wiki Home Page** updated with organized navigation to consolidated guides
- **User Onboarding Journey** simplified from overwhelming choice to clear path

#### Removed
- **26+ Redundant Documentation Files** safely archived while preserving git history
- **Choice Paralysis** from 6 installation guides, 5 integration guides, 4 platform guides
- **Maintenance Burden** of updating multiple files for single concept changes
- **Inconsistent Information** across overlapping documentation

#### Fixed
- **User Confusion** from overwhelming documentation choices and unclear paths
- **Maintenance Complexity** requiring updates to 6+ files for installation changes
- **Professional Image** with organized structure reflecting code quality
- **Information Discovery** through logical wiki organization vs scattered files

#### Documentation Impact
This release transforms MCP Memory Service from having overwhelming documentation chaos to organized, professional, maintainable structure. Users now have clear paths from discovery to success (README → Quick Start → Wiki → Success), while maintainers benefit from single-source-of-truth updates. **90% reduction in repository documentation files** while **improving comprehensiveness** through organized wiki structure.

**Key Achievements:**
- Installation guides: 6 → 1 comprehensive wiki page
- Integration guides: 5 → 1 complete reference  
- Platform guides: 4 → 1 optimized guide
- User experience: Confusion → Clarity
- Maintenance: 6+ places → 1 place per topic

## [6.3.3] - 2025-08-22

### 🔧 **Enhancement: Version Synchronization**

#### Fixed
- **API Documentation Version**: Fixed API docs dashboard showing outdated version `1.0.0` instead of current version
- **Version Inconsistencies**: Synchronized hardcoded versions across FastAPI app, web module, and server configuration
- **Maintenance Overhead**: Established single source of truth for version management

#### Enhanced
- **Dynamic Version Management**: All version references now import from main `__version__` variable
- **Future-Proofing**: Version updates now only require changes in 2 files (pyproject.toml + __init__.py)
- **Developer Experience**: Consistent version display across all interfaces and documentation

#### Technical Details
- **FastAPI App**: Changed from hardcoded `version="1.0.0"` to dynamic `version=__version__`
- **Web Module**: Removed separate version `0.2.0`, now imports from parent package
- **Server Config**: Updated `SERVER_VERSION` from `0.2.2` to use main version import
- **Impact**: All dashboards, API docs, and mDNS advertisements now show consistent version

## [6.3.2] - 2025-08-22

### 🚨 **Critical Fix: Claude Desktop Compatibility Regression**

#### Fixed
- **Claude Desktop Integration**: Restored backward compatibility for `uv run memory` command
- **MCP Protocol Errors**: Fixed JSON parsing errors when Claude Desktop tried to parse CLI help text as MCP messages
- **Regression from v6.3.1**: CLI consolidation accidentally broke existing Claude Desktop configurations

#### Technical Details
- **Root Cause**: CLI consolidation removed ability to start MCP server with `uv run memory` (without `server` subcommand)
- **Impact**: Claude Desktop configurations calling `uv run memory` received help text instead of MCP server
- **Solution**: Added backward compatibility logic to default to `server` command when no subcommand provided

#### Added
- **Backward Compatibility**: `uv run memory` now starts MCP server automatically for existing integrations
- **Deprecation Warning**: Informative warning guides users to explicit `memory server` syntax
- **Integration Test**: New test case verifies backward compatibility warning functionality
- **MCP Protocol Validation**: Confirmed proper JSON-RPC responses instead of help text parsing errors

#### For Users
- **Claude Desktop works again**: Existing configurations continue working without changes
- **Migration Encouraged**: Warning message guides users toward preferred `memory server` syntax
- **No Breaking Changes**: All existing usage patterns maintained while encouraging modern syntax

## [6.3.1] - 2025-08-22

### 🔧 **Major Enhancement: CLI Architecture Consolidation**

#### Fixed
- **CLI Conflicts Eliminated**: Resolved installation conflicts between argparse and Click CLI implementations
- **Command Consistency**: Established `uv run memory server` as the single, reliable server start pattern
- **Installation Issues**: Fixed stale entry point problems that caused "command not found" errors

#### Enhanced  
- **Unified CLI Interface**: All server commands now route through Click-based CLI for consistency
- **Deprecation Warnings**: Added graceful migration path with informative deprecation warnings for `memory-server`
- **Error Handling**: Improved error messages and graceful failure handling across all CLI commands
- **Documentation**: Added comprehensive CLI Migration Guide with clear upgrade paths

#### Added
- **CLI Integration Tests**: 16 comprehensive test cases covering all CLI interfaces and edge cases
- **Performance Testing**: CLI startup time validation and version command performance monitoring
- **Backward Compatibility**: `memory-server` command maintained with deprecation warnings
- **Migration Guide**: Complete documentation for transitioning from legacy commands

#### Technical Improvements
- **Environment Variable Integration**: Seamless config passing via MCP_MEMORY_CHROMA_PATH
- **Code Cleanup**: Removed duplicate argparse implementation from server.py
- **Entry Point Simplification**: Streamlined from 3 conflicting implementations to 1 clear interface
- **Robustness Testing**: Added error handling, argument parity, and isolation testing

#### Benefits for Users
- **Eliminates MCP startup failures** that were caused by CLI conflicts
- **Clear command interface**: `uv run memory server` works reliably every time
- **Better error messages** when something goes wrong
- **Smooth migration path** from legacy patterns
- **Full Claude Code compatibility** confirmed and tested

## [6.3.0] - 2025-08-22

### 🚀 **Major Feature: Distributed Memory Synchronization**

#### Added
- **Git-like Sync Workflow**: Complete distributed synchronization system with offline capabilities
  - `memory_sync.sh` - Main orchestrator for sync operations with colored output and status reporting
  - **Stash → Pull → Apply → Push** workflow similar to Git's distributed version control
  - Intelligent conflict detection and resolution with user control
  - Remote-first architecture with automatic fallback to local staging

- **Enhanced Memory Store**: `enhanced_memory_store.sh` with hybrid remote-first approach
  - **Primary**: Direct storage to remote API (`https://narrowbox.local:8443/api/memories`)
  - **Fallback**: Automatic local staging when remote is unavailable
  - Smart context detection (project, git branch, hostname, tags)
  - Seamless offline-to-online transition with automatic sync

- **Real-time Database Replication**: Litestream-based synchronization infrastructure
  - **Master/Replica Setup**: Remote server as master with HTTP-served replica data
  - **Automatic Replication**: 10-second sync intervals for near real-time updates
  - **Lightweight HTTP Server**: Python built-in server for serving replica files
  - **Cross-platform Compatibility**: macOS LaunchDaemon and Linux systemd services

- **Staging Database System**: SQLite-based local change tracking
  - `staging_db_init.sql` - Complete schema with triggers and status tracking
  - **Conflict Detection**: Content hash-based duplicate detection
  - **Operation Tracking**: INSERT/UPDATE/DELETE operation logging
  - **Source Attribution**: Machine hostname and timestamp tracking

- **Comprehensive Sync Scripts**: Complete workflow automation
  - `stash_local_changes.sh` - Capture local changes before remote sync
  - `pull_remote_changes.sh` - Download remote changes with conflict awareness  
  - `apply_local_changes.sh` - Intelligent merge of staged changes
  - `push_to_remote.sh` - Upload changes via HTTPS API with retry logic
  - `resolve_conflicts.sh` - Interactive conflict resolution helper

- **Litestream Integration**: Production-ready replication setup
  - **Automated Setup Scripts**: `setup_remote_litestream.sh` and `setup_local_litestream.sh`
  - **Service Configurations**: systemd and LaunchDaemon service files
  - **Master/Replica Configs**: Complete Litestream YAML configurations
  - **Comprehensive Documentation**: `LITESTREAM_SETUP_GUIDE.md` with troubleshooting

#### Enhanced
- **Claude Commands**: Updated `memory-store.md` to document remote-first approach
  - **Hybrid Strategy**: Remote API primary, local staging fallback
  - **Sync Status**: Integration with `./sync/memory_sync.sh status` for pending changes
  - **Automatic Context**: Git branch, project, and hostname detection

#### Architecture
- **Remote-first Design**: Single source of truth on remote server with local caching
- **Conflict Resolution**: Last-write-wins with comprehensive logging and user notification
- **Network Resilience**: Graceful degradation from online → staging → read-only local
- **Git-like Workflow**: Familiar distributed workflow for developers

#### Technical Details
- **Database Schema**: New staging database with triggers for change counting
- **HTTP Integration**: Secure HTTPS API communication with bearer token auth
- **Platform Support**: Cross-platform service management (systemd/LaunchDaemon)
- **Performance**: Lz4 compression for efficient snapshot transfers
- **Security**: Content hash verification and source machine tracking

This release transforms MCP Memory Service from a local-only system into a fully distributed memory platform, enabling seamless synchronization across multiple devices while maintaining robust offline capabilities.

## [6.2.5] - 2025-08-21

### 🐛 **Bug Fix: SQLite-Vec Backend Debug Utilities**

This release fixes a critical AttributeError in debug utilities when using the SQLite-Vec storage backend.

#### Fixed
- **Debug Utilities Compatibility** ([#89](https://github.com/doobidoo/mcp-memory-service/issues/89)): Fixed `'SqliteVecMemoryStorage' object has no attribute 'model'` error
  - Added compatibility helper `_get_embedding_model()` to handle different attribute names between storage backends
  - ChromaDB backend uses `storage.model` while SQLite-Vec uses `storage.embedding_model`
  - Updated all debug functions (`get_raw_embedding`, `check_embedding_model`, `debug_retrieve_memory`) to use the compatibility helper
  
#### Technical Details
- **Affected Functions**: 
  - `get_raw_embedding()` - Now works with both backends
  - `check_embedding_model()` - Properly detects model regardless of backend
  - `debug_retrieve_memory()` - Semantic search debugging works for SQLite-Vec users
  
#### Impact
- Users with SQLite-Vec backend can now use all MCP debug operations
- Semantic search and embedding inspection features work correctly
- No breaking changes for ChromaDB backend users

## [6.2.4] - 2025-08-21

### 🐛 **CRITICAL BUG FIXES: Claude Code Hooks Compatibility**

This release fixes critical compatibility issues with Claude Code hooks that prevented automatic memory injection at session start.

#### Fixed
- **Claude Code Hooks API Parameters**: Fixed incorrect API parameters in `claude-hooks/core/session-start.js`
  - Replaced invalid `tags`, `limit`, `time_filter` parameters with correct `n_results` for `retrieve_memory` API
  - Hooks now work correctly with MCP Memory Service API without parameter errors
  
- **Python Dict Response Parsing**: Fixed critical parsing bug where hooks couldn't process MCP service responses
  - Added proper Python dictionary format to JavaScript object conversion 
  - Implemented fallback parsing for different response formats
  - Hooks now successfully parse memory service responses and inject context

- **Memory Export Security**: Enhanced security for memory export files
  - Added `local_export*.json` to .gitignore to prevent accidental commits of sensitive data
  - Created safe template files in `examples/` directory for documentation and testing

#### Added
- **Memory Export Templates**: New example files showing export format structure
  - `examples/memory_export_template.json`: Basic example with 3 sample memories
  - Clean, sanitized examples safe for sharing and documentation

#### Technical Details
- **Response Format Handling**: Hooks now handle Python dict format responses with proper conversion to JavaScript objects
- **Error Handling**: Added multiple fallback mechanisms for response parsing
- **API Compatibility**: Updated to use correct MCP protocol parameters for memory retrieval

#### Impact
- Claude Code hooks will now work out-of-the-box without manual fixes
- Memory context injection at session start now functions correctly
- Users can install hooks directly from repository without encountering parsing errors
- Enhanced security prevents accidental exposure of sensitive data in exports

## [6.2.3] - 2025-08-20

### 🛠️ **Cross-Platform Path Detection & Claude Code Integration**

This release provides comprehensive cross-platform fixes for path detection issues and complete Claude Code hooks integration across Linux, macOS, and Windows.

#### Fixed
- **Linux Path Detection**: Enhanced `scripts/remote_ingest.sh` to auto-detect mcp-memory-service repository location anywhere in user's home directory
  - Resolves path case sensitivity issues (Repositories vs repositories)
  - Works regardless of where users clone the repository
  - Validates found directory contains pyproject.toml to ensure correct repository

- **Windows Path Detection**: Added comprehensive Windows support with PowerShell and batch scripts
  - New: `claude-hooks/install_claude_hooks_windows.ps1` - Full-featured PowerShell installation
  - New: `claude-hooks/install_claude_hooks_windows.bat` - Batch wrapper for easy execution
  - Dynamic repository location detection using PSScriptRoot resolution
  - Comprehensive Claude Code hooks directory detection with fallbacks
  - Improved error handling and validation for source/target directories
  - Resolves hardcoded Unix path issues (`\home\hkr\...`) on Windows systems
  - Tested with 100% success rate across Windows environments

- **Claude Code Commands Documentation**: Fixed and enhanced memory commands documentation
  - Updated command usage from `/memory-store` to `claude /memory-store`
  - Added comprehensive troubleshooting section for command installation issues
  - Documented both Claude Code commands and direct API alternatives
  - Added installation instructions and quick fixes for common problems

#### Technical Improvements
- Repository detection now works on any platform and directory structure
- Claude Code hooks installation handles Windows-specific path formats
- Improved error messages and debug output across all platforms
- Consistent behavior across Windows, Linux, and macOS platforms

## [6.2.2] - 2025-08-20

### 🔧 Fixed  
- **Remote Ingestion Script Path Detection**: Enhanced `scripts/remote_ingest.sh` to auto-detect mcp-memory-service repository location anywhere in user's home directory instead of hardcoded path assumptions
  - Resolves path case sensitivity issues (Repositories vs repositories)
  - Works regardless of where users clone the repository  
  - Validates found directory contains pyproject.toml to ensure correct repository

## [6.2.1] - 2025-08-20

### 🐛 **CRITICAL BUG FIXES: Memory Listing and Search Index**

This patch release resolves critical issues with memory pagination and search functionality that were preventing users from accessing the full dataset.

#### Fixed
- **Memory API Pagination**: Fixed `/api/memories` endpoint returning only 82 of 904 total memories
  - **Root Cause**: API was using semantic search fallback instead of proper chronological listing
  - **Solution**: Implemented dedicated `get_all_memories()` method with SQL-based LIMIT/OFFSET pagination
  - **Impact**: Web dashboard and API clients now see accurate memory counts and can access complete dataset

- **Missing Storage Backend Methods**: Added missing pagination methods in SqliteVecMemoryStorage
  - `get_all_memories(limit, offset)` - Chronological memory listing with pagination support
  - `get_recent_memories(n)` - Get n most recent memories efficiently
  - `count_all_memories()` - Accurate total count for pagination calculations
  - `_row_to_memory(row)` - Proper database row to Memory object conversion with JSON parsing

- **Search Index Issues**: Resolved problems with recently stored memories not appearing in searches
  - **Tag Search**: Newly stored memories now immediately appear in tag-based filtering
  - **Semantic Search**: MCP protocol semantic search verified working with similarity scoring
  - **Memory Context**: `/memory-context` command functionality confirmed end-to-end

#### Technical Details
- **Files Modified**: 
  - `src/mcp_memory_service/storage/sqlite_vec.py` - Added 75+ lines of pagination methods
  - `src/mcp_memory_service/web/api/memories.py` - Fixed pagination logic to use proper SQL queries
- **Database Access**: Replaced unreliable semantic search with direct SQL `ORDER BY created_at DESC`
- **Error Handling**: Added comprehensive JSON parsing for tags and metadata with graceful fallbacks
- **Verification**: All 904 memories now accessible via REST API with proper page navigation

#### Verification Results
- ✅ **API Pagination**: Returns accurate 904 total count (was showing 82)
- ✅ **Search Functionality**: Tag searches work immediately after storage
- ✅ **Memory Context**: Session storage and retrieval verified end-to-end
- ✅ **Semantic Search**: MCP protocol search confirmed functional with similarity scoring
- ✅ **Performance**: No performance degradation despite handling full dataset

This release ensures reliable access to the complete memory dataset with proper pagination and search capabilities.

---

## [6.2.0] - 2025-08-20

### 🚀 **MAJOR FEATURE: Native Cloudflare Backend Integration**

This major release introduces native Cloudflare integration as a third storage backend option alongside SQLite-vec and ChromaDB, providing global distribution, automatic scaling, and enterprise-grade infrastructure, integrated with the existing Memory Awareness system.

#### Added
- **Native Cloudflare Backend Support**: Complete implementation using Cloudflare's edge computing platform
  - **Vectorize**: 768-dimensional vector storage with cosine similarity for semantic search
  - **D1 Database**: SQLite-compatible database for metadata storage
  - **Workers AI**: Embedding generation using @cf/baai/bge-base-en-v1.5 model
  - **R2 Storage** (optional): Object storage for large content exceeding 1MB threshold
  
- **Implementation Files**:
  - `src/mcp_memory_service/storage/cloudflare.py` - Complete CloudflareStorage implementation (740 lines)
  - `scripts/migrate_to_cloudflare.py` - Migration tool for existing SQLite-vec and ChromaDB users
  - `scripts/test_cloudflare_backend.py` - Comprehensive test suite with automated validation
  - `scripts/setup_cloudflare_resources.py` - Automated Cloudflare resource provisioning
  - `docs/cloudflare-setup.md` - Complete setup, configuration, and troubleshooting guide
  - `tests/unit/test_cloudflare_storage.py` - 15 unit tests for CloudflareStorage class

- **Features**:
  - Automatic retry logic with exponential backoff for API rate limiting
  - Connection pooling for optimal HTTP performance
  - NDJSON format support for Vectorize v2 API endpoints
  - LRU caching (1000 entries) for embedding deduplication
  - Batch operations support for efficient data processing
  - Global distribution with <100ms latency from most locations
  - Pay-per-use pricing model with no upfront costs

#### Changed
- Updated `config.py` to include 'cloudflare' in SUPPORTED_BACKENDS
- Enhanced server initialization in `mcp_server.py` to support Cloudflare backend
- Updated storage factory in `storage/__init__.py` to create CloudflareStorage instances
- Consolidated documentation, removing redundant setup files

#### Technical Details
- **Environment Variables**:
  - `MCP_MEMORY_STORAGE_BACKEND=cloudflare` - Activate Cloudflare backend
  - `CLOUDFLARE_API_TOKEN` - API token with Vectorize, D1, Workers AI permissions
  - `CLOUDFLARE_ACCOUNT_ID` - Cloudflare account identifier
  - `CLOUDFLARE_VECTORIZE_INDEX` - Name of Vectorize index (768 dimensions)
  - `CLOUDFLARE_D1_DATABASE_ID` - D1 database UUID
  - `CLOUDFLARE_R2_BUCKET` (optional) - R2 bucket for large content
  
- **Performance Characteristics**:
  - Storage: ~200ms per memory (including embedding generation)
  - Search: ~100ms for semantic search (5 results)
  - Batch operations: ~50ms per memory in batches of 100
  - Global latency: <100ms from most global locations

#### Migration Path
Users can migrate from existing backends using provided scripts:
```bash
# From SQLite-vec
python scripts/migrate_to_cloudflare.py --source sqlite

# From ChromaDB  
python scripts/migrate_to_cloudflare.py --source chroma
```

#### Memory Awareness Integration
- **Full Compatibility**: Cloudflare backend works seamlessly with Phase 1 and Phase 2 Memory Awareness
- **Cross-Session Intelligence**: Session tracking and conversation threading supported
- **Dynamic Context Updates**: Real-time memory loading during conversations
- **Global Performance**: Enhances Memory Awareness with <100ms global response times

#### Compatibility
- Fully backward compatible with existing SQLite-vec and ChromaDB backends
- No breaking changes to existing APIs or configurations
- Supports all existing MCP operations and features
- Compatible with all existing Memory Awareness hooks and functionality

## [6.1.1] - 2025-08-20

### 🐛 **CRITICAL BUG FIX: Memory Retrieval by Hash**

#### Fixed
- **Memory Retrieval 404 Issue**: Fixed HTTP API returning 404 errors for valid memory hashes
- **Direct Hash Lookup**: Added `get_by_hash()` method to `SqliteVecMemoryStorage` for proper content hash retrieval
- **API Endpoint Correction**: Updated `/api/memories/{content_hash}` to use direct hash lookup instead of semantic search
- **Production Deployment**: Successfully deployed fix to production servers and verified functionality

#### Technical Details
- **Root Cause**: HTTP API was incorrectly using `storage.retrieve()` (semantic search) instead of direct hash-based lookup
- **Solution**: Implemented dedicated hash lookup method that queries database directly using content hash as primary key
- **Impact**: Web dashboard memory retrieval by hash now works correctly without SSL certificate issues or false 404 responses
- **Testing**: Verified with multiple memory hashes including previously failing hash `812d361cbfd1b79a49737e6ea34e24459b4d064908e222d98af6a504aa09ff19`

#### Deployment
- Version 6.1.1 deployed to production server `10.0.1.30:8443`
- Service restart completed successfully
- Health check confirmed: Version 6.1.1 running with full functionality

## [6.1.0] - 2025-08-20

### 🚀 **MAJOR FEATURE: Intelligent Context Updates (Phase 2)**

#### Conversation-Aware Dynamic Memory Loading
This release introduces **Phase 2 of Claude Code Memory Awareness** - transforming the system from static memory injection to intelligent, real-time conversation analysis with dynamic context updates during active coding sessions.

#### Added

##### 🧠 **Dynamic Memory Loading System**
- **Real-time Topic Detection**: Analyzes conversation flow to detect significant topic shifts
- **Automatic Context Updates**: Injects relevant memories as conversations evolve naturally
- **Smart Deduplication**: Prevents re-injection of already loaded memories
- **Intelligent Rate Limiting**: 30-second cooldown and session limits prevent context overload
- **Cross-Session Intelligence**: Links related conversations across different sessions

##### 🔍 **Advanced Conversation Analysis Engine** 
- **Natural Language Processing**: Extracts topics, entities, intent, and code context from conversations
- **15+ Technical Topic Categories**: database, debugging, architecture, testing, deployment, etc.
- **Entity Recognition**: Technologies, frameworks, languages, tools (JavaScript, Python, React, etc.)
- **Intent Classification**: learning, problem-solving, development, optimization, review, planning
- **Code Context Detection**: Identifies code blocks, file paths, error messages, commands

##### 📊 **Enhanced Memory Scoring with Conversation Context**
- **Multi-Factor Relevance Algorithm**: 5-factor scoring system including conversation context (25% weight)
- **Dynamic Weight Adjustment**: Adapts scoring based on conversation analysis
- **Topic Alignment Matching**: Prioritizes memories matching current conversation topics
- **Intent-Based Scoring**: Aligns memory relevance with conversation purpose
- **Semantic Content Analysis**: Advanced content matching with conversation context

##### 🔗 **Cross-Session Intelligence & Conversation Threading**
- **Session Tracking**: Links related sessions across time with unique thread IDs  
- **Conversation Continuity**: Builds conversation threads over multiple sessions
- **Progress Context**: Connects outcomes from previous sessions to current work
- **Pattern Recognition**: Identifies recurring topics and workflow patterns
- **Historical Context**: Includes insights from recent related sessions (up to 3 sessions, 7 days back)

##### ⚡ **Performance & Reliability**
- **Efficient Processing**: <500ms response time for topic detection and memory queries
- **Scalable Architecture**: Handles 100+ active memories per session
- **Smart Debouncing**: 5-second debounce prevents rapid-fire updates
- **Resource Optimization**: Intelligent memory management and context deduplication
- **Comprehensive Testing**: 100% test pass rate (15/15 tests) with full integration coverage

#### Technical Implementation

##### Core Phase 2 Components
- **conversation-analyzer.js**: NLP engine for topic/entity/intent detection
- **topic-change.js**: Monitors conversation flow and triggers dynamic updates
- **memory-scorer.js**: Enhanced scoring with conversation context awareness  
- **session-tracker.js**: Cross-session intelligence and conversation threading
- **dynamic-context-updater.js**: Orchestrates all Phase 2 components with rate limiting

##### Configuration Enhancements
- **Phase 2 Settings**: New configuration sections for conversation analysis, dynamic updates, session tracking
- **Flexible Thresholds**: Configurable significance scores, update limits, and confidence levels
- **Feature Toggles**: Independent enable/disable for each Phase 2 capability

#### User Experience Improvements
- **Zero Cognitive Load**: Context updates happen automatically during conversations
- **Perfect Timing**: Memories appear exactly when relevant to current discussion  
- **Seamless Integration**: Works transparently during normal coding sessions
- **Progressive Learning**: Each conversation builds upon previous knowledge intelligently

#### Migration from Phase 1
- **Backward Compatible**: Phase 1 features remain fully functional
- **Additive Enhancement**: Phase 2 builds upon Phase 1 session-start memory injection
- **Unified Configuration**: Single config.json manages both Phase 1 and Phase 2 settings

## [6.0.0] - 2025-08-19

### 🧠 **MAJOR FEATURE: Claude Code Memory Awareness (Phase 1)**

#### Revolutionary Memory-Aware Development Experience
This major release introduces **automatic memory awareness for Claude Code** - a sophisticated hook system that transforms how developers interact with their project knowledge and conversation history.

#### Added

##### 🔄 **Session Lifecycle Hooks**
- **Session-Start Hook**: Automatically injects relevant memories when starting Claude Code sessions
  - Intelligent project detection supporting JavaScript, Python, Rust, Go, Java, C++, and more
  - Multi-factor memory relevance scoring with time decay, tag matching, and content analysis
  - Context-aware memory selection (up to 8 most relevant memories per session)
  - Beautiful markdown formatting for seamless context integration
  
- **Session-End Hook**: Captures and consolidates session outcomes automatically
  - Conversation analysis and intelligent summarization
  - Automatic tagging with project context and session insights
  - Long-term knowledge building through outcome storage
  - Session relationship tracking for continuity

##### 🎯 **Advanced Project Detection System**
- **Multi-Language Support**: Detects 15+ project types and frameworks
  - Package managers: `package.json`, `Cargo.toml`, `go.mod`, `requirements.txt`, `pom.xml`
  - Build systems: `Makefile`, `CMakeLists.txt`, `build.gradle`, `setup.py`
  - Configuration files: `tsconfig.json`, `pyproject.toml`, `.gitignore`
- **Git Integration**: Repository context analysis with branch and commit information
- **Framework Detection**: React, Vue, Angular, Django, Flask, Express, and more
- **Technology Stack Analysis**: Automatic identification of languages, databases, and tools

##### 🧮 **Intelligent Memory Scoring System**
- **Time Decay Algorithm**: Recent memories weighted higher with configurable decay curves
- **Tag Relevance Matching**: Project-specific and technology-specific tag scoring
- **Content Similarity Analysis**: Semantic matching with current project context
- **Memory Type Bonuses**: Prioritizes decisions, insights, and architecture notes
- **Relevance Threshold**: Only injects memories above significance threshold (>0.3)

##### 🎨 **Context Formatting & Presentation**
- **Categorized Memory Display**: Organized by Recent Insights, Key Decisions, and Project Context
- **Markdown-Rich Formatting**: Beautiful presentation with metadata, timestamps, and tags
- **Configurable Limits**: Prevents context overload with smart memory selection
- **Source Attribution**: Clear memory source tracking with content hashes

##### 💻 **Complete Installation & Testing System**
- **One-Command Installation**: `./install.sh` deploys entire system to Claude Code hooks
- **Comprehensive Test Suite**: 10 integration tests with 100% pass rate
  - Project detection testing across multiple languages
  - Memory scoring algorithm validation
  - Context formatting verification
  - Hook structure and configuration validation
  - MCP service connectivity testing
- **Configuration Management**: Production-ready config with memory service endpoints
- **Backup and Recovery**: Automatic backup of existing hooks during installation

#### Technical Architecture

##### 🏗️ **Claude Code Hooks System**
```javascript
claude-hooks/
├── core/
│   ├── session-start.js    # Automatic memory injection hook
│   └── session-end.js      # Session consolidation hook
├── utilities/
│   ├── project-detector.js  # Multi-language project detection
│   ├── memory-scorer.js     # Relevance scoring algorithms
│   └── context-formatter.js # Memory presentation utilities
├── tests/
│   └── integration-test.js  # Complete test suite (100% pass)
├── config.json             # Production configuration
└── install.sh             # One-command installation
```

##### 🔗 **MCP Memory Service Integration**
- **JSON-RPC Protocol**: Direct communication with MCP Memory Service
- **Error Handling**: Graceful degradation when memory service unavailable
- **Performance Optimization**: Efficient memory querying with result caching
- **Security**: Content hash verification and safe JSON parsing

##### 📊 **Memory Selection Algorithm**
```javascript
// Multi-factor scoring system
const relevanceScore = (
  timeDecayScore * 0.4 +         // Recent memories preferred
  tagRelevanceScore * 0.3 +      // Project-specific tags
  contentSimilarityScore * 0.2 + // Semantic matching
  memoryTypeBonusScore * 0.1     // Decision/insight bonus
);
```

#### Usage Examples

##### Automatic Session Context
```markdown
# 🧠 Relevant Memory Context

## Recent Insights (Last 7 days)
- **Database Performance Issue** - Resolved SQLite-vec query optimization (yesterday)
- **Authentication Flow** - Implemented JWT token validation in API (3 days ago)

## Key Decisions
- **Architecture Decision** - Chose React over Vue for frontend consistency (1 week ago)
- **Database Choice** - Selected PostgreSQL for production scalability (2 weeks ago)

## Project Context: mcp-memory-service
- **Language**: JavaScript, Python
- **Frameworks**: Node.js, FastAPI
- **Recent Activity**: Bug fixes, feature implementation
```

##### Session Outcome Storage
```markdown
Session consolidated and stored with tags:
- mcp-memory-service, claude-code-session
- bug-fix, performance-optimization
- javascript, api-development
Content hash: abc123...def456
```

#### Benefits & Impact

##### 🚀 **Productivity Enhancements**
- **Zero Cognitive Load**: Memory context appears automatically without user intervention
- **Perfect Continuity**: Never lose track of decisions, insights, or architectural choices
- **Intelligent Context**: Only relevant memories shown, preventing information overload
- **Session Learning**: Each coding session builds upon previous knowledge automatically

##### 🧠 **Memory-Aware Development**
- **Decision Tracking**: Automatic capture of technical decisions and rationale
- **Knowledge Building**: Progressive accumulation of project understanding
- **Context Preservation**: Important insights never get lost between sessions
- **Team Knowledge Sharing**: Shareable memory context across team members

##### ⚡ **Performance Optimized**
- **Fast Startup**: Memory injection adds <2 seconds to session startup
- **Smart Caching**: Efficient memory retrieval with minimal API calls
- **Configurable Limits**: Prevents memory service overload with request throttling
- **Graceful Fallback**: Works with or without memory service availability

#### Migration & Compatibility

##### 🔄 **Seamless Integration**
- **Non-Intrusive**: Works alongside existing Claude Code workflows
- **Backward Compatible**: No changes required to existing development processes
- **Optional Feature**: Can be enabled/disabled per project or globally
- **Multi-Environment**: Works with local, remote, and distributed memory services

##### 📋 **Installation Requirements**
- Claude Code CLI installed and configured
- MCP Memory Service running (local or remote)
- Node.js environment for hook execution
- Git repository for optimal project detection

#### Roadmap Integration

This release completes **Phase 1** of the Memory Awareness Enhancement Roadmap (Issue #14):
- ✅ Session startup hooks with automatic memory injection
- ✅ Project-aware memory selection algorithms  
- ✅ Context formatting and injection utilities
- ✅ Comprehensive testing and installation system
- ✅ Production-ready configuration and deployment

**Next Phase**: Dynamic memory loading, cross-session intelligence, and advanced consolidation features.

#### Breaking Changes
None - This is a purely additive feature that enhances existing Claude Code functionality.

---

## [5.2.0] - 2025-08-18

### 🚀 **New Features**

#### Command Line Interface (CLI)
- **Comprehensive CLI**: Added `memory` command with subcommands for document ingestion and management
- **Document Ingestion Commands**: 
  - `memory ingest-document <file>` - Ingest single documents with customizable chunking
  - `memory ingest-directory <path>` - Batch process entire directories 
  - `memory list-formats` - Show all supported document formats
- **Management Commands**:
  - `memory server` - Start the MCP server (replaces old `memory` command)
  - `memory status` - Show service status and statistics
- **Advanced Options**: Tags, chunk sizing, storage backend selection, verbose output, dry-run mode
- **Progress Tracking**: Real-time progress bars and detailed error reporting
- **Cross-Platform**: Works on Windows, macOS, and Linux with proper path handling

#### Enhanced Document Processing
- **Click Framework**: Professional CLI with help system and tab completion support
- **Async Operations**: Non-blocking document processing with proper resource management
- **Error Recovery**: Graceful handling of processing errors with detailed diagnostics
- **Batch Limits**: Configurable file limits and extension filtering for large directories

**New Dependencies**: `click>=8.0.0` for CLI framework

**Examples**:
```bash
memory ingest-document manual.pdf --tags documentation,manual --verbose
memory ingest-directory ./docs --recursive --max-files 50
memory list-formats
memory status
```

**Backward Compatibility**: Old `memory` server command now available as `memory server` and `memory-server`

## [5.1.0] - 2025-08-18

### 🚀 **New Features**

#### Remote ChromaDB Support
- **Enterprise-Ready**: Connect to remote ChromaDB servers, Chroma Cloud, or self-hosted instances
- **HttpClient Implementation**: Full support for remote ChromaDB connectivity
- **Authentication**: API key authentication via `X_CHROMA_TOKEN` header
- **SSL/HTTPS Support**: Secure connections to remote ChromaDB servers
- **Custom Collections**: Specify collection names for multi-tenant deployments

**New Environment Variables:**
- `MCP_MEMORY_CHROMADB_HOST`: Remote server hostname (enables remote mode)
- `MCP_MEMORY_CHROMADB_PORT`: Server port (default: 8000)
- `MCP_MEMORY_CHROMADB_SSL`: Use HTTPS ('true'/'false')
- `MCP_MEMORY_CHROMADB_API_KEY`: Authentication token
- `MCP_MEMORY_COLLECTION_NAME`: Custom collection name (default: 'memory_collection')

**Perfect Timing**: Arrives just as Chroma Cloud launches Q1 2025 (early access available)

**Resolves**: #36 (Remote ChromaDB support request)

## [5.0.5] - 2025-08-18

### 🐛 **Bug Fixes**

#### Code Quality & Future Compatibility
- **Fixed datetime deprecation warnings**: Replaced all `datetime.utcnow()` usage with `datetime.now(timezone.utc)`
  - Updated `src/mcp_memory_service/web/api/health.py` (2 occurrences)
  - Updated `src/mcp_memory_service/web/sse.py` (3 occurrences)
  - Eliminates deprecation warnings in Python 3.12+
  - Future-proof timezone-aware datetime handling

### 🎨 **UI Improvements**

#### Dashboard Mobile Responsiveness
- **Enhanced mobile UX**: Added responsive design for action buttons
  - Buttons now stack vertically on screens < 768px width
  - Improved touch-friendly spacing and sizing
  - Better mobile experience for API documentation links
  - Maintains desktop horizontal layout on larger screens

**Issues Resolved**: #68 (Code Quality & Deprecation Fixes), #80 (Dashboard Mobile Responsiveness)

## [5.0.4] - 2025-08-18

### 🐛 **Critical Bug Fixes**

#### SQLite-vec Embedding Model Loading
- **Fixed UnboundLocalError**: Removed redundant `import os` statement at line 285 in `sqlite_vec.py`
  - Variable shadowing prevented ONNX embedding model initialization
  - Caused "cannot access local variable 'os'" error in production
  - Restored full embedding functionality for memory storage

#### Docker HTTP Server Support
- **Fixed Missing Files**: Added `run_server.py` to Docker image (reported by Joe Esposito)
  - HTTP server wouldn't start without this critical file
  - Now properly copied in Dockerfile for HTTP/API mode
- **Fixed Python Path**: Corrected `PYTHONPATH` from `/app` to `/app/src`
  - Modules couldn't be found with incorrect path
  - Essential for both MCP and HTTP modes

### 🚀 **Major Docker Improvements**

#### Simplified Docker Architecture
- **Reduced Complexity by 60%**: Consolidated from 4 confusing compose files to 2 clear options
  - `docker-compose.yml` for MCP protocol mode (Claude Desktop, VS Code)
  - `docker-compose.http.yml` for HTTP/API mode (REST API, Web Dashboard)
- **Unified Entrypoint**: Created single smart entrypoint script
  - Auto-detects mode from `MCP_MODE` environment variable
  - Eliminates confusion about which script to use
- **Pre-download Models**: Embedding models now downloaded during Docker build
  - Prevents runtime failures from network/DNS issues
  - Makes containers fully self-contained
  - Faster startup times

#### Deprecated Docker Files
- Marked 4 redundant Docker files as deprecated:
  - `docker-compose.standalone.yml` → Use `docker-compose.http.yml`
  - `docker-compose.uv.yml` → UV now built into main Dockerfile
  - `docker-compose.pythonpath.yml` → Fix applied to main Dockerfile
  - `docker-entrypoint-persistent.sh` → Replaced by unified entrypoint

### 📝 **Documentation**

#### Docker Documentation Overhaul
- **Added Docker README**: Clear instructions for both MCP and HTTP modes
- **Created DEPRECATED.md**: Migration guide for old Docker setups
- **Added Test Script**: `test-docker-modes.sh` to verify both modes work
- **Troubleshooting Guide**: Added comprehensive debugging section to CLAUDE.md
  - Web frontend debugging (CSS/format string conflicts)
  - Cache clearing procedures
  - Environment reset steps
  - Backend method compatibility

### 🙏 **Credits**
- Special thanks to **Joe Esposito** for identifying and helping fix critical Docker issues

## [5.0.3] - 2025-08-17

### 🐛 **Bug Fixes**

#### SQLite-vec Backend Method Support
- **Fixed Missing Method**: Added `search_by_tags` method to SQLite-vec backend
  - Web API was calling `search_by_tags` (plural) but backend only had `search_by_tag` (singular)
  - This caused 500 errors when using tag-based search via HTTP/MCP endpoints
  - New method supports both AND/OR operations for tag matching
  - Fixes network distribution and memory retrieval functionality

### 🚀 **Enhancements**

#### Version Information in Health Checks
- **Added Version Field**: All health endpoints now return service version
  - Basic health endpoint (`/api/health`) includes version field
  - Detailed health endpoint (`/api/health/detailed`) includes version field
  - MCP `check_database_health` tool returns version in response
  - Enables easier debugging and version tracking across deployments

### 🚀 **New Features**

#### Memory Distribution and Network Sharing
- **Export Tool**: Added `scripts/export_distributable_memories.sh` for memory export
  - Export memories tagged with `distributable-reference` for team sharing
  - JSON format for easy import to other MCP instances
  - Support for cross-network memory synchronization
- **Personalized CLAUDE.md Generator**: Added `scripts/generate_personalized_claude_md.sh`
  - Generate CLAUDE.md with embedded memory service endpoints
  - Customize for different network deployments
  - Include memory retrieval commands for each environment
- **Memory Context Templates**: Added `prompts/load_memory_context.md`
  - Ready-to-use prompts for loading project context
  - Quick retrieval commands for Claude Code sessions
  - Network distribution instructions

### 📝 **Documentation**

#### Network Distribution Updates
- **Fixed Memory Retrieval Commands**: Updated scripts to use working API methods
  - Changed from non-existent `search_by_tag` to `retrieve_memory` for current deployments
  - Updated prompt templates and distribution scripts
  - Improved error handling for memory context loading
- **CLAUDE.md Enhancements**: Added optional memory context section
  - Instructions for setting up local memory service integration
  - Guidelines for creating CLAUDE_MEMORY.md (git-ignored) for local configurations
  - Best practices for memory management and quarterly reviews

## [5.0.2] - 2025-08-17

### 🚀 **New Features**

#### ONNX Runtime Support
- **PyTorch-free operation**: True PyTorch-free embedding generation using ONNX Runtime
  - Reduced dependencies (~500MB less disk space without PyTorch)
  - Faster startup with pre-optimized ONNX models  
  - Automatic fallback to SentenceTransformers when needed
  - Compatible with the same `all-MiniLM-L6-v2` model embeddings
  - Enable with `export MCP_MEMORY_USE_ONNX=true`

### 🐛 **Bug Fixes**

#### SQLite-vec Consolidation Support (Issue #84)
- **Missing Methods Fixed**: Added all required methods for consolidation support
  - Implemented `get_memories_by_time_range()` for time-based queries
  - Added `get_memory_connections()` for relationship tracking statistics
  - Added `get_access_patterns()` for access pattern analysis
  - Added `get_all_memories()` method with proper SQL implementation

#### Installer Accuracy  
- **False ONNX Claims**: Fixed misleading installer messages about ONNX support
  - Removed false claims about "ONNX runtime for embeddings" when no implementation existed
  - Updated installer messages to accurately reflect capabilities
  - Now actually implements the ONNX support that was previously claimed

#### Docker Configuration
- **SQLite-vec Defaults**: Updated Docker configuration to reflect SQLite-vec as default backend
  - Updated `Dockerfile` environment variables to use `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`
  - Changed paths from `/app/chroma_db` to `/app/sqlite_db` 
  - Updated `docker-compose.yml` with SQLite-vec configuration
  - Fixed volume mounts and environment variables

### 📝 **Documentation**

#### Enhanced README
- **ONNX Feature Documentation**: Added comprehensive ONNX Runtime feature section
- **Installation Updates**: Updated installation instructions with ONNX dependencies
- **Feature Visibility**: Highlighted ONNX support in main features list

#### Technical Implementation
- **New Module**: Created `src/mcp_memory_service/embeddings/onnx_embeddings.py`
  - Complete ONNX embedding implementation based on ChromaDB's ONNXMiniLM_L6_V2
  - Automatic model downloading and caching
  - Hardware-aware provider selection (CPU, CUDA, DirectML, CoreML)
  - Error handling with graceful fallbacks

- **Enhanced Configuration**: Added ONNX configuration support in `config.py`
  - `USE_ONNX` configuration option  
  - ONNX model cache directory management
  - Environment variable support for `MCP_MEMORY_USE_ONNX`

### Technical Notes
- Full backward compatibility maintained for existing SQLite-vec users
- ONNX support is optional and falls back gracefully to SentenceTransformers
- All consolidation methods implemented with proper error handling
- Docker images now correctly reflect the SQLite-vec default backend

This release resolves all issues reported in GitHub issue #84, implementing true ONNX support and completing the SQLite-vec consolidation feature set.
## [6.2.0] - 2025-08-20

### 🚀 **MAJOR FEATURE: Native Cloudflare Backend Integration**

This major release introduces native Cloudflare integration as a third storage backend option alongside SQLite-vec and ChromaDB, providing global distribution, automatic scaling, and enterprise-grade infrastructure, integrated with the existing Memory Awareness system.

#### Added
- **Native Cloudflare Backend Support**: Complete implementation using Cloudflare's edge computing platform
  - **Vectorize**: 768-dimensional vector storage with cosine similarity for semantic search
  - **D1 Database**: SQLite-compatible database for metadata storage
  - **Workers AI**: Embedding generation using @cf/baai/bge-base-en-v1.5 model
  - **R2 Storage** (optional): Object storage for large content exceeding 1MB threshold
  
- **Implementation Files**:
  - `src/mcp_memory_service/storage/cloudflare.py` - Complete CloudflareStorage implementation (740 lines)
  - `scripts/migrate_to_cloudflare.py` - Migration tool for existing SQLite-vec and ChromaDB users
  - `scripts/test_cloudflare_backend.py` - Comprehensive test suite with automated validation
  - `scripts/setup_cloudflare_resources.py` - Automated Cloudflare resource provisioning
  - `docs/cloudflare-setup.md` - Complete setup, configuration, and troubleshooting guide
  - `tests/unit/test_cloudflare_storage.py` - 15 unit tests for CloudflareStorage class

- **Features**:
  - Automatic retry logic with exponential backoff for API rate limiting
  - Connection pooling for optimal HTTP performance
  - NDJSON format support for Vectorize v2 API endpoints
  - LRU caching (1000 entries) for embedding deduplication
  - Batch operations support for efficient data processing
  - Global distribution with <100ms latency from most locations
  - Pay-per-use pricing model with no upfront costs

#### Changed
- Updated `config.py` to include 'cloudflare' in SUPPORTED_BACKENDS
- Enhanced server initialization in `mcp_server.py` to support Cloudflare backend
- Updated storage factory in `storage/__init__.py` to create CloudflareStorage instances
- Consolidated documentation, removing redundant setup files

#### Technical Details
- **Environment Variables**:
  - `MCP_MEMORY_STORAGE_BACKEND=cloudflare` - Activate Cloudflare backend
  - `CLOUDFLARE_API_TOKEN` - API token with Vectorize, D1, Workers AI permissions
  - `CLOUDFLARE_ACCOUNT_ID` - Cloudflare account identifier
  - `CLOUDFLARE_VECTORIZE_INDEX` - Name of Vectorize index (768 dimensions)
  - `CLOUDFLARE_D1_DATABASE_ID` - D1 database UUID
  - `CLOUDFLARE_R2_BUCKET` (optional) - R2 bucket for large content
  
- **Performance Characteristics**:
  - Storage: ~200ms per memory (including embedding generation)
  - Search: ~100ms for semantic search (5 results)
  - Batch operations: ~50ms per memory in batches of 100
  - Global latency: <100ms from most global locations

#### Migration Path
Users can migrate from existing backends using provided scripts:
```bash
# From SQLite-vec
python scripts/migrate_to_cloudflare.py --source sqlite

# From ChromaDB  
python scripts/migrate_to_cloudflare.py --source chroma
```

#### Memory Awareness Integration
- **Full Compatibility**: Cloudflare backend works seamlessly with Phase 1 and Phase 2 Memory Awareness
- **Cross-Session Intelligence**: Session tracking and conversation threading supported
- **Dynamic Context Updates**: Real-time memory loading during conversations
- **Global Performance**: Enhances Memory Awareness with <100ms global response times

#### Compatibility
- Fully backward compatible with existing SQLite-vec and ChromaDB backends
- No breaking changes to existing APIs or configurations
- Supports all existing MCP operations and features
- Compatible with all existing Memory Awareness hooks and functionality

## [5.0.1] - 2025-08-15

### 🐛 **Critical Migration Fixes**

This patch release addresses critical issues in the v5.0.0 ChromaDB to SQLite-vec migration process reported in [Issue #83](https://github.com/doobidoo/mcp-memory-service/issues/83).

#### Fixed
- **Custom Data Paths**: Migration scripts now properly support custom ChromaDB locations via CLI arguments and environment variables
  - Added `--chroma-path` and `--sqlite-path` arguments to migration scripts
  - Support for `MCP_MEMORY_CHROMA_PATH` and `MCP_MEMORY_SQLITE_PATH` environment variables
  - Fixed hardcoded path assumptions that ignored user configurations

- **Content Hash Generation**: Fixed critical bug where ChromaDB document IDs were used instead of proper SHA256 hashes
  - Now generates correct content hashes using SHA256 algorithm
  - Prevents "NOT NULL constraint failed" errors during migration
  - Ensures data integrity and proper deduplication

- **Tag Format Corruption**: Fixed issue where 60% of tags became malformed during migration
  - Improved tag validation and format conversion
  - Handles comma-separated strings, arrays, and single tags correctly
  - Prevents array syntax from being stored as strings

- **Migration Progress**: Added progress indicators and better error reporting
  - Shows percentage completion during migration
  - Batch processing with configurable batch size
  - Verbose mode for detailed debugging
  - Clear error messages for troubleshooting

#### Added
- **Enhanced Migration Script** (`scripts/migrate_v5_enhanced.py`):
  - Comprehensive migration tool with all fixes
  - Dry-run mode for testing migrations
  - Transaction-based migration with rollback support
  - Progress bars with `tqdm` support
  - JSON backup creation
  - Automatic path detection for common installations

- **Migration Validator** (`scripts/validate_migration.py`):
  - Validates migrated SQLite database integrity
  - Checks for missing fields and corrupted data
  - Compares with original ChromaDB data
  - Generates detailed validation report
  - Identifies specific issues like hash mismatches and tag corruption

- **Comprehensive Documentation**:
  - Updated migration guide with troubleshooting section
  - Documented all known v5.0.0 issues and solutions
  - Added recovery procedures for failed migrations
  - Migration best practices and validation steps

#### Enhanced
- **Original Migration Script** (`scripts/migrate_chroma_to_sqlite.py`):
  - Added CLI argument support for custom paths
  - Fixed content hash generation
  - Improved tag handling
  - Better duplicate detection
  - Progress percentage display

#### Documentation
- **Migration Troubleshooting Guide**: Added comprehensive troubleshooting section covering:
  - Custom data location issues
  - Content hash errors
  - Tag corruption problems
  - Migration hangs
  - Dependency conflicts
  - Recovery procedures

## [5.0.0] - 2025-08-15

### ⚠️ **BREAKING CHANGES**

#### ChromaDB Deprecation
- **DEPRECATED**: ChromaDB backend is now deprecated and will be removed in v6.0.0
- **DEFAULT CHANGE**: SQLite-vec is now the default storage backend (previously ChromaDB)
- **MIGRATION REQUIRED**: Users with existing ChromaDB data should migrate to SQLite-vec
  - Run `python scripts/migrate_to_sqlite_vec.py` to migrate your data
  - Migration is one-way only (ChromaDB → SQLite-vec)
  - Backup your data before migration

#### Why This Change?
- **Network Issues**: ChromaDB requires downloading models from Hugging Face, causing frequent failures
- **Performance**: SQLite-vec is 10x faster at startup (2-3 seconds vs 15-30 seconds)
- **Resource Usage**: SQLite-vec uses 75% less memory than ChromaDB
- **Reliability**: Zero network dependencies means no download failures or connection issues
- **Simplicity**: Single-file database, easier backup and portability

#### Changed
- **Default Backend**: Changed from ChromaDB to SQLite-vec in all configurations
- **Installation**: `install.py` now installs SQLite-vec dependencies by default
- **Documentation**: Updated all guides to recommend SQLite-vec as primary backend
- **Warnings**: Added deprecation warnings when ChromaDB backend is used
- **Migration Prompts**: Server now prompts for migration when ChromaDB data is detected

#### Migration Guide
1. **Backup your data**: `python scripts/create_backup.py`
2. **Run migration**: `python scripts/migrate_to_sqlite_vec.py`
3. **Update configuration**: Set `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`
4. **Restart server**: Your memories are now in SQLite-vec format

#### Backward Compatibility
- ChromaDB backend still functions but displays deprecation warnings
- Existing ChromaDB installations continue to work until v6.0.0
- Migration tools provided for smooth transition
- All APIs remain unchanged regardless of backend

## [4.6.1] - 2025-08-14

### 🐛 **Bug Fixes**

#### Fixed
- **Export/Import Script Database Path Detection**: Fixed critical bug in memory export and import scripts
  - Export script now properly respects `SQLITE_VEC_PATH` configuration from `config.py`
  - Import script now properly respects `SQLITE_VEC_PATH` configuration from `config.py`
  - Scripts now use environment variables like `MCP_MEMORY_SQLITE_PATH` correctly
  - Fixed issue where export/import would use wrong database path, missing actual memories
  - Added support for custom database paths via `--db-path` argument
  - Ensures export captures all memories from the configured database location
  - Ensures import writes to the correct configured database location

#### Enhanced
- **Export/Import Script Configuration**: Improved database path detection logic
  - Falls back gracefully when SQLite-vec backend is not configured
  - Maintains compatibility with different storage backend configurations
  - Added proper imports for configuration variables

#### Technical Details
- Modified `scripts/sync/export_memories.py` to use `SQLITE_VEC_PATH` instead of `BASE_DIR`
- Modified `scripts/sync/import_memories.py` to use `SQLITE_VEC_PATH` instead of `BASE_DIR`
- Updated `get_default_db_path()` functions in both scripts to check storage backend configuration
- Added version bump to exporter metadata for tracking
- Added `get_all_memories()` method to `SqliteVecMemoryStorage` for proper export functionality

## [4.6.0] - 2025-08-14

### ✨ **New Features**

#### Added
- **Custom SSL Certificate Support**: Added environment variable configuration for SSL certificates
  - New `MCP_SSL_CERT_FILE` environment variable for custom certificate path
  - New `MCP_SSL_KEY_FILE` environment variable for custom private key path
  - Maintains backward compatibility with self-signed certificate generation
  - Enables production deployments with proper SSL certificates (e.g., mkcert, Let's Encrypt)

#### Enhanced
- **HTTPS Server Configuration**: Improved certificate validation and error handling
  - Certificate file existence validation before server startup
  - Clear error messages for missing certificate files
  - Logging improvements for certificate source identification

#### Documentation
- **SSL/TLS Setup Guide**: Added comprehensive SSL configuration documentation
  - Integration guide for [mkcert](https://github.com/FiloSottile/mkcert) for local development
  - Example HTTPS startup script template
  - Client CA installation instructions for multiple operating systems

## [4.5.2] - 2025-08-14

### 🐛 **Bug Fixes & Documentation**

#### Fixed
- **JSON Protocol Compatibility**: Resolved debug output contaminating MCP JSON-RPC communication
  - Fixed unconditional debug print statements causing "Unexpected token" errors in Claude Desktop logs
  - Added client detection checks to `TOOL CALL INTERCEPTED` and `Processing tool` debug messages
  - Ensures clean JSON-only output for Claude Desktop while preserving debug output for LM Studio

#### Enhanced
- **Universal README Documentation**: Transformed from Claude Desktop-specific to universal AI client focus
  - Updated opening description to emphasize compatibility with "AI applications and development environments"
  - Added prominent compatibility badges for Cursor, WindSurf, LM Studio, Zed, and other AI clients
  - Moved comprehensive client compatibility table to prominent position in documentation
  - Expanded client support details for 13+ different AI applications and IDEs
  - Added multi-client benefits section highlighting cross-tool memory sharing capabilities
  - Updated examples and Docker configurations to be client-agnostic

#### Documentation
- **Improved Client Visibility**: Enhanced documentation structure for broader MCP ecosystem appeal
- **Balanced Examples**: Updated API examples to focus on universal MCP access rather than specific clients
- **Clear Compatibility Matrix**: Detailed status and configuration for each supported AI client

## [4.5.1] - 2025-08-13

### 🎯 **Enhanced Multi-Client Support**

#### Added
- **Intelligent Client Detection**: Automatic detection of MCP client type
  - Detects Claude Desktop, LM Studio, and other MCP clients
  - Uses process inspection and environment variables for robust detection
  - Falls back to strict JSON mode for unknown clients
  
- **Client-Aware Logging System**: Optimized output for different MCP clients
  - **Claude Desktop Mode**: Pure JSON-RPC protocol compliance
    - Suppresses diagnostic output to maintain clean JSON communication
    - Routes only WARNING/ERROR messages to stderr
    - Ensures maximum compatibility with Claude's strict parsing
  - **LM Studio Mode**: Enhanced diagnostic experience
    - Shows system diagnostics, dependency checks, and initialization status
    - Provides detailed feedback for troubleshooting
    - Maintains full INFO/DEBUG output to stdout

#### Enhanced
- **Improved Stability**: All diagnostic output is now conditional based on client type
  - 15+ print statements updated with client-aware logic
  - System diagnostics, dependency checks, and initialization messages
  - Docker mode detection and standalone mode indicators

#### Technical Details
- Added `psutil` dependency for process-based client detection
- Implemented `DualStreamHandler` with client-aware routing
- Environment variable support: `CLAUDE_DESKTOP=1` or `LM_STUDIO=1` for manual override
- Maintains full backward compatibility with existing integrations

## [4.5.0] - 2025-08-12

### 🔄 **Database Synchronization System**

#### Added
- **Multi-Node Database Sync**: Complete Litestream-based synchronization for SQLite-vec databases
  - **JSON Export/Import**: Preserve timestamps and metadata across database migrations
  - **Litestream Integration**: Real-time database replication with conflict resolution
  - **3-Node Architecture**: Central server with replica nodes for distributed workflows
  - **Deduplication Logic**: Content hash-based duplicate prevention during imports
  - **Source Tracking**: Automatic tagging to identify memory origin machines

- **New Sync Module**: `src/mcp_memory_service/sync/`
  - `MemoryExporter`: Export memories to JSON with full metadata preservation
  - `MemoryImporter`: Import with intelligent deduplication and source tracking
  - `LitestreamManager`: Automated Litestream configuration and management

- **Sync Scripts Suite**: `scripts/sync/`
  - `export_memories.py`: Platform-aware memory export utility
  - `import_memories.py`: Central server import with merge statistics
  - `README.md`: Comprehensive usage documentation

#### Enhanced
- **Migration Tools**: Extended existing migration scripts to support sync workflows
- **Backup Integration**: Sync capabilities integrate with existing backup system
- **Health Monitoring**: Added sync status to health endpoints and monitoring

#### Documentation
- **Complete Sync Guide**: `docs/deployment/database-synchronization.md`
- **Technical Architecture**: Detailed setup and troubleshooting documentation
- **Migration Examples**: Updated migration documentation with sync procedures

#### Use Cases
- **Multi-Device Workflows**: Keep memories synchronized across Windows, macOS, and server
- **Team Collaboration**: Shared memory databases with individual client access
- **Backup and Recovery**: Real-time replication provides instant backup capability
- **Offline Capability**: Local replicas work offline, sync when reconnected

This release enables seamless database synchronization across multiple machines while preserving all memory metadata, timestamps, and source attribution.

## [4.4.0] - 2025-08-12

### 🚀 **Backup System Enhancements**

#### Added
- **SQLite-vec Backup Support**: Enhanced MCP backup system to fully support SQLite-vec backend
  - **Multi-Backend Support**: `dashboard_create_backup` now handles both ChromaDB and SQLite-vec databases
  - **Complete File Coverage**: Backs up main database, WAL, and SHM files for data integrity
  - **Metadata Generation**: Creates comprehensive backup metadata with size, file count, and backend info
  - **Error Handling**: Robust error handling and validation during backup operations

- **Automated Backup Infrastructure**: Complete automation solution for production deployments
  - **Backup Script**: `scripts/backup_sqlite_vec.sh` with 7-day retention policy
  - **Cron Setup**: `scripts/setup_backup_cron.sh` for easy daily backup scheduling
  - **Metadata Tracking**: JSON metadata files with backup timestamp, size, and source information
  - **Automatic Cleanup**: Old backup removal to prevent disk space issues

#### Enhanced
- **Backup Reliability**: Improved backup system architecture for production use
  - **Backend Detection**: Automatic detection and appropriate handling of storage backend
  - **File Integrity**: Proper handling of SQLite WAL mode with transaction log files
  - **Consistent Naming**: Standardized backup naming with timestamps
  - **Validation**: Pre-backup validation of source files and post-backup verification

#### Technical Details
- **Storage Backend**: Seamless support for both `sqlite_vec` and `chroma` backends
- **File Operations**: Safe file copying with proper permission handling
- **Scheduling**: Cron integration for hands-off automated backups
- **Monitoring**: Backup logs and status tracking for operational visibility

## [4.3.5] - 2025-08-12

### 🔧 **Critical Fix: Client Hostname Capture**

#### Fixed
- **Architecture Correction**: Fixed hostname capture to identify CLIENT machine instead of server machine
  - **Before**: Always captured server hostname (`narrowbox`) regardless of client
  - **After**: Prioritizes client-provided hostname, fallback to server hostname
  - **HTTP API**: Supports `client_hostname` in request body + `X-Client-Hostname` header
  - **MCP Server**: Added `client_hostname` parameter to store_memory tool
  - **Legacy Server**: Supports `client_hostname` in arguments dictionary
  - **Priority Order**: request body > HTTP header > server hostname fallback

#### Changed
- **Client Detection Logic**: Updated all three interfaces with proper client hostname detection
  - `memories.py`: Added Request parameter and header/body hostname extraction
  - `mcp_server.py`: Added client_hostname parameter with priority logic
  - `server.py`: Added client_hostname argument extraction with fallback
  - Maintains backward compatibility when `MCP_MEMORY_INCLUDE_HOSTNAME=false`

#### Documentation
- **Command Templates**: Updated repository templates with client hostname detection guidance
- **API Documentation**: Enhanced descriptions to clarify client vs server hostname capture
- **Test Documentation**: Added comprehensive test scenarios and verification steps

#### Technical Impact
- ✅ **Multi-device workflows**: Memories now correctly identify originating client machine
- ✅ **Audit trails**: Proper source attribution across different client connections
- ✅ **Remote deployments**: Works correctly when client and server are different machines
- ✅ **Backward compatible**: No breaking changes, respects environment variable setting

## [4.3.4] - 2025-08-12

### 🔧 **Optional Machine Identification**

#### Added
- **Environment-Controlled Machine Tracking**: Made machine identification optional via environment variable
  - New environment variable: `MCP_MEMORY_INCLUDE_HOSTNAME` (default: `false`)
  - When enabled, automatically adds machine hostname to all stored memories
  - Adds both `source:hostname` tag and hostname metadata field
  - Supports all interfaces: MCP server, HTTP API, and legacy server
  - Privacy-focused: disabled by default, enables multi-device workflows when needed

#### Changed
- **Memory Storage Enhancement**: All memory storage operations now support optional machine tracking
  - Updated `mcp_server.py` store_memory function with hostname logic
  - Enhanced HTTP API `/memories` endpoint with machine identification
  - Updated legacy `server.py` with consistent hostname tracking
  - Maintains backward compatibility with existing memory operations

#### Documentation
- **CLAUDE.md Updated**: Added `MCP_MEMORY_INCLUDE_HOSTNAME` environment variable documentation
- **Configuration Guide**: Explains optional hostname tracking for audit trails and multi-device setups

## [4.3.3] - 2025-08-12

### 🎯 **Claude Code Command Templates Enhancement**

#### Added
- **Machine Source Tracking**: All memory storage commands now automatically include machine hostname as a tag
  - Enables filtering memories by originating machine (e.g., `source:machine-name`)
  - Adds hostname to both tags and metadata for redundancy
  - Supports multi-device workflows and audit trails

#### Changed
- **Command Templates Updated**: All five memory command templates enhanced with:
  - Updated to use generic HTTPS endpoint (`https://memory.local:8443/`)
  - Proper API endpoint paths documented for all operations
  - Auto-save functionality without confirmation prompts
  - curl with `-k` flag for HTTPS self-signed certificates
  - Machine hostname tracking integrated throughout

#### Documentation
- `memory-store.md`: Added machine context and HTTPS configuration
- `memory-health.md`: Updated with specific health API endpoints
- `memory-search.md`: Added all search API endpoints and machine source search
- `memory-context.md`: Integrated machine tracking for session captures
- `memory-recall.md`: Updated with API endpoints and time parser limitations

## [4.3.2] - 2025-08-11

### 🎯 **Repository Organization & PyTorch Download Fix**

#### Fixed
- **PyTorch Repeated Downloads**: Completely resolved Claude Desktop downloading PyTorch (230MB+) on every startup
  - Root cause: UV package manager isolation prevented offline environment variables from taking effect
  - Solution: Created `scripts/memory_offline.py` launcher that sets offline mode BEFORE any imports
  - Updated Claude Desktop config to use Python directly instead of UV isolation
  - Added comprehensive offline mode configuration for HuggingFace transformers

- **Environment Variable Inheritance**: Fixed UV environment isolation issues
  - Implemented direct Python execution bypass for Claude Desktop integration
  - Added code-level offline setup in `src/mcp_memory_service/__init__.py` as fallback
  - Ensured cached model usage without network requests

#### Changed
- **Repository Structure**: Major cleanup and reorganization of root directory
  - Moved documentation files to appropriate `/docs` subdirectories
  - Consolidated guides in `docs/guides/`, technical docs in `docs/technical/`
  - Moved deployment guides to `docs/deployment/`, installation to `docs/installation/`
  - Removed obsolete debug scripts and development notes
  - Moved service management scripts to `/scripts` directory

- **Documentation Organization**: Improved logical hierarchy
  - Claude Code compatibility → `docs/guides/claude-code-compatibility.md`
  - Setup guides → `docs/installation/` and `docs/guides/`
  - Technical documentation → `docs/technical/`
  - Integration guides → `docs/integrations/`

#### Technical Details
- **Offline Mode Implementation**: `scripts/memory_offline.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before ML library imports
- **Config Optimization**: Updated Claude Desktop config templates for both Windows and general use
- **Cache Management**: Proper Windows cache path configuration for sentence-transformers and HuggingFace

#### Impact
- ✅ **Eliminated 230MB PyTorch downloads** - Startup time reduced from ~60s to ~3s
- ✅ **Professional repository structure** - Clean root directory with logical documentation hierarchy  
- ✅ **Improved maintainability** - Consolidated scripts and removed redundant files
- ✅ **Enhanced user experience** - No more frustrating download delays in Claude Desktop

This release resolves the persistent PyTorch download issue that affected Windows users and establishes a clean, professional repository structure suitable for enterprise deployment.

## [4.3.1] - 2025-08-11

### 🔧 **Critical Windows Installation Fixes**

#### Fixed
- **PyTorch-DirectML Compatibility**: Resolved major installation issues on Windows 11
  - Fixed installer attempting to install incompatible PyTorch 2.5.1 over working 2.4.1+DirectML setup
  - Added smart compatibility checking: PyTorch 2.4.x works with DirectML, 2.5.x doesn't
  - Enhanced `install_pytorch_windows()` to preserve existing compatible installations
  - Only installs torch-directml if PyTorch 2.4.1 exists without DirectML extensions
  
- **Corrupted Virtual Environment Recovery**: Fixed "module 'torch' has no attribute 'version'" errors
  - Implemented complete cleanup of corrupted `~orch` and `functorch` directories  
  - Added robust uninstall and reinstall process for broken PyTorch installations
  - Restored proper torch.version attribute functionality
  
- **Windows 11 Detection**: Fixed incorrect OS identification
  - Implemented registry-based Windows 11 detection using build numbers (≥22000)
  - Replaced unreliable platform detection with accurate registry lookups
  - Added system info caching to prevent duplicate detection calls

- **Installation Logging Improvements**: Enhanced installer feedback and debugging
  - Created built-in DualOutput logging system with UTF-8 encoding
  - Fixed character encoding issues in installation logs
  - Added comprehensive logging for PyTorch compatibility decisions

#### Changed
- **Installation Intelligence**: Installer now preserves working DirectML setups instead of force-upgrading
- **Error Prevention**: Added extensive pre-checks to prevent corrupted package installations
- **User Experience**: Clear messaging about PyTorch version compatibility and preservation decisions

#### Technical Details
- Enhanced PyTorch version detection and compatibility matrix
- Smart preservation of PyTorch 2.4.1 + torch-directml 0.2.5.dev240914 combinations
- Automatic cleanup of corrupted package directories during installation recovery
- Registry-based Windows version detection via `SOFTWARE\Microsoft\Windows NT\CurrentVersion`

This release resolves critical Windows installation failures that prevented successful PyTorch-DirectML setup, ensuring reliable DirectML acceleration on Windows 11 systems.

## [4.3.0] - 2025-08-10

### ⚡ **Developer Experience Improvements**

#### Added
- **Automated uv.lock Conflict Resolution**: Eliminates manual merge conflict resolution
  - Custom git merge driver automatically resolves `uv.lock` conflicts
  - Auto-runs `uv sync` after conflict resolution to ensure consistency
  - One-time setup script for contributors: `./scripts/setup-git-merge-drivers.sh`
  - Comprehensive documentation in README.md and CLAUDE.md

#### Technical
- Added `.gitattributes` configuration for `uv.lock` merge handling
- Created `scripts/uv-lock-merge.sh` custom merge driver script
- Added contributor setup script with automatic git configuration
- Enhanced development documentation with git setup instructions

This release significantly improves the contributor experience by automating the resolution of the most common merge conflicts in the repository.

## [4.2.0] - 2025-08-10

### 🔧 **Improved Client Compatibility**

#### Added
- **LM Studio Compatibility Layer**: Automatic handling of non-standard MCP notifications
  - Monkey patch for `notifications/cancelled` messages that aren't in the MCP specification
  - Graceful error handling prevents server crashes when LM Studio cancels operations
  - Debug logging for troubleshooting compatibility issues
  - Comprehensive documentation in `docs/LM_STUDIO_COMPATIBILITY.md`

#### Technical
- Added `lm_studio_compat.py` module with compatibility patches
- Applied patches automatically during server initialization
- Enhanced error handling in MCP protocol communication

This release significantly improves compatibility with LM Studio and other MCP clients while maintaining full backward compatibility with existing Claude Desktop integrations.

## [4.1.1] - 2025-08-10

### Fixed
- **macOS ARM64 Support**: Enhanced PyTorch installation for Apple Silicon
  - Proper dependency resolution for M1/M2/M3 Mac systems
  - Updated torch dependency requirements from `>=1.6.0` to `>=2.0.0` in `pyproject.toml`
  - Platform-specific installation instructions in `install.py`
  - Improved cross-platform dependency management

## [4.1.0] - 2025-08-06

### 🎯 **Full MCP Specification Compliance**

#### Added
- **Enhanced Resources System**: URI-based access to memory collections
  - `memory://stats` - Real-time database statistics and health metrics
  - `memory://tags` - Complete list of available memory tags
  - `memory://recent/{n}` - Access to N most recent memories
  - `memory://tag/{tagname}` - Query memories by specific tag
  - `memory://search/{query}` - Dynamic search with structured results
  - Resource templates for parameterized queries
  - JSON responses for all resource endpoints

- **Guided Prompts Framework**: Interactive workflows for memory operations
  - `memory_review` - Review and organize memories from specific time periods
  - `memory_analysis` - Analyze patterns, themes, and tag distributions
  - `knowledge_export` - Export memories in JSON, Markdown, or Text formats
  - `memory_cleanup` - Identify and remove duplicate or outdated memories
  - `learning_session` - Store structured learning notes with automatic categorization
  - Each prompt includes proper argument schemas and validation

- **Progress Tracking System**: Real-time notifications for long operations
  - Progress notifications with percentage completion
  - Operation IDs for tracking concurrent tasks
  - Enhanced `delete_by_tags` with step-by-step progress
  - Enhanced `dashboard_optimize_db` with operation stages
  - MCP-compliant progress notification protocol

#### Changed
- Extended `MemoryStorage` base class with helper methods for resources
- Enhanced `Memory` and `MemoryQueryResult` models with `to_dict()` methods
- Improved server initialization with progress tracking state management

#### Technical
- Added `send_progress_notification()` method to MemoryServer
- Implemented `get_stats()`, `get_all_tags()`, `get_recent_memories()` in storage base
- Full backward compatibility maintained with existing operations

This release brings the MCP Memory Service to full compliance with the Model Context Protocol specification, enabling richer client interactions and better user experience through structured data access and guided workflows.

## [4.0.1] - 2025-08-04

### Fixed
- **MCP Protocol Validation**: Resolved critical ID validation errors affecting integer/string ID handling
- **Embedding Model Loading**: Fixed model loading failures in offline environments
- **Semantic Search**: Restored semantic search functionality that was broken in 4.0.0
- **Version Consistency**: Fixed version mismatch between `__init__.py` and `pyproject.toml`

### Technical
- Enhanced flexible ID validation for MCP protocol compliance
- Improved error handling for embedding model initialization
- Corrected version bumping process for patch releases

## [4.0.0] - 2025-08-04

### 🚀 **Major Release: Production-Ready Remote MCP Memory Service**

#### Added
- **Native MCP-over-HTTP Protocol**: Direct MCP protocol support via FastAPI without Node.js bridge
- **Remote Server Deployment**: Full production deployment capability with remote access
- **Cross-Device Memory Access**: Validated multi-device memory synchronization
- **Comprehensive Documentation**: Complete deployment guides and remote access documentation

#### Changed
- **Architecture Evolution**: Transitioned from local experimental service to production infrastructure
- **Protocol Compliance**: Applied MCP protocol refactorings with flexible ID validation
- **Docker CI/CD**: Fixed and operationalized Docker workflows for automated deployment
- **Repository Maintenance**: Comprehensive cleanup and branch management

#### Production Validation
- Successfully deployed server running at remote endpoints with 65+ memories
- SQLite-vec backend validated (1.7MB database, 384-dim embeddings)
- all-MiniLM-L6-v2 model loaded and operational
- Full MCP tool suite available and tested

#### Milestones Achieved
- GitHub Issue #71 (remote access) completed
- GitHub Issue #72 (bridge deprecation) resolved
- Production deployment proven successful

## [4.0.0-beta.1] - 2025-08-03

### Added
- **Dual-Service Architecture**: Combined HTTP Dashboard + Native MCP Protocol
- **FastAPI MCP Integration**: Complete integration for native remote access
- **Direct MCP-over-HTTP**: Eliminated dependency on Node.js bridge

### Changed
- **Remote Access Solution**: Resolved remote memory service access (Issue #71)
- **Bridge Deprecation**: Deprecated Node.js bridge in favor of direct protocol
- **Docker Workflows**: Fixed CI/CD pipeline for automated testing

### Technical
- Maintained backward compatibility for existing HTTP API users
- Repository cleanup and branch management improvements
- Significant architectural evolution while preserving existing functionality

## [4.0.0-alpha.1] - 2025-08-03

### Added
- **Initial FastAPI MCP Server**: First implementation of native MCP server structure
- **MCP Protocol Endpoints**: Added core MCP protocol endpoints to FastAPI server
- **Hybrid Support**: Initial HTTP+MCP hybrid architecture support

### Changed
- **Server Architecture**: Began transition from pure HTTP to MCP-native implementation
- **Remote Access Configuration**: Initial configuration for remote server access
- **Protocol Implementation**: Started implementing MCP specification compliance

### Technical
- Validated local testing with FastAPI MCP server
- Fixed `mcp.run()` syntax issues
- Established foundation for dual-protocol support

## [3.3.4] - 2025-08-03

### Fixed
- **Multi-Client Backend Selection**: Fixed hardcoded sqlite_vec backend in multi-client configuration
  - Configuration functions now properly accept and use storage_backend parameter
  - Chosen backend is correctly passed through entire multi-client setup flow
  - M1 Macs with MPS acceleration now correctly use ChromaDB when selected
  - SQLite pragmas only applied when sqlite_vec is actually chosen

### Changed
- **Configuration Instructions**: Updated generic configuration to reflect chosen backend
- **Backend Flexibility**: All systems now get optimal backend configuration in multi-client mode

### Technical
- Resolved Issue #73 affecting M1 Mac users
- Ensures proper backend-specific configuration for all platforms
- Version bump to 3.3.4 for critical fix release

## [3.3.3] - 2025-08-02

### 🔒 **SSL Certificate & MCP Bridge Compatibility**

#### Fixed
- **SSL Certificate Generation**: Now generates certificates with proper Subject Alternative Names (SANs) for multi-hostname/IP compatibility
  - Auto-detects local machine IP address dynamically (no hardcoded IPs)
  - Includes `DNS:memory.local`, `DNS:localhost`, `DNS:*.local` 
  - Includes `IP:127.0.0.1`, `IP:::1` (IPv6), and auto-detected local IP
  - Environment variable support: `MCP_SSL_ADDITIONAL_IPS`, `MCP_SSL_ADDITIONAL_HOSTNAMES`
- **Node.js MCP Bridge Compatibility**: Resolved SSL handshake failures when connecting from Claude Code
  - Added missing MCP protocol methods: `initialize`, `tools/list`, `tools/call`, `notifications/initialized`
  - Enhanced error handling with exponential backoff retry logic (3 attempts, max 5s delay)
  - Comprehensive request/response logging with unique request IDs
  - Improved HTTPS client configuration with custom SSL agent
  - Reduced timeout from 30s to 10s for faster failure detection
  - Removed conflicting Host headers that caused SSL verification issues

#### Changed
- **Certificate Security**: CN changed from `localhost` to `memory.local` for better hostname matching
- **HTTP Client**: Enhanced connection management with explicit port handling and connection close headers
- **Logging**: Added detailed SSL handshake and request flow debugging

#### Environment Variables
- `MCP_SSL_ADDITIONAL_IPS`: Comma-separated list of additional IP addresses to include in certificate
- `MCP_SSL_ADDITIONAL_HOSTNAMES`: Comma-separated list of additional hostnames to include in certificate

This release resolves SSL connectivity issues that prevented Claude Code from connecting to remote MCP Memory Service instances across different networks and deployment environments.

## [3.3.2] - 2025-08-02

### 📚 **Enhanced Documentation & API Key Management**

#### Changed
- **API Key Documentation**: Comprehensive improvements to authentication guides
  - Enhanced multi-client server documentation with security best practices
  - Detailed API key generation and configuration instructions
  - Updated service installation guide with authentication setup
  - Improved CLAUDE.md with API key environment variable explanations

#### Technical
- **Documentation Quality**: Enhanced authentication documentation across multiple guides
- **Security Guidance**: Clear instructions for production API key management
- **Cross-Reference Links**: Better navigation between related documentation sections

This release significantly improves the user experience for setting up secure, authenticated MCP Memory Service deployments.

## [3.3.1] - 2025-08-01

### 🔧 **Memory Statistics & Health Monitoring**

#### Added
- **Enhanced Health Endpoint**: Memory statistics integration for dashboard display
  - Added memory statistics to `/health` endpoint for real-time monitoring
  - Integration with dashboard UI for comprehensive system overview
  - Better visibility into database health and memory usage

#### Fixed
- **Dashboard Display**: Improved dashboard data integration and visualization support

#### Technical
- **Web App Enhancement**: Updated FastAPI app with integrated statistics endpoints
- **Version Synchronization**: Updated package version to maintain consistency

This release enhances monitoring capabilities and prepares the foundation for advanced dashboard features.

## [3.3.0] - 2025-07-31

### 🎨 **Modern Professional Dashboard UI**

#### Added
- **Professional Dashboard Interface**: Complete UI overhaul for web interface
  - Modern, responsive design with professional styling
  - Real-time memory statistics display
  - Interactive memory search and management interface
  - Enhanced user experience for memory operations
  
#### Changed
- **Visual Identity**: Updated project branding with professional dashboard preview
- **User Interface**: Complete redesign of web-based memory management
- **Documentation Assets**: Added dashboard screenshots and visual documentation

#### Technical
- **Web App Modernization**: Updated FastAPI application with modern UI components
- **Asset Organization**: Proper structure for dashboard images and visual assets

This release transforms the web interface from a basic API into a professional, user-friendly dashboard for memory management.

## [3.2.0] - 2025-07-30

### 🛠️ **SQLite-vec Diagnostic & Repair Tools**

#### Added
- **Comprehensive Diagnostic Tools**: Advanced SQLite-vec backend analysis and repair
  - Database integrity checking and validation
  - Embedding consistency verification tools
  - Memory preservation during repairs and migrations  
  - Automated repair workflows for corrupted databases

#### Fixed
- **SQLite-vec Embedding Issues**: Resolved critical embedding problems causing zero search results
  - Fixed embedding dimension mismatches
  - Resolved database schema inconsistencies
  - Improved embedding generation and storage reliability

#### Technical
- **Migration Tools**: Enhanced migration utilities to preserve existing memories during backend transitions
- **Diagnostic Scripts**: Comprehensive database analysis and repair automation

This release significantly improves SQLite-vec backend reliability and provides tools for database maintenance and recovery.

## [3.1.0] - 2025-07-30

### 🔧 **Cross-Platform Service Installation**

#### Added
- **Universal Service Installation**: Complete cross-platform service management
  - Linux systemd service installation and configuration
  - macOS LaunchAgent/LaunchDaemon support
  - Windows Service installation and management
  - Unified service utilities across all platforms

#### Changed
- **Installation Experience**: Streamlined service setup for all operating systems
- **Service Management**: Consistent service control across platforms
- **Documentation**: Enhanced service installation guides

#### Technical
- **Platform-Specific Scripts**: Dedicated installation scripts for each operating system
- **Service Configuration**: Proper service definitions and startup configurations
- **Cross-Platform Utilities**: Unified service management tools

This release enables easy deployment of MCP Memory Service as a system service on any major operating system.

## [3.0.0] - 2025-07-29

### 🚀 MAJOR RELEASE: Autonomous Multi-Client Memory Service

This is a **major architectural evolution** transforming MCP Memory Service from a development tool into a production-ready, intelligent memory system with autonomous processing capabilities.

### Added
#### 🧠 **Dream-Inspired Consolidation System**
- **Autonomous Memory Processing**: Fully autonomous consolidation system inspired by human cognitive processes
- **Exponential Decay Scoring**: Memory aging with configurable retention periods (critical: 365d, reference: 180d, standard: 30d, temporary: 7d)
- **Creative Association Discovery**: Automatic discovery of semantic connections between memories (similarity range 0.3-0.7)
- **Semantic Clustering**: DBSCAN algorithm for intelligent memory grouping (minimum 5 memories per cluster)
- **Memory Compression**: Statistical summarization with 500-character limits while preserving originals
- **Controlled Forgetting**: Relevance-based memory archival system (threshold 0.1, 90-day access window)
- **Automated Scheduling**: Configurable consolidation schedules (daily 2AM, weekly Sunday 3AM, monthly 1st 4AM)
- **Zero-AI Dependencies**: Operates entirely offline using existing embeddings and mathematical algorithms

#### 🌐 **Multi-Client Server Architecture**
- **HTTPS Server**: Production-ready FastAPI server with auto-generated SSL certificates
- **mDNS Service Discovery**: Zero-configuration automatic service discovery (`MCP Memory._mcp-memory._tcp.local.`)
- **Server-Sent Events (SSE)**: Real-time updates with 30s heartbeat intervals for all connected clients
- **Multi-Interface Support**: Service advertisement across all network interfaces (WiFi, Ethernet, Docker, etc.)
- **API Authentication**: Secure API key-based authentication system
- **Cross-Platform Discovery**: Works on Windows, macOS, and Linux with standard mDNS/Bonjour

#### 🚀 **Production Deployment System**
- **Systemd Auto-Startup**: Complete systemd service integration for automatic startup on boot
- **Service Management**: Professional service control scripts with start/stop/restart/status/logs/health commands
- **User-Space Service**: Runs as regular user (not root) for enhanced security
- **Auto-Restart**: Automatic service recovery on failures with 10-second restart delay
- **Journal Logging**: Integrated with systemd journal for professional log management
- **Health Monitoring**: Built-in health checks and monitoring endpoints

#### 📖 **Comprehensive Documentation**
- **Complete Setup Guide**: 100+ line comprehensive production deployment guide
- **Production Quick Start**: Streamlined production deployment instructions
- **Service Management**: Full service lifecycle documentation
- **Troubleshooting**: Detailed problem resolution guides
- **Network Configuration**: Firewall and mDNS setup instructions

### Enhanced
#### 🔧 **Improved Server Features**
- **Enhanced SSE Implementation**: Restored full Server-Sent Events functionality with connection statistics
- **Network Optimization**: Multi-interface service discovery and connection handling
- **Configuration Management**: Environment-based configuration with secure defaults
- **Error Handling**: Comprehensive error handling and recovery mechanisms

#### 🛠️ **Developer Experience**
- **Debug Tools**: Service debugging and testing utilities
- **Installation Scripts**: One-command installation and configuration
- **Management Scripts**: Easy service lifecycle management
- **Archive Organization**: Clean separation of development and production files

### Configuration
#### 🔧 **New Environment Variables**
- `MCP_CONSOLIDATION_ENABLED`: Enable/disable autonomous consolidation (default: true)
- `MCP_MDNS_ENABLED`: Enable/disable mDNS service discovery (default: true)
- `MCP_MDNS_SERVICE_NAME`: Customizable service name for discovery (default: "MCP Memory")
- `MCP_HTTPS_ENABLED`: Enable HTTPS with auto-generated certificates (default: true)
- `MCP_HTTP_HOST`: Server bind address (default: 0.0.0.0 for multi-client)
- `MCP_HTTP_PORT`: Server port (default: 8000)
- Consolidation timing controls: `MCP_SCHEDULE_DAILY`, `MCP_SCHEDULE_WEEKLY`, `MCP_SCHEDULE_MONTHLY`

### Breaking Changes
- **Architecture Change**: Single-client MCP protocol → Multi-client HTTPS server architecture
- **Service Discovery**: Manual configuration → Automatic mDNS discovery
- **Deployment Model**: Development script → Production systemd service
- **Access Method**: Direct library import → HTTP API with authentication

### Migration
- **Client Configuration**: Update to use HTTP-MCP bridge with auto-discovery
- **Service Deployment**: Install systemd service for production use
- **Network Setup**: Configure firewall for ports 8000/tcp (HTTPS) and 5353/udp (mDNS)
- **API Access**: Use generated API key for authentication

### Technical Details
- **Consolidation Algorithm**: Mathematical approach using existing embeddings without external AI
- **Service Architecture**: FastAPI + uvicorn + systemd for production deployment
- **Discovery Protocol**: RFC-compliant mDNS service advertisement
- **Security**: User-space execution, API key authentication, HTTPS encryption
- **Storage**: Continues to support both ChromaDB and SQLite-vec backends

---

## [2.2.0] - 2025-07-29

### Added
- **Claude Code Commands Integration**: 5 conversational memory commands following CCPlugins pattern
  - `/memory-store`: Store information with context and smart tagging
  - `/memory-recall`: Time-based memory retrieval with natural language
  - `/memory-search`: Tag and content-based semantic search
  - `/memory-context`: Capture current session and project context
  - `/memory-health`: Service health diagnostics and statistics
- **Optional Installation System**: Integrated command installation into main installer
  - New CLI arguments: `--install-claude-commands`, `--skip-claude-commands-prompt`
  - Intelligent prompting during installation when Claude Code CLI is detected
  - Automatic backup of existing commands with timestamps
- **Command Management Utilities**: Standalone installation and management script
  - `scripts/claude_commands_utils.py` for manual command installation
  - Cross-platform support with comprehensive error handling
  - Prerequisites testing and service connectivity validation
- **Context-Aware Operations**: Commands understand project and session context
  - Automatic project detection from current directory and git repository
  - Smart tag generation based on file types and development context
  - Session analysis and summarization capabilities
- **Auto-Discovery Integration**: Commands automatically locate MCP Memory Service
  - Uses existing mDNS service discovery functionality
  - Graceful fallback when service is unavailable
  - Backend-agnostic operation (works with both ChromaDB and SQLite-vec)

### Changed
- Updated main README.md with Claude Code commands feature documentation
- Enhanced `docs/guides/claude-code-integration.md` with comprehensive command usage guide
- Updated installation documentation to include new command options
- Version bump from 2.1.0 to 2.2.0 for significant feature addition

### Documentation
- Added detailed command descriptions and usage examples
- Created comparison guide between conversational commands and MCP server registration
- Enhanced troubleshooting documentation for both integration methods
- Added `claude_commands/README.md` with complete command reference

## [2.1.0] - 2025-07-XX

### Added
- **mDNS Service Discovery**: Zero-configuration networking with automatic service discovery
- **HTTPS Support**: SSL/TLS support with automatic self-signed certificate generation
- **Enhanced HTTP-MCP Bridge**: Auto-discovery mode with health validation and fallback
- **Zero-Config Deployment**: No manual endpoint configuration needed for local networks

### Changed
- Updated service discovery to use `_mcp-memory._tcp.local.` service type
- Enhanced HTTP server with SSL certificate generation capabilities
- Improved multi-client coordination with automatic discovery

## [2.0.0] - 2025-07-XX

### Added
- **Dream-Inspired Memory Consolidation**: Autonomous memory management system
- **Multi-layered Time Horizons**: Daily, weekly, monthly, quarterly, yearly consolidation
- **Creative Association Discovery**: Finding non-obvious connections between memories
- **Semantic Clustering**: Automatic organization of related memories
- **Intelligent Compression**: Preserving key information while reducing storage
- **Controlled Forgetting**: Safe archival and recovery systems
- **Performance Optimization**: Efficient processing of 10k+ memories
- **Health Monitoring**: Comprehensive error handling and alerts

### Changed
- Major architecture updates for consolidation system
- Enhanced storage backends with consolidation support
- Improved multi-client coordination capabilities

## [1.0.0] - 2025-07-XX

### Added
- Initial stable release
- Core memory operations (store, retrieve, search, recall)
- ChromaDB and SQLite-vec storage backends
- Cross-platform compatibility
- Claude Desktop integration
- Basic multi-client support

## [0.1.0] - 2025-07-XX

### Added
- Initial development release
- Basic memory storage and retrieval functionality
- ChromaDB integration
- MCP server implementation