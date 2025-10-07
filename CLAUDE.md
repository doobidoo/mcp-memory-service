# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **Note**: Comprehensive project context has been stored in memory with tags `claude-code-reference`. Use memory retrieval to access detailed information during development.

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop using ChromaDB and sentence transformers.

> **🎉 v7.2.2**: **Interactive Dashboard Production-Ready** - Comprehensive validation complete with excellent performance (25ms page load, 26ms memory ops), advanced search capabilities, real-time SSE updates, and mobile responsiveness!

> **🧠 v7.1.0**: Now features **Natural Memory Triggers** with intelligent automatic memory retrieval, 85%+ trigger accuracy, and multi-tier performance optimization!

> **🚀 v7.0.0**: Features **OAuth 2.1 Dynamic Client Registration** and **Dual Protocol Memory Hooks** for Claude Code with automatic HTTP/MCP protocol detection.

## Essential Commands

```bash
# Setup & Development
python scripts/installation/install.py         # Platform-aware installation with backend selection
python scripts/installation/install.py --storage-backend hybrid      # Hybrid setup (RECOMMENDED)
python scripts/installation/install.py --storage-backend cloudflare  # Direct Cloudflare setup
uv run memory server                           # Start server (v6.3.0+ consolidated CLI)
pytest tests/                                 # Run tests
python scripts/validation/verify_environment.py # Check environment
python scripts/validation/validate_configuration_complete.py   # Comprehensive configuration validation

# HTTP Server Management (Required for Claude Code hooks)
python scripts/server/check_http_server.py -v  # Check if HTTP server is running
scripts/server/start_http_server.sh            # Start HTTP server (Unix/macOS)
scripts/server/start_http_server.bat           # Start HTTP server (Windows)

# Memory Operations (requires: python scripts/utils/claude_commands_utils.py)
claude /memory-store "content"                 # Store information
claude /memory-recall "query"                  # Retrieve information
claude /memory-health                         # Check service status

# Configuration Validation
python scripts/validation/diagnose_backend_config.py  # Validate Cloudflare configuration

# Backend Synchronization
python scripts/sync/sync_memory_backends.py --status    # Check sync status
python scripts/sync/sync_memory_backends.py --dry-run   # Preview sync
python scripts/sync/claude_sync_commands.py backup      # Cloudflare → SQLite
python scripts/sync/claude_sync_commands.py restore     # SQLite → Cloudflare

# Service Management
scripts/service/memory_service_manager.sh status       # Check service status
scripts/service/memory_service_manager.sh start-cloudflare # Start with Cloudflare

# Natural Memory Triggers v7.1.0 (Latest)
node ~/.claude/hooks/memory-mode-controller.js status   # Check trigger system status
node ~/.claude/hooks/memory-mode-controller.js profile balanced  # Switch performance profile
node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.7   # Adjust trigger sensitivity
node ~/.claude/hooks/test-natural-triggers.js          # Test trigger system

# Context-Provider Integration (Latest)
# Note: Context-provider commands are integrated into MCP client automatically
# No manual commands needed - contexts activate automatically during sessions

# Debug & Troubleshooting
npx @modelcontextprotocol/inspector uv run memory server # MCP Inspector
python scripts/database/simple_timestamp_check.py       # Database health check
df -h /                                               # Check disk space (critical for Litestream)
journalctl -u mcp-memory-service -f                   # Monitor service logs

# Interactive Dashboard Testing & Validation
curl "http://127.0.0.1:8888/api/health"              # Health check (expect 200 OK)
curl "http://127.0.0.1:8888/api/search" -H "Content-Type: application/json" -d '{"query":"test"}' # Semantic search
curl "http://127.0.0.1:8888/api/search/by-tag" -H "Content-Type: application/json" -d '{"tags":["test"]}' # Tag search
curl "http://127.0.0.1:8888/api/search/by-time" -H "Content-Type: application/json" -d '{"query":"last week"}' # Time search
curl -N "http://127.0.0.1:8888/api/events"           # Test SSE real-time updates
time curl -s "http://127.0.0.1:8888/" > /dev/null     # Dashboard page load performance
```

## Architecture

**Core Components:**
- **Server Layer**: MCP protocol implementation with async handlers and global caches (`src/mcp_memory_service/server.py`)
- **Storage Backends**: SQLite-Vec (fast, single-client), ChromaDB (multi-client), Cloudflare (production)
- **Web Interface**: FastAPI dashboard at `https://localhost:8443/` with REST API
- **Document Ingestion**: Pluggable loaders for PDF, DOCX, PPTX, text with semtools support
- **Dual Protocol Memory Hooks** 🆕: Advanced Claude Code integration with HTTP + MCP support
  - **HTTP Protocol**: Web-based memory service connection (`https://localhost:8443/api/*`)
  - **MCP Protocol**: Direct server process communication (`uv run memory server`)
  - **Smart Auto-Detection**: MCP preferred → HTTP fallback → Environment detection
  - **Unified Interface**: Transparent protocol switching via `MemoryClient` wrapper

**Key Design Patterns:**
- Async/await for all I/O operations
- Type safety with Python 3.10+ hints
- Platform detection for hardware optimization (CUDA, MPS, DirectML, ROCm)
- Global model and embedding caches for performance
- **Protocol Abstraction** 🆕: Single interface for multi-protocol memory operations

## Document Ingestion (v7.6.0+) 📄

**Enhanced document parsing** with optional semtools integration for superior quality extraction.

### Supported Formats

| Format | Native Parser | With Semtools | Quality |
|--------|--------------|---------------|---------|
| PDF | PyPDF2/pdfplumber | ✅ LlamaParse | Excellent (OCR, tables) |
| DOCX | ❌ Not supported | ✅ LlamaParse | Excellent |
| PPTX | ❌ Not supported | ✅ LlamaParse | Excellent |
| TXT/MD | ✅ Built-in | N/A | Perfect |

### Semtools Integration (Optional)

Install [semtools](https://github.com/run-llama/semtools) for enhanced document parsing:

```bash
# Install via npm (recommended)
npm i -g @llamaindex/semtools

# Or via cargo
cargo install semtools

# Optional: Configure LlamaParse API key for best quality
export LLAMAPARSE_API_KEY="your-api-key"
```

### Configuration

```bash
# Document chunking settings
export MCP_DOCUMENT_CHUNK_SIZE=1000          # Characters per chunk
export MCP_DOCUMENT_CHUNK_OVERLAP=200        # Overlap between chunks

# LlamaParse API key (optional, improves quality)
export LLAMAPARSE_API_KEY="llx-..."
```

### Usage Examples

```bash
# Ingest a single document
claude /memory-ingest document.pdf --tags documentation

# Ingest directory
claude /memory-ingest-dir ./docs --tags knowledge-base

# Via Python
from mcp_memory_service.ingestion import get_loader_for_file

loader = get_loader_for_file(Path("document.pdf"))
async for chunk in loader.extract_chunks(Path("document.pdf")):
    await store_memory(chunk.content, tags=["doc"])
```

### Features

- ✅ **Automatic format detection** - Selects best loader for each file
- ✅ **Intelligent chunking** - Respects paragraph/sentence boundaries
- ✅ **Metadata enrichment** - Preserves file info, extraction method, page numbers
- ✅ **Graceful fallback** - Uses native parsers if semtools unavailable
- ✅ **Progress tracking** - Reports chunks processed during ingestion

## Interactive Dashboard (v7.2.2+) 🎉

**Production-ready web interface** providing complete memory management capabilities with excellent performance.

### ✅ **Core Features**
- **Complete CRUD Operations**: Create, read, update, delete memories with intuitive UI
- **Advanced Search**: Semantic search, tag-based filtering, and time-based queries
- **Real-time Updates**: Server-Sent Events (SSE) with 30-second heartbeat for live dashboard updates
- **Mobile Responsive**: CSS breakpoints for mobile (768px) and tablet (1024px) devices
- **Security**: XSS protection via `escapeHtml()` function throughout frontend
- **OAuth Integration**: Seamless conditional loading for both enabled/disabled OAuth modes

### 📊 **Performance Benchmarks** (Validated v7.2.2)
| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Page Load | <2s | 25ms | ✅ EXCELLENT |
| Memory Operations | <1s | 26ms | ✅ EXCELLENT |
| Tag Search | <500ms | <100ms | ✅ EXCELLENT |
| Large Dataset | 1000+ | 994+ tested | ✅ EXCELLENT |

### 🔍 **Search API Endpoints**
```bash
# Semantic search (similarity-based)
POST /api/search
{"query": "documentation", "limit": 10}

# Tag-based search (exact tag matching)
POST /api/search/by-tag
{"tags": ["important", "reference"], "limit": 10}

# Time-based search (natural language)
POST /api/search/by-time
{"query": "last week", "n_results": 10}
```

### 🎯 **Usage**
- **Dashboard Access**: `http://127.0.0.1:8888/` (HTTP) or `https://localhost:8443/` (HTTPS)
- **API Base**: `/api/` for programmatic access
- **SSE Events**: `/api/events` for real-time updates
- **File Structure**: `src/mcp_memory_service/web/static/` (index.html, app.js, style.css)

## Environment Variables

**Essential Configuration:**
```bash
# Storage Backend (Cloudflare is PRODUCTION default)
export MCP_MEMORY_STORAGE_BACKEND=cloudflare  # cloudflare|sqlite_vec|chroma

# Cloudflare Production Configuration (REQUIRED)
export CLOUDFLARE_API_TOKEN="your-token"      # Required for Cloudflare backend
export CLOUDFLARE_ACCOUNT_ID="your-account"   # Required for Cloudflare backend
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id" # Required for Cloudflare backend
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index" # Required for Cloudflare backend

# Web Interface (Optional)
export MCP_HTTP_ENABLED=true                  # Enable HTTP server
export MCP_HTTPS_ENABLED=true                 # Enable HTTPS (production)
export MCP_API_KEY="$(openssl rand -base64 32)" # Generate secure API key
```

**Configuration Precedence:** Global Claude Config > .env file > Environment variables

**✅ Automatic Configuration Loading (v6.16.0+):** The service now automatically loads `.env` files and respects environment variable precedence. CLI defaults no longer override environment configuration.

**⚠️  Important:** This system uses **Cloudflare as the primary backend**. If health checks show SQLite-vec instead of Cloudflare, this indicates a configuration issue that needs to be resolved.

**Platform Support:** macOS (MPS/CPU), Windows (CUDA/DirectML/CPU), Linux (CUDA/ROCm/CPU)

## Claude Code Hooks Configuration 🆕

### Protocol Selection Guide (v8.2.6+)

**Choose the right protocol for your environment** - Natural Memory Triggers support both HTTP and MCP protocols with automatic fallback.

#### **When to Use HTTP Protocol (Recommended for Claude Code)**

HTTP is the **default and recommended** protocol when using Natural Memory Triggers within Claude Code:

**Why HTTP?**
- ✅ Avoids MCP server conflicts (Claude Code already uses the MCP server)
- ✅ Works alongside HTTP-MCP Bridge for persistent server connection
- ✅ Better for production hooks that run automatically on session start
- ✅ More reliable in Claude Code environment

**Configuration Example:**
```json
// ~/.claude/hooks/config.json
{
  "memoryService": {
    "protocol": "http",
    "preferredProtocol": "http",
    "fallbackEnabled": true,
    "http": {
      "endpoint": "http://localhost:8000",
      "apiKey": "your-api-key",
      "healthCheckTimeout": 3000,
      "useDetailedHealthCheck": true
    }
  }
}
```

**Prerequisites:**
- HTTP server must be running: `uv run python scripts/server/run_http_server.py`
- Or use HTTPS: `MCP_HTTPS_ENABLED=true uv run python scripts/server/run_http_server.py`

**Checking Server Status:**
```bash
# Check if HTTP server is running
python scripts/server/check_http_server.py -v

# Auto-start server (checks first, starts if needed)
scripts/server/start_http_server.sh    # Unix/macOS
scripts/server/start_http_server.bat   # Windows
```

**⚠️ Important:** If the HTTP server is not running, the session-start hook will fail silently and no memories will be injected. Always verify the server is running before starting Claude Code.

#### **When to Use MCP Protocol (Optional)**

MCP protocol is available for **standalone scenarios** where Claude Code isn't running:

**Valid MCP Use Cases:**
- 🔧 Standalone hook testing and development
- 📝 VSCode, Cursor, Windsurf, Cline, RooCode, Zed IDE integrations
- 🧪 Direct MCP server access without HTTP overhead
- 🚀 Scenarios where Claude Code isn't actively using the MCP server

**Configuration Example:**
```json
// ~/.claude/hooks/config.json (standalone/other IDEs)
{
  "memoryService": {
    "protocol": "mcp",
    "preferredProtocol": "mcp",
    "fallbackEnabled": true,
    "mcp": {
      "serverCommand": ["uv", "run", "memory", "server"],
      "serverWorkingDir": "/path/to/mcp-memory-service",
      "connectionTimeout": 5000,
      "toolCallTimeout": 10000
    }
  }
}
```

**⚠️ Important:** MCP protocol will **fail in Claude Code** because the MCP server is already in use. The installer auto-detects your environment and configures the appropriate protocol.

#### **Auto Protocol (Smart Fallback)**

The `"auto"` protocol tries preferred protocol first, then falls back:

```json
{
  "memoryService": {
    "protocol": "auto",
    "preferredProtocol": "http",  // Try HTTP first
    "fallbackEnabled": true,
    "http": { ... },
    "mcp": { ... }
  }
}
```

**Recommended:** Use explicit `"http"` protocol in Claude Code instead of `"auto"` to avoid unnecessary MCP connection attempts.

#### **Troubleshooting**

For comprehensive troubleshooting of hooks configuration issues, see:
- **[Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#claude-code-hooks-timeout)** - Detailed solutions for timeout issues, backend configuration, and more
- **Configuration Validation:** `python scripts/validation/validate_configuration_complete.py`
- **Diagnostic Script:** `python scripts/validation/diagnose_backend_config.py`

---

### Natural Memory Triggers v7.1.0 (Latest)

**Intelligent automatic memory retrieval** with advanced semantic analysis and multi-tier performance optimization:

```bash
# Installation (Zero-restart required, auto-detects environment)
cd claude-hooks && python install_hooks.py --natural-triggers

# CLI Management
node ~/.claude/hooks/memory-mode-controller.js status
node ~/.claude/hooks/memory-mode-controller.js profile balanced
node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.6
```

**Key Features:**
- ✅ **85%+ trigger accuracy** for memory-seeking pattern detection
- ✅ **Multi-tier processing**: 50ms instant → 150ms fast → 500ms intensive
- ✅ **CLI management system** for real-time configuration without restart
- ✅ **Git-aware context** integration for enhanced memory relevance
- ✅ **Adaptive learning** based on user preferences and usage patterns

**Configuration (`~/.claude/hooks/config.json`):**
```json
{
  "naturalTriggers": {
    "enabled": true,
    "triggerThreshold": 0.6,
    "cooldownPeriod": 30000,
    "maxMemoriesPerTrigger": 5
  },
  "performance": {
    "defaultProfile": "balanced",
    "enableMonitoring": true,
    "autoAdjust": true
  }
}
```

**Performance Profiles:**
- `speed_focused`: <100ms, instant tier only - minimal memory awareness for speed
- `balanced`: <200ms, instant + fast tiers - optimal for general development (recommended)
- `memory_aware`: <500ms, all tiers - maximum context awareness for complex work
- `adaptive`: Dynamic adjustment based on usage patterns and user feedback

### Context-Provider Integration 🆕

**Rule-based context management** that complements Natural Memory Triggers with structured, project-specific patterns:

```bash
# Context-Provider Commands
mcp context list                                # List available contexts
mcp context status                             # Check session initialization status
mcp context optimize                           # Get optimization suggestions
```

#### **Available Contexts:**

**1. Python MCP Memory Service Context** (`python_mcp_memory`)
- Project-specific patterns for FastAPI, MCP protocol, and storage backends
- Auto-store: MCP protocol changes, backend configs, performance optimizations
- Auto-retrieve: Troubleshooting, setup queries, implementation examples
- Smart tagging: Auto-detects tools (fastapi, cloudflare, sqlite, chromadb, etc.)

**2. Release Workflow Context** 🆕 (`mcp_memory_release_workflow`)
- **PR Review Cycle**: Iterative Gemini Code Assist workflow (Fix → Comment → /gemini review → Wait 1min → Repeat)
- **Version Management**: Three-file procedure (__init__.py → pyproject.toml → uv lock)
- **CHANGELOG Management**: Format guidelines, conflict resolution (combine PR entries)
- **Documentation Matrix**: When to use CHANGELOG vs Wiki vs CLAUDE.md vs code comments
- **Release Procedure**: Merge → Tag → Push → Verify workflows (Docker Publish, Publish and Test, HTTP-MCP Bridge)
- **Issue Management** 🆕: Auto-tracking, post-release workflow, smart closing comments
  - **Auto-Detection**: Tracks "fixes #", "closes #", "resolves #" patterns in PRs
  - **Post-Release Workflow**: Retrieves issues from release, suggests closures with context
  - **Smart Comments**: Auto-generates closing comments with PR links, CHANGELOG entries, wiki references
  - **Triage Intelligence**: Auto-categorizes issues (bug, feature, docs, performance) based on patterns

**Auto-Store Patterns:**
- **Technical**: `MCP protocol`, `tool handler`, `storage backend switch`, `25ms page load`, `embedding cache`
- **Configuration**: `cloudflare configuration`, `hybrid backend setup`, `oauth integration`
- **Release Workflow** 🆕: `merged PR`, `gemini review`, `created tag`, `CHANGELOG conflict`, `version bump`
- **Documentation** 🆕: `updated CHANGELOG`, `wiki page created`, `CLAUDE.md updated`
- **Issue Tracking** 🆕: `fixes #`, `closes #`, `resolves #`, `created issue`, `closed issue #`

**Auto-Retrieve Patterns:**
- **Troubleshooting**: `cloudflare backend error`, `MCP client connection`, `storage backend failed`
- **Setup**: `backend configuration`, `environment setup`, `claude desktop config`
- **Development**: `MCP handler example`, `API endpoint pattern`, `async error handling`
- **Release Workflow** 🆕: `how to release`, `PR workflow`, `gemini iteration`, `version bump procedure`, `where to document`
- **Issue Management** 🆕: `review open issues`, `what issues fixed`, `can we close`, `issue status`, `which issues resolved`

**Documentation Decision Matrix:**
| Change Type | CHANGELOG | CLAUDE.md | Wiki | Code Comments |
|-------------|-----------|-----------|------|---------------|
| Bug fix | ✅ Always | If affects workflow | If complex | ✅ Non-obvious |
| New feature | ✅ Always | If adds commands | ✅ Major features | ✅ API changes |
| Performance | ✅ Always | If measurable | If >20% improvement | Rationale |
| Config change | ✅ Always | ✅ User-facing | If requires migration | Validation logic |
| Troubleshooting | In notes | If common | ✅ Detailed guide | For maintainers |

**Integration Benefits:**
- **Structured Memory Management**: Rule-based triggers complement AI-based Natural Memory Triggers
- **Project-Specific Intelligence**: Captures MCP Memory Service-specific terminology and workflows
- **Enhanced Git Workflow**: Automatic semantic commit formatting and branch naming conventions
- **Release Automation** 🆕: Never miss version bumps, CHANGELOG updates, or workflow verification
- **Knowledge Retention** 🆕: Capture what works/doesn't work in PR review cycles
- **Intelligent Issue Management** 🆕: Auto-track issue-PR relationships, suggest closures after releases, generate smart closing comments
- **Post-Release Efficiency** 🆕: Automated checklist retrieves related issues, suggests verification steps, includes all context
- **Zero Performance Impact**: Lightweight rule processing with minimal overhead

### Dual Protocol Memory Hooks (Legacy)

**Dual Protocol Memory Hooks** (v7.0.0+) provide intelligent memory awareness with automatic protocol detection:

```json
{
  "memoryService": {
    "protocol": "auto",
    "preferredProtocol": "mcp",
    "fallbackEnabled": true,
    "http": {
      "endpoint": "https://localhost:8443",
      "apiKey": "your-api-key",
      "healthCheckTimeout": 3000,
      "useDetailedHealthCheck": true
    },
    "mcp": {
      "serverCommand": ["uv", "run", "memory", "server", "-s", "cloudflare"],
      "serverWorkingDir": "/Users/yourname/path/to/mcp-memory-service",
      "connectionTimeout": 5000,
      "toolCallTimeout": 10000
    }
  }
}
```

**Protocol Options:**
- `"auto"`: Smart detection (MCP → HTTP → Environment fallback)
- `"http"`: HTTP-only mode (web server at localhost:8443)
- `"mcp"`: MCP-only mode (direct server process)

**Benefits:**
- **Reliability**: Multiple connection methods ensure hooks always work
- **Performance**: MCP direct for speed, HTTP for stability
- **Flexibility**: Works with local development or remote deployments
- **Compatibility**: Full backward compatibility with existing configurations

## Storage Backends

| Backend | Performance | Use Case | Installation |
|---------|-------------|----------|--------------|
| **Hybrid** ⚡ | **Fast (5ms read)** | **🌟 Production (Recommended)** | `install.py --storage-backend hybrid` |
| **Cloudflare** ☁️ | Network dependent | Legacy cloud-only | `install.py --storage-backend cloudflare` |
| SQLite-Vec 🪶 | Fast (5ms read) | Development, single-user | `install.py --storage-backend sqlite_vec` |
| ChromaDB 👥 | Medium (15ms read) | Team, multi-client local | `install.py --storage-backend chromadb` |

### 🚀 **Hybrid Backend (v6.21.0+) - RECOMMENDED**

The **Hybrid backend** provides the best of both worlds - **SQLite-vec speed with Cloudflare persistence**:

```bash
# Enable hybrid backend
export MCP_MEMORY_STORAGE_BACKEND=hybrid

# Hybrid-specific configuration
export MCP_HYBRID_SYNC_INTERVAL=300    # Background sync every 5 minutes
export MCP_HYBRID_BATCH_SIZE=50        # Sync 50 operations at a time
export MCP_HYBRID_SYNC_ON_STARTUP=true # Initial sync on startup

# Requires Cloudflare credentials (same as cloudflare backend)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
```

**Key Benefits:**
- ✅ **5ms read/write performance** (SQLite-vec speed)
- ✅ **Zero user-facing latency** - Cloud sync happens in background
- ✅ **Multi-device synchronization** - Access memories everywhere
- ✅ **Graceful offline operation** - Works without internet, syncs when available
- ✅ **Automatic failover** - Falls back to SQLite-only if Cloudflare unavailable

**Architecture:**
- **Primary Storage**: SQLite-vec (all user operations)
- **Secondary Storage**: Cloudflare (background sync)
- **Background Service**: Async queue with retry logic and health monitoring

**v6.16.0+ Installer Enhancements:**
- **Interactive backend selection** with usage-based recommendations
- **Automatic Cloudflare credential setup** and `.env` file generation
- **Connection testing** during installation to validate configuration
- **Graceful fallbacks** from cloud to local backends if setup fails

## Development Guidelines

### 🧠 **Memory & Documentation**
- Use `claude /memory-store` to capture decisions during development
- Memory operations handle duplicates via content hashing
- Time parsing supports natural language ("yesterday", "last week")
- Use semantic commit messages for version management

### 🏗️ **Architecture & Testing**
- Storage backends must implement abstract base class
- All features require corresponding tests
- **Comprehensive UI Testing**: Validate performance benchmarks (page load <2s, operations <1s)
- **Security Validation**: Verify XSS protection, input validation, and OAuth integration
- **Mobile Testing**: Confirm responsive design at 768px and 1024px breakpoints

### 🚀 **Version Management Best Practices**
- Document major milestones in CHANGELOG.md with performance metrics
- Create descriptive git tags for releases (`git tag -a v7.2.2 -m "description"`)
- Sync develop/main branches after releases
- Update version in both `__init__.py` and `pyproject.toml`

### 🔧 **Configuration & Deployment**
- Run `python scripts/validation/validate_configuration_complete.py` when troubleshooting setup issues
- Use sync utilities for hybrid Cloudflare/SQLite deployments
- Test both OAuth enabled/disabled modes for web interface
- Validate search endpoints: semantic (`/api/search`), tag (`/api/search/by-tag`), time (`/api/search/by-time`)

## Key Endpoints

### 🌐 **Web Interface**
- **Dashboard**: `http://127.0.0.1:8888/` (HTTP) or `https://localhost:8443/` (HTTPS)
- **Health Check**: `/api/health` - Server status and version
- **SSE Events**: `/api/events` - Real-time updates via Server-Sent Events

### 📋 **Memory Management**
- **CRUD Operations**: `/api/memories` - Create, read, update, delete memories
- **Memory Details**: `/api/memories/{hash}` - Get specific memory by content hash
- **Tags**: `/api/tags` - Get all available tags with counts

### 🔍 **Search APIs**
- **Semantic Search**: `POST /api/search` - Similarity-based search
- **Tag Search**: `POST /api/search/by-tag` - Filter by specific tags
- **Time Search**: `POST /api/search/by-time` - Natural language time queries
- **Similar**: `GET /api/search/similar/{hash}` - Find memories similar to given hash

### 📚 **Documentation**
- **Wiki**: `https://github.com/doobidoo/mcp-memory-service/wiki`
- **API Reference**: Available in dashboard at `/api/docs` (when enabled)

## Configuration Management

**Validation & Troubleshooting:**
```bash
python scripts/validation/validate_configuration_complete.py  # Comprehensive configuration validation
```

**Single Source of Truth:**
- **Global Configuration**: `~/.claude.json` (authoritative for all projects)
- **Project Environment**: `.env` file (Cloudflare credentials only)
- **No Local Overrides**: Project `.mcp.json` should NOT contain memory server config

**Common Configuration Issues (Pre-v6.16.0):**
- **✅ FIXED**: CLI defaults overriding environment variables
- **✅ FIXED**: Manual .env file loading required
- **Multiple Backends**: Conflicting SQLite/Cloudflare configurations
- **Credential Conflicts**: Old macOS paths or missing Cloudflare credentials
- **Cache Issues**: Restart Claude Code to refresh MCP connections

**v6.16.0+ Configuration Benefits:**
- **Automatic .env loading**: No manual configuration required
- **Proper precedence**: Environment variables respected over CLI defaults
- **Better error messages**: Clear indication of configuration loading issues

**Cloudflare Backend Troubleshooting:**
- **Enhanced Initialization Logging**: Look for these indicators in Claude Desktop logs:
  - 🚀 SERVER INIT: Main server initialization flow
  - ☁️ Cloudflare-specific initialization steps
  - ✅ Success markers for each phase
  - ❌ Error details with full tracebacks
  - 🔍 Storage type verification (confirms final backend)
- **Common Issues**:
  - Silent fallback to SQLite-vec: Check logs for eager initialization timeout or API errors
  - Configuration validation: Environment variables are logged during startup
  - Network timeouts: Enhanced error messages show specific Cloudflare API failures

**Dual Environment Setup (Claude Desktop + Claude Code):**
```bash
# Quick setup for both environments - see docs/quick-setup-cloudflare-dual-environment.md
python scripts/validation/diagnose_backend_config.py  # Validate Cloudflare configuration
claude mcp list                             # Check Claude Code MCP servers
```

**Troubleshooting Health Check Showing Wrong Backend:**
```bash
# If health check shows "sqlite-vec" instead of "cloudflare":
python scripts/validation/diagnose_backend_config.py  # Check configuration
claude mcp remove memory && claude mcp add memory python -e MCP_MEMORY_STORAGE_BACKEND=cloudflare -e CLOUDFLARE_API_TOKEN=your-token -- -m mcp_memory_service.server
```

**Emergency Debugging:**
```bash
/mcp                                         # Check active MCP servers in Claude
python scripts/validation/diagnose_backend_config.py  # Run configuration validation
rm -f .mcp.json                             # Remove conflicting local MCP config
python debug_server_initialization.py       # Test initialization flows (v6.15.1+)
tail -50 ~/Library/Logs/Claude/mcp-server-memory.log | grep -E "(🚀|☁️|✅|❌)" # View enhanced logs
```

> **For detailed troubleshooting, architecture, and deployment guides:**
> - **Backend Configuration Issues**: See [Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#backend-configuration-issues) for comprehensive solutions to missing memories, environment variable issues, Cloudflare auth, hooks timeouts, and more
> - **Historical Context**: Retrieve memories tagged with `claude-code-reference`
> - **Quick Diagnostic**: Run `python scripts/validation/diagnose_backend_config.py`
- always use "/gemini review" when commenting on a PR
- make sure to have commited and pushed every change to the branch before commenting on the PR and before triggering gemini Code Assist