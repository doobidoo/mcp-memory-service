# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **📝 Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Note**: Comprehensive project context has been stored in memory with tags `claude-code-reference`. Use memory retrieval to access detailed information during development.

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop with SQLite-vec, Cloudflare, and Hybrid storage backends.

> **🚨 v8.13.3**: **MCP Tools Restored** - CRITICAL patch fixing v8.12.0 regression that broke all MCP memory operations. Transform MemoryService responses to proper MCP TypedDict format. Requires MCP server restart (/mcp command) to load fix.

> **🔄 v8.13.2**: **Sync Script Restored** - Fixed broken backend synchronization (store_memory API migration). Proper Memory object creation with storage.store() method.

> **🔧 v8.13.1**: **Concurrent Access Fix** - Zero database locks restored. Connection timeout now set BEFORE opening database, detects already-initialized database to skip DDL operations.

> **📊 v8.13.0**: **HTTP Integration Tests** - 32 comprehensive tests prevent production bugs. Server startup validation, dependency injection tests, storage interface compatibility checks.

> **🧠 v8.5.1**: **Dynamic Memory Weight Adjustment** - Intelligent auto-calibration prevents stale memories from dominating context! Automatically detects memory age vs git activity mismatches and adapts weights/boosts accordingly. No more manual config tweaks!

> **🆕 v8.4.0**: **Memory Hooks Recency Optimization** - Recent memory prioritization with 80% better context accuracy. Comprehensive scoring rebalancing ensures development work from the last 7 days surfaces automatically!

> **🎉 v8.3.1**: **HTTP Server Management** - Cross-platform auto-start scripts and health check utilities for seamless Natural Memory Triggers integration!

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

# Database Maintenance (NEW v8.16.0)
python scripts/maintenance/check_memory_types.py                  # Check memory type distribution
python scripts/maintenance/consolidate_memory_types.py --dry-run  # Preview type consolidation (safe)
python scripts/maintenance/consolidate_memory_types.py            # Execute type consolidation
python scripts/maintenance/find_all_duplicates.py                 # Find duplicate memories
bash scripts/maintenance/fast_cleanup_duplicates.sh               # Remove duplicates quickly

# Service Management
scripts/service/memory_service_manager.sh status       # Check service status
scripts/service/memory_service_manager.sh start-cloudflare # Start with Cloudflare

# HTTP Server (Linux systemd)
systemctl --user start/stop/restart mcp-memory-http.service  # Control service
systemctl --user status mcp-memory-http.service              # Check status
journalctl --user -u mcp-memory-http.service -f              # View logs
bash scripts/service/install_http_service.sh                 # Install service

# Natural Memory Triggers v7.1.0 (Latest)
node ~/.claude/hooks/memory-mode-controller.js status   # Check trigger system status
node ~/.claude/hooks/memory-mode-controller.js profile balanced  # Switch performance profile
node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.7   # Adjust trigger sensitivity
node ~/.claude/hooks/test-natural-triggers.js          # Test trigger system

# Context-Provider Integration (Latest)
# Note: Context-provider commands are integrated into MCP client automatically
# No manual commands needed - contexts activate automatically during sessions

# Amp CLI Bridge (Semi-Automated File-Based Workflow)
# Usage: Claude Code creates prompt → Shows you command → You run: amp @{prompt-file} → Response appears

# Debug & Troubleshooting
npx @modelcontextprotocol/inspector uv run memory server # MCP Inspector
python scripts/database/simple_timestamp_check.py       # Database health check
python scripts/maintenance/consolidate_memory_types.py --dry-run  # Preview type consolidation
python scripts/maintenance/consolidate_memory_types.py  # Execute type consolidation
df -h /                                               # Check disk space (critical for Litestream)
journalctl -u mcp-memory-service -f                   # Monitor service logs

# Interactive Dashboard Testing & Validation
curl "http://127.0.0.1:8000/api/health"              # Health check (expect 200 OK)
curl "http://127.0.0.1:8000/api/search" -H "Content-Type: application/json" -d '{"query":"test"}' # Semantic search
curl "http://127.0.0.1:8000/api/search/by-tag" -H "Content-Type: application/json" -d '{"tags":["test"]}' # Tag search
curl "http://127.0.0.1:8000/api/search/by-time" -H "Content-Type: application/json" -d '{"query":"last week"}' # Time search
curl -N "http://127.0.0.1:8000/api/events"           # Test SSE real-time updates
time curl -s "http://127.0.0.1:8000/" > /dev/null     # Dashboard page load performance

# Critical: Post-v8.12.0 Testing Requirements
# After architecture changes, ALWAYS test:
# 1. HTTP server actually starts (uv run memory server --http)
# 2. Dashboard loads in browser without errors
# 3. API endpoints return valid responses (not 500 errors)
# 4. All storage backends have compatible interfaces
```

## Architecture

**Core Components:**
- **Server Layer**: MCP protocol implementation with async handlers and global caches (`src/mcp_memory_service/server.py`)
- **Storage Backends**: SQLite-Vec (fast local, 5ms reads), Cloudflare (edge distribution), Hybrid (SQLite+Cloudflare sync)
- **Web Interface**: FastAPI dashboard at `http://127.0.0.1:8000/` (HTTP) or `https://localhost:8443/` (HTTPS) with REST API
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
- **Dashboard Access**:
  - HTTP mode (default): `http://127.0.0.1:8000/`
  - HTTPS mode (when enabled): `https://localhost:8443/`
- **API Base**: `/api/` for programmatic access
- **SSE Events**: `/api/events` for real-time updates
- **Server Ports**: Same port for both HTTP/HTTPS API and MCP protocol (default: 8000)
- **File Structure**: `src/mcp_memory_service/web/static/` (index.html, app.js, style.css)

## Environment Variables

**Essential Configuration:**
```bash
# Storage Backend (Hybrid is RECOMMENDED for production)
export MCP_MEMORY_STORAGE_BACKEND=hybrid  # hybrid|cloudflare|sqlite_vec

# Cloudflare Configuration (REQUIRED for hybrid/cloudflare backends)
export CLOUDFLARE_API_TOKEN="your-token"      # Required for Cloudflare backend
export CLOUDFLARE_ACCOUNT_ID="your-account"   # Required for Cloudflare backend
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id" # Required for Cloudflare backend
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index" # Required for Cloudflare backend

# Web Interface (Optional)
export MCP_HTTP_ENABLED=true                  # Enable HTTP server
export MCP_HTTPS_ENABLED=true                 # Enable HTTPS (production)
export MCP_API_KEY="$(openssl rand -base64 32)" # Generate secure API key
```

**Configuration Precedence:** Environment variables > .env file > Global Claude Config > defaults

**✅ Automatic Configuration Loading (v6.16.0+):** The service now automatically loads `.env` files and respects environment variable precedence. CLI defaults no longer override environment configuration.

**⚠️  Important:** When using hybrid or cloudflare backends, ensure Cloudflare credentials are properly configured. If health checks show "sqlite-vec" when you expect "cloudflare" or "hybrid", this indicates a configuration issue that needs to be resolved.

**Platform Support:** macOS (MPS/CPU), Windows (CUDA/DirectML/CPU), Linux (CUDA/ROCm/CPU)

## Claude Code Hooks Configuration 🆕

> **🚨 CRITICAL - Windows Users**: SessionStart hooks with `matchers: ["*"]` cause Claude Code to hang indefinitely on Windows. This is a confirmed bug (#160). **Workaround**: Disable SessionStart hooks or use UserPromptSubmit hooks instead. See [Windows SessionStart Hook Issue](#windows-sessionstart-hook-issue) below.

### Natural Memory Triggers v7.1.0 (Latest)

**Intelligent automatic memory retrieval** with advanced semantic analysis and multi-tier performance optimization:

```bash
# Installation (Zero-restart required)
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
- Smart tagging: Auto-detects tools (fastapi, cloudflare, sqlite-vec, hybrid, etc.)

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
| **Cloudflare** ☁️ | Network dependent | Cloud-only deployment | `install.py --storage-backend cloudflare` |
| **SQLite-Vec** 🪶 | Fast (5ms read) | Development, single-user local | `install.py --storage-backend sqlite_vec` |

### ⚠️ **Database Lock Prevention (v8.9.0+)**

**CRITICAL**: After adding `MCP_MEMORY_SQLITE_PRAGMAS` to `.env`, you **MUST restart all servers**:
- HTTP server: `kill <PID>` then restart with `uv run python scripts/server/run_http_server.py`
- MCP servers: Use `/mcp` in Claude Code to reconnect, or restart Claude Desktop
- Verify: Check logs for `Custom pragma from env: busy_timeout=15000`

SQLite pragmas are **per-connection**, not global. Long-running servers (days/weeks old) won't pick up new `.env` settings automatically.

**Symptoms of missing pragmas**:
- "database is locked" errors despite v8.9.0+ installation
- `PRAGMA busy_timeout` returns `0` instead of `15000`
- Concurrent HTTP + MCP access fails

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

#### **Memory Type Taxonomy** (Updated Nov 2025)
Database consolidated from 342 fragmented types to 128 organized types. Use these **24 core types** for all new memories:

**Content Types:**
- `note` - General notes, observations, summaries
- `reference` - Reference materials, knowledge base entries
- `document` - Formal documents, code snippets
- `guide` - How-to guides, tutorials, troubleshooting guides

**Activity Types:**
- `session` - Work sessions, development sessions
- `implementation` - Implementation work, integrations
- `analysis` - Analysis, reports, investigations
- `troubleshooting` - Problem-solving, debugging
- `test` - Testing activities, validation

**Artifact Types:**
- `fix` - Bug fixes, corrections
- `feature` - New features, enhancements
- `release` - Releases, release notes
- `deployment` - Deployments, deployment records

**Progress Types:**
- `milestone` - Milestones, completions, achievements
- `status` - Status updates, progress reports

**Infrastructure Types:**
- `configuration` - Configurations, setups, settings
- `infrastructure` - Infrastructure changes, system updates
- `process` - Processes, workflows, procedures
- `security` - Security-related memories
- `architecture` - Architecture decisions, design patterns

**Other Types:**
- `documentation` - Documentation artifacts
- `solution` - Solutions, resolutions
- `achievement` - Accomplishments, successes

**Usage Notes:**
- Avoid creating new type variations (e.g., use `fix` not `bug-fix`, `bugfix`, `technical-fix`)
- Avoid redundant prefixes (e.g., use `solution` not `technical-solution`, `project-solution`)
- Run `python scripts/maintenance/consolidate_memory_types.py --dry-run` to preview type cleanup

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
- **Dashboard**:
  - HTTP mode (default): `http://127.0.0.1:8000/`
  - HTTPS mode (when enabled): `https://localhost:8443/`
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

**Troubleshooting Hooks Not Retrieving Memories:**
```bash
# Check if HTTP server is running
systemctl --user status mcp-memory-http.service  # Linux
# or
uv run python scripts/server/check_http_server.py  # All platforms

# Verify hooks endpoint matches server port
cat ~/.claude/hooks/config.json | grep endpoint
# Should show: http://127.0.0.1:8000 (not 8889 or other port)

# See detailed guide: docs/http-server-management.md
```

**⚠️ CRITICAL: Hook Configuration Synchronization**

When configuring Claude Code hooks, **all HTTP endpoints MUST use the same port** across configuration files:

**Configuration Files to Check:**
1. **`~/.claude/hooks/config.json`** - Line 7: `"endpoint": "http://127.0.0.1:8000"`
2. **HTTP Server** - Default port: `8000` (check `scripts/server/run_http_server.py`)
3. **Dashboard/Web Interface** - Separate port: `8000` (HTTP) or `8443` (HTTPS)

**Common Mistakes:**
- ❌ Port mismatch (config.json shows 8889 but server runs on 8000)
- ❌ Using dashboard port (8000/8443) instead of API server port (8000)
- ❌ Different ports in `settings.json` MCP server env vs hooks config

**Quick Verification:**
```bash
# Windows
netstat -ano | findstr "8000"

# Linux/macOS
lsof -i :8000

# Check hooks config
grep endpoint ~/.claude/hooks/config.json
```

**Symptoms of Port Mismatch:**
- SessionStart hook hangs/times out
- Claude Code becomes unresponsive on startup
- Hooks show "connection timeout" in logs
- No memories injected despite hook firing

**Troubleshooting Schema Validation Errors After PR Merges:**

**Symptom**: After merging a PR that changes tool schemas, you still see validation errors like:
```
Input validation error: 'value' is not of type 'expected_type'
```

**Root Cause**: MCP clients (like Claude Code) cache tool schemas when they first connect. Even after:
- ✅ PR is merged
- ✅ Git pull completes
- ✅ Code is updated
- ❌ **MCP server process is still running old code**

The old MCP server continues advertising the old schema, and the client validates against this cached schema.

**Diagnosis**:
```bash
# 1. Check when PR was merged
gh pr view <PR_NUMBER> --json mergedAt,title

# 2. Check when MCP server process started
ps aux | grep "memory.*server" | grep -v grep

# 3. If server started BEFORE merge time, it's running old code
```

**Solution**:
```bash
# In Claude Code, reconnect MCP:
/mcp

# This will:
# 1. Terminate old MCP server process
# 2. Start new MCP server with latest code
# 3. Re-fetch updated tool schemas
# 4. Clear client-side schema cache

# For HTTP server (separate from MCP):
systemctl --user restart mcp-memory-http.service
```

**Example**: PR #162 (comma-separated tags fix)
- Merged: Oct 20, 2025 17:22 UTC
- Error persisted: "Input validation error: 'tag1,tag2' is not of type 'array'"
- Server process: Started Oct 21 10:43 (before git pull)
- Fix: `/mcp` command to reconnect with new schema

**See**: `docs/troubleshooting/pr162-schema-caching-issue.md` for detailed analysis

**Emergency Debugging:**
```bash
/mcp                                         # Check active MCP servers in Claude
python scripts/validation/diagnose_backend_config.py  # Run configuration validation
rm -f .mcp.json                             # Remove conflicting local MCP config
python debug_server_initialization.py       # Test initialization flows (v6.15.1+)
tail -50 ~/Library/Logs/Claude/mcp-server-memory.log | grep -E "(🚀|☁️|✅|❌)" # View enhanced logs
```

**⚠️ Accidental Database Creation:**

If you find a `data/memory.db` file in your project directory:
- This is **not** the configured database location
- It may be created accidentally by tools running in the project directory
- Safe to delete: `rm -rf data/` (already in `.gitignore`)
- Configured location: `~/Library/Application Support/mcp-memory/sqlite_vec.db` (macOS)
- Verify: `curl http://localhost:8000/api/health` should show correct memory count

### SessionEnd Hook Troubleshooting

**Common Confusion**: Many users expect SessionEnd hooks to fire when they press Ctrl+C twice, but this is **not** how Claude Code works.

#### 🔍 When SessionEnd Hooks Actually Trigger

**SessionEnd fires ONLY on actual session termination**:
- ✅ `/exit` command - Graceful session termination
- ✅ Terminal/window close - Process termination
- ✅ Normal Claude Code exit - Graceful shutdown

**SessionEnd does NOT fire on**:
- ❌ Ctrl+C (once) - Interrupts input only
- ❌ Ctrl+C (twice) - Suspends session
- ❌ Session resume - Continues existing session (fires `SessionStart:resume` instead)

**Key Insight**: When you press Ctrl+C twice and later resume, you see:
```
SessionStart:resume hook success
```
This confirms you **resumed** an existing session - no SessionEnd was triggered.

#### 🐛 Common Issue: "My Session Didn't Create a Memory"

**Symptom**: You exited Claude Code with Ctrl+C, resumed later, but no `session-consolidation` memory exists.

**Root Cause**: Ctrl+C **suspends** the session, it doesn't **end** it. Only `/exit` or terminal close triggers SessionEnd.

**Solution**: Always use `/exit` to properly terminate sessions if you want memories created.

#### 🔌 Common Issue: Connection Failures

**Symptom**:
```
⚠️ Memory Connection → Failed to connect using any available protocol
💾 Storage → 💾 Unknown Storage (http://127.0.0.1:8000)
```

**Root Cause**: HTTP/HTTPS protocol mismatch between hook config and server.

**Diagnosis**:
```bash
# Check server protocol
systemctl --user status mcp-memory-http.service
# Look for: "Uvicorn running on https://..." or "http://..."

# Check hook config
grep endpoint ~/.claude/hooks/config.json
```

**Solution**: Update `~/.claude/hooks/config.json` to match server:
```json
{
  "memoryService": {
    "http": {
      "endpoint": "https://localhost:8000",  // Match your server protocol
      "apiKey": "your-api-key"
    }
  }
}
```

#### 📋 SessionEnd Memory Requirements

Even if SessionEnd fires, memory creation requires:
1. **Minimum session length**: 100+ characters (configurable)
2. **Minimum confidence**: > 0.1 from conversation analysis
3. **Session consolidation enabled**: `enableSessionConsolidation: true` in config

**What gets extracted**:
- Topics (implementation, debugging, architecture, etc.)
- Decisions ("decided to", "will use", "chose to")
- Insights ("learned that", "discovered", "realized")
- Code changes ("implemented", "created", "refactored")
- Next steps ("next we need", "TODO", "remaining")

#### 🔧 Quick Verification

```bash
# Check recent session memories
curl -sk "https://localhost:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["session-consolidation"], "limit": 5}' | \
  python -m json.tool | grep created_at_iso

# Test SessionEnd hook manually
node ~/.claude/hooks/core/session-end.js

# Verify connection
curl -sk "https://localhost:8000/api/health"
```

**Detailed Guide**: See `docs/troubleshooting/session-end-hooks.md` for comprehensive troubleshooting steps, diagnosis checklist, and technical details.

### Windows SessionStart Hook Issue

**🚨 CRITICAL BUG**: SessionStart hooks with `matchers: ["*"]` cause Claude Code to hang indefinitely on Windows.

**Issue**: [#160](https://github.com/doobidoo/mcp-memory-service/issues/160)

**Symptoms**:
- Claude Code becomes completely unresponsive when starting
- Hook executes but process never terminates
- Cannot enter prompts or cancel with Ctrl+C
- Must force-close terminal to exit

**Root Cause**:
Windows-specific subprocess management issue. Even with `process.exit(0)`, Node.js subprocesses with open connections (HTTP client, etc.) don't close all file descriptors properly on Windows, causing the parent process (Claude Code) to wait indefinitely.

**Tested Solutions** (None worked on Windows):
- ❌ Multiple `process.exit(0)` calls
- ❌ `.finally()` blocks with forced exit
- ❌ Minimal hook (just print + exit)
- ❌ Windows batch wrapper with forced exit
- ❌ Increased timeouts (no timeout enforcement occurs)

**Workarounds**:

1. **Use `/session-start` slash command** (recommended):
```bash
claude /session-start
```
- Provides same functionality as automatic SessionStart hook
- Works on all platforms (Windows, macOS, Linux)
- Safe manual alternative - no configuration changes needed
- Installer automatically skips SessionStart configuration on Windows
- See: `claude_commands/session-start.md` for full documentation

2. **Disable SessionStart hooks** (if manually configured):
```json
{
  "hooks": {
    "SessionStart": []
  }
}
```

3. **Use UserPromptSubmit hooks instead** (these work on Windows):
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matchers": ["*"],
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/hooks/core/mid-conversation.js",
            "timeout": 8
          }
        ]
      }
    ]
  }
}
```

4. **Manual hook invocation** (advanced):
```bash
node C:\Users\username\.claude\hooks\core\session-start.js
```

**Platform Status**:
- macOS: Works correctly ✅
- Linux: Works correctly ✅ (assumed)
- Windows: Fatal hang ❌

**Impact**: Critical for Windows users. SessionStart hooks are completely unusable until Claude Code fixes subprocess management on Windows.

---

## Amp CLI Bridge (Semi-Automated Workflow) 🆕

**Purpose**: Leverage Amp CLI capabilities (research, code analysis, web search) from Claude Code without consuming Claude Code credits, using a semi-automated file-based workflow where you manually run Amp commands.

### Architecture

```
Claude Code (@agent-amp-bridge) → .claude/amp/prompts/pending/{uuid}.json
                                            ↓
                          You run: amp @prompts/pending/{uuid}.json
                                            ↓
                          Amp writes: responses/ready/{uuid}.json
                                            ↓
                   Claude Code reads response ← Workflow continues
```

### Quick Start

**1. Claude Code creates prompt**:
```
You: "Use @agent-amp-bridge to research TypeScript 5.0 features"
Claude: [Creates prompt file and shows command]
```

**2. Run the command shown**:
```bash
amp @.claude/amp/prompts/pending/research-xyz.json
```

**3. Amp processes and writes response**:
- Amp reads the prompt from the file
- Processes using your authenticated free-tier session
- Writes response to `.claude/amp/responses/ready/research-xyz.json`

**4. Claude Code continues automatically**:
- Detects response file
- Reads and presents results
- Archives processed files

### How It Works

1. **Claude Code Agent** (`@agent-amp-bridge`):
   - Generates UUID for each request
   - Writes prompt to `.claude/amp/prompts/pending/{uuid}.json`
   - **Includes file-write instructions** in the prompt
   - Polls `.claude/amp/responses/ready/{uuid}.json` for response
   - Presents results when available

2. **You (Manual Step)**:
   - Run: `./scripts/amp/next-prompt.sh` to see command
   - Execute: `amp @.claude/amp/prompts/pending/{uuid}.json`
   - Amp reads prompt and writes response file

3. **Amp Processing**:
   - Reads prompt from JSON file using `@` file reference
   - Processes using your authenticated session (free tier credits)
   - Follows embedded instructions to write response JSON
   - Writes to `.claude/amp/responses/ready/{uuid}.json`

4. **File Structure**:
   ```
   .claude/amp/
   ├── prompts/
   │   └── pending/        # Prompts waiting for you to process
   ├── responses/
   │   ├── ready/          # Responses written by Amp
   │   └── consumed/       # Archive of processed responses
   └── README.md           # Documentation
   ```

### Message Format

**Prompt** (`.claude/amp/prompts/pending/{uuid}.json`):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-04T20:00:00.000Z",
  "prompt": "Research async/await best practices in Python",
  "context": {
    "project": "mcp-memory-service",
    "cwd": "/path/to/project"
  },
  "options": {
    "timeout": 300000,
    "format": "markdown"
  }
}
```

**Response** (`.claude/amp/responses/ready/{uuid}.json`):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-04T20:05:00.000Z",
  "success": true,
  "output": "## Async/Await Best Practices\n\n...",
  "error": null,
  "duration": 300000
}
```

### Configuration

**File**: `.claude/amp/config.json`

```json
{
  "pollInterval": 1000,      // Check for new prompts every 1s
  "timeout": 300000,          // 5 minute timeout per prompt
  "debug": false,             // Enable debug logging
  "ampCommand": "amp"         // Amp CLI command
}
```

### Use Cases

✅ **Web Research**: "Research latest React 18 features"
✅ **Code Analysis**: "Analyze our storage backend architecture"
✅ **Documentation**: "Generate API docs for MCP tools"
✅ **Code Generation**: "Create TypeScript type definitions"
✅ **Best Practices**: "Find OAuth 2.1 security recommendations"

### Manual Inspection (Optional)

**List pending prompts:**
```bash
ls -lt .claude/amp/prompts/pending/
```

**View prompt content:**
```bash
cat .claude/amp/prompts/pending/{uuid}.json | jq -r '.prompt'
```

**Note**: The agent automatically shows you the exact command when creating each prompt.

### Troubleshooting

**Amp CLI credit errors:**
```bash
# Test if Amp is authenticated
amp

# If credits exhausted:
# - Free tier limits may be daily/monthly
# - Check status: https://ampcode.com/settings
# - Wait for refresh or upgrade subscription
```

**Response not appearing:**
```bash
# Verify Amp wrote the file
ls -lt .claude/amp/responses/ready/

# Check if Amp followed file-write instructions
# The prompt explicitly tells Amp where to write
```

**Permission issues:**
```bash
# Ensure directories exist
ls -la .claude/amp/

# Check write permissions
touch .claude/amp/responses/ready/test.json && rm .claude/amp/responses/ready/test.json
```

### Benefits

- ✅ **Zero Claude Code Credits**: All Amp API calls use your separate authenticated session
- ✅ **Uses Free Tier**: Works with Amp's free tier (when credits available)
- ✅ **Simple Workflow**: No background processes, just manual command execution
- ✅ **Full Control**: You decide when/what to process
- ✅ **Fault Tolerant**: File-based queue survives crashes
- ✅ **Audit Trail**: All prompts/responses saved for debugging
- ✅ **Reusable**: Can replay prompts or review past responses

### Limitations

- 🖐️ **Manual Step Required**: You must run the `amp @` command for each prompt
- 💰 **Amp Credits**: Still consumes Amp API credits (free tier limits apply)
- ⏱️ **Semi-Async**: Claude Code waits for you to process the prompt
- 📝 **Best for Research**: Optimized for async research tasks, not real-time chat

---

> **For detailed troubleshooting, architecture, and deployment guides:**
> - **Backend Configuration Issues**: See [Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#backend-configuration-issues) for comprehensive solutions to missing memories, environment variable issues, Cloudflare auth, hooks timeouts, and more
> - **Historical Context**: Retrieve memories tagged with `claude-code-reference`
> - **Quick Diagnostic**: Run `python scripts/validation/diagnose_backend_config.py`
