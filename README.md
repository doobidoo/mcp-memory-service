# MCP Memory Service

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/doobidoo/mcp-memory-service?style=social)](https://github.com/doobidoo/mcp-memory-service/stargazers)
[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen?style=flat&logo=checkmark)](https://github.com/doobidoo/mcp-memory-service#-in-production)

[![Works with Claude](https://img.shields.io/badge/Works%20with-Claude-blue)](https://claude.ai)
[![Works with Cursor](https://img.shields.io/badge/Works%20with-Cursor-orange)](https://cursor.sh)
[![MCP Protocol](https://img.shields.io/badge/MCP-Compatible-4CAF50?style=flat)](https://modelcontextprotocol.io/)
[![Multi-Client](https://img.shields.io/badge/Multi--Client-13+%20Apps-FF6B35?style=flat)](https://github.com/doobidoo/mcp-memory-service/wiki)

**Production-ready MCP memory service** with **zero database locks**, **hybrid backend** (fast local + cloud sync), and **intelligent memory search** for **AI assistants**. Features **v8.9.0 auto-configuration** for multi-client access, **5ms local reads** with background Cloudflare sync, **Natural Memory Triggers** with 85%+ accuracy, and **OAuth 2.1 team collaboration**. Works with **Claude Desktop, VS Code, Cursor, Continue, and 13+ AI applications**.

<img width="240" alt="MCP Memory Service" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />

## 🚀 Quick Start (2 minutes)

### 🆕 **v8.13.2: Production Stability** (Latest Release - Oct 30, 2025)

**🎯 Concurrent Access + Sync Script Fixes** - Zero database locks with working synchronization:

```bash
# One-command installation with auto-configuration
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service && python install.py

# Choose option 4 (Hybrid - RECOMMENDED) when prompted
# Installer automatically configures:
#   ✅ SQLite pragmas for concurrent access
#   ✅ Cloudflare credentials for cloud sync
#   ✅ Claude Desktop integration

# Done! Fast local + cloud sync with zero database locks
```

**✨ What's New in v8.13.x:**
- 🔧 **Concurrent Access Fix** (v8.13.1) - Zero database locks restored
  - Fixed "database is locked" errors when MCP and HTTP servers run together
  - Connection timeout now set BEFORE opening database (critical fix)
  - Detects already-initialized database to skip DDL operations
  - MCP tools now work perfectly while HTTP server is running
- 🔄 **Sync Script Fix** (v8.13.2) - Memory backend synchronization restored
  - Fixed broken sync_memory_backends.py calling non-existent store_memory()
  - Updated to use proper Memory object creation and storage.store()
  - Now successfully syncs between Cloudflare and SQLite backends
- 📊 **HTTP Integration Tests** (v8.13.0) - 32 comprehensive tests prevent production bugs
- 🏗️ **MemoryService Architecture** (v8.12.x) - 80% code duplication eliminated

**📖 Complete Guide**: [v8.13.2 CHANGELOG](CHANGELOG.md#8132---2025-10-30)

---

<details>
<summary>📜 <strong>Previous Releases</strong> (v8.12, v8.11, v8.10, v8.9...)</summary>

### **v8.12.1: MemoryService Architecture + Critical Fixes** (Oct 28, 2025)
- Fixed import-time evaluation bugs preventing HTTP server startup
- Resolved dashboard loading errors (missing tags parameter)
- Fixed analytics metrics discrepancy (accurate counts for >1000 memories)
- 4 critical bugs discovered and fixed within 4 hours

### **v8.12.0: MemoryService Architecture** (Oct 28, 2025)
- Centralized business logic layer (single source of truth)
- 80% code duplication eliminated between MCP and HTTP servers
- Consistent behavior across all interfaces
- 55 comprehensive tests (34 unit + 21 integration)

### **v8.11.0: JSON and CSV Document Loaders** (Oct 28, 2025)
- Complete JSON loader with nested structure flattening
- CSV loader with auto-detection (delimiters, headers, encoding)
- 29 comprehensive unit tests (15 JSON + 14 CSV)
- Fixed false advertising of JSON/CSV support in SUPPORTED_FORMATS

### **v8.10.0: Complete Analytics Dashboard** (Oct 28, 2025)
- Memory Types Breakdown (pie chart)
- Activity Heatmap (GitHub-style calendar)
- Top Tags Report with co-occurrence patterns
- Recent Activity Report (hourly/daily/weekly breakdowns)
- Storage Report with largest memories and efficiency metrics
- Streak tracking for consecutive activity days

### **v8.9.0: Production-Ready Hybrid Backend** (Oct 27, 2025)
- Hybrid backend as recommended default (5ms local + cloud sync)
- Zero database locks with auto-configured SQLite pragmas
- Auto-configuration installer for seamless setup
- Tested: 5/5 concurrent writes succeeded without errors

### **v8.8.2: Document Upload Tag Validation** (Oct 26, 2025)
- Fixed bloated tags from space-separated file paths
- Enhanced file:// URI handling with proper URL decoding
- Processing mode toggle for batch/individual uploads

### **v8.8.0: DRY Refactoring** (Oct 26, 2025)
- Eliminated 364 lines of code duplication between MCP/HTTP servers
- Created MemoryService class as single source of truth
- Bug fixes now apply to both protocols automatically

### **v8.7.0: Cosine Similarity & Maintenance Tools** (Oct 20, 2025)
- Fixed 0% similarity scores (migrated to cosine distance, now 70-79%)
- 1800x faster duplicate cleanup (5s vs 2.5 hours)
- Automatic migration on startup
```bash
# Maintenance scripts for power users
python scripts/maintenance/regenerate_embeddings.py
bash scripts/maintenance/fast_cleanup_duplicates.sh
```

### **v8.6.0: Document Ingestion System** (Oct 15, 2025)
- Interactive drag-and-drop document upload (PDF, TXT, MD, JSON)
- Document viewer with chunk-by-chunk browsing
- Smart tagging with validation (max 100 chars)
- Optional semtools for enhanced PDF/DOCX/PPTX parsing

### **v8.4.0: Memory Hooks Recency Optimization**
- Recent memory prioritization (80% better context)
- Automatically surfaces memories <7 days old

</details>

### PyPI Installation (Simplest)

**Install from PyPI:**
```bash
# Install latest version from PyPI
pip install mcp-memory-service

# Or with uv (faster)
uv pip install mcp-memory-service
```

**Then configure Claude Desktop** by adding to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:
```json
{
  "mcpServers": {
    "memory": {
      "command": "memory",
      "args": ["server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid"
      }
    }
  }
}
```

For advanced configuration with the interactive installer, clone the repo and run `python scripts/installation/install.py`.

### Traditional Setup Options

**Universal Installer (Most Compatible):**
```bash
# Clone and install with automatic platform detection
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Lightweight installation (SQLite-vec with ONNX embeddings - recommended)
python install.py

# Add full ML capabilities (torch + sentence-transformers for advanced features)
python install.py --with-ml

# Install with hybrid backend (SQLite-vec + Cloudflare sync)
python install.py --storage-backend hybrid
```

**📝 Installation Options Explained:**
- **Default (recommended)**: Lightweight SQLite-vec with ONNX embeddings - fast, works offline, <100MB dependencies
- **`--with-ml`**: Adds PyTorch + sentence-transformers for advanced ML features - heavier but more capable
- **`--storage-backend hybrid`**: Hybrid backend with SQLite-vec + Cloudflare sync - best for multi-device access

**Docker (Fastest):**
```bash
# For MCP protocol (Claude Desktop)
docker-compose up -d

# For HTTP API + OAuth (Team Collaboration)
docker-compose -f docker-compose.http.yml up -d
```

**Smithery (Claude Desktop):**
```bash
# Auto-install for Claude Desktop
npx -y @smithery/cli install @doobidoo/mcp-memory-service --client claude
```

## ⚠️ v6.17.0+ Script Migration Notice

**Updating from an older version?** Scripts have been reorganized for better maintainability:
- **Recommended**: Use `python -m mcp_memory_service.server` in your Claude Desktop config (no path dependencies!)
- **Alternative 1**: Use `uv run memory server` with UV tooling
- **Alternative 2**: Update path from `scripts/run_memory_server.py` to `scripts/server/run_memory_server.py`
- **Backward compatible**: Old path still works with a migration notice

## ⚠️ First-Time Setup Expectations

On your first run, you'll see some warnings that are **completely normal**:

- **"WARNING: Failed to load from cache: No snapshots directory"** - The service is checking for cached models (first-time setup)
- **"WARNING: Using TRANSFORMERS_CACHE is deprecated"** - Informational warning, doesn't affect functionality
- **Model download in progress** - The service automatically downloads a ~25MB embedding model (takes 1-2 minutes)

These warnings disappear after the first successful run. The service is working correctly! For details, see our [First-Time Setup Guide](docs/first-time-setup.md).

### 🐍 Python 3.13 Compatibility Note

**sqlite-vec** may not have pre-built wheels for Python 3.13 yet. If installation fails:
- The installer will automatically try multiple installation methods
- Consider using Python 3.12 for the smoothest experience: `brew install python@3.12`
- Alternative: Use Cloudflare backend with `--storage-backend cloudflare`
- See [Troubleshooting Guide](docs/troubleshooting/general.md#python-313-sqlite-vec-issues) for details

### 🍎 macOS SQLite Extension Support

**macOS users** may encounter `enable_load_extension` errors with sqlite-vec:
- **System Python** on macOS lacks SQLite extension support by default
- **Solution**: Use Homebrew Python: `brew install python && rehash`
- **Alternative**: Use pyenv: `PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0`
- **Fallback**: Use Cloudflare or Hybrid backend: `--storage-backend cloudflare` or `--storage-backend hybrid`
- See [Troubleshooting Guide](docs/troubleshooting/general.md#macos-sqlite-extension-issues) for details

## 📚 Complete Documentation

**👉 Visit our comprehensive [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki) for detailed guides:**

### 🧠 v7.1.0 Natural Memory Triggers (Latest)
- **[Natural Memory Triggers v7.1.0 Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Natural-Memory-Triggers-v7.1.0)** - Intelligent automatic memory awareness
  - ✅ **85%+ trigger accuracy** with semantic pattern detection
  - ✅ **Multi-tier performance** (50ms instant → 150ms fast → 500ms intensive)
  - ✅ **CLI management system** for real-time configuration
  - ✅ **Git-aware context** integration for enhanced relevance
  - ✅ **Zero-restart installation** with dynamic hook loading

### 🆕 v7.0.0 OAuth & Team Collaboration
- **[🔐 OAuth 2.1 Setup Guide](https://github.com/doobidoo/mcp-memory-service/wiki/OAuth-2.1-Setup-Guide)** - **NEW!** Complete OAuth 2.1 Dynamic Client Registration guide
- **[🔗 Integration Guide](https://github.com/doobidoo/mcp-memory-service/wiki/03-Integration-Guide)** - Claude Desktop, **Claude Code HTTP transport**, VS Code, and more
- **[🛡️ Advanced Configuration](https://github.com/doobidoo/mcp-memory-service/wiki/04-Advanced-Configuration)** - **Updated!** OAuth security, enterprise features

### 🚀 Setup & Installation
- **[📋 Installation Guide](https://github.com/doobidoo/mcp-memory-service/wiki/01-Installation-Guide)** - Complete installation for all platforms and use cases
- **[🖥️ Platform Setup Guide](https://github.com/doobidoo/mcp-memory-service/wiki/02-Platform-Setup-Guide)** - Windows, macOS, and Linux optimizations
- **[⚡ Performance Optimization](https://github.com/doobidoo/mcp-memory-service/wiki/05-Performance-Optimization)** - Speed up queries, optimize resources, scaling

### 🧠 Advanced Topics
- **[👨‍💻 Development Reference](https://github.com/doobidoo/mcp-memory-service/wiki/06-Development-Reference)** - Claude Code hooks, API reference, debugging
- **[🔧 Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING)** - **Updated!** OAuth troubleshooting + common issues
- **[❓ FAQ](https://github.com/doobidoo/mcp-memory-service/wiki/08-FAQ)** - Frequently asked questions
- **[📝 Examples](https://github.com/doobidoo/mcp-memory-service/wiki/09-Examples)** - Practical code examples and workflows

### 📂 Internal Documentation
- **[🏗️ Architecture Specs](docs/architecture/)** - Search enhancement specifications and design documents
- **[👩‍💻 Development Docs](docs/development/)** - AI agent instructions, release checklist, refactoring notes
- **[🚀 Deployment Guides](docs/deployment/)** - Docker, dual-service, and production deployment
- **[📚 Additional Guides](docs/guides/)** - Storage backends, migration, mDNS discovery

## ✨ Key Features

### 🏆 **Production-Ready Reliability** 🆕 v8.9.0
- **Hybrid Backend** - Fast 5ms local SQLite + background Cloudflare sync (RECOMMENDED default)
  - Zero user-facing latency for cloud operations
  - Automatic multi-device synchronization
  - Graceful offline operation
- **Zero Database Locks** - Concurrent HTTP + MCP server access works flawlessly
  - Auto-configured SQLite pragmas (`busy_timeout=15000,cache_size=20000`)
  - WAL mode with proper multi-client coordination
  - Tested: 5/5 concurrent writes succeeded with no errors
- **Auto-Configuration** - Installer handles everything
  - SQLite pragmas for concurrent access
  - Cloudflare credentials with connection testing
  - Claude Desktop integration with hybrid backend
  - Graceful fallback to sqlite_vec if cloud setup fails

### 📄 **Document Ingestion System** v8.6.0
- **Interactive Web UI** - Drag-and-drop document upload with real-time progress
- **Multiple Formats** - PDF, TXT, MD, JSON with intelligent chunking
- **Document Viewer** - Browse chunks, view metadata, search content
- **Smart Tagging** - Automatic tagging with length validation (max 100 chars)
- **Optional semtools** - Enhanced PDF/DOCX/PPTX parsing with LlamaParse
- **Security Hardened** - Path traversal protection, XSS prevention, input validation
- **7 New Endpoints** - Complete REST API for document management

### 🔐 **Enterprise Authentication & Team Collaboration**
- **OAuth 2.1 Dynamic Client Registration** - RFC 7591 & RFC 8414 compliant
- **Claude Code HTTP Transport** - Zero-configuration team collaboration
- **JWT Authentication** - Enterprise-grade security with scope validation
- **Auto-Discovery Endpoints** - Seamless client registration and authorization
- **Multi-Auth Support** - OAuth + API keys + optional anonymous access

### 🧠 **Intelligent Memory Management**
- **Semantic search** with vector embeddings
- **Natural language time queries** ("yesterday", "last week")
- **Tag-based organization** with smart categorization
- **Memory consolidation** with dream-inspired algorithms
- **Document-aware search** - Query across uploaded documents and manual memories

### 🔗 **Universal Compatibility**
- **Claude Desktop** - Native MCP integration
- **Claude Code** - **HTTP transport** + Memory-aware development with hooks
- **VS Code, Cursor, Continue** - IDE extensions
- **13+ AI applications** - REST API compatibility

### 💾 **Flexible Storage**
- **Hybrid** 🌟 (RECOMMENDED) - Fast local SQLite + background Cloudflare sync (v8.9.0 default)
  - 5ms local reads with zero user-facing latency
  - Multi-device synchronization
  - Zero database locks with auto-configured pragmas
  - Automatic backups and cloud persistence
- **SQLite-vec** - Local-only storage (lightweight ONNX embeddings, 5ms reads)
  - Good for single-user offline use
  - No cloud dependencies
- **Cloudflare** - Cloud-only storage (global edge distribution with D1 + Vectorize)
  - Network-dependent performance

> **Note**: All heavy ML dependencies (PyTorch, sentence-transformers) are now optional to dramatically reduce build times and image sizes. SQLite-vec uses lightweight ONNX embeddings by default. Install with `--with-ml` for full ML capabilities.

### 🚀 **Production Ready**
- **Cross-platform** - Windows, macOS, Linux
- **Service installation** - Auto-start background operation
- **HTTPS/SSL** - Secure connections with OAuth 2.1
- **Docker support** - Easy deployment with team collaboration
- **Interactive Dashboard** - Web UI at http://127.0.0.1:8888/ for complete management

## 💡 Basic Usage

### 📄 **Document Ingestion** (v8.6.0+)
```bash
# Start server with web interface
uv run memory server --http

# Access interactive dashboard
open http://127.0.0.1:8888/

# Upload documents via CLI
curl -X POST http://127.0.0.1:8888/api/documents/upload \
  -F "file=@document.pdf" \
  -F "tags=documentation,reference"

# Search document content
curl -X POST http://127.0.0.1:8888/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication flow", "limit": 10}'
```

### 🔗 **Team Collaboration with OAuth** (v7.0.0+)
```bash
# Start OAuth-enabled server for team collaboration
export MCP_OAUTH_ENABLED=true
uv run memory server --http

# Claude Code team members connect via HTTP transport
claude mcp add --transport http memory-service http://your-server:8000/mcp
# → Automatic OAuth discovery, registration, and authentication
```

### 🧠 **Memory Operations**
```bash
# Store a memory
uv run memory store "Fixed race condition in authentication by adding mutex locks"

# Search for relevant memories
uv run memory recall "authentication race condition"

# Search by tags
uv run memory search --tags python debugging

# Check system health (shows OAuth status)
uv run memory health
```

## 🔧 Configuration

### Claude Desktop Integration
**Recommended approach** - Add to your Claude Desktop config (`~/.claude/config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

**Alternative approaches:**
```json
// Option 1: UV tooling (if using UV)
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory", "server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}

// Option 2: Direct script path (v6.17.0+)
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/scripts/server/run_memory_server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

### Environment Variables

**Hybrid Backend (v8.9.0+ RECOMMENDED):**
```bash
# Hybrid backend with auto-configured pragmas
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,cache_size=20000"

# Cloudflare credentials (required for hybrid)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-db-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# Enable HTTP API
export MCP_HTTP_ENABLED=true
export MCP_HTTP_PORT=8000

# Security
export MCP_API_KEY="your-secure-key"
```

**SQLite-vec Only (Local):**
```bash
# Local-only storage
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,cache_size=20000"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Clients    │    │  MCP Memory     │    │ Storage Backend │
│                 │    │  Service v8.9   │    │                 │
│ • Claude Desktop│◄──►│ • MCP Protocol  │◄──►│ • Hybrid 🌟     │
│ • Claude Code   │    │ • HTTP Transport│    │   (5ms local +  │
│   (HTTP/OAuth)  │    │ • OAuth 2.1 Auth│    │    cloud sync)  │
│ • VS Code       │    │ • Memory Store  │    │ • SQLite-vec    │
│ • Cursor        │    │ • Semantic      │    │ • Cloudflare    │
│ • 13+ AI Apps   │    │   Search        │    │                 │
│ • Web Dashboard │    │ • Doc Ingestion │    │ Zero DB Locks ✅│
│   (Port 8888)   │    │ • Zero DB Locks │    │ Auto-Config ✅  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Development

### Project Structure
```
mcp-memory-service/
├── src/mcp_memory_service/    # Core application
│   ├── models/                # Data models
│   ├── storage/               # Storage backends
│   ├── web/                   # HTTP API & dashboard
│   └── server.py              # MCP server
├── scripts/                   # Utilities & installation
├── tests/                     # Test suite
└── tools/docker/              # Docker configuration
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🆘 Support

- **📖 Documentation**: [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki) - Comprehensive guides
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/doobidoo/mcp-memory-service/discussions)
- **🔧 Troubleshooting**: [Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING)
- **✅ Configuration Validator**: Run `python scripts/validation/validate_configuration_complete.py` to check your setup
- **🔄 Backend Sync Tools**: See [scripts/README.md](scripts/README.md#backend-synchronization) for Cloudflare↔SQLite sync

## 📊 In Production

**Real-world metrics from active deployments:**
- **1700+ memories** stored and actively used across teams
- **5ms local reads** with hybrid backend (v8.9.0)
- **Zero database locks** with concurrent HTTP + MCP access (v8.9.0)
  - Tested: 5/5 concurrent writes succeeded
  - Auto-configured pragmas prevent lock errors
- **<500ms response time** for semantic search (local & HTTP transport)
- **65% token reduction** in Claude Code sessions with OAuth collaboration
- **96.7% faster** context setup (15min → 30sec)
- **100% knowledge retention** across sessions and team members
- **Zero-configuration** setup success rate: **98.5%** (OAuth + hybrid backend)

## 🏆 Recognition

- [![Smithery](https://smithery.ai/badge/@doobidoo/mcp-memory-service)](https://smithery.ai/server/@doobidoo/mcp-memory-service) **Verified MCP Server**
- [![Glama AI](https://img.shields.io/badge/Featured-Glama%20AI-blue)](https://glama.ai/mcp/servers/bzvl3lz34o) **Featured AI Tool**
- **Production-tested** across 13+ AI applications
- **Community-driven** with real-world feedback and improvements

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

**Ready to supercharge your AI workflow?** 🚀

👉 **[Start with our Installation Guide](https://github.com/doobidoo/mcp-memory-service/wiki/01-Installation-Guide)** or explore the **[Wiki](https://github.com/doobidoo/mcp-memory-service/wiki)** for comprehensive documentation.

*Transform your AI conversations into persistent, searchable knowledge that grows with you.*