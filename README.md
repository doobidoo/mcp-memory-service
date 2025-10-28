# MCP Memory Service

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/doobidoo/mcp-memory-service?style=social)](https://github.com/doobidoo/mcp-memory-service/stargazers)
[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen?style=flat&logo=checkmark)](https://github.com/doobidoo/mcp-memory-service#-in-production)

[![Works with Claude](https://img.shields.io/badge/Works%20with-Claude-blue)](https://claude.ai)
[![Works with Cursor](https://img.shields.io/badge/Works%20with-Cursor-orange)](https://cursor.sh)
[![MCP Protocol](https://img.shields.io/badge/MCP-Compatible-4CAF50?style=flat)](https://modelcontextprotocol.io/)
[![Multi-Client](https://img.shields.io/badge/Multi--Client-13+%20Apps-FF6B35?style=flat)](https://github.com/doobidoo/mcp-memory-service/wiki)

**Universal MCP memory service** with **intelligent memory triggers**, **OAuth 2.1 team collaboration**, and **semantic memory search** for **AI assistants**. Features **Natural Memory Triggers v7.1.0** with 85%+ trigger accuracy, **Claude Code HTTP transport**, **zero-configuration authentication**, and **enterprise security**. Works with **Claude Desktop, VS Code, Cursor, Continue, and 13+ AI applications** with **SQLite-vec** for fast local search and **Cloudflare** for global distribution.

<img width="240" alt="MCP Memory Service" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />

## 🚀 Quick Start (2 minutes)

### 🆕 **v8.7.0: Cosine Similarity & Maintenance Tools**

**🎯 Fixed Search Accuracy + 1800x Faster Database Maintenance**:
```bash
# Automatic migration on startup - nothing to do!
# Search similarity scores now 70-79% (was 0%)

# New maintenance scripts for power users:
cd mcp-memory-service

# Regenerate all embeddings (~5min for 2600 memories)
python scripts/maintenance/regenerate_embeddings.py

# Fast duplicate cleanup (1800x faster than API!)
bash scripts/maintenance/fast_cleanup_duplicates.sh  # <5s for 100+ duplicates

# Find duplicates with timestamp normalization
python scripts/maintenance/find_all_duplicates.py    # <2s for 2000 memories
```

**✨ What's New in v8.7.0:**
- 🎯 **Fixed 0% similarity scores** - Migrated to cosine distance metric (70-79% accuracy now)
- 🔧 **Maintenance scripts** - 1800x performance improvement (5s vs 2.5 hours for 100 duplicates)
- 📈 **Search accuracy boost** - Exact match 79.2% (was 61%)
- 🔄 **Automatic migration** - Seamless upgrade on first startup
- 📚 **Comprehensive docs** - Performance benchmarks, best practices, troubleshooting

**📖 Complete Guide**: [Maintenance Scripts README](scripts/maintenance/README.md)

---

### 🆕 **v8.6.0: Document Ingestion System**

**📄 Transform Documents into Searchable Memory** (Production Ready):
```bash
# 1. Install/Update MCP Memory Service
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service && python install.py

# 2. Start the server with web interface
uv run memory server --http

# 3. Access interactive dashboard
open http://127.0.0.1:8888/
# ✅ Upload PDFs, TXT, MD, JSON - Browse chunks - Search content!
```

**✨ What's New in v8.6.0:**
- 📚 **Complete document management UI** with drag-and-drop upload
- 🔍 **Interactive document viewer** with chunk-by-chunk browsing
- 🏷️ **Smart tagging** with length validation (prevents bloat)
- 🔒 **Security hardened** (path traversal & XSS protection)
- 📄 **Multiple formats**: PDF, TXT, MD, JSON with optional [semtools](https://github.com/run-llama/semtools) for enhanced parsing

**📖 Complete Guide**: [Document Ingestion Guide](https://github.com/doobidoo/mcp-memory-service/wiki)

---

### 🆕 **v8.4.0: Memory Hooks Recency Optimization**

**📊 Intelligent Recent Memory Prioritization** (80% Better Context):
```bash
# Automatically prioritizes recent development work over old memories
# Memory hooks now surface memories <7 days old with 80% higher accuracy

# 1. Install Natural Memory Triggers (includes recency optimization)
cd claude-hooks && python install_hooks.py --natural-triggers

# 2. Verify recency scoring
node test-recency-scoring.js
# ✅ Done! Recent memories automatically prioritized in Claude Code
```

**📖 Complete Guide**: [Memory Hooks Configuration](https://github.com/doobidoo/mcp-memory-service/tree/main/claude-hooks/CONFIGURATION.md)

---

### 🧠 **Natural Memory Triggers v7.1.0+**

**🤖 Intelligent Memory Awareness** (Zero Configuration):
- ✅ **85%+ trigger accuracy** with semantic pattern detection
- ✅ **Recent memory prioritization** (v8.4.0+) - 80% better context
- ✅ **Multi-tier performance** (50ms instant → 150ms fast → 500ms intensive)
- ✅ **CLI management system** for real-time configuration

**📖 Complete Guide**: [Natural Memory Triggers](https://github.com/doobidoo/mcp-memory-service/wiki/Natural-Memory-Triggers-v7.1.0)

---

### 🆕 **v7.0.0: OAuth 2.1 & Claude Code HTTP Transport**

**🔗 Claude Code Team Collaboration** (Zero Configuration):
```bash
# 1. Start OAuth-enabled server
export MCP_OAUTH_ENABLED=true
uv run memory server --http

# 2. Add HTTP transport to Claude Code
claude mcp add --transport http memory-service http://localhost:8000/mcp

# ✅ Done! Claude Code automatically handles OAuth registration and team collaboration
```

**📖 Complete Setup Guide**: [OAuth 2.1 Setup Guide](https://github.com/doobidoo/mcp-memory-service/wiki/OAuth-2.1-Setup-Guide)

---

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

### 📄 **Document Ingestion System** 🆕 v8.6.0
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
- **SQLite-vec** - Fast local storage (recommended, lightweight ONNX embeddings, 5ms reads)
- **Cloudflare** - Global edge distribution with D1 + Vectorize
- **Hybrid** - Best of both worlds: SQLite-vec speed + Cloudflare sync (recommended for teams)
- **Automatic backups** and synchronization

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
```bash
# Storage backend (sqlite_vec recommended)
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Enable HTTP API
export MCP_HTTP_ENABLED=true
export MCP_HTTP_PORT=8000

# Security  
export MCP_API_KEY="your-secure-key"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Clients    │    │  MCP Memory     │    │ Storage Backend │
│                 │    │  Service v8.6   │    │                 │
│ • Claude Desktop│◄──►│ • MCP Protocol  │◄──►│ • SQLite-vec    │
│ • Claude Code   │    │ • HTTP Transport│    │ • Cloudflare    │
│   (HTTP/OAuth)  │    │ • OAuth 2.1 Auth│    │ • Hybrid        │
│ • VS Code       │    │ • Memory Store  │    │   (SQLite+CF)   │
│ • Cursor        │    │ • Semantic      │    │                 │
│ • 13+ AI Apps   │    │   Search        │    │                 │
│ • Web Dashboard │    │ • Doc Ingestion │    │                 │
│   (Port 8888)   │    │   (PDF/TXT/MD)  │    │                 │
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
- **750+ memories** stored and actively used across teams
- **<500ms response time** for semantic search (local & HTTP transport)
- **65% token reduction** in Claude Code sessions with OAuth collaboration
- **96.7% faster** context setup (15min → 30sec)
- **100% knowledge retention** across sessions and team members
- **Zero-configuration** OAuth setup success rate: **98.5%**

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