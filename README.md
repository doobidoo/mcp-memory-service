# MCP Memory Service

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/doobidoo/mcp-memory-service?style=social)](https://github.com/doobidoo/mcp-memory-service/stargazers)
[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen?style=flat&logo=checkmark)](https://github.com/doobidoo/mcp-memory-service#-in-production)

[![Works with Claude](https://img.shields.io/badge/Works%20with-Claude-blue)](https://claude.ai)
[![Works with Cursor](https://img.shields.io/badge/Works%20with-Cursor-orange)](https://cursor.sh)
[![MCP Protocol](https://img.shields.io/badge/MCP-Compatible-4CAF50?style=flat)](https://modelcontextprotocol.io/)
[![Multi-Client](https://img.shields.io/badge/Multi--Client-13+%20Apps-FF6B35?style=flat)](https://github.com/doobidoo/mcp-memory-service/wiki)

**Universal MCP memory service** providing **semantic memory search** and persistent storage for **AI assistants**. Works with **Claude Desktop, VS Code, Cursor, Continue, and 13+ AI applications** with **SQLite-vec** for fast local search and **Cloudflare** for global distribution.

<img width="240" alt="MCP Memory Service" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />

## 🚀 Quick Start (2 minutes)

### Universal Installer (Recommended)
```bash
# Clone and install with automatic platform detection
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python install.py
```

### Docker (Fastest)
```bash
# For MCP protocol (Claude Desktop)
docker-compose up -d

# For HTTP API (Web Dashboard)
docker-compose -f docker-compose.http.yml up -d
```

### Smithery (Claude Desktop)
```bash
# Auto-install for Claude Desktop
npx -y @smithery/cli install @doobidoo/mcp-memory-service --client claude
```

## 📚 Complete Documentation

**👉 Visit our comprehensive [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki) for detailed guides:**

### 🚀 Setup & Installation
- **[📋 Installation Guide](https://github.com/doobidoo/mcp-memory-service/wiki/01-Installation-Guide)** - Complete installation for all platforms and use cases
- **[🖥️ Platform Setup Guide](https://github.com/doobidoo/mcp-memory-service/wiki/02-Platform-Setup-Guide)** - Windows, macOS, and Linux optimizations  
- **[🔗 Integration Guide](https://github.com/doobidoo/mcp-memory-service/wiki/03-Integration-Guide)** - Claude Desktop, Claude Code, VS Code, and more

### 🧠 Advanced Topics
- **[🧠 Advanced Configuration](https://github.com/doobidoo/mcp-memory-service/wiki/04-Advanced-Configuration)** - Integration patterns, best practices, workflows
- **[⚡ Performance Optimization](https://github.com/doobidoo/mcp-memory-service/wiki/05-Performance-Optimization)** - Speed up queries, optimize resources, scaling
- **[👨‍💻 Development Reference](https://github.com/doobidoo/mcp-memory-service/wiki/06-Development-Reference)** - Claude Code hooks, API reference, debugging

### 🔧 Help & Reference
- **[🔧 Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING)** - Solutions for common issues
- **[❓ FAQ](https://github.com/doobidoo/mcp-memory-service/wiki/08-FAQ)** - Frequently asked questions
- **[📝 Examples](https://github.com/doobidoo/mcp-memory-service/wiki/09-Examples)** - Practical code examples and workflows

## ✨ Key Features

### 🧠 **Intelligent Memory Management**
- **Semantic search** with vector embeddings
- **Natural language time queries** ("yesterday", "last week")
- **Tag-based organization** with smart categorization
- **Memory consolidation** with dream-inspired algorithms

### 🔗 **Universal Compatibility**
- **Claude Desktop** - Native MCP integration
- **Claude Code** - Memory-aware development with hooks
- **VS Code, Cursor, Continue** - IDE extensions
- **13+ AI applications** - REST API compatibility

### 💾 **Flexible Storage**
- **SQLite-vec** - Fast local storage (recommended)
- **ChromaDB** - Multi-client collaboration  
- **Cloudflare** - Global edge distribution
- **Automatic backups** and synchronization

### 🚀 **Production Ready**
- **Cross-platform** - Windows, macOS, Linux
- **Service installation** - Auto-start background operation
- **HTTPS/SSL** - Secure connections
- **Docker support** - Easy deployment

## 💡 Basic Usage

```bash
# Store a memory
uv run memory store "Fixed race condition in authentication by adding mutex locks"

# Search for relevant memories  
uv run memory recall "authentication race condition"

# Search by tags
uv run memory search --tags python debugging

# Check system health
uv run memory health
```

## 🔧 Configuration

### Claude Desktop Integration
Add to your Claude Desktop config (`~/.claude/config.json`):

```json
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
│   AI Clients    │    │  MCP Protocol   │    │ Storage Backend │
│                 │    │                 │    │                 │
│ • Claude Desktop│◄──►│ • Memory Store  │◄──►│ • SQLite-vec    │
│ • Claude Code   │    │ • Semantic      │    │ • ChromaDB      │
│ • VS Code       │    │   Search        │    │ • Cloudflare    │
│ • Cursor        │    │ • Tag System    │    │                 │
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

## 📊 In Production

**Real-world metrics from active deployments:**
- **750+ memories** stored and actively used
- **<500ms response time** for semantic search
- **65% token reduction** in Claude Code sessions  
- **96.7% faster** context setup (15min → 30sec)
- **100% knowledge retention** across sessions

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