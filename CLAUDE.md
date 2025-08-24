# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **Note**: Comprehensive project context has been stored in memory with tags `claude-code-reference`. Use memory retrieval to access detailed information during development.

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop using ChromaDB and sentence transformers.

## Essential Commands

```bash
# Setup & Development
python install.py                    # Platform-aware installation
uv run memory server                 # Start server (v6.3.0+ consolidated CLI)
pytest tests/                       # Run tests
python scripts/verify_environment.py # Check environment

# Memory Operations (requires: python scripts/claude_commands_utils.py)
claude /memory-store "content"       # Store information
claude /memory-recall "query"        # Retrieve information
claude /memory-health               # Check service status

# Debug & Troubleshooting
npx @modelcontextprotocol/inspector uv run memory server  # MCP Inspector
df -h /                             # Check disk space (critical for Litestream)
journalctl -u mcp-memory-service -f # Monitor service logs
```

## Architecture

**Core Components:**
- **Server Layer**: MCP protocol implementation with async handlers and global caches (`src/mcp_memory_service/server.py`)
- **Storage Backends**: SQLite-Vec (fast, single-client), ChromaDB (multi-client), Cloudflare (production)
- **Web Interface**: FastAPI dashboard at `https://localhost:8443/` with REST API
- **Claude Code Hooks**: Session lifecycle management and automatic memory awareness

**Key Design Patterns:**
- Async/await for all I/O operations
- Type safety with Python 3.10+ hints
- Platform detection for hardware optimization (CUDA, MPS, DirectML, ROCm)
- Global model and embedding caches for performance

## Environment Variables

**Essential Configuration:**
```bash
# Storage Backend
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec  # sqlite_vec|chroma|cloudflare

# Web Interface
export MCP_HTTP_ENABLED=true                  # Enable HTTP server
export MCP_HTTPS_ENABLED=true                 # Enable HTTPS (production)
export MCP_API_KEY="$(openssl rand -base64 32)" # Generate secure API key

# Cloudflare Production (if using cloudflare backend)
export CLOUDFLARE_API_TOKEN="your-token"      # Required for Cloudflare backend
export CLOUDFLARE_ACCOUNT_ID="your-account"   # Required for Cloudflare backend
```

**Platform Support:** macOS (MPS/CPU), Windows (CUDA/DirectML/CPU), Linux (CUDA/ROCm/CPU)

## Storage Backends

| Backend | Performance | Use Case |
|---------|-------------|----------|
| SQLite-Vec | Fast (5ms read) | Development, single-client |
| ChromaDB | Medium (15ms read) | Multi-client, team collaboration |
| Cloudflare | Network dependent | Production, global scale |

## Development Guidelines

- Use `claude /memory-store` to capture decisions during development  
- Memory operations handle duplicates via content hashing
- Time parsing supports natural language ("yesterday", "last week")
- Storage backends must implement abstract base class
- All features require corresponding tests
- Use semantic commit messages for version management

## Key Endpoints

- **Health**: `https://localhost:8443/api/health`
- **Web UI**: `https://localhost:8443/`  
- **API**: `https://localhost:8443/api/memories`
- **Wiki**: `https://github.com/doobidoo/mcp-memory-service/wiki`

## Critical Issues

**Litestream Sync Issues:**
- Disk space >95%: Litestream fails silently, clean logs: `sudo rm /var/log/syslog.*`
- Database conflicts: Remove WAL files after DB replacement: `rm *.db-shm *.db-wal`
- MCP service caching: Restart Claude Desktop to refresh connections

**Common Problems:**
- Version mismatches: Use `__version__` imports, not hardcoded values
- Path conflicts: Check Claude Desktop config vs MCP tools paths
- Cache issues: Clear Python bytecode and UV caches, restart fresh

> **For detailed troubleshooting, architecture, and deployment guides, retrieve memories tagged with `claude-code-reference` or visit the project wiki.**