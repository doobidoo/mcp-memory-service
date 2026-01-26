# CLAUDE.md

MCP Memory Service - Semantic memory server for Claude with SQLite-vec and Qdrant backends.

## Quick Start

```bash
# Start server
uv run memory server

# Run tests
pytest tests/

# Check status
uv run memory status
```

## Storage Backends

| Backend | Performance | Use Case |
|---------|-------------|----------|
| **SQLite-Vec** | 5ms read | Local dev, single-node (DEFAULT) |
| **Qdrant** | 5ms read | Production, ARM64-optimized |

Backend selection via env: `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec` (or `qdrant`)

## Architecture

```
mcp_memory_service/
├── mcp_server.py         # FastMCP 2.0 MCP protocol adapter
├── unified_server.py     # HTTP + MCP dual-mode server
├── services/
│   └── memory_service.py # Core business logic
├── storage/
│   ├── base.py           # MemoryStorage ABC
│   ├── sqlite_vec.py     # SQLite-vec backend
│   └── qdrant_storage.py # Qdrant backend
├── config.py             # pydantic-settings configuration
└── cli/main.py           # CLI entry point
```

## Embedding Models

Default: `intfloat/e5-base-v2` (768-dim, ~63 MTEB avg)

| Model | Dims | Use Case |
|-------|------|----------|
| **intfloat/e5-base-v2** ⭐ | 768 | Default (best balance) |
| **intfloat/e5-small-v2** | 384 | Speed > accuracy |
| **intfloat/e5-large-v2** | 1024 | Best quality |

## Environment Variables

```bash
# Backend selection
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec  # sqlite_vec|qdrant

# HTTP server
export MCP_HTTP_ENABLED=true
export MCP_HTTP_PORT=8000
export MCP_HTTP_HOST=0.0.0.0

# Security
export MCP_MEMORY_API_KEY="your-api-key"  # Optional API key auth
export MCP_MEMORY_CORS_ORIGINS='[]'       # Default: no CORS

# Debug
export MCP_MEMORY_EXPOSE_DEBUG_TOOLS=false
```

## Development

- Storage backends implement `BaseStorage` Protocol (structural typing)
- All features require tests
- Version updates: `__init__.py` → `pyproject.toml` → `uv lock`
- Config uses pydantic-settings with thread-safe lazy loading

## Key Files

- `src/mcp_memory_service/config.py` - All configuration (pydantic-settings)
- `src/mcp_memory_service/mcp_server.py` - MCP tool definitions
- `src/mcp_memory_service/storage/factory.py` - Backend factory

## Documentation

- **Wiki**: https://github.com/doobidoo/mcp-memory-service/wiki
