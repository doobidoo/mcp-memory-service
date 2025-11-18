# CLAUDE.md

MCP Memory Service - Semantic memory server for Claude with SQLite-vec, Cloudflare, Hybrid, and Qdrant backends.

## Quick Start

```bash
# Install (interactive backend selection)
python scripts/installation/install.py

# Start server
uv run memory server

# Run tests
pytest tests/

# Validate config
python scripts/validation/validate_configuration_complete.py
```

## Storage Backends

| Backend | Performance | Use Case |
|---------|-------------|----------|
| **Hybrid** | 5ms read | Production (RECOMMENDED) |
| **Qdrant** | 5ms read | ARM64-optimized |
| **SQLite-Vec** | 5ms read | Local dev |
| **Cloudflare** | Network | Cloud-only |

## Environment Variables

```bash
# Backend selection
export MCP_MEMORY_STORAGE_BACKEND=hybrid  # hybrid|cloudflare|sqlite_vec|qdrant

# Cloudflare (required for hybrid/cloudflare)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-db-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# Optional
export MCP_HTTP_ENABLED=true
export MCP_MEMORY_EXPOSE_DEBUG_TOOLS=false
```

## Development

- Storage backends must implement abstract base class
- All features require tests
- Version updates: `__init__.py` → `pyproject.toml` → `uv lock`
- Use `claude /memory-store` to capture decisions

## Documentation

- **Wiki**: https://github.com/doobidoo/mcp-memory-service/wiki
- **Troubleshooting**: Run `python scripts/validation/diagnose_backend_config.py`
- **Detailed context**: Retrieve memories tagged `claude-code-reference`
