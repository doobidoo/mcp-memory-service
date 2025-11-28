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

## Embedding Models

Configure embedding model at Docker build time. Default: `intfloat/e5-base-v2` (768-dim, ~63 MTEB avg).

### Recommended Models (by priority)

| Model | Dims | MTEB | Speed | Integration | Use Case |
|-------|------|------|-------|-------------|----------|
| **intfloat/e5-base-v2** ⭐ | 768 | ~63 | 20.2ms | Simple | **Default (best balance)** |
| **infgrad/stella-base-en-v2** | 768 | 62.61 | Good | Simple | Alternative, no prefixes |
| **BAAI/bge-base-en-v1.5** | 768 | 63.55 | 22.5ms | **Complex*** | Highest MTEB |
| **intfloat/e5-small-v2** | 384 | ~60 | Fast | Simple | Speed > accuracy |
| **intfloat/e5-large-v2** | 1024 | ~64 | Slow | Simple | Best quality |
| **BAAI/bge-large-en-v1.5** | 1024 | 64.23 | Slower | **Complex*** | Best quality + MTEB |

**\*Complex**: BGE models require prefix prompts ("Represent this sentence for searching...") for optimal performance. E5/Stella work without prefixes.

### Why E5-base-v2 is Default

1. **No prefix complexity** - Works out of the box
2. **11% faster** than BGE-base (20.2ms vs 22.5ms)
3. **Nearly identical accuracy** - 63 vs 63.55 (<1% difference)
4. **Better CPU performance** - Optimized for ONNX
5. **Proven stable** - No prompt-engineering risks

### Build with Custom Model

```bash
# Default build (e5-base-v2)
docker build -t mcp-memory .

# Use different model at build time
docker build --build-arg EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 .

# Or override at runtime (must match build-time model dimensions)
docker run -e MCP_MEMORY_EMBEDDING_MODEL=intfloat/e5-base-v2 mcp-memory
```

### Model Migration

**WARNING**: Changing models requires migration if dimensions differ.

```bash
# Startup will detect mismatch and fail with instructions
# Example error: "Vector dimension mismatch: 384 != 768"

# Run migration (2K memories ~= 2 minutes)
docker run --rm -v mcp-data:/data mcp-memory:new-model \
  python -m mcp_memory_service.migrate_model \
  --old-model intfloat/e5-small \
  --new-model intfloat/e5-base-v2
```

**Safe workflow:**
1. Build new image: `docker build --build-arg EMBEDDING_MODEL=intfloat/e5-base-v2 .`
2. Stop old container: `docker stop mcp-memory`
3. Run migration script (if dimensions changed)
4. Start new container: `docker run mcp-memory:new-model`

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
