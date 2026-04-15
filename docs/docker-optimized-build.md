# Docker Optimized Build Guide

## Overview

The MCP Memory Service Docker images have been optimized to use **sqlite_vec** as the default storage backend with **lightweight ONNX embeddings**, removing heavy ML dependencies (PyTorch, sentence-transformers) from the default build. This results in:

- **70-80% faster build times**
- **1-2GB smaller image size**
- **Lower memory footprint**
- **Faster container startup**

## Building Docker Images

### Standard Build (Optimized Default)

```bash
# Build the optimized image with lightweight embeddings
docker build -f tools/docker/Dockerfile -t mcp-memory-service:latest .

# Or use docker-compose
docker-compose -f tools/docker/docker-compose.yml build
```

**Includes**: SQLite-vec + ONNX Runtime for embeddings (~100MB total dependencies)

### Slim Build (Minimal Installation)

```bash
# Build the slim image without ML capabilities
docker build -f tools/docker/Dockerfile.slim -t mcp-memory-service:slim .
```

**Includes**: Core MCP Memory Service without embeddings (~50MB dependencies)

### Full ML Build (All features)

```bash
# Build with full ML capabilities (custom build)
docker build -f tools/docker/Dockerfile -t mcp-memory-service:full \
  --build-arg INSTALL_EXTRA="[sqlite-ml]" .
```

**Includes**: SQLite-vec (core) + PyTorch + sentence-transformers + ONNX (~2GB dependencies)

## Running Containers

### Using Docker Run

```bash
# Run with sqlite_vec backend
docker run -it \
  -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
  -v ./data:/app/data \
  mcp-memory-service:latest
```

### Using Docker Compose

```bash
# Start the service
docker-compose -f tools/docker/docker-compose.yml up -d

# View logs
docker-compose -f tools/docker/docker-compose.yml logs -f

# Stop the service
docker-compose -f tools/docker/docker-compose.yml down
```

## Storage Backend Configuration

The Docker images default to **sqlite_vec** for optimal performance. For Cloudflare/Hybrid backends, set the environment variables described in [cloudflare-setup.md](cloudflare-setup.md).

### Option 1: Install ML Dependencies at Runtime

```dockerfile
# Base installation (SQLite-vec only, no embeddings)
RUN pip install -e .

# Add ONNX Runtime for lightweight embeddings (recommended)
RUN pip install -e .[sqlite]

# Add full ML capabilities (PyTorch + sentence-transformers)
RUN pip install -e .[sqlite-ml]
```

### Option 2: Use Full Installation

```bash
# Install locally with lightweight SQLite-vec (default)
python scripts/installation/install.py

# Install locally with full ML support for SQLite-vec
python scripts/installation/install.py --with-ml

# Then build Docker image
docker build -t mcp-memory-service:full .
```

## Environment Variables

```yaml
environment:
  # Storage backend (sqlite_vec recommended)
  - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

  # Data paths
  - MCP_MEMORY_SQLITE_PATH=/app/data/sqlite_vec.db
  - MCP_MEMORY_BACKUPS_PATH=/app/data/backups

  # Performance
  - MCP_MEMORY_USE_ONNX=1  # For CPU-only deployments

  # Logging
  - LOG_LEVEL=INFO
```

## Multi-Architecture Builds

The optimized Dockerfiles support multi-platform builds:

```bash
# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f tools/docker/Dockerfile \
  -t mcp-memory-service:latest \
  --push .
```

## Image Sizes Comparison

| Image Type | With full ML stack | Default (ONNX only) | Reduction |
|------------|--------------------|---------------------|-----------|
| Standard   | ~2.5GB             | ~800MB              | 68%       |
| Slim       | N/A                | ~400MB              | N/A       |

## Build Time Comparison

| Build Type | With full ML stack | Default (ONNX only) | Speedup |
|------------|--------------------|---------------------|---------|
| Standard   | ~10-15 min         | ~2-3 min            | 5x      |
| Slim       | N/A                | ~1-2 min            | N/A     |

## Migrating Legacy ChromaDB Data

If you still have data from a pre-v8.0 ChromaDB deployment, follow the dedicated migration guide: [guides/chromadb-migration.md](guides/chromadb-migration.md). The script lives on the `chromadb-legacy` branch; once exported you can start a sqlite_vec container and import from the JSON backup.

## Troubleshooting

### Issue: Need Full ML Capabilities

If the default ONNX embeddings are not enough (e.g. you want PyTorch-based sentence-transformers for custom models):

1. Install with ML dependencies:
```bash
# For ML capabilities on top of SQLite-vec
python scripts/installation/install.py --with-ml
```

2. Set environment variables:
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec  # or hybrid / cloudflare
```

3. Build Docker image with full dependencies

## Best Practices

1. **Start with lightweight default** - ONNX embeddings cover most use cases
2. **Add ML capabilities when needed** - Use `[ml]` optional dependencies for semantic search
3. **Use sqlite_vec for single-user deployments** - Fast and lightweight
4. **Use Cloudflare for edge / cloud-only deployments** - Global distribution without local state
5. **Use Hybrid for production** - 5 ms local reads + background Cloudflare sync (recommended)
6. **Leverage Docker layer caching** - Build dependencies separately
7. **Use slim images for production** - Minimal attack surface

## CI/CD Integration

For GitHub Actions:

```yaml
- name: Build optimized Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: ./tools/docker/Dockerfile
    platforms: linux/amd64,linux/arm64
    push: true
    tags: ${{ steps.meta.outputs.tags }}
    build-args: |
      SKIP_MODEL_DOWNLOAD=true
```

The `SKIP_MODEL_DOWNLOAD=true` build arg further reduces build time by deferring model downloads to runtime.