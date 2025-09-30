# Docker Optimized Build Guide

## Overview

The MCP Memory Service Docker images have been optimized to use **sqlite_vec** as the default storage backend with **lightweight ONNX embeddings**, removing all heavy ML dependencies (ChromaDB, PyTorch, sentence-transformers). This results in:

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

The Docker images default to **sqlite_vec** for optimal performance. If you need ChromaDB support:

### Option 1: Install ML Dependencies at Runtime

```dockerfile
# Base installation (SQLite-vec only, no embeddings)
RUN pip install -e .

# Add ONNX Runtime for lightweight embeddings (recommended)
RUN pip install -e .[sqlite]

# Add full ML capabilities (PyTorch + sentence-transformers)
RUN pip install -e .[sqlite-ml]

# Add ChromaDB backend support (includes full ML stack)
RUN pip install -e .[chromadb]
```

### Option 2: Use Full Installation

```bash
# Install locally with lightweight SQLite-vec (default)
python scripts/installation/install.py

# Install locally with full ML support for SQLite-vec
python scripts/installation/install.py --with-ml

# Install locally with ChromaDB support (includes ML)
python scripts/installation/install.py --with-chromadb

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

| Image Type | With ChromaDB | Without ChromaDB | Reduction |
|------------|---------------|------------------|-----------|
| Standard   | ~2.5GB        | ~800MB          | 68%       |
| Slim       | N/A           | ~400MB          | N/A       |

## Build Time Comparison

| Build Type | With ChromaDB | Without ChromaDB | Speedup |
|------------|---------------|------------------|---------|
| Standard   | ~10-15 min    | ~2-3 min        | 5x      |
| Slim       | N/A           | ~1-2 min        | N/A     |

## Migration from ChromaDB

If you have existing ChromaDB data:

1. Export data from ChromaDB container:
```bash
docker exec mcp-memory-chromadb python scripts/backup/backup_memories.py
```

2. Start new sqlite_vec container:
```bash
docker-compose -f tools/docker/docker-compose.yml up -d
```

3. Import data to sqlite_vec:
```bash
docker exec mcp-memory-sqlite python scripts/backup/restore_memories.py
```

## Troubleshooting

### Issue: Need ML Capabilities or ChromaDB

If you need semantic search, embeddings, or ChromaDB support:

1. Install with ML dependencies:
```bash
# For ML capabilities only
python scripts/installation/install.py --with-ml

# For ChromaDB (includes ML automatically)
python scripts/installation/install.py --with-chromadb
```

2. Set environment variables:
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec  # or chromadb
```

3. Build Docker image with full dependencies

### Issue: Import error for ChromaDB

If you see ChromaDB import errors:

```
ImportError: ChromaDB backend selected but chromadb package not installed
```

This is expected behavior. The system will:
1. Log a clear error message
2. Suggest installing with `--with-chromadb`
3. Recommend switching to sqlite_vec

## Best Practices

1. **Start with lightweight default** - No ML dependencies for basic functionality
2. **Add ML capabilities when needed** - Use `[ml]` optional dependencies for semantic search
3. **Use sqlite_vec for single-user deployments** - Fast and lightweight
4. **Use Cloudflare for production** - Global distribution without heavy dependencies
5. **Only use ChromaDB when necessary** - Multi-client local deployments
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