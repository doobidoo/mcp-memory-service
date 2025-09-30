# Docker Optimized Build Guide

## Overview

The MCP Memory Service Docker images have been optimized to use **sqlite_vec** as the default storage backend, removing the heavy ChromaDB dependencies. This results in:

- **70-80% faster build times**
- **1-2GB smaller image size**
- **Lower memory footprint**
- **Faster container startup**

## Building Docker Images

### Standard Build (SQLite-vec only)

```bash
# Build the optimized image
docker build -f tools/docker/Dockerfile -t mcp-memory-service:latest .

# Or use docker-compose
docker-compose -f tools/docker/docker-compose.yml build
```

### Slim Build (Ultra-lightweight with ONNX)

```bash
# Build the slim image (no PyTorch)
docker build -f tools/docker/Dockerfile.slim -t mcp-memory-service:slim .
```

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

### Option 1: Install ChromaDB at Runtime

```dockerfile
# In your Dockerfile, add:
RUN pip install chromadb>=0.5.0
```

### Option 2: Use Full Installation

```bash
# Install locally with ChromaDB support
python scripts/installation/install.py --with-chromadb

# Then build Docker image
docker build -t mcp-memory-service:chromadb .
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

### Issue: Need ChromaDB for multi-client support

If you specifically need ChromaDB for multi-client support:

1. Install with ChromaDB flag:
```bash
python scripts/installation/install.py --with-chromadb
```

2. Set environment variable:
```bash
export MCP_MEMORY_STORAGE_BACKEND=chromadb
```

3. Build Docker image with ChromaDB dependencies

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

1. **Use sqlite_vec for single-user deployments** - Fast and lightweight
2. **Use Cloudflare for production** - Global distribution without heavy dependencies
3. **Only use ChromaDB when necessary** - Multi-client local deployments
4. **Leverage Docker layer caching** - Build dependencies separately
5. **Use slim images for production** - Minimal attack surface

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