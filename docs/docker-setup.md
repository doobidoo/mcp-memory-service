# Docker Setup Guide

Complete Docker Compose setup for MCP Memory Service with ARM64 support.

## Files

- **`docker-compose.yml`** - Main configuration (amd64 + arm64)
- **`docker-compose.override.yml`** - ARM64-specific optimizations
- **`Dockerfile`** - Multi-stage production build
- **`.dockerignore`** - Optimized build context

## Quick Start

### Standard Build (amd64)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f mcp-memory-http

# Health check
curl http://localhost:8000/api/health

# Stop services
docker-compose down
```

### ARM64 Build (Apple Silicon, Graviton, Raspberry Pi)

```bash
# Start with ARM64 optimizations
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# View logs
docker-compose -f docker-compose.yml -f docker-compose.override.yml logs -f mcp-memory-http

# Health check
curl http://localhost:8000/api/health
```

## Configuration

### Environment Variables

Edit `docker-compose.yml` to configure:

```yaml
environment:
  # Storage backend (default: qdrant)
  - MCP_MEMORY_STORAGE_BACKEND=qdrant|sqlite_vec|cloudflare|hybrid

  # Qdrant settings
  - MCP_QDRANT_STORAGE_PATH=/data/qdrant
  - MCP_QDRANT_QUANTIZATION_ENABLED=false

  # Cloudflare (if using cloudflare/hybrid backend)
  - CLOUDFLARE_API_TOKEN=${CLOUDFLARE_API_TOKEN}
  - CLOUDFLARE_ACCOUNT_ID=${CLOUDFLARE_ACCOUNT_ID}
  - CLOUDFLARE_D1_DATABASE_ID=${CLOUDFLARE_D1_DATABASE_ID}
  - CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

  # Logging
  - MCP_LOG_LEVEL=INFO|DEBUG
```

### Cloudflare Backend

For Cloudflare backend, create `.env` file:

```bash
cat > .env << 'EOF'
CLOUDFLARE_API_TOKEN=your-api-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_D1_DATABASE_ID=your-d1-id
EOF

docker-compose up -d
```

## Building Images

### Build locally

```bash
# Standard build
docker-compose build

# Rebuild without cache
docker-compose build --no-cache

# ARM64-optimized build
docker-compose -f docker-compose.yml -f docker-compose.override.yml build
```

### Multi-arch builds (requires buildx)

```bash
# Create builder
docker buildx create --name memorybuilder

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t mcp-memory-service:latest \
  --push \
  .
```

## Volumes

Services use named volumes for persistence:

```
mcp-memory-data              # Qdrant data (http service)
mcp-memory-data-mcp         # Qdrant data (MCP service)
mcp-memory-backups          # SQLite backups
```

### Inspect volumes

```bash
# List volumes
docker volume ls | grep mcp-memory

# View volume data
docker volume inspect mcp-memory-data

# Mount for inspection
docker run -v mcp-memory-data:/data alpine sh -c "ls -la /data"
```

### Backup volumes

```bash
# Backup Qdrant data
docker run --rm -v mcp-memory-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant-$(date +%Y%m%d).tar.gz -C /data .

# Restore Qdrant data
docker run --rm -v mcp-memory-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/qdrant-20240101.tar.gz -C /data
```

## Services

### HTTP Server (mcp-memory-http)

REST API on port 8000

```bash
# Health check
curl http://localhost:8000/api/health

# List memories
curl http://localhost:8000/api/memories

# Dashboard
open http://localhost:8000
```

### MCP Server (mcp-memory-mcp)

Model Context Protocol server (stdio-based)

- Used by Claude Desktop, VS Code, other MCP clients
- No exposed ports (uses stdin/stdout)
- Shares same Qdrant storage

## Networking

### Expose to network

Edit `docker-compose.yml` ports:

```yaml
# Default: localhost only
ports:
  - "127.0.0.1:8000:8000"

# Expose to network
ports:
  - "0.0.0.0:8000:8000"

# Custom port
ports:
  - "8080:8000"
```

### Connect containers

Services communicate via service name:

```bash
# From inside container
curl http://mcp-memory-http:8000/api/health
```

## Resource Limits

Adjust resource allocation in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'        # Max CPU cores
      memory: 2G       # Max memory
    reservations:
      cpus: '1'        # Guaranteed CPU
      memory: 1G       # Guaranteed memory
```

ARM64-specific limits in `docker-compose.override.yml` (more conservative).

## Troubleshooting

### Check logs

```bash
# HTTP server
docker-compose logs mcp-memory-http

# MCP server
docker-compose logs mcp-memory-mcp

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Health issues

```bash
# Manual health check
docker exec mcp-memory-http curl http://localhost:8000/api/health

# Check container status
docker ps | grep mcp-memory

# Inspect container
docker inspect mcp-memory-http
```

### Storage issues

```bash
# Check volume usage
docker system df

# Clean up unused volumes
docker volume prune

# Check Qdrant directory
docker exec mcp-memory-http ls -la /data/qdrant
```

### Network issues

```bash
# Check network
docker network inspect mcp-memory-service_default

# Test connectivity
docker exec mcp-memory-http curl http://localhost:8000/api/health

# DNS from container
docker exec mcp-memory-http nslookup mcp-memory-http
```

## Production Deployment

### Use specific versions

```yaml
services:
  mcp-memory-http:
    image: mcp-memory-service:8.13.0
```

### Enable restart policy

```yaml
restart: always  # or unless-stopped
```

### Add resource monitoring

```bash
# CPU and memory usage
docker stats mcp-memory-http

# Monitor with docker-compose
docker-compose stats
```

### Logging

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## Migration from Native to Docker

If you have existing data:

```bash
# Stop native server
pkill -f "python3 scripts/server/run_http_server.py"

# Copy Qdrant data to Docker volume
docker volume create mcp-memory-data
docker run -v ~/Library/Application\ Support/mcp-memory/qdrant:/source \
  -v mcp-memory-data:/dest \
  alpine cp -r /source/* /dest/

# Start Docker services
docker-compose up -d

# Verify data
curl http://localhost:8000/api/health
```

## Performance Tuning

### ARM64 Optimization

Already configured in `docker-compose.override.yml`:

- **Quantization enabled** - 32x memory savings
- **Reduced CPU threads** - Efficient on ARM64
- **Conservative resource limits** - Better for SoC environments

### amd64 Optimization

For high-performance amd64 deployments:

```yaml
environment:
  - MCP_QDRANT_QUANTIZATION_ENABLED=false  # Full precision
  - OMP_NUM_THREADS=4  # Increase for more cores

deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## See Also

- [Configuration Guide](./docs/configuration-fix-embedding-model.md)
- [Troubleshooting Guide](./docs/troubleshooting/)
- [Architecture](./CLAUDE.md#architecture)
