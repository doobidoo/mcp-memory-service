# Docker Deployment Guide

This comprehensive guide covers deploying MCP Memory Service using Docker, including various configurations for different use cases and environments.

## Overview

MCP Memory Service provides Docker support with multiple deployment configurations:

- **Standard Mode**: For MCP clients (Claude Desktop, VS Code, etc.)
- **Standalone Mode**: For testing and development (prevents boot loops)
- **HTTP/SSE Mode**: For web services and multi-client access
- **Production Mode**: For scalable server deployments

## Prerequisites

- **Docker** 20.10+ installed on your system
- **Docker Compose** 2.0+ (recommended for simplified deployment)
- Basic knowledge of Docker concepts
- Sufficient disk space for Docker images and container volumes

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Start with standard configuration
docker-compose up -d

# View logs
docker-compose logs -f
```

This will:
- Build a Docker image for the Memory Service
- Create persistent volumes for the database and backups
- Start the service configured for MCP clients

## Docker Compose Configurations

### 1. Standard Configuration (`docker-compose.yml`)

**Best for**: MCP clients like Claude Desktop, VS Code with MCP extension

```yaml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    stdin_open: true
    tty: true
    volumes:
      - ./data/chroma_db:/app/chroma_db
      - ./data/backups:/app/backups
    environment:
      - MCP_MEMORY_STORAGE_BACKEND=chromadb
    restart: unless-stopped
```

```bash
# Deploy standard configuration
docker-compose up -d
```

### 2. Standalone Configuration (`docker-compose.standalone.yml`)

**Best for**: Testing, development, and preventing boot loops when no MCP client is connected

```yaml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    stdin_open: true
    tty: true
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma_db:/app/chroma_db
      - ./data/backups:/app/backups
    environment:
      - MCP_STANDALONE_MODE=1
      - MCP_HTTP_HOST=0.0.0.0
      - MCP_HTTP_PORT=8000
    restart: unless-stopped
```

```bash
# Deploy standalone configuration
docker-compose -f docker-compose.standalone.yml up -d

# Test connectivity
curl http://localhost:8000/health
```

### 3. UV Configuration (`docker-compose.uv.yml`)

**Best for**: Enhanced dependency management with UV package manager

```yaml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    stdin_open: true
    tty: true
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma_db:/app/chroma_db
      - ./data/backups:/app/backups
    environment:
      - UV_ACTIVE=1
      - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
    restart: unless-stopped
```

### 4. Python Path Configuration (`docker-compose.pythonpath.yml`)

**Best for**: Custom Python path configurations and development mode

```bash
# Deploy with Python path configuration
docker-compose -f docker-compose.pythonpath.yml up -d
```

## Manual Docker Commands

### Basic Docker Deployment

```bash
# Build the Docker image
docker build -t mcp-memory-service .

# Create directories for persistent storage
mkdir -p ./data/chroma_db ./data/backups

# Run in standard mode (for MCP clients)
docker run -d --name memory-service \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  -e MCP_MEMORY_STORAGE_BACKEND=chromadb \
  --stdin --tty \
  mcp-memory-service

# Run in standalone/HTTP mode
docker run -d -p 8000:8000 --name memory-service \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  -e MCP_STANDALONE_MODE=1 \
  -e MCP_HTTP_HOST=0.0.0.0 \
  -e MCP_HTTP_PORT=8000 \
  --stdin --tty \
  mcp-memory-service
```

### Using Specific Docker Images

```bash
# Use pre-built Glama deployment image
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e MCP_API_KEY=your-api-key \
  --name memory-service \
  mcp-memory-service:glama

# Use SQLite-vec optimized image
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
  --name memory-service \
  mcp-memory-service:sqlite-vec
```

## Environment Configuration

### Core Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_STORAGE_BACKEND` | `chromadb` | Storage backend (chromadb, sqlite_vec) |
| `MCP_HTTP_HOST` | `0.0.0.0` | HTTP server bind address |
| `MCP_HTTP_PORT` | `8000` | HTTP server port |
| `MCP_STANDALONE_MODE` | `false` | Enable standalone HTTP mode |
| `MCP_API_KEY` | `none` | API key for authentication |

### Docker-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_CONTAINER` | `auto-detect` | Indicates running in Docker |
| `UV_ACTIVE` | `false` | Use UV package manager |
| `PYTHONPATH` | `/app/src` | Python module search path |

### Storage Configuration

```bash
# ChromaDB backend
docker run -d \
  -e MCP_MEMORY_STORAGE_BACKEND=chromadb \
  -e MCP_MEMORY_CHROMA_PATH=/app/chroma_db \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  mcp-memory-service

# SQLite-vec backend (recommended for containers)
docker run -d \
  -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
  -e MCP_MEMORY_SQLITE_PATH=/app/sqlite_data/memory.db \
  -v $(pwd)/data/sqlite_data:/app/sqlite_data \
  mcp-memory-service
```

## Production Deployment

### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: '3.8'
services:
  mcp-memory-service:
    image: mcp-memory-service:latest
    ports:
      - "8000:8000"
    environment:
      - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
      - MCP_HTTP_HOST=0.0.0.0
      - MCP_API_KEY=REDACTED
    volumes:
      - memory_data:/app/data
    secrets:
      - api_key
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

volumes:
  memory_data:

secrets:
  api_key:
    external: true
```

```bash
# Deploy to Docker Swarm
docker stack deploy -c docker-stack.yml mcp-memory
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-memory-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-memory-service
  template:
    metadata:
      labels:
        app: mcp-memory-service
    spec:
      containers:
      - name: mcp-memory-service
        image: mcp-memory-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: MCP_MEMORY_STORAGE_BACKEND
          value: "sqlite_vec"
        - name: MCP_HTTP_HOST
          value: "0.0.0.0"
        - name: MCP_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-api-key
              key: api-key
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 1Gi
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: mcp-memory-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-memory-service
spec:
  selector:
    app: mcp-memory-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Volume Management

### Data Persistence

```bash
# Create named volumes
docker volume create mcp_memory_data
docker volume create mcp_memory_backups

# Use named volumes
docker run -d \
  -v mcp_memory_data:/app/data \
  -v mcp_memory_backups:/app/backups \
  mcp-memory-service

# Backup volumes
docker run --rm \
  -v mcp_memory_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/mcp_memory_$(date +%Y%m%d).tar.gz /data
```

### Database Migration

```bash
# Export data from running container
docker exec memory-service python scripts/backup_memories.py

# Import data to new container
docker cp ./backup.json new-memory-service:/app/
docker exec new-memory-service python scripts/restore_memories.py /app/backup.json
```

## Monitoring and Logging

### Container Health Checks

```yaml
# Add to docker-compose.yml
services:
  mcp-memory-service:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Log Management

```bash
# View container logs
docker-compose logs -f mcp-memory-service

# Configure log rotation
docker-compose -f docker-compose.yml -f docker-compose.logging.yml up -d
```

```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  mcp-memory-service:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Monitoring with Prometheus

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  mcp-memory-service:
    environment:
      - MCP_MEMORY_ENABLE_METRICS=true
      - MCP_MEMORY_METRICS_PORT=9090
    ports:
      - "9090:9090"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Troubleshooting

### Common Docker Issues

#### 1. Container Boot Loop

**Symptom**: Container exits immediately with code 0

**Solution**: Use standalone mode or ensure proper TTY configuration:

```yaml
services:
  mcp-memory-service:
    stdin_open: true
    tty: true
    environment:
      - MCP_STANDALONE_MODE=1
```

#### 2. Permission Issues

**Symptom**: Permission denied errors in container

**Solution**: Fix volume permissions:

```bash
# Set proper ownership
sudo chown -R 1000:1000 ./data

# Or run with specific user
docker run --user $(id -u):$(id -g) mcp-memory-service
```

#### 3. Storage Backend Issues

**Symptom**: Database initialization failures

**Solution**: Use SQLite-vec for containers:

```bash
docker run -d \
  -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
  -v $(pwd)/data:/app/data \
  mcp-memory-service
```

#### 4. Network Connectivity

**Symptom**: Cannot connect to containerized service

**Solution**: Check port mapping and firewall:

```bash
# Test container networking
docker exec memory-service netstat -tlnp

# Check port mapping
docker port memory-service

# Test external connectivity
curl http://localhost:8000/health
```

#### 5. Model Download Issues

**Symptom**: `OSError: We couldn't connect to 'https://huggingface.co'` when starting container

**Issue**: Container cannot download sentence-transformer models due to network restrictions

**Solutions**:

1. **Pre-download models and mount cache (Recommended)**:

```bash
# Step 1: Download models on host machine first
python -c "from sentence_transformers import SentenceTransformer; \
          model = SentenceTransformer('all-MiniLM-L6-v2'); \
          print('Model downloaded successfully')"

# Step 2: Run container with model cache mounted
docker run -d --name memory-service \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -e MCP_MEMORY_STORAGE_BACKEND=chromadb \
  mcp-memory-service
```

2. **Configure proxy for Docker Desktop (Windows/Corporate networks)**:

```bash
# With proxy environment variables
docker run -d --name memory-service \
  -e HTTPS_PROXY=http://your-proxy:port \
  -e HTTP_PROXY=http://your-proxy:port \
  -e NO_PROXY=localhost,127.0.0.1 \
  -v $(pwd)/data:/app/data \
  mcp-memory-service
```

3. **Use offline mode with pre-cached models**:

```bash
# Ensure models are in mounted volume, then run offline
docker run -d --name memory-service \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -v $(pwd)/data:/app/data \
  mcp-memory-service
```

4. **Docker Compose with model cache**:

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    volumes:
      # Mount model cache from host
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ./data/chroma_db:/app/chroma_db
      - ./data/backups:/app/backups
    environment:
      - MCP_MEMORY_STORAGE_BACKEND=chromadb
      # Optional: force offline mode if models are pre-cached
      # - HF_HUB_OFFLINE=1
      # - TRANSFORMERS_OFFLINE=1
```

**Prevention**: Always mount the Hugging Face cache directory as a volume to persist models between container runs and avoid re-downloading.

### Diagnostic Commands

#### Container Status

```bash
# Check container status
docker ps -a

# View container logs
docker logs memory-service

# Execute commands in container
docker exec -it memory-service bash

# Check resource usage
docker stats memory-service
```

#### Service Health

```bash
# Test HTTP endpoints
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Check database connectivity
docker exec memory-service python -c "
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecStorage
storage = SqliteVecStorage()
print('Database accessible')
"
```

#### Model Cache Verification

```bash
# Check if models are cached on host
ls -la ~/.cache/huggingface/hub/

# Verify model availability in container
docker exec memory-service ls -la /root/.cache/huggingface/hub/

# Test model loading in container
docker exec memory-service python -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

## Security Considerations

### API Key Authentication

```bash
# Generate secure API key
API_KEY=$(openssl rand -hex 32)

# Use with Docker
docker run -d \
  -e MCP_API_KEY=$API_KEY \
  -p 8000:8000 \
  mcp-memory-service
```

### HTTPS Configuration

```yaml
# docker-compose.https.yml
services:
  mcp-memory-service:
    environment:
      - MCP_HTTPS_ENABLED=true
      - MCP_HTTP_PORT=8443
      - MCP_SSL_CERT_FILE=/app/certs/cert.pem
      - MCP_SSL_KEY_FILE=/app/certs/key.pem
    volumes:
      - ./certs:/app/certs:ro
    ports:
      - "8443:8443"
```

### Container Security

```bash
# Run with security options
docker run -d \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --read-only \
  --tmpfs /tmp \
  mcp-memory-service
```

## Performance Optimization

### Resource Limits

```yaml
services:
  mcp-memory-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Multi-Stage Builds

```dockerfile
# Optimized Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "src/mcp_memory_service/server.py"]
```

## Development Workflow

### Development with Docker

```bash
# Development with live reload
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker exec memory-service pytest tests/

# Debug with interactive shell
docker exec -it memory-service bash
```

### Building Custom Images

```bash
# Build with specific tag
docker build -t mcp-memory-service:v1.2.3 .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t mcp-memory-service:latest .

# Push to registry
docker push mcp-memory-service:latest
```

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Multi-Client Setup](../integration/multi-client.md) - Multi-client configuration
- [Ubuntu Setup](../platforms/ubuntu.md) - Ubuntu Docker deployment
- [Windows Setup](../platforms/windows.md) - Windows Docker deployment
- [Troubleshooting](../troubleshooting/general.md) - Docker-specific troubleshooting
