# Glama Deployment Guide

This guide provides instructions for deploying the MCP Memory Service on the Glama platform.

## Overview

The MCP Memory Service is now available on Glama at: https://glama.ai/mcp/servers/bzvl3lz34o

Glama is a directory for MCP servers that provides easy discovery and deployment options for users.

## Docker Configuration for Glama

### Primary Dockerfile

The repository includes an optimized Dockerfile specifically for Glama deployment:
- `Dockerfile` - Main production Dockerfile
- `Dockerfile.glama` - Glama-optimized version with enhanced labels and health checks

### Key Features

1. **Multi-platform Support**: Works on x86_64 and ARM64 architectures
2. **Health Checks**: Built-in health monitoring for container status
3. **Data Persistence**: Proper volume configuration for ChromaDB and backups
4. **Environment Configuration**: Pre-configured for optimal performance
5. **Security**: Minimal attack surface with slim Python base image

### Quick Start from Glama

Users can deploy the service using:

```bash
# Using the Glama-provided configuration
docker run -d -p 8000:8000 \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  doobidoo/mcp-memory-service:latest
```

### Environment Variables

The following environment variables are pre-configured:

| Variable | Value | Purpose |
|----------|-------|---------|
| `MCP_MEMORY_CHROMA_PATH` | `/app/chroma_db` | ChromaDB storage location |
| `MCP_MEMORY_BACKUPS_PATH` | `/app/backups` | Backup storage location |
| `DOCKER_CONTAINER` | `1` | Indicates Docker environment |
| `CHROMA_TELEMETRY_IMPL` | `none` | Disables ChromaDB telemetry |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Enables MPS fallback for Apple Silicon |

### Standalone Mode

For deployment without an active MCP client, use:

```bash
docker run -d -p 8000:8000 \
  -e MCP_STANDALONE_MODE=1 \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  doobidoo/mcp-memory-service:latest
```

## Glama Platform Integration

### Server Verification

The Dockerfile passes all Glama server checks:
- ✅ Valid Dockerfile syntax
- ✅ Proper base image
- ✅ Security best practices
- ✅ Health check implementation
- ✅ Volume configuration
- ✅ Port exposure

### User Experience

Glama users benefit from:
1. **One-click deployment** from the Glama interface
2. **Pre-configured settings** for immediate use
3. **Documentation integration** with setup instructions
4. **Community feedback** and ratings
5. **Version tracking** and update notifications

### Monitoring and Health

The Docker image includes health checks that verify:
- Python environment is working
- MCP Memory Service can be imported
- Dependencies are properly loaded

## Maintenance

### Updates

The Glama listing is automatically updated when:
1. New versions are tagged in the GitHub repository
2. Docker images are published to Docker Hub
3. Documentation is updated

### Support

For Glama-specific issues:
1. Check the Glama platform documentation
2. Verify Docker configuration
3. Review container logs for errors
4. Test with standalone mode for debugging

## Contributing

To improve the Glama integration:
1. Test the deployment on different platforms
2. Provide feedback on the installation experience
3. Suggest improvements to the Docker configuration
4. Report any platform-specific issues

The goal is to make the MCP Memory Service as accessible as possible to the 60k+ monthly Glama users.
