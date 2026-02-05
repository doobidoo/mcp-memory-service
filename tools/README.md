# Development Tools and Utilities

This directory contains development tools, build utilities, and deployment configurations for MCP Memory Service.

## Directory Structure

### `/build/` - Build Tools
- `setup.py` - Python package build configuration
- Build scripts and packaging utilities

### `/deployments/` - Deployment Tools
- `cloudflare/` - Cloudflare Workers deployment configuration
- Cloud platform deployment scripts and configurations

### `/docker/` - Docker Tools
- Multiple Docker configurations and Dockerfiles
- Docker Compose files for different deployment scenarios
- Docker utility scripts and entrypoints

## Usage

### Build Tools
```bash
# Build Python package
cd tools/build
python setup.py sdist bdist_wheel
```

### Docker Deployment
```bash
# Use various Docker configurations
cd tools/docker
docker-compose -f docker-compose.yml up
docker-compose -f docker-compose.standalone.yml up
```

### Cloudflare Workers
```bash
# Deploy to Cloudflare Workers
cd tools/deployments/cloudflare
npm install
wrangler deploy
```

## Related Documentation

- [Docker Deployment Guide](../docs/deployment/docker.md) - Comprehensive Docker setup
- [Installation Guide](../docs/installation/master-guide.md) - General installation
- [Development Guide](../docs/technical/development.md) - Development setup
