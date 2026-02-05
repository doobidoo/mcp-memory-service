#!/bin/bash

# Setup script for MCP Memory Service with Consolidation and mDNS
echo "Setting up MCP Memory Service with Consolidation and mDNS HTTPS..."

# Enable consolidation system
export MCP_CONSOLIDATION_ENABLED=true

# Configure consolidation settings
export MCP_DECAY_ENABLED=true
export MCP_RETENTION_CRITICAL=365
export MCP_RETENTION_REFERENCE=180
export MCP_RETENTION_STANDARD=30
export MCP_RETENTION_TEMPORARY=7

export MCP_ASSOCIATIONS_ENABLED=true
export MCP_ASSOCIATION_MIN_SIMILARITY=0.3
export MCP_ASSOCIATION_MAX_SIMILARITY=0.7
export MCP_ASSOCIATION_MAX_PAIRS=100

export MCP_CLUSTERING_ENABLED=true
export MCP_CLUSTERING_MIN_SIZE=5
export MCP_CLUSTERING_ALGORITHM=dbscan

export MCP_COMPRESSION_ENABLED=true
export MCP_COMPRESSION_MAX_LENGTH=500
export MCP_COMPRESSION_PRESERVE_ORIGINALS=true

export MCP_FORGETTING_ENABLED=true
export MCP_FORGETTING_RELEVANCE_THRESHOLD=0.1
export MCP_FORGETTING_ACCESS_THRESHOLD=90

# Set consolidation schedule (cron-like)
export MCP_SCHEDULE_DAILY="02:00"
export MCP_SCHEDULE_WEEKLY="SUN 03:00"
export MCP_SCHEDULE_MONTHLY="01 04:00"

# Configure mDNS multi-client server with HTTPS
export MCP_MDNS_ENABLED=true
export MCP_MDNS_SERVICE_NAME="memory"
export MCP_HTTPS_ENABLED=true

# HTTP server configuration
export MCP_HTTP_ENABLED=true
export MCP_HTTP_HOST=0.0.0.0
export MCP_HTTP_PORT=8000

# Storage backend
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# API security
export MCP_API_KEY="$(openssl rand -base64 32)"

echo "Configuration set! Environment variables:"
echo "- Consolidation enabled: $MCP_CONSOLIDATION_ENABLED"
echo "- mDNS enabled: $MCP_MDNS_ENABLED"
echo "- HTTPS enabled: $MCP_HTTPS_ENABLED"
echo "- Service name: $MCP_MDNS_SERVICE_NAME"
echo "- API Key generated: [SET]"
echo ""
echo "Starting MCP Memory Service HTTP server..."

# Activate virtual environment and start the server
source venv/bin/activate && python scripts/run_http_server.py
