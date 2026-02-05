#!/bin/bash
# Example HTTPS startup script for MCP Memory Service
# Copy and customize this file for your deployment
#
# This example shows how to configure the MCP Memory Service with custom SSL certificates.
# For easy local development with trusted certificates, consider using mkcert:
# https://github.com/FiloSottile/mkcert

# Storage configuration
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# API authentication - CHANGE THIS TO A SECURE KEY!
# Generate a secure key with: openssl rand -base64 32
export MCP_API_KEY="your-secure-api-key-here"

# HTTPS configuration with custom certificates
export MCP_HTTPS_ENABLED=true
export MCP_HTTPS_PORT=8443

# SSL Certificate paths - UPDATE THESE PATHS TO YOUR CERTIFICATES
#
# For mkcert certificates (recommended for development):
# 1. Install mkcert: https://github.com/FiloSottile/mkcert#installation
# 2. Create local CA: mkcert -install
# 3. Generate certificate: mkcert your-domain.local localhost 127.0.0.1
# 4. Update paths below to point to generated certificate files
#
# Example paths:
# export MCP_SSL_CERT_FILE="/path/to/your-domain.local+2.pem"
# export MCP_SSL_KEY_FILE="/path/to/your-domain.local+2-key.pem"
#
# For production, use certificates from your certificate authority:
export MCP_SSL_CERT_FILE="/path/to/your/certificate.pem"
export MCP_SSL_KEY_FILE="/path/to/your/certificate-key.pem"

# Optional: Disable HTTP if only HTTPS is needed
export MCP_HTTP_ENABLED=false
export MCP_HTTP_PORT=8080

# mDNS service discovery
export MCP_MDNS_ENABLED=true
export MCP_MDNS_SERVICE_NAME="MCP Memory Service"

# Optional: Additional configuration
# export MCP_MEMORY_INCLUDE_HOSTNAME=true
# export MCP_CONSOLIDATION_ENABLED=false

echo "Starting MCP Memory Service with HTTPS on port $MCP_HTTPS_PORT"
echo "Certificate: $MCP_SSL_CERT_FILE"
echo "Private Key: $MCP_SSL_KEY_FILE"

# Change to script directory and start server
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Start the server
exec ./.venv/bin/python run_server.py
