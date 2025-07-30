# MCP Memory Service Examples

This directory contains example configurations, scripts, and setup utilities for deploying MCP Memory Service in various scenarios.

## Directory Structure

### `/config/` - Configuration Examples
- Example Claude Desktop configurations
- Template configuration files for different deployment scenarios
- MCP server configuration samples

### `/setup/` - Setup Scripts and Utilities  
- Multi-client setup scripts
- Automated configuration tools
- Installation helpers

## Core Files

### `http-mcp-bridge-robust.js`
Production-ready Node.js bridge that connects MCP clients to remote HTTP memory services. Features retry logic, timeout handling, and comprehensive error management.

**Key Features:**
- Automatic retry with exponential backoff
- Health check verification
- Certificate validation bypass for self-signed certificates
- Support for both semantic search and tag-based search
- Comprehensive logging and error handling

### `README-remote-memory.md`
Complete setup guide for connecting Claude Desktop to remote memory services, including troubleshooting and configuration options.

### Configuration Templates
- `claude-desktop-memory-config.json` - Production Claude Desktop configuration
- `memory_local_config.json` - Local server configuration example  
- `custom-memory-config.json` - Customizable template for various setups

### Development Archive
- `archive/remote-development/` - Development artifacts and test files from remote memory implementation

## Quick Start

### Remote Memory Connection
For connecting to an existing remote memory server:

```bash
# 1. Copy the configuration template
cp examples/claude-desktop-memory-config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# 2. Update configuration with your server details
# Edit the endpoint URL and API key in the configuration file

# 3. Restart Claude Desktop
```

### Local Server Setup
For setting up your own memory server:

```bash
# On your server machine
cd mcp-memory-service
python install.py --server-mode --enable-http-api
export MCP_HTTP_HOST=0.0.0.0
export MCP_API_KEY="your-secure-key"
python scripts/run_http_server.py
```

### Test Connection
```bash
# Test the HTTP API directly
curl -k -H "Authorization: Bearer your-api-key" \
  https://your-server/api/health
```

## Advanced Usage

See the complete [Multi-Client Deployment Guide](../docs/deployment/multi-client-server.md) for detailed configuration options, security setup, and troubleshooting.