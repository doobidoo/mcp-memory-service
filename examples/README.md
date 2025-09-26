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

### `http-mcp-bridge.js`
A Node.js script that bridges MCP JSON-RPC protocol to HTTP REST API calls. This allows MCP clients like Claude Desktop to connect to a remote HTTP server instead of running a local instance.

**Usage:**
1. Configure your server endpoint and API key as environment variables
2. Use this script as the MCP server command in your client configuration

### `claude-desktop-http-config.json`
Example Claude Desktop configuration for connecting to a remote MCP Memory Service HTTP server via the bridge script.

**Setup:**
1. Update the path to `http-mcp-bridge.js`
2. Set your server endpoint URL
3. Add your API key (if authentication is enabled)
4. Copy this configuration to your Claude Desktop config file

### `codex-mcp-config.json`
Example Codex configuration using `mcp-proxy` to bridge stdio to the service’s Streamable HTTP endpoint at `/mcp`.

**Setup:**
1. Install the proxy: `pipx install mcp-proxy` (or `uv tool install mcp-proxy`)
2. Set server API key on the server: `export MCP_API_KEY=...`
3. Copy this file and adjust `your-server` and API key
4. Place it in Codex’s MCP config location (see Codex docs)

Why proxy? Codex does not support HTTP transports natively and requires a stdio bridge.

## Quick Start

### 1. Server Setup
```bash
# On your server machine
cd mcp-memory-service
python install.py --server-mode --storage-backend sqlite_vec
export MCP_HTTP_HOST=0.0.0.0
export MCP_API_KEY="your-secure-key"
python scripts/run_http_server.py
```

### 2. Client Configuration
```bash
# Update the bridge script path and server details
cp examples/claude-desktop-http-config.json ~/.config/claude-desktop/
```

### 3. Test Connection
```bash
# Test the HTTP API directly
curl -H "Authorization: Bearer your-secure-key" \
  http://your-server:8000/api/health
```

## Advanced Usage

- See the [Multi-Client Setup Guide](../docs/integration/multi-client.md) for Codex, Cursor, Qwen, and Gemini recipes.
- For Cursor/Qwen/Gemini direct HTTP usage, prefer the Streamable HTTP endpoint: `http(s)://<host>:8000/mcp` with header `Authorization: Bearer <MCP_API_KEY>`.
