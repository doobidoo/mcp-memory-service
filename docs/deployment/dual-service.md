# Dual Service Deployment - FastMCP + HTTP Dashboard

## Overview

This deployment provides both **FastMCP Protocol** and **HTTP Dashboard** services running simultaneously, eliminating Node.js SSL issues while maintaining full functionality.

## Architecture

### Service 1: FastMCP Server (Port 8000)
- **Purpose**: Native MCP protocol for Claude Code clients
- **Protocol**: JSON-RPC 2.0 over Server-Sent Events
- **Access**: `http://[IP]:8000/mcp`
- **Service**: `mcp-memory.service`

### Service 2: HTTP Dashboard (Port 8080)
- **Purpose**: Web dashboard and HTTP API
- **Protocol**: Standard HTTP/REST
- **Access**: `http://[IP]:8080/`
- **API**: `http://[IP]:8080/api/*`
- **Service**: `mcp-http-dashboard.service`

## Deployment

### Quick Deploy
```bash
./deploy_dual_services.sh
```

### Manual Setup
```bash
# Install FastMCP service
sudo cp /tmp/fastmcp-server-with-mdns.service /etc/systemd/system/mcp-memory.service

# Install HTTP Dashboard service
sudo cp /tmp/mcp-http-dashboard.service /etc/systemd/system/mcp-http-dashboard.service

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable mcp-memory mcp-http-dashboard
sudo systemctl start mcp-memory mcp-http-dashboard
```

## Access URLs

Replace `[IP]` with your actual server IP address (e.g., `10.0.1.30`):

- **FastMCP Protocol**: `http://[IP]:8000/mcp` (for Claude Code)
- **Web Dashboard**: `http://[IP]:8080/` (for monitoring)
- **Health API**: `http://[IP]:8080/api/health`
- **Memory API**: `http://[IP]:8080/api/memories`
- **Search API**: `http://[IP]:8080/api/search`

## Service Management

### Status Checks
```bash
sudo systemctl status mcp-memory          # FastMCP server
sudo systemctl status mcp-http-dashboard  # HTTP dashboard
```

### View Logs
```bash
sudo journalctl -u mcp-memory -f          # FastMCP logs
sudo journalctl -u mcp-http-dashboard -f  # Dashboard logs
```

### Control Services
```bash
# Start services
sudo systemctl start mcp-memory mcp-http-dashboard

# Stop services
sudo systemctl stop mcp-memory mcp-http-dashboard

# Restart services
sudo systemctl restart mcp-memory mcp-http-dashboard
```

## mDNS Discovery

Both services advertise via mDNS for network discovery:

```bash
# Browse HTTP services
avahi-browse -t _http._tcp

# Browse MCP services (if supported)
avahi-browse -t _mcp._tcp

# Resolve hostname
avahi-resolve-host-name memory.local
```

**Services Advertised:**
- `MCP Memory Dashboard._http._tcp.local.` (port 8080)
- `MCP Memory FastMCP._mcp._tcp.local.` (port 8000)

## Dependencies

Ensure these packages are installed in the virtual environment:
- `mcp` - MCP Protocol support
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `zeroconf` - mDNS advertising
- `aiohttp` - HTTP client/server
- `sqlite-vec` - Vector database
- `sentence-transformers` - Embeddings

## Configuration

### Environment Variables
- `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`
- `MCP_MDNS_ENABLED=true`
- `MCP_HTTP_ENABLED=true`
- `MCP_SERVER_HOST=0.0.0.0`
- `MCP_SERVER_PORT=8000`
- `MCP_HTTP_PORT=8080`

### Storage
Both services share the same SQLite-vec database:
- **Path**: `~/.local/share/mcp-memory/sqlite_vec.db`
- **Backend**: `sqlite_vec`
- **Model**: `all-MiniLM-L6-v2`

## Troubleshooting

### Services Not Accessible
1. Check if services are running: `systemctl status [service]`
2. Verify ports are listening: `ss -tlnp | grep -E ":800[08]"`
3. Test direct IP access instead of hostname
4. Check firewall rules if accessing remotely

### mDNS Not Working
1. Ensure avahi-daemon is running: `systemctl status avahi-daemon`
2. Install missing dependencies: `pip install zeroconf aiohttp`
3. Restart services after installing dependencies

### FastMCP Protocol Issues
1. Ensure client accepts `text/event-stream` headers
2. Use JSON-RPC 2.0 format for requests
3. Access `/mcp` endpoint, not root `/`

## Client Configuration

### Claude Code
Configure MCP client to use:
```
http://[SERVER_IP]:8000/mcp
```

### curl Examples
```bash
# Health check
curl http://[SERVER_IP]:8080/api/health

# Store memory
curl -X POST http://[SERVER_IP]:8080/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "tags": ["test"]}'

# Search memories
curl -X POST http://[SERVER_IP]:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 5}'
```

## Benefits

✅ **No Node.js SSL Issues** - Pure Python implementation
✅ **Dual Protocol Support** - Both MCP and HTTP available
✅ **Network Discovery** - mDNS advertising for easy access
✅ **Production Ready** - systemd managed services
✅ **Backward Compatible** - HTTP API preserved for existing tools
✅ **Claude Code Ready** - Native MCP protocol support
