# Multi-Client Setup Guide

This comprehensive guide covers setting up MCP Memory Service for multiple clients, enabling shared memory access across different applications and devices.

## Overview

MCP Memory Service supports multi-client access through several deployment patterns:

1. **🌟 Integrated Setup** (Easiest - during installation)
2. **📁 Shared File Access** (Local networks with shared storage)
3. **🌐 Centralized HTTP/SSE Server** (Distributed teams and cloud deployment)

## 🌟 Integrated Setup (Recommended)

### During Installation

The easiest way to configure multi-client access is during the initial installation:

```bash
# Run the installer - you'll be prompted for multi-client setup
python install.py

# When prompted, choose 'y':
# 🌐 Multi-Client Access Available!
# Would you like to configure multi-client access? (y/N): y
```

**Benefits of integrated setup:**
- ✅ Automatic detection of Claude Desktop, VS Code, Continue, Cursor, and other MCP clients
- ✅ Universal compatibility beyond just Claude applications
- ✅ Zero manual configuration required
- ✅ Future-proof setup for any MCP application

### Command Line Options

```bash
# Automatic multi-client setup (no prompts)
python install.py --setup-multi-client

# Skip the multi-client prompt entirely
python install.py --skip-multi-client-prompt

# Combined with other options
python install.py --storage-backend sqlite_vec --setup-multi-client
```

### Supported Applications

The integrated setup automatically detects and configures:

#### Automatically Configured
- **Claude Desktop**: Updates `claude_desktop_config.json` with multi-client settings
- **Continue IDE**: Modifies Continue configuration files
- **VS Code MCP Extension**: Updates VS Code MCP settings
- **Cursor**: Configures Cursor MCP integration
- **Generic MCP Clients**: Updates `.mcp.json` and similar configuration files

#### Manual Configuration Required
- **Custom MCP implementations**: May require manual configuration file updates
- **Enterprise MCP clients**: Check with your IT department for configuration requirements

## 📁 Shared File Access (Local Networks)

### Overview

For local networks with shared storage, multiple clients can access the same SQLite database using Write-Ahead Logging (WAL) mode.

### Quick Setup

1. **Run the setup script:**
   ```bash
   python setup_multi_client_complete.py
   ```

2. **Configure shared database location:**
   ```bash
   # Path to the SQLite-vec database file (folder will be created if needed)
   export MCP_MEMORY_SQLITE_PATH="/shared/network/mcp_memory/memory.db"

   # WAL is enabled by default by the service; no extra env needed
   ```

3. **Update each client configuration** to point to the shared location.

### Technical Implementation

The shared file access uses SQLite's WAL (Write-Ahead Logging) mode for concurrent access:

- **WAL Mode**: Enables multiple readers and one writer simultaneously
- **File Locking**: Handles concurrent access safely
- **Automatic Recovery**: SQLite handles crash recovery automatically

### Configuration Example

For Claude Desktop on each client machine:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/src/mcp_memory_service/server.py"],
      "env": {
        "MCP_MEMORY_SQLITE_PATH": "/shared/network/mcp_memory/memory.db",
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

### Network Storage Requirements

- **NFS/SMB Share**: Properly configured network file system
- **File Permissions**: Read/write access for all client users
- **Network Reliability**: Stable network connection to prevent corruption

## 🌐 Centralized HTTP/SSE Server (Cloud Deployment)

### Why This Approach?

- ✅ **True Concurrency**: Proper handling of multiple simultaneous clients
- ✅ **Real-time Updates**: Server-Sent Events (SSE) push changes to all clients instantly
- ✅ **Cross-platform**: Works from any device with HTTP access
- ✅ **Secure**: Optional API key authentication
- ✅ **Scalable**: Can handle many concurrent clients
- ✅ **Cloud-friendly**: No file locking issues

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client PC 1   │    │   Client PC 2   │    │   Client PC 3   │
│   (Claude App)  │    │   (VS Code)     │    │   (Web Client)  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │         HTTP/SSE API │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │     Central Server         │
                    │  ┌─────────────────────┐   │
                    │  │ MCP Memory Service  │   │
                    │  │   HTTP/SSE Server   │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │   SQLite-vec DB     │   │
                    │  │   (Single Source)   │   │
                    │  └─────────────────────┘   │
                    └────────────────────────────┘
```

### Server Installation

1. **Install on your server machine:**
   ```bash
   git clone https://github.com/doobidoo/mcp-memory-service.git
   cd mcp-memory-service
   python install.py --server-mode --storage-backend sqlite_vec
   ```

2. **Configure HTTP server:**
   ```bash
   export MCP_HTTP_HOST=0.0.0.0
   export MCP_HTTP_PORT=8000
   export MCP_API_KEY=your-secure-api-key
   ```

3. **Start the HTTP server:**
   ```bash
   python scripts/run_http_server.py
   ```

### Client Configuration (HTTP Mode)

There are two reliable ways for clients to connect to the centralized server:

- Direct Streamable HTTP (for clients that natively support MCP Streamable HTTP)
- Via mcp-proxy (for stdio-only clients like Codex)

Option A — Direct Streamable HTTP (preferred when supported):

```json
{
  "mcpServers": {
    "memory": {
      "transport": "streamablehttp",
      "url": "http://your-server:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-secure-api-key"
      }
    }
  }
}
```

Option B — mcp-proxy bridge (works with any stdio-only client):

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-proxy",
      "args": [
        "http://your-server:8000/mcp",
        "--transport=streamablehttp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "your-secure-api-key"
      }
    }
  }
}
```

### Security Configuration

#### API Key Authentication

```bash
# Generate a secure API key
export MCP_API_KEY=$(openssl rand -hex 32)

# Configure HTTPS (recommended for production)
export MCP_HTTPS_ENABLED=true
export MCP_SSL_CERT_FILE=/path/to/cert.pem
export MCP_SSL_KEY_FILE=/path/to/key.pem
```

#### Firewall Configuration

```bash
# Allow HTTP/HTTPS access (adjust port as needed)
sudo ufw allow 8000/tcp
sudo ufw allow 8443/tcp  # For HTTPS
```

### Docker Deployment

For containerized deployment:

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_HTTP_HOST=0.0.0.0
      - MCP_HTTP_PORT=8000
      - MCP_API_KEY=${MCP_API_KEY}
      - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

```bash
# Deploy with Docker Compose
docker-compose up -d
```

## Advanced Configuration

Note: For the HTTP server interface, use `MCP_HTTP_HOST`, `MCP_HTTP_PORT`, and `MCP_API_KEY`. These supersede older `MCP_MEMORY_HTTP_*` names in legacy docs. Client-side tools may use different env vars (see below).

### Client Environment Variables

- mcp-proxy: set `API_ACCESS_TOKEN` to pass the bearer token automatically.
- Memory MCP Bridge (`docker-compose/mcp-gateway/scripts/memory-mcp-bridge.js`): set `MCP_MEMORY_API_KEY` and optionally `MCP_MEMORY_HTTP_ENDPOINT`, `MCP_MEMORY_AUTO_DISCOVER`, `MCP_MEMORY_PREFER_HTTPS`.
- Direct Streamable HTTP clients: provide `Authorization: Bearer <MCP_API_KEY>` via headers (no special env var required).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HTTP_ENABLED` | `false` | Enable HTTP mode (FastAPI + Streamable HTTP) |
| `MCP_HTTP_HOST` | `0.0.0.0` | HTTP server bind address |
| `MCP_HTTP_PORT` | `8000` | HTTP server port |
| `MCP_API_KEY` | `none` | API key for auth (sent as `Authorization: Bearer ...`) |
| `MCP_HTTPS_ENABLED` | `false` | Enable HTTPS termination |
| `MCP_SSL_CERT_FILE` | `none` | Path to TLS certificate |
| `MCP_SSL_KEY_FILE` | `none` | Path to TLS private key |
| `MCP_CORS_ORIGINS` | `*` | CSV list of allowed origins |
| `MCP_SSE_HEARTBEAT` | `30` | SSE heartbeat interval (seconds) |
| `MCP_MEMORY_STORAGE_BACKEND` | `sqlite_vec` | `sqlite_vec`, `chroma`, or `cloudflare` |
| `MCP_MEMORY_SQLITE_PATH` | `<base>/sqlite_vec.db` | SQLite-vec database file path |
| `MCP_MEMORY_SQLITEVEC_PATH` | `none` | Alternate var for SQLite path (if set, used) |
| `MCP_MEMORY_SQLITE_PRAGMAS` | `none` | Override SQLite pragmas (e.g. `journal_mode=WAL,busy_timeout=5000`) |
| `MCP_MDNS_ENABLED` | `true` | Enable mDNS advertising/discovery |
| `MCP_MDNS_SERVICE_NAME` | `MCP Memory Service` | mDNS service name |
| `MCP_MDNS_SERVICE_TYPE` | `_mcp-memory._tcp.local.` | mDNS service type |
| `MCP_MDNS_DISCOVERY_TIMEOUT` | `5` | mDNS discovery timeout (seconds) |

Deprecated (replaced):
- `MCP_MEMORY_HTTP_HOST` → `MCP_HTTP_HOST`
- `MCP_MEMORY_HTTP_PORT` → `MCP_HTTP_PORT`
- `MCP_MEMORY_API_KEY` → `MCP_API_KEY` (server HTTP mode). Note: the standalone Memory MCP Bridge continues to use `MCP_MEMORY_API_KEY`.
- `MCP_MEMORY_ENABLE_WAL`: not needed; WAL is enabled by default. Use `MCP_MEMORY_SQLITE_PRAGMAS` to change.
- `MCP_MEMORY_ENABLE_SSE`: not required; SSE events are enabled with the HTTP server.
- `MCP_MEMORY_MULTI_CLIENT`, `MCP_MEMORY_MAX_CLIENTS`: not used.

### Performance Tuning

#### SQLite Configuration

```bash
# Optimize for concurrent access
export MCP_MEMORY_SQLITE_BUSY_TIMEOUT=5000
export MCP_MEMORY_SQLITE_CACHE_SIZE=10000
export MCP_MEMORY_SQLITE_JOURNAL_MODE=WAL
```

#### HTTP Server Tuning

```bash
# Adjust for high concurrency
export MCP_HTTP_WORKERS=4
export MCP_HTTP_TIMEOUT=30
export MCP_HTTP_KEEPALIVE=true
```

## Troubleshooting

### Common Issues

#### 1. Database Lock Errors

**Symptom**: `database is locked` errors
**Solution**: Enable WAL mode and check file permissions:

```bash
# WAL is enabled by default; verify file permissions instead
chmod 666 /path/to/memory.db
chmod 777 /path/to/memory.db-wal || true
```

#### 2. Network Access Issues

**Symptom**: Clients can't connect to HTTP server
**Solution**: Check firewall and network connectivity:

```bash
# Test server connectivity
curl http://your-server:8000/health

# Check firewall rules
sudo ufw status
```

#### 3. Configuration Conflicts

**Symptom**: Clients use different configurations
**Solution**: Verify all clients use the same settings:

```bash
# Check environment variables on each client
env | grep MCP_MEMORY

# Verify database file path matches
ls -la "$MCP_MEMORY_SQLITE_PATH"
```

### Diagnostic Commands

#### Check Multi-Client Status

```bash
# Test multi-client setup
python scripts/test_multi_client.py

# Verify database access
python -c "
import os, sqlite3
db = os.environ.get('MCP_MEMORY_SQLITE_PATH', '')
conn = sqlite3.connect(db) if db else None
print(f'Database accessible: {bool(conn)} (path={db})')
conn and conn.close()
"
```

#### Monitor Client Connections

```bash
# For HTTP server deployment
curl http://your-server:8000/stats

# Check active connections
netstat -an | grep :8000
```

## Migration from Single-Client

### Upgrading Existing Installation

1. **Backup existing data:**
   ```bash
   python scripts/backup_memories.py
   ```

2. **Run multi-client setup:**
   ```bash
   python install.py --setup-multi-client --migrate-existing
   ```

3. **Update client configurations** as needed.

### Data Migration

The installer automatically handles data migration, but you can also run it manually:

```bash
# Migrate to shared database location
python scripts/migrate_to_multi_client.py \
  --source ~/.mcp_memory_chroma \
  --target /shared/mcp_memory
```

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Deployment Guide](../deployment/docker.md) - Docker and cloud deployment
- [Troubleshooting](../troubleshooting/general.md) - Multi-client specific issues
- [API Reference](../IMPLEMENTATION_PLAN_HTTP_SSE.md) - HTTP/SSE API documentation

## Client Setup Recipes (Codex, Cursor, Qwen, Gemini)

This section provides practical, copy-pasteable setups for popular MCP clients. Use Streamable HTTP at `http://<host>:8000/mcp` when supported, or bridge via `mcp-proxy` for stdio-only clients.

Important:
- Server API key: set `MCP_API_KEY` on the server. Clients must send `Authorization: Bearer <MCP_API_KEY>`.
- Our MCP endpoint is Streamable HTTP at `/mcp` (not the SSE events feed at `/api/events`).

### Codex (via mcp-proxy)

Codex does not natively support HTTP transport. Use `mcp-proxy` to bridge stdio ⇄ Streamable HTTP.

1) Install mcp-proxy
```bash
pipx install mcp-proxy  # or: uv tool install mcp-proxy
```

2) Update Codex MCP config (see Codex docs for exact file location):
```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-proxy",
      "args": [
        "http://your-server:8000/mcp",
        "--transport=streamablehttp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "your-secure-api-key"
      }
    }
  }
}
```

Reference template: `examples/codex-mcp-config.json` in this repository.

Notes:
- Replace `your-server` and `your-secure-api-key` accordingly. For local testing use `http://127.0.0.1:8000/mcp`.
- Alternatively pass headers explicitly: `"args": ["http://.../mcp", "--transport=streamablehttp", "--headers", "Authorization", "Bearer your-secure-api-key"]`.

### Cursor

Pick one of these depending on your deployment:

- Option A — Local stdio (single machine):
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

- Option B — Remote central server via mcp-proxy (recommended for multi-client):
```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-proxy",
      "args": [
        "http://your-server:8000/mcp",
        "--transport=streamablehttp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "your-secure-api-key"
      }
    }
  }
}
```

- Option C — Direct Streamable HTTP (if your Cursor version supports it):
```json
{
  "mcpServers": {
    "memory": {
      "transport": "streamablehttp",
      "url": "http://your-server:8000/mcp",
      "headers": { "Authorization": "Bearer your-secure-api-key" }
    }
  }
}
```

### Qwen

Qwen clients that support MCP can connect either directly via Streamable HTTP or through `mcp-proxy` when only stdio is available. If your Qwen client UI accepts an MCP server list, use one of the Cursor-style examples above. If it only lets you specify a command, use the `mcp-proxy` form:

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-proxy",
      "args": [
        "http://your-server:8000/mcp",
        "--transport=streamablehttp"
      ],
      "env": { "API_ACCESS_TOKEN": "your-secure-api-key" }
    }
  }
}
```

Tips:
- Some Qwen distributions expose MCP configuration in a UI. Map fields as: transport = Streamable HTTP, URL = `http://<host>:8000/mcp`, header `Authorization: Bearer <key>`.

### Gemini

Gemini-based IDE integrations (e.g., Gemini Code Assist in VS Code/JetBrains) typically support MCP via a config file or UI. Use either direct Streamable HTTP or `mcp-proxy`:

- Direct Streamable HTTP (when supported):
```json
{
  "mcpServers": {
    "memory": {
      "transport": "streamablehttp",
      "url": "https://your-server:8443/mcp",
      "headers": { "Authorization": "Bearer your-secure-api-key" }
    }
  }
}
```

- Via mcp-proxy (works everywhere):
```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-proxy",
      "args": [
        "https://your-server:8443/mcp",
        "--transport=streamablehttp"
      ],
      "env": { "API_ACCESS_TOKEN": "your-secure-api-key" }
    }
  }
}
```

If your Gemini client expects a command-only entry, prefer the `mcp-proxy` form.

---

Troubleshooting client connections:
- Ensure you’re using `/mcp` (Streamable HTTP), not `/api/events` (SSE).
- Verify server exports `MCP_API_KEY` and clients send `Authorization: Bearer ...`.
- For remote setups, test reachability: `curl -i http://your-server:8000/api/health`.
- If a client doesn’t support Streamable HTTP, use `mcp-proxy`.
