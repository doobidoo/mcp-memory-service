# Multi-Client Deployment Guide

This guide explains how to deploy MCP Memory Service for multiple clients accessing a shared memory database. The service supports two primary deployment patterns: **Centralized Server Deployment** and **Shared File Access**.

## Overview

MCP Memory Service provides several deployment options to support multiple clients:

1. **🌟 Centralized HTTP/SSE Server** (Recommended for distributed teams)
2. **📁 Shared File Access** (For local networks with shared storage)
3. **☁️ Cloud Storage Limitations** (Why direct cloud storage doesn't work)

## 🌟 Centralized HTTP/SSE Server Deployment (Recommended)

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

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with server mode and HTTP API
python install.py --server-mode --enable-http-api
```

2. **Configure server settings:**
```bash
# Allow external connections
export MCP_HTTP_HOST=0.0.0.0

# Set custom port (optional)
export MCP_HTTP_PORT=8000

# Enable CORS for web clients (optional)
export MCP_CORS_ORIGINS="*"

# Set API key for authentication (recommended)
export MCP_API_KEY="$(openssl rand -base64 32)"  # Generate secure key

# Configure database location
export MCP_MEMORY_SQLITE_PATH="/path/to/shared/memory.db"
```

3. **Start the server:**
```bash
python scripts/run_http_server.py
```

The server will be available at:
- **API Documentation**: `http://your-server:8000/api/docs`
- **Web Dashboard**: `http://your-server:8000/`
- **SSE Endpoint**: `http://your-server:8000/api/events/stream`

### Client Configuration

#### For Claude Desktop

Configure each client to use the HTTP server instead of local MCP:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["-e", "
        const http = require('http');
        const url = 'http://your-server:8000/api';
        // HTTP-to-MCP bridge implementation
        // (See examples/http-mcp-bridge.js)
      "],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "http://your-server:8000/api",
        "MCP_MEMORY_API_KEY": "your-secure-api-key"
      }
    }
  }
}
```

#### For Web Applications

```javascript
// Connect to the HTTP API
const apiBase = 'http://your-server:8000/api';
const apiKey = 'your-secure-api-key';

// Store memory
const response = await fetch(`${apiBase}/memories`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  },
  body: JSON.stringify({
    content: 'Memory content',
    tags: ['tag1', 'tag2'],
    memory_type: 'note'
  })
});

// Real-time updates via SSE
const eventSource = new EventSource(`${apiBase}/events/stream?api_key=${apiKey}`);
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Memory update:', data);
};
```

### Server Management

#### Monitoring

Check server health:
```bash
curl http://your-server:8000/api/health
```

View server logs:
```bash
# Server logs include all client operations
tail -f /var/log/mcp-memory-service.log
```

#### Backup and Recovery

```bash
# Backup database
curl -X POST http://your-server:8000/api/admin/backup \
  -H "Authorization: Bearer your-api-key"

# List backups
curl http://your-server:8000/api/admin/backups \
  -H "Authorization: Bearer your-api-key"

# Restore from backup
curl -X POST http://your-server:8000/api/admin/restore \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"backup_id": "backup-timestamp"}'
```

## 📁 Shared File Access (Local Networks)

### When to Use

- ✅ **Local network environment** with shared storage (NFS, SMB)
- ✅ **Trusted users** who coordinate access
- ✅ **Limited concurrent usage** (2-3 users max)

### ❌ When NOT to Use

- ❌ **Cloud storage** (OneDrive, Dropbox, Google Drive)
- ❌ **High concurrency** (many simultaneous users)
- ❌ **Distributed teams** across different networks

### Setup

1. **Install on shared storage location:**
```bash
# Install to shared network drive
export MCP_MEMORY_SQLITE_PATH="/shared/network/drive/memory.db"
python install.py --storage-backend sqlite_vec
```

2. **Configure all clients to use the same database:**
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_SQLITE_PATH": "/shared/network/drive/memory.db",
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

### Important Considerations

- **File Locking**: SQLite WAL mode provides some concurrency, but conflicts can occur
- **Network Latency**: Database operations may be slower over network
- **Reliability**: Network interruptions can cause corruption
- **Coordination**: Users should avoid intensive operations simultaneously

## ☁️ Cloud Storage: Why It Doesn't Work

### The Problem

Direct SQLite file storage on cloud services has fundamental issues:

```
❌ BROKEN: Multiple PCs → Cloud Storage → SQLite File
┌──────────┐    ┌─────────────┐    ┌──────────────┐
│ Client 1 │───▶│   Dropbox   │◀───│   Client 2   │
└──────────┘    │ Google Drive│    └──────────────┘
                │   OneDrive  │
                └─────────────┘
                      │
                ┌─────▼─────┐
                │ SQLite DB │  ← Corruption Risk!
                └───────────┘

Issues:
• File locking doesn't work over cloud sync
• Sync conflicts when multiple clients write
• Database corruption from incomplete syncs
• Poor performance due to full file uploads
```

### Why SQLite + Cloud Storage Fails

1. **File Locking**: Cloud storage doesn't support SQLite's fcntl() locking
2. **Sync Conflicts**: Multiple edits create "conflicted copy" files
3. **Atomicity**: Cloud sync can interrupt SQLite transactions
4. **Performance**: Every change uploads the entire database file
5. **Corruption**: Partial syncs can corrupt the database

### Alternative: Cloud-Hosted Server

Instead of storing the database in cloud storage, host the MCP Memory Service server in the cloud:

```
✅ RECOMMENDED: Cloud Server Deployment
┌──────────┐    ┌─────────────┐    ┌──────────────┐
│ Client 1 │───▶│             │◀───│   Client 2   │
└──────────┘    │ Cloud Server│    └──────────────┘
                │   (AWS/GCP/ │
                │   Digital   │
                │   Ocean)    │
                └─────────────┘
                      │
                ┌─────▼─────┐
                │ MCP Memory│
                │  Service  │
                │ SQLite DB │
                └───────────┘
```

## 🔍 Zero-Configuration Service Discovery (mDNS)

### Overview

Starting with version 2.1.0, MCP Memory Service supports automatic service discovery using mDNS (Multicast DNS), also known as Bonjour or Zeroconf. This eliminates the need for manual endpoint configuration on local networks.

### Benefits

- ✅ **Zero Configuration**: Clients automatically discover services without manual setup
- ✅ **HTTPS Support**: Automatic discovery of both HTTP and HTTPS endpoints
- ✅ **Health Validation**: Clients verify service health before connecting
- ✅ **Fallback Support**: Graceful degradation to manual configuration if discovery fails

### Server Setup with mDNS

#### Enable mDNS Advertisement

```bash
# Enable mDNS service discovery (enabled by default)
export MCP_MDNS_ENABLED=true

# Optional: Customize service name (default: "MCP Memory Service")
export MCP_MDNS_SERVICE_NAME="My Team Memory Service"

# Optional: Enable HTTPS with auto-generated certificates
export MCP_HTTPS_ENABLED=true

# Start server with mDNS advertisement
python scripts/run_http_server.py
```

#### mDNS Configuration Options

```bash
# Core mDNS settings
export MCP_MDNS_ENABLED=true                           # Enable/disable mDNS (default: true)
export MCP_MDNS_SERVICE_NAME="MCP Memory Service"      # Service display name
export MCP_MDNS_SERVICE_TYPE="_mcp-memory._tcp.local." # Service type (RFC compliant)
export MCP_MDNS_DISCOVERY_TIMEOUT=5                    # Discovery timeout in seconds

# HTTPS settings for secure discovery
export MCP_HTTPS_ENABLED=true                          # Enable HTTPS mode
export MCP_SSL_CERT_FILE="/path/to/cert.pem"          # Custom certificate (optional)
export MCP_SSL_KEY_FILE="/path/to/key.pem"            # Custom private key (optional)
```

#### Server Output with mDNS

```
Starting MCP Memory Service HTTPS server on 0.0.0.0:8000
Dashboard: https://localhost:8000
API Docs: https://localhost:8000/api/docs
SSL Certificate: /tmp/mcp-memory-certs/cert.pem
SSL Key: /tmp/mcp-memory-certs/key.pem
NOTE: Browsers may show security warnings for self-signed certificates
mDNS service advertisement started
Press Ctrl+C to stop
```

### Client Auto-Discovery Setup

#### Claude Desktop Configuration

**Option 1: Full Auto-Discovery (Recommended)**

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_PREFER_HTTPS": "true",
        "MCP_MEMORY_API_KEY": "your-secure-api-key"
      }
    }
  }
}
```

**Option 2: Auto-Discovery with Fallback**

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_HTTP_ENDPOINT": "https://backup-server:8000/api",
        "MCP_MEMORY_PREFER_HTTPS": "true",
        "MCP_MEMORY_API_KEY": "your-secure-api-key"
      }
    }
  }
}
```

#### Client Configuration Options

```bash
# Auto-discovery settings
MCP_MEMORY_AUTO_DISCOVER=true              # Enable mDNS discovery
MCP_MEMORY_PREFER_HTTPS=true               # Prefer HTTPS over HTTP services
MCP_MEMORY_HTTP_ENDPOINT=""                # Manual fallback endpoint (optional)
MCP_MEMORY_API_KEY="your-api-key"          # API key for authentication
```

#### Client Discovery Output

```
MCP HTTP Bridge starting...
Attempting to discover MCP Memory Service via mDNS...
Discovered service: https://192.168.1.100:8000/api
Endpoint: https://192.168.1.100:8000/api
API Key: [SET]
Auto-discovery: ENABLED
Prefer HTTPS: YES
Service discovered automatically via mDNS
```

### Advanced mDNS Scenarios

#### Multiple Services Discovery

When multiple MCP Memory Services are available on the network:

```bash
# Client automatically selects the best service based on:
# 1. HTTPS preference (if MCP_MEMORY_PREFER_HTTPS=true)
# 2. Health check results
# 3. Response time
# 4. Port preference (standard ports first)
```

#### Network Troubleshooting

**Check mDNS Services on Network:**

```bash
# macOS: Browse for MCP Memory Services
dns-sd -B _mcp-memory._tcp

# Linux: Use avahi-browse
avahi-browse -t _mcp-memory._tcp

# Windows: Use Bonjour Browser or equivalent
```

**Manual Service Query:**

```bash
# Query specific service details
dns-sd -L "MCP Memory Service" _mcp-memory._tcp
```

#### Firewall Configuration

Ensure mDNS traffic is allowed:

```bash
# Linux (UFW)
sudo ufw allow 5353/udp  # mDNS port

# macOS/Windows: mDNS typically allowed by default
```

### Security Considerations

#### mDNS Security Best Practices

1. **Local Network Only**: mDNS only works on local networks (security by isolation)
2. **API Key Authentication**: Always use API keys even with mDNS discovery
3. **HTTPS Enforcement**: Enable HTTPS for encrypted communication
4. **Network Segmentation**: Use VLANs to isolate service discovery if needed

#### Production Deployment

```bash
# Production server with custom certificates
export MCP_HTTPS_ENABLED=true
export MCP_SSL_CERT_FILE="/etc/ssl/certs/mcp-memory.crt"
export MCP_SSL_KEY_FILE="/etc/ssl/private/mcp-memory.key"
export MCP_MDNS_ENABLED=true
export MCP_API_KEY="$(openssl rand -base64 32)"

python scripts/run_http_server.py
```

### Troubleshooting mDNS

#### Common Issues

**Q: No services discovered**
```bash
# Check if mDNS is enabled on server
grep "mDNS service advertisement" server.log

# Verify network connectivity
ping 224.0.0.251  # mDNS multicast address

# Check firewall rules
sudo ufw status | grep 5353
```

**Q: Discovery timeout**
```bash
# Increase discovery timeout
export MCP_MDNS_DISCOVERY_TIMEOUT=10

# Check network latency
# Large networks may need longer timeouts
```

**Q: Client prefers wrong service**
```bash
# Force HTTPS preference
export MCP_MEMORY_PREFER_HTTPS=true

# Or disable auto-discovery and use manual endpoint
export MCP_MEMORY_AUTO_DISCOVER=false
export MCP_MEMORY_HTTP_ENDPOINT="https://preferred-server:8000/api"
```

#### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Server debug logging
export LOG_LEVEL=DEBUG
python scripts/run_http_server.py

# Client will show discovery details in stderr
node examples/http-mcp-bridge.js 2>debug.log
```

## Deployment Examples

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN python install.py --server-mode --enable-http-api

# Expose both HTTP and HTTPS ports
EXPOSE 8000 8443

ENV MCP_HTTP_HOST=0.0.0.0
ENV MCP_HTTP_PORT=8000
ENV MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
ENV MCP_MDNS_ENABLED=true
ENV MCP_HTTPS_ENABLED=false

CMD ["python", "scripts/run_http_server.py"]
```

#### Basic HTTP with mDNS:
```bash
docker build -t mcp-memory-service .
docker run -p 8000:8000 -v ./data:/app/data mcp-memory-service
```

#### HTTPS with mDNS:
```bash
docker run -p 8000:8000 \
  -v ./data:/app/data \
  -e MCP_HTTPS_ENABLED=true \
  -e MCP_MDNS_ENABLED=true \
  mcp-memory-service
```

#### Production with Custom Certificates:
```bash
docker run -p 8000:8000 \
  -v ./data:/app/data \
  -v ./certs:/app/certs \
  -e MCP_HTTPS_ENABLED=true \
  -e MCP_SSL_CERT_FILE=/app/certs/cert.pem \
  -e MCP_SSL_KEY_FILE=/app/certs/key.pem \
  -e MCP_MDNS_ENABLED=true \
  -e MCP_API_KEY="your-secure-api-key" \
  mcp-memory-service
```

**Note**: For mDNS to work with Docker, you may need to use `--network host` mode on Linux:
```bash
docker run --network host \
  -v ./data:/app/data \
  -e MCP_MDNS_ENABLED=true \
  mcp-memory-service
```

### Cloud Platform Deployment

#### AWS EC2 / DigitalOcean / Linode

```bash
# Install on cloud instance
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Install and configure
python install.py --server-mode --enable-http-api

# Set environment for production
export MCP_HTTP_HOST=0.0.0.0
export MCP_API_KEY="$(openssl rand -base64 32)"
export MCP_MEMORY_SQLITE_PATH="/opt/mcp-memory/memory.db"

# Run with process manager
nohup python scripts/run_http_server.py > /var/log/mcp-memory.log 2>&1 &
```

#### Using systemd Service

Create `/etc/systemd/system/mcp-memory.service`:

```ini
[Unit]
Description=MCP Memory Service HTTP Server
After=network.target

[Service]
Type=simple
User=mcp-memory
WorkingDirectory=/opt/mcp-memory-service
Environment=MCP_HTTP_HOST=0.0.0.0
Environment=MCP_HTTP_PORT=8000
Environment=MCP_API_KEY=your-secure-key
Environment=MCP_MEMORY_SQLITE_PATH=/opt/mcp-memory/memory.db
ExecStart=/opt/mcp-memory-service/venv/bin/python scripts/run_http_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable mcp-memory
sudo systemctl start mcp-memory
```

## Security Considerations

### API Authentication

#### Generating API Keys

Always use a secure API key for production deployments:

```bash
# Generate a secure 32-byte base64 encoded key (recommended)
export MCP_API_KEY="$(openssl rand -base64 32)"

# Alternative: Generate a hex key
export MCP_API_KEY="$(openssl rand -hex 32)"

# Manual key (ensure it's secure)
export MCP_API_KEY="your-very-secure-random-key-here"
```

#### Using API Keys in Requests

All HTTP requests must include the API key in the Authorization header:

```bash
# Store memory with API key
curl -X POST http://your-server:8000/api/memories \
  -H "Authorization: Bearer $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "tags": ["test"]}'

# Retrieve memories with API key
curl -H "Authorization: Bearer $MCP_API_KEY" \
  http://your-server:8000/api/memories
```

#### Client Configuration with API Keys

Ensure all clients are configured with the same API key:

**JavaScript/Web Clients:**
```javascript
const headers = {
  'Authorization': `Bearer ${process.env.MCP_API_KEY}`,
  'Content-Type': 'application/json'
};
```

**Claude Desktop Configuration:**
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "http://your-server:8000/api",
        "MCP_MEMORY_API_KEY": "your-actual-api-key-here"
      }
    }
  }
}
```

#### API Key Security Best Practices

1. **Environment Variables**: Store API keys in environment variables, never in code
2. **Unique Keys**: Use different API keys for different environments (dev/staging/prod)
3. **Key Rotation**: Rotate API keys regularly, especially after team changes
4. **Secure Storage**: Use secrets management systems for production deployments
5. **Monitoring**: Log failed authentication attempts (but not the keys themselves)

#### Troubleshooting Authentication

**Common Authentication Errors:**

```bash
# 401 Unauthorized - Missing or invalid API key
curl -v http://your-server:8000/api/memories  # No auth header

# 403 Forbidden - API key provided but incorrect
curl -H "Authorization: Bearer wrong-key" http://your-server:8000/api/memories

# Success - Correct API key format
curl -H "Authorization: Bearer $MCP_API_KEY" http://your-server:8000/api/memories
```

**Server Logs for Debugging:**
- Check server logs for authentication failures
- Verify the API key is properly set in server environment
- Ensure clients are sending the correct Authorization header format

### Network Security

- **HTTPS**: Use reverse proxy (nginx/caddy) for TLS encryption
- **Firewall**: Restrict access to specific IP ranges
- **VPN**: Use VPN for secure remote access

### Database Security

- **Backups**: Regular automated backups
- **Encryption**: Encrypt database file at rest
- **Access Control**: Limit file system permissions

## Troubleshooting

### Common Issues

**Q: Server not accessible from other machines**
```bash
# Check if server is binding to all interfaces
export MCP_HTTP_HOST=0.0.0.0  # Not 127.0.0.1 or localhost

# Check firewall
sudo ufw allow 8000  # Ubuntu
firewall-cmd --add-port=8000/tcp  # CentOS/RHEL
```

**Q: CORS errors in web browsers**
```bash
# Allow all origins (development only)
export MCP_CORS_ORIGINS="*"

# Allow specific domains (production)
export MCP_CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

**Q: Database locked errors**
```bash
# Check database permissions
ls -la /path/to/memory.db

# Ensure only one server instance is running
ps aux | grep run_http_server

# Check SQLite WAL mode is enabled
sqlite3 /path/to/memory.db "PRAGMA journal_mode;"
# Should return: wal
```

**Q: Performance issues with many clients**
```bash
# Increase SQLite cache size
export MCP_MEMORY_SQLITE_PRAGMAS="cache_size=20000,temp_store=MEMORY"

# Monitor server resources
htop
iotop
```

### Performance Tuning

For high-load deployments:

```bash
# Optimize SQLite settings
export MCP_MEMORY_SQLITE_PRAGMAS="
  cache_size=50000,
  temp_store=MEMORY,
  mmap_size=268435456,
  synchronous=NORMAL,
  wal_autocheckpoint=1000
"

# Increase connection limits
ulimit -n 4096

# Use production ASGI server
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  mcp_memory_service.web.app:app
```

## Conclusion

The **Centralized HTTP/SSE Server** approach is the recommended solution for multi-client deployments. It provides:

- **Reliable concurrency** without file locking issues
- **Real-time synchronization** across all clients
- **Cloud deployment flexibility** 
- **Professional scalability** for team environments

For local networks with trusted users, **Shared File Access** can work with careful coordination, but the HTTP server approach is more robust and future-proof.

**Avoid direct cloud storage** (Dropbox/OneDrive/Google Drive) for SQLite databases due to fundamental concurrency and corruption issues.