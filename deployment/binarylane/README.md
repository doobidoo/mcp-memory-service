# BinaryLane Deployment Guide

Deploy MCP Memory Service to a BinaryLane VPS in Australia with Cloudflare Tunnel for secure access.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Claude Desktop │────▶│ Cloudflare Tunnel│────▶│  BinaryLane VPS (SYD)   │
│  (your machine) │     │   (encrypted)    │     │  ├── MCP Memory Service │
└─────────────────┘     └──────────────────┘     │  └── SQLite-vec/Qdrant  │
                                                 └─────────────────────────┘
```

## Prerequisites

- BinaryLane VPS (1GB+ RAM, Sydney datacenter)
- Cloudflare account (free tier works)
- Domain managed by Cloudflare (or use `*.trycloudflare.com`)

## Quick Start

### 1. Bootstrap the VPS

```bash
# SSH to your VPS
ssh root@your-vps-ip

# Download and run bootstrap
curl -sSL https://raw.githubusercontent.com/27Bslash6/mcp-memory-service/main/deployment/binarylane/bootstrap.sh | bash

# Log out and back in (for Docker group)
exit
ssh root@your-vps-ip
```

### 2. Deploy the Service

```bash
cd /opt/mcp-memory

# Download the compose file (SQLite-vec recommended for 1GB RAM)
curl -O https://raw.githubusercontent.com/27Bslash6/mcp-memory-service/main/deployment/binarylane/docker-compose.sqlite.yml

# Or for Qdrant (needs 2GB swap - already set up by bootstrap)
# curl -O https://raw.githubusercontent.com/27Bslash6/mcp-memory-service/main/deployment/binarylane/docker-compose.qdrant.yml

# Start the service
docker compose -f docker-compose.sqlite.yml up -d

# Check logs
docker compose -f docker-compose.sqlite.yml logs -f
```

### 3. Set Up Cloudflare Tunnel

#### Option A: Quick Tunnel (Testing)

```bash
# Creates a random *.trycloudflare.com URL
cloudflared tunnel --url http://localhost:8000
```

#### Option B: Named Tunnel (Production)

```bash
# Login to Cloudflare
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create mcp-memory

# Note the tunnel ID (e.g., abc123-def456-...)
# Create config file
cat > ~/.cloudflared/config.yml << EOF
tunnel: YOUR_TUNNEL_ID
credentials-file: /root/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  # HTTP API
  - hostname: memory-api.yourdomain.com
    service: http://localhost:8000
  # MCP Protocol
  - hostname: memory-mcp.yourdomain.com
    service: http://localhost:8001
  # Catch-all
  - service: http_status:404
EOF

# Create DNS records
cloudflared tunnel route dns mcp-memory memory-api.yourdomain.com
cloudflared tunnel route dns mcp-memory memory-mcp.yourdomain.com

# Install as system service
sudo cloudflared service install

# Start the tunnel
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### 4. Configure Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "url": "https://memory-mcp.yourdomain.com/mcp/v1"
    }
  }
}
```

### 5. Migrate Existing Memories

```bash
# On your LOCAL machine (where memories currently exist)

# Clone the repo if you haven't
git clone https://github.com/27Bslash6/mcp-memory-service.git
cd mcp-memory-service/deployment/binarylane

# Edit the script with your settings
export TARGET_HOST="your-vps-ip"
export SOURCE_TYPE="qdrant"  # or sqlite_vec
export SOURCE_QDRANT_URL="http://localhost:6333"

# Run full migration
chmod +x migrate-memories.sh
./migrate-memories.sh full
```

## Memory Optimization (1GB RAM)

### SQLite-vec (Recommended)

- Vectors stored on disk, not RAM
- ~500MB footprint with e5-small model
- Works reliably on 1GB VPS

### Qdrant (Advanced)

- Needs MMAP mode for low memory
- ~900MB footprint (tight on 1GB)
- Requires 2GB swap (bootstrap creates this)
- May swap under heavy load

### Embedding Model Options

| Model | Dimensions | RAM Usage | Quality |
|-------|------------|-----------|---------|
| `intfloat/e5-small-v2` | 384 | ~400MB | Good |
| `intfloat/e5-base-v2` | 768 | ~600MB | Better |

For 1GB VPS, use `e5-small-v2` (default in compose files).

## Monitoring

```bash
# Check service status
docker compose -f docker-compose.sqlite.yml ps

# View logs
docker compose -f docker-compose.sqlite.yml logs -f

# Check memory usage
free -h
docker stats --no-stream

# Test health endpoint
curl http://localhost:8000/api/health
```

## Backup

```bash
# SQLite-vec backup
docker compose -f docker-compose.sqlite.yml stop
cp -r data/sqlite backups/sqlite-$(date +%Y%m%d)
docker compose -f docker-compose.sqlite.yml start

# Qdrant backup
docker compose -f docker-compose.qdrant.yml exec qdrant \
  curl -X POST 'http://localhost:6333/collections/memories/snapshots'
```

## Troubleshooting

### Service won't start

```bash
# Check logs
docker compose logs mcp-memory

# Check memory
free -h
dmesg | tail -20  # Look for OOM killer
```

### Out of Memory

1. Ensure swap is enabled: `swapon --show`
2. Switch to SQLite-vec backend
3. Use smaller embedding model
4. Consider upgrading to 2GB VPS

### Tunnel not connecting

```bash
# Check cloudflared status
sudo systemctl status cloudflared

# Check logs
sudo journalctl -u cloudflared -f

# Test local connectivity
curl http://localhost:8000/api/health
```

### Slow responses

- First request after idle loads embedding model (~10-15s)
- Subsequent requests should be fast
- Check if swapping: `vmstat 1 5`

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| BinaryLane VPS 1GB | ~$9 AUD (~$6 USD) |
| Cloudflare Tunnel | Free |
| Domain (optional) | ~$10-15/year |
| **Total** | **~$6 USD/month** |
