# Deploying MCP Memory Service to Fly.io

## Cost Estimate

| Usage Pattern | Monthly Cost |
|---------------|-------------|
| Auto-stop (2-4 hrs/day) | ~$2-4 |
| Always-on (shared-cpu-1x, 1GB) | ~$6 |
| Performance (shared-cpu-2x, 2GB) | ~$12 |

With annual reservation ($36/year): 40% discount on compute.

## Quick Start

```bash
# 1. Install flyctl
brew install flyctl  # macOS
# or: curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Launch (creates app + volume)
cd /path/to/mcp-memory-service
fly launch --copy-config --no-deploy

# 4. Create persistent volume (1GB is plenty for years of memories)
fly volumes create mcp_memory_data --region sjc --size 1

# 5. Deploy
fly deploy

# 6. Check status
fly status
fly logs
```

## Configuration

### Environment Variables

Set secrets for sensitive values:

```bash
# If using Cloudflare hybrid backend (optional)
fly secrets set CLOUDFLARE_API_TOKEN="your-token"
fly secrets set CLOUDFLARE_ACCOUNT_ID="your-account"
fly secrets set CLOUDFLARE_D1_DATABASE_ID="your-db"
```

### Regions

Change `primary_region` in `fly.toml` to your nearest:
- `sjc` - San Jose, US
- `iad` - Virginia, US
- `lhr` - London, UK
- `fra` - Frankfurt, DE
- `nrt` - Tokyo, JP
- `syd` - Sydney, AU

Full list: `fly platform regions`

### Machine Size

Edit `fly.toml` [vm] section:

```toml
# Minimal (personal use)
[vm]
  size = "shared-cpu-1x"
  memory = "1gb"

# Comfortable (faster embeddings)
[vm]
  size = "shared-cpu-2x"
  memory = "2gb"

# Performance (multiple users)
[vm]
  size = "performance-1x"
  memory = "2gb"
```

## Connecting Claude Desktop

### Option 1: Direct HTTPS (Recommended)

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "url": "https://mcp-memory-service.fly.dev:8081/mcp/v1"
    }
  }
}
```

### Option 2: Via HTTP API

Use the REST API directly:
- Health: `https://mcp-memory-service.fly.dev/api/health`
- Store: `POST https://mcp-memory-service.fly.dev/api/memories`
- Retrieve: `GET https://mcp-memory-service.fly.dev/api/memories/search?query=...`

## Auto-Stop Behavior

By default, the machine stops after ~5 minutes of no requests:
- **Saves money** - Only pay when active
- **Cold start** - ~10-15 seconds (embedding model load)

To keep always-on, edit `fly.toml`:

```toml
[[services]]
  auto_stop_machines = "off"
  min_machines_running = 1
```

## Monitoring

```bash
# Live logs
fly logs

# SSH into machine
fly ssh console

# Check volume usage
fly ssh console -C "df -h /data"

# Database stats
fly ssh console -C "ls -la /data"
```

## Backup / Migration

### Export from Fly

```bash
# SSH and copy database
fly ssh console
# Inside container:
tar -czvf /tmp/backup.tar.gz /data

# From local machine:
fly ssh sftp get /tmp/backup.tar.gz ./backup.tar.gz
```

### Import to Fly

```bash
# Copy to machine
fly ssh sftp put ./backup.tar.gz /tmp/backup.tar.gz

# SSH and extract
fly ssh console
tar -xzvf /tmp/backup.tar.gz -C /
```

## Troubleshooting

### Machine won't start

```bash
fly logs --app mcp-memory-service
fly status
```

### Volume not mounting

```bash
fly volumes list
# Ensure volume exists in same region as app
```

### Out of memory

Upgrade machine size in `fly.toml` and redeploy:
```bash
fly deploy
```

### Cold start too slow

1. Increase `min_machines_running = 1`
2. Use smaller embedding model (`intfloat/e5-small-v2`)
3. Upgrade to `performance-1x` CPU

## Security Considerations

For production/multi-user deployments:

1. **Enable OAuth**:
   ```bash
   fly secrets set MCP_OAUTH_ENABLED=true
   fly secrets set MCP_OAUTH_CLIENT_ID="your-client-id"
   fly secrets set MCP_OAUTH_CLIENT_SECRET="your-secret"
   ```

2. **Restrict access** via Fly.io private networking or IP allowlists

3. **Enable HTTPS only** (already configured in fly.toml)

## Scaling

### Horizontal (multiple machines)

```bash
fly scale count 2 --region sjc
```

Note: SQLite-vec doesn't support multi-machine. Use Qdrant or Cloudflare backend for horizontal scaling.

### Vertical (bigger machine)

Edit `fly.toml` and redeploy, or:

```bash
fly scale vm shared-cpu-2x --memory 2048
```
