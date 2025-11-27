# Oracle Server Deployment Guide

**MCP Memory Service** - Self-hosted deployment to Oracle Cloud Free Tier with Tailscale VPN access.

This guide provides step-by-step instructions for deploying a centralized, containerized MCP Memory Service instance on Oracle Cloud Infrastructure Free Tier, eliminating SQLite database locking issues when running multiple concurrent MCP clients.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Client Configuration](#client-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Architecture](#architecture)
9. [Maintenance](#maintenance)

---

## Prerequisites

Before starting deployment, ensure you have the following:

### Required Accounts

1. **Oracle Cloud Infrastructure (OCI) Free Tier Account**
   - Sign up: https://www.oracle.com/cloud/free/
   - Free Tier includes: VM.Standard.A1.Flex (4 ARM cores, 24GB RAM)
   - Melbourne region (ap-melbourne-1) recommended for AU users

2. **Tailscale Account**
   - Sign up: https://login.tailscale.com/start
   - Free tier supports up to 100 devices
   - Required for secure VPN access to your server

3. **Cloudflare Account (Optional - for backups)**
   - Sign up: https://dash.cloudflare.com/sign-up
   - R2 storage: ~$1/month for typical usage
   - Only needed if you want automated backups

### Required Tools (Local Machine)

```bash
# macOS
brew install oci-cli jq rsync

# Linux (Ubuntu/Debian)
sudo apt-get install oci-cli jq rsync

# Verify installations
oci --version        # OCI CLI 3.x or later
jq --version         # jq 1.6 or later
rsync --version      # rsync 3.x or later
ssh -V               # OpenSSH 7.x or later
```

### SSH Key Pair

If you don't have an SSH key pair:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
# Press Enter to accept defaults (no passphrase recommended for automation)
```

---

## Quick Start

**Single-command deployment** (recommended for most users):

```bash
# 1. Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# 2. Set required environment variables
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
export TAILSCALE_AUTH_KEY="tskey-auth-your-tailscale-key"

# 3. Run deployment (includes migration from Cloudflare if desired)
bash scripts/deploy/deploy_to_oracle.sh --with-migration

# 4. Wait for deployment to complete (~10-15 minutes)
# Script will display client configuration at the end
```

**With existing Cloudflare backend?** Add migration flag to import your memories:

```bash
# Export Cloudflare credentials first
export CLOUDFLARE_API_TOKEN="your-cloudflare-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-database-id"

# Run deployment with migration
bash scripts/deploy/deploy_to_oracle.sh --with-migration
```

---

## Step-by-Step Setup

For users who want to understand each step or customize the deployment.

### Step 1: Configure OCI CLI

```bash
# Configure OCI CLI (interactive setup)
oci setup config

# You'll need:
# - User OCID (from OCI console -> Profile -> User Settings)
# - Tenancy OCID (from OCI console -> Profile -> Tenancy)
# - Region: ap-melbourne-1 (or your preferred region)
# - Generate new API key (follow prompts)

# Find your compartment ID
oci iam compartment list --all

# Export compartment ID
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..aaaaaa..."
```

### Step 2: Generate Tailscale Auth Key

```bash
# 1. Visit Tailscale admin console
open https://login.tailscale.com/admin/settings/keys

# 2. Click "Generate auth key"
# 3. Settings:
#    - Reusable: Yes (for redeployment)
#    - Ephemeral: No (persistent device)
#    - Preauthorized: Yes (no manual approval)
#    - Tags: oracle-mcp-memory (optional)
# 4. Copy the key (starts with tskey-auth-)

# 5. Export auth key
export TAILSCALE_AUTH_KEY="tskey-auth-your-key-here"
```

### Step 3: (Optional) Configure R2 Backups

If you want automated backups to Cloudflare R2:

```bash
# 1. Create R2 bucket
# Visit: https://dash.cloudflare.com/ -> R2
# Click "Create bucket" -> Name: "mcp-memory-backups"

# 2. Generate R2 API token
# R2 -> Manage R2 API Tokens -> Create API Token
# Permissions: Object Read & Write
# Copy access_key_id and secret_access_key

# 3. Create deployment/.env file
cd deployment/oracle
cp .env.example .env

# 4. Edit .env and fill in R2 credentials:
nano .env
# Set:
# CLOUDFLARE_ACCOUNT_ID=your-account-id
# R2_ACCESS_KEY_ID=your-access-key
# R2_SECRET_ACCESS_KEY=your-secret-key
# R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com

# 5. Create rclone.conf
cp rclone.conf.example rclone.conf
nano rclone.conf
# Update endpoint with your account ID
```

### Step 4: Provision Oracle Instance

```bash
# Run provisioning script
bash scripts/deploy/provision_oracle.sh

# Script will:
# - Check OCI compartment access
# - Create VM.Standard.A1.Flex instance (4 cores, 24GB RAM)
# - Configure security list (SSH only, port 22)
# - Wait for instance to be RUNNING
# - Output public IP and instance OCID

# Save the public IP for SSH access
# Example output:
# {
#   "instance_ocid": "ocid1.instance.oc1.ap-melbourne-1...",
#   "public_ip": "123.45.67.89",
#   "instance_name": "oracle-vps-au"
# }
```

**Troubleshooting Provisioning:**

- **"Out of host capacity"**: Oracle Free Tier capacity is limited. Script retries 3 times with 30s delay. If still failing, try different availability domain or wait a few hours.
- **"Authorization failed"**: Verify OCI CLI configuration with `oci iam region list`
- **"Compartment not found"**: Double-check `OCI_COMPARTMENT_ID` is correct

### Step 5: Setup Tailscale VPN

```bash
# SSH into your Oracle instance (replace with your public IP)
ssh -i ~/.ssh/id_rsa ubuntu@123.45.67.89

# Copy and run setup_tailscale.sh
# (Or use the master deployment script which does this automatically)

# Manual approach:
sudo bash /tmp/setup_tailscale.sh

# Script will:
# - Install Tailscale from official repository
# - Authenticate with your tailnet using auth key
# - Obtain Tailscale IP (100.x.x.x)
# - Write TAILSCALE_IP to /opt/mcp-memory/.env

# Save the Tailscale IP for client configuration
# Example output:
# {
#   "tailscale_ip": "100.64.1.2",
#   "device_name": "oracle-vps-au",
#   "tailnet": ".example.ts.net"
# }

# Verify Tailscale connection from another device
tailscale ping 100.64.1.2
```

### Step 6: Deploy Docker Containers

```bash
# Option A: Manual deployment (if not using master script)
ssh ubuntu@123.45.67.89

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker

# Create deployment directory
sudo mkdir -p /opt/mcp-memory
sudo chown ubuntu:ubuntu /opt/mcp-memory

# Copy deployment files (from local machine)
rsync -avz deployment/oracle/ ubuntu@123.45.67.89:/opt/mcp-memory/

# SSH back into instance
ssh ubuntu@123.45.67.89

# Build and start containers
cd /opt/mcp-memory
docker compose build
docker compose up -d

# Option B: Use master deployment script (recommended)
bash scripts/deploy/deploy_to_oracle.sh
```

### Step 7: Verify Deployment

```bash
# From local machine (on Tailscale network)
TAILSCALE_IP=100.64.1.2  # Replace with your Tailscale IP

# Test health endpoint
curl http://$TAILSCALE_IP:8000/api/health

# Expected response:
# {
#   "status": "ok",
#   "version": "8.x.x",
#   "storage_backend": "sqlite_vec",
#   "memory_count": 0,
#   "queue_size": 0
# }

# Run automated test suite
bash scripts/verify/test_deployment.sh --tailscale-ip $TAILSCALE_IP

# All tests should pass (8/8)
```

---

## Configuration

### Environment Variables

All configuration is in `deployment/oracle/.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAILSCALE_IP` | Yes | Auto-populated | Tailscale IP address (100.x.x.x) |
| `MCP_MEMORY_STORAGE_BACKEND` | Yes | `sqlite_vec` | Storage backend (use sqlite_vec for oracle-server) |
| `MCP_MEMORY_SQLITE_DATABASE_PATH` | Yes | `/data/sqlite_vec.db` | Database file path inside container |
| `MCP_MEMORY_SQLITE_PRAGMAS` | Yes | `busy_timeout=15000,journal_mode=WAL` | SQLite performance settings |
| `MCP_HTTP_ENABLED` | Yes | `true` | Enable HTTP server |
| `MCP_API_KEY` | Recommended | (generate) | API key for authentication (`openssl rand -base64 32`) |
| `CLOUDFLARE_ACCOUNT_ID` | Optional | - | For R2 backups |
| `R2_ACCESS_KEY_ID` | Optional | - | For R2 backups |
| `R2_SECRET_ACCESS_KEY` | Optional | - | For R2 backups |
| `R2_BUCKET_NAME` | Optional | `mcp-memory-backups` | R2 bucket name |
| `UVICORN_WORKERS` | No | `4` | Number of worker processes |

**Security Note:** Generate a strong API key:

```bash
openssl rand -base64 32
```

Add to `.env`:

```bash
MCP_API_KEY=your-generated-key-here
```

### Docker Resource Limits

Resource limits are defined in `docker-compose.yml` to match Oracle Free Tier (4 cores, 24GB RAM):

```yaml
services:
  mcp-memory:
    deploy:
      resources:
        limits:
          cpus: '3'
          memory: 16GB

  backup:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2GB
```

**Note:** Leaves 6GB RAM for OS and system processes.

### Backup Configuration

Backup schedule is controlled via `BACKUP_SCHEDULE` in `.env` (cron format):

```bash
# Every 6 hours (default)
BACKUP_SCHEDULE=0 */6 * * *

# Daily at 2 AM UTC
BACKUP_SCHEDULE=0 2 * * *

# Twice daily (noon and midnight UTC)
BACKUP_SCHEDULE=0 0,12 * * *
```

---

## Verification

After deployment, verify all components are working:

### 1. Automated Test Suite

```bash
# Run complete test suite
bash scripts/verify/test_deployment.sh \
  --tailscale-ip 100.64.1.2 \
  --api-key your-api-key

# Tests include:
# ✓ Health endpoint accessibility
# ✓ Memory creation (POST)
# ✓ Semantic search
# ✓ Memory retrieval by hash
# ✓ Memory deletion
# ✓ Docker container status
# ✓ Environment configuration
# ✓ Backup configuration
```

### 2. Manual Verification

```bash
# Check container status
ssh ubuntu@<public-ip> "cd /opt/mcp-memory && docker compose ps"

# View logs
ssh ubuntu@<public-ip> "cd /opt/mcp-memory && docker compose logs --tail=50"

# Test from Tailscale network
curl http://<tailscale-ip>:8000/api/health

# Access web dashboard
open http://<tailscale-ip>:8000/
```

### 3. Performance Validation

Expected performance (Melbourne to AU region):

- **Query latency**: <100ms
- **Memory operations**: <50ms
- **Concurrent writes**: 20 requests queued without errors

Test concurrency:

```bash
# Install hey (HTTP load tester)
go install github.com/rakyll/hey@latest

# Test concurrent writes
hey -n 25 -c 5 -m POST \
  -H "Content-Type: application/json" \
  -d '{"content":"concurrent test","tags":["load-test"]}' \
  http://<tailscale-ip>:8000/api/memories
```

---

## Client Configuration

Configure your MCP clients to connect to the Oracle server:

### Claude Desktop

Edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memory", "server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "http_client",
        "MCP_HTTP_CLIENT_ENDPOINT": "http://100.64.1.2:8000",
        "MCP_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Claude Code

In VS Code settings or `.vscode/settings.json`:

```json
{
  "claude.mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memory", "server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "http_client",
        "MCP_HTTP_CLIENT_ENDPOINT": "http://100.64.1.2:8000",
        "MCP_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Environment Variables (Alternative)

Set globally in `~/.bashrc` or `~/.zshrc`:

```bash
export MCP_MEMORY_STORAGE_BACKEND=http_client
export MCP_HTTP_CLIENT_ENDPOINT=http://100.64.1.2:8000
export MCP_API_KEY=your-api-key-here
```

### Verification

Test client connection:

```bash
# From Claude Desktop or Claude Code, try:
# /memory-store test "Deployment verification from client"
# /memory-recall test "deployment verification"
```

---

## Troubleshooting

### Common Issues

#### 1. "Out of host capacity" during provisioning

**Symptom:** Oracle provisioning fails with capacity error

**Solutions:**

```bash
# Try different availability domain
export OCI_AVAILABILITY_DOMAIN="different-AD-name"
bash scripts/deploy/provision_oracle.sh

# Wait and retry (capacity fluctuates)
# Script automatically retries 3 times with 30s delay

# Alternative: Provision via OCI console manually
# Then skip provisioning: deploy_to_oracle.sh --skip-provisioning
```

#### 2. Tailscale auth key timeout

**Symptom:** Setup hangs at "Authenticating with Tailscale..."

**Solutions:**

```bash
# Check auth key is valid
# Generate new key: https://login.tailscale.com/admin/settings/keys

# Use interactive login instead (SSH into instance)
ssh ubuntu@<public-ip>
sudo tailscale up
# Follow browser link to authorize

# Get Tailscale IP
tailscale ip -4

# Manually update .env
echo "TAILSCALE_IP=$(tailscale ip -4)" >> /opt/mcp-memory/.env
```

#### 3. Health check timeout

**Symptom:** Deployment fails at "Waiting for health check..."

**Solutions:**

```bash
# SSH into instance and check logs
ssh ubuntu@<public-ip>
cd /opt/mcp-memory
docker compose logs mcp-memory

# Common causes:
# - Container crashed: docker compose ps (check Status)
# - Port binding error: docker compose logs | grep "Address already in use"
# - Database locked: docker compose logs | grep "database is locked"

# Restart containers
docker compose restart

# Check if health endpoint works locally
docker compose exec mcp-memory curl localhost:8000/api/health
```

#### 4. "Database is locked" errors

**Symptom:** Multiple clients get database locked errors

**Solution:**

```bash
# Verify write queue is enabled
ssh ubuntu@<public-ip>
cd /opt/mcp-memory
grep "queue_size" .env  # Should show queue in health check

# Check SQLite pragmas
grep "SQLITE_PRAGMAS" .env
# Should include: busy_timeout=15000,journal_mode=WAL

# Restart with correct settings
docker compose down
docker compose up -d

# Monitor queue
curl http://<tailscale-ip>:8000/api/health | jq .queue_size
```

#### 5. Migration fails

**Symptom:** `--with-migration` flag errors out

**Solutions:**

```bash
# Verify Cloudflare credentials
env | grep CLOUDFLARE_

# Test Cloudflare connection
python scripts/migrate/cloudflare_to_oracle.py --dry-run

# Check migration script logs
python scripts/migrate/cloudflare_to_oracle.py \
  --target-url http://<tailscale-ip>:8000 \
  2>&1 | tee migration.log

# Resume failed migration (idempotent)
python scripts/migrate/cloudflare_to_oracle.py \
  --target-url http://<tailscale-ip>:8000
```

#### 6. Can't access from client (Tailscale network)

**Symptom:** `curl: (7) Failed to connect to <tailscale-ip>:8000`

**Solutions:**

```bash
# Verify you're on Tailscale network
tailscale status

# Test connection to Oracle instance
tailscale ping <tailscale-ip>

# Check if HTTP server is bound correctly
ssh ubuntu@<public-ip>
netstat -tuln | grep 8000
# Should show: tcp 0 0 100.x.x.x:8000 LISTEN

# Verify .env has correct TAILSCALE_IP
cat /opt/mcp-memory/.env | grep TAILSCALE_IP

# Check firewall (shouldn't block Tailscale)
sudo iptables -L -n | grep 8000
```

#### 7. Backup container exits

**Symptom:** `docker compose ps` shows backup with Exit status

**Solutions:**

```bash
# Check logs
docker compose logs backup

# Common errors:
# - Invalid R2 credentials: Check .env R2_* variables
# - Bucket doesn't exist: Create in Cloudflare dashboard
# - rclone.conf syntax error: Validate with rclone config show

# Test rclone connection
docker compose run backup rclone lsd r2:

# Manually create bucket if needed
docker compose run backup rclone mkdir r2:mcp-memory-backups

# Restart backup container
docker compose restart backup
```

### Performance Issues

#### High latency (>100ms)

```bash
# Test network latency
ping <tailscale-ip>

# Check CPU/RAM usage
ssh ubuntu@<public-ip>
docker stats

# Reduce worker count if memory pressure
# Edit .env: UVICORN_WORKERS=2
docker compose restart
```

#### Memory exhaustion

```bash
# Check Docker stats
docker stats --no-stream

# Identify memory leak
docker compose logs mcp-memory | grep -i "memory\|oom"

# Restart containers to clear
docker compose restart
```

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Layer (Melbourne)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │Claude Desktop│  │  Claude Code │  │   VS Code    │  ...      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           │ http_client.py (MCP_HTTP_CLIENT_     │
│                           │ ENDPOINT=http://100.x.x.x:8000)      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                     Tailscale VPN (WireGuard mesh)
                            │
┌───────────────────────────┼──────────────────────────────────────┐
│              Oracle Cloud Free Tier (AU Region)                  │
│                           │                                       │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │              Tailscale Container (100.x.x.x)              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │             Docker Compose Stack                          │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  mcp-memory Service                              │     │   │
│  │  │  ┌──────────────────────────────────────────┐   │     │   │
│  │  │  │ FastAPI HTTP Server (port 8000)          │   │     │   │
│  │  │  │  - Health checks (/api/health)           │   │     │   │
│  │  │  │  - Write queue (BackgroundTasks)         │   │     │   │
│  │  │  │  - Input validation (10MB, 50 tags)      │   │     │   │
│  │  │  └──────────────┬───────────────────────────┘   │     │   │
│  │  │                 │                                │     │   │
│  │  │  ┌──────────────▼───────────────────────────┐   │     │   │
│  │  │  │ SQLite-vec (WAL mode, busy_timeout)      │   │     │   │
│  │  │  │ Named Volume: sqlite-data                │   │     │   │
│  │  │  └──────────────────────────────────────────┘   │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  rclone Backup Sidecar                          │     │   │
│  │  │  - Mounts sqlite-data volume (read-only)        │     │   │
│  │  │  - Syncs to Cloudflare R2 every 6 hours         │     │   │
│  │  │  - Retains 7 local + 30 cloud backups           │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  VM.Standard.A1.Flex (4 ARM cores, 24GB RAM) - Ubuntu 24.04 LTS  │
└───────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **MCP Clients** | Connect to memory service | http_client.py backend |
| **Tailscale VPN** | Secure private network | WireGuard mesh protocol |
| **FastAPI Server** | HTTP API + write queue | Python, uvicorn (4 workers) |
| **SQLite-vec** | Vector database + storage | SQLite with vec extension |
| **Backup Sidecar** | Automated backups | rclone + Cloudflare R2 |

### Data Flow

1. **Client Request** → Sent over Tailscale VPN to Oracle server
2. **FastAPI Receives** → Validates input (max 10MB, 50 tags)
3. **Write Queue** → Serializes concurrent writes (backpressure at 20)
4. **SQLite Write** → Stored with WAL journaling (busy_timeout=15s)
5. **Response** → Returns content_hash to client
6. **Backup** → rclone syncs to R2 every 6 hours

---

## Maintenance

### Updating Containers

```bash
# SSH into Oracle instance
ssh ubuntu@<public-ip>

# Pull latest code (if updates available)
cd /opt/mcp-memory
git pull origin main

# Rebuild and restart
docker compose build
docker compose up -d

# Verify health
curl http://localhost:8000/api/health
```

### Backup Restore Procedure

**Scenario:** Database corrupted or need to restore previous state

```bash
# SSH into Oracle instance
ssh ubuntu@<public-ip>

# Stop containers
cd /opt/mcp-memory
docker compose down

# List available backups
docker compose run backup rclone ls r2:mcp-memory-backups/

# Restore from R2
docker compose run backup rclone copy \
  r2:mcp-memory-backups/sqlite_vec.db \
  /data/sqlite_vec.db

# Or restore from timestamped backup
docker compose run backup rclone copy \
  r2:mcp-memory-backups/archives/sqlite_vec-2025-01-15-120000.db \
  /data/sqlite_vec.db

# Restart containers
docker compose up -d

# Verify restoration
curl http://localhost:8000/api/health
```

### Log Management

```bash
# View real-time logs
docker compose logs -f

# View last 100 lines
docker compose logs --tail=100

# View specific service
docker compose logs mcp-memory
docker compose logs backup

# Save logs to file
docker compose logs > deployment.log
```

### Monitoring

```bash
# Container stats (CPU, RAM, network)
docker stats

# Disk usage
docker system df

# Database size
ssh ubuntu@<public-ip>
du -h /var/lib/docker/volumes/oracle_sqlite-data/_data/sqlite_vec.db
```

### Cost Management

**Oracle Free Tier (Always Free):**
- ✅ VM.Standard.A1.Flex (4 ARM cores, 24GB RAM)
- ✅ 200GB block storage
- ✅ 10TB outbound traffic/month
- ✅ No expiration

**Cloudflare R2 (Optional Backups):**
- Storage: $0.015/GB/month (~$0.50/month for 30GB)
- Operations: 10M free Class A, 10M free Class B per month
- Estimated: **~$1/month**

**Total monthly cost: ~$1** (R2 only, Oracle is free)

---

## Additional Resources

- **Project Repository**: https://github.com/doobidoo/mcp-memory-service
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Oracle Cloud Free Tier**: https://www.oracle.com/cloud/free/
- **Tailscale**: https://tailscale.com/kb/
- **Cloudflare R2**: https://developers.cloudflare.com/r2/

---

## Support

**Issues or Questions?**

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing issues: https://github.com/doobidoo/mcp-memory-service/issues
3. Open new issue with:
   - Deployment logs: `docker compose logs`
   - Environment: `cat deployment/oracle/.env | grep -v "SECRET\|KEY"`
   - Test results: `bash scripts/verify/test_deployment.sh`

**Community:**

- GitHub Discussions: https://github.com/doobidoo/mcp-memory-service/discussions
- Project Wiki: https://github.com/doobidoo/mcp-memory-service/wiki

---

**Deployment Status Checklist:**

- [ ] Oracle instance provisioned
- [ ] Tailscale VPN configured
- [ ] Docker containers running
- [ ] Health check passing
- [ ] Test suite passing (8/8 tests)
- [ ] Client configuration complete
- [ ] Backups configured (optional)
- [ ] Performance validated (<100ms latency)

**Happy deploying!** 🚀
