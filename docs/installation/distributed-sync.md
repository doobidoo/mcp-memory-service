# Distributed Sync Installation Guide

This guide covers the installation and initial setup of the distributed memory synchronization system for MCP Memory Service v6.3.0+.

## Prerequisites

### System Requirements
- **MCP Memory Service v6.3.0+** installed and configured
- **SQLite-vec backend** (distributed sync is optimized for SQLite-vec)
- **Remote server** with HTTPS-enabled MCP Memory Service
- **Network connectivity** between client and server
- **Bash shell** (for sync scripts)

### Supported Platforms
- **Linux**: All distributions with systemd support
- **macOS**: macOS 10.14+ with Homebrew recommended
- **Windows**: Windows 10+ with Git Bash or WSL

### Dependencies

The following tools are required and will be installed automatically if missing:

- `curl` - HTTP client for API communication
- `sqlite3` - SQLite database management
- `jq` - JSON processing (optional but recommended)
- `litestream` - Real-time database replication (optional)

## Quick Installation

### 1. Install Base MCP Memory Service

If not already installed:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Install with SQLite-vec backend
python install.py --storage-backend sqlite_vec
```

### 2. Install Distributed Sync System

```bash
# Install sync system
./scripts/memory_sync.sh install

# Configure remote server
export REMOTE_MEMORY_HOST="your-server.local"
export REMOTE_MEMORY_PORT="8443"

# Initialize synchronization
./scripts/memory_sync.sh init

# Verify installation
./scripts/memory_sync.sh status
```

### 3. Quick Verification Test

```bash
# Test remote connectivity
curl -k https://your-server.local:8443/api/health

# Test memory storage (remote-first)
./scripts/enhanced_memory_store.sh "Test sync installation"

# Verify sync status
./scripts/memory_sync.sh status
```

## Detailed Installation Steps

### Step 1: Server Setup

First, ensure your remote server has MCP Memory Service running with HTTP API enabled:

```bash
# On the remote server
cd mcp-memory-service

# Configure for remote access
export MCP_HTTP_ENABLED=true
export MCP_HTTP_HOST="0.0.0.0"  # Allow external connections
export MCP_HTTP_PORT="8443"
export MCP_HTTPS_ENABLED=true   # Enable HTTPS
export MCP_API_KEY="$(openssl rand -base64 32)"  # Generate secure API key

# Start the server
python scripts/run_http_server.py
```

**Verify server is accessible:**
```bash
# From client machine
curl -k https://your-server.local:8443/api/health
```

### Step 2: Client Installation

#### Install Sync Components

```bash
cd mcp-memory-service

# Run installation script
./scripts/memory_sync.sh install
```

**The installer will:**
1. Create staging database directory (`~/.mcp_memory_staging/`)
2. Initialize staging SQLite database
3. Create configuration files
4. Set up log directories
5. Install service files (optional)
6. Configure automatic sync (optional)

#### Configure Remote Connection

```bash
# Create configuration file
cat > ~/.mcp_memory_staging/sync_config.json << EOF
{
  "remote": {
    "host": "your-server.local",
    "port": 8443,
    "protocol": "https",
    "api_key": "your-api-key-if-required",
    "timeout": 30,
    "retry_attempts": 3
  },
  "local": {
    "staging_db": "~/.mcp_memory_staging/staging.db",
    "log_file": "~/.mcp_memory_staging/sync.log",
    "batch_size": 100
  },
  "sync": {
    "conflict_resolution": "merge",
    "auto_sync_interval": 900,
    "offline_threshold": 5
  }
}
EOF
```

**Or use environment variables:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export REMOTE_MEMORY_HOST="your-server.local"
export REMOTE_MEMORY_PORT="8443"
export REMOTE_MEMORY_PROTOCOL="https"
export REMOTE_MEMORY_API_KEY="your-api-key"  # if required
```

### Step 3: Initialize Synchronization

```bash
# Initialize sync system
./scripts/memory_sync.sh init

# This will:
# 1. Test remote connectivity
# 2. Create staging database schema
# 3. Perform initial sync
# 4. Set up conflict resolution rules
```

### Step 4: Verification and Testing

```bash
# Check overall system status
./scripts/memory_sync.sh status

# Test remote storage
./scripts/enhanced_memory_store.sh "Installation test message"

# Test offline storage (simulate network failure)
./scripts/enhanced_memory_store.sh "Offline test message" --offline

# Test synchronization
./scripts/memory_sync.sh sync

# Check sync status again
./scripts/memory_sync.sh status
```

## Platform-Specific Installation

### Linux Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install curl sqlite3 jq  # Ubuntu/Debian
# OR
sudo yum install curl sqlite jq    # RHEL/CentOS
# OR
sudo pacman -S curl sqlite jq      # Arch Linux

# Install sync system
./scripts/memory_sync.sh install

# Optional: Install as systemd service
./scripts/memory_sync.sh install-service
systemctl --user enable mcp-memory-sync
systemctl --user start mcp-memory-sync
```

### macOS Installation

```bash
# Install dependencies via Homebrew
brew install curl sqlite jq litestream

# Install sync system
./scripts/memory_sync.sh install

# Optional: Install as LaunchAgent
./scripts/memory_sync.sh install-service
launchctl load ~/Library/LaunchAgents/com.mcp.memory.sync.plist
```

### Windows Installation

Using Git Bash or WSL:

```bash
# Ensure dependencies are available
which curl sqlite3 jq

# Install sync system
./scripts/memory_sync.sh install

# Optional: Install as Windows Service (requires admin privileges)
./scripts/memory_sync.sh install-service
```

## Advanced Installation Options

### Custom Installation Paths

```bash
# Custom staging database location
export STAGING_DB_PATH="/custom/path/staging.db"
export SYNC_LOG_PATH="/custom/path/sync.log"

# Install with custom paths
./scripts/memory_sync.sh install --custom-paths
```

### Litestream Integration

For real-time database replication:

```bash
# Install Litestream
# Linux
curl -sSfL https://github.com/benbjohnson/litestream/releases/download/v0.3.13/litestream-v0.3.13-linux-amd64.tar.gz | tar -xzf - -C /tmp
sudo mv /tmp/litestream /usr/local/bin/

# macOS
brew install litestream

# Configure Litestream
cat > litestream.yml << EOF
dbs:
  - path: ~/.mcp_memory/sqlite_vec.db
    replicas:
      - name: "local-backup"
        path: ~/.mcp_memory_staging/backup.db
        sync-interval: 1s
        retention: 72h
EOF

# Start Litestream replication
litestream replicate -config litestream.yml &
```

### Service Installation Options

```bash
# Install with automatic startup
./scripts/memory_sync.sh install-service --auto-start

# Install with custom sync interval (minutes)
./scripts/memory_sync.sh install-service --interval 10

# Install with custom log rotation
./scripts/memory_sync.sh install-service --log-rotate daily
```

## Configuration Validation

After installation, validate your configuration:

### Network Connectivity
```bash
# Test basic connectivity
ping your-server.local

# Test HTTPS endpoint
curl -k https://your-server.local:8443/api/health

# Test API authentication (if enabled)
curl -k -H "Authorization: Bearer your-api-key" \
  https://your-server.local:8443/api/health
```

### Database Setup
```bash
# Verify staging database
sqlite3 ~/.mcp_memory_staging/staging.db ".tables"

# Check database schema
sqlite3 ~/.mcp_memory_staging/staging.db ".schema memories"

# Test staging database functionality
./scripts/enhanced_memory_store.sh "Database test" --offline
sqlite3 ~/.mcp_memory_staging/staging.db "SELECT * FROM memories;"
```

### Sync Functionality
```bash
# Test full sync workflow
./scripts/memory_sync.sh sync --dry-run  # Preview changes
./scripts/memory_sync.sh sync            # Perform sync
./scripts/memory_sync.sh status          # Check results
```

## Post-Installation Setup

### 1. Configure Automatic Sync

Add to crontab for automatic synchronization:
```bash
crontab -e
# Add line (sync every 15 minutes):
*/15 * * * * /path/to/mcp-memory-service/scripts/memory_sync.sh sync >/dev/null 2>&1
```

### 2. Set Up Shell Aliases

Add to `~/.bashrc` or `~/.zshrc`:
```bash
# Memory sync aliases
alias msync='cd /path/to/mcp-memory-service && ./scripts/memory_sync.sh sync'
alias mstore='cd /path/to/mcp-memory-service && ./scripts/enhanced_memory_store.sh'
alias mstatus='cd /path/to/mcp-memory-service && ./scripts/memory_sync.sh status'
```

### 3. Configure IDE Integration

For VS Code, add to `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Sync Memories",
      "type": "shell",
      "command": "./scripts/memory_sync.sh",
      "args": ["sync"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

## Troubleshooting Installation

### Common Installation Issues

#### Permission Errors
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix staging directory permissions
chmod 755 ~/.mcp_memory_staging/
chmod 644 ~/.mcp_memory_staging/*
```

#### Missing Dependencies
```bash
# Check for required tools
./scripts/memory_sync.sh check-dependencies

# Install missing dependencies manually
sudo apt install curl sqlite3 jq  # Linux
brew install curl sqlite jq       # macOS
```

#### Network Issues
```bash
# Test with IP address instead of hostname
export REMOTE_MEMORY_HOST="10.0.1.30"
./scripts/memory_sync.sh status

# Test without HTTPS (if server supports HTTP)
export REMOTE_MEMORY_PROTOCOL="http"
export REMOTE_MEMORY_PORT="8000"
```

#### Database Issues
```bash
# Recreate staging database
rm ~/.mcp_memory_staging/staging.db
./scripts/memory_sync.sh init

# Check SQLite version compatibility
sqlite3 --version
```

### Installation Logs

Check installation logs for detailed error information:

```bash
# View installation log
cat ~/.mcp_memory_staging/install.log

# View sync logs
tail -f ~/.mcp_memory_staging/sync.log

# Enable debug logging for next installation
export SYNC_DEBUG=1
./scripts/memory_sync.sh install
```

### Getting Help

If you encounter issues:

1. **Check the logs** first: `~/.mcp_memory_staging/sync.log`
2. **Test components individually**: server connectivity, database access, sync operations
3. **Enable debug mode**: `export SYNC_DEBUG=1`
4. **Review configuration**: ensure all paths and endpoints are correct
5. **Check file permissions**: all scripts should be executable

### Support Commands

```bash
# Generate system information for support
./scripts/memory_sync.sh system-info

# Create diagnostic report
./scripts/memory_sync.sh diagnose

# Test all components
./scripts/memory_sync.sh test-all
```

## Next Steps

After successful installation:

1. **Read the [Distributed Sync Guide](distributed-sync.md)** for detailed usage instructions
2. **Set up automatic synchronization** using the service installation
3. **Configure your development workflow** with the new sync commands
4. **Test disaster recovery** by simulating network failures
5. **Monitor sync performance** and adjust configuration as needed

The distributed sync system is now ready for production use with your MCP Memory Service.