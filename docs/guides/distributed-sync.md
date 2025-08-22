# Distributed Memory Synchronization Guide

This guide provides comprehensive documentation for the distributed memory synchronization system introduced in MCP Memory Service v6.3.0.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Advanced Configuration](#advanced-configuration)
- [Command Reference](#command-reference)
- [Service Management](#service-management)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The distributed memory synchronization system transforms your MCP Memory Service into a distributed architecture with seamless synchronization between local development environments and remote production servers. It features:

- **Git-like workflow**: Stage → Pull → Apply → Push operations
- **Remote-first architecture**: Direct API communication with local fallback
- **Real-time replication**: Litestream integration for live database sync
- **Offline capability**: Local staging database for disconnected development
- **Intelligent conflict resolution**: Content hash-based deduplication
- **Cross-platform support**: Linux, macOS, and Windows compatibility

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Local Client   │    │  Remote Server  │    │ Backup Storage  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Staging DB  │ │◄──►│ │Production DB│ │◄──►│ │ Litestream  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ │  Replicas   │ │
│                 │    │                 │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │                 │
│ │ Sync Engine │ │◄──►│ │  HTTP API   │ │    │                 │
│ └─────────────┘ │    │ └─────────────┘ │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Remote-First Storage**: `enhanced_memory_store.sh` attempts direct API storage first
2. **Local Staging**: Falls back to local SQLite database if remote unavailable
3. **Background Sync**: `memory_sync.sh` periodically synchronizes staged changes
4. **Real-time Replication**: Litestream provides continuous database backup/sync
5. **Conflict Resolution**: Content hash comparison prevents duplicate memories

## Quick Start

### Prerequisites

- MCP Memory Service v6.3.0+ installed
- Remote server accessible via HTTPS (e.g., `narrowbox.local:8443`)
- SQLite-vec backend configured on both client and server

### Initial Setup

1. **Install the sync system:**
```bash
cd mcp-memory-service
./scripts/memory_sync.sh install
```

2. **Configure remote connection:**
```bash
export REMOTE_MEMORY_HOST="narrowbox.local"
export REMOTE_MEMORY_PORT="8443"
export REMOTE_MEMORY_PROTOCOL="https"
```

3. **Initialize synchronization:**
```bash
./scripts/memory_sync.sh init
```

4. **Verify connectivity:**
```bash
./scripts/memory_sync.sh status
```

### Basic Usage

**Store memories (remote-first):**
```bash
./scripts/enhanced_memory_store.sh "Important development decision"
```

**Synchronize changes:**
```bash
./scripts/memory_sync.sh sync
```

**Check sync status:**
```bash
./scripts/memory_sync.sh status
```

## Advanced Configuration

### Environment Variables

#### Core Sync Settings
```bash
# Remote server configuration
REMOTE_MEMORY_HOST="narrowbox.local"          # Server hostname
REMOTE_MEMORY_PORT="8443"                     # Server port
REMOTE_MEMORY_PROTOCOL="https"                # http or https
REMOTE_MEMORY_API_KEY="your-api-key"          # Optional authentication

# Local staging configuration
STAGING_DB_PATH="$HOME/.mcp_memory_staging/staging.db"
SYNC_LOG_PATH="$HOME/.mcp_memory_staging/sync.log"

# Sync behavior
SYNC_CONFLICT_RESOLUTION="merge"              # merge, local, remote
SYNC_BATCH_SIZE="100"                         # Memories per batch
SYNC_RETRY_ATTEMPTS="3"                       # Network retry count
SYNC_OFFLINE_THRESHOLD="5"                    # Seconds before offline mode
SYNC_INTERVAL="900"                           # Auto-sync interval (seconds)
```

#### Litestream Configuration
```bash
# Replication settings
LITESTREAM_REPLICA_PATH="/backup/sqlite_vec.db"
LITESTREAM_SYNC_INTERVAL="1s"                 # Real-time replication
LITESTREAM_RETENTION="72h"                    # Backup retention period
LITESTREAM_COMPRESS="true"                    # Enable compression
```

### Custom Configuration Files

#### Litestream Configuration (`litestream.yml`)
```yaml
dbs:
  - path: /path/to/sqlite_vec.db
    replicas:
      - name: "local-backup"
        type: file
        path: /backup/sqlite_vec.db
        sync-interval: 1s
        retention: 72h
        
      - name: "remote-backup"
        url: "http://backup-server:8080/replica"
        sync-interval: 10s
        retention: 168h
```

#### Sync Configuration (`sync_config.json`)
```json
{
  "remote": {
    "host": "narrowbox.local",
    "port": 8443,
    "protocol": "https",
    "timeout": 30,
    "retry_attempts": 3
  },
  "local": {
    "staging_db": "~/.mcp_memory_staging/staging.db",
    "log_file": "~/.mcp_memory_staging/sync.log",
    "batch_size": 100
  },
  "conflict_resolution": "merge",
  "auto_sync_interval": 900,
  "litestream_enabled": true
}
```

## Command Reference

### Primary Commands

#### `memory_sync.sh` - Main Orchestrator
```bash
# Basic operations
./scripts/memory_sync.sh status           # Show sync status
./scripts/memory_sync.sh sync             # Full synchronization
./scripts/memory_sync.sh pull             # Pull remote changes
./scripts/memory_sync.sh push             # Push local changes

# Advanced operations
./scripts/memory_sync.sh init             # Initialize sync system
./scripts/memory_sync.sh reset            # Reset sync state
./scripts/memory_sync.sh repair           # Repair corrupted sync
./scripts/memory_sync.sh export           # Export sync configuration

# Service management
./scripts/memory_sync.sh install-service # Install as system service
./scripts/memory_sync.sh start-service   # Start sync service
./scripts/memory_sync.sh stop-service    # Stop sync service
```

#### `enhanced_memory_store.sh` - Remote-First Storage
```bash
# Basic usage
./scripts/enhanced_memory_store.sh "Memory content"

# With tags
./scripts/enhanced_memory_store.sh "Content" --tags "tag1,tag2"

# Force offline mode
./scripts/enhanced_memory_store.sh "Content" --offline

# Custom project context
./scripts/enhanced_memory_store.sh "Content" --project "my-project"
```

#### `pull_remote_changes.sh` - Remote Synchronization
```bash
# Basic pull
./scripts/pull_remote_changes.sh

# Force overwrite local changes
./scripts/pull_remote_changes.sh --force

# Dry run (show what would be synchronized)
./scripts/pull_remote_changes.sh --dry-run

# Selective sync by tags
./scripts/pull_remote_changes.sh --tags "urgent,important"
```

#### `manual_sync.sh` - HTTP-Based Sync
```bash
# Sync with conflict detection
./scripts/manual_sync.sh

# Force conflict resolution
./scripts/manual_sync.sh --resolve-conflicts

# Detailed logging
./scripts/manual_sync.sh --verbose
```

### Utility Commands

#### Database Management
```bash
# Initialize staging database
sqlite3 ~/.mcp_memory_staging/staging.db < scripts/staging_db_init.sql

# Check database integrity
./scripts/memory_sync.sh verify-db

# Optimize databases
./scripts/memory_sync.sh optimize
```

#### Litestream Integration
```bash
# Start Litestream replication
litestream replicate -config litestream.yml

# Restore from backup
litestream restore -o sqlite_vec_restored.db backup/sqlite_vec.db

# Monitor replication
litestream snapshots backup/sqlite_vec.db
```

## Service Management

### Installation as System Service

The sync system can be installed as a native system service for automatic operation:

#### Linux (systemd)
```bash
# Install service
./scripts/memory_sync.sh install-service

# Service control
systemctl --user enable mcp-memory-sync
systemctl --user start mcp-memory-sync
systemctl --user status mcp-memory-sync

# View logs
journalctl --user -u mcp-memory-sync -f
```

#### macOS (LaunchAgent)
```bash
# Install service
./scripts/memory_sync.sh install-service

# Service control  
launchctl load ~/Library/LaunchAgents/com.mcp.memory.sync.plist
launchctl start com.mcp.memory.sync
launchctl list | grep mcp.memory

# View logs
tail -f ~/Library/Logs/mcp-memory-sync.log
```

#### Windows (Windows Service)
```bash
# Install service (run as administrator)
./scripts/memory_sync.sh install-service

# Service control
net start "MCP Memory Sync"
net stop "MCP Memory Sync"
sc query "MCP Memory Sync"
```

### Service Configuration

Service files are automatically generated during installation:

- **Linux**: `~/.config/systemd/user/mcp-memory-sync.service`
- **macOS**: `~/Library/LaunchAgents/com.mcp.memory.sync.plist`
- **Windows**: Registry entries under `HKLM\SYSTEM\CurrentControlSet\Services`

## Troubleshooting

### Common Issues

#### Connection Problems
```bash
# Test remote connectivity
curl -k https://narrowbox.local:8443/api/health

# Check DNS resolution
nslookup narrowbox.local

# Test with IP address
export REMOTE_MEMORY_HOST="10.0.1.30"
./scripts/memory_sync.sh status
```

#### Database Conflicts
```bash
# Reset staging database
rm ~/.mcp_memory_staging/staging.db
./scripts/memory_sync.sh init

# Force remote state
./scripts/memory_sync.sh pull --force

# Manual conflict resolution
./scripts/manual_sync.sh --resolve-conflicts
```

#### Service Issues
```bash
# Check service status
./scripts/memory_sync.sh status-service

# Restart service
./scripts/memory_sync.sh restart-service

# View service logs
./scripts/memory_sync.sh logs
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
export SYNC_DEBUG=1
export SYNC_VERBOSE=1
./scripts/memory_sync.sh sync
```

### Log Analysis

Sync logs provide detailed operation history:

```bash
# View recent sync activity
tail -f ~/.mcp_memory_staging/sync.log

# Search for errors
grep -i error ~/.mcp_memory_staging/sync.log

# Analyze sync performance
grep "sync completed" ~/.mcp_memory_staging/sync.log | tail -10
```

## Best Practices

### Development Workflow

1. **Start each session with sync:**
   ```bash
   ./scripts/memory_sync.sh status
   ./scripts/memory_sync.sh pull
   ```

2. **Use remote-first storage:**
   ```bash
   # Preferred method
   ./scripts/enhanced_memory_store.sh "Important insight"
   
   # Instead of direct MCP operations
   ```

3. **Regular synchronization:**
   ```bash
   # Set up automatic sync
   crontab -e
   # Add: */15 * * * * /path/to/memory_sync.sh sync >/dev/null 2>&1
   ```

4. **End session with push:**
   ```bash
   ./scripts/memory_sync.sh push
   ./scripts/memory_sync.sh status
   ```

### Production Deployment

1. **Server configuration:**
   ```bash
   # Enable API access
   export MCP_HTTP_ENABLED=true
   export MCP_API_KEY="$(openssl rand -base64 32)"
   
   # Start server
   python scripts/run_http_server.py
   ```

2. **Security considerations:**
   - Use HTTPS with valid certificates
   - Configure API key authentication
   - Set up firewall rules for port 8443
   - Enable request rate limiting

3. **Monitoring setup:**
   ```bash
   # Health check endpoint
   curl -f https://server:8443/api/health
   
   # Monitor sync performance
   ./scripts/memory_sync.sh metrics
   ```

### Data Management

1. **Regular backups:**
   ```bash
   # Enable Litestream replication
   export LITESTREAM_ENABLED=true
   litestream replicate -config litestream.yml
   ```

2. **Content organization:**
   ```bash
   # Use meaningful tags
   ./scripts/enhanced_memory_store.sh "Feature decision" --tags "architecture,decisions,v6.3"
   ```

3. **Cleanup old data:**
   ```bash
   # Archive old memories
   ./scripts/memory_sync.sh archive --older-than "90 days"
   ```

### Performance Optimization

1. **Batch operations:**
   ```bash
   export SYNC_BATCH_SIZE=50  # Smaller batches for slow networks
   ```

2. **Network optimization:**
   ```bash
   export SYNC_RETRY_ATTEMPTS=5
   export SYNC_OFFLINE_THRESHOLD=10
   ```

3. **Storage optimization:**
   ```bash
   # Optimize databases regularly
   ./scripts/memory_sync.sh optimize
   ```

## Integration Examples

### CI/CD Pipeline Integration
```yaml
# GitHub Actions example
- name: Sync Memory Context
  run: |
    ./scripts/memory_sync.sh pull
    ./scripts/enhanced_memory_store.sh "Build ${{ github.run_number }} completed" --tags "ci,builds"
    ./scripts/memory_sync.sh push
```

### IDE Integration
```json
// VS Code tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Sync Memories",
      "type": "shell",
      "command": "./scripts/memory_sync.sh",
      "args": ["sync"],
      "group": "build"
    }
  ]
}
```

### Shell Aliases
```bash
# Add to .bashrc/.zshrc
alias msync='cd /path/to/mcp-memory-service && ./scripts/memory_sync.sh sync'
alias mstore='cd /path/to/mcp-memory-service && ./scripts/enhanced_memory_store.sh'
alias mstatus='cd /path/to/mcp-memory-service && ./scripts/memory_sync.sh status'
```

This completes the comprehensive distributed synchronization documentation. The system provides enterprise-grade distributed memory management with offline capability, real-time replication, and Git-like workflow patterns.