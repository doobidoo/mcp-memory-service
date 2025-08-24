# Database Synchronization Guide

This guide covers setting up database synchronization for MCP Memory Service across multiple machines using JSON export/import and Litestream replication.

## Overview

The synchronization system enables you to maintain consistent memory databases across multiple devices while preserving all original timestamps, metadata, and source attribution.

### Architecture

```
                  Central Server
                  ┌──────────────────────┐
                  │  MASTER DATABASE     │
                  │  SQLite-vec (Master) │
                  │  + MCP Service API   │
                  │  + Litestream Hub    │
                  └──────┬───────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                                  │
        ▼                                  ▼
Windows PC (Replica)              MacBook (Replica)
┌─────────────────┐              ┌─────────────────┐
│ SQLite-vec DB   │              │ SQLite-vec DB   │
│ (synced copy)   │              │ (synced copy)   │
│ + Litestream    │              │ + Litestream    │
└─────────────────┘              └─────────────────┘
```

### Key Features

- **Timestamp Preservation**: Original creation and update times maintained
- **Source Tracking**: Each memory tagged with originating machine
- **Deduplication**: Content hash prevents duplicate memories
- **Real-time Sync**: Litestream provides near-instant replication
- **Conflict Resolution**: Last-write-wins with comprehensive logging

## Prerequisites

### Software Requirements

- **Python 3.10+** with MCP Memory Service installed
- **Litestream** for database replication
- **SQLite-vec** backend configured

### System Requirements

- **Network connectivity** between all machines
- **Sufficient disk space** for database replicas
- **Write permissions** to database directories

## Phase 1: Initial Database Consolidation

### Step 1: Export from Source Machines

Export memories from each machine that has existing data:

**On Windows PC:**
```bash
cd /path/to/mcp-memory-service
python scripts/sync/export_memories.py --output windows_memories.json
```

**On MacBook:**
```bash
cd /path/to/mcp-memory-service
python scripts/sync/export_memories.py --output macbook_memories.json
```

**Export Options:**
```bash
# Include embedding vectors (increases file size significantly)
python scripts/sync/export_memories.py --include-embeddings

# Export only specific tags
python scripts/sync/export_memories.py --filter-tags claude-code,architecture

# Export from custom database location
python scripts/sync/export_memories.py --db-path /custom/path/sqlite_vec.db
```

### Step 2: Transfer Files to Central Server

Transfer the JSON export files to your central server:

```bash
# Using SCP
scp windows_memories.json user@central-server:/tmp/
scp macbook_memories.json user@central-server:/tmp/

# Using rsync
rsync -av *.json user@central-server:/tmp/

# Or use any file transfer method (USB, cloud storage, etc.)
```

### Step 3: Import to Central Database

On the central server, import all memories:

```bash
cd /path/to/mcp-memory-service

# First, analyze what will be imported
python scripts/sync/import_memories.py --dry-run /tmp/*.json

# Then import for real
python scripts/sync/import_memories.py /tmp/windows_memories.json /tmp/macbook_memories.json
```

### Step 4: Verify Consolidation

Check that all memories were imported correctly:

```bash
# Using the health API
curl -k https://localhost:8443/api/health/detailed

# Using the web interface
# Visit https://localhost:8443 and check memory count

# Using the CLI
python -c "
import asyncio
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

async def check():
    storage = SqliteVecMemoryStorage('/path/to/sqlite_vec.db')
    await storage.initialize()
    memories = await storage.get_all_memories()
    
    print(f'Total memories: {len(memories)}')
    
    # Count by source
    sources = {}
    for m in memories:
        for tag in m.tags:
            if tag.startswith('source:'):
                source = tag.replace('source:', '')
                sources[source] = sources.get(source, 0) + 1
    
    for source, count in sources.items():
        print(f'  {source}: {count} memories')

asyncio.run(check())
"
```

## Phase 2: Litestream Setup

### Install Litestream

**On Linux (Central Server):**
```bash
curl -LsS https://github.com/benbjohnson/litestream/releases/latest/download/litestream-linux-amd64.tar.gz | tar -xzf -
sudo mv litestream /usr/local/bin/
sudo chmod +x /usr/local/bin/litestream
```

**On macOS:**
```bash
brew install benbjohnson/litestream/litestream
```

**On Windows:**
1. Download from [Litestream Releases](https://github.com/benbjohnson/litestream/releases)
2. Extract `litestream.exe` to `C:\Program Files\Litestream\`
3. Add to PATH environment variable

### Configure Central Server (Master)

Create `/etc/litestream.yml` on the central server:

```yaml
dbs:
  - path: /home/user/.local/share/mcp-memory/sqlite_vec.db
    replicas:
      # HTTP endpoint for replica downloads
      - type: file
        path: /var/www/litestream/mcp-memory
        sync-interval: 10s
      
      # Local backup
      - type: file
        path: /backup/litestream/mcp-memory
        sync-interval: 1m
    
    # Performance settings
    checkpoint-interval: 30s
    wal-retention: 10m
```

### Configure Replica Machines

**Windows Client** (`C:\ProgramData\litestream\litestream.yml`):
```yaml
dbs:
  - path: C:\Users\YourUser\.local\share\mcp-memory\sqlite_vec.db
    replicas:
      - type: file
        url: http://YOUR_CENTRAL_SERVER/litestream/mcp-memory
        sync-interval: 10s
```

**macOS Client** (`/usr/local/etc/litestream.yml`):
```yaml
dbs:
  - path: /Users/YourUser/.local/share/mcp-memory/sqlite_vec.db
    replicas:
      - type: file
        url: http://YOUR_CENTRAL_SERVER/litestream/mcp-memory
        sync-interval: 10s
```

### Set Up HTTP Server for Replicas

On the central server, set up a web server to serve Litestream replicas:

**Using nginx:**
```nginx
server {
    listen 80;
    server_name YOUR_CENTRAL_SERVER;
    
    location /litestream/ {
        alias /var/www/litestream/;
        autoindex on;
        
        # Allow cross-origin requests
        add_header Access-Control-Allow-Origin *;
    }
}
```

**Using Python HTTP server (for testing):**
```bash
cd /var/www/litestream
python -m http.server 8080
```

### Start Litestream Services

**On Central Server:**
```bash
# Create systemd service
sudo tee /etc/systemd/system/litestream.service > /dev/null <<EOF
[Unit]
Description=Litestream replication service
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=1
User=root
ExecStart=/usr/local/bin/litestream replicate -config /etc/litestream.yml

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable litestream
sudo systemctl start litestream
sudo systemctl status litestream
```

**On macOS:**
```bash
# Create LaunchDaemon
sudo tee /Library/LaunchDaemons/io.litestream.replication.plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.litestream.replication</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/litestream</string>
        <string>replicate</string>
        <string>-config</string>
        <string>/usr/local/etc/litestream.yml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Load and start
sudo launchctl load /Library/LaunchDaemons/io.litestream.replication.plist
sudo launchctl start io.litestream.replication
```

**On Windows:**
```cmd
# Install as Windows Service (requires NSSM or similar)
# Alternative: Run from Task Scheduler
litestream replicate -config C:\ProgramData\litestream\litestream.yml
```

### Initial Replica Sync

**On replica machines, perform initial sync:**

```bash
# Stop MCP Memory Service first
sudo systemctl stop mcp-memory  # Linux
launchctl unload ~/Library/LaunchAgents/mcp-memory.plist  # macOS

# Restore from master
litestream restore -config /path/to/litestream.yml /path/to/local/sqlite_vec.db

# Restart MCP Memory Service
sudo systemctl start mcp-memory  # Linux
launchctl load ~/Library/LaunchAgents/mcp-memory.plist  # macOS

# Start Litestream replication
litestream replicate -config /path/to/litestream.yml
```

## Phase 3: Monitoring and Maintenance

### Health Monitoring

Create a monitoring script to check sync status:

```python
#!/usr/bin/env python3
import asyncio
import aiohttp
import logging

async def check_sync_status():
    """Check synchronization status across all nodes."""
    
    nodes = {
        "central": "https://central-server:8443/api/health",
        "windows": "https://windows-pc:8443/api/health",
        "macbook": "https://macbook.local:8443/api/health"
    }
    
    async with aiohttp.ClientSession() as session:
        for name, url in nodes.items():
            try:
                async with session.get(url, ssl=False) as response:
                    data = await response.json()
                    print(f"{name}: {data['memory_count']} memories, "
                          f"last modified: {data.get('last_modified', 'unknown')}")
            except Exception as e:
                print(f"{name}: ERROR - {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_sync_status())
```

### Litestream Status

Check Litestream replication status:

```bash
# View replication positions
litestream wal -config /etc/litestream.yml

# Check replica status
litestream replicas -config /etc/litestream.yml

# Monitor logs
journalctl -u litestream -f  # Linux
tail -f /var/log/litestream.log  # macOS
```

### Backup Strategy

Even with replication, maintain regular backups:

```bash
#!/bin/bash
# backup_master.sh

BACKUP_DIR="/backup/mcp-memory/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Backup main database
cp /home/user/.local/share/mcp-memory/sqlite_vec.db "$BACKUP_DIR/"

# Backup Litestream replica
cp -r /var/www/litestream/mcp-memory "$BACKUP_DIR/litestream/"

# Compress and retain last 30 days
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Cleanup old backups
find /backup/mcp-memory/ -name "*.tar.gz" -mtime +30 -delete
```

## Troubleshooting

### Common Issues

1. **Sync Lag**
   ```bash
   # Check network connectivity
   ping central-server
   
   # Verify HTTP endpoint
   curl http://central-server/litestream/mcp-memory/
   
   # Check Litestream logs
   litestream replicas
   ```

2. **Permission Errors**
   ```bash
   # Fix database permissions
   chmod 644 /path/to/sqlite_vec.db
   chown user:group /path/to/sqlite_vec.db
   
   # Fix directory permissions
   chmod 755 /path/to/database/directory/
   ```

3. **Conflicts**
   ```bash
   # Check for WAL mode conflicts
   sqlite3 /path/to/sqlite_vec.db "PRAGMA journal_mode;"
   
   # Force checkpoint
   sqlite3 /path/to/sqlite_vec.db "PRAGMA wal_checkpoint(FULL);"
   ```

### Recovery Procedures

**If replica becomes corrupted:**
```bash
# Stop services
sudo systemctl stop mcp-memory litestream

# Restore from master
litestream restore -o /path/to/sqlite_vec.db http://central-server/litestream/mcp-memory/

# Restart services
sudo systemctl start litestream mcp-memory
```

**If master fails:**
```bash
# Promote replica to master
# 1. Stop Litestream on all nodes
# 2. Update configurations to point to new master
# 3. Start Litestream with new topology
```

## Performance Optimization

### Network Optimization
- Use local network for replication when possible
- Configure appropriate sync intervals based on usage
- Monitor bandwidth usage

### Database Optimization
- Regular VACUUM operations on master
- Monitor database size growth
- Optimize checkpoint intervals

### Monitoring Alerts
Set up alerts for:
- Replication lag > 60 seconds
- Database size growth > 10% per day
- Service downtime
- Network connectivity issues

## Security Considerations

- Use HTTPS for replication endpoints in production
- Implement proper authentication for replica access
- Encrypt backups and transfer files
- Monitor access logs for anomalies
- Regular security audits of sync infrastructure

This synchronization setup provides robust, real-time database replication while preserving all memory metadata and enabling distributed workflows across multiple machines.