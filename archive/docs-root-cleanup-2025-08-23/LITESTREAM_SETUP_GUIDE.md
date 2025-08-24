# Litestream Synchronization Setup Guide

This guide will help you set up real-time database synchronization between your local macOS machine and your remote server at `your-remote-server:8443`.

## Overview

- **Master**: `your-remote-server` (serves replica data via HTTP on port 8080)
- **Replica**: Local macOS machine (syncs from master every 10 seconds)
- **HTTP Server**: Python built-in server (lightweight, no additional dependencies)

## Files Created

The following configuration files have been generated:

### Configuration Files
- `litestream_master_config.yml` - Litestream master configuration for remote server
- `litestream_replica_config.yml` - Litestream replica configuration for local machine

### Service Files
- `litestream.service` - Systemd service for Litestream master
- `litestream-http.service` - Systemd service for HTTP server
- `io.litestream.replication.plist` - macOS LaunchDaemon for replica

### Setup Scripts
- `setup_remote_litestream.sh` - Automated setup for remote server
- `setup_local_litestream.sh` - Automated setup for local machine

## Step 1: Remote Server Setup (your-remote-server)

### Option A: Automated Setup
```bash
# Copy files to remote server
scp litestream_master_config.yml litestream.service litestream-http.service setup_remote_litestream.sh user@your-remote-server:/tmp/

# SSH to remote server and run setup
ssh user@your-remote-server
cd /tmp
sudo ./setup_remote_litestream.sh
```

### Option B: Manual Setup
```bash
# Install Litestream
curl -LsS https://github.com/benbjohnson/litestream/releases/latest/download/litestream-linux-amd64.tar.gz | tar -xzf -
sudo mv litestream /usr/local/bin/
sudo chmod +x /usr/local/bin/litestream

# Create directories
sudo mkdir -p /var/www/litestream/mcp-memory
sudo mkdir -p /backup/litestream/mcp-memory
sudo chown -R www-data:www-data /var/www/litestream
sudo chmod -R 755 /var/www/litestream

# Install configuration
sudo cp litestream_master_config.yml /etc/litestream.yml

# Install systemd services
sudo cp litestream.service litestream-http.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable litestream litestream-http
```

### Start Services
```bash
# Start both services
sudo systemctl start litestream litestream-http

# Check status
sudo systemctl status litestream litestream-http

# Verify HTTP endpoint
curl http://localhost:8080/mcp-memory/
```

## Step 2: Local Machine Setup (macOS)

### Option A: Automated Setup
```bash
# Run the setup script
sudo ./setup_local_litestream.sh
```

### Option B: Manual Setup
```bash
# Install configuration
sudo mkdir -p /usr/local/etc
sudo cp litestream_replica_config.yml /usr/local/etc/litestream.yml

# Create log directory
sudo mkdir -p /var/log
sudo touch /var/log/litestream.log
sudo chmod 644 /var/log/litestream.log

# Install LaunchDaemon
sudo cp io.litestream.replication.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/io.litestream.replication.plist
sudo chmod 644 /Library/LaunchDaemons/io.litestream.replication.plist
```

## Step 3: Initialize Synchronization

### Perform Initial Restore (if needed)
```bash
# Stop MCP Memory Service if running
# launchctl unload ~/Library/LaunchAgents/mcp-memory.plist  # if you have it

# Restore database from master (only needed if local DB is empty/outdated)
litestream restore -config /usr/local/etc/litestream.yml "http://your-remote-server:8080/mcp-memory" "$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
```

### Start Replica Service
```bash
# Load and start the service
sudo launchctl load /Library/LaunchDaemons/io.litestream.replication.plist
sudo launchctl start io.litestream.replication

# Check status
litestream replicas -config /usr/local/etc/litestream.yml
```

## Step 4: Verification and Testing

### Check Remote Server
```bash
# On your-remote-server
sudo systemctl status litestream litestream-http
journalctl -u litestream -f
curl http://localhost:8080/mcp-memory/
```

### Check Local Machine
```bash
# Check replica status
litestream replicas -config /usr/local/etc/litestream.yml

# Monitor logs
tail -f /var/log/litestream.log

# Check if service is running
sudo launchctl list | grep litestream
```

### Test Synchronization
```bash
# Add a test memory to the remote database (via MCP service)
curl -k -H "Content-Type: application/json" -d '{"content": "Test sync memory", "tags": ["test", "sync"]}' https://your-remote-server:8443/api/memories

# Wait 10-15 seconds, then check if it appears locally
# (You'll need to query your local database or MCP service)
```

## Monitoring and Maintenance

### Health Check Script
Create a monitoring script to check sync status:

```bash
#!/bin/bash
# health_check.sh
echo "=== Litestream Health Check ==="
echo "Remote server status:"
ssh user@your-remote-server "sudo systemctl is-active litestream litestream-http"

echo "Local replica status:"
litestream replicas -config /usr/local/etc/litestream.yml

echo "HTTP endpoint test:"
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://your-remote-server:8080/mcp-memory/
```

### Troubleshooting

**Sync lag issues:**
```bash
# Check network connectivity
ping your-remote-server

# Verify HTTP endpoint
curl http://your-remote-server:8080/mcp-memory/

# Check Litestream logs
journalctl -u litestream -f  # Remote
tail -f /var/log/litestream.log  # Local
```

**Permission errors:**
```bash
# Fix database permissions
chmod 644 "$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
```

**Service issues:**
```bash
# Restart services
sudo systemctl restart litestream litestream-http  # Remote
sudo launchctl stop io.litestream.replication && sudo launchctl start io.litestream.replication  # Local
```

## Important Notes

1. **Database Path**: Make sure the database path in the master config matches your actual SQLite-vec database location on the remote server.

2. **Network**: The local machine needs to reach `your-remote-server:8080`. Ensure firewall rules allow this.

3. **SSL/TLS**: The HTTP server runs on plain HTTP (port 8080) for simplicity. For production, consider HTTPS.

4. **Backup**: The master config includes local backups to `/backup/litestream/mcp-memory`.

5. **Performance**: Sync interval is set to 10 seconds. Adjust if needed in the configuration files.

## Next Steps

After successful setup:
1. Monitor sync performance and adjust intervals if needed
2. Set up automated health checks
3. Configure backup retention policies
4. Consider setting up alerts for sync failures