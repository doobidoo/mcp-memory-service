# Distributed Sync Troubleshooting Guide

This guide helps diagnose and resolve common issues with the distributed memory synchronization system in MCP Memory Service v6.3.0+.

## Table of Contents
- [Diagnostic Commands](#diagnostic-commands)
- [Network Connectivity Issues](#network-connectivity-issues)
- [Database Problems](#database-problems)
- [Sync Conflicts](#sync-conflicts)
- [Service Issues](#service-issues)
- [Performance Problems](#performance-problems)
- [Recovery Procedures](#recovery-procedures)

## Diagnostic Commands

Before troubleshooting specific issues, use these commands to gather information:

### System Status Check
```bash
# Overall sync system health
./sync/memory_sync.sh status

# Detailed system information
./sync/memory_sync.sh system-info

# Full diagnostic report
./sync/memory_sync.sh diagnose
```

### Component Testing
```bash
# Test individual components
./sync/memory_sync.sh test-connectivity    # Network tests
./sync/memory_sync.sh test-database       # Database integrity
./sync/memory_sync.sh test-sync           # Sync functionality
./sync/memory_sync.sh test-all            # Complete test suite
```

### Enable Debug Mode
```bash
# Enable verbose logging
export SYNC_DEBUG=1
export SYNC_VERBOSE=1

# Run commands with detailed output
./sync/memory_sync.sh sync
```

## Network Connectivity Issues

### Problem: Cannot Connect to Remote Server

**Symptoms:**
- Connection timeout errors
- "Remote server unreachable" messages
- Sync operations fail immediately

**Diagnostic Steps:**
```bash
# Test basic network connectivity
ping your-remote-server

# Test specific port
telnet your-remote-server 8443

# Test HTTP/HTTPS endpoint
curl -v -k https://your-remote-server:8443/api/health
```

**Solutions:**

#### DNS Resolution Issues
```bash
# Try with IP address instead of hostname
export REMOTE_MEMORY_HOST="your-server-ip"
./sync/memory_sync.sh status

# Add to /etc/hosts if DNS fails
echo "your-server-ip your-remote-server" | sudo tee -a /etc/hosts
```

#### Firewall/Port Issues
```bash
# Check if port is open
nmap -p 8443 your-remote-server

# Test alternative ports
export REMOTE_MEMORY_PORT="8000"  # Try HTTP port
export REMOTE_MEMORY_PROTOCOL="http"
```

#### SSL/TLS Certificate Issues
```bash
# Bypass SSL verification (testing only)
curl -k https://your-remote-server:8443/api/health

# Check certificate details
openssl s_client -connect your-remote-server:8443 -servername your-remote-server
```

### Problem: API Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- "Invalid API key" messages
- Authentication required warnings

**Solutions:**
```bash
# Check if API key is required
curl -k https://your-remote-server:8443/api/health

# Set API key if required
export REMOTE_MEMORY_API_KEY="your-api-key"

# Test with API key
curl -k -H "Authorization: Bearer your-api-key" \
  https://your-remote-server:8443/api/health
```

### Problem: Slow Network Performance

**Symptoms:**
- Sync operations taking too long
- Timeout errors during large syncs
- Network latency warnings

**Solutions:**
```bash
# Reduce batch size
export SYNC_BATCH_SIZE=25

# Increase timeout values
export SYNC_TIMEOUT=60
export SYNC_RETRY_ATTEMPTS=5

# Test network performance
./sync/memory_sync.sh benchmark-network
```

## Database Problems

### Problem: Staging Database Corruption

**Symptoms:**
- "Database is locked" errors
- SQLite integrity check failures
- Corrupt database warnings

**Diagnostic Steps:**
```bash
# Check database integrity
sqlite3 ~/.mcp_memory_staging/staging.db "PRAGMA integrity_check;"

# Check for database locks
lsof ~/.mcp_memory_staging/staging.db

# View database schema
sqlite3 ~/.mcp_memory_staging/staging.db ".schema"
```

**Recovery Procedures:**
```bash
# Backup current database
cp ~/.mcp_memory_staging/staging.db ~/.mcp_memory_staging/staging.db.backup

# Attempt repair
sqlite3 ~/.mcp_memory_staging/staging.db ".recover" > recovered.sql
rm ~/.mcp_memory_staging/staging.db
sqlite3 ~/.mcp_memory_staging/staging.db < recovered.sql

# If repair fails, reinitialize
rm ~/.mcp_memory_staging/staging.db
./sync/memory_sync.sh init
```

### Problem: Database Version Mismatch

**Symptoms:**
- Schema incompatibility errors
- "Database version not supported" messages
- Migration failures

**Solutions:**
```bash
# Check database version
sqlite3 ~/.mcp_memory_staging/staging.db "PRAGMA user_version;"

# Upgrade database schema
./sync/memory_sync.sh upgrade-db

# Force schema recreation
./sync/memory_sync.sh init --force-schema
```

### Problem: Insufficient Disk Space

**Symptoms:**
- "No space left on device" errors
- Database write failures
- Sync operations abort

**Solutions:**
```bash
# Check disk space
df -h ~/.mcp_memory_staging/

# Clean up old logs
find ~/.mcp_memory_staging/ -name "*.log.*" -mtime +30 -delete

# Compact databases
./sync/memory_sync.sh optimize
```

## Sync Conflicts

### Problem: Content Hash Conflicts

**Symptoms:**
- "Duplicate content detected" warnings
- Sync operations skip memories
- Hash mismatch errors

**Understanding:**
Content hash conflicts occur when the same memory content exists in both local staging and remote databases but with different metadata or timestamps.

**Resolution Strategies:**
```bash
# View conflict details
./sync/memory_sync.sh show-conflicts

# Auto-resolve using merge strategy
export SYNC_CONFLICT_RESOLUTION="merge"
./sync/memory_sync.sh sync

# Manual conflict resolution
./sync/memory_sync.sh resolve-conflicts --interactive
```

### Problem: Tag Conflicts

**Symptoms:**
- Memories with same content but different tags
- Tag merge warnings
- Inconsistent tag application

**Solutions:**
```bash
# Configure tag merging behavior
export TAG_MERGE_STRATEGY="union"  # union, intersection, local, remote

# Manual tag resolution
./sync/memory_sync.sh resolve-tags --memory-hash "abc123..."

# Bulk tag cleanup
./sync/memory_sync.sh cleanup-tags
```

### Problem: Timestamp Conflicts

**Symptoms:**
- Memories appear out of chronological order
- "Future timestamp" warnings
- Time synchronization issues

**Solutions:**
```bash
# Check system time synchronization
timedatectl status  # Linux
sntp -sS time.apple.com  # macOS

# Force timestamp update during sync
./sync/memory_sync.sh sync --update-timestamps

# Configure timestamp handling
export SYNC_TIMESTAMP_STRATEGY="newest"  # newest, oldest, local, remote
```

## Service Issues

### Problem: Service Won't Start

**Symptoms:**
- systemctl/launchctl start fails
- Service immediately exits
- "Service failed to start" errors

**Diagnostic Steps:**
```bash
# Check service status
./sync/memory_sync.sh status-service

# View service logs
./sync/memory_sync.sh logs

# Test service configuration
./sync/memory_sync.sh test-service-config
```

**Linux (systemd) Solutions:**
```bash
# Check service file
cat ~/.config/systemd/user/mcp-memory-sync.service

# Reload systemd
systemctl --user daemon-reload

# Check for permission issues
systemctl --user status mcp-memory-sync

# View detailed logs
journalctl --user -u mcp-memory-sync -n 50
```

**macOS (LaunchAgent) Solutions:**
```bash
# Check plist file
cat ~/Library/LaunchAgents/com.mcp.memory.sync.plist

# Unload and reload
launchctl unload ~/Library/LaunchAgents/com.mcp.memory.sync.plist
launchctl load ~/Library/LaunchAgents/com.mcp.memory.sync.plist

# Check logs
tail -f ~/Library/Logs/mcp-memory-sync.log
```

### Problem: Service Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- System becomes slow
- Out of memory errors

**Solutions:**
```bash
# Monitor memory usage
./sync/memory_sync.sh monitor-resources

# Restart service periodically
./sync/memory_sync.sh install-service --restart-interval daily

# Optimize memory usage
export SYNC_MEMORY_LIMIT="100MB"
./sync/memory_sync.sh restart-service
```

## Performance Problems

### Problem: Slow Sync Operations

**Symptoms:**
- Sync takes several minutes
- High CPU usage during sync
- Network timeouts

**Optimization Strategies:**
```bash
# Reduce batch size for large datasets
export SYNC_BATCH_SIZE=25

# Enable parallel processing
export SYNC_PARALLEL_JOBS=4

# Optimize database operations
./sync/memory_sync.sh optimize

# Profile sync performance
./sync/memory_sync.sh profile-sync
```

### Problem: High Resource Usage

**Symptoms:**
- High CPU usage
- Excessive disk I/O
- Memory consumption warnings

**Solutions:**
```bash
# Set resource limits
export SYNC_CPU_LIMIT=50      # Percentage
export SYNC_MEMORY_LIMIT=200  # MB
export SYNC_IO_PRIORITY=3     # Lower priority

# Use nice/ionice for background sync
nice -n 10 ionice -c 3 ./sync/memory_sync.sh sync

# Schedule sync during off-hours
crontab -e
# Change from: */15 * * * *
# To: 0 2,6,10,14,18,22 * * *
```

## Recovery Procedures

### Complete System Reset

If all else fails, perform a complete reset:

```bash
# 1. Stop all sync services
./sync/memory_sync.sh stop-service

# 2. Backup important data
cp -r ~/.mcp_memory_staging ~/.mcp_memory_staging.backup

# 3. Remove sync system
./sync/memory_sync.sh uninstall --remove-data

# 4. Reinstall from scratch
./sync/memory_sync.sh install

# 5. Restore configuration
./sync/memory_sync.sh init
```

### Disaster Recovery

For complete system failure:

```bash
# 1. Recover from Litestream backup (if configured)
litestream restore -o recovered_sqlite_vec.db /backup/path

# 2. Restore staging database from backup
cp ~/.mcp_memory_staging.backup/staging.db ~/.mcp_memory_staging/

# 3. Force sync from remote
./sync/memory_sync.sh pull --force

# 4. Verify data integrity
./sync/memory_sync.sh verify-integrity
```

### Data Migration

To migrate to a different server:

```bash
# 1. Export all local data
./sync/memory_sync.sh export --format json --output backup.json

# 2. Update configuration for new server
export REMOTE_MEMORY_HOST="new-server.local"

# 3. Import data to new server
./sync/memory_sync.sh import --input backup.json

# 4. Verify migration
./sync/memory_sync.sh status
```

## Logging and Monitoring

### Log File Locations

- **Sync logs**: `~/.mcp_memory_staging/sync.log`
- **Error logs**: `~/.mcp_memory_staging/error.log`
- **Service logs**: System-dependent (journalctl, Console.app, Event Viewer)
- **Debug logs**: `~/.mcp_memory_staging/debug.log` (when SYNC_DEBUG=1)

### Log Analysis

```bash
# View recent sync activity
tail -f ~/.mcp_memory_staging/sync.log

# Find sync errors
grep -i error ~/.mcp_memory_staging/sync.log | tail -10

# Analyze sync performance
grep "sync completed" ~/.mcp_memory_staging/sync.log | \
  awk '{print $(NF-1)}' | sort -n

# Count sync operations
grep -c "sync started" ~/.mcp_memory_staging/sync.log
```

### Monitoring Setup

Create monitoring scripts:

```bash
# Health check script
#!/bin/bash
if ! ./sync/memory_sync.sh status | grep -q "healthy"; then
  echo "Sync system unhealthy" | mail -s "MCP Sync Alert" admin@example.com
fi

# Performance monitoring
#!/bin/bash
SYNC_TIME=$(./sync/memory_sync.sh sync --dry-run 2>&1 | grep "would take" | awk '{print $3}')
if [ "$SYNC_TIME" -gt 300 ]; then
  echo "Sync taking too long: ${SYNC_TIME}s" | mail -s "MCP Sync Performance" admin@example.com
fi
```

## Getting Additional Help

### Support Information Generation

```bash
# Generate comprehensive support report
./sync/memory_sync.sh support-report > support_info.txt

# Include anonymized memory samples
./sync/memory_sync.sh support-report --include-samples >> support_info.txt
```

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check latest docs for updates
- **Wiki**: Community troubleshooting tips
- **Discussions**: Ask questions and share solutions

### Emergency Contacts

For critical production issues:
1. Check the GitHub issues for similar problems
2. Create a detailed bug report with support information
3. Tag the issue as "urgent" if it affects production systems
4. Include logs, configuration, and system information

Remember: The sync system is designed to be resilient. Most issues can be resolved by understanding the specific error messages and following the appropriate recovery procedures outlined in this guide.