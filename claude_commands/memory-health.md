# Check Memory Service Health

I'll help you check the health and status of your MCP Memory Service, providing detailed diagnostics and statistics about your memory storage and service connectivity.

## What I'll do:

1. **Service Discovery**: I'll locate your MCP Memory Service using mDNS auto-discovery or configured endpoints.

2. **Connectivity Test**: I'll verify that the service is running and accessible from Claude Code.

3. **Health Assessment**: I'll check the database health, memory statistics, and service performance.

4. **Storage Analysis**: I'll provide insights about your stored memories, database size, and usage patterns.

5. **Troubleshooting**: If issues are found, I'll provide specific recommendations for resolution.

## Usage Examples:

```bash
claude /memory-health
claude /memory-health --detailed
claude /memory-health --test-operations
claude /memory-health --export-report
```

## Implementation:

I'll perform comprehensive health checks using your MCP Memory Service at `https://memory.local:8443/`:

1. **Service Detection**:
   - Connect to configured HTTPS endpoint
   - Check health endpoint at `/api/health/detailed`
   - Verify HTTPS connectivity with `-k` flag for self-signed certificates

2. **Basic Health Check**:
   - Service responsiveness via `/api/health` endpoint
   - Detailed diagnostics via `/api/health/detailed`
   - Database connection status and embedding model availability

3. **Detailed Diagnostics**:
   - Memory count and database statistics from API response
   - Storage backend type (SQLite-vec, ChromaDB, etc.)
   - Performance metrics, uptime, and response times
   - Disk usage warnings and system resource status

4. **Operational Testing** (if requested):
   - Test memory storage and retrieval
   - Verify search functionality
   - Check tag operations

## Health Report Includes:

### Service Status
- **Running**: Whether the MCP Memory Service is active
- **Endpoint**: Service URL and port information
- **Response Time**: Average API response latency
- **Version**: Service version and backend type
- **Uptime**: How long the service has been running

### Database Health
- **Backend Type**: ChromaDB, SQLite-vec, or other storage backend
- **Connection Status**: Database connectivity and integrity
- **Total Memories**: Number of stored memory entries
- **Database Size**: Storage space used by the database
- **Last Backup**: When backups were last created (if applicable)

### Performance Metrics
- **Query Performance**: Average search and retrieval times
- **Embedding Model**: Model type and loading status
- **Memory Usage**: Service memory consumption
- **Cache Status**: Embedding and model cache statistics

### Storage Statistics
- **Popular Tags**: Most frequently used tags
- **Memory Types**: Distribution of memory types (note, decision, etc.)
- **Recent Activity**: Recent storage and retrieval operations
- **Growth Trends**: Database growth over time

## Troubleshooting Features:

### Common Issues I'll Check:
- **Service Not Running**: Instructions to start the MCP Memory Service
- **Port Conflicts**: Detection of port usage conflicts
- **Database Corruption**: Database integrity verification
- **Permission Issues**: File system access and write permissions
- **Model Loading**: Embedding model download and loading status

### Auto-Fix Capabilities:
- **Restart Service**: Attempt to restart a hung service
- **Clear Cache**: Clean corrupted cache files
- **Repair Database**: Basic database repair operations
- **Update Configuration**: Fix common configuration issues

## Arguments:

- `$ARGUMENTS` - Optional specific health check focus
- `--detailed` - Show comprehensive diagnostics and statistics
- `--test-operations` - Perform functional tests of store/retrieve operations
- `--check-backups` - Verify backup system health and recent backups
- `--performance-test` - Run performance benchmarks
- `--export-report` - Save health report to file for sharing
- `--fix-issues` - Attempt automatic fixes for detected problems
- `--quiet` - Show only critical issues (minimal output)

## Advanced Diagnostics:

- **Network Connectivity**: Test mDNS discovery and HTTP endpoints
- **Database Integrity**: Verify database consistency and repair if needed
- **Model Health**: Check embedding model files and performance
- **Configuration Validation**: Verify environment variables and settings
- **Resource Usage**: Monitor CPU, memory, and disk usage

If critical issues are detected, I'll provide step-by-step resolution instructions and can attempt automatic fixes with your permission. For complex issues, I'll recommend specific diagnostic commands or log file locations to investigate further.

The health check helps ensure your memory service is running optimally and can identify potential issues before they impact your workflow.
