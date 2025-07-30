# Check Memory Service Health via Direct API

I'll help you check the health and status of your remote MCP Memory Service using direct HTTP API calls. This provides comprehensive diagnostics without requiring any MCP protocol overhead.

## What I'll do:

1. **Direct Health Check**: I'll query the memory server health endpoints directly:
   - `/api/health` - Basic health status
   - `/api/health/detailed` - Comprehensive system information

2. **System Diagnostics**: I'll analyze and report:
   - Service uptime and availability
   - Storage backend status and configuration
   - System resources (memory, disk, CPU)
   - Database connectivity and performance
   - Embedding model status

3. **Performance Metrics**: I'll gather:
   - Response time measurements
   - Storage backend performance
   - Memory usage and optimization recommendations
   - Processing capabilities

4. **Connectivity Testing**: I'll verify:
   - Network connectivity to the server
   - API authentication status
   - SSL certificate validation
   - Endpoint availability

## Usage Examples:

```bash
claude /memory-api-health
claude /memory-api-health --detailed
claude /memory-api-health --server "https://memory.local/api"
claude /memory-api-health --test-endpoints
```

## API Implementation:

I'll execute these health checks:

**Basic Health:**
```bash
curl -k -H "Authorization: Bearer API_KEY" \\
  https://memory.local/api/health
```

**Detailed Health:**
```bash
curl -k -H "Authorization: Bearer API_KEY" \\
  https://memory.local/api/health/detailed
```

## Health Report Sections:

### 🟢 Service Status
- Overall health status (healthy/unhealthy)
- Service uptime and availability
- Last restart time and reason

### 🗄️ Storage Backend
- Backend type (SQLite-vec, ChromaDB)
- Database path and accessibility
- Embedding model (all-MiniLM-L6-v2)
- Connection status and performance

### 💾 System Resources
- Memory usage (total, available, percentage)
- Disk usage (total, free, percentage)
- CPU information and load
- Platform and version details

### ⚡ Performance Metrics
- API response times
- Database query performance
- Embedding processing capabilities
- Recent operation statistics

### 🔗 Connectivity
- Network endpoint accessibility
- SSL certificate status
- Authentication verification
- API endpoint availability

## Configuration:

- **Server Endpoint**: `https://memory.local/api` (default)
- **API Key**: `mcp-0b1ccbde2197a08dcb12d41af4044be6` (default)
- **SSL Verification**: Disabled for self-signed certificates
- **Timeout**: 10 seconds for health checks

## Arguments:

- `$ARGUMENTS` - Health check options:
  - `--detailed` - Request comprehensive system information
  - `--server "https://server/api"` - Override server endpoint
  - `--key "api-key"` - Override API key
  - `--test-endpoints` - Test all API endpoints for availability
  - `--benchmark` - Include performance benchmarking
  - `--no-auth` - Skip authentication (for public endpoints)

## Health Indicators:

### ✅ Healthy Status Indicators:
- HTTP 200 response from health endpoints
- "healthy" status in response body
- Reasonable response times (< 1 second)
- Storage backend "connected" status
- Available system resources

### ⚠️ Warning Indicators:
- High memory usage (> 80%)
- High disk usage (> 90%)
- Slow response times (> 2 seconds)
- Certificate expiration warnings

### ❌ Error Indicators:
- HTTP error responses (4xx, 5xx)
- "unhealthy" status in response
- Connection timeouts or failures
- Storage backend disconnected
- Authentication failures

## Troubleshooting Guidance:

Based on health check results, I'll provide:

**Connection Issues:**
- Network connectivity troubleshooting
- SSL certificate validation fixes
- API endpoint verification steps

**Performance Issues:**
- Resource optimization recommendations
- Database maintenance suggestions
- Configuration tuning advice

**Storage Issues:**
- Backend-specific diagnostics
- Database repair procedures
- Migration recommendations

## Sample Health Report:

```
🟢 Memory Service Health Check
Server: https://memory.local/api
Status: HEALTHY ✅
Uptime: 22.7 hours

Storage Backend: SQLite-vec
- Status: Connected ✅
- Model: all-MiniLM-L6-v2
- Database: Accessible ✅

System Resources:
- Memory: 71.8% used (2.6GB / 3.7GB)
- Disk: 84.3% used (24GB / 27GB) ⚠️
- Platform: Linux Ubuntu

Performance:
- Health check: 156ms
- API response: Fast ✅
```

## Error Handling:

If health checks fail, I'll:
- Provide exact error messages and HTTP status codes
- Suggest specific troubleshooting steps
- Show manual curl commands for direct testing
- Recommend alternative servers or configurations

This command gives you complete visibility into your remote memory service health without any MCP dependencies.