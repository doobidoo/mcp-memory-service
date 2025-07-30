# Remote Memory Setup Guide

This comprehensive guide covers setting up remote memory connections between Claude Desktop and MCP Memory Service servers.

## Overview

The remote memory feature allows Claude Desktop to connect to a memory service running on a different machine, enabling shared memory across multiple clients or offloading memory processing to dedicated servers.

## Architecture

```
Claude Desktop → HTTP-to-MCP Bridge → Remote Memory Server
    (Client)         (Node.js)           (Python/HTTP API)
```

## Prerequisites

- Node.js installed on the client machine
- Remote memory server running with HTTP API enabled
- Network connectivity between client and server
- Valid API key (if authentication is enabled)

## Quick Setup

### 1. Configuration File

Copy the production configuration template:

```bash
# macOS
cp examples/claude-desktop-memory-config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
cp examples/claude-desktop-memory-config.json %APPDATA%\Claude\claude_desktop_config.json

# Linux
cp examples/claude-desktop-memory-config.json ~/.config/Claude/claude_desktop_config.json
```

### 2. Update Configuration

Edit the configuration file with your server details:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/full/path/to/mcp-memory-service/examples/http-mcp-bridge-robust.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "https://your-server.local/api",
        "MCP_MEMORY_API_KEY": "your-api-key",
        "NODE_TLS_REJECT_UNAUTHORIZED": "0"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Restart Claude Desktop to load the new configuration.

## Configuration Options

### Environment Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `MCP_MEMORY_HTTP_ENDPOINT` | Server API endpoint URL | `https://memory.local/api` | Yes |
| `MCP_MEMORY_API_KEY` | Authentication API key | `mcp-abc123...` | If auth enabled |
| `NODE_TLS_REJECT_UNAUTHORIZED` | Disable cert validation | `0` | For self-signed certs |

### Bridge Features

The production bridge (`http-mcp-bridge-robust.js`) includes:

- **Retry Logic**: Automatic retry with exponential backoff (max 3 attempts)
- **Timeout Handling**: Configurable timeouts (default: 60 seconds)
- **Health Checks**: Automatic endpoint health verification
- **Error Recovery**: Graceful error handling and reporting
- **Certificate Bypass**: Support for self-signed certificates

## Testing Your Connection

### 1. Manual API Test

Test the server API directly:

```bash
curl -k -H "Authorization: Bearer your-api-key" \
  https://your-server.local/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-30T18:45:43.565844",
  "uptime_seconds": 81374.24145579338,
  "storage": {
    "backend": "sqlite-vec",
    "status": "connected"
  }
}
```

### 2. Bridge Test

Test the bridge directly:

```bash
export MCP_MEMORY_HTTP_ENDPOINT="https://your-server.local/api"
export MCP_MEMORY_API_KEY="your-api-key"
export NODE_TLS_REJECT_UNAUTHORIZED="0"

echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  node examples/http-mcp-bridge-robust.js
```

### 3. Claude Desktop Logs

Monitor the Claude Desktop logs for connection status:

```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp-server-memory.log

# Windows
# Check %APPDATA%\Claude\Logs\mcp-server-memory.log

# Linux
tail -f ~/.config/Claude/logs/mcp-server-memory.log
```

## Memory Operations

### Available Tools

Once connected, the following memory tools become available in Claude Desktop:

1. **store_memory** - Save new information with optional tags
2. **retrieve_memory** - Semantic search for stored information
3. **search_by_tag** - Find memories by specific tags
4. **delete_memory** - Remove specific memories
5. **check_database_health** - Verify server health

### Known Limitations

- **Semantic Search Issue**: Current implementation may have embedding indexing delays
- **Workaround**: Use `search_by_tag` for reliable memory retrieval
- **Performance**: Network latency affects response times

## Troubleshooting

### Connection Issues

#### Certificate Errors
```
Error: self signed certificate
```
**Solution**: Set `NODE_TLS_REJECT_UNAUTHORIZED=0` in configuration

#### Network Timeout
```
Error: Request timeout after 60000ms
```
**Solution**: Check network connectivity and server performance

#### Authentication Errors
```
HTTP 401: Unauthorized
```
**Solution**: Verify API key is correct and properly formatted

### Performance Issues

#### Slow Response Times
- Check network latency between client and server
- Monitor server resource usage
- Consider local caching strategies

#### Memory Search Returns 0 Results
- Use `search_by_tag` instead of `retrieve_memory`
- Check if memories are properly stored with tags
- Verify embedding model is functioning on server

### Debugging Steps

1. **Verify API Connectivity**:
   ```bash
   curl -k https://your-server.local/api/health
   ```

2. **Check Bridge Logs**:
   - Look for "Robust MCP HTTP Bridge starting..." message
   - Verify endpoint and API key are loaded
   - Check for health check warnings

3. **Monitor Claude Desktop Logs**:
   - Confirm tools are registered successfully
   - Watch for tool call requests and responses
   - Check for JSON-RPC errors

4. **Test Individual Operations**:
   ```bash
   # Test storing a memory
   curl -k -H "Authorization: Bearer your-key" \
     -H "Content-Type: application/json" \
     -X POST https://server/api/memories \
     -d '{"content":"test memory","tags":["test"]}'
   
   # Test tag search
   curl -k -H "Authorization: Bearer your-key" \
     -H "Content-Type: application/json" \
     -X POST https://server/api/search/by-tag \
     -d '{"tags":["test"],"n_results":5}'
   ```

## Security Considerations

### Network Security
- Use HTTPS whenever possible
- Implement proper firewall rules
- Consider VPN for sensitive deployments

### Authentication
- Use strong, unique API keys
- Rotate API keys regularly
- Monitor access logs

### Certificate Management
- Use proper SSL certificates in production
- Only disable certificate validation for testing
- Implement certificate pinning for enhanced security

## Advanced Configuration

### Custom Bridge Configuration

You can modify the bridge behavior by editing `http-mcp-bridge-robust.js`:

```javascript
// Adjust timeout values
const DEFAULT_TIMEOUT = 60000; // 60 seconds

// Modify retry behavior
const MAX_RETRIES = 3;
const BASE_DELAY = 1000; // 1 second
```

### Multi-Server Setup

Configure multiple memory servers:

```json
{
  "mcpServers": {
    "memory-primary": {
      "command": "node",
      "args": ["/path/to/http-mcp-bridge-robust.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "https://primary.local/api",
        "MCP_MEMORY_API_KEY": "primary-key"
      }
    },
    "memory-backup": {
      "command": "node", 
      "args": ["/path/to/http-mcp-bridge-robust.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "https://backup.local/api",
        "MCP_MEMORY_API_KEY": "backup-key"
      }
    }
  }
}
```

## Support

For additional help:

1. Check the [main documentation](../README.md)
2. Review [troubleshooting guides](../troubleshooting/general.md)
3. Submit issues to the project repository
4. Join the community discussions

## Changelog

- **v3.1.0**: Initial remote memory bridge implementation
- **v3.2.0**: Enhanced error handling and retry logic (planned)