# Remote Server Configuration (Wiki Section)

This content can be added to the **03 Integration Guide** wiki page under the "1. Claude Desktop Integration" section.

---

## Remote Server Configuration

For users who want to connect Claude Desktop or Cursor to a remote MCP Memory Service instance (running on a VPS, server, or different machine), use the HTTP-to-MCP bridge included in the repository.

### Quick Setup

The MCP Memory Service includes a Node.js bridge that translates HTTP API calls to MCP protocol messages, allowing remote connections.

**Configuration for Claude Desktop:**

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_HTTP_ENDPOINT": "https://your-server:8000/api",
        "MCP_MEMORY_API_KEY": "your-secure-api-key"
      }
    }
  }
}
```

### Configuration Options

#### Manual Endpoint Configuration (Recommended for Remote Servers)
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_HTTP_ENDPOINT": "https://your-server:8000/api",
        "MCP_MEMORY_API_KEY": "your-secure-api-key",
        "MCP_MEMORY_AUTO_DISCOVER": "false",
        "MCP_MEMORY_PREFER_HTTPS": "true"
      }
    }
  }
}
```

#### Auto-Discovery (For Local Network)
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_PREFER_HTTPS": "true",
        "MCP_MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Step-by-Step Setup

1. **Download the HTTP Bridge**
   - Copy [`examples/http-mcp-bridge.js`](https://github.com/doobidoo/mcp-memory-service/blob/main/examples/http-mcp-bridge.js) to your local machine

2. **Update Configuration**
   - Open your Claude Desktop configuration file:
     - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
     - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - Add the remote server configuration (see examples above)
   - Replace `/path/to/mcp-memory-service/examples/http-mcp-bridge.js` with the actual path
   - Replace `https://your-server:8000/api` with your server's endpoint
   - Replace `your-secure-api-key` with your actual API key

3. **Verify Connection**
   - Restart Claude Desktop
   - Test the connection with a simple memory operation
   - Check the bridge logs for any connection issues

### Bridge Features

The HTTP-to-MCP bridge supports:

- ✅ **Manual endpoint configuration** - Direct connection to your remote server
- ✅ **API key authentication** - Secure access to your memory service
- ✅ **HTTPS with self-signed certificates** - Works with development SSL certificates
- ✅ **Automatic service discovery via mDNS** - Auto-detects local network services
- ✅ **Retry logic and error handling** - Robust connection management
- ✅ **Comprehensive logging** - Detailed logs for troubleshooting

### Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MCP_HTTP_ENDPOINT` | Remote server API endpoint (Streamable HTTP lives at `/mcp`; REST at `/api`) | `http://localhost:8000/api` | `https://myserver.com:8000/api` |
| `MCP_MEMORY_API_KEY` | Authentication token for client bridge (server uses `MCP_API_KEY`) | None | `abc123xyz789` |
| `MCP_MEMORY_AUTO_DISCOVER` | Enable mDNS service discovery | `false` | `true` |
| `MCP_MEMORY_PREFER_HTTPS` | Prefer HTTPS over HTTP when discovering | `true` | `false` |

### Troubleshooting Remote Connections

#### Connection Refused
- **Issue**: Bridge can't connect to the remote server
- **Solutions**:
  - Verify the server is running and accessible
  - Check firewall rules allow connections on port 8000
  - Confirm the endpoint URL is correct
  - Test with curl: `curl https://your-server:8000/api/health`

#### SSL Certificate Issues
- **Issue**: HTTPS connections fail with SSL errors
- **Solutions**:
  - The bridge automatically accepts self-signed certificates
  - Ensure your server is running with HTTPS enabled
  - Check server logs for SSL configuration issues

#### API Key Authentication Failed
- **Issue**: Server returns 401 Unauthorized
- **Solutions**:
  - Verify the API key is correctly set on the server
  - Check the key is properly configured in the bridge environment
  - Ensure no extra whitespace in the API key value

#### Service Discovery Not Working
- **Issue**: Auto-discovery can't find the service
- **Solutions**:
  - Use manual endpoint configuration instead
  - Ensure both devices are on the same network
  - Check if mDNS/Bonjour is enabled on your network

#### Bridge Logs Not Appearing
- **Issue**: Can't see bridge connection logs
- **Solutions**:
  - Bridge logs appear in Claude Desktop's console/stderr
  - On macOS, use Console.app to view logs
  - On Windows, check Event Viewer or run Claude Desktop from command line

### Complete Example Files

For complete working examples, see:
- [`examples/claude-desktop-http-config.json`](https://github.com/doobidoo/mcp-memory-service/blob/main/examples/claude-desktop-http-config.json) - Complete configuration template
- [`examples/http-mcp-bridge.js`](https://github.com/doobidoo/mcp-memory-service/blob/main/examples/http-mcp-bridge.js) - Full bridge implementation with documentation

---

*This section should be added to the existing "1. Claude Desktop Integration" section of the 03 Integration Guide wiki page, positioned after the basic local configuration examples.*
