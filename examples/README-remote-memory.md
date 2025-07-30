# Remote Memory Connection Guide

This guide explains how to connect Claude Desktop to a remote MCP Memory Service.

## Quick Setup

1. Copy the `claude-desktop-memory-config.json` file to your Claude Desktop configuration directory:

```bash
# On macOS:
cp claude-desktop-memory-config.json ~/Library/Application\ Support/claude-desktop/

# On Windows:
# cp claude-desktop-memory-config.json %APPDATA%\claude-desktop\

# On Linux:
# cp claude-desktop-memory-config.json ~/.config/claude-desktop/
```

2. Restart Claude Desktop to use the remote memory service

## Customizing the Configuration

You can edit the `claude-desktop-memory-config.json` file to change:

1. **Endpoint URL**: Update the `MCP_MEMORY_HTTP_ENDPOINT` value if your memory server is at a different location
2. **API Key**: Change the `MCP_MEMORY_API_KEY` if your server requires a different key
3. **Certificate Validation**: The `NODE_TLS_REJECT_UNAUTHORIZED` setting disables certificate validation for self-signed certificates

## Testing the Connection

Run the `simple-memory-test.js` script to verify your connection:

```bash
node examples/simple-memory-test.js
```

## Using Auto-Discovery

If your memory server advertises itself using mDNS (Bonjour/Zeroconf), you can use automatic discovery:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_PREFER_HTTPS": "true",
        "MCP_MEMORY_API_KEY": "your-api-key",
        "NODE_TLS_REJECT_UNAUTHORIZED": "0"
      }
    }
  }
}
```

## Troubleshooting

1. **Certificate Issues**: If you see TLS/SSL errors, make sure `NODE_TLS_REJECT_UNAUTHORIZED` is set to "0"
2. **Connection Refused**: Verify that the memory server is running and accessible (try pinging the hostname)
3. **Authentication Errors**: Check that your API key is correct
4. **Timeout Errors**: The server may be slow to respond - try increasing timeouts in the bridge code

## Memory Operations

The remote memory service supports these operations:

- `store_memory`: Save a new memory
- `retrieve_memory`: Search memories by semantic content
- `search_by_tag`: Find memories with specific tags
- `delete_memory`: Remove a specific memory
- `check_database_health`: Verify server health