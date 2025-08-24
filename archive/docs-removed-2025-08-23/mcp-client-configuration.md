# MCP Client Configuration Guide

## Overview

This guide provides complete configuration examples for connecting various IDEs and MCP clients to the remote MCP Memory Service. With v4.0.1, the service provides native MCP-over-HTTP protocol support, enabling seamless integration across devices and platforms.

## Server Information

- **Endpoint**: `http://your-server:8000/mcp`
- **Protocol**: JSON-RPC 2.0 over HTTP/HTTPS
- **Authentication**: Bearer token (API key)
- **Version**: 4.0.1+

## IDE Configurations

### Claude Code (Desktop)

**Configuration File Location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "memory": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "http://your-server:8000/mcp",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer YOUR_API_KEY",
        "-d", "@-"
      ],
      "env": {
        "MCP_SERVER_NAME": "memory-service",
        "MCP_SERVER_VERSION": "4.0.1"
      }
    }
  }
}
```

### Direct HTTP MCP Connection

For IDEs with native HTTP MCP support:

```json
{
  "mcpServers": {
    "memory": {
      "transport": "http",
      "endpoint": "http://your-server:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
      },
      "protocol": "jsonrpc-2.0"
    }
  }
}
```

### VS Code with MCP Extension

```json
{
  "mcp.servers": {
    "memory-service": {
      "url": "http://your-server:8000/mcp",
      "authentication": {
        "type": "bearer",
        "token": "YOUR_API_KEY"
      },
      "tools": [
        "store_memory",
        "retrieve_memory", 
        "search_by_tag",
        "delete_memory",
        "check_database_health"
      ]
    }
  }
}
```

## Programming Language Clients

### Python MCP Client

```python
import asyncio
import aiohttp
import json

class MCPMemoryClient:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def request(self, method, params=None):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def store_memory(self, content, tags=None, memory_type=None):
        params = {
            "name": "store_memory",
            "arguments": {
                "content": content,
                "tags": tags or [],
                "memory_type": memory_type
            }
        }
        return await self.request("tools/call", params)
    
    async def retrieve_memory(self, query, limit=10):
        params = {
            "name": "retrieve_memory", 
            "arguments": {
                "query": query,
                "limit": limit
            }
        }
        return await self.request("tools/call", params)

# Usage
client = MCPMemoryClient("http://your-server:8000/mcp", "YOUR_API_KEY")
result = await client.store_memory("Important project decision", ["decisions", "project"])
```

### Node.js MCP Client

```javascript
const axios = require('axios');

class MCPMemoryClient {
  constructor(endpoint, apiKey) {
    this.endpoint = endpoint;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async request(method, params = {}) {
    const payload = {
      jsonrpc: "2.0",
      id: 1,
      method: method,
      params: params
    };

    const response = await axios.post(this.endpoint, payload, {
      headers: this.headers
    });

    return response.data;
  }

  async storeMemory(content, tags = [], memoryType = null) {
    const params = {
      name: "store_memory",
      arguments: {
        content: content,
        tags: tags,
        memory_type: memoryType
      }
    };

    return await this.request("tools/call", params);
  }

  async retrieveMemory(query, limit = 10) {
    const params = {
      name: "retrieve_memory",
      arguments: {
        query: query,
        limit: limit
      }
    };

    return await this.request("tools/call", params);
  }

  async listTools() {
    return await this.request("tools/list");
  }
}

// Usage
const client = new MCPMemoryClient("http://your-server:8000/mcp", "YOUR_API_KEY");
const result = await client.storeMemory("Meeting notes", ["meetings", "important"]);
```

### cURL Examples

**List Available Tools**:
```bash
curl -X POST http://your-server:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'
```

**Store Memory**:
```bash
curl -X POST http://your-server:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "store_memory",
      "arguments": {
        "content": "Important project decision about architecture",
        "tags": ["decisions", "architecture", "project"],
        "memory_type": "decision"
      }
    }
  }'
```

**Retrieve Memory**:
```bash
curl -X POST http://your-server:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "retrieve_memory",
      "arguments": {
        "query": "architecture decisions",
        "limit": 5
      }
    }
  }'
```

## Available MCP Tools

The MCP Memory Service provides these tools:

### 1. store_memory
**Description**: Store new memories with tags and metadata
**Parameters**:
- `content` (string, required): The memory content
- `tags` (array[string], optional): Tags for categorization
- `memory_type` (string, optional): Type of memory (e.g., "note", "decision")

### 2. retrieve_memory
**Description**: Semantic search and retrieval of memories
**Parameters**:
- `query` (string, required): Search query
- `limit` (integer, optional): Maximum results to return (default: 10)

### 3. search_by_tag
**Description**: Search memories by specific tags
**Parameters**:
- `tags` (array[string], required): Tags to search for
- `operation` (string, optional): "AND" or "OR" logic (default: "AND")

### 4. delete_memory
**Description**: Delete a specific memory by content hash
**Parameters**:
- `content_hash` (string, required): Hash of the memory to delete

### 5. check_database_health
**Description**: Check the health and status of the memory database
**Parameters**: None

## Security Configuration

### API Key Setup

Generate a secure API key:
```bash
# Generate a secure API key
export MCP_API_KEY="$(openssl rand -base64 32)"
echo "Your API Key: $MCP_API_KEY"
```

Set the API key on your server:
```bash
# On the server
export MCP_API_KEY="your-secure-api-key"
python scripts/run_http_server.py
```

### HTTPS Setup (Production)

For production deployments, use HTTPS:

1. **Generate SSL certificates** (or use Let's Encrypt)
2. **Configure HTTPS** in the server
3. **Update client endpoints** to use `https://`

Example with HTTPS:
```json
{
  "mcpServers": {
    "memory": {
      "transport": "http",
      "endpoint": "https://your-domain.com:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

## Troubleshooting

### Connection Issues

1. **Check server status**:
   ```bash
   curl -s http://your-server:8000/api/health
   ```

2. **Verify MCP endpoint**:
   ```bash
   curl -X POST http://your-server:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
   ```

3. **Check authentication**:
   - Ensure API key is correctly set
   - Verify Authorization header format: `Bearer YOUR_API_KEY`

### Common Errors

- **404 Not Found**: Check endpoint URL and server status
- **401 Unauthorized**: Verify API key and Authorization header
- **422 Validation Error**: Check JSON-RPC payload format
- **500 Internal Error**: Check server logs for embedding model issues

### Network Configuration

- **Firewall**: Ensure port 8000 is accessible
- **CORS**: Server includes CORS headers for web clients
- **DNS**: Use IP address if hostname resolution fails

## Best Practices

1. **Use HTTPS** in production environments
2. **Secure API keys** with proper rotation
3. **Implement retries** for network failures
4. **Cache tool lists** to reduce overhead
5. **Use appropriate timeouts** for requests
6. **Monitor server health** regularly

## Advanced Configuration

### Load Balancing

For high-availability deployments:

```json
{
  "mcpServers": {
    "memory": {
      "endpoints": [
        "https://memory1.yourdomain.com:8000/mcp",
        "https://memory2.yourdomain.com:8000/mcp"
      ],
      "loadBalancing": "round-robin",
      "failover": true
    }
  }
}
```

### Custom Headers

Add custom headers for monitoring or routing:

```json
{
  "headers": {
    "Authorization": "Bearer YOUR_API_KEY",
    "X-Client-ID": "claude-desktop-v1.0",
    "X-Session-ID": "unique-session-identifier"
  }
}
```

---

## Summary

The MCP Memory Service v4.0.1 provides a robust, production-ready remote memory solution with native MCP protocol support. This guide ensures seamless integration across various IDEs and programming environments, enabling powerful semantic memory capabilities in your development workflow.

For additional support, refer to the main documentation or check the GitHub repository for updates and examples.