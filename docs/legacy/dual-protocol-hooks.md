# Dual Protocol Memory Hooks (Legacy)

> **Note**: This feature has been superseded by Natural Memory Triggers v7.1.0+. This documentation is kept for reference only.

**Dual Protocol Memory Hooks** (v7.0.0+) provide intelligent memory awareness with automatic protocol detection:

## Configuration

```json
{
  "memoryService": {
    "protocol": "auto",
    "preferredProtocol": "mcp",
    "fallbackEnabled": true,
    "http": {
      "endpoint": "https://localhost:8443",
      "apiKey": "your-api-key",
      "healthCheckTimeout": 3000,
      "useDetailedHealthCheck": true
    },
    "mcp": {
      "serverCommand": ["uv", "run", "memory", "server", "-s", "cloudflare"],
      "serverWorkingDir": "/Users/yourname/path/to/mcp-memory-service",
      "connectionTimeout": 5000,
      "toolCallTimeout": 10000
    }
  }
}
```

## Protocol Options

- `"auto"`: Smart detection (MCP → HTTP → Environment fallback)
- `"http"`: HTTP-only mode (web server at localhost:8443)
- `"mcp"`: MCP-only mode (direct server process)

## Benefits

- **Reliability**: Multiple connection methods ensure hooks always work
- **Performance**: MCP direct for speed, HTTP for stability
- **Flexibility**: Works with local development or remote deployments
- **Compatibility**: Full backward compatibility with existing configurations

## Migration to Natural Memory Triggers

If you're using Dual Protocol Hooks, consider migrating to Natural Memory Triggers v7.1.0+ which offers:
- 85%+ trigger accuracy
- Multi-tier performance optimization
- CLI management system
- Git-aware context integration
- Adaptive learning

See main CLAUDE.md for migration instructions.
