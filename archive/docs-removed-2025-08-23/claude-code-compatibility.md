# Claude Code Compatibility Guide

## Overview

The MCP Memory Service FastAPI server v4.0.0-alpha.1 implements the official MCP protocol but has specific compatibility considerations with Claude Code's SSE client implementation.

## Current Status

### ✅ Working MCP Clients
- **Standard MCP Libraries**: Python `mcp` package, JavaScript MCP SDK
- **Claude Desktop**: Works with proper MCP configuration
- **Custom MCP Clients**: Any client implementing standard MCP protocol
- **HTTP API**: Full REST API access via port 8080

### ❌ Claude Code SSE Client Issue

**Problem**: Claude Code's SSE client has specific header and protocol requirements that don't match the FastMCP server implementation.

**Technical Details**:
- FastMCP server requires `Accept: application/json, text/event-stream` headers
- Claude Code's SSE client doesn't send the required header combination
- Server correctly rejects invalid SSE connections with proper error messages

**Error Symptoms**:
```bash
claude mcp list
# Output: memory: http://10.0.1.30:8000/mcp (SSE) - ✗ Failed to connect
```

## Workarounds for Claude Code Users

### Option 1: Use HTTP Dashboard
```bash
# Access memory service via web interface
open http://memory.local:8080/

# Use API endpoints directly
curl -X POST http://memory.local:8080/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "My memory", "tags": ["important"]}'
```

### Option 2: Use Claude Commands (Recommended)
```bash
# Install Claude Code commands (bypass MCP entirely)
python install.py --install-claude-commands

# Use conversational memory commands
claude /memory-store "Important information"
claude /memory-recall "what did we discuss?"
claude /memory-search --tags "project,architecture"
```

### Option 3: Use Alternative MCP Client
```python
# Python example with standard MCP client
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def test_memory():
    # This works with standard MCP protocol
    # Implementation details for your specific needs
    pass
```

## Technical Investigation Results

### Server Verification ✅
```bash
# Server correctly implements MCP protocol
curl -H "Accept: text/event-stream, application/json" \
     -H "Content-Type: application/json" \
     -X POST http://memory.local:8000/mcp \
     -d '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}'
# Result: 200 OK, SSE stream established
```

### Claude Code Client Issue ❌
```bash
# Claude Code client fails header negotiation
# Missing required Accept header combination
# Connection rejected with 406 Not Acceptable
```

## Development Roadmap

### Short Term (Next Release)
- [ ] Investigate Claude Code's exact SSE client requirements
- [ ] Consider server-side compatibility layer
- [ ] Expand client compatibility testing

### Medium Term 
- [ ] Custom SSE implementation for Claude Code compatibility
- [ ] Alternative transport protocols (WebSocket, HTTP long-polling)
- [ ] Client library development

### Long Term
- [ ] Collaborate with Claude Code team on SSE standardization
- [ ] MCP protocol enhancement proposals
- [ ] Universal client compatibility layer

## Conclusion

The FastAPI MCP migration successfully achieved its primary goals:
- ✅ SSL connectivity issues resolved
- ✅ Standard MCP protocol compliance
- ✅ Production-ready architecture

The Claude Code compatibility issue is a client-specific limitation that doesn't impact the core migration success. Users have multiple viable workarounds available.

## Support

- **HTTP Dashboard**: http://memory.local:8080/
- **Documentation**: See `DUAL_SERVICE_DEPLOYMENT.md`
- **Issues**: Report at https://github.com/doobidoo/mcp-memory-service/issues
- **Claude Commands**: See `docs/guides/claude-code-quickstart.md`