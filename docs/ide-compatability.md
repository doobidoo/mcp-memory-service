## IDE Compatibility

[![Works with Claude](https://img.shields.io/badge/Works%20with-Claude-blue)](https://claude.ai)
[![Works with Cursor](https://img.shields.io/badge/Works%20with-Cursor-orange)](https://cursor.sh)
[![Works with WindSurf](https://img.shields.io/badge/Works%20with-WindSurf-green)](https://codeium.com/windsurf)
[![Works with Cline](https://img.shields.io/badge/Works%20with-Cline-purple)](https://github.com/saoudrizwan/claude-dev)
[![Works with RooCode](https://img.shields.io/badge/Works%20with-RooCode-red)](https://roo.ai)

As of June 2025, MCP (Model Context Protocol) has become the standard for AI-IDE integrations. The MCP Memory Service is **fully compatible** with all major AI-powered development environments:

### Supported IDEs

| IDE | MCP Support | Configuration Location | Notes |
|-----|------------|----------------------|--------|
| **Claude Desktop** | ✅ Full | `claude_desktop_config.json` | Official MCP support |
| **Claude Code** | ✅ Full | CLI configuration | Official MCP support |
| **Cursor** | ✅ Full | `.cursor/mcp.json` or global config | Supports stdio, SSE, HTTP |
| **WindSurf** | ✅ Full | MCP config file | Built-in server management |
| **Cline** | ✅ Full | VS Code MCP config | Can create/share MCP servers |
| **RooCode** | ✅ Full | IDE config | Full MCP client implementation |
| **VS Code** | ✅ Full | `.vscode/mcp.json` | Via MCP extension |
| **Zed** | ✅ Full | Built-in config | Native MCP support |

### Quick Setup for Popular IDEs

#### Cursor
Add to `.cursor/mcp.json` in your project or global Cursor config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-memory-service",
        "run",
        "memory"
      ],
      "env": {
        "MCP_MEMORY_CHROMA_PATH": "/path/to/chroma_db",
        "MCP_MEMORY_BACKUPS_PATH": "/path/to/backups"
      }
    }
  }
}
```

#### WindSurf
WindSurf offers the easiest setup with built-in server management. Add to your WindSurf MCP configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_CHROMA_PATH": "/path/to/chroma_db",
        "MCP_MEMORY_BACKUPS_PATH": "/path/to/backups"
      }
    }
  }
}
```

#### Cline (VS Code)
1. Open the Cline extension in VS Code
2. Click the MCP Servers icon
3. Click "Configure MCP Servers"
4. Add the memory service configuration (same format as above)

#### RooCode
RooCode uses a similar configuration format. Refer to RooCode's MCP documentation for the exact config file location.

### Working with Multiple MCP Servers

MCP servers are designed to be composable. You can use the Memory Service alongside other popular MCP servers:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-github-token"
      }
    },
    "task-master": {
      "command": "npx",
      "args": ["-y", "task-master-mcp"]
    }
  }
}
```

### Alternative Installation Methods

#### Using NPX (if published to npm)
```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@doobidoo/mcp-memory-service"]
    }
  }
}
```

#### Using Python directly
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/memory_wrapper.py"]
    }
  }
}
```

### Why Choose MCP Memory Service?

Unlike IDE-specific memory solutions, MCP Memory Service offers:

- **Cross-IDE Compatibility**: Your memories work across ALL supported IDEs
- **Persistent Storage**: Memories survive IDE restarts and updates
- **Semantic Search**: Find memories by meaning, not just keywords
- **Natural Language Time Queries**: "What did I work on last week?"
- **Tag-based Organization**: Organize memories with flexible tagging
- **Cross-Platform**: Works on macOS, Windows, and Linux

### Troubleshooting IDE Connections

If the memory service isn't connecting in your IDE:

1. **Verify Installation**: Run `python scripts/test_installation.py`
2. **Check Logs**: Most IDEs show MCP server logs in their output panels
3. **Test Standalone**: Try running the server directly: `uv run memory`
4. **Path Issues**: Use absolute paths in your configuration
5. **Python Environment**: Ensure the IDE can access your Python environment

### IDE-Specific Tips

**Cursor**: If using multiple MCP servers, be aware of Cursor's server limit. Prioritize based on your needs.

**WindSurf**: Take advantage of WindSurf's built-in server management UI for easier configuration.

**Cline**: Cline can display MCP server status - check for green indicators after configuration.

**VS Code with MCP Extension**: Install the official MCP extension from the marketplace for better integration.
