# Claude Code Commands for MCP Memory Service

This directory contains conversational Claude Code commands that integrate memory functionality into your Claude Code workflow. These commands follow the CCPlugins pattern of markdown-based conversational instructions.

## Available Commands

### MCP-Based Commands (Traditional)

### `/memory-store` - Store Current Context
Store information in your MCP Memory Service with proper context and tagging. Automatically detects project context and applies relevant tags.

**Usage:**
```bash
claude /memory-store "Important architectural decision about database backend"
claude /memory-store --tags "decision,architecture" "We chose SQLite-vec for performance"
```

### `/memory-recall` - Time-based Memory Retrieval
Retrieve memories using natural language time expressions. Perfect for finding past conversations and decisions.

**Usage:**
```bash
claude /memory-recall "what did we decide about the database last week?"
claude /memory-recall "yesterday's architectural discussions"
```

### `/memory-search` - Tag and Content Search
Search through stored memories using tags, content keywords, and semantic similarity.

**Usage:**
```bash
claude /memory-search --tags "architecture,database"
claude /memory-search "SQLite performance optimization"
```

### `/memory-context` - Session Context Integration
Capture the current conversation and project context as a memory for future reference.

**Usage:**
```bash
claude /memory-context
claude /memory-context --summary "Architecture planning session"
```

### `/memory-health` - Service Health Check
Check the health and status of your MCP Memory Service, providing diagnostics and statistics.

**Usage:**
```bash
claude /memory-health
claude /memory-health --detailed
```

### Direct API Commands (No MCP Required)

### `/memory-api-store` - Direct API Storage
Store memories directly via HTTP API, bypassing MCP entirely. Maximum reliability and simplicity.

**Usage:**
```bash
claude /memory-api-store "Remote memory bridge implementation completed"
claude /memory-api-store --tags "implementation,bridge" --server "https://memory.local/api"
```

### `/memory-api-search` - Direct API Search
Search memories using direct HTTP API calls. Uses reliable tag-based search as primary method.

**Usage:**
```bash
claude /memory-api-search "remote memory bridge"
claude /memory-api-search --tags "implementation,July 30 2025"
claude /memory-api-search --semantic "database optimization"
```

### `/memory-api-health` - Direct API Health Check
Comprehensive health diagnostics via direct API access. No MCP dependencies required.

**Usage:**
```bash
claude /memory-api-health
claude /memory-api-health --detailed --server "https://memory.local/api"
```

## Command Types Comparison

| Feature | MCP Commands | Direct API Commands |
|---------|--------------|-------------------|
| **Setup Complexity** | Requires MCP server + bridge | Just HTTP API endpoint |
| **Dependencies** | Node.js, MCP bridge, protocol | Only curl (built-in) |
| **Reliability** | Multiple layers, potential failures | Direct HTTP, fewer failure points |
| **Performance** | Protocol overhead, JSON-RPC | Direct REST API calls |
| **Debugging** | Complex (MCP logs, bridge logs) | Simple (HTTP status codes) |
| **Network** | MCP protocol over stdio/network | Standard HTTPS |
| **Authentication** | MCP protocol auth | Standard Bearer tokens |
| **Error Handling** | MCP error codes | HTTP status codes |
| **Compatibility** | Requires MCP client | Works anywhere |

## When to Use Each Type

### Use MCP Commands When:
- You have a local MCP Memory Service running
- You want full MCP protocol integration
- You need advanced features like auto-discovery
- You're already using other MCP services

### Use Direct API Commands When:
- Connecting to remote memory servers
- Maximum reliability is required
- You want to bypass MCP complexity  
- You need to debug connection issues
- Working with self-signed certificates
- Cross-platform compatibility is important

## Installation

### Automatic Installation (Recommended)

The commands can be installed automatically during the main MCP Memory Service installation:

```bash
# Install with commands (will prompt if Claude Code CLI is detected)
python install.py

# Force install commands
python install.py --install-claude-commands

# Skip command installation prompt
python install.py --skip-claude-commands-prompt
```

### Manual Installation

You can also install the commands manually:

```bash
# Install commands directly
python scripts/claude_commands_utils.py

# Test installation prerequisites
python scripts/claude_commands_utils.py --test

# Uninstall commands
python scripts/claude_commands_utils.py --uninstall
```

## Requirements

- **Claude Code CLI**: Must be installed and available in PATH
- **MCP Memory Service**: Should be installed and configured
- **File System Access**: Write access to `~/.claude/commands/` directory

## How It Works

1. **Command Files**: Each command is a markdown file with conversational instructions
2. **Claude Code Integration**: Commands are installed to `~/.claude/commands/`
3. **Auto-Discovery**: Commands automatically discover and connect to your MCP Memory Service
4. **Context Awareness**: Commands understand your current project and session context
5. **Fallback Handling**: Graceful degradation when the memory service is unavailable

## Command Features

- **Conversational Interface**: Natural language interactions following CCPlugins pattern
- **Context Detection**: Automatic project and session context recognition
- **Smart Tagging**: Intelligent tag generation based on current work
- **Error Recovery**: Helpful error messages and fallback suggestions
- **Backend Agnostic**: Works with both ChromaDB and SQLite-vec backends

## Example Workflow

```bash
# Start a development session
claude /memory-context --summary "Starting work on mDNS integration"

# Store important decisions
claude /memory-store --tags "mDNS,architecture" "Decided to use zeroconf library for service discovery"

# Continue development...

# Later, recall what was decided
claude /memory-recall "what did we decide about mDNS last week?"

# Search for related information
claude /memory-search --tags "mDNS,zeroconf"

# Check service health
claude /memory-health
```

## Troubleshooting

### Commands Not Available
- Ensure Claude Code CLI is installed: `claude --version`
- Check if commands are installed: `ls ~/.claude/commands/memory-*.md`
- Reinstall commands: `python scripts/claude_commands_utils.py`

### MCP Service Connection Issues
- Verify MCP Memory Service is running: `memory --help`
- Check service health: `claude /memory-health`
- Review service configuration in your Claude Code settings

### Permission Issues
- Check directory permissions: `ls -la ~/.claude/commands/`
- Ensure write access to the commands directory
- Try running installation with appropriate permissions

## Integration with MCP Memory Service

These commands seamlessly integrate with your MCP Memory Service installation:

- **Auto-Discovery**: Commands automatically find running memory services via mDNS
- **Backend Compatibility**: Works with both ChromaDB and SQLite-vec storage backends  
- **Configuration Aware**: Respects your memory service configuration settings
- **Health Monitoring**: Built-in service health checking and diagnostics

## Development

The commands are implemented using:

- **Markdown Format**: Conversational instructions in markdown files
- **Python Utilities**: Installation and management scripts in `scripts/claude_commands_utils.py`
- **Integration Logic**: Seamless installation via main `install.py` script
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

For more information about the MCP Memory Service, see the main project documentation.