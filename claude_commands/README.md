# Claude Code Commands for MCP Memory Service

This directory contains conversational Claude Code commands that integrate memory functionality into your Claude Code workflow. These commands follow the CCPlugins pattern of markdown-based conversational instructions.

## Available Commands

### `/memory-store` - Store Current Context
Store information in your MCP Memory Service with proper context and tagging. Automatically detects project context, applies relevant tags, and includes machine hostname for source tracking.

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
Capture the current conversation and project context as a memory for future reference. Automatically includes machine source identification for multi-device workflows.

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
3. **Service Connection**: Commands connect via mDNS discovery or direct HTTPS endpoints
4. **Context Awareness**: Commands understand your current project and session context
5. **API Integration**: Uses standard MCP Memory Service API endpoints for all operations

## Command Features

- **Machine Source Tracking**: Automatic hostname tagging for multi-device memory filtering
- **Conversational Interface**: Natural language interactions following CCPlugins pattern
- **Context Detection**: Automatic project and session context recognition
- **Smart Tagging**: Intelligent tag generation based on current work
- **Auto-Save**: Immediate storage without confirmation prompts
- **Flexible Configuration**: Supports both mDNS discovery and direct HTTPS endpoints
- **API Compatibility**: Works with standard MCP Memory Service API endpoints

## Example Workflow

```bash
# Start a development session (automatically tagged with machine hostname)
claude /memory-context --summary "Starting work on mDNS integration"

# Store important decisions (includes source:machine-name tag)
claude /memory-store --tags "mDNS,architecture" "Decided to use zeroconf library for service discovery"

# Continue development...

# Later, recall what was decided
claude /memory-recall "what did we decide about mDNS last week?"

# Search for related information
claude /memory-search --tags "mDNS,zeroconf"

# Search memories from specific machine
claude /memory-search --tags "source:laptop-work"

# Check service health
claude /memory-health
```

## Troubleshooting

### Commands Not Available
- Ensure Claude Code CLI is installed: `claude --version`
- Check if commands are installed: `ls ~/.claude/commands/memory-*.md`
- Reinstall commands: `python scripts/claude_commands_utils.py`

### Service Connection Issues
- **Service Health**: Check if service is accessible: `curl -k https://your-endpoint:8443/api/health`
- **Comprehensive Diagnostics**: Use `claude /memory-health` for detailed service status
- **Local Development**: Verify server can start: `python scripts/run_memory_server.py --help`
- **Endpoint Configuration**: Verify API endpoints match your service deployment

### Permission Issues
- Check directory permissions: `ls -la ~/.claude/commands/`
- Ensure write access to the commands directory
- Try running installation with appropriate permissions

## Configuration & API Compatibility

These commands integrate with your MCP Memory Service through configurable endpoints:

- **Flexible Connection**: Supports both mDNS auto-discovery and direct HTTPS endpoints
- **Configurable Endpoints**: Template uses `memory.local:8443` but can be customized to your setup
- **API Requirements**: Commands must use these standard API endpoints to function:
  - `POST /api/search` - Semantic similarity search
  - `POST /api/search/by-tag` - Tag-based search (AND/OR matching)
  - `POST /api/search/by-time` - Time-based natural language queries
  - `GET /api/memories` - List memories with pagination
  - `POST /api/memories` - Store new memories
  - `GET /api/health/detailed` - Service health diagnostics
- **Backend Compatibility**: Works with both ChromaDB and SQLite-vec storage backends
- **HTTPS Support**: Uses curl with `-k` flag for self-signed certificates

**Important**: Commands are configurable but must maintain compatibility with the documented API endpoints to function properly. Users can customize server locations but not the API contract.

## Development

The commands are implemented using:

- **Markdown Format**: Conversational instructions in markdown files
- **Python Utilities**: Installation and management scripts in `scripts/claude_commands_utils.py`
- **Integration Logic**: Seamless installation via main `install.py` script
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

For more information about the MCP Memory Service, see the main project documentation.
