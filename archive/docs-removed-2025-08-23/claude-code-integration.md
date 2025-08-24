# Using MCP Memory Service with Claude Code

This guide explains how to integrate the MCP Memory Service with Claude Code, providing two powerful approaches for using persistent memory capabilities in the Claude CLI environment.

## Prerequisites

Before you begin, ensure you have:

1. Installed [Claude Code](https://www.anthropic.com/news/introducing-claude-code) CLI tool
2. Set up the MCP Memory Service on your machine
3. Basic familiarity with command-line interfaces

## Integration Approaches

The MCP Memory Service offers **two integration methods** with Claude Code:

### 🎯 Method 1: Conversational Commands (Recommended) 
**New in v2.2.0** - Direct command syntax following the CCPlugins pattern

### 🔧 Method 2: MCP Server Registration
Traditional MCP server approach for deep integration

---

## Method 1: Conversational Commands (v2.2.0)

The **conversational commands approach** provides direct memory operations through Claude Code commands that follow the CCPlugins pattern. This is the **recommended approach** for most users as it provides immediate access to memory operations without MCP server configuration.

### Installation

The commands can be installed during the main MCP Memory Service installation:

```bash
# Install with commands (will prompt if Claude Code CLI is detected)
python install.py

# Force install commands without prompting
python install.py --install-claude-commands

# Skip command installation prompt
python install.py --skip-claude-commands-prompt
```

Or install them manually:

```bash
# Install commands directly
python scripts/claude_commands_utils.py

# Test installation prerequisites
python scripts/claude_commands_utils.py --test

# Uninstall commands
python scripts/claude_commands_utils.py --uninstall
```

### Available Commands

#### `/memory-store` - Store Information with Context
Store information in your memory service with automatic context detection and smart tagging.

```bash
claude /memory-store "Important architectural decision about database backend"
claude /memory-store --tags "decision,architecture" "We chose SQLite-vec for performance"
claude /memory-store --type "note" "Remember to update Docker configuration"
```

**Features:**
- Automatic project context detection from current directory
- Smart tag generation based on file types and git repository
- Session context integration
- Metadata enrichment with timestamps and paths

#### `/memory-recall` - Time-based Memory Retrieval
Retrieve memories using natural language time expressions.

```bash
claude /memory-recall "what did we decide about the database last week?"
claude /memory-recall "yesterday's architectural discussions"
claude /memory-recall --project "mcp-memory-service" "last month's progress"
```

**Features:**
- Natural language time parsing ("yesterday", "last Tuesday", "two months ago")
- Context-aware filtering based on current project
- Temporal relevance scoring
- Seasonal and event-based queries

#### `/memory-search` - Tag and Content Search
Search through stored memories using tags, content keywords, and semantic similarity.

```bash
claude /memory-search --tags "architecture,database"
claude /memory-search "SQLite performance optimization"
claude /memory-search --project "current" --type "decision"
```

**Features:**
- Tag-based filtering with partial matching
- Semantic content search using embeddings
- Combined query support (tags + content)
- Relevance scoring and ranking

#### `/memory-context` - Session Context Integration
Capture the current conversation and project context as a memory.

```bash
claude /memory-context
claude /memory-context --summary "Architecture planning session"
claude /memory-context --include-files --include-commits
```

**Features:**
- Automatic session analysis and summarization
- Git repository state capture
- File change detection and inclusion
- Key decision and insight extraction

#### `/memory-health` - Service Health and Diagnostics
Check the health and status of your MCP Memory Service.

```bash
claude /memory-health
claude /memory-health --detailed
claude /memory-health --test-operations
```

**Features:**
- Service connectivity verification
- Database health and statistics
- Performance metrics and diagnostics
- Auto-discovery testing and troubleshooting

### Command Features

- **Auto-Discovery**: Commands automatically locate your MCP Memory Service via mDNS
- **Context Awareness**: Understand current project, git repository, and session state
- **Error Recovery**: Graceful handling when memory service is unavailable
- **Cross-Platform**: Full support for Windows, macOS, and Linux
- **Backend Agnostic**: Works with both ChromaDB and SQLite-vec storage backends

### Example Workflow

```bash
# Start a development session
claude /memory-context --summary "Starting work on mDNS integration"

# Store important decisions during development
claude /memory-store --tags "mDNS,architecture" "Decided to use zeroconf library for service discovery"

# Continue development work...

# Later, recall what was decided
claude /memory-recall "what did we decide about mDNS last week?"

# Search for related technical information  
claude /memory-search --tags "mDNS,zeroconf"

# Check if everything is working correctly
claude /memory-health
```

---

## Method 2: MCP Server Registration

For users who prefer the traditional MCP server approach or need deeper integration, you can register the MCP Memory Service directly with Claude Code.

### Registering the Memory Service with Claude Code

You can register the MCP Memory Service to work with Claude Code using the `claude mcp add` command.

### Check Existing MCP Servers

To see which MCP servers are already registered with Claude:

```bash
claude mcp list
```

### Add the Memory Service

To add the memory service that's running on your local machine:

```bash
claude mcp add memory-service spawn -- /path/to/your/command
```

For example, if you've installed the memory service using UV (recommended):

```bash
claude mcp add memory-service spawn -- /opt/homebrew/bin/uv --directory /Users/yourusername/path/to/mcp-memory-service run memory
```

Replace the path elements with the actual paths on your system.

## Example Configuration

Here's a real-world example of adding the memory service to Claude Code:

```bash
claude mcp add memory-service spawn -- /opt/homebrew/bin/uv --directory /Users/yourusername/Documents/GitHub/mcp-memory-service run memory
```

This command:
1. Registers a new MCP server named "memory-service"
2. Uses the "spawn" transport method, which runs the command when needed
3. Specifies the full path to the UV command
4. Sets the working directory to your mcp-memory-service location
5. Runs the "memory" module

After running this command, you should see a confirmation message like:

```
Added stdio MCP server memory-service with command: spawn /opt/homebrew/bin/uv --directory /Users/yourusername/Documents/GitHub/mcp-memory-service run memory to local config
```

## Using Memory Functions in Claude Code

Once registered, you can use the memory service directly in your conversations with Claude Code. The memory functions available include:

- Storing memories
- Retrieving memories based on semantic search
- Recalling information from specific time periods
- Searching by tags
- And many more

---

## Troubleshooting

### For Conversational Commands (Method 1)

If you encounter issues with the commands:

1. **Commands Not Available**:
   - Verify Claude Code CLI is installed: `claude --version`
   - Check commands are installed: `ls ~/.claude/commands/memory-*.md`
   - Reinstall commands: `python scripts/claude_commands_utils.py`

2. **MCP Service Connection Issues**:
   - Verify MCP Memory Service is running: `memory --help`
   - Check service health: `claude /memory-health`
   - Ensure service is discoverable via mDNS

3. **Permission Issues**:
   - Check commands directory permissions: `ls -la ~/.claude/commands/`
   - Ensure write access to the commands directory

### For MCP Server Registration (Method 2)

If you encounter issues with MCP server registration:

1. Verify the memory service is running properly as a standalone application
2. Check that the paths in your `claude mcp add` command are correct
3. Ensure you have the necessary permissions to execute the specified commands
4. Try running `claude mcp list` to verify the server was added correctly

## Which Method Should You Use?

### Choose **Conversational Commands (Method 1)** if:
- ✅ You want quick setup with minimal configuration
- ✅ You prefer direct command syntax (`claude /memory-store`)
- ✅ You want automatic service discovery and context awareness
- ✅ You're new to MCP and want the simplest approach
- ✅ You want CCPlugins-compatible command integration

### Choose **MCP Server Registration (Method 2)** if:
- ✅ You need deep integration with Claude Code's MCP system
- ✅ You want to use the service alongside other MCP servers
- ✅ You prefer traditional MCP tool-based interactions
- ✅ You need maximum control over the server configuration
- ✅ You're building complex multi-server MCP workflows

## Benefits of Claude Code Integration

Both integration methods provide powerful advantages:

### Core Benefits
1. **Persistent Memory**: Your conversations and stored information persist across sessions
2. **Semantic Search**: Claude can retrieve relevant information even when not phrased exactly the same way
3. **Temporal Recall**: Ask about information from specific time periods (e.g., "last week", "yesterday")
4. **Organized Knowledge**: Use tags to categorize and later retrieve information by category
5. **Project Context**: Commands understand your current project and development context

### Method 1 (Commands) Additional Benefits
- **Immediate Access**: Direct command syntax without MCP server complexity
- **Context Integration**: Automatic project and git repository detection
- **Error Recovery**: Graceful fallback when service is unavailable
- **User-Friendly**: Conversational interface following established patterns

### Method 2 (MCP Server) Additional Benefits
- **Deep Integration**: Full MCP protocol support with rich tool interactions
- **Flexible Configuration**: Advanced server configuration options
- **Multi-Server Support**: Seamless integration with other MCP servers
- **Protocol Compliance**: Standard MCP tool-based interactions

This integration transforms Claude Code into a powerful knowledge management system with long-term memory capabilities, making your development workflow more efficient and context-aware.
