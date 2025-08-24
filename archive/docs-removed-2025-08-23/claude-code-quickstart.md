# Claude Code Commands - Quick Start Guide

Get up and running with MCP Memory Service Claude Code commands in just 2 minutes!

## Prerequisites

✅ [Claude Code CLI](https://claude.ai/code) installed and working  
✅ Python 3.10+ with pip  
✅ 5 minutes of your time  

## Step 1: Install MCP Memory Service with Commands

```bash
# Clone and install with Claude Code commands
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python install.py --install-claude-commands
```

The installer will:
- ✅ Detect your Claude Code CLI automatically
- ✅ Install the memory service with optimal settings for your system
- ✅ Install 5 conversational memory commands
- ✅ Test everything to ensure it works

## Step 2: Test Your Installation

```bash
# Check if everything is working
claude /memory-health
```

You should see a comprehensive health check interface. If you see the command description and interface, you're all set! 🎉

## Step 3: Store Your First Memory

```bash
# Store something important
claude /memory-store "I successfully set up MCP Memory Service with Claude Code commands on $(date)"
```

## Step 4: Try the Core Commands

```bash
# Recall memories by time
claude /memory-recall "what did I store today?"

# Search by content
claude /memory-search "MCP Memory Service"

# Capture current session context
claude /memory-context --summary "Initial setup and testing"
```

## 🎯 You're Done!

That's it! You now have powerful memory capabilities integrated directly into Claude Code. 

## Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `claude /memory-store` | Store information with context | `claude /memory-store "Important decision about architecture"` |
| `claude /memory-recall` | Retrieve by time expressions | `claude /memory-recall "what did we decide last week?"` |
| `claude /memory-search` | Search by tags or content | `claude /memory-search --tags "architecture,database"` |
| `claude /memory-context` | Capture session context | `claude /memory-context --summary "Planning session"` |
| `claude /memory-health` | Check service status | `claude /memory-health --detailed` |

## Next Steps

### Explore Advanced Features
- **Context-aware operations**: Commands automatically detect your current project
- **Smart tagging**: Automatic tag generation based on your work
- **Time-based queries**: Natural language like "yesterday", "last week", "two months ago"
- **Semantic search**: Find related information even with different wording

### Learn More
- 📖 [**Full Integration Guide**](claude-code-integration.md) - Complete documentation
- 🔧 [**Installation Master Guide**](../installation/master-guide.md) - Advanced installation options
- ❓ [**Troubleshooting**](../troubleshooting/general.md) - Solutions to common issues

## Troubleshooting Quick Fixes

### Commands Not Working?
```bash
# Check if Claude Code CLI is working
claude --version

# Check if commands are installed
ls ~/.claude/commands/memory-*.md

# Reinstall commands
python scripts/claude_commands_utils.py
```

### Memory Service Not Connecting?
```bash
# Check if service is running
memory --help

# Check service health
claude /memory-health

# Start the service if needed
memory
```

### Need Help?
- 💬 [GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues)
- 📚 [Full Documentation](../README.md)
- 🔍 [Search Existing Solutions](https://github.com/doobidoo/mcp-memory-service/issues?q=is%3Aissue)

---

## What Makes This Special?

🚀 **Zero Configuration**: No MCP server setup required  
🧠 **Context Intelligence**: Understands your current project and session  
💬 **Conversational Interface**: Natural, CCPlugins-compatible commands  
⚡ **Instant Access**: Direct command-line memory operations  
🛠️ **Professional Grade**: Enterprise-level capabilities through simple commands  

**Enjoy your enhanced Claude Code experience with persistent memory!** 🎉