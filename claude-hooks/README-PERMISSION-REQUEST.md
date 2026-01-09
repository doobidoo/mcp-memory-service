# Universal Permission Request Hook for Claude Code

Automatically manages MCP tool permission requests, reducing repetitive dialogs by intelligently auto-approving safe, read-only operations while blocking potentially destructive actions.

## Overview

This hook intercepts Claude Code permission requests and makes intelligent decisions based on tool naming patterns:

- **Auto-approves** read-only operations (get, list, retrieve, search, etc.)
- **Blocks** potentially destructive actions (delete, remove, update, execute, etc.)
- **Works universally** across all MCP servers (memory, browser, context7, etc.)
- **Safe-by-default** - unknown patterns require user confirmation

## Features

✅ **Smart Pattern Matching** - Analyzes tool names to determine safety
✅ **Universal Support** - Works with any MCP server (strips `mcp__server__` prefixes)
✅ **Safe Defaults** - Unknown operations always prompt for confirmation
✅ **Zero Configuration** - Works out of the box with sensible defaults
✅ **Extensible** - Easy to add custom patterns via configuration

## Installation

### Via Unified Installer (Recommended)

```bash
cd claude-hooks
python install_hooks.py  # Installs all hooks including permission-request
```

### Manual Installation

```bash
# Copy hook to Claude Code directory
cp claude-hooks/core/permission-request.js ~/.claude/hooks/core/

# Make executable (Unix/macOS)
chmod +x ~/.claude/hooks/core/permission-request.js

# Update Claude Code settings (automatic if using installer)
```

The hook will be automatically configured in your Claude Code settings by the installer.

## How It Works

### Decision Flow

```
Permission Request
    ↓
Extract Tool Name (strip mcp__server__ prefix)
    ↓
Check for Destructive Patterns?
    ↓ YES → Prompt User
    ↓ NO
Check for Safe Patterns?
    ↓ YES → Auto-Approve
    ↓ NO → Prompt User (unknown pattern)
```

### Pattern Categories

#### Safe Patterns (Auto-Approved)
```javascript
get, list, read, retrieve, fetch, search, find, query,
recall, check, status, health, stats, analyze, view,
show, describe, inspect
```

**Examples:**
- `mcp__memory__retrieve_memory` → Auto-approved
- `mcp__browser__get_screenshot` → Auto-approved
- `mcp__Context7__query-docs` → Auto-approved

#### Destructive Patterns (Require Confirmation)
```javascript
delete, remove, destroy, drop, clear, wipe, purge,
forget, erase, reset, update, modify, edit, change,
write, create, deploy, publish, execute, run, eval,
consolidate
```

**Examples:**
- `mcp__memory__delete_memory` → User prompt
- `mcp__memory__update_memory_metadata` → User prompt
- `mcp__memory__consolidate_memories` → User prompt

## Configuration

### Basic Configuration (config.json)

```json
{
  "permissionRequest": {
    "enabled": true,
    "autoApprove": true,
    "customSafePatterns": [],
    "customDestructivePatterns": [],
    "logDecisions": false
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | Boolean | `true` | Enable/disable the permission hook |
| `autoApprove` | Boolean | `true` | Auto-approve safe patterns |
| `customSafePatterns` | Array | `[]` | Additional safe patterns to auto-approve |
| `customDestructivePatterns` | Array | `[]` | Additional patterns to block |
| `logDecisions` | Boolean | `false` | Log decisions to console for debugging |

### Adding Custom Patterns

To add organization-specific patterns:

```json
{
  "permissionRequest": {
    "customSafePatterns": ["audit", "report", "export"],
    "customDestructivePatterns": ["archive", "migrate"]
  }
}
```

## Usage Examples

### Typical Session Flow

```
User: "Retrieve my recent memories"
→ Claude calls mcp__memory__retrieve_memory
→ Hook: Auto-approved (safe pattern: retrieve)
→ Tool executes immediately

User: "Delete old memories"
→ Claude calls mcp__memory__delete_by_timeframe
→ Hook: User prompt (destructive pattern: delete)
→ User confirms/denies
```

### With Multiple MCP Servers

```
User: "Search the codebase and recall relevant memories"
→ Claude calls:
  1. mcp__code-context__search_code (auto-approved)
  2. mcp__memory__recall_memory (auto-approved)
→ Both execute without user prompts
```

## Debugging

### Enable Debug Logging

```json
{
  "permissionRequest": {
    "logDecisions": true
  }
}
```

Output format:
```
[PermissionRequest Hook] Tool: retrieve_memory, Server: memory, Decision: allow
[PermissionRequest Hook] Tool: delete_memory, Server: memory, Decision: prompt
```

### Testing Hook Behavior

```bash
# Test with sample payload
echo '{"tool_name": "mcp__memory__retrieve_memory", "server_name": "memory"}' | \
  node ~/.claude/hooks/core/permission-request.js
```

Expected output:
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PermissionRequest",
    "decision": {
      "behavior": "allow"
    }
  }
}
```

## Security Considerations

### Safe-by-Default Philosophy

The hook follows a **conservative approach**:

1. **Unknown patterns require confirmation** - If a tool name doesn't match any pattern, the user is prompted
2. **Destructive patterns checked first** - Ensures dangerous operations can't slip through
3. **No server-specific logic** - All MCP servers treated equally (no special cases)

### Pattern Maintenance

Review patterns regularly:

```bash
# View current patterns
node -e "require('~/.claude/hooks/core/permission-request.js')"
```

Add new safe patterns as you identify trustworthy operations in your workflow.

## Troubleshooting

### Hook Not Auto-Approving

**Symptom:** All tools still prompt for permission

**Solutions:**
1. Check hook is installed: `ls ~/.claude/hooks/core/permission-request.js`
2. Verify settings: `cat ~/.claude/settings.json | grep permission-request`
3. Check configuration: `cat ~/.claude/hooks/config.json | jq .permissionRequest`
4. Enable debug logging to see decisions

### Unexpected Auto-Approvals

**Symptom:** Tool auto-approved when it should prompt

**Solutions:**
1. Check if tool name contains safe pattern: `echo "toolname" | grep -E "(get|list|read)"`
2. Add to `customDestructivePatterns` if needed
3. Report pattern mismatch for review

### Tool Name Parsing Issues

**Symptom:** Hook behavior inconsistent across MCP servers

**Solutions:**
1. Verify tool name format: should be `mcp__server__tool_name`
2. Check server name extraction is working
3. Enable debug logging to see extracted names

## Related Documentation

- [Main Hooks README](README.md) - Overview of all Claude Code hooks
- [CONFIGURATION.md](CONFIGURATION.md) - Detailed configuration guide
- [MCP Tool Annotations](https://modelcontextprotocol.io/docs/tools/annotations) - Upstream MCP documentation

## Changelog

### v1.0.0 (2026-01-08)
- Initial release
- Universal MCP server support
- Safe/destructive pattern matching
- Zero-config operation

## Credits

Original implementation: [Universal Permission Request Hook Gist](https://gist.github.com/doobidoo/fa84d31c0819a9faace345ca227b268f)

## License

Part of the MCP Memory Service project. See main LICENSE file.
