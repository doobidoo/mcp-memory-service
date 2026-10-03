# Smart Auto-Capture for Claude Code

Intelligent automatic memory capture that stores valuable conversation content after tool operations.

## Overview

Smart Auto-Capture extends the MCP Memory Service hooks by automatically detecting and storing valuable conversation patterns. Ported from [shodh-cloudflare](https://github.com/doobidoo/shodh-cloudflare), it uses pattern detection to identify decisions, errors, learnings, and implementations.

## Features

- **Pattern Detection**: Automatically identifies 6 memory types (Decision, Error, Learning, Implementation, Important, Code)
- **Bilingual Support**: English and German pattern matching
- **User Overrides**: `#remember` and `#skip` markers for explicit control
- **Cross-Platform**: Node.js (primary) and PowerShell (Windows) implementations
- **Non-Blocking**: 5-second timeout, graceful failures

## Installation

```bash
cd claude-hooks
python install_hooks.py --auto-capture

# Or install everything:
python install_hooks.py --all
```

## Configuration

Configuration is stored in `~/.claude/hooks/config.json`:

```json
{
  "autoCapture": {
    "enabled": true,
    "minLength": 300,
    "maxLength": 4000,
    "patterns": ["decision", "error", "learning", "implementation", "important", "code"],
    "debugMode": false,
    "userOverrides": {
      "forceRemember": "#remember",
      "forceSkip": "#skip"
    }
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable auto-capture |
| `minLength` | `300` | Minimum content length to consider |
| `maxLength` | `4000` | Maximum content length (truncated if exceeded) |
| `patterns` | all | Which patterns to detect |
| `debugMode` | `false` | Enable verbose logging |

## Pattern Detection

Auto-capture analyzes conversation content for these patterns:

| Pattern | Type | Example Triggers |
|---------|------|------------------|
| **Decision** | `Decision` | "decided", "chose", "will use", "settled on", "entschieden" |
| **Error** | `Error` | "error", "fixed", "bug", "resolved", "behoben", "gefixt" |
| **Learning** | `Learning` | "learned", "discovered", "realized", "turns out", "gelernt" |
| **Implementation** | `Learning` | "implemented", "created", "built", "configured", "implementiert" |
| **Important** | `Context` | "critical", "important", "remember", "key", "wichtig" |
| **Code** | `Context` | "function", "class", "api", "database" (requires 600+ chars) |

### Priority Order

Patterns are evaluated in priority order (first match wins):
1. Decision (highest priority)
2. Error
3. Learning
4. Implementation
5. Important
6. Code (lowest priority, requires longer content)

## User Overrides

Control memory hook behavior directly in your messages.

> **Note**: As of v8.70.0, `#skip` and `#remember` apply to **all** memory hooks, not just auto-capture.
> See the main [README.md](README.md#-user-overrides-skip--remember) for complete documentation.

### Force Capture

Add `#remember` to ensure the conversation is stored:

```
Please implement the authentication flow #remember
```

### Skip Capture

Add `#skip` to prevent auto-capture:

```
Just testing something quickly #skip
```

### All Supported Hooks

| Hook | `#skip` | `#remember` |
|------|---------|-------------|
| **auto-capture** | Skips storage | Forces storage |
| session-start | Skips retrieval | Forces retrieval |
| mid-conversation | Skips analysis | Bypasses cooldown |
| session-end | Skips consolidation | Bypasses thresholds |

## Hook Trigger

Auto-capture triggers on `PostToolUse` for specific tools:

- **Edit** - Code edits
- **Write** - File creation
- **Bash** - Command execution

This ensures meaningful work is captured while avoiding noise from read-only operations.

## Generated Tags

Each auto-captured memory receives automatic tags:

- `auto-captured` - Identifies automatic capture
- `smart-ingest` - From shodh-cloudflare lineage
- Memory type (e.g., `decision`, `error`, `learning`)
- Matched pattern name
- Project name (from current working directory)

## Debug Mode

Enable debug mode to see pattern matching in action:

```json
{
  "autoCapture": {
    "debugMode": true
  }
}
```

Output example:
```
[auto-capture] Matched pattern: decision
[auto-capture] Storing Decision memory...
[auto-capture] Tags: auto-captured, smart-ingest, decision, mcp-memory-service
[auto-capture] Stored successfully in 45ms
```

## Architecture

```
PostToolUse (Edit/Write/Bash)
        │
        ▼
┌─────────────────────────────────────┐
│  auto-capture-hook.js/.ps1          │
│  ├── Read transcript                │
│  ├── Check user overrides           │
│  ├── Length validation              │
│  └── Pattern detection              │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  auto-capture-patterns.js           │
│  ├── PATTERNS definitions           │
│  ├── detectPatterns()               │
│  ├── hasUserOverride()              │
│  └── generateTags()                 │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Memory Service HTTP API            │
│  POST /api/memories                 │
└─────────────────────────────────────┘
```

## Troubleshooting

### Auto-capture not triggering

1. Check if enabled in config:
   ```bash
   cat ~/.claude/hooks/config.json | grep -A5 autoCapture
   ```

2. Verify hook is registered:
   ```bash
   cat ~/.claude/settings.json | grep -A10 PostToolUse
   ```

3. Enable debug mode and check output

### Memories not being stored

1. Check if HTTP server is running:
   ```bash
   curl http://127.0.0.1:8000/api/health
   ```

2. Verify API key matches:
   ```bash
   # In config.json
   grep apiKey ~/.claude/hooks/config.json

   # In .env
   grep MCP_API_KEY .env
   ```

### Pattern not matching

1. Content may be too short (< 300 chars default)
2. No matching keywords found
3. User override `#skip` in message

## Related Documentation

- [Claude Code Hooks README](README.md)
- [Memory Awareness Hooks Guide](README.md)
- [Natural Memory Triggers](README-NATURAL-TRIGGERS.md)

## Version History

- **v1.0.0** (2025-12-23): Initial release
  - Ported pattern detection from shodh-cloudflare
  - Node.js and PowerShell implementations
  - 6 pattern types with bilingual support
  - User override markers

## Credits

Smart Auto-Capture is based on the intelligent memory capture system from [shodh-cloudflare](https://github.com/doobidoo/shodh-cloudflare).
