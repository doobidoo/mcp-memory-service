# MCP Memory Wrapper with Orphan Cleanup

## Problem

When Claude Desktop or Claude Code crashes unexpectedly, child MCP processes may become "orphaned" - they continue running without a parent process. These orphaned processes hold locks on the SQLite database, causing "database is locked" errors for new MCP sessions.

## Solution

These wrapper scripts clean up orphaned processes before starting the MCP Memory Server, ensuring a clean database connection every time.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `memory_wrapper_cleanup.py` | All | Cross-platform Python wrapper |
| `memory_wrapper_cleanup.sh` | macOS/Linux | Native Bash wrapper |
| `memory_wrapper_cleanup.ps1` | Windows | Native PowerShell wrapper |

## How Orphan Detection Works

### macOS / Linux
- Orphaned processes are adopted by init (PID 1) or launchd
- The wrapper checks if `ppid == 1` to identify orphans
- Uses `pgrep` and `ps` for process inspection

### Windows
- Orphaned processes have a non-existent parent process
- The wrapper checks if the parent PID exists in the process list
- Uses WMI and `Get-Process` for inspection

## Configuration Examples

### Claude Desktop (`claude_desktop_config.json`)

**Using Python wrapper (recommended - works everywhere):**
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/scripts/run/memory_wrapper_cleanup.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
        "MCP_MEMORY_SQLITE_PATH": "/path/to/sqlite_vec.db"
      }
    }
  }
}
```

**Using Bash wrapper (macOS/Linux):**
```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/mcp-memory-service/scripts/run/memory_wrapper_cleanup.sh",
      "args": [],
      "env": { ... }
    }
  }
}
```

**Using PowerShell wrapper (Windows):**
```json
{
  "mcpServers": {
    "memory": {
      "command": "powershell",
      "args": [
        "-ExecutionPolicy", "Bypass",
        "-File", "C:\\path\\to\\mcp-memory-service\\scripts\\run\\memory_wrapper_cleanup.ps1"
      ],
      "env": { ... }
    }
  }
}
```

### Claude Code (`~/.mcp.json` or project `.mcp.json`)

Same format as Claude Desktop config.

## What Gets Cleaned Up

The wrapper identifies and terminates:
- Python processes with `mcp-memory-service` in command line
- Only if they are orphaned (parent process is init/launchd/non-existent)
- Never kills processes that have a valid parent (e.g., active Claude sessions)

## Logging

All wrappers log to stderr (visible in MCP server logs):
```
[mcp-memory-wrapper] Starting (Darwin 23.0.0)
[mcp-memory-wrapper] Found 2 orphaned process(es): [1234, 5678]
[mcp-memory-wrapper] Terminated orphaned process: 1234
[mcp-memory-wrapper] Terminated orphaned process: 5678
[mcp-memory-wrapper] Starting server with: uv run memory
```

## Requirements

- Python 3.8+ (for Python wrapper)
- Bash (for shell wrapper)
- PowerShell 5.1+ (for Windows wrapper)
- `uv` package manager installed

## Troubleshooting

### "uv not found"
Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
irm https://astral.sh/uv/install.ps1 | iex
```

### Still getting "database locked" errors
1. Manually check for stuck processes:
   ```bash
   # macOS/Linux
   pgrep -fl mcp-memory-service
   
   # Windows PowerShell
   Get-Process | Where-Object {$_.CommandLine -like "*mcp-memory-service*"}
   ```

2. Kill all MCP memory processes and restart:
   ```bash
   # macOS/Linux
   pkill -9 -f mcp-memory-service
   
   # Windows PowerShell
   Get-Process | Where-Object {$_.CommandLine -like "*mcp-memory-service*"} | Stop-Process -Force
   ```

### SQLite busy_timeout
For additional resilience, set in your environment:
```bash
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,cache_size=20000"
```

## Contributing

These wrappers are part of the [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) project. Issues and PRs welcome!
