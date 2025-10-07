# HTTP Server Management

The MCP Memory Service HTTP server is **required** for Claude Code hooks (Natural Memory Triggers) to work. This guide explains how to check and manage the HTTP server.

## Why is the HTTP Server Required?

When using **Natural Memory Triggers** in Claude Code:
- The session-start hook needs the HTTP server to retrieve relevant memories
- Without the HTTP server, hooks fail silently and no memories are injected
- HTTP protocol avoids conflicts with Claude Code's MCP server

## Checking Server Status

### Quick Check

```bash
# Verbose output (default, recommended for troubleshooting)
uv run python scripts/server/check_http_server.py

# Quiet mode (only exit code, useful for scripts)
uv run python scripts/server/check_http_server.py -q
```

**Sample Output (Running):**
```
[OK] HTTP server is running
   Version: 8.3.0
   Endpoint: http://localhost:8000/api/health
   Status: healthy
```

**Sample Output (Not Running):**
```
[ERROR] HTTP server is NOT running

To start the HTTP server, run:
   uv run python scripts/server/run_http_server.py

   Or for HTTPS:
   MCP_HTTPS_ENABLED=true uv run python scripts/server/run_http_server.py

Error: [WinError 10061] No connection could be made...
```

## Starting the Server

### Manual Start

```bash
# HTTP mode (default, port 8000)
uv run python scripts/server/run_http_server.py

# HTTPS mode (port 8443)
MCP_HTTPS_ENABLED=true uv run python scripts/server/run_http_server.py
```

### Auto-Start Scripts

These scripts check if the server is running and start it only if needed:

**Unix/macOS:**
```bash
./scripts/server/start_http_server.sh
```

**Windows:**
```cmd
scripts\server\start_http_server.bat
```

**Features:**
- Checks if server is already running (avoids duplicate instances)
- Starts server in background/new window
- Verifies successful startup
- Shows server status and logs location

## Troubleshooting

### Hook Not Injecting Memories

**Symptom:** Claude Code starts but no memories are shown

**Solution:**
1. Check if HTTP server is running:
   ```bash
   uv run python scripts/server/check_http_server.py
   ```

2. If not running, start it:
   ```bash
   uv run python scripts/server/run_http_server.py
   ```

3. Restart Claude Code to trigger session-start hook

### Wrong Port or Endpoint

**Check your hooks configuration:**
```bash
cat ~/.claude/hooks/config.json | grep -A5 "http"
```

Should match your server configuration:
- Default HTTP: `http://localhost:8000`
- Default HTTPS: `https://localhost:8443`

### Server Startup Issues

**Common causes:**
- Port already in use
- Missing dependencies
- Configuration errors

**Debug steps:**
1. Check if port is in use:
   ```bash
   # Unix/macOS
   lsof -i :8000
   ```

   ```cmd
   # Windows
   netstat -ano | findstr :8000
   ```

2. Check server logs (when using auto-start scripts):
   ```bash
   # Unix/macOS
   tail -f /tmp/mcp-http-server.log

   # Windows
   # Check the server window
   ```

## Integration with Hooks

The session-start hook automatically:
1. Attempts to connect to HTTP server (preferred)
2. Falls back to MCP if HTTP unavailable
3. Falls back to environment-only if both fail

**Recommended setup for Claude Code** (`~/.claude/hooks/config.json`):
```json
{
  "memoryService": {
    "protocol": "http",
    "preferredProtocol": "http",
    "http": {
      "endpoint": "http://localhost:8000",
      "healthCheckTimeout": 3000
    }
  }
}
```

## Automation

### Start Server on System Boot

**Unix/macOS (launchd):**
Create `~/Library/LaunchAgents/com.mcp.memory.http.plist` and replace `/path/to/repository` with the absolute path to this repository:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mcp.memory.http</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/repository/scripts/server/start_http_server.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At log on
4. Action: Start a program
5. Program: `C:\path\to\repository\scripts\server\start_http_server.bat` (replace `C:\path\to\repository` with the full path to this repository)

### Pre-Claude Code Script

Add to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
# Auto-start MCP Memory HTTP server before Claude Code
# Replace /path/to/repository with the absolute path to this project
alias claude-code='/path/to/repository/scripts/server/start_http_server.sh && claude'
```

## See Also

- [Claude Code Hooks Configuration](../CLAUDE.md#claude-code-hooks-configuration-)
- [Natural Memory Triggers](../CLAUDE.md#natural-memory-triggers-v710-latest)
- [Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING)
