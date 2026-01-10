# Fix: Support Flexible MCP Memory Server Naming Conventions

## Problem

The hook installer (`install_hooks.py`) was hardcoded to check for an MCP server named exactly `memory`:

```python
result = subprocess.run(['claude', 'mcp', 'get', 'memory'], ...)
```

However, Claude Code allows users to choose their own MCP server names when running `claude mcp add`, and many users name their memory servers differently:

- `memory-service` (common for HTTP MCP servers)
- `extended-memory` (used by older installations)
- `mcp-memory-service` (descriptive naming convention)

This hardcoded assumption caused the installer to incorrectly report:

```
❌ No existing memory server found in Claude Code MCP configuration
❌ Standalone environment detected (no active MCP server)
```

Even when `claude mcp list` showed the memory server was properly configured and connected!

## User Impact

Users with properly configured memory MCP servers were told they had "no MCP defined" and were forced into "standalone mode" unnecessarily. This was confusing and prevented the installer from using the existing MCP configuration (DRY principle).

## Solution

Updated `install_hooks.py` to support flexible memory server naming:

### 1. Check Multiple Common Names

Instead of only checking `memory`, now tries multiple common naming patterns:

```python
memory_server_names = ['memory-service', 'memory', 'mcp-memory-service', 'extended-memory']

for server_name in memory_server_names:
    result = subprocess.run(['claude', 'mcp', 'get', server_name], ...)
    if result.returncode == 0:
        return config_info
```

### 2. Fallback to List Detection

If no specific name matches, falls back to searching all MCP servers:

```python
result = subprocess.run(['claude', 'mcp', 'list'], ...)
if result.returncode == 0 and 'memory' in result.stdout.lower():
    return {'status': 'Connected', 'type': 'detected', ...}
```

### 3. Support HTTP Server Format

Updated the output parser to handle HTTP MCP servers, which show `URL:` instead of `Command:`:

```python
elif line.startswith('URL:'):
    # HTTP servers show URL instead of Command
    config['command'] = line.replace('URL:', '').strip()
    config['url'] = line.replace('URL:', '').strip()
```

### 4. Updated Validation Logic

Modified validation to accept both stdio and HTTP server types with appropriate format checking:

```python
if server_type == 'http':
    # HTTP servers use URL
    if not ('http://' in command or 'https://' in command):
        issues.append(f"HTTP server should have URL: {command}")
```

## Changes Made

### Files Modified
- `claude-hooks/install_hooks.py`
  - Lines 198-236: `detect_claude_mcp_configuration()` - Check multiple server names
  - Lines 238-268: `_parse_mcp_get_output()` - Support HTTP server format
  - Lines 367-383: Validation logic - Accept http type

### Backward Compatibility
✅ Still checks for `memory` (original hardcoded name)
✅ Existing installations continue to work
✅ No breaking changes to the installer API

## Testing

### Before Fix
```bash
$ claude mcp list
memory-service: http://mcp-memory:8000/mcp (HTTP) - ✓ Connected

$ python install_hooks.py --natural-triggers
❌ No existing memory server found in Claude Code MCP configuration
❌ Standalone environment detected (no active MCP server)
```

### After Fix
```bash
$ claude mcp list
memory-service: http://mcp-memory:8000/mcp (HTTP) - ✓ Connected

$ python install_hooks.py --natural-triggers
✅ Found existing memory server 'memory-service': http://mcp-memory:8000/mcp
✅ Status: ✓ Connected
✅ Type: http
✅ Valid MCP configuration detected!
✅ Claude Code environment detected (MCP server active)
```

### Test Scenarios Covered
- ✅ HTTP MCP server with custom name (`memory-service`)
- ✅ Stdio MCP server with standard name (`memory`)
- ✅ Alternative naming conventions (`extended-memory`, `mcp-memory-service`)
- ✅ Fallback detection via `claude mcp list`
- ✅ Backward compatibility with existing `memory` name

## Related Issues

Addresses the assumption in the installer that all users will name their memory MCP server exactly `memory`. The Claude Code MCP system is designed to be flexible, and the installer should respect that flexibility.

## Additional Context

Per the [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp), users configure MCP servers with their own chosen names:

```bash
claude mcp add memory-service --type http --url http://localhost:8000/mcp
```

The installer should detect any memory-related MCP server regardless of the specific name chosen by the user.

## Checklist

- [x] Code follows project style guidelines
- [x] Changes are backward compatible
- [x] Testing performed with multiple naming conventions
- [x] Documentation updated (this PR description)
- [x] No breaking changes introduced
- [x] Supports both HTTP and stdio MCP transports
