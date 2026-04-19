# Claude Code Plugin Installation

This directory can be installed as a Claude Code plugin via the GitHub
marketplace. Two setup paths exist: the plugin (recommended for new users)
and the legacy Python installer (`install_hooks.py`). Both work; they should
not be active simultaneously.

## Install via plugin marketplace

```
/plugin marketplace add doobidoo/mcp-memory-service
/plugin install mcp-memory-service
```

Claude Code registers the MCP server (`.mcp.json`) and the hook wiring
(`.claude-plugin/hooks.json`) automatically. Updates come via `git pull`
on the marketplace entry.

## Configuration

The plugin reads `~/.claude/hooks/config.json`. If that file does not
exist, copy `config.template.json` from this directory:

```
mkdir -p ~/.claude/hooks
cp ~/.claude/plugins/marketplaces/doobidoo/mcp-memory-service/claude-hooks/config.template.json \
   ~/.claude/hooks/config.json
```

Then edit `~/.claude/hooks/config.json` — at minimum set
`memoryService.http.endpoint` and `memoryService.http.apiKey`.

## Server lifecycle

On SessionStart the plugin runs `scripts/ensure-server.js`, which:

1. Probes `GET /api/health` on the configured endpoint
2. If unreachable, tries (in order):
   - `memory server --http` (CLI entry point, works from any directory if `mcp-memory-service` is pip-installed)
   - `python -m mcp_memory_service.cli.main server --http` (module form)
   - `python scripts/server/run_http_server.py` (dev-mode fallback, requires repo checkout as cwd)
3. Polls for health for up to 10 seconds

The script never blocks session start — all failure paths exit 0 with a
warning on stderr. Logs go to `~/.mcp-memory-service/http.log` (fallback:
`/tmp/mcp-memory-service-http.log` if the preferred location is unwritable).

Set `ENSURE_SERVER_NO_SPAWN=1` to disable the spawn path (probe-only mode).

## Migrating from `install_hooks.py`

If you already installed via the Python installer, remove its hook entries
from `~/.claude/settings.json` before installing the plugin — otherwise
every hook runs twice.

```
grep -n "claude-hooks" ~/.claude/settings.json
```

Remove the matching entries from the `hooks` object and save.

## Uninstall

```
/plugin uninstall mcp-memory-service
```

This removes hooks and MCP server registration. It does not delete
`~/.claude/hooks/config.json` or `~/.mcp-memory-service/` logs.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hooks run twice | Both installer and plugin active | Remove installer hook entries from `~/.claude/settings.json` |
| `ensure-server.js` warns "could not spawn" | `mcp-memory-service` not pip-installed, no `memory` on PATH, no Python available | `pip install mcp-memory-service`, or set `MCP_MEMORY_PYTHON` to a Python with the package installed |
| Writes silently fail | API key missing | Set `memoryService.http.apiKey` in `~/.claude/hooks/config.json` |
| Server starts but never becomes healthy | Port collision | Check `~/.mcp-memory-service/http.log`, change port in `.env` |

## Status

v1.0.0 is **experimental**. Please file issues at
https://github.com/doobidoo/mcp-memory-service/issues with the `plugin` label.
