# Claude Code Plugin Packaging — Design Spec

**Date:** 2026-04-19
**Status:** Approved, pending implementation plan
**Related:** Issue #530 (Option A/B/Plugin), PR #532 (Option C done)

## Problem

The `claude-hooks/` suite ships 8 Node.js hooks plus a Python installer. Setup
requires editing `~/.claude.json`, generating API keys, running
`install_hooks.py`, and keeping two server processes alive. Writes fail
silently when the HTTP server is not running (issue #530: 218+ sessions, zero
stored consolidations).

Option C (auto-start fix) shipped in PR #532. Options A (embedded HTTP) and B
(hooks via MCP protocol) remain unaddressed. Claude Code's plugin system is
now mature enough to be a viable distribution channel and replaces most of
the manual setup with `/plugin install`.

## Goals

1. Ship the hooks as an installable Claude Code plugin via GitHub marketplace
2. Remove the raw-HTTP write path from hooks (protocol-native writes)
3. Preserve the existing installer flow during rollout (no breaking change)
4. Minimize scope: core hooks only, no feature additions

## Non-Goals

- Removing `install_hooks.py` in this release
- Windows-specific packaging beyond what hooks already support
- Self-upgrade, telemetry, or server-lifecycle-on-session-end
- Plugins for the extended hook set (harvest, topic-change, permission-request,
  memory-retrieval) — deferred to v2
- Embedded HTTP in the stdio MCP process (Option A) — not pursued; hybrid mode
  with `MCP_HYBRID_SYNC_OWNER=http` cleanly separates concerns

## Architecture

Two sequential pull requests in `doobidoo/mcp-memory-service`:

### PR 1 — `storeMemory()` on MemoryClient

**Scope:** Add protocol-native write path, mirrors existing read pattern.

- `claude-hooks/utilities/memory-client.js` gains `storeMemory(content, { tags, memoryType, metadata })`
  - Primary: HTTP POST `/api/memories` (existing behavior, extracted)
  - Fallback: MCP `store_memory` tool via existing `mcp-client.js`
  - Returns: `{ success, contentHash, error }`
- `claude-hooks/core/session-end.js:274` refactored to call `memoryClient.storeMemory()`
- `claude-hooks/core/auto-capture-hook.js:166` same refactor
- Tests: `claude-hooks/tests/memory-client-store.test.js`
  - Mocked `fetch` for HTTP path
  - Mocked MCP client for fallback path
  - Error-handling contract (network failures, 401s, malformed responses)

**Size estimate:** ~150 LOC, isolated, no dependency on PR 2.

**Unblocks:** All existing users (not just plugin adopters) from the silent-write-failure class of bugs.

### PR 2 — Plugin Packaging

**Scope:** New files only. Existing hooks/utilities untouched.

#### Directory layout

```
claude-hooks/
├── .claude-plugin/
│   ├── plugin.json           # Plugin manifest
│   └── hooks.json            # Hook event wiring
├── .mcp.json                 # MCP server registration
├── scripts/
│   ├── ensure-server.js      # Self-healing HTTP starter
│   └── test-plugin-install.sh # Structural smoke test
├── core/                     # UNCHANGED (session-start, session-end, mid-conversation, auto-capture)
├── utilities/                # UNCHANGED (includes storeMemory from PR 1)
├── PLUGIN.md                 # NEW: plugin-specific docs
└── [existing files untouched]

Repo root:
.claude-plugin/
└── marketplace.json          # Makes repo discoverable via /plugin marketplace add
```

#### `plugin.json`

```json
{
  "name": "mcp-memory-service",
  "version": "1.0.0",
  "description": "Automatic memory capture and injection for Claude Code via MCP Memory Service",
  "author": "doobidoo",
  "homepage": "https://github.com/doobidoo/mcp-memory-service",
  "mcpServers": "./.mcp.json",
  "hooks": "./.claude-plugin/hooks.json"
}
```

#### `hooks.json` (v1 scope: 4 hooks)

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup|resume",
      "hooks": [
        { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/scripts/ensure-server.js\"" },
        { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/session-start.js\"" }
      ]
    }],
    "SessionEnd": [{
      "hooks": [{ "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/session-end.js\"" }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{ "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/mid-conversation.js\"" }]
    }],
    "PostToolUse": [{
      "hooks": [{ "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/auto-capture-hook.js\"" }]
    }]
  }
}
```

Per-event filtering (e.g. which tools trigger auto-capture) remains inside the
hook scripts. No matcher changes at the plugin level.

#### `.mcp.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
        "MCP_HYBRID_SYNC_OWNER": "http"
      }
    }
  }
}
```

#### `marketplace.json` (repo root)

```json
{
  "name": "mcp-memory-service",
  "owner": { "name": "doobidoo", "url": "https://github.com/doobidoo" },
  "plugins": [{
    "name": "mcp-memory-service",
    "source": "./claude-hooks",
    "description": "Semantic memory for Claude Code sessions"
  }]
}
```

### `ensure-server.js` behavior

Invoked at SessionStart before any memory-dependent hook.

1. HTTP GET `/api/health` on the configured endpoint, 500ms timeout
2. 200 OK → exit 0
3. Connection refused / non-200 → spawn background process:
   ```
   python scripts/server/run_http_server.py > <log-path> 2>&1 &
   ```
   with `detached: true`, `stdio: 'ignore'`, `unref()`
4. Poll `/api/health` every 500ms for up to 10 seconds
5. Success → exit 0; timeout → stderr warning, exit 0

**Edge cases:**
- **Custom port:** Read endpoint from `~/.claude/hooks/config.json`, not hardcoded
- **Python missing:** Try `python`, `python3`, `.venv/bin/python`; on failure → stderr + exit 0
- **Log path:** `~/.mcp-memory-service/http.log`, fallback `/tmp/mcp-memory-service-http.log`
- **Stale lockfile:** Check PID in `~/.mcp-memory-service/http.pid`; remove if process is gone
- **Never block session start:** All failure paths end in `exit 0` with stderr warning

## Configuration

Plugin reads `~/.claude/hooks/config.json` (existing installer path).

- Existing installer users: no migration step
- Fresh plugin installs without pre-existing config: `ensure-server.js` detects
  missing file on first run, logs a one-time setup hint to stderr (does not
  auto-create; user follows `PLUGIN.md`)

**Rationale:** Config asymmetry between installer and plugin users would break
"both workflows work" invariant.

## Distribution

Plugin installable via:

```
/plugin marketplace add doobidoo/mcp-memory-service
/plugin install mcp-memory-service
```

`git pull` on the marketplace keeps users updated. No external hosting required.

## Testing

### Unit tests (Node, runs in CI)

- `claude-hooks/tests/memory-client-store.test.js` (in PR 1)
- `claude-hooks/tests/ensure-server.test.js` (in PR 2, mocks `fetch` and
  `child_process.spawn`)

### Smoke test (shell, runs in CI)

`scripts/test-plugin-install.sh` validates:
- `.claude-plugin/plugin.json` parseable
- `.claude-plugin/hooks.json` parseable
- Every script referenced in `hooks.json` exists
- Every `require()` in referenced scripts resolves
- `.mcp.json` parseable

### Manual E2E (PR description checklist)

1. Fresh Claude Code install, no prior MCP Memory Service
2. `/plugin marketplace add doobidoo/mcp-memory-service`
3. `/plugin install mcp-memory-service`
4. Start new session, verify HTTP server autostarts
5. Trigger auto-capture event, verify memory stored via MCP protocol
6. SessionEnd, verify consolidation memory stored

## Migration & Rollout

Installer (`install_hooks.py`) and plugin coexist.

**Risk:** User with both active runs hooks twice. Documented mitigation in
`PLUGIN.md`: remove installer hook entries from `~/.claude/settings.json`
before installing plugin.

**Rollout phases:**
1. PR 1 merges (storeMemory on MemoryClient)
2. PR 2 merges as experimental, flagged in `PLUGIN.md`
3. v11.0 minor release — plugin announced in CHANGELOG, README updated with
   both installation paths
4. ~4 weeks beta feedback
5. README flips: plugin becomes primary recommendation, installer marked legacy

**No breaking change** at any point. Installer removal is a future v12+ decision.

## Changelog entries

- PR 1: `feat(hooks): add storeMemory() to MemoryClient for protocol-native writes (closes #530 Option B)`
- PR 2: `feat: Claude Code plugin packaging for mcp-memory-service hooks`

## Open questions (for implementation)

- Plugin `name` collision with MCP server `name: "memory"` in `.mcp.json` —
  verify Claude Code namespaces these separately
- Whether `matcher: "startup|resume"` regex is supported or requires two
  separate entries (spec-check against current Claude Code hooks schema)
- Whether `PostToolUse` fires per-tool or once per turn (affects
  auto-capture-hook scheduling cost)

These are implementation-time verification items, not design blockers.
