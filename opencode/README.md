# OpenCode Memory Awareness Plugin
Automatic memory retrieval, context injection, and write-back for OpenCode using the `mcp-memory-service` HTTP API.

This integration provides:

- **Session Start**: load relevant memories when an OpenCode session starts
- **Auto-Capture**: detect and store valuable conversation content (decisions, errors, learnings) in real-time via `message.part.updated`
- **Session-End**: consolidate buffered outcomes via `session.idle` (replaces the previous summary by deleting it and storing a new one)
- **Harvest**: optional pattern-harvesting via `/api/harvest` at session end
- **Compact Injection**: inject condensed memory context into `experimental.session.compacting`

## OpenCode V1 and V2

OpenCode V2 loads a plugin through `id` + `setup(context)`; V1 used `server(client)`
and `tui(api)`. V2 also renamed every hook and event this integration uses, so the
plugin maps them onto one implementation instead of forking it. Both fields are
exported, and the same file works in either host:

| V1 | V2 |
| --- | --- |
| `server(client)` | `setup(context)`, events arrive from `ctx.event.subscribe()` |
| `session.created` | `session.created` |
| `session.idle` | `session.step.ended` with `finish !== "tool-calls"` |
| `message.part.updated` | no equivalent; the adapter reads the `context` hook transcript |
| `command.execute.before` | registered in `setup` via `ctx.command.transform` |
| `client.app.log()` | no server-side equivalent (verbose logs become no-ops) |
| `client.tui.showToast()` | no server-side equivalent (toasts become no-ops) |
| `tui.json` + `{id, tui}` export | plugin package with a `"./tui"` export |

What that means in practice:

- The status file is the shared channel on V2; without toast and log support, the
  verbose/notification sections below only apply to V1.
- `/memory` needs no `opencode.json` entry on V2, because `setup` registers it.
- The sidebar widget is installed as a plugin package on V2 and needs no
  `tui.json` entry.

## Prerequisites

- An installed OpenCode version with plugin support; check with `opencode --version`.
- Python and pip for the service, and Git to obtain this repository's plugin files.
- `mcp-memory-service` running in HTTP mode, or an existing reachable HTTP deployment.
- A Bash-compatible shell for the commands below (macOS, Linux, or WSL).

The basic JavaScript plugin uses Node built-ins and the OpenCode runtime; no
separate npm dependency installation or sidebar build is required. The optional
sidebar additionally needs Node.js, Bun, and an OpenCode build with the TUI plugin API.

Start the service locally:

```bash
pip install mcp-memory-service
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

If you secure the API with `MCP_API_KEY`, set the client-side plugin key explicitly with `memoryService.apiKey` or `OPENCODE_MEMORY_API_KEY`.

Keep the local service running in a separate terminal and verify reachability:

```bash
curl --fail http://127.0.0.1:8000/api/health
```

The response must contain `"status": "healthy"`. For an authenticated endpoint,
include `-H "X-API-Key: $OPENCODE_MEMORY_API_KEY"` when testing protected API routes.

`http://127.0.0.1:8000` is only the default fallback. The plugin can target any reachable HTTP deployment of `mcp-memory-service`.

## Install

OpenCode loads local plugins automatically from:
- `~/.config/opencode/plugins/` for global plugins
- `.opencode/plugins/` for project-local plugins

Copy the plugin file to one of those locations:

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
mkdir -p ~/.config/opencode/plugins
cp opencode/memory-plugin.js ~/.config/opencode/plugins/
```

For a local anonymous service, no configuration file is needed. To customize the
plugin, copy the example only when the destination does not already exist:

```bash
cp -n opencode/memory-plugin.config.example.json ~/.config/opencode/memory-plugin.json
```

The example uses `http://127.0.0.1:8000`. Older copies used
`https://memory.example.com`; replace that placeholder before use. Preserve and
edit an existing configuration rather than overwriting it with the example.

No `plugin` entry is required in `opencode.json` when loading this JavaScript file
from the local plugin directory. Install it once, globally or per project, and
avoid also registering the same file through `plugin`.

For a project-only installation, run these commands from your project's root,
replacing the source path with this repository's actual checkout:

```bash
mkdir -p .opencode/plugins
cp /absolute/path/to/mcp-memory-service/opencode/memory-plugin.js .opencode/plugins/
```

Fully exit and restart OpenCode after installation or configuration changes.
Launch it from the project whose context you want to load.

## Configuration

The plugin reads the first valid, readable configuration file in this order:
- `OPENCODE_MEMORY_PLUGIN_CONFIG`
- `~/.config/opencode/memory-plugin.json`
- `~/.config/opencode/memory-awareness.json`
- `.opencode/memory-plugin.json`
- `.opencode/memory-awareness.json`

Then it applies environment overrides:
- `OPENCODE_MEMORY_ENDPOINT` or `OPENCODE_MEMORY_URL`
- `OPENCODE_MEMORY_API_KEY`
- `OPENCODE_MEMORY_TIMEOUT_MS`
- `OPENCODE_MEMORY_LOAD_TIMEOUT_MS`
- `OPENCODE_MEMORY_ALLOW_SELF_SIGNED_CERTS` — set to `true` or `1` to disable TLS certificate verification (self-signed certs only, off by default)

The files are not layered: an existing global file takes precedence over a
project-local file. To select a project's configuration explicitly, launch
OpenCode with an absolute path in `OPENCODE_MEMORY_PLUGIN_CONFIG`.
Environment overrides above take precedence over the selected file.

The memory configuration lookup currently uses these home-directory paths even
when OpenCode itself uses a custom config directory. Set
`OPENCODE_MEMORY_PLUGIN_CONFIG` explicitly for that case.

`MCP_API_KEY` is intentionally not consumed by the plugin. That avoids accidentally reusing the server-side secret from a shared shell environment.

Example:

```json
{
  "memoryService": {
    "endpoint": "https://memory.example.com",
    "apiKey": "",
    "maxMemoriesPerSession": 8,
    "searchTags": ["decision"],
    "includeProjectTag": false,
    "projectQueries": [
      "{project} architecture decisions",
      "{project} recent work",
      "{project} open issues"
    ]
  },
  "output": {
    "verbose": true,
    "includeTimestamps": true,
    "maxContentLength": 280
  }
}
```

For a purely local setup, change `endpoint` back to `http://127.0.0.1:8000`.

Environment-only example:

```bash
export OPENCODE_MEMORY_ENDPOINT="https://memory.example.com"
export OPENCODE_MEMORY_API_KEY="your-api-key"
```

Set these in the environment that launches OpenCode, not just in an MCP server's
environment block. With these client values configured, restart OpenCode.

## How It Works

The event names below are V1's. On V2 the adapter translates the equivalent V2
events into these before they reach the same handlers (see the table above), so
the behavior is identical.

On `session.created`, the plugin:
- derives the project name from the working directory
- runs a few semantic searches against the memory service
- stores the best matches in per-session plugin state
- `experimental.chat.system.transform` injects full memory context into the system prompt
- `experimental.session.compacting` injects a smaller memory summary into compaction context

On `message.part.updated` (every text part), the plugin:
- buffers parts for session-end analysis (dedupes by part id)
- detects valuable patterns (decisions, errors, learnings, etc.) via regex, ignoring fenced code blocks
- requires `minSentenceLength` (default 40) before considering a sentence
- stores matched content immediately as memories via `POST /api/memories`
- respects `#skip` (skip auto-capture) and `#remember` (force capture) overrides

On `session.idle` (after each assistant turn), the plugin:
- analyzes buffered messages for topics, decisions, insights, code changes, next steps
- stores a fresh session summary memory; deletes the previous one for this session via `DELETE /api/memories/<hash>` to avoid DB pollution
- optionally triggers pattern harvest via `POST /api/harvest` (opt-in, dry-run first-use safety)

`session.deleted` is also handled when received but tends to fire after the plugin's event subscribers have already been torn down, so it cannot be relied on as the only end-of-session trigger.

## Verification

1. Check `/api/health` before launching OpenCode.
2. Install the plugin in one of the directories above and restart OpenCode.
3. Start OpenCode in your project. After the first response, check the loaded count
   with the optional `/memory` command. A new or unrelated project can have zero matches.
4. Make a substantive decision, wait for the turn to finish, and confirm a capture
   or summary appears in the Memory Service dashboard. Check `/memory` for the
   capture count and last summary time.
5. If using the sidebar, check the status JSON file before investigating rendering.

If `verbose` is enabled, the plugin writes structured logs through `client.app.log()` under the `opencode-memory` service name.

## Slash Command: `/memory`

This command is optional. On OpenCode V2 the plugin registers it in `setup`, so
copying `memory-plugin.js` is enough and no configuration entry is needed. On V1,
where `command.execute.before` is the only hook, the command has to be declared:
merge the following entry into `~/.config/opencode/opencode.json`, preserving
your other commands and settings:

```json
{
  "command": {
    "memory": {
      "description": "Show MCP Memory Service status. Usage: /memory, /memory search <query>, /memory health",
      "template": ""
    }
  }
}
```

Then in OpenCode:
- `/memory` — current session status (project, loaded count, captured count, last action, last summary)
- `/memory search <query>` — top 5 semantic matches for the query
- `/memory health` — backend type, status, total memory count, endpoint

The plugin intercepts the command via `command.execute.before`, fetches data from the memory service, and replaces the user message with a formatted block plus a "reply verbatim" instruction. One small LLM round-trip per call. If your default model adds commentary anyway, set `command.memory.model` to a smaller/cheaper model. On V2 the plugin registers the command itself, so there is no `command.memory` block to set that field on.

## Optional MCP Tools

The HTTP plugin provides automatic retrieval and write-back. To also expose
interactive memory tools to the model, configure a separate MCP connection.
For a local stdio server, merge this into `opencode.json`:

```json
{
  "mcp": {
    "memory": {
      "type": "local",
      "command": ["memory", "server"],
      "enabled": true
    }
  }
}
```

The `memory` executable must be on OpenCode's PATH; use its absolute path if it
is installed in a virtual environment. Check the connection with `opencode mcp list`.
The HTTP API still needs to run for the plugin. See OpenCode's
[MCP server configuration](https://opencode.ai/docs/mcp-servers/) for remote connections.

## TUI Toasts

V1 only. OpenCode V2 has no server-side toast API, so the calls below are skipped
on V2 and the status file carries the same information.

The plugin shows transient toasts via `client.tui.showToast` in three situations (rate-limited to once per session):

- on first memory load: `Loaded N memories for <project>`
- on each auto-capture: `Captured <type> memory for <project>`
- on first session summary store: `Storing session summary for <project>`

`variant` must be one of `info | success | warning | error` — omitting it returns 400 from `/tui/show-toast`.

## Status File Bridge

The plugin writes a JSON snapshot to `~/.local/state/opencode/.memory-status.json` on each significant event (load, capture, summary). Override via `OPENCODE_MEMORY_STATUS_FILE`. Schema:

```json
{
  "projectName": "string",
  "loadedCount": 0,
  "capturedCount": 0,
  "lastAction": "string",
  "lastSummaryAt": "ISO timestamp",
  "updatedAt": "ISO timestamp"
}
```

This file is consumed by the TUI sidebar widget (next section) but is also useful for any external tool that wants the latest snapshot of memory activity.

## TUI Sidebar Widget

The repo ships the same panel for both generations:

- `opencode/memory-status-tui.v2.tsx` — V2: `Plugin.define` and the `sidebar.content` slot
- `opencode/memory-status-tui.tsx` — V1: `{id, tui}` and `sidebar_content`

It shows project, loaded count, captured count, and the last action, and polls the
status file every 1.5 seconds. The widget is optional; retrieval and capture work
without it.

### OpenCode V2

V2 needs no TUI config file: a directory under `plugins/` whose `package.json`
exports `"./tui"` is a plugin package, and the widget ships as the second export of
the HTTP plugin's directory.

```bash
cd ~/.config/opencode
bun add @babel/core@7 @babel/preset-typescript@7 babel-preset-solid
node /absolute/path/to/mcp-memory-service/opencode/build-tui-plugin.mjs
```

The build writes `~/.config/opencode/plugins/opencode-memory/`:

| File | Content |
| --- | --- |
| `index.js` | copy of `memory-plugin.js` |
| `tui.js` | compiled V2 widget |
| `package.json` | `exports: {".": "./index.js", "./tui": "./tui.js"}` |

That directory is the whole V2 installation; no `opencode.json` or `tui.json` entry
is needed. A `Could not deploy` warning means the Babel modules were not found.

### OpenCode V1

V1 registers the widget through `tui.json`, from a directory that resolves through
its `package.json` "main". The build script compiles the V1 widget too, but leaves
that directory and its `package.json` to you: a stray plugin package under
`plugins/` is picked up by V2 plugin discovery and fails to load there.

**Install (one-time):**
```

Good. Let me verify what I wrote stays inside the fence... Let me just do the edit.</think>

<｜DSML｜ calls>
<｜DSML｜ invoke name="edit">
<｜DSML｜ parameter name="newString" string="true">The repo ships the same panel for both generations:

- `opencode/memory-status-tui.v2.tsx` — V2, `Plugin.define` and the `sidebar.content` slot
- `opencode/memory-status-tui.tsx` — V1, `{id, tui}` and `sidebar_content`

It shows the project, loaded count, captured count, and the last action, and polls
the status file every 1.5 seconds. The panel is optional: retrieval and capture
work without it.

### OpenCode V2

V2 needs no TUI config file. A directory under `plugins/` whose `package.json`
exports `"./tui"` is a plugin package, and the widget ships as the second export of
the HTTP plugin's directory:

```bash
cd ~/.config/opencode
bun add @babel/core@7 @babel/preset-typescript@7 babel-preset-solid
node /absolute/path/to/mcp-memory-service/opencode/build-tui-plugin.mjs
```

The build writes `~/.config/opencode/plugins/opencode-memory/`:

| File | Content |
| --- | --- |
| `index.js` | copy of `memory-plugin.js` |
| `tui.js` | compiled V2 widget |
| `package.json` | `exports: {".": "./index.js", "./tui": "./tui.js"}` |

That directory is the whole V2 installation; no `opencode.json` and no `tui.json`
entry is needed. A `Could not deploy` warning means the Babel modules were not
found.

### OpenCode V1

V1 registers the widget through `tui.json`, from a directory that resolves through
its `package.json` "main". The build compiles the V1 widget too, but leaves that
directory to you: a stray plugin package under `plugins/` is picked up by V2 plugin
discovery and fails to load there.

This is optional; automatic memory retrieval and capture work without the widget.
It uses the TUI plugin API shown below, which is separate from the HTTP plugin's
server hooks. Do not assume these instructions apply unchanged to a different
OpenCode TUI API or a development version.

**Install (one-time):**

```bash
# 1. Create the deployment directory before running the build script.
mkdir -p ~/.config/opencode/plugins/memory-status-tui

# 2. Install build dependencies and the widget's runtime dependencies explicitly.
cd ~/.config/opencode
bun add @babel/core@7 @babel/preset-typescript@7 babel-preset-solid solid-js @opentui/solid @opentui/core
```

Use Babel 7 for this build script. Babel 8 removes the TypeScript preset options
it currently uses and fails the build.

Create `~/.config/opencode/plugins/memory-status-tui/package.json` if it does not
exist, or merge these fields into an existing file:

```json
{
  "name": "opencode-memory-status",
  "type": "module",
  "main": "./index.js"
}
```

Compile and deploy from the repository checkout:

```bash
node /absolute/path/to/mcp-memory-service/opencode/build-tui-plugin.mjs
```

The build script writes the compiled file to `opencode/memory-status-tui.js`, then
copies it to the deployment directory's `index.js` and writes that directory's
`package.json`. It does not create the directory itself, so a `Could not deploy`
warning means the directory is missing, even if compilation succeeded. For a custom
build/config directory, set `OPENCODE_CONFIG_DIR` and install the dependencies and
widget there.

**Merge into `~/.config/opencode/tui.json`**, preserving existing settings
(TUI plugins use a separate config file from server plugins):

```json
{
  "$schema": "https://opencode.ai/tui.json",
  "plugin": [
    "file:///absolute/path/to/.config/opencode/plugins/memory-status-tui"
  ]
}
```

Replace the file URL with the actual absolute deployment path; `~` is not
expanded in a file URL. Confirm that `index.js` and `package.json` exist, then
restart OpenCode. If you override `OPENCODE_MEMORY_STATUS_FILE`, give the HTTP
plugin and the TUI process the same value. The sidebar shows a new "Memory"
section above Context. The separately registered `/memory` command renders a
status block in chat:

![Memory sidebar widget and /memory slash command in OpenCode](screenshots/sidebar.png)

**Key takeaway for plugin authors:** the two OpenCode generations disagree about how
a plugin is found. V2 loads `id` + `setup(context)` from any package under `plugins/`,
which is why one directory there carries both the server plugin and the `"./tui"`
widget. V1 splits config across two files: server plugins (`{id, server}` exports) go
in `opencode.json["plugin"]`, TUI plugins (`{id, tui}` exports) go in
`tui.json["plugin"]`. Putting a TUI plugin in `opencode.json` triggers the loader
error `must default export an object with server()`, and a TUI-only module under
`plugins/` fails V2 discovery for the same reason.

## Limitations

- depends on the HTTP API being reachable
- relevance is intentionally simple and project-name driven in the first cut
- auto-capture uses regex-based pattern detection (no LLM-based classification)
- session-end consolidation may overlap with auto-capture entries (both write to `/api/memories`)
- mid-conversation memory injection (when a user asks "what did we do before?") is not yet implemented

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No memory activity | Confirm the HTTP endpoint, the plugin file location, and a full OpenCode restart. |
| Requests go to `memory.example.com` | Edit an older copied configuration or override `OPENCODE_MEMORY_ENDPOINT`. |
| HTTP 401 or 403 | Set `OPENCODE_MEMORY_API_KEY` in OpenCode's launch environment; configuring MCP OAuth does not authenticate this HTTP plugin. |
| Project configuration is ignored | A global file wins; use `OPENCODE_MEMORY_PLUGIN_CONFIG` to select the project file explicitly. |
| `/memory` is missing | On V2 look for a plugin setup error in the log; on V1 merge `command.memory` into `opencode.json` and restart. |
| `Loaded 0` | The database may be empty, the query may not match, or configured tags may exclude the project. Check `/memory search <query>`. |
| Widget build reports a missing Babel module | Install all explicit build dependencies in the build's config directory. |
| V1 widget compiles but does not deploy | Create the deployment directory and `package.json` first; check for `Could not deploy`. |
| Status JSON updates but sidebar is absent | Check the TUI plugin API, the `tui.json` registration on V1 or the package `exports["./tui"]` on V2, and a shared status-file path. |

OpenCode's [local plugin loading documentation](https://opencode.ai/docs/plugins/)
describes the automatic plugin directories and dependency lookup.
