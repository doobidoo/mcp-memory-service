# OpenCode Memory Awareness Plugin
Automatic memory retrieval, context injection, and write-back for OpenCode using the `mcp-memory-service` HTTP API.

This integration provides:

- **Session Start**: load relevant memories when an OpenCode session starts
- **Auto-Capture**: detect and store valuable conversation content (decisions, errors, learnings) in real-time via `chat.message`
- **Session-End**: consolidate full-session outcomes and store as summary memories when a session ends
- **Harvest**: optional pattern-harvesting via `/api/harvest` at session end
- **Compact Injection**: inject condensed memory context into `experimental.session.compacting`

## Prerequisites

- OpenCode with plugin support
- `mcp-memory-service` running in HTTP mode

Start the service locally:

```bash
pip install mcp-memory-service
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

If you secure the API with `MCP_API_KEY`, set the client-side plugin key explicitly with `memoryService.apiKey` or `OPENCODE_MEMORY_API_KEY`.

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

Optional: install the example config as a starting point:

```bash
cp opencode/memory-plugin.config.example.json ~/.config/opencode/memory-plugin.json
```

No `plugin` entry is required in `opencode.json` when loading from the local plugin directory.

## Configuration

The plugin looks for config in this order:
- `options.configPath` when the plugin is loaded programmatically
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

If you load the plugin with explicit plugin options, those win last.

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

## How It Works

On `session.created`, the plugin:
- derives the project name from the working directory
- runs a few semantic searches against the memory service
- stores the best matches in per-session plugin state
- `experimental.chat.system.transform` injects full memory context into the system prompt
- `experimental.session.compacting` injects a smaller memory summary into compaction context

On `chat.message` (every new message), the plugin:
- buffers user messages for session-end analysis
- detects valuable patterns (decisions, errors, learnings, etc.) via regex
- stores matched content immediately as memories via `POST /api/memories`
- respects `#skip` (skip auto-capture) and `#remember` (force capture) overrides

On `session.deleted`, the plugin:
- analyzes all buffered messages for topics, decisions, insights, code changes, and next steps
- stores a session summary memory with extracted analysis
- optionally triggers pattern harvest via `POST /api/harvest` (opt-in, dry-run first-use safety)

## Verification

1. Start `mcp-memory-service` in HTTP mode.
2. Install the plugin under `~/.config/opencode/plugins/`.
3. Start OpenCode inside a project you already have memories for.
4. Ask a question about the project and confirm the assistant can use prior context.

If `verbose` is enabled, the plugin writes structured logs through `client.app.log()` under the `opencode-memory` service name.

## Limitations

- depends on the HTTP API being reachable
- relevance is intentionally simple and project-name driven in the first cut
- auto-capture uses regex-based pattern detection (no LLM-based classification)
- session-end consolidation may overlap with auto-capture entries (both write to `/api/memories`)
- mid-conversation memory injection (when a user asks "what did we do before?") is not yet implemented
