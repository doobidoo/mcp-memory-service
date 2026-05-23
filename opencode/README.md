# OpenCode Memory Awareness Plugin

Automatic memory retrieval and context injection for OpenCode using the `mcp-memory-service` HTTP API.

This integration is intentionally minimal in its first upstream form:
- load relevant memories when an OpenCode session starts
- inject memory context into `experimental.chat.system.transform`
- inject condensed memory context into `experimental.session.compacting`

It does **not** do automatic write-back or session harvesting yet. That is deferred to a later step so the first integration stays small, reviewable, and aligned with stable OpenCode plugin hooks.

## Why HTTP Instead of Direct Python Imports?

This plugin uses the documented HTTP API instead of importing Python internals directly.

That keeps the integration:
- host-agnostic
- easier to configure across platforms
- aligned with the public `mcp-memory-service` surface

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

Then:
- `experimental.chat.system.transform` injects full memory context into the system prompt
- `experimental.session.compacting` injects a smaller memory summary into compaction context

## Verification

1. Start `mcp-memory-service` in HTTP mode.
2. Install the plugin under `~/.config/opencode/plugins/`.
3. Start OpenCode inside a project you already have memories for.
4. Ask a question about the project and confirm the assistant can use prior context.

If `verbose` is enabled, the plugin writes structured logs through `client.app.log()` under the `opencode-memory` service name.

## Limitations

- read-only retrieval/injection only
- depends on the HTTP API being reachable
- relevance is intentionally simple and project-name driven in the first cut

Future work can add richer retrieval, manual refresh, and safe write-back once the host lifecycle hooks are proven stable for that path.
