# MCP Memory Service for DeepSeek Harness

This bundle contributes a short **project memory snapshot** to the system prompt of
every new DeepSeek Harness agent. It asks a running MCP Memory Service for memories
that match the working directory's project and wraps the result as recalled data,
not as instructions.

It is retrieval only. It does not write memories; capture is tracked in
[#2111](https://github.com/doobidoo/mcp-memory-service/issues/2111). Until that
lands, the `mcp__memory__*` tools remain the way to store and search on demand.

## Behaviour

- On `agent/created` with source `startup` or `clear`, the plugin reads the session
  working directory, derives the project name from its last path segment, and calls
  `POST /api/search` with `{project} architecture decisions, recent work and open issues`.
- Up to `limit` results are rendered as one `project-memory` block of at most
  `maxChars` characters. The block carries an explicit "data, not instructions"
  line and is registered with `interpolate: false`.
- The snapshot is stored per agent and registered once in the system prompt. On
  `agent/disposed` the agent's entry is dropped.
- `resume` and `compact` are skipped: those sessions already carry the earlier
  snapshot in their history.
- A missing, slow or unreachable service is ignored. Agent creation never fails
  because memory is unavailable.

## Configuration

Set these in the plugin row's `config` (the bundle patch reads `MCP_MEMORY_API` and
`MCP_API_KEY` from the DSH process environment by default):

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turn the snapshot off without uninstalling the bundle. |
| `endpoint` | `http://127.0.0.1:8000` | Base URL of the HTTP service; `/api/search` is appended. |
| `apiKey` | empty | Sent as `X-API-Key` when set. |
| `limit` | `5` | Number of search results requested and rendered. |
| `timeoutMs` | `1500` | Hard budget for the search; agent creation waits no longer. |
| `maxChars` | `2000` | Upper bound for the whole injected block. |
| `order` | `150` | System-prompt context order. |
| `query` | `{project} architecture decisions, recent work and open issues` | Search query; `{project}` is replaced. |

The DSH process must trust the endpoint's certificate. This bundle has no TLS
override, so use loopback HTTP or a chain the process trusts.

## Install

Install the directory through **Plugins → Add plugin** (local folder), or point
`plugin_manager` at `dsh/memory-hooks/` with `install_bundle`. Then start a new
session and ask the agent to quote any project-memory block from its system
prompt. Removing the bundle removes the context; no repository files are touched.
