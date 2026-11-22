# MCP Memory Service for DeepSeek Harness

This bundle connects a DeepSeek Harness project to a running MCP Memory Service:

- **Read** — a short project-memory snapshot enters the system prompt of every new agent.
- **Write** — the end of a turn is captured when it contains a decision, error, learning or other pattern worth keeping, or when the prompt says `#remember`.

Both halves fail open: a missing or slow service never fails agent creation or a turn. Recalled text is labelled as data, not instructions.

Capture is opt-in per turn through the classifier; the plugin never stores tool output or injected context, only the assistant response and the human prompt.

## Behaviour

### Project snapshot

- On `agent/created` with source `startup` or `clear`, the plugin resolves the project root by walking up to the nearest marker (`.git`, `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, `composer.json`, `Gemfile`) and uses its directory name.
- Directories without a marker are skipped: a workspace like `default-workspace` has no project identity, so a semantic search would only return arbitrary high-scoring memories.
- It calls `POST /api/search` with `{project} architecture decisions, recent work and open issues`.
- Up to `limit` results are rendered as one `<project-memory>` block of at most
  `maxChars` characters. Entries are escaped (`&`, `<`, `>`) so a stored `</project-memory>` cannot close the block; the block carries an explicit "data, not instructions" line and is registered with `interpolate: false`.
- `resume` and `compact` are skipped: those sessions already carry the snapshot.

### Turn capture

- `session/event` buffers the turn's human prompts (`source.kind === 'user'`) and assistant text. Every other user-role message is injected context and is ignored.
- On `agent/turn-stopping` the buffered turn is classified with the shared patterns from `claude-hooks/utilities/auto-capture-patterns.js` (decision, error, learning, implementation, important, code).
- Captures shorter than `capture.minChars` are skipped; `#skip` in the prompt forces a skip and `#remember` forces a capture.
- The stored memory contains `Project:`, `Assistant:` and `User:` sections, is capped at `capture.maxChars`, and is tagged `dsh-capture`, `auto-captured`, the project name and the matched pattern. `memory_type` comes from the classifier.
- Subagent sessions (`header.parentSession`) are not captured by default; their turns usually repeat the parent's work.

### Configuration

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turn the whole bundle off without uninstalling it. |
| `endpoint` | `http://127.0.0.1:8000` | Base URL of the HTTP service. |
| `apiKey` | empty | Sent as `X-API-Key` when set. |
| `limit` | `5` | Search results requested and rendered. |
| `timeoutMs` | `1500` | Budget for each request. |
| `maxChars` | `2000` | Upper bound for the snapshot. |
| `order` | `150` | System-prompt context order. |
| `query` | `{project} architecture decisions, recent work and open issues` | Snapshot search query; `{project}` is replaced. |
| `requireProjectMarker` | `true` | Skip a working directory that has no project marker. |
| `projectMarkers` | see above | Marker names used to find the project root. |
| `capture.enabled` | `true` | Turn capture on or off. |
| `capture.minChars` | `300` | Minimum assistant text before classification. |
| `capture.maxChars` | `4000` | Upper bound for one captured memory. |
| `capture.includeSubagents` | `false` | Capture subagent turns too. |

The DSH process must trust the endpoint's certificate. This bundle has no TLS
override, so use loopback HTTP or a chain the process trusts.

## Install

Install the directory through **Plugins → Add plugin** (local folder), or point
`plugin_manager` at `dsh/memory-hooks/` with `install_bundle`. Then start a new
session and ask the agent to quote any project-memory block from its system
prompt; a turn with a decision stores a memory tagged `dsh-capture`. Removing the
bundle removes both halves; no repository files are touched.
