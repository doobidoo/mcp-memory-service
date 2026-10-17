# Codex Memory Hooks

Automatic project memory retrieval, capture, and session write-back for Codex,
using the MCP Memory Service HTTP API. The adapter reuses the Claude Code
integration's pattern classifier and understands Codex rollout messages.

## Requirements

- A current local Codex runtime with native lifecycle hooks and `/hooks`.
- Node.js 18 or newer and Python 3.10 or newer for installation.
- A running MCP Memory Service HTTP endpoint.
- A checkout of this repository; these scripts are not installed by the Python package.

For a local service:

```bash
pip install mcp-memory-service
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

An existing authenticated local or remote deployment also works. The hooks use
HTTP directly and do not require an MCP connection in Codex.

## Install

From the repository root:

```bash
python3 codex/install_hooks.py --dry-run
python3 codex/install_hooks.py
```

The default destination is `$CODEX_HOME`, falling back to `~/.codex`. The installer:

- Copies the adapter and shared pattern classifier to `memory-service/`.
- Adds hooks to `hooks.json`, retaining other events, matchers, and handlers.
- Backs up an existing `hooks.json` before changing it.
- Creates `memory-service/config.json` only when it does not already exist.

Running it again updates the scripts without duplicating hooks or replacing your
memory configuration. It does not modify `config.toml`. If that file also defines
hooks, Codex loads both sources; review them for overlapping memory integrations.

Restart Codex and use **`/hooks` to review and trust the new or changed hooks**.
Installation does not bypass Codex's hook trust requirements.

For project-local installation:

```bash
python3 codex/install_hooks.py --project /absolute/path/to/project
```

This uses `<project>/.codex/`. The project configuration layer must be trusted in
Codex. The installed adapter reads its own configuration, including when Codex
starts in a subdirectory. State still lives under `$CODEX_HOME/memory-service/state`.

## Configuration

Edit the installed `memory-service/config.json`; see
[config.example.json](config.example.json) for all defaults. For example:

```json
{
  "endpoint": "https://memory.example.com",
  "apiKey": "",
  "projectName": "my-project",
  "autoCapture": true,
  "sessionSummary": true
}
```

Environment overrides:

| Variable | Purpose |
| --- | --- |
| `CODEX_MEMORY_CONFIG` | Absolute path to an alternative JSON configuration |
| `CODEX_MEMORY_ENDPOINT` | HTTP API base URL |
| `CODEX_MEMORY_API_KEY` | Client API key, sent as `X-API-Key` |

Environment overrides win over file configuration. The server-side `MCP_API_KEY`
is not read automatically. TLS certificates are verified normally.

Project names default to the Git root directory name, lowercased, with the local
`--wt-...` worktree suffix removed. Outside Git, the working directory name is
used. Set `projectName` to match the project tag already used by your memories.
Retrieval filters search results by that tag on the client because the semantic
search endpoint does not enforce tag filters. It returns at most eight memories
and 6,000 characters by default.

## Lifecycle

| Event | Behavior |
| --- | --- |
| `SessionStart` | Retrieves project memories and returns Codex `additionalContext`. Also runs for `source: compact` after context compaction. |
| `Stop` | Captures new decisions, errors, learnings, and other configured patterns after each response; saves an incremental outcome summary. |
| `PreCompact` | Saves pending outcomes before compaction. |
| `SessionEnd` | Attempts a final checkpoint within the short session-end deadline. |

Capture runs at response boundaries, rather than on every tool call. It ignores
developer/system messages, reasoning, tool output, assistant commentary phases,
workspace instruction envelopes, and fenced code blocks. Both Codex
`response_item` messages and older `event_msg` message records are supported;
when both exist, only the response messages are used.

- **`#skip`** in a user message excludes that turn, including the assistant's reply.
- **`#remember`** forces capture of that user message even below the length threshold.
- `autoCapture: false` disables individual captures.
- `sessionSummary: false` disables incremental summaries. Disable both for retrieval only.

Summaries contain newly observed substantive outcomes, not another copy of the
entire conversation. They use the ontology's `session` subtype. Successful writes
are checkpointed immediately so subsequent `Stop`, compaction, and session-end
events do not duplicate them. An exact-duplicate rejection is checkpointed only
after reading the existing memory by its normalized content hash and confirming
the content in the same store. Other storage failures remain retryable. Repeated
pending outcomes occupy one summary slot. There are no server-side memory deletions.

## State and failure behavior

State contains deduplication hashes and the last retrieved context, stored in
private files under `$CODEX_HOME/memory-service/state`. Session IDs are hashed
before use in filenames. A per-session lock prevents concurrent callbacks from
duplicating writes. A lock left by a killed process expires after one minute.
The same callback removes an expired lock, retries exclusive acquisition, and
continues its work if it acquires the lock. Fresh locks are left in place.

Requests have an absolute timeout. `SessionEnd` has a 1.8-second internal budget
inside Codex's three-second limit; the other hooks have a 7.5-second budget.
Errors are advisory and do not block Codex. Failed writes remain eligible for a
later checkpoint. A cached context can be reused when retrieval fails.

Codex documents its transcript format as unstable. Unknown records are ignored,
and malformed or unfinished JSONL lines are skipped. Transcripts above 32 MiB
are skipped. A killed hook or permanent server outage can leave outcomes unsaved;
there is no detached background worker. This integration targets local Codex
execution, not hosted cloud hook execution.

## Verify and disable

1. Use `/hooks` to check that all four memory hooks are trusted.
2. Start Codex in a project with memories tagged with its project name.
3. Confirm that prior context is available, then make a substantive decision and
   check the Memory Service dashboard for `codex` memories.
4. Repeat a checkpoint without new messages and confirm no additional writes.

Disable the entries in `/hooks` to stop the integration. The installer does not
automatically enable hooks disabled globally with `[features].hooks = false`.

Run the offline tests from the repository root:

```bash
node codex/tests/memory-hook.test.js
node codex/tests/install-hooks.test.js
bash scripts/ci/run_hooks_tests.sh
```

The tests use temporary installations and a local mock HTTP server. They do not
read your real transcripts or write to your memory database.

Codex event, input, output, and trust contracts:
[OpenAI Hooks documentation](https://learn.chatgpt.com/docs/hooks).
