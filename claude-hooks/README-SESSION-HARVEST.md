# Session End Auto-Harvest Hook

Lightweight Claude Code SessionEnd hook that calls `POST /api/harvest` on
session end to extract key information (decisions, bugs, conventions,
learnings, context) from the session transcript.

Implements issue #631. Requires the harvest endpoint from PR #710
(mcp-memory-service v10.37.0+).

## Why opt-in?

The harvest endpoint can:

- Store memories in your backend (when `dry_run=false`).
- Call an LLM for classification (when `use_llm=true`), which costs money
  (~$0.01–$0.02 per session via Groq/Gemini).

Both behaviors are off by default. The first time the hook ever runs it
additionally forces `dry_run=true` regardless of your config, so you can see
what it would extract before it writes anything. A marker file at
`~/.claude/mcp-memory-harvest-first-run.done` records that the first-run
preview happened; delete it to replay the preview.

## Enable

1. Make sure `claude-hooks/config.json` exists (copy from `config.template.json`).
2. Add or edit the `sessionHarvest` section:

```json
{
  "sessionHarvest": {
    "enabled": true,
    "dryRun": true,
    "dryRunOnFirstUse": true,
    "minSessionMessages": 10,
    "sessions": 1,
    "useLlm": false,
    "minConfidence": 0.6,
    "types": ["decision", "bug", "convention", "learning", "context"],
    "endpoint": "http://127.0.0.1:8000",
    "apiKey": null,
    "timeoutMs": 5000
  },
  "hooks": {
    "sessionEndHarvest": { "enabled": true, "timeout": 7000, "priority": "low" }
  }
}
```

3. Register the hook in `~/.claude/settings.json` (you do this yourself;
   the repo does not modify your user settings). Example snippet:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "node /absolute/path/to/mcp-memory-service/claude-hooks/core/session-end-harvest.js" }
        ]
      }
    ]
  }
}
```

See the [Claude Code hooks docs](https://docs.claude.com/en/docs/claude-code/hooks)
for how to register multiple SessionEnd hooks alongside the existing
`session-end.js` (consolidation) hook.

## Config reference

| Key                  | Default                                                   | Meaning                                                                       |
|----------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------|
| `enabled`            | `false`                                                   | Master switch. Must be `true` for the hook to do anything.                    |
| `dryRun`             | `true`                                                    | Sent as `dry_run` to `/api/harvest`. `true` = preview only, no writes.        |
| `dryRunOnFirstUse`   | `true`                                                    | First run ever forces `dry_run=true` regardless of `dryRun`.                  |
| `minSessionMessages` | `10`                                                      | Skip entirely if the transcript has fewer messages than this.                 |
| `sessions`           | `1`                                                       | Number of recent sessions to harvest.                                         |
| `useLlm`             | `false`                                                   | Enable LLM classification (incurs cost).                                      |
| `minConfidence`      | `0.6`                                                     | Minimum candidate confidence.                                                 |
| `types`              | `["decision","bug","convention","learning","context"]`    | Candidate types to extract.                                                   |
| `endpoint`           | `http://127.0.0.1:8000`                                   | Memory service HTTP endpoint (port 8000).                                     |
| `apiKey`             | `null`                                                    | Bearer token. Falls back to `MCP_API_KEY` env var.                            |
| `timeoutMs`          | `5000`                                                    | Hard timeout. On timeout the hook logs a warning and exits cleanly.           |
| `allowSelfSignedCerts` | `false`                                                 | **HTTPS only.** When `true`, disables TLS certificate validation (vulnerable to MITM). Use **only** for local dev with self-signed certs. |

## API key precedence

1. `context.apiKey` (provided by the Claude Code runner, if any)
2. `sessionHarvest.apiKey` in `config.json`
3. `memoryService.http.apiKey` / `memoryService.apiKey` in `config.json`
4. `MCP_API_KEY` environment variable
5. No `Authorization` header

## Example output

First run:

```
[Memory Hook] Harvest: first run detected, forcing dry_run=true
[Memory Hook] Harvest: POST http://127.0.0.1:8000/api/harvest (project=-Users-hkr-Repositories-mcp-memory-service, dry_run=true)
[Memory Hook] Session harvest: 7 candidates found, 0 stored (dry_run=true)
```

Subsequent run with `dryRun: false`:

```
[Memory Hook] Harvest: POST http://127.0.0.1:8000/api/harvest (project=-Users-hkr-Repositories-mcp-memory-service, dry_run=false)
[Memory Hook] Session harvest: 7 candidates found, 5 stored (dry_run=false)
```

Any failure (timeout, non-2xx, network error) logs as a warning and does
**not** block session end:

```
[Memory Hook] Harvest: request failed (non-fatal): timeout after 5000ms
```

## Costs

With `useLlm: true`, each session harvest routes through the memory
service's quality/classification pipeline, which typically uses Groq
(Llama 3) or Gemini Flash. Expected cost: **~$0.01–$0.02 per session**.
Keep `useLlm: false` (default) for the Phase-1 pattern-based harvester if
you want zero cost.

## Project path

The hook derives `project_path` from `process.cwd()` by replacing path
separators with `-`, mirroring the Python side
(`str(Path.cwd()).replace(os.sep, "-")`). For example,
`/Users/hkr/Repositories/mcp-memory-service` becomes
`-Users-hkr-Repositories-mcp-memory-service`, which must match a
directory under `~/.claude/projects/`. The endpoint rejects absolute paths
and `..` components (CodeQL #383/#384).

## Tests

```bash
node claude-hooks/tests/session-end-harvest.test.js
```

7 tests cover: disabled-by-default, short-session skip, first-run forced
dry-run, subsequent-run honoring config, timeout resilience, HTTP 5xx
resilience, and API key precedence.
