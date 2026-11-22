# MCP Memory Service for DeepSeek Harness (DSH)

DeepSeek Harness ships an MCP client (`@deepseek-ai/dsh-mcp-client`), and the
memory service exposes MCP over Streamable HTTP at `/mcp`. One loader row is
enough to give DSH the full tool surface; no code change is needed on either
side.

Verified against a running service (v13.0.0) with DSH's bundled
`@modelcontextprotocol/client` 2.2.0: the handshake negotiates `2024-11-05`,
`tools/list` returns 26 tools, and `memory_search` returns results.

## Prerequisites

- A running HTTP service. For a local, unauthenticated first try:

  ```bash
  MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
  ```

  The endpoint is `http://127.0.0.1:8000/mcp`. The REST API on the same port is
  not used by DSH.
- DeepSeek Harness 0.2.x. The row below was written against the Desktop build
  whose CLI reports `0.2.2-rc.5`.

## Configure

DSH composes its profile from layers; the user layer is
`$DSH_HOME/profiles/<profile>/cordis.patch.yml` (on macOS
`~/.dsh/profiles/<profile>/cordis.patch.yml`). It is a YAML array: keep the
existing rows and append the one from [`cordis.patch.yml`](cordis.patch.yml).

```yaml
- insert:
    - id: memory-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: memory
        transport: streamable-http
        url: http://127.0.0.1:8000/mcp
        failOnStartupError: true
```

Restart DeepSeek Harness after the edit. The client discovers the tools during
startup; they appear as `mcp__memory__<tool>`.

For an authenticated service, start DSH with `MCP_API_KEY` in its environment
and add the header:

```yaml
        headers:
          X-API-Key: !!js process.env.MCP_API_KEY
```

The MCP client also accepts `headers: { X-API-Key: '${credential:MCP_API_KEY}' }`
with `expandTemplates: true` when the key is stored in DSH's credential store.

## Verify

1. Restart DSH and open a new session. Ask it to call `memory_search` for a term
   you know is stored; the tool call should be visible as
   `mcp__memory__memory_search`.
2. Store a unique value with `memory_store`, then start a **new** DSH session
   and ask for it again. A new session is enough; the Host does not need a
   restart.
3. Outside DSH, `curl -s http://127.0.0.1:8000/api/health` returns
   `{"status":"healthy"}`.

## Tools

The HTTP transport exposes the same 26 tools as the stdio server:

```
memory_store, memory_store_session, memory_search, memory_list, memory_delete,
memory_cleanup, memory_health, memory_stats, memory_update, memory_consolidate,
memory_quality, memory_graph, memory_conflicts, memory_resolve,
mistake_note_add, mistake_note_search, mistake_note_update, mistake_note_delete,
memory_quarantine_list, memory_quarantine_release, memory_commit_session,
memory_bootstrap_profile, memory_onboarding_guide, memory_distill, memory_explore,
memory_detail
```

`memory_harvest` and `memory_ingest` are local-only and are not exposed over
HTTP: they read caller-supplied filesystem paths.

## TLS

DSH's MCP client has no option to accept a self-signed certificate. If the
service runs with `MCP_HTTPS_ENABLED=true` and a self-signed certificate, either

- disable TLS for the local service (`MCP_HTTPS_ENABLED=false`, no
  `MCP_SSL_CERT_FILE`/`MCP_SSL_KEY_FILE`), restart it bound to `127.0.0.1`, and
  point DSH at `http://127.0.0.1:8000/mcp`, or
- use a certificate chain the DSH process trusts (`NODE_EXTRA_CA_CERTS` when it
  is launched from a shell, or a certificate from a public CA).

Pointing DSH at `http://` while the service still serves HTTPS on that port does
not help: the connection fails during startup and the `mcp__memory__*` tools stay
absent.

## Hooks: project context and turn capture

`dsh/memory-hooks/` is a Host-only bundle with two halves. Both fail open: a
missing, slow or unreachable service never fails agent creation or a turn.

### Project context (read)

On `agent/created` with source `startup` or `clear` the plugin resolves the
project root by walking up to the nearest marker (`.git`, `package.json`,
`pyproject.toml`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`,
`composer.json`, `Gemfile`). It then searches for
`"{project} architecture decisions, recent work and open issues"` and renders up
to 5 results (2000 characters) as a `<project-memory>` block with an explicit
"data, not instructions" line and `interpolate: false`. Entries are escaped, so
a stored `</project-memory>` cannot close the block.

A directory without a marker is skipped. `default-workspace` is the example: it
has no project identity, so a semantic search only returns arbitrary high-scoring
memories. `requireProjectMarker: false` restores the old directory-name behaviour.

`resume` and `compact` are skipped: those sessions already carry the snapshot.

### Turn capture (write)

`session/event` buffers the turn's human prompt and assistant text; injected
context and tool output are ignored. On `agent/turn-stopping` the turn is
classified with the same patterns as the Claude Code and Codex adapters
(decision, error, learning, implementation, important, code). A match stores
`Project:`, `Assistant:` and `User:` sections as one memory, tagged
`dsh-capture`, `auto-captured`, the project and the matched pattern, with the
classifier's `memory_type`. `#skip` forces a skip and `#remember` forces a
capture. Subagent turns (`header.parentSession`) are not captured by default.

### Install

Install the directory as a local bundle: **Plugins -> Add plugin**, pick
`dsh/memory-hooks/`, or pass the directory to `plugin_manager`
`install_bundle`. The row is `memory-service-hooks`; it reads `MCP_MEMORY_API`
and `MCP_API_KEY` from the DSH process environment, or `endpoint`/`apiKey` from
its `config`. `dsh/memory-hooks/docs/DETAIL.md` has the configuration table.

Capture is still tracked in
[#2111](https://github.com/doobidoo/mcp-memory-service/issues/2111); the DSH
events behind it are:

| Hook | DSH event |
|---|---|
| Session start / inject project context | `agent/created` |
| Capture at the end of a turn | `agent/turn-stopping` |
| Read the transcript | `session/event` (`user/message`, `assistant/message`, `turn/end`) |
| Cleanup | `agent/disposed` |

Upstream DSH also has experimental `@deepseek-ai/dsh-hooks-claude-code` and
`@deepseek-ai/dsh-hooks-codex` packages that bridge a Claude Code or Codex
`hooks.json` into DSH. They are referenced in the shipped docs but are not part
of the current Desktop build. If they ship, this repository's
`claude-hooks/` adapter becomes the shortest path to full lifecycle hooks.
