---
name: benites-memory
description: How to use the team's shared MCP memory service (benites-memory). Tag conventions, recall patterns, recovery from connection / auth failures, and how to set the API key.
---

# benites-memory — Team Shared Memory

`benites-memory` is the team's **shared, persistent memory** behind `https://benites-memory.hive.letzdoo.com`. Use it instead of inventing your own ad-hoc note files. Anything stored here is visible to the whole team.

## When to Use

- The user asks to **remember**, **recall**, **search memory**, **what did we learn about**, **what was the decision on**, **find docs about**…
- You have just learned something **durable and team-relevant** (an architectural decision, a fix root cause, a convention) — proactively offer to store it.
- Before answering a project-knowledge question (best practices, conventions, prior decisions), **search memory first** so your answer reflects the team's actual context, not generic advice.

Don't store: ephemeral conversation state, secrets, temporary debugging output, raw tool output. Store the **insight**, not the trace.

## Tag Taxonomy (REQUIRED reading)

Every tag must use a known namespace. Stack multiple namespaces freely.

| Namespace | Purpose | Examples |
|---|---|---|
| `proj:` | Project / repo / product | `proj:hive`, `proj:wasfaty`, `proj:platform-lss` |
| `topic:` | Subject matter | `topic:auth`, `topic:rag-pipeline`, `topic:postgres` |
| `t:` | Time / sprint | `t:2026-04`, `t:sprint-12`, `t:q2-2026` |
| `q:` | Quality / curation | `q:high` (vetted, durable), `q:medium`, `q:draft` |
| `user:` | Personal / per-developer notes | `user:jerome`, `user:onboarding` |
| `agent:` | Agent identity | `agent:planner`, `agent:reviewer` |
| `sys:` | **Reserved for the service.** Don't write `sys:` tags yourself. | `sys:source_file=…`, `sys:upload_id=…` |

### Rules

1. **Use `proj:<slug>`** when the memory is tied to a specific project — and only then. For general knowledge (patterns, language tips, team conventions) use `topic:` alone.
2. **Always include a `topic:`** so the memory is discoverable by subject.
3. **Lowercase-kebab values** — `topic:rag-pipeline`, not `topic:RagPipeline`.
4. **No legacy un-namespaced tags.** They still load for backward compat but won't appear in namespace queries — avoid them.
5. **`q:high` is a curation signal** — only apply to memories you've verified are accurate and lasting.

## Standard Workflow

### Storing a memory

1. Decide tags (must include at least one `topic:`; add `proj:` if project-specific).
2. Call `mcp__benites-memory__store_memory` with `content` and `tags`.
3. Confirm to the user what was stored and with which tags.

```
mcp__benites-memory__store_memory({
  content: "ADR-003: admin reference data validation runs at the rules-engine entry, not in the API layer. Reason: the rules engine has the canonical schema and the API is treated as untrusted input.",
  tags: ["proj:wasfaty", "topic:adr", "topic:rules-engine", "topic:admin-portal", "t:2026-04", "q:high"]
})
```

### Recalling memories

Prefer **tag-filtered search** for precision; fall back to semantic search when tags are unknown.

```
# By tag (most precise)
mcp__benites-memory__search_by_tag({ tags: ["proj:wasfaty", "topic:adr"] })

# Semantic (when you don't know the tags)
mcp__benites-memory__retrieve_memory({ query: "admin reference data validation rules" })

# Time-bounded recall
mcp__benites-memory__recall_memory({ query: "auth changes from last sprint" })
```

### Proactive memory before answering

When a user asks about project conventions, prior decisions, or "how do we do X here":

```
1. mcp__benites-memory__search_by_tag({ tags: ["proj:<current-project>", "topic:<subject>"] })
2. If 0 results, broaden: search by topic alone, then semantic retrieve_memory.
3. Cite what you found (or note "no team memory on this yet — answering from general knowledge").
```

## Setup & Recovery

### One-time setup (each developer)

1. **Get the API key** from the team password manager (entry: *MCP Memory — shared API key*).
2. **Export it** so the ingestion script and any curl tests work:
   ```bash
   export MCP_API_KEY='paste-the-team-key-here'
   ```
   Add this to your shell rc (`~/.zshrc` / `~/.bashrc`) so it's persistent.
3. **Install `mcp-proxy`** once:
   ```bash
   pipx install mcp-proxy   # or: uv tool install mcp-proxy
   ```
4. **Register the MCP server** — see `docs/team/CLAUDE_CODE_SETUP.md` for the exact `.mcp.json` snippet (the server name MUST be `benites-memory`).

### Health check from the shell

```bash
curl -fsS -H "Authorization: Bearer $MCP_API_KEY" \
     https://benites-memory.hive.letzdoo.com/api/health
# Expected: {"status":"healthy"}
```

If this fails, no point trying Claude Code — the issue is upstream of MCP.

### Troubleshooting tree

| Symptom | First check | Fix |
|---|---|---|
| `/mcp` shows `benites-memory` as **failed** | `curl …/api/health` returns 200? | If yes: re-add the server (`claude mcp remove benites-memory && claude mcp add …`). If no: see next row. |
| `curl …/api/health` returns 401 | Is `MCP_API_KEY` set in your shell and correct? | Re-fetch from password manager; re-export; restart Claude Code |
| `curl …/api/health` returns 5xx or times out | `dig benites-memory.hive.letzdoo.com` resolves? Are you on the team VPN/network? | Get on-network; if still failing, ping #infra — service may be down |
| Tools work but searches return nothing | Are you searching with namespaced tags? Legacy tags don't match namespaced searches | Use `proj:`/`topic:` prefixes; ask a teammate if the area has been ingested |
| `mcp-proxy: command not found` | Is `~/.local/bin` on your `$PATH`? | `pipx install mcp-proxy` and add `~/.local/bin` to `$PATH` |
| Connection drops mid-session | Network blip — Claude Code retries with backoff | Run `/mcp` to force a reconnect; if it stays failed, restart Claude Code |
| Stored memory but can't find it later | Did you actually pass tags? Is the namespace prefix correct? | Search semantically with `retrieve_memory` to confirm it's there, then check tags |

### Reconnecting after a failure

In the Claude Code session:

```
/mcp
```

This re-handshakes with the MCP servers. If `benites-memory` still shows failed:

```bash
# From your terminal
claude mcp remove benites-memory --scope user
claude mcp add benites-memory \
  --scope user \
  -- mcp-proxy https://benites-memory.hive.letzdoo.com/mcp --transport=streamablehttp
# Re-add API_ACCESS_TOKEN to ~/.claude.json under the env block
```

Then restart Claude Code.

### Setting / rotating the API key

The key lives in **two places** for each developer:
1. **Shell env** (`MCP_API_KEY`) — used by curl, the ingestion script, ad-hoc testing.
2. **MCP server config** (`API_ACCESS_TOKEN` under the `env` block of the `benites-memory` entry) — used by Claude Code via `mcp-proxy`.

After a key rotation:
1. Pull the new key from the team password manager.
2. Update `~/.zshrc` / `~/.bashrc` and re-source.
3. Update `~/.claude.json` (or your project's `.mcp.json`) — replace the value of `API_ACCESS_TOKEN`.
4. Restart Claude Code.

## Quick Cheat-Sheet

```
STORE:    mcp__benites-memory__store_memory({content, tags: ["proj:X", "topic:Y", "q:high"]})
SEARCH:   mcp__benites-memory__search_by_tag({tags: ["proj:X", "topic:Y"]})
SEMANTIC: mcp__benites-memory__retrieve_memory({query: "..."})
RECALL:   mcp__benites-memory__recall_memory({query: "..."})  # natural language + time
HEALTH:   curl -H "Authorization: Bearer $MCP_API_KEY" https://benites-memory.hive.letzdoo.com/api/health
RECONNECT: /mcp inside Claude Code
INGEST:   ./scripts/team/ingest_docs.sh <path> --tags "proj:X,topic:Y"
```

## Resources

| URL | What you get |
|---|---|
| `https://benites-memory.hive.letzdoo.com/` | Browser dashboard — search and manage memories visually |
| `https://benites-memory.hive.letzdoo.com/api/health` | Service health check |
| `https://benites-memory.hive.letzdoo.com/api/documents/batch-upload` | REST endpoint behind `scripts/team/ingest_docs.sh` |
| `https://benites-memory.hive.letzdoo.com/mcp` | Streamable-HTTP MCP transport (this is what `mcp-proxy` connects to) |
