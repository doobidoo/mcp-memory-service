# Team Memory — Claude Code Setup

The team runs a shared MCP Memory Service at:

| Endpoint | URL |
|---|---|
| Health check | `https://benites-memory.hive.letzdoo.com/api/health` |
| Dashboard (browser) | `https://benites-memory.hive.letzdoo.com/` |
| Doc ingestion (REST) | `https://benites-memory.hive.letzdoo.com/api/documents/batch-upload` |
| MCP transport (Claude Code) | `https://benites-memory.hive.letzdoo.com/mcp` |

Auth is a single shared Bearer token. **Get the token from the team password manager** (entry: *MCP Memory — shared API key*) and export it locally:

```bash
export MCP_API_KEY='paste-the-team-key-here'
```

---

## 1. Install `mcp-proxy` once

Claude Code talks to the remote service over Streamable HTTP via the `mcp-proxy` bridge.

```bash
pipx install mcp-proxy        # or: uv tool install mcp-proxy
```

## 2. Add the benites-memory MCP server

Pick the scope that fits your workflow.

### Option A — Per-project (recommended)

Create `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "benites-memory": {
      "command": "mcp-proxy",
      "args": [
        "https://benites-memory.hive.letzdoo.com/mcp",
        "--transport=streamablehttp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "paste-the-team-key-here"
      }
    }
  }
}
```

### Option B — User-scope (one-time, all projects)

```bash
claude mcp add benites-memory \
  --scope user \
  -- mcp-proxy https://benites-memory.hive.letzdoo.com/mcp --transport=streamablehttp
```

Then edit `~/.claude.json` and add `"API_ACCESS_TOKEN": "<the team key>"` to the `env` block of the `benites-memory` entry.

### Option C — Native HTTP transport (newer Claude Code builds)

If `claude mcp add --transport http` is available in your Claude Code version:

```bash
claude mcp add benites-memory \
  --scope user \
  --transport http \
  https://benites-memory.hive.letzdoo.com/mcp \
  --header "Authorization: Bearer ${MCP_API_KEY}"
```

This skips the `mcp-proxy` bridge entirely.

## 3. Install the `benites-memory` skill (recommended)

The skill teaches Claude Code **when** to use the memory server, the **tag conventions**, and the **troubleshooting tree** for connection / auth issues. Install it once into your user-scope skills directory:

```bash
mkdir -p ~/.claude/skills/benites-memory
cp docs/team/skills/benites-memory/SKILL.md ~/.claude/skills/benites-memory/SKILL.md
```

(Replace the `cp` source path with wherever you cloned this repo.)

After installing, restart Claude Code. The skill will load automatically and Claude will follow its conventions when it stores or recalls memories.

## 4. Verify

In a Claude Code session:

```
/mcp
```

You should see `benites-memory` as **connected** with tools like `store_memory`, `retrieve_memory`, `search_by_tag`, etc.

Quick smoke test — ask Claude:
> Recall any memories tagged `proj:team-docs`.

It should hit the shared service and return team docs (assuming someone has run the ingestion script — see § 6).

---

## 5. Tag your memories using namespaces (IMPORTANT)

The service uses a **structured tag taxonomy**. Tags should start with one of these prefixes so memories stay findable across the whole team:

| Namespace | Purpose | Examples |
|---|---|---|
| `proj:` | Project / repository / product | `proj:hive`, `proj:platform-lss`, `proj:wifaq` |
| `topic:` | Subject matter | `topic:auth`, `topic:rag-pipeline`, `topic:postgres` |
| `t:` | Time-based / sprints | `t:2026-04`, `t:sprint-12`, `t:q2-2026` |
| `user:` | Personal / per-developer notes | `user:jerome`, `user:onboarding`, `user:jerome-todo` |
| `agent:` | Agent identity (multi-agent setups) | `agent:planner`, `agent:reviewer`, `crew:research` |
| `q:` | Quality / curation level | `q:high`, `q:medium`, `q:draft` |
| `sys:` | System-generated (set by the service itself; don't use manually) | `sys:source_file=…`, `sys:upload_id=…` |

**Rules of thumb:**
- **Use `proj:<slug>` when the memory is tied to a specific project / repo / product.** Skip it for general, cross-cutting knowledge (team conventions, language tips, agentic patterns) — those are better tagged with `topic:` alone.
- **Stack multiple namespaces freely** — a memory can be tagged `proj:hive`, `topic:auth`, `t:2026-04`, `q:high` all at once.
- **Every tag must use a known namespace.** Legacy un-prefixed tags still load for backward compatibility but won't appear in namespace-aware searches — avoid them in new memories.
- **Tags are case-sensitive.** Stick with lowercase-kebab inside the value (`topic:rag-pipeline`, not `topic:RagPipeline`).

### How to tell Claude Code to use them

Add a short instruction to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for user-scope):

```markdown
## Memory tagging

When you store memories via the `benites-memory` MCP server, **every tag must use a known namespace**:
- `proj:<this-project's-slug>` — when the memory is project-specific
- `topic:<subject>` — always (e.g. `topic:auth`, `topic:postgres`, `topic:rag-pipeline`)
- `t:<YYYY-MM>` — for time-sensitive context
- `q:high` — only for vetted, durable knowledge
- `user:<who>` — for personal notes
- `agent:<name>` — for memories tied to a specific agent role

For **general cross-cutting knowledge** (team conventions, language tips, patterns), use `topic:` alone — `proj:` is not required.

Never use legacy (un-namespaced) tags — they won't be found by namespace queries.
```

Examples:

```
"Store this as a memory tagged proj:hive, topic:rag-pipeline, q:high: <fact>"

"Search memories tagged proj:hive AND topic:auth"

"Recall everything tagged user:jerome from last sprint (t:sprint-12)"
```

---

## 6. Ingest documentation (any dev can do this)

Use the helper script in this repo to bulk-upload markdown / PDF / Office docs:

```bash
export MCP_API_KEY='paste-the-team-key-here'

# Project-specific docs — include proj:
./scripts/team/ingest_docs.sh ~/repos/handbook --tags "proj:handbook,topic:onboarding"

# A single .docx, vetted as high-quality
./scripts/team/ingest_docs.sh ./architecture-overview.docx --tags "proj:platform,topic:architecture,q:high"

# General / cross-cutting knowledge — no proj: needed
./scripts/team/ingest_docs.sh ./guides --tags "topic:agentic-patterns,q:high"

# Preview what would be uploaded
./scripts/team/ingest_docs.sh ./docs --dry-run --tags "proj:foo,topic:bar"
```

Pass tags as a **comma-separated list of namespaced tags**. Include `proj:<slug>` for project-specific docs (optional for cross-cutting general knowledge); always include `topic:`. See § 5.

Supported file types: `.md`, `.markdown`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.xlsx`.

The script is **re-runnable** — the server deduplicates by content hash, so you can run it nightly via cron, or re-run after editing docs, without creating duplicates.

### What gets stored

Each file is chunked (~1000 chars, 200 overlap by default) and stored with:
- Your `--tags` (namespaced, see above)
- `memory_type` (`document` by default)
- System-generated `sys:` tags for source file, file type, upload ID
- Markdown structure preserved (headers, code blocks)

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `/mcp` shows memory as **failed** | Verify the API key: `curl -H "Authorization: Bearer $MCP_API_KEY" https://benites-memory.hive.letzdoo.com/api/health` should return 200 |
| `mcp-proxy: command not found` | Run `pipx install mcp-proxy` (or `uv tool install mcp-proxy`) and ensure `~/.local/bin` is on your `$PATH` |
| Tools work but no team docs come back | Someone needs to run `scripts/team/ingest_docs.sh` against your docs folder |
| 401 Unauthorized | Wrong API key — re-fetch from the team password manager |
| 413 Payload Too Large | Reduce `BATCH_SIZE` env var (default 10): `BATCH_SIZE=3 ./scripts/team/ingest_docs.sh …` |
