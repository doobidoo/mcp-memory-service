# Consolidated Setup Guide Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `docs/setup-guide.md` as the canonical setup entry point with a decision tree and 4 progressive paths, fix 3 broken README links, and update the outdated `docs/first-time-setup.md`.

**Architecture:** Pure documentation change. New file is the single source of truth for setup decisions. README links are updated to point here. `first-time-setup.md` is pruned of stale content (ChromaDB, wrong port, legacy installer). No code changes.

**Tech Stack:** Markdown, GitHub-flavored MD. No tests needed — verification is link checking + manual review of rendered output.

---

### Task 1: Create `docs/setup-guide.md`

**Files:**
- Create: `docs/setup-guide.md`

**Step 1: Create the file with decision tree and all 4 paths**

Write `docs/setup-guide.md` with the following content:

```markdown
# Setup Guide

Not sure where to start? Use the decision tree below to find your path.

## Which path do you need?

```
What do you need?
│
├─ Connect Claude Desktop or Claude Code via MCP only?
│   └─ → Path 1: MCP Only (5 min)
│
├─ Also want the web dashboard + REST API?
│   └─ → Path 2: MCP + HTTP Dashboard (10 min)
│
├─ Sync memories across devices via Cloudflare?
│   └─ → Path 3: MCP + Cloud Sync (20 min)
│
└─ Team deployment, OAuth, or full production setup?
    └─ → Path 4: Full Stack — Hybrid + Dashboard + OAuth (30 min)
```

Each path builds on the previous one. If you want Path 3, follow 1 → 2 → 3 in order.

---

## Path 1: MCP Only

**For whom:** Individual users who want Claude Desktop or Claude Code to remember things across sessions.

**Prerequisites:** Python 3.10–3.13, pip

**Duration:** ~5 min

### 1. Install

```bash
pip install mcp-memory-service
```

### 2. Configure your AI client

**Claude Desktop** — edit the config file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory",
      "args": ["server"]
    }
  }
}
```

**Claude Code** — run once:

```bash
claude mcp add memory -- memory server
```

### 3. Verify

Restart your AI client. Ask it: *"Do you have memory tools available?"* — it should confirm tools like `memory_store` and `memory_search` are present.

Or check directly:

```bash
memory server --help
```

### 4. Next Steps

- [Store your first memory](https://github.com/doobidoo/mcp-memory-service/wiki)
- [See what else it works with](../../README.md#-works-with-your-favorite-ai-tools)
- Want the dashboard? → [Path 2](#path-2-mcp--http-dashboard)

---

## Path 2: MCP + HTTP Dashboard

**For whom:** Users who want a browser UI to browse, search, and manage memories, or need the REST API for scripting.

**Prerequisites:** Complete [Path 1](#path-1-mcp-only) first.

**Duration:** ~10 min total (from scratch)

### 1. Start the HTTP server

```bash
python scripts/server/run_http_server.py
```

The server starts on port 8000 by default.

### 2. Open the dashboard

Visit [http://localhost:8000](http://localhost:8000) in your browser.

### 3. (Optional) Secure with an API key

Add to your `.env` file (create it in the project root if it doesn't exist):

```bash
MCP_API_KEY=your-secure-key-here
```

All REST API requests then require `Authorization: Bearer your-secure-key-here`.

### 4. Verify

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{"status": "healthy", "storage_backend": "sqlite_vec"}
```

### 5. Next Steps

- [REST API reference](https://github.com/doobidoo/mcp-memory-service/wiki)
- Want cloud sync? → [Path 3](#path-3-mcp--cloud-sync)

---

## Path 3: MCP + Cloud Sync

**For whom:** Users who want memories synced across multiple devices via Cloudflare (D1 + Vectorize).

**Prerequisites:** Complete [Path 2](#path-2-mcp--http-dashboard) first. Cloudflare account (free tier works).

**Duration:** ~20 min total (from scratch)

### 1. Create Cloudflare resources

Follow [docs/cloudflare-setup.md](../cloudflare-setup.md) to create:
- D1 database
- Vectorize index

Note down: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_D1_DATABASE_ID`, `CLOUDFLARE_VECTORIZE_INDEX`.

### 2. Configure `.env`

```bash
MCP_MEMORY_STORAGE_BACKEND=hybrid
MCP_HYBRID_SYNC_OWNER=http
MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000,cache_size=20000

CLOUDFLARE_API_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_D1_DATABASE_ID=your-db-id
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index
```

> **Why `MCP_HYBRID_SYNC_OWNER=http`?** This makes the HTTP server handle all Cloudflare sync. Your MCP server (Claude Desktop) uses local SQLite only — no Cloudflare credentials needed in `claude_desktop_config.json`.

### 3. Restart servers

```bash
./scripts/update_and_restart.sh
```

### 4. Verify

```bash
python scripts/validation/diagnose_backend_config.py
```

Expected: backend reported as `hybrid`, Cloudflare connection confirmed.

### 5. Next Steps

- Want team access + OAuth? → [Path 4](#path-4-full-stack)

---

## Path 4: Full Stack

**For whom:** Teams who need multi-user access, OAuth authentication, or a production-hardened deployment.

**Prerequisites:** Complete [Path 3](#path-3-mcp--cloud-sync) first.

**Duration:** ~30 min total (from scratch)

### 1. Enable OAuth

Add to `.env`:

```bash
MCP_OAUTH_ENABLED=true
MCP_OAUTH_STORAGE_BACKEND=sqlite
MCP_OAUTH_SQLITE_PATH=./data/oauth.db
```

See [docs/oauth-setup.md](../oauth-setup.md) for full OAuth configuration including Dynamic Client Registration.

### 2. (Optional) Protect the DCR endpoint

If your server is internet-facing, set a registration key so only authorized clients can register:

```bash
MCP_DCR_REGISTRATION_KEY=your-secret-registration-key
```

### 3. (Optional) Docker deployment

See [docs/docker-optimized-build.md](../docker-optimized-build.md) for a production Docker Compose setup.

### 4. Verify

```bash
python scripts/validation/validate_configuration_complete.py
```

### 5. Next Steps

- [Remote MCP setup (claude.ai browser)](../remote-mcp-setup.md)
- [Architecture overview](../architecture.md)
- [Troubleshooting](../troubleshooting/)

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Memory tools don't appear in Claude | Restart the client after config change |
| `command not found: memory` | Run `pip install mcp-memory-service` again, check your PATH |
| Dashboard not loading | Check `python scripts/server/run_http_server.py` is running, default port 8000 |
| "database is locked" errors | Add `journal_mode=WAL` to `MCP_MEMORY_SQLITE_PRAGMAS` in `.env` |
| Cloudflare 401 on startup | Set `MCP_HYBRID_SYNC_OWNER=http` — MCP server then skips Cloudflare entirely |
| Slow first start (30–60s) | Normal — ONNX model loads on first run, cached afterwards |

For more: [docs/troubleshooting/](../troubleshooting/)
```

**Step 2: Commit**

```bash
git add docs/setup-guide.md
git commit -m "docs: add consolidated setup guide with decision tree (issue #658)"
```

---

### Task 2: Update `docs/first-time-setup.md`

**Files:**
- Modify: `docs/first-time-setup.md`

**Step 1: Remove ChromaDB references**

The file currently mentions ChromaDB as a backend option in the Python 3.13 and macOS sections. ChromaDB is not a supported backend — remove all `--storage-backend chromadb` options and `install.py --storage-backend` references.

Specifically remove:
- "Option 2: Use ChromaDB Backend" block in the Python 3.13 section (lines ~105–108)
- "Option 3: Use ChromaDB Backend" block in the macOS SQLite section (lines ~156–159)

**Step 2: Fix the port**

Replace all occurrences of port `8443` with `8000`.

The health check command becomes:
```bash
curl http://127.0.0.1:8000/api/health
```

**Step 3: Remove the `install.py` workflow**

The Verification section references `python scripts/verify_environment.py` and `uv run memory server --debug`. Replace with:

```bash
# Verify installation
memory server --help

# Check health (requires HTTP server running)
curl http://localhost:8000/api/health
```

**Step 4: Update the "Next Steps" links at the bottom**

Replace:
```markdown
- [Configure Claude Desktop](../README.md#claude-desktop-integration)
- [Store your first memory](../README.md#basic-usage)
- [Explore the API](https://github.com/doobidoo/mcp-memory-service/wiki)
```

With:
```markdown
- [Setup Guide](setup-guide.md) — choose your path
- [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki) — full documentation
```

**Step 5: Commit**

```bash
git add docs/first-time-setup.md
git commit -m "docs: update first-time-setup — remove ChromaDB refs, fix port 8443→8000"
```

---

### Task 3: Fix broken links in `README.md`

**Files:**
- Modify: `README.md` (lines 225, 552, 555, 558)

**Step 1: Add setup guide link in "Get Started in 60 Seconds" (line 225)**

After the `## 🚀 Get Started in 60 Seconds` heading, add one line:

```markdown
> Not sure which setup fits your needs? See the **[Setup Guide](docs/setup-guide.md)** — a decision tree walks you to the right path in under a minute.
```

**Step 2: Fix the three broken links in the Documentation section (lines 552–558)**

Replace:
```markdown
- **[Installation Guide](docs/installation.md)** – Detailed setup instructions
```
With:
```markdown
- **[Setup Guide](docs/setup-guide.md)** – Decision tree + step-by-step paths for all use cases
```

Replace:
```markdown
- **[Team Setup Guide](docs/teams.md)** – OAuth and cloud collaboration
```
With:
```markdown
- **[Team Setup Guide](docs/setup-guide.md#path-4-full-stack)** – OAuth and cloud collaboration
```

Replace:
```markdown
- **[API Reference](docs/api.md)** – Programmatic usage
```
With:
```markdown
- **[API Reference](https://github.com/doobidoo/mcp-memory-service/wiki)** – Programmatic usage
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: fix 3 broken doc links, add setup guide pointer in Get Started"
```

---

### Task 4: Push branch and create PR

**Step 1: Push to origin**

```bash
git push https://github.com/doobidoo/mcp-memory-service.git main
```

Note: SSH may be down — use HTTPS push.

**Step 2: Verify**

Check that:
- https://github.com/doobidoo/mcp-memory-service/blob/main/docs/setup-guide.md renders correctly
- The three previously broken README links now resolve
- Issue #658 can be closed (add "Closes #658" to the commit or PR body)

**Step 3: Close issue #658**

```bash
gh issue close 658 --comment "Resolved in the just-merged setup guide (docs/setup-guide.md). The new guide has a decision tree at the top that leads to 4 clearly separated paths (MCP only → Dashboard → Cloud Sync → Full Stack). Broken README links are also fixed. Let us know if anything is still unclear!"
```
