# Setup Guide

Not sure where to start? Use the decision tree below to find your path.

## Which path do you need?

```
What do you need?
│
├─ Connect Claude Desktop or Claude Code via MCP only?
│   └─ → Path 1: MCP Only (5 min)
│
├─ Use OpenCode with memory-aware local plugin?
│   └─ → Path 1A: OpenCode Plugin (5 min)
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

Each path builds on the previous one where noted. OpenCode users can start with Path 1A directly. If you want Path 3, follow 1 or 1A → 2 → 3 in order.

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
- [See what else it works with](README.md#works-with-your-favorite-ai-tools)
- Want OpenCode integration? → [Path 1A](#path-1a-opencode-plugin)
- Want the dashboard? → [Path 2](#path-2-mcp--http-dashboard)

---

## Path 1A: OpenCode Plugin

**For whom:** OpenCode users who want memory-aware sessions via a local plugin.

**Prerequisites:** Python 3.10–3.13, pip

**Duration:** ~5 min

### 1. Install

```bash
pip install mcp-memory-service
```

### 2. Start the HTTP API

```bash
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

> The current OpenCode integration uses the documented HTTP API, not direct MCP transport.
> `http://127.0.0.1:8000` is only the default. Set `memoryService.endpoint` or `OPENCODE_MEMORY_ENDPOINT` to use a remote/shared deployment.

### 3. Install the local plugin

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
mkdir -p ~/.config/opencode/plugins
cp opencode/memory-plugin.js ~/.config/opencode/plugins/
cp opencode/memory-plugin.config.example.json ~/.config/opencode/memory-plugin.json
```

OpenCode loads local plugins automatically from `~/.config/opencode/plugins/` and `.opencode/plugins/`.
See [opencode/README.md](../opencode/README.md) for configuration and current limitations.

> If you installed only the PyPI package, clone the repository once to copy the plugin files into your OpenCode config directory.

### 4. Verify

Start OpenCode inside a project that already has stored memories, then ask a project-specific question and confirm prior context is available.

### 5. Next Steps

- Want the dashboard too? → [Path 2](#path-2-mcp--http-dashboard)

## Path 2: MCP + HTTP Dashboard

**For whom:** Users who want a browser UI to browse, search, and manage memories, or need the REST API for scripting.

**Prerequisites:** Complete [Path 1](#path-1-mcp-only) first.

**Duration:** ~10 min total (from scratch)

### 1. Start the HTTP server

> **Note:** The HTTP dashboard requires the repository source code. If you installed via PyPI only, clone it first:
> ```bash
> git clone https://github.com/doobidoo/mcp-memory-service.git
> cd mcp-memory-service
> pip install -e .
> ```

```bash
memory launch
```

The server starts on port 8000 by default (in the background with PID tracking).

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

> The response may include additional fields such as `version`, `uptime`, and backend details.

### 5. Next Steps

- [REST API reference](https://github.com/doobidoo/mcp-memory-service/wiki)
- Want cloud sync? → [Path 3](#path-3-mcp--cloud-sync)

---

## Path 3: MCP + Cloud Sync

**For whom:** Users who want memories synced across multiple devices via Cloudflare (D1 + Vectorize).

**Prerequisites:** Complete [Path 2](#path-2-mcp--http-dashboard) first. Cloudflare account (free tier works).

**Duration:** ~20 min total (from scratch)

### 1. Create Cloudflare resources

Follow [docs/cloudflare-setup.md](cloudflare-setup.md) to create:
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
memory restart
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

See [docs/oauth-setup.md](oauth-setup.md) for full OAuth configuration including Dynamic Client Registration.

### 2. (Optional) Protect the DCR endpoint

If your server is internet-facing, set a registration key so only authorized clients can register:

```bash
MCP_DCR_REGISTRATION_KEY=your-secret-registration-key
```

### 3. (Optional) Docker deployment

See [docs/docker-optimized-build.md](docker-optimized-build.md) for a production Docker Compose setup.

### 4. Verify

```bash
python scripts/validation/validate_configuration_complete.py
```

### 5. Next Steps

- [Remote MCP setup (claude.ai browser)](remote-mcp-setup.md)
- [Architecture overview](architecture.md)
- [Troubleshooting](troubleshooting/)

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Memory tools don't appear in Claude | Restart the client after config change |
| `command not found: memory` | Run `pip install mcp-memory-service` again, check your PATH |
| Dashboard not loading | Check `memory info` (server should be running, default port 8000) |
| "database is locked" errors | Add `journal_mode=WAL` to `MCP_MEMORY_SQLITE_PRAGMAS` in `.env` |
| Cloudflare 401 on startup | Set `MCP_HYBRID_SYNC_OWNER=http` — MCP server then skips Cloudflare entirely |
| Slow first start (30–60s) | Normal — ONNX model loads on first run, cached afterwards |

For more: [docs/troubleshooting/](troubleshooting/)
