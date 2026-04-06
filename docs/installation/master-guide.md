# MCP Memory Service — Complete Setup Guide

This guide walks you through every step required for a fully working installation, from installing the package to verifying that memories persist across sessions. Follow the path that matches your use case.

---

## Step 0 — Pick Your Path

Answer two questions before you begin:

**1. How do you want to install it?**

| Method | When to use |
|--------|-------------|
| **PyPI install** (`pip install mcp-memory-service`) | Most users. No git clone needed, always gets the latest stable release. |
| **Source install** (git clone + `pip install -e .`) | Developers who want to modify the code, run the test suite, or contribute. |

**2. Which storage backend?**

| Backend | Best for |
|---------|----------|
| **SQLite** (default) | Single user, local machine, no cloud account needed. Fast, zero configuration. |
| **Cloudflare** | Multi-device sync, team use. Requires a Cloudflare account with D1 + Vectorize. |
| **Hybrid** (recommended for production) | Best of both: 5 ms local reads + background Cloudflare sync. |

If you are unsure, start with **SQLite + PyPI install**. You can migrate to Hybrid later without losing memories.

---

## Path A — PyPI Install (Recommended for Most Users)

### A1. Prerequisites

- Python 3.10–3.12
- **macOS**: use Homebrew Python (`brew install python`) — the system Python on macOS lacks loadable SQLite extensions, which are required for sqlite-vec.
- **Linux**: `sudo apt install python3-dev build-essential` (Ubuntu/Debian)
- **Windows**: Visual Studio Build Tools (C++ workload)

Verify your Python build supports SQLite extensions:

```bash
python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.enable_load_extension(True); print('OK')"
```

If this prints `OK`, continue. If it raises `AttributeError`, see [macOS SQLite Extension Issues](../first-time-setup.md#-macos-sqlite-extension-issues).

### A2. Install the Package

```bash
pip install mcp-memory-service
```

For all optional features (quality scoring, document ingestion, OAuth):

```bash
pip install "mcp-memory-service[full]"
```

### A3. Verify the Installation

```bash
memory server --help
```

You should see the available options. If the command is not found, ensure your Python `Scripts` (Windows) or `bin` (macOS/Linux) directory is on `PATH`.

---

## Path B — Source Install (Developers)

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
pip install -e ".[full]"
```

Verify:

```bash
python -m mcp_memory_service.server --help
```

For the rest of this guide, replace `memory server` with `python -m mcp_memory_service.server` if you are using a source install without the `memory` entry point on your PATH.

---

## Step 1 — Configure Storage

Skip this step if you are using the default SQLite backend — it requires zero configuration and auto-creates the database at `~/.mcp_memory_service/memories.db` on first run.

### SQLite (Default — No Configuration Needed)

Nothing to do. The database is created automatically on first use.

Optional tuning (add to `.env` or your shell profile):

```bash
# Required when running the HTTP dashboard alongside the MCP server
MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000,cache_size=20000
```

### Cloudflare

1. Create a Cloudflare API token at `https://dash.cloudflare.com/profile/api-tokens` with these permissions:
   - D1: Edit
   - Vectorize: Edit
   - Workers AI: Read

2. Create the Cloudflare resources (requires the `wrangler` CLI):

```bash
wrangler vectorize create mcp-memory-index --dimensions=384 --metric=cosine
wrangler d1 create mcp-memory-database
```

3. Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env   # source install
# or create .env manually for PyPI install
```

```dotenv
MCP_MEMORY_STORAGE_BACKEND=cloudflare
CLOUDFLARE_API_TOKEN=<your-token>
CLOUDFLARE_ACCOUNT_ID=<your-account-id>
CLOUDFLARE_D1_DATABASE_ID=<your-d1-database-id>
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index
```

### Hybrid (Local + Cloud Sync)

Hybrid mode uses SQLite for fast local reads and syncs to Cloudflare in the background. This is the recommended mode for production.

```dotenv
MCP_MEMORY_STORAGE_BACKEND=hybrid
CLOUDFLARE_API_TOKEN=<your-token>
CLOUDFLARE_ACCOUNT_ID=<your-account-id>
CLOUDFLARE_D1_DATABASE_ID=<your-d1-database-id>
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

# Recommended: let the HTTP server own the cloud sync
# The MCP server (used by Claude Desktop) then uses SQLite only,
# and does not need Cloudflare credentials in its environment.
MCP_HYBRID_SYNC_OWNER=http

# Required for concurrent HTTP dashboard + MCP server access
MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000,cache_size=20000
```

---

## Step 2 — Connect Your AI Client

Choose the section for the client you use. You can connect multiple clients to the same service.

### Claude Desktop

Add the following to your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Minimal configuration (SQLite, PyPI install):**

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

**With environment variables (Hybrid or Cloudflare):**

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory",
      "args": ["server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
        "CLOUDFLARE_API_TOKEN": "<your-token>",
        "CLOUDFLARE_ACCOUNT_ID": "<your-account-id>",
        "CLOUDFLARE_D1_DATABASE_ID": "<your-d1-database-id>",
        "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index"
      }
    }
  }
}
```

> **Tip for Hybrid mode with `MCP_HYBRID_SYNC_OWNER=http`**: Claude Desktop does not need Cloudflare credentials. Set only `MCP_MEMORY_STORAGE_BACKEND=hybrid` — the HTTP dashboard handles cloud sync.

Restart Claude Desktop after saving. The memory tools appear automatically in Claude's tool list.

### Claude Code (CLI)

```bash
claude mcp add memory -- memory server
```

Restart Claude Code. Memory tools will appear in the tool list.

For a source install, use the full module path:

```bash
claude mcp add memory -- python -m mcp_memory_service.server
```

### claude.ai (Browser — Remote MCP)

Claude.ai connects over HTTPS, so you need a publicly reachable URL with a valid TLS certificate.

**Quickest option — Cloudflare Tunnel (free, no certificate management):**

```bash
# 1. Install cloudflared
brew install cloudflared        # macOS
# sudo dpkg -i cloudflared.deb  # Ubuntu

# 2. Start the server in Remote MCP mode
MCP_STREAMABLE_HTTP_MODE=1 \
MCP_SSE_HOST=0.0.0.0 \
MCP_SSE_PORT=8765 \
MCP_OAUTH_ENABLED=true \
memory server

# 3. Open a tunnel (separate terminal)
cloudflared tunnel --url http://localhost:8765
# Note the printed URL, e.g. https://random-name.trycloudflare.com

# 4. In claude.ai: Settings → Connectors → Add Connector
# Enter: https://random-name.trycloudflare.com/mcp
```

For a permanent deployment (Let's Encrypt, nginx, Docker), see [Remote MCP Setup Guide](../remote-mcp-setup.md).

### Other MCP Clients (Cursor, Windsurf, VS Code, etc.)

Most MCP-compatible clients accept a server command in a JSON config. Use the same pattern as Claude Desktop:

```json
{
  "memory": {
    "command": "memory",
    "args": ["server"]
  }
}
```

Consult your client's documentation for the exact config file location.

### Agent Frameworks (LangGraph, CrewAI, AutoGen, HTTP)

Start the HTTP server and call the REST API directly — no MCP client library needed:

```bash
# Start the HTTP API server (separate from the MCP server)
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
# REST API running at http://localhost:8000
```

See [docs/agents/](../agents/) for framework-specific integration examples.

---

## Step 3 — Optional: Enable the Web Dashboard

The web dashboard is a separate HTTP server that runs alongside the MCP server. It provides a UI for browsing, searching, and managing memories.

```bash
# Start the dashboard server (requires source install or full package)
python scripts/server/run_http_server.py  # Requires repository clone
# Open http://localhost:8000
```

Or start both servers together:

```bash
./start_all_servers.sh
```

To secure the dashboard with an API key:

```dotenv
MCP_API_KEY=your-secret-key
```

For team access with OAuth 2.0:

```dotenv
MCP_OAUTH_ENABLED=true
MCP_OAUTH_STORAGE_BACKEND=sqlite
MCP_OAUTH_SQLITE_PATH=./data/oauth.db
```

Full OAuth setup: [docs/oauth-setup.md](../oauth-setup.md)

---

## Step 4 — Verify Your Setup

### Check the Server Starts

Run the server in a terminal and look for the ready message:

```bash
memory server
```

Expected output (first run will also download the embedding model, ~25 MB, one-time):

```
INFO: SQLite-vec storage initialized successfully with embedding dimension: 384
INFO: Ready to accept connections
```

First-run warnings like `No snapshots directory` and `TRANSFORMERS_CACHE is deprecated` are normal — see [First-Time Setup Guide](../first-time-setup.md) for details.

### Check the HTTP API (if dashboard is running)

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "healthy",
  "storage_backend": "sqlite_vec"
}
```

### Validate Configuration

Run the configuration validator for a detailed diagnostic:

```bash
python scripts/validation/validate_configuration_complete.py  # Requires repository clone
```

For storage-backend specific issues:

```bash
python scripts/validation/diagnose_backend_config.py  # Requires repository clone
```

---

## Troubleshooting Quick Reference

| Symptom | Fix |
|---------|-----|
| `AttributeError: enable_load_extension` on macOS | Use Homebrew Python: `brew install python` |
| `sqlite-vec` fails to install on Python 3.13 | Use Python 3.12: `brew install python@3.12` |
| "Storage initialization timed out" (Windows) | Add `MCP_INIT_TIMEOUT=120` to the MCP server env |
| "database is locked" when dashboard + MCP run together | Add `MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000` |
| Cloudflare 401 on MCP server startup (hybrid mode) | Set `MCP_HYBRID_SYNC_OWNER=http`; MCP server uses SQLite only |
| MCP tools not showing in Claude Desktop | Restart Claude Desktop; check config file path and JSON syntax |
| `memory: command not found` | Ensure pip's `bin` directory is on `PATH`; try `python -m mcp_memory_service.server` instead |
| Pre-commit hook "Package not installed" | Use `PATH=".venv/bin:$PATH" git commit -m "..."` |

Comprehensive troubleshooting: [docs/troubleshooting/general.md](../troubleshooting/general.md)

---

## What to Read Next

| Goal | Guide |
|------|-------|
| Understand storage backend trade-offs | [docs/guides/STORAGE_BACKENDS.md](../guides/STORAGE_BACKENDS.md) |
| Set up Cloudflare resources step by step | [docs/cloudflare-setup.md](../cloudflare-setup.md) |
| Deploy to production (systemd, Docker) | [docs/deployment/production-guide.md](../deployment/production-guide.md) |
| Expose the server for claude.ai | [docs/remote-mcp-setup.md](../remote-mcp-setup.md) |
| Configure OAuth for teams | [docs/oauth-setup.md](../oauth-setup.md) |
| Use agent frameworks (LangGraph, CrewAI) | [docs/agents/](../agents/) |
| Migrate between storage backends | [docs/guides/migration.md](../guides/migration.md) |
| Understand first-run warnings | [docs/first-time-setup.md](../first-time-setup.md) |
