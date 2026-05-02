# Production Deployment Guide

Pick the deployment topology that fits your setup. All topologies use the same storage backends (`hybrid`, `cloudflare`, or `sqlite_vec`) and the same configuration via environment variables — they differ only in how the MCP and HTTP surfaces are exposed.

## Topology Selector

| Topology | When to use | Guide |
|----------|-------------|-------|
| **Docker** | Container-based deploys, CI, Kubernetes, fastest path to running | [docker.md](docker.md) |
| **Dual-Service (FastMCP + HTTP)** | Linux box serving both MCP clients and the web dashboard, no Node.js SSL bridge needed | [dual-service.md](dual-service.md) |
| **Single-Process systemd** | Linux box, one service, simplest operational footprint | [systemd-service.md](systemd-service.md) |
| **External Embeddings** | Offload embedding generation to vLLM / Ollama / TEI / OpenAI-compatible endpoint | [external-embeddings.md](external-embeddings.md) |

## Common Production Checklist

These apply regardless of topology:

- [ ] **Storage backend chosen and tested** — see [Storage Backends](../guides/STORAGE_BACKENDS.md). Hybrid is recommended for production (5 ms local reads + Cloudflare sync).
- [ ] **WAL mode enabled for SQLite-Vec / Hybrid** — set `MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000,cache_size=20000`. Without this, concurrent HTTP + MCP access causes "database is locked" errors.
- [ ] **API key set via environment** — never commit keys to docs or config. Example: `export MCP_API_KEY="$(openssl rand -hex 16 | sed 's/^/mcp-/')"` then put it in `.env` (gitignored) or your secrets manager.
- [ ] **OAuth storage backend** for production — `MCP_OAUTH_STORAGE_BACKEND=sqlite` (not `memory`), so tokens survive restarts. See [oauth-storage-backends.md](../oauth-storage-backends.md).
- [ ] **Health check wired up** — `curl -fsS https://your-host/api/health` from a monitoring system. Returns 200 + JSON when ready.
- [ ] **Backups configured** — for SQLite-Vec and Hybrid, snapshot the SQLite file regularly. For Cloudflare-only, exports go via the dashboard.
- [ ] **Hybrid sync owner pinned** (Hybrid only) — set `MCP_HYBRID_SYNC_OWNER=http` so only the HTTP server syncs to Cloudflare. The MCP server then runs SQLite-Vec only and needs no Cloudflare credentials.

## After Deploy

- Verify health: `curl -fsS https://<your-host>/api/health`
- Verify mDNS (Linux, optional): `avahi-browse -t _mcp-memory._tcp`
- Confirm storage backend and version: `curl -fsS https://<your-host>/api/health/detailed | jq '.backend,.version'`

## Related

- [Multi-Client Setup](../integration/multi-client.md) — share memory across Claude Desktop, VS Code, OpenCode, etc.
- [Storage Backends](../guides/STORAGE_BACKENDS.md) — backend comparison + tuning
- [General Troubleshooting](../troubleshooting/general.md)
