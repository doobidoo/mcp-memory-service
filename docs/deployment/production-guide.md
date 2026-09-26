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

### Exposing the service through Cloudflare Tunnel

From a production deployment that puts the Docker containers behind a Cloudflare tunnel
with an auth gateway in front:

- [ ] **Cloudflare ZeroTrust with subnet-based access control** — allow only the addresses that need in: your own IPs plus the published ranges of every client you connect. Anthropic's for claude.ai, OpenAI's as well if you connect a ChatGPT connector; an allowlist built for one blocks the other.
- [ ] **Client IP Address Filtering on every Cloudflare API token** (Dashboard → My Profile → API Tokens → Edit → Client IP Address Filtering). It limits the damage if a token leaks.
- [ ] **IPv6 in the allowlist** — include your IPv6 /64 network. Python prefers IPv6 by default, so an IPv4-only allowlist silently blocks it.
- [ ] **`offline_access` scope for long-running browser sessions** — request it during authorization to get a rotating `refresh_token` (lifetime via `MCP_OAUTH_REFRESH_TOKEN_EXPIRE_DAYS`, default 30 days). Without it the access token is the only credential; extend `MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES` up to `1440` (24h) if you need longer single-shot sessions.
- [ ] **An auth proxy in front** — [AuthMCP Gateway](https://github.com/loglux/authmcp-gateway) or [mcp-auth-proxy](https://github.com/sigbit/mcp-auth-proxy) for session management, locally managed users and per-user server access control.

## After Deploy

- Verify health: `curl -fsS https://<your-host>/api/health`
- Verify mDNS (Linux, optional): `avahi-browse -t _mcp-memory._tcp`
- Confirm storage backend and version: `curl -fsS https://<your-host>/api/health/detailed | jq '.backend,.version'`

## Related

- [Multi-Client Setup](../integration/multi-client.md) — share memory across Claude Desktop, VS Code, OpenCode, etc.
- [Storage Backends](../guides/STORAGE_BACKENDS.md) — backend comparison + tuning
- [General Troubleshooting](../troubleshooting/general.md)
