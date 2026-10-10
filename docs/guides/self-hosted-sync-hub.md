# Self-Hosted Sync Hub

Run your own MCP Memory Service as the "cloud" side of the hybrid backend, instead of Cloudflare.
Every client keeps its fast local SQLite-vec database and syncs in the background to one
server you control — the **hub**.

This page covers the hub itself: what it runs, how clients authenticate to it, how to re-embed it.
The client-side variables are summarised here and detailed in
[STORAGE_BACKENDS.md](STORAGE_BACKENDS.md#http-secondary-backend-alternative-to-cloudflare) and the
[configuration guide](../mastery/configuration-guide.md#hybrid-http-secondary-backend).

Tracking issue: [#1482](https://github.com/doobidoo/mcp-memory-service/issues/1482).

## Topology

```
 laptop  (hybrid, secondary=http) ──┐
 desktop (hybrid, secondary=http) ──┼──► hub (sqlite_vec, HTTP server on :8443)
 server  (hybrid, secondary=http) ──┘
```

- **Clients** run `MCP_MEMORY_STORAGE_BACKEND=hybrid` with `MCP_HYBRID_SECONDARY_BACKEND=http`.
  Reads are served locally; writes go to SQLite-vec first and are queued to the hub.
- **The hub is terminal.** It runs `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec` and syncs nowhere.
  Nothing in the code stops you from starting the hub as `hybrid` with another secondary, but
  then every memory a client pushes is forwarded again, and a hub whose own secondary is
  unreachable logs sync failures that have nothing to do with your clients. Keep it plain.
- All clients and the hub must use the **same embedding model** (`MCP_EMBEDDING_MODEL`); a
  client refuses to start otherwise (see [Re-embedding the hub](#re-embedding-the-hub)).

## Hub setup

Minimal environment for the hub process (HTTP server mode):

```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_HTTP_ENABLED=true
export MCP_HTTP_HOST=0.0.0.0
export MCP_HTTP_PORT=8443
export MCP_HTTPS_ENABLED=true            # or terminate TLS in nginx/caddy in front
export MCP_API_KEY="$(openssl rand -base64 32)"
export MCP_EMBEDDING_MODEL=all-MiniLM-L6-v2   # identical on every client
```

Start it the way you start any HTTP deployment (`uv run memory server --http`, the systemd
unit from `scripts/service/`, or Docker — see the [production guide](../deployment/production-guide.md) and [docker.md](../deployment/docker.md)).

### One key, full access

`MCP_API_KEY` is a **single key**, and a request that presents it gets the scope
`read write admin` (`web/oauth/middleware.py`, `authenticate_api_key`). There is no read-only
API key: every client that can sync can also delete on the hub, and can call the admin
endpoints. Treat the hub key like a database password — one shared secret for a set of
machines you trust equally, rotated by changing it on the hub and on every client.

Do **not** set `MCP_ALLOW_ANONYMOUS_ACCESS=true` on a hub that is reachable from outside your
network: anonymous callers get `read write`.

### Endpoints a client calls

Scope is what the middleware demands; the key above satisfies all of them.

| Client action | Hub endpoint | Scope |
|---|---|---|
| Startup embedding-model check | `GET /api/health/model` | read |
| Stats, existence checks | `GET /api/memories?page=…`, `GET /api/memories/{hash}`, `GET /api/memories/hashes` | read |
| Store / update / delete | `POST /api/memories`, `PUT /api/memories/{hash}`, `DELETE /api/memories/{hash}` | write |
| Delta-sync pull | `GET /api/sync/events`, `GET /api/sync/baseline` | read |
| Delta-sync push | `POST /api/sync/events` | write (+ agent allow-list, below) |

The client sends the key as `Authorization: Bearer <key>` by default, or as `X-API-Key` when
`MCP_HYBRID_SECONDARY_AUTH_STYLE=x-api-key` — use that when the hub sits behind nginx
`auth_basic`, because the `Authorization` header is then taken by nginx, and give the client
the nginx credentials via `MCP_HYBRID_SECONDARY_BASIC_USER` / `_PASS`.

### `MCP_HYBRID_SYNC_OWNER` still applies

`MCP_HYBRID_SYNC_OWNER` is read for any hybrid client, whatever its secondary
(`config/storage.py`; the override lives in `storage/factory.py` and only looks at
`STORAGE_BACKEND == 'hybrid'` and the owner value). On a client that runs both the MCP server
and the HTTP dashboard, set it to `http` or `mcp` so only one of the two processes keeps a
sync queue to the hub; the other falls back to plain SQLite-vec on the same database. Leave it
at `both` (default) on single-process clients. It is meaningless on the hub, which is not hybrid.

## Client setup

```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_HYBRID_SECONDARY_BACKEND=http
export MCP_HYBRID_SECONDARY_URL=https://hub.example.com:8443
export MCP_HYBRID_SECONDARY_API_KEY="<the hub's MCP_API_KEY>"
export MCP_EMBEDDING_MODEL=all-MiniLM-L6-v2        # same as the hub
# optional: tune MCP_HYBRID_SYNC_INTERVAL / MCP_HYBRID_BATCH_SIZE as for Cloudflare
```

None of the `CLOUDFLARE_*` variables are needed. After startup, `get_stats()` (and the
dashboard) report `Hybrid (SQLite-vec + HTTP)` with `secondary_backend: HTTP`.

## Delta-sync between clients (optional)

The hybrid queue above is one-way: client → hub. To also **pull** what other clients stored,
and to keep deletes consistent, enable the event log and the scheduled sync job.

On the **hub** (it must record events for clients to pull):

```bash
export MCP_SYNC_EVENTLOG=on
# optional: only accept pushed batches from these agent ids (whole batch rejected otherwise)
export MCP_SYNC_PUSH_ALLOWED_AGENTS=laptop,desktop
```

On each **client**:

```bash
export MCP_SYNC_EVENTLOG=on
export MCP_AGENT_ID=laptop                 # stamped on this node's events
export MCP_SYNC_SCHEDULE=15m               # pull+push every 15 minutes (also 6h, 90s …)
export MCP_SYNC_PEERS=hub
export MCP_SYNC_PEER_URL=https://hub.example.com:8443
export MCP_SYNC_PEER_API_KEY="<the hub's MCP_API_KEY>"
# behind nginx auth_basic:
# export MCP_SYNC_BASIC_USER=… ; export MCP_SYNC_BASIC_PASS=…
```

**`MCP_SYNC_PEER_API_KEY` is deliberately not `MCP_API_KEY`.** `MCP_API_KEY` on a process is
the key *this* server demands from its own callers; setting it on a client would turn on
authentication for that client's own `/mcp` endpoint and break its local Claude Desktop /
Claude Code connection. The peer key names the hub's key without touching the client's own
auth (`consolidation/scheduler.py`, `_resolve_sync_peers`).

`MCP_SYNC_SCHEDULE` unset or `disabled` means no job; it is opt-in on purpose
(`web/app.py`, `_OPTIN_SCHEDULE_ENV_VARS`). The hub sets no `MCP_SYNC_*` peer variables: it
is pulled from, never pulls.

## Multi-client semantics

<!-- Section to be written by @filhocf (Claudio) — see #1482: concurrent writers, conflict
     resolution between clients, delete propagation, what a client sees after another client's
     write. -->

## Re-embedding the hub

Changing `MCP_EMBEDDING_MODEL` means every stored vector on the hub is stale, and every client
will refuse to start until the hub's model matches theirs again
([troubleshooting](../mastery/troubleshooting.md#http-secondary-refuses-to-start-embedding-model-mismatch)).
Two scripts exist; pick by **vector dimension**, not by preference:

- **Same dimension** (e.g. `all-MiniLM-L6-v2` → another 384-dim sentence-transformers model):
  set the new `MCP_EMBEDDING_MODEL` on the hub, stop the service, run
  `python scripts/maintenance/regenerate_embeddings.py`. It takes no arguments and re-embeds
  every memory with the model the storage itself loads, so it uses exactly what the hub will
  serve with afterwards. Then set the same model on every client and restart them.

- **Different dimension** (e.g. 384 → 768): `scripts/maintenance/migrate_embeddings.py` rebuilds
  the vector table, but it fetches embeddings from an **OpenAI-compatible HTTP API**
  (`--url` and `--model` are required). It has no path for a hub that embeds with a local
  sentence-transformers model, so it is a fit only if you move the hub to an external
  embedding server (see [external embeddings](../deployment/external-embeddings.md)). For a
  local model with a new dimension, export the memories, start a fresh database with the new
  model, and re-import.

Either way, stop the hub while re-embedding and let clients re-run their startup model check
afterwards.

## Checklist

- [ ] Hub: `sqlite_vec`, `MCP_API_KEY` set, TLS, same `MCP_EMBEDDING_MODEL` as clients.
- [ ] Hub does **not** run hybrid and does **not** allow anonymous access.
- [ ] Clients: `hybrid` + `MCP_HYBRID_SECONDARY_BACKEND=http` + URL + key; no `CLOUDFLARE_*`.
- [ ] Dual-process clients: `MCP_HYBRID_SYNC_OWNER` set to one of `http` / `mcp`.
- [ ] Delta-sync, if wanted: `MCP_SYNC_EVENTLOG=on` on hub and clients, `MCP_SYNC_*` peer
      variables on clients only, `MCP_SYNC_PEER_API_KEY` ≠ the client's own `MCP_API_KEY`.
- [ ] Model change: re-embed the hub (`regenerate_embeddings.py` for same dimension) before
      restarting clients.
