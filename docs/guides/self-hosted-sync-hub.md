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

Start it with an entry point that honours `MCP_HTTPS_ENABLED`:

- `uv run memory launch` (add `--foreground` to stay attached): resolves the certificate
  paths and passes them to uvicorn (`cli/lifecycle.py`, `_resolve_server_tls`), and refuses
  to come up on plain HTTP when TLS was requested but cannot be provided;
- `python scripts/server/run_http_server.py`, which the systemd unit
  `scripts/service/mcp-memory-http.service` runs, does the same;
- or Docker — see the [production guide](../deployment/production-guide.md) and
  [docker.md](../deployment/docker.md).

**Do not start the hub with `memory server --http`.** That command passes no TLS arguments to
uvicorn at all (`cli/main.py`, `server`), so it serves plain HTTP whatever `MCP_HTTPS_ENABLED`
says, and clients configured with `https://hub.example.com:8443` cannot connect. With
`MCP_HTTPS_ENABLED=true` and no `MCP_SSL_CERT_FILE` / `MCP_SSL_KEY_FILE`, both entry points
above generate a self-signed development certificate; the hybrid client verifies the hub's
certificate against the system trust store and has no switch to accept a self-signed one, so
for a hub that other machines connect to, either point the two variables at a certificate the
clients trust, or leave `MCP_HTTPS_ENABLED` unset and terminate TLS in nginx/caddy in front.

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
```

Optionally, restrict who may push new events:

```bash
export MCP_SYNC_PUSH_ALLOWED_AGENTS=laptop,desktop   # the clients' MCP_AGENT_ID values
```

A pushed batch that contains a new event from an agent outside the list is rejected whole
(403), and nothing in it is applied. Events the hub already holds are exempt, because clients
push back the events they pulled, including the hub's own, which are stamped `unknown`. So
every client needs its `MCP_AGENT_ID` set and listed here. A client without it stamps its
events with the memory's own `agent_id` metadata or `unknown`, and those events get rejected.

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

**The job only exists in a process that starts the scheduler — on a client, that is the HTTP
server.** The HTTP server (`memory launch`, `run_http_server.py`) starts the scheduler as soon
as `MCP_SYNC_SCHEDULE` is set, even with consolidation off (`web/app.py`, lifespan). The stdio
`memory server` that Claude Desktop / Claude Code spawn starts it only when
`MCP_CONSOLIDATION_ENABLED=true` and at least one consolidation cadence is not `disabled`
(`server_impl.py`, `_initialize_consolidation`); with consolidation off, the stdio process
never creates the job and the settings above do nothing. So a client that only runs the stdio
server must also run the HTTP process with these variables, e.g.

```bash
# on the client, same environment as above
uv run memory launch            # background HTTP process that owns the pull+push job
```

and, since the client now runs two processes on one database, set `MCP_HYBRID_SYNC_OWNER`
(see above) so only one of them keeps the hybrid queue to the hub.

## Multi-client semantics

The two layers give different guarantees. The hybrid queue alone is a one-way backup to the hub,
and only delta-sync makes the clients converge.

### Hybrid queue only

- **Other clients see a write only after their own restart.** A client pulls from the hub in
  its startup sync (`MCP_HYBRID_SYNC_ON_STARTUP`, default on) or when `POST /api/sync/force`
  is called. The periodic drift check is skipped for an HTTP secondary (`storage/hybrid.py`,
  the `supports_delete_operations` guard), so `MCP_HYBRID_SYNC_UPDATES` and
  `MCP_HYBRID_DRIFT_CHECK_INTERVAL` have no effect here. `MCP_HYBRID_SYNC_INTERVAL` only
  paces health probes and retries.
- **The startup sync adds missing memories, nothing else.** It compares content hashes, so
  updates and deletes that another client made never reach a client that already holds the
  memory. Each client keeps its own version.
- **Deletes stay local to the hub.** The hub soft-deletes the row, and every other client
  keeps its copy. `POST /api/sync/force` on such a client pushes its whole database and
  re-creates the deleted memory on the hub.
- **Identity is the content hash.** If two clients store the same content, the hub keeps the
  first one. It rejects the second as a duplicate, and that client's tags and metadata are lost.
  Near-identical content inside the semantic-dedup window (24 h by default) can be rejected the
  same way, because the hub's `POST /api/memories` runs semantic deduplication.
- **Updates land in arrival order, with no timestamp comparison.** Tags and `memory_type` are
  replaced, while metadata is merged key by key, so the last arrival wins.
- **The queue lives in memory.** A normal shutdown tries to send what is left
  (`BackgroundSyncService.stop`). A crash, a forced kill, or a failed send during shutdown
  leaves those operations unsynced.

### With delta-sync

- **Writes reach the other clients on their next sync.** A client's queued write creates an
  event on the hub (with `MCP_SYNC_EVENTLOG=on` there) within seconds, and every other client
  pulls it at its next `MCP_SYNC_SCHEDULE` tick. Updates and deletes arrive the same way.
- **Conflicts resolve by hybrid logical clock.** Every event carries an HLC (wall-clock
  milliseconds plus a counter). The later HLC wins. On a tie, a delete wins, then the agent id
  and event id decide (`storage/sync/resolver.py`, ADR-0010 to 0013). Metadata updates are
  merged field by field, with the later HLC winning per field. A node whose clock runs ahead
  wins concurrent conflicts, so keep the clocks synced (NTP).
- **A delete wins only against older events.** A create with a later HLC than the delete
  brings the memory back. A replayed old create under a new identity does not. ADR-0012
  intends tombstones to block re-creation, but the apply step does not enforce that yet
  (#1522).
- **Events are never compacted.** `sync_events` keeps every event, on the clients and on the
  hub. A client hard-deletes soft-deleted memory rows after 30 days, while the delete event
  itself stays.
- **One write produces two events on the hub:** one from the hybrid queue and one from the
  client's push. The resolver keeps a single row per content hash, but the two events carry
  different clocks. The queued REST request gets the hub's clock when it arrives, while the
  pushed event keeps the client's. A REST request that arrives late can therefore outrank a
  newer change from another client and decide which version wins.
- **A new client starts from the event log, not from a snapshot.** Its first pull covers the
  events since the hub turned the event log on. Memories that are older than that arrive
  through the hybrid startup sync. The baseline bootstrap from Phase 5 has no command that
  runs it yet (#1522).
- **Memories from a different embedding model stay invisible.** A pulled create stamped with
  another `embedding_model` is stored without a vector, and search and tag queries skip it.
  Nothing clears that state later, even after the models match again (#1522). This is one
  more reason for one model everywhere.

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
  **On a database that has the `store` column, this script currently writes vectors without
  their store** (it inserts only `rowid` and `content_embedding` into `memory_embeddings`,
  where `store` is the partition key), so store-scoped searches miss them. Until that is
  fixed (#1521), use the export / re-import route below for a same-dimension change too,
  with the limits listed there.

- **Different dimension** (e.g. 384 → 768): `scripts/maintenance/migrate_embeddings.py` rebuilds
  the vector table, but it fetches embeddings from an **OpenAI-compatible HTTP API**
  (`--url` and `--model` are required). It has no path for a hub that embeds with a local
  sentence-transformers model, so it is a fit only if you move the hub to an external
  embedding server (see [external embeddings](../deployment/external-embeddings.md)). For a
  local model with a new dimension, export the memories
  (`python scripts/sync/export_memories.py --db-path <the hub's sqlite_vec.db>`; the export
  only reads rows, so it opens the database whatever model it was built with), move the old
  database aside, set the new `MCP_EMBEDDING_MODEL`, and re-import
  (`python scripts/sync/import_memories.py <export.json> --db-path <new sqlite_vec.db>`):
  the import builds the fresh database at the new model's dimension and embeds every memory
  with it. Both scripts take the model from `MCP_EMBEDDING_MODEL`, as the server does.

  The export walks every store and writes each memory's `store` into the file, and the import
  writes each memory back to the store it came from. The import skips semantic deduplication
  (an export holds memories the source already accepted; the exact-hash check stays) and counts
  a memory as imported only when the storage accepted it: whatever the storage refused is
  listed under "Rejected by storage", with the reason. Compare that line, and the memory count
  of the new database, with the export before you delete the old one.

**Re-embedding the hub is not enough: every client holds its own vectors.** A hybrid client
answers searches from its local SQLite-vec database, not from the hub (`storage/hybrid.py`,
`retrieve` / `search` delegate to the primary). After a model change, a client's local
vectors are still the old model's:

- **Same dimension**: startup accepts the old vectors (the existing-database check compares
  dimensions only, `storage/mixins/migrations.py`), so the client silently compares new query
  vectors with old stored ones and misses memories. Re-embed each client as you did the hub:
  stop its processes, set the new `MCP_EMBEDDING_MODEL`, run
  `python scripts/maintenance/regenerate_embeddings.py` with the client's own environment
  (on a hybrid it re-embeds the local primary, `_resolve_actual_storage`; its startup includes
  the model check against the hub, so do it after the hub is back up with the new model), then
  restart.
- **Different dimension**: the client's local database refuses to start (`RuntimeError`,
  strict dimension check). Either export / re-import it as for the hub, or rebuild it from the
  hub: move the local database aside and start the client, and the hybrid initial sync pulls
  every memory it is missing from the hub and stores it locally, which embeds it with the
  client's new model. **A rebuild only restores what reached the hub.** A hybrid write succeeds
  locally first and is queued for the hub afterwards (`HybridMemoryStorage.store`); a request
  that failed stays in the client's failed-operations list, so a client can hold memories the
  hub never got. Before moving the database aside, with the client still on the old model,
  check `GET /api/sync/status` on the client's HTTP process (or `get_sync_status()`): both
  `pending_operations` and `failed_operations` must be zero — if they are not, wait for the
  queue to drain or export the local database first with
  `python scripts/sync/export_memories.py --db-path <the client's sqlite_vec.db>` (it opens the
  database whatever model built it, so it works before or after you switch
  `MCP_EMBEDDING_MODEL`). Keep the old database file until the rebuilt client reports the
  memory count you expect.

Order: stop everything → re-embed the hub → start the hub → re-embed (or rebuild) each client →
start the clients.

## Checklist

- [ ] Hub: `sqlite_vec`, `MCP_API_KEY` set, TLS, same `MCP_EMBEDDING_MODEL` as clients.
- [ ] Hub does **not** run hybrid and does **not** allow anonymous access.
- [ ] Clients: `hybrid` + `MCP_HYBRID_SECONDARY_BACKEND=http` + URL + key; no `CLOUDFLARE_*`.
- [ ] Dual-process clients: `MCP_HYBRID_SYNC_OWNER` set to one of `http` / `mcp`.
- [ ] Delta-sync, if wanted: `MCP_SYNC_EVENTLOG=on` on hub and clients, `MCP_SYNC_*` peer
      variables on clients only, `MCP_SYNC_PEER_API_KEY` ≠ the client's own `MCP_API_KEY`.
- [ ] Model change: re-embed the hub (`regenerate_embeddings.py` for same dimension), then
      re-embed or rebuild **every client's local database** before restarting it.
- [ ] Hub started with `memory launch` or `run_http_server.py`, never `memory server --http`
      (it ignores `MCP_HTTPS_ENABLED`).
- [ ] Scheduled sync on a client: the HTTP process runs there too (stdio alone never schedules it).
