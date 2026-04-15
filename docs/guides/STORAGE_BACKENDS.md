# Storage Backend Comparison and Selection Guide

**MCP Memory Service** supports three storage backends, each optimized for different deployment patterns — single-user development, cloud-only edge deployments, and production use with local reads and cloud durability.

> **Looking for the legacy backend?** The previous vector-database backend was removed in v7.x. See the historical migration guide under `docs/guides/` for the upgrade path.

## Quick Comparison

| Feature | SQLite-vec | Cloudflare | Hybrid |
|---------|------------|------------|--------|
| **Read latency** | ~5ms (local) | Network-dependent | ~5ms (local reads) |
| **Write latency** | Local disk | Network round-trip | Local disk + async cloud sync |
| **Persistence** | Single file on disk | D1 + Vectorize + optional R2 | Local SQLite + Cloudflare mirror |
| **Setup complexity** | Minimal — no credentials | Requires Cloudflare account + API token | Requires Cloudflare credentials |
| **Best for** | Development, single-user | Edge / cloud-only deployments | **Production (recommended default)** |
| **Offline operation** | Yes | No | Yes (syncs when online) |
| **Multi-device sync** | No (file-based) | Yes (via Cloudflare) | Yes (via Cloudflare background sync) |
| **Scale ceiling** | ~100K+ memories | Vectorize-limited (millions) | Same as Cloudflare |
| **External embedding APIs** | Supported (Ollama, vLLM, TEI, OpenAI-compatible) | Not supported | Not supported |

All three backends implement the same `BaseStorage` interface and expose identical MCP tools and REST endpoints. The choice is about where data lives, not what you can do with it.

## When to Choose SQLite-vec

### Ideal For:
- **Local development** and rapid iteration
- **Single-user deployments** where cloud sync isn't needed
- **Air-gapped or offline** environments
- **External embedding APIs** — the only backend compatible with `MCP_EXTERNAL_EMBEDDING_URL` (Ollama, vLLM, TEI, OpenAI-compatible endpoints)
- **Portable deployments** — the entire store is a single `.db` file you can copy, back up, or move

### Technical Characteristics:
- Uses the `sqlite-vec` extension for KNN semantic search
- ONNX embeddings (sentence-transformers/all-MiniLM-L6-v2) by default — no PyTorch required
- ACID transactions via SQLite
- ~5ms read latency on typical hardware
- Supports WAL mode for concurrent HTTP + MCP server access

### Example:
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
pip install -e ".[sqlite]"
```

## When to Choose Cloudflare

### Ideal For:
- **Cloud-only deployments** where no local storage is available (Workers, serverless runtimes)
- **Edge-deployed agents** running close to Cloudflare infrastructure
- **Shared team stores** where multiple clients read/write the same source of truth without a local cache

### Technical Characteristics:
- **D1** (Cloudflare's managed SQLite) stores memory rows and metadata
- **Vectorize** stores and queries embeddings (KNN)
- **R2** (optional) for large content payloads above the configured threshold
- Cloudflare-hosted embedding models (e.g. `@cf/baai/bge-base-en-v1.5`)
- Performance is network-dependent — every read is a round-trip

### Example:
```bash
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
pip install -e ".[full]"
```

## When to Choose Hybrid (Recommended)

### Ideal For:
- **Production deployments** — this is the recommended default
- **Multi-device setups** — Claude Desktop on laptop, HTTP dashboard on a server, all reading consistent state
- **Any use case that wants local-read speed + cloud durability**

### Technical Characteristics:
- Local SQLite-vec handles all reads → ~5ms latency, same as pure SQLite-vec
- Background worker syncs writes to Cloudflare (D1 + Vectorize) on a configurable interval
- Survives Cloudflare outages (keeps serving local reads) and local disk loss (can restore from Cloudflare)
- Default configuration: sync every 5 minutes, batch size 50

### Recommended Configuration:
Set `MCP_HYBRID_SYNC_OWNER=http` so that **only the HTTP server** performs Cloudflare sync. The MCP server (Claude Desktop) then runs in pure SQLite-vec mode and doesn't need Cloudflare credentials in `claude_desktop_config.json`. This is the correct separation of concerns: Claude Desktop = memory access, HTTP server = sync infrastructure.

### Example:
```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_HYBRID_SYNC_OWNER=http
pip install -e ".[full]"
```

## Deployment Compatibility Matrix

The right backend is driven by connectivity, privacy, and scale — not by hardware. Use this matrix to pick based on deployment pattern:

### Single-user laptop, offline-capable
```
Recommended: sqlite_vec (or hybrid if you also want cloud backup)
Why:         No network dependency; single-file portability.
Config:
  MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
```

### Production workstation with cloud backup
```
Recommended: hybrid
Why:         Local 5ms reads + Cloudflare durability. Best of both.
Config:
  MCP_MEMORY_STORAGE_BACKEND=hybrid
  MCP_HYBRID_SYNC_OWNER=http
```

### Team collaboration (multiple clients, shared state)
```
Recommended: hybrid (on each client) OR cloudflare (if no local disk)
Why:         Hybrid keeps each client fast; Cloudflare is the shared ground truth.
```

### Serverless / edge (Cloudflare Workers, no local FS)
```
Recommended: cloudflare
Why:         Only backend that doesn't require persistent local storage.
Config:
  MCP_MEMORY_STORAGE_BACKEND=cloudflare
```

### Air-gapped / high-privacy environment
```
Recommended: sqlite_vec
Why:         No network egress; all data stays on-host.
Config:
  MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
```

### Using an external embedding server (Ollama, vLLM, TEI)
```
Recommended: sqlite_vec (required)
Why:         External embedding APIs are only supported with the sqlite_vec backend.
Config:
  MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
  MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
  MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
```

## Performance Notes

**Concrete numbers** (from production and CLAUDE.md):
- **~5ms reads** on SQLite-vec local storage
- **~600MB** resident memory for a warmed service with embedding model loaded
- **534,628× speedup** from global singleton caching introduced in v8.26.0 — prevents redundant storage re-initialization across MCP tool calls

**Qualitative characteristics:**
- **SQLite-vec**: disk-bound; performance scales with SSD speed and memory-mapped cache size (`cache_size` pragma)
- **Cloudflare**: network-bound; latency tracks your connection RTT to the nearest Cloudflare PoP. Expect 10× higher read latency than local
- **Hybrid**: reads match SQLite-vec; writes return as soon as the local commit lands (cloud sync is asynchronous and batched)

**Storage footprint:**
- SQLite-vec: single `.db` file (memories + embeddings + FTS5 + graph tables in one database)
- Cloudflare: D1 rows + Vectorize index + optional R2 for large payloads
- Hybrid: local `.db` file **plus** Cloudflare footprint

## Feature Parity

All three backends expose identical capabilities via the `BaseStorage` interface:
- Semantic search (KNN over embeddings)
- Tag-based filtering
- Natural-language time queries ("yesterday", "last week")
- Full-text search (SQLite-vec and Hybrid use FTS5; Cloudflare uses D1's SQL LIKE with indexes)
- Duplicate detection via content hashes
- Memory consolidation and quality scoring
- Knowledge graph associations (v9.0.0+)

## Migration Between Backends

Migration scripts live in `scripts/migration/`. Not every direction has a dedicated script — the supported paths are below.

### SQLite-vec → Cloudflare
```bash
python scripts/migration/migrate_to_cloudflare.py migrate \
  --source sqlite_vec \
  --source-path /path/to/your/sqlite_vec.db
```
Reads from your local `.db`, writes to D1 + Vectorize. Preserves content, embeddings, tags, timestamps, and metadata. Use the `export` / `import` subcommands instead of `migrate` if you want an intermediate JSON file (useful for backup before cutting over).

### Cloudflare → SQLite-vec
**No dedicated script exists.** The workaround is to run the service in `hybrid` mode temporarily: set `MCP_MEMORY_STORAGE_BACKEND=hybrid` and let the background sync populate a local SQLite-vec database from Cloudflare. Then switch `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec` once the local file is populated.

If you need a fully offline migration, export memories via the REST API (`GET /api/memories`) and re-import with [`mcp-migration.py`](https://github.com/doobidoo/mcp-memory-service/blob/main/scripts/migration/mcp-migration.py) or a small custom script. Open an issue if this is a blocker — a dedicated Cloudflare → SQLite-vec script is a reasonable feature request.

### SQLite-vec → Hybrid / Cloudflare → Hybrid
Set `MCP_MEMORY_STORAGE_BACKEND=hybrid` and restart. Hybrid initialization reconciles the local SQLite-vec store and the Cloudflare mirror on startup — it picks whichever side is populated and syncs the other direction on the next interval. For a clean cutover from SQLite-vec, run the SQLite-vec → Cloudflare migration above first, then switch to `hybrid`.

### ChromaDB → SQLite-vec (historical)
ChromaDB is no longer a supported backend. If you have a legacy ChromaDB store, use:
```bash
python scripts/migration/migrate_to_sqlite_vec.py
```
See [`chromadb-migration.md`](chromadb-migration.md) for the full legacy migration guide.

See [`docs/guides/migration.md`](migration.md) for operational details and [`docs/troubleshooting/database-transfer-migration.md`](../troubleshooting/database-transfer-migration.md) for recovery scenarios.

## Configuration Reference

Full environment variable list: see [`.env.example`](../../.env.example).

### SQLite-vec
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH="$HOME/.mcp-memory/memory.db"

# CRITICAL when running HTTP + MCP servers concurrently
export MCP_MEMORY_SQLITE_PRAGMAS="journal_mode=WAL,busy_timeout=15000,cache_size=20000"

# Optional: use an external embedding API (sqlite_vec only)
# export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
# export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
```

**Claude Desktop config:**
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PATH": "/path/to/memory.db"
      }
    }
  }
}
```

### Cloudflare
```bash
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-db-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# Optional: R2 for large content (>1MB)
# export CLOUDFLARE_R2_BUCKET=mcp-memory-content
# export CLOUDFLARE_LARGE_CONTENT_THRESHOLD=1048576

# Optional: override embedding model
# export CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5
```

Setup details: [`docs/troubleshooting/cloudflare-api-token-setup.md`](../troubleshooting/cloudflare-api-token-setup.md).

### Hybrid
```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid

# All Cloudflare vars from above are required
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-db-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# Recommended: only the HTTP server syncs to Cloudflare
export MCP_HYBRID_SYNC_OWNER=http

# Optional sync tuning
# export MCP_HYBRID_SYNC_INTERVAL=300     # seconds
# export MCP_HYBRID_BATCH_SIZE=50
# export MCP_HYBRID_SYNC_ON_STARTUP=true
```

**Claude Desktop config (hybrid with `MCP_HYBRID_SYNC_OWNER=http` on the HTTP server):**
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid"
      }
    }
  }
}
```
No Cloudflare token needed in the MCP server's env when the HTTP server owns sync — the MCP server reads directly from SQLite-vec and the HTTP server (with the `.env` file) handles all Cloudflare I/O.

## Decision Flowchart

```
Start: Choose Storage Backend
│
├── Do you have no local disk (serverless / edge)?
│   └── Yes → cloudflare
│
├── Do you need offline operation / no network egress?
│   └── Yes → sqlite_vec
│
├── Do you need external embedding APIs (Ollama, vLLM, TEI)?
│   └── Yes → sqlite_vec (the only compatible backend)
│
├── Single-user development / local prototyping?
│   └── Yes → sqlite_vec
│
└── Production / multi-device / want durability + local speed?
    └── hybrid (recommended default)
```

## Troubleshooting

| Issue | Backend | Fix |
|-------|---------|-----|
| `database is locked` under concurrent HTTP + MCP access | sqlite_vec, hybrid | Add `journal_mode=WAL` to `MCP_MEMORY_SQLITE_PRAGMAS`, restart both servers |
| Cloudflare 401 on MCP server startup | hybrid | Set `MCP_HYBRID_SYNC_OWNER=http` so MCP server skips Cloudflare init entirely |
| Cloudflare 403 / sync stopped working | cloudflare, hybrid | IPv6 token allowlist mismatch — see [cloudflare-ipv6-issue](../troubleshooting/cloudflare-authentication.md) |
| Token rotation didn't take effect | cloudflare, hybrid | Rotating the token in the Cloudflare dashboard does **not** update local `.env` — update `CLOUDFLARE_API_TOKEN` manually and restart |
| Wrong backend reported on startup | any | Run `python scripts/validation/diagnose_backend_config.py` |
| Sync not running in hybrid mode | hybrid | Check the HTTP server logs; verify `MCP_HYBRID_SYNC_ON_STARTUP` and that the HTTP server (not just MCP) is running |

Full troubleshooting index: [`docs/troubleshooting/`](../troubleshooting/).

## Documentation Links
- [SQLite-vec Backend Deep Dive](../sqlite-vec-backend.md)
- [Migration Guide](migration.md)
- [Setup Guide](../setup-guide.md)
- [Cloudflare API Token Setup](../troubleshooting/cloudflare-api-token-setup.md)
- [Sync Issues Troubleshooting](../troubleshooting/sync-issues.md)
