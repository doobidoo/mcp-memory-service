# Milvus Storage Backend

The Milvus backend stores memories in a [Milvus](https://milvus.io) collection using the `pymilvus` `MilvusClient` API. One configuration supports three deployment targets:

| Target | Use case | URI example |
|--------|----------|-------------|
| **Milvus Lite** (default) | Local development, single-file portability, CI tests | `./milvus.db` |
| **Self-hosted Milvus** | Dedicated vector DB running in Docker, Kubernetes, bare metal | `http://localhost:19530` |
| **Zilliz Cloud** | Managed Milvus on Zilliz — fully hosted, pay-per-use | `https://xxx.zillizcloud.com` (with token) |

All three modes share the same schema, index, and code path — switching targets is a one-line change to `MCP_MILVUS_URI`.

## Which URI to use

| URI form | Use case | Stability |
|----------|----------|-----------|
| `uri="./milvus.db"` (Milvus Lite) | Batch scripts, unit tests, short-lived CLIs. | ✅ for sequential use. ⚠️ For long-lived processes (MCP servers, web services, notebooks with idle gaps, parallel tool calls) Milvus Lite is **not recommended** — the embedded daemon is not designed for this pattern. See [milvus-lite#334](https://github.com/milvus-io/milvus-lite/issues/334) and [milvus-lite#264](https://github.com/milvus-io/milvus-lite/issues/264). The backend auto-reconnects after a single idle-induced daemon exit, but cannot work around concurrent-daemon limitations on the same `.db` file. |
| `uri="http://host:19530"` (self-hosted Milvus) | **Recommended for MCP servers and any long-running service.** Stable across idle gaps and concurrent clients. Easy to run via the official `milvusdb/milvus` Docker image. | ✅ production-grade |
| `uri="https://<cluster>.zillizcloud.com"` + `token="..."` (Zilliz Cloud) | Managed Milvus at scale, no ops burden. | ✅ production-grade |

> **MCP deployments**: use a Docker-hosted Milvus or Zilliz Cloud endpoint. Milvus Lite's embedded daemon is unsuitable for long-lived, multi-call MCP scenarios.

The backend auto-detects Lite vs. remote from the URI — file paths ending in `.db` are treated as Lite (matching pymilvus's own heuristic). Only the Lite code path reconnects on a dead channel; remote targets fail fast on network errors so real infrastructure problems are visible instead of masked.

## Quick Start

### 1. Install

```bash
pip install -e ".[milvus]"
```

The `milvus` extra pulls in `pymilvus>=2.5.0` and `milvus-lite>=2.4.10`. Milvus Lite is the default and requires no external service.

### 2. Configure

The minimum configuration to activate the backend:

```bash
export MCP_MEMORY_STORAGE_BACKEND=milvus
```

That's it — `MCP_MILVUS_URI` defaults to `<project_base>/milvus.db` (Milvus Lite). To point at a server or Zilliz Cloud, set `MCP_MILVUS_URI` (and `MCP_MILVUS_TOKEN` for Zilliz Cloud):

```bash
# Milvus Lite (default)
export MCP_MILVUS_URI=./milvus.db

# Self-hosted Milvus server (e.g. Docker)
export MCP_MILVUS_URI=http://localhost:19530

# Zilliz Cloud
export MCP_MILVUS_URI=https://xxx.zillizcloud.com
export MCP_MILVUS_TOKEN=your-zilliz-token

# Optional: custom collection name (default: mcp_memory)
export MCP_MILVUS_COLLECTION_NAME=mcp_memory
```

### 3. Run

```bash
python -m mcp_memory_service.server
```

The first call to `initialize()` creates the collection with an AUTOINDEX on the vector field and the COSINE metric. Subsequent runs reuse the existing collection.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_STORAGE_BACKEND` | `sqlite_vec` | Set to `milvus` to use this backend. |
| `MCP_MILVUS_URI` | `<project_base>/milvus.db` | Milvus endpoint. Local file path for Lite, `http(s)://` URL for server/Cloud. |
| `MCP_MILVUS_TOKEN` | — | Auth token (required for Zilliz Cloud, optional for self-hosted with auth enabled, ignored by Lite). |
| `MCP_MILVUS_COLLECTION_NAME` | `mcp_memory` | Name of the Milvus collection that holds the memories. |
| `MCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model used for local embedding. |

> **Why `MCP_MILVUS_*` instead of `MILVUS_*`?**
> `pymilvus`'s ORM reads the `MILVUS_URI` environment variable at import time and requires an `http(s)://` URL — a Milvus Lite file path in that variable would fail with `ConnectionConfigException` before any application code runs. Using the `MCP_MILVUS_*` prefix avoids the clash.

## Schema

The collection is created once by `MilvusMemoryStorage.initialize()` with the following schema:

| Field | Type | Notes |
|-------|------|-------|
| `id` | VARCHAR (PK, 128) | The memory's content hash. Guarantees exact-hash deduplication for free. |
| `vector` | FLOAT_VECTOR | Dimension matches the embedding model (384 for `all-MiniLM-L6-v2`). |
| `content` | VARCHAR (65535) | Memory text. Milvus's VARCHAR hard cap. |
| `tags` | VARCHAR (8192) | Comma-delimited with leading and trailing commas (`,tag1,tag2,`) for exact `like` matching. |
| `memory_type` | VARCHAR (128) | Ontology-validated memory type. |
| `metadata` | VARCHAR (65535) | JSON-serialized metadata dict. |
| `created_at` / `updated_at` | DOUBLE | Unix timestamps. |
| `created_at_iso` / `updated_at_iso` | VARCHAR (64) | ISO-8601 timestamp strings. |

**Index**: single vector index on `vector` with `index_type=AUTOINDEX`, `metric_type=COSINE`. This works identically on Milvus Lite, self-hosted Milvus, and Zilliz Cloud.

## Semantics and limitations

- **Primary key = content hash.** Duplicate stores with the same content are rejected at the client before insert. `cleanup_duplicates()` is a no-op for this reason.
- **Tag matching is exact.** Tags are stored with leading/trailing commas and matched with `tags like "%,<tag>,%"`. Substrings never match — `test` does not match `testing`.
- **Hard delete.** Unlike `sqlite_vec`, deletions are permanent. There is no tombstone table and no `is_deleted()` / `purge_deleted()` semantics (the base class defaults apply). If you need soft-delete for multi-device sync, use the `hybrid` backend instead.
- **Content length.** Limited to `65535 - 256 = 65279` characters by the `max_content_length` property. Milvus's VARCHAR hard cap applies; longer content should be chunked by the calling service.
- **Semantic deduplication** honours the same `MCP_SEMANTIC_DEDUP_*` environment variables as `sqlite_vec`, so behaviour is consistent across backends.
- **External embedding APIs** (Ollama, vLLM, TEI) are not supported — the Milvus backend always uses local sentence-transformers. Use `sqlite_vec` if you need an external embedding endpoint.

## Deployment recipes

### Milvus Lite (local file)

Zero external dependencies. Good for development, CI, and single-user deployments.

```bash
export MCP_MEMORY_STORAGE_BACKEND=milvus
export MCP_MILVUS_URI="$HOME/.mcp-memory/milvus.db"
```

The `milvus-lite` package spawns an in-process daemon the first time a client connects. Subsequent connections to the same URI reuse the daemon.

### Self-hosted Milvus (Docker)

Start Milvus with the official compose file:

```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
docker compose up -d
```

Then:

```bash
export MCP_MEMORY_STORAGE_BACKEND=milvus
export MCP_MILVUS_URI=http://localhost:19530
```

If auth is enabled on the Milvus server, also set `MCP_MILVUS_TOKEN`.

### Zilliz Cloud

Create a cluster at [cloud.zilliz.com](https://cloud.zilliz.com) and copy the **Public Endpoint** and **Token** into:

```bash
export MCP_MEMORY_STORAGE_BACKEND=milvus
export MCP_MILVUS_URI=https://in03-xxxxxxx.api.gcp-us-west1.zillizcloud.com
export MCP_MILVUS_TOKEN=your-zilliz-token
```

## Claude Desktop configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "milvus",
        "MCP_MILVUS_URI": "/path/to/milvus.db"
      }
    }
  }
}
```

For a Zilliz Cloud setup, also add `"MCP_MILVUS_TOKEN": "your-token"`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ConnectionConfigException: Illegal uri` on startup | Something set `MILVUS_URI` (no `MCP_` prefix) to a non-HTTP value. Unset `MILVUS_URI` or use `MCP_MILVUS_URI` instead. |
| `pymilvus.MilvusException: illegal connection params` with Milvus Lite | Too many per-test databases — `milvus-lite` spawns a daemon per distinct DB path. Reuse a single DB file and use a unique collection name per test. |
| `Cannot invoke RPC on closed channel!` / `server unavailable` on Milvus Lite after long idle | The Lite daemon exited after idle (upstream [milvus-lite#334](https://github.com/milvus-io/milvus-lite/issues/334)). The backend detects this and auto-reconnects once on the next RPC — the failure surfaces as a `WARNING` log followed by a successful retry. If it recurs every call, check for Lite-on-Lite collisions (multiple processes writing the same `.db`). For production services, switch to self-hosted Milvus or Zilliz Cloud. |
| Embedding dimension mismatch warning | An existing collection has a different vector dim than the current model. Drop the collection (`MilvusClient.drop_collection(...)`) or switch to a model that matches the stored dim. |
| First `initialize()` takes a long time | The sentence-transformers model is downloading from Hugging Face. Once cached, later runs are fast (the backend automatically enables `HF_HUB_OFFLINE` when the cache is present). |
| `pkg_resources is deprecated` warning on Python 3.13 | Harmless deprecation warning from `milvus-lite`; our `milvus` extra pins `setuptools<81` on Python 3.13+ to silence it. |

## Support

The Milvus backend is maintained by [@zc277584121](https://github.com/zc277584121). For Milvus-specific issues:

1. File a GitHub issue with the `backend:milvus` label.
2. Include: deployment mode (Lite / self-hosted / Zilliz Cloud), `pymilvus` version, and the relevant section of your server logs.
3. First responder SLA: best-effort, typically within a few business days (through ~October 2026).

For questions about `pymilvus` itself or Milvus server behavior, the upstream [Milvus community channels](https://milvus.io/community) are usually faster than this repo.

## See also

- [Storage Backend Comparison](guides/STORAGE_BACKENDS.md) — picking between `sqlite_vec`, `cloudflare`, `hybrid`, and `milvus`
- [pymilvus MilvusClient reference](https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Client/MilvusClient.md)
- [Milvus Lite quickstart](https://milvus.io/docs/milvus_lite.md)
- [Zilliz Cloud documentation](https://docs.zilliz.com/)
