# MCP Memory Service — Configuration Guide

All configuration is driven via environment variables and sensible defaults resolved in `src/mcp_memory_service/config.py`.

## Base Paths

- `MCP_MEMORY_BASE_DIR`: Root directory for service data. Defaults per-OS to an app-data directory and is created if missing.
- Derived paths (auto-created):
  - Chroma path: `${BASE_DIR}/chroma_db` unless overridden.
  - Backups path: `${BASE_DIR}/backups` unless overridden.

Overrides:

- `MCP_MEMORY_CHROMA_PATH` or `mcpMemoryChromaPath`: ChromaDB directory path.
- `MCP_MEMORY_BACKUPS_PATH` or `mcpMemoryBackupsPath`: Backups directory path.

## Storage Backend Selection

- `MCP_MEMORY_STORAGE_BACKEND`: `sqlite_vec` (default), `chroma`, or `cloudflare`.
  - `sqlite-vec` aliases to `sqlite_vec`.
  - Unknown values default to `sqlite_vec` with a warning.

SQLite-vec options:

- `MCP_MEMORY_SQLITE_PATH` or `MCP_MEMORY_SQLITEVEC_PATH`: Path to `.db` file. Default `${BASE_DIR}/sqlite_vec.db`.
- `MCP_MEMORY_SQLITE_PRAGMAS`: CSV list of custom pragmas e.g. `busy_timeout=10000,cache_size=20000`.

Chroma options:

- `MCP_MEMORY_CHROMADB_HOST`: Hostname for remote Chroma.
- `MCP_MEMORY_CHROMADB_PORT`: Port (default 8000).
- `MCP_MEMORY_CHROMADB_SSL`: `true|false` for HTTPS.
- `MCP_MEMORY_CHROMADB_API_KEY`: API key when remote.
- `MCP_MEMORY_COLLECTION_NAME`: Collection name (default `memory_collection`).

Cloudflare options (required unless otherwise noted):

- `CLOUDFLARE_API_TOKEN` (required)
- `CLOUDFLARE_ACCOUNT_ID` (required)
- `CLOUDFLARE_VECTORIZE_INDEX` (required)
- `CLOUDFLARE_D1_DATABASE_ID` (required)
- `CLOUDFLARE_R2_BUCKET` (optional, for large content)
- `CLOUDFLARE_EMBEDDING_MODEL` (default `@cf/baai/bge-base-en-v1.5`)
- `CLOUDFLARE_LARGE_CONTENT_THRESHOLD` (bytes; default 1,048,576)
- `CLOUDFLARE_MAX_RETRIES` (default 3)
- `CLOUDFLARE_BASE_DELAY` (seconds; default 1.0)

## Embedding Model

- `MCP_EMBEDDING_MODEL`: Model name (default `all-MiniLM-L6-v2`).
- `MCP_MEMORY_USE_ONNX`: `true|false` toggle for ONNX path.

## HTTP/HTTPS Interface

- `MCP_HTTP_ENABLED`: `true|false` to enable HTTP interface.
- `MCP_HTTP_HOST`: Bind address (default `0.0.0.0`).
- `MCP_HTTP_PORT`: Port (default `8000`).
- `MCP_CORS_ORIGINS`: Comma-separated origins (default `*`).
- `MCP_SSE_HEARTBEAT`: SSE heartbeat interval seconds (default 30).
- `MCP_API_KEY`: Optional API key for HTTP.

TLS:

- `MCP_HTTPS_ENABLED`: `true|false`.
- `MCP_SSL_CERT_FILE`, `MCP_SSL_KEY_FILE`: Certificate and key paths.

## mDNS Service Discovery

- `MCP_MDNS_ENABLED`: `true|false` (default `true`).
- `MCP_MDNS_SERVICE_NAME`: Service display name (default `MCP Memory Service`).
- `MCP_MDNS_SERVICE_TYPE`: Service type (default `_mcp-memory._tcp.local.`).
- `MCP_MDNS_DISCOVERY_TIMEOUT`: Seconds to wait for discovery (default 5).

## Consolidation (Optional)

- `MCP_CONSOLIDATION_ENABLED`: `true|false`.
- Archive location:
  - `MCP_CONSOLIDATION_ARCHIVE_PATH` or `MCP_MEMORY_ARCHIVE_PATH` (default `${BASE_DIR}/consolidation_archive`).
- Config knobs:
  - Decay: `MCP_DECAY_ENABLED`, retention by type: `MCP_RETENTION_CRITICAL`, `MCP_RETENTION_REFERENCE`, `MCP_RETENTION_STANDARD`, `MCP_RETENTION_TEMPORARY`.
  - Associations: `MCP_ASSOCIATIONS_ENABLED`, `MCP_ASSOCIATION_MIN_SIMILARITY`, `MCP_ASSOCIATION_MAX_SIMILARITY`, `MCP_ASSOCIATION_MAX_PAIRS`.
  - Clustering: `MCP_CLUSTERING_ENABLED`, `MCP_CLUSTERING_MIN_SIZE`, `MCP_CLUSTERING_ALGORITHM`.
  - Compression: `MCP_COMPRESSION_ENABLED`, `MCP_COMPRESSION_MAX_LENGTH`, `MCP_COMPRESSION_PRESERVE_ORIGINALS`.
  - Forgetting: `MCP_FORGETTING_ENABLED`, `MCP_FORGETTING_RELEVANCE_THRESHOLD`, `MCP_FORGETTING_ACCESS_THRESHOLD`.
- Scheduling (APScheduler-ready):
  - `MCP_SCHEDULE_DAILY` (default `02:00`), `MCP_SCHEDULE_WEEKLY` (default `SUN 03:00`), `MCP_SCHEDULE_MONTHLY` (default `01 04:00`), `MCP_SCHEDULE_QUARTERLY` (default `disabled`), `MCP_SCHEDULE_YEARLY` (default `disabled`).

## Machine Identification

- `MCP_MEMORY_INCLUDE_HOSTNAME`: `true|false` to tag memories with `source:<hostname>` and include `hostname` metadata.

## Logging and Performance

- `LOG_LEVEL`: Root logging level (default `WARNING`).
- `DEBUG_MODE`: When unset, the service raises log levels for performance-critical libs (`chromadb`, `sentence_transformers`, `transformers`, `torch`, `numpy`).

## Recommended Defaults by Backend

- SQLite-vec:
  - Defaults enable WAL, busy timeout, optimized cache; customize with `MCP_MEMORY_SQLITE_PRAGMAS`.
  - For multi-client setups, the service auto-detects and may start/use an HTTP coordinator.
- ChromaDB:
  - HNSW space/ef/M values tuned for balanced accuracy and speed; migration messaging warns of deprecation and recommends moving to SQLite-vec.
- Cloudflare:
  - Ensure required variables are set or the process exits with a clear error and checklist.

