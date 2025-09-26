# MCP Memory Service — Architecture Overview

This document summarizes the Memory Service architecture, components, data flow, and how MCP integration is implemented.

## High-Level Design

- Clients: Claude Desktop/Code, VS Code, Cursor, Continue, and other MCP-compatible clients.
- Protocol Layer:
  - MCP stdio server: `src/mcp_memory_service/server.py` (uses `mcp.server.Server`).
  - FastAPI MCP server: `src/mcp_memory_service/mcp_server.py` (via `FastMCP`, exposes streamable HTTP for remote access).
- Core Domain:
  - Models: `src/mcp_memory_service/models/memory.py` defining `Memory` and `MemoryQueryResult`.
  - Utilities: hashing, time parsing, system detection, HTTP server coordination.
- Storage Abstraction:
  - Interface: `src/mcp_memory_service/storage/base.py` (`MemoryStorage` ABC).
  - Backends:
    - SQLite-vec: `src/mcp_memory_service/storage/sqlite_vec.py` (recommended default).
    - ChromaDB: `src/mcp_memory_service/storage/chroma.py` (deprecated; migration path provided).
    - Cloudflare: `src/mcp_memory_service/storage/cloudflare.py` (Vectorize + D1 + optional R2).
    - HTTP client: `src/mcp_memory_service/storage/http_client.py` (multi-client coordination).
- CLI:
  - Entry points: `memory`, `memory-server`, `mcp-memory-server` (pyproject scripts).
  - Implementation: `src/mcp_memory_service/cli/main.py` (server, status, ingestion commands).
- Config and Env:
  - Central config: `src/mcp_memory_service/config.py` (paths, backend selection, HTTP/HTTPS, mDNS, consolidation, hostname tagging, Cloudflare settings).
- Consolidation (optional): associations, clustering, compression, forgetting; initialized lazily when enabled.

## Data Flow

1. Client invokes MCP tool/prompt (stdio or FastMCP HTTP transport).
2. Server resolves the configured backend via `config.py` and lazy/eager initializes storage.
3. For SQLite-vec:
   - Embeddings generated via `sentence-transformers` (or ONNX disabled path) and stored alongside content and metadata in SQLite; vector search via `vec0` virtual table.
   - WAL mode + busy timeouts for concurrent access; optional HTTP coordination for multi-client scenarios.
4. For ChromaDB: uses DuckDB+Parquet persistence and HNSW settings (deprecated path; migration messaging built-in).
5. For Cloudflare: Vectorize (vectors), D1 (metadata), R2 (large content); HTTPx for API calls.
6. Results map back to `Memory`/`MemoryQueryResult` and are returned to the MCP client.

## MCP Integration Patterns

- Stdio MCP (`server.py`):
  - Uses `mcp.server.Server` and registers tools/prompts for memory operations, diagnostics, and analysis.
  - Client-aware logging (`DualStreamHandler`) to keep JSON wire clean for Claude Desktop; richer stdout for LM Studio.
  - Coordination: detects if an HTTP sidecar is needed for multi-client access; starts/uses `HTTPClientStorage` when appropriate.

- FastMCP (`mcp_server.py`):
  - Wraps storage via `lifespan` context; exposes core tools like `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `check_database_health` using `@mcp.tool()`.
  - Designed for remote/HTTP access and Claude Code compatibility via `streamable-http` transport.

## Storage Layer Abstraction

- `MemoryStorage` interface defines: `initialize`, `store`, `retrieve/search`, `search_by_tag(s)`, `delete`, `delete_by_tag`, `cleanup_duplicates`, `update_memory_metadata`, `get_stats`, plus optional helpers for tags/time ranges.
- Backends adhere to the interface and can be swapped via `MCP_MEMORY_STORAGE_BACKEND`.

## Configuration Management

- Paths: base dir and per-backend storage paths (auto-created, validated for writability).
- Backend selection: `MCP_MEMORY_STORAGE_BACKEND` ∈ `{sqlite_vec, chroma, cloudflare}` (normalized).
- HTTP/HTTPS server, CORS, API key, SSE heartbeat.
- mDNS discovery toggles and timeouts.
- Consolidation: enabled flag, archive path, decay/association/clustering/compression/forgetting knobs; schedules for APScheduler.
- Hostname tagging: `MCP_MEMORY_INCLUDE_HOSTNAME` annotates source host.
- Cloudflare: tokens, account, Vectorize index, D1 DB, optional R2, retry behavior.

## Dependencies and Roles

- `mcp`: MCP protocol server/runtime.
- `sqlite-vec`: vector index for SQLite; provides `vec0` virtual table.
- `sentence-transformers`, `torch`: embedding generation; can be disabled.
- `chromadb`: legacy backend (DuckDB+Parquet).
- `fastapi`, `uvicorn`, `sse-starlette`, `aiofiles`, `aiohttp/httpx`: HTTP transports and Cloudflare/API.
- `psutil`, `zeroconf`: client detection and mDNS discovery.

## Logging and Diagnostics

- Client-aware logging handler prevents stdout noise for Claude (keeps JSON clean) and surfaces info on LM Studio.
- `LOG_LEVEL` env to set root logger (defaults WARNING). Performance-critical third-party loggers elevated to WARNING unless `DEBUG_MODE` set.

## Performance and Concurrency

- SQLite-vec pragmas: WAL, busy_timeout, synchronous=NORMAL, cache_size, temp_store.
- Custom pragmas via `MCP_MEMORY_SQLITE_PRAGMAS`.
- Embedding model is cached and loaded once; ONNX path available when enabled.
- Average query time tracking, async operations, and optional consolidation scheduler.

