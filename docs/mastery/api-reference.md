# MCP Memory Service — API Reference

This document catalogs available APIs exposed via the MCP servers and summarizes request and response patterns.

## MCP (FastMCP HTTP) Tools

Defined in `src/mcp_memory_service/mcp_server.py` using `@mcp.tool()`:

- `store_memory(content, tags=None, memory_type="note", metadata=None, client_hostname=None)`
  - Stores a new memory; tags and metadata optional. If `INCLUDE_HOSTNAME=true`, a `source:<hostname>` tag and `hostname` metadata are added.
  - Response: `{ success: bool, message: str, content_hash: str }`.

- `retrieve_memory(query, n_results=5, min_similarity=0.0)`
  - Semantic search by query; returns up to `n_results` matching memories.
  - Response: `{ memories: [{ content, content_hash, tags, memory_type, created_at, similarity_score }...], query, total_results }`.

- `search_by_tag(tags, match_all=False)`
  - Search by a tag or list of tags. `match_all=true` requires all tags; otherwise any.
  - Response: `{ memories: [{ content, content_hash, tags, memory_type, created_at }...], search_tags: [...], match_all, total_results }`.

- `delete_memory(content_hash)`
  - Deletes a memory by its content hash.
  - Response: `{ success: bool, message: str, content_hash }`.

- `check_database_health()`
  - Health and status of the configured backend.
  - Response: `{ status: "healthy"|"error", backend, statistics: { total_memories, total_tags, storage_size, last_backup }, timestamp? }`.

Transport: `mcp.run("streamable-http")`, default host `0.0.0.0`, default port `8000` or `MCP_SERVER_PORT`/`MCP_SERVER_HOST`.

## MCP (stdio) Server Tools and Prompts

Defined in `src/mcp_memory_service/server.py` using `mcp.server.Server`. Exposes a broader set of tools/prompts beyond the core FastMCP tools above.

Highlights:

- Core memory ops: store, retrieve/search, search_by_tag(s), delete, delete_by_tag, cleanup_duplicates, update_memory_metadata, time-based recall.
- Analysis/export: knowledge_analysis, knowledge_export (supports `format: json|markdown|text`, optional filters).
- Maintenance: memory_cleanup (duplicate detection heuristics), health/stats, tag listing.
- Consolidation (optional): association, clustering, compression, forgetting tasks and schedulers when enabled.

Note: The stdio server dynamically picks storage mode for multi-client scenarios (direct SQLite-vec with WAL vs. HTTP coordination), suppresses stdout for Claude Desktop, and prints richer diagnostics for LM Studio.

## HTTP Interface

- For FastMCP, HTTP transport is used to carry MCP protocol; endpoints are handled by the FastMCP layer and not intended as a REST API surface.
- A dedicated HTTP API and dashboard exist under `src/mcp_memory_service/web/` in some distributions. In this repo version, coordination HTTP is internal and the recommended external interface is MCP.

## Error Model and Logging

- MCP tool errors are surfaced as `{ success: false, message: <details> }` or include `error` fields.
- Logging routes WARNING+ to stderr (Claude Desktop strict mode), info/debug to stdout only for LM Studio; set `LOG_LEVEL` for verbosity.

## Examples

Store memory:

```
tool: store_memory
args: { "content": "Refactored auth flow to use OAuth 2.1", "tags": ["auth", "refactor"], "memory_type": "note" }
```

Retrieve by query:

```
tool: retrieve_memory
args: { "query": "OAuth refactor", "n_results": 5 }
```

Search by tags:

```
tool: search_by_tag
args: { "tags": ["auth", "refactor"], "match_all": true }
```

Delete by hash:

```
tool: delete_memory
args: { "content_hash": "<hash>" }
```

