# MCP Memory Service — API Reference

This document catalogs available APIs exposed via the MCP servers and summarizes request and response patterns.

## MCP (FastMCP HTTP) Tools

The core of the unified MCP tool surface introduced in v10.0.0. This table is the
orientation set, not the full list — the authoritative list is
`TOOL_REGISTRY` in `src/mcp_memory_service/tools/registry.py`, which has grown well past
these twelve (`grep -c 'ToolDef(' src/mcp_memory_service/tools/registry.py`):

| Tool | Purpose |
|------|---------|
| `memory_store` | Store a new memory (replaces `store_memory`) |
| `memory_search` | Hybrid semantic + keyword search (replaces `retrieve_memory`, `search_by_tag` family) |
| `memory_list` | List/filter memories (replaces tag-listing variants) |
| `memory_delete` | Delete by hash, tag, timeframe, or filter (replaces `delete_*` family) |
| `memory_health` | Server + storage health check (replaces `check_database_health`) |
| `memory_stats` | Usage and storage statistics |
| `memory_update` | Update memory metadata (replaces `update_memory_metadata`) |
| `memory_cleanup` | Remove duplicates / orphans (replaces `cleanup_duplicates`) |
| `memory_consolidate` | Run/manage consolidation (replaces `trigger_consolidation`, `scheduler_status`) |
| `memory_quality` | Rate / get / analyze quality (replaces `rate_memory`, `get_memory_quality`, `analyze_quality_distribution`) |
| `memory_ingest` | Ingest documents (PDF/DOCX/TXT/JSON) |
| `memory_graph` | Knowledge-graph queries (`connected`, `path`, `subgraph`, `extract_entities`, `infer`, `suggest`, `abduct`, `list_entities`, `entity_profile`) |
| `memory_explore` | Knowledge map of entities for a query, with ranked chunks per entity |
| `memory_detail` | Full ranked chunk list for one entity, plus its neighbours |

`memory_explore` and `memory_detail` are the two-phase retrieval pair: explore for an
overview, detail to drill in. They require a populated entity graph and are unavailable on
the Cloudflare backend. Setup, prerequisites and limitations:
[Token-Efficient Retrieval](../guides/token-efficient-retrieval.md).

Deprecated v9-and-earlier tool names **no longer resolve**. The alias layer
(`compat.DEPRECATED_TOOLS`) was removed in v11.0.0, so old names now fail rather than
warn — see `docs/MIGRATION.md` for the mapping.

Transport: `mcp.run("streamable-http")`, default host `0.0.0.0`, default port `8000` or `MCP_SERVER_PORT`/`MCP_SERVER_HOST`.

## MCP (stdio) Server Tools and Prompts

Defined in `src/mcp_memory_service/server.py` using `mcp.server.Server`. Exposes a broader set of tools/prompts beyond the core FastMCP tools above.

Highlights:

- Core memory ops via the unified memory_* tool surface: memory_store, memory_search, memory_list (tag/filter), memory_delete (by hash/tag/timeframe), memory_cleanup, memory_update.
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
tool: memory_store
args: { "content": "Refactored auth flow to use OAuth 2.1", "tags": ["auth", "refactor"], "memory_type": "note" }
```

Retrieve by query:

```
tool: memory_search
args: { "query": "OAuth refactor", "limit": 5 }
```

Search by tags:

```
tool: memory_list
args: { "tags": ["auth", "refactor"], "match_all": true }
```

Delete by hash:

```
tool: memory_delete
args: { "content_hash": "<hash>" }
```

Bound an oversized search response:

```
tool: memory_search
args: { "query": "OAuth refactor", "limit": 5, "max_response_chars": 30000 }
```

Query-aware summary with source hashes (opt-in; requires a configured Harvest LLM
provider chain or `GROQ_API_KEY`):

```
tool: memory_search
args: { "query": "What caused the replication consistency issue?", "summarize": true }
```

Requires the operator to set `MCP_SEARCH_SUMMARIZE_ENABLED=true` (off by default)
and restart the server, in addition to configuring an LLM provider. This permits
both local and remote read-scope clients to request summaries. While disabled,
the tool returns raw results with a warning and makes no summarization provider call.

The successful response is JSON containing `summary`, `source_hashes`, a metadata
`snapshot`, and counts of summarized/omitted records. Original memories stay
stored; failures return raw results with a warning. See
[token-efficient retrieval](../guides/token-efficient-retrieval.md#2-query-aware-search-summaries)
for provider configuration, budgets, source validation, and follow-up retrieval.

Overview, then drill in:

```
tool: memory_explore
args: { "query": "authentication design", "max_entities": 5 }

tool: memory_detail
args: { "entity_id": "authentication-design", "limit": 20 }
```
