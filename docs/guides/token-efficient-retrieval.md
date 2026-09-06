# Token-Efficient Retrieval

How to stop `memory_search` from flooding an agent's context, and how to use the
two-phase `memory_explore` / `memory_detail` pair.

Two independent levers:

1. **Bounding a search response** — works on every backend, no setup. Start here.
2. **The two-phase knowledge map** — needs a populated entity graph, and a graph-capable
   backend.

## 1. Bounding a search response

An unbounded `memory_search` on a large store routinely returns 10k+ tokens, most of it
content the caller will not read. Two parameters control that:

| Parameter | Default | Effect |
|---|---|---|
| `limit` | 10 | Maximum number of memories returned |
| `max_response_chars` | unlimited | Caps total response size, truncating **at memory boundaries** so no record is cut mid-content |

```json
{"query": "replication message bus consistency", "limit": 5, "max_response_chars": 30000}
```

30000-50000 is the range to start with. Truncation reports what was dropped, so the caller
can narrow the query or raise the cap deliberately rather than silently losing results.

If you are calling the service from code rather than as an MCP tool, the code-execution
interface has compact result types (`CompactMemory`, `CompactSearchResult`) that cut
85-91% of the tokens of full `Memory` objects — see
[`docs/api/code-execution-interface.md`](../api/code-execution-interface.md).

## 2. The two-phase knowledge map

`memory_explore` returns a map of entities with a few ranked chunks each;
`memory_detail` then returns the full ranked chunk list for one entity. The point is to
let a caller decide where to drill in before pulling content into context.

```json
{"query": "authentication design", "max_entities": 5, "chunks_per_entity": 2}
```

```json
{"entity_id": "authentication-design", "limit": 20, "include_related": false}
```

`memory_explore` parameters: `max_entities` (default 10), `chunks_per_entity` (default 3),
`tags`, `min_score`, `store`, and `scoring: "composite"` to re-rank by relevance plus graph
proximity and entity centrality. `memory_detail` takes the `entity_id` from the map, plus
`limit` (default 50, max 200), `include_related` (default true) and `max_hops` (default 1).

### Prerequisite: the entity graph must be populated

**`memory_explore` returns `{"entities": [], "count": 0}` for every query until entities
exist.** Nothing populates them by default. Check first:

```json
{"action": "list_entities", "limit": 10}
```

via `memory_graph`. If that returns `{"entities": [], "total": 0}`, the two-phase tools
have nothing to work with, and the fix is one of:

- **Batch, for memories you already have** — `memory_quality` with
  `{"action": "maintain", "dry_run": false}`. The `dry_run` default is `true`, which counts
  candidate entities and stores none, so a default call looks like it did nothing.
- **Ongoing, for memories stored from now on** — set `MCP_ENTITY_LINKING_ENABLED=1`. This
  links at store time and does not touch existing memories.
- **A single memory** — `memory_graph` with
  `{"action": "extract_entities", "hash": "<content hash>"}`.

Graph storage exists for the `sqlite_vec`, `hybrid` and `milvus` backends. Cloudflare has
none, so these tools are unavailable there.

### The scan limit will surprise you on a large store

`maintain` scans at most `MCP_MAINTAIN_SCAN_LIMIT` memories per run, **default 2000**, and
it takes the first N in the order the storage returns them. Running it repeatedly re-scans
the same 2000 — it does not walk forward through the store. On any store larger than the
limit, everything past the first N is never reached, however many times you run it.

Set `MCP_MAINTAIN_SCAN_LIMIT=0` for unlimited, and expect the run to take proportionally
longer:

```bash
MCP_MAINTAIN_SCAN_LIMIT=0 memory restart
```

### What counts as an entity

Extraction is deliberately pattern-based and high-precision, not semantic NER. It finds:

- `@mentions` (typed as `person`)
- `#hashtags` (typed as `tag`)
- URLs
- file paths
- the memory's **tags**
- any terms listed in `MCP_ENTITY_CUSTOM_TERMS` (comma-separated), matched on word
  boundaries

It does **not** recognise "Postgres", "Kafka" or "OAuth" as entities just because they
appear in prose. On a store whose memories are well tagged but whose content carries none
of the patterns above, tags do most of the work — which is why tags being dropped
(#218) left graphs empty.

For domain-specific extraction, `MCP_ENTITY_EXTRACTOR_MODULES` accepts comma-separated
`module.path:ClassName` entries implementing the `DomainExtractor` protocol.

### What `summary` actually is

The `summary` field on each entity is the first sentence of up to three of its top chunks,
each truncated to 150 characters. Chunk content itself is capped at 500 characters. It is
snippet concatenation, useful for deciding where to drill in — not a summarization of the
entity.

### Query-scoped entity selection

`memory_explore` selects entities linked to the retrieved query chunks, starting with the
highest-relevance chunks and stopping at `max_entities`. If none of those chunks has an
entity link, it falls back to the graph's global entity list so partially populated graphs
remain discoverable. A fallback entity receives an empty `top_chunks` list and summary
unless one of its linked memories is present in the query results; unrelated query chunks
are never attached to it.

## Which to reach for

| Situation | Use |
|---|---|
| Search responses too large | `limit` + `max_response_chars` |
| Want an overview before pulling content | `memory_explore` → `memory_detail` |
| Restrict results to one known entity | `memory_search` with `entity` (needs a populated graph) |
| Calling from Python rather than as an MCP tool | code-execution interface, compact types |
