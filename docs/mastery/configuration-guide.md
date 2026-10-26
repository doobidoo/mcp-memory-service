# MCP Memory Service — Configuration Guide

All configuration is driven via environment variables and sensible defaults resolved in `src/mcp_memory_service/config.py`.

## Base Paths

- `MCP_MEMORY_BASE_DIR`: Root directory for service data. Defaults per-OS to an app-data directory and is created if missing.
- Derived paths (auto-created):
  - SQLite database: `${BASE_DIR}/sqlite_vec.db` unless overridden.
  - Backups path: `${BASE_DIR}/backups` unless overridden.

Overrides:

- `MCP_MEMORY_SQLITE_PATH`: SQLite-vec database file path.
- `MCP_MEMORY_BACKUPS_PATH` or `mcpMemoryBackupsPath`: Backups directory path.

## Storage Backend Selection

- `MCP_MEMORY_STORAGE_BACKEND`: `sqlite_vec` (default), `cloudflare`, or `hybrid`.
  - `sqlite-vec` aliases to `sqlite_vec`.
  - Unknown values default to `sqlite_vec` with a warning.

SQLite-vec options:

- `MCP_MEMORY_SQLITE_PATH` or `MCP_MEMORY_SQLITEVEC_PATH`: Path to `.db` file. Default `${BASE_DIR}/sqlite_vec.db`.
- `MCP_MEMORY_SQLITE_PRAGMAS`: CSV list of custom pragmas e.g. `journal_mode=WAL,busy_timeout=15000,cache_size=20000` (recommended for concurrent access).

> **Note:** The service will warn about unrecognized `MCP_MEMORY_*` environment variables that contain `PATH`, `DB`, or `DIR` in their names when falling back to the default database path. This helps catch common typos like `MCP_MEMORY_DB_PATH` instead of `MCP_MEMORY_SQLITE_PATH`.

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

### Choosing a model

The default works well for English-only content. For memories in other languages, switch
to a multilingual model:

| Model | Languages | Dimensions | Use case |
|-------|-----------|-----------|----------|
| `all-MiniLM-L6-v2` (default) | English only | 384 | Fastest, English-only deployments |
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ languages | 384 | Mixed-language or non-English content |

```bash
export MCP_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

> **Switching models requires re-embedding existing memories.** Cross-language cosine
> similarity drops from roughly 0.95 to 0.10 otherwise. Both models above are 384-dim,
> so the switch is a straight re-embed: stop the service, run
> `python scripts/maintenance/regenerate_embeddings.py` with the new model env var, then
> restart.
>
> A model with a **different dimension** is not that. The script writes new vectors into
> the existing vector table and cannot change its width, so storage initialization fails
> with `Dimension mismatch for inserted vector` before anything is re-embedded. Moving to
> a different dimension means rebuilding the store — see
> [external embeddings](../deployment/external-embeddings.md) for the failure mode and
> the migration path.

### Pinning a non-default model

If a custom embedding model fails to load, the service can fall back to the default
MiniLM (384-dim) — and then every subsequent write fails with a dimension mismatch
against the existing store. When you pin a non-default model, pin the model path as well
and set the Hugging Face offline flags, so a load failure surfaces loudly instead of
degrading into a silent fallback.

## HTTP/HTTPS Interface

- `MCP_HTTP_ENABLED`: `true|false` to enable HTTP interface.
- `MCP_HTTP_HOST`: Bind address (default `127.0.0.1`, localhost only).
- `MCP_HTTP_PORT`: Port (default `8000`).
- `MCP_HTTP_ROOT_PATH`: External path prefix when a reverse proxy strips the
  prefix before forwarding (for example, `/memory`). Defaults to empty.
- `MCP_CORS_ORIGINS`: Comma-separated origins (default `*`).
- `MCP_SSE_HEARTBEAT`: SSE heartbeat interval seconds (default 30).
- `MCP_API_KEY`: Optional API key for HTTP.

> **Binding to `0.0.0.0` exposes the API to your network.** The default is localhost
> only. Do this in trusted environments only, with authentication and firewall rules in
> place. On untrusted networks, terminate TLS in front of the service (reverse proxy with
> HTTPS) or put it behind a VPN overlay.

For a proxy that exposes the service at `https://host.example/memory/` and
forwards the request without `/memory`, set:

```bash
MCP_HTTP_ROOT_PATH=/memory
```

The dashboard, static assets, REST/SSE requests, API documentation, and
auto-detected OAuth endpoint URLs then use the same prefix. If OAuth uses a
public hostname, continue to set `MCP_OAUTH_ISSUER` to the complete external
issuer URL, including the prefix.

TLS:

- `MCP_HTTPS_ENABLED`: `true|false`.
- `MCP_SSL_CERT_FILE`, `MCP_SSL_KEY_FILE`: Certificate and key paths.

## Quality Scoring

- `MCP_QUALITY_SYSTEM_ENABLED`: `true|false` (default `true`).
- `MCP_QUALITY_AI_PROVIDER`: `local` (ONNX, default), `openai-compatible`, `groq`, `gemini`, `auto`, `none`.
- `MCP_QUALITY_LOCAL_MODEL` (default `ms-marco-MiniLM-L-6-v2`), `MCP_QUALITY_LOCAL_DEVICE` (`auto|cpu|cuda|mps|directml`).
- `openai-compatible` requires `MCP_QUALITY_AI_BASE_URL` and `MCP_QUALITY_AI_MODEL`; `MCP_QUALITY_AI_API_KEY` is optional. On macOS 27 this can point at Apple's built-in `fm serve`, see [Apple Foundation Models](../guides/apple-foundation-models.md).
- Cloud providers read `GROQ_API_KEY` and `GEMINI_API_KEY`.
- Search and storage weighting: `MCP_QUALITY_BOOST_ENABLED`, `MCP_QUALITY_BOOST_WEIGHT`, `MCP_QUALITY_IMPLICIT_BLEND_ENABLED`, `MCP_QUALITY_IMPLICIT_WEIGHT`.
- Retention by tier: `MCP_QUALITY_RETENTION_HIGH`, `MCP_QUALITY_RETENTION_MEDIUM`, `MCP_QUALITY_RETENTION_LOW_MIN`, `MCP_QUALITY_RETENTION_LOW_MAX`.

Scoring tiers, the homelab setup against your own LLM, and what each score component
measures: [Memory Quality Guide](../guides/memory-quality-guide.md).

### Rerank (optional, second stage)

`memory_search` can re-order its results with a cross-encoder before truncating to `limit`.
It is off by default and takes **two** switches:

- `MCP_RERANK_ENABLED=true` on the server, and
- `rerank=true` on the individual `memory_search` call.

With either one missing, nothing happens: no model is downloaded or loaded, and the response
order is unchanged.

- `MCP_RERANK_CANDIDATES` (default `40`): the floor of the candidate pool. The pool the
  cross-encoder scores is `max(limit, MCP_RERANK_CANDIDATES)`, so a call that asks for more
  than 40 results scores all of them — this setting is not a cost cap on a large `limit`.
- `MCP_RERANK_MODEL` (default `Xenova/bge-reranker-base`): the cross-encoder, int8 ONNX, about
  266 MB on disk.

Cost, from the measurements in [#1661](https://github.com/doobidoo/mcp-memory-service/pull/1661):
about 6 s p50 per call at 40 candidates and about 3.9 GB peak RSS, against about 0.5 s and 1 GB
without rerank.

Gain: +9.2 p.p. recall@10 on a multilingual 120-case set. The English-only
`cross-encoder/ms-marco-MiniLM-L-6-v2` scored worse there (-8.3 p.p.), so it is not a cheap
substitute on non-English content.

Two limits worth knowing. It is fail-open: if the model cannot be loaded, search returns the
normal order and logs one warning per process. And it can only reorder what the pool holds. The
benchmark in #1661 saw recall@30 unchanged, but that is an observation, not a guarantee: with
`limit=30` the pool is still 40 by default, so a result ranked 31-40 can enter the returned 30.

On the negation pairs introduced for [#1674](https://github.com/doobidoo/mcp-memory-service/issues/1674),
this stage is the only measured answer: cosine scores 0.63 pairwise accuracy, reranking the top
40 lifts it to 0.87 (+24.0 p.p., paired 95% CI [+14.0, +34.0]; 27 of 100 queries improved, 3
regressed; 0.27 s p50 on that corpus, which is 115 short memories). Explicit negation goes
0.88 → 1.00. Implicit negation — a query about the negated fact with no "not" to match on —
moves 0.32 → 0.48 and stays below chance: reranking narrows the gap that motivated the
resonance experiment without closing it. On a multilingual index (`paraphrase-multilingual-MiniLM-L12-v2`)
the baseline is already stronger on this corpus (0.77) and the stage takes it to 0.89 (+12.0 p.p.,
95% CI [+5.0, +19.0], 13 improved / 1 regressed); implicit negation then crosses chance
(0.28 → 0.56). `scripts/benchmarks/benchmark_negation_rerank.py`
repeats the comparison and refuses to score a stage that degraded to fail-open.

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
  - Decay: `MCP_DECAY_ENABLED`, retention by type: `MCP_RETENTION_DECISION`, `MCP_RETENTION_LEARNING`, `MCP_RETENTION_PATTERN`, `MCP_RETENTION_ERROR`, `MCP_RETENTION_OBSERVATION`; for memories stored under legacy type names, use the independently configured `MCP_RETENTION_CRITICAL`, `MCP_RETENTION_REFERENCE`, `MCP_RETENTION_STANDARD`, `MCP_RETENTION_TEMPORARY`. Subtypes inherit their base type's period (e.g. an `insight` memory decays on the `learning` period).
  - Associations: `MCP_ASSOCIATIONS_ENABLED`, `MCP_ASSOCIATION_MIN_SIMILARITY`, `MCP_ASSOCIATION_MAX_SIMILARITY`, `MCP_ASSOCIATION_MAX_PAIRS`.
    - Inferred graph edges store `relationship_confidence` (rounded to three decimals) and `needs_review` in their metadata. Confidence ≤ 0.60 downgrades a typed label to `related`; 0.60 < confidence < 0.85 retains it with `needs_review: true`; confidence ≥ 0.85 retains it with `needs_review: false`. Band decisions use the unrounded score. Similarity, domain-overlap, and ontology guards still apply. These cutoffs are review policy, not calibrated probabilities, and do not imply that a human or model has reviewed an edge.
    - `MCP_CONSOLIDATION_AUTO_SUPERSEDE` (default `false`): keeps inferred edges without hiding either memory. Explicitly setting `true` permits only top-band `contradicts` edges to mark the older memory superseded. The current keyword-only contradiction scorer returns at most 0.75, so those edges always require review and cannot trigger supersession, even with this setting enabled. A dedicated, validated local judgment model is separate follow-up work (#1460). This setting only prevents future supersession: memories that are already superseded stay hidden until their `superseded_by` is cleared, which is a separate step.
  - Clustering: `MCP_CLUSTERING_ENABLED`, `MCP_CLUSTERING_MIN_SIZE`, `MCP_CLUSTERING_ALGORITHM`.
  - Compression: `MCP_COMPRESSION_ENABLED`, `MCP_COMPRESSION_MAX_LENGTH`, `MCP_COMPRESSION_PRESERVE_ORIGINALS`.
  - Forgetting: `MCP_FORGETTING_ENABLED`, `MCP_FORGETTING_RELEVANCE_THRESHOLD`, `MCP_FORGETTING_ACCESS_THRESHOLD`.
- Scheduling (APScheduler-ready):
  - `MCP_SCHEDULE_DAILY` (default `02:00`), `MCP_SCHEDULE_WEEKLY` (default `SUN 03:00`), `MCP_SCHEDULE_MONTHLY` (default `01 04:00`), `MCP_SCHEDULE_QUARTERLY` (default `disabled`), `MCP_SCHEDULE_YEARLY` (default `disabled`).

## Scheduled Session Harvest (Optional)

Autonomously harvests learnings from session transcripts on a timer, in-process (via the consolidation scheduler). Backfills sessions that ended abruptly or were never harvested — the `memory_harvest` tool is local-only (not exposed over remote transports), so the scheduler is the safe place for autonomous harvest.

- `MCP_HARVEST_SCHEDULE`: interval like `6h`, `30m`, `90s`, or a bare number of hours (`6`). Unset/`disabled` → no job (default; opt-in).
- `MCP_HARVEST_SESSION_DIR`: transcripts directory (default `~/.kiro/sessions/cli`; shared with `memory_harvest`).
- `MCP_HARVEST_SCHEDULE_SESSIONS`: max sessions per run (default `50`; delta only — already-harvested sessions are skipped via the harvest tracker).
- `MCP_HARVEST_SCHEDULE_USE_LLM`: `true|false` (default `true`) — use the LLM classifier during scheduled harvest.

Stored candidates carry the `session-harvest` tag and feed consolidation/beliefs on the next cycle.

Both the store path and the evolve path (`_try_evolve`, when a candidate is similar enough to an
existing memory to update it instead of duplicating) stamp `harvest:method:{llm|heuristic}`, so an
evolved memory keeps the same provenance trace as a freshly stored one.

- **Safe pre-deletion**: `SessionHarvester.verify_session_coverage(session_id, threshold=0.9)`
  re-harvests a session in-memory and checks each insight against stored memories, returning
  `{coverage, missing_insights, low_quality_matches, safe_to_delete}`. Use it before deleting
  a source transcript — a session is only `safe_to_delete` when every insight already has a
  strong stored match, so freeing disk never silently loses knowledge.

## Contradiction Detection / NLI (Optional)

Flags contradictions between a newly stored memory and semantically similar existing ones.

- `MCP_NLI_ENABLED`: `true|false` (default `false`). Master switch; nothing runs unless truthy.
- `MCP_NLI_ON_STORE`: `true|false` (default `false`). Also run the pass inline on every store, not just on demand. Only takes effect when `MCP_NLI_ENABLED` is on.
- `MCP_NLI_CONFIDENCE_THRESHOLD`: float (default `0.4`). Minimum NLI confidence for a pair to be registered as a contradiction.
- `MCP_NLI_BACKEND`: `heuristic|cascade|llm` (default `heuristic`). `heuristic` is keyword/pattern-based with no ML deps. `cascade` (alias `llm`) uses the harvest provider chain (`HARVEST_LLM_PROVIDERS`) and degrades gracefully to the heuristic on any error. When unset it resolves to `heuristic`, so no LLM is ever called by accident.
- `MCP_NLI_LLM_TIMEOUT`: seconds (default `30`). Applied once **per provider attempt** inside the harvest chain, so the worst case for a single pair is roughly `timeout × number of providers` before it falls back to the heuristic.

## Harvest LLM Classifier — Pacing & Backoff (Optional)

Rate-limit handling for the harvest classifier (validates candidate memories via the `HARVEST_LLM_PROVIDERS` chain). All optional; unset = prior behavior (switch provider on 429, no pacing).

- `MCP_HARVEST_LLM_MAX_RETRIES`: int (default `0`). Retries against the **same** provider on a 429 before moving to the next. Default `0` preserves the prior behavior (switch provider immediately); set `>0` to opt into exponential backoff.
- `MCP_HARVEST_LLM_BACKOFF_BASE`: seconds (default `1.0`). Exponential backoff base — waits `base × 2**attempt` between retries (1s, 2s, 4s, …).
- `MCP_HARVEST_LLM_REQUEST_DELAY`: seconds (default `0.0` = disabled). Inter-request delay applied before each classifier call, to pace throughput regardless of provider.

## Machine Identification

- `MCP_MEMORY_INCLUDE_HOSTNAME`: `true|false` to tag memories with `source:<hostname>` and include `hostname` metadata.

## Agent Identity (Optional)

- `MCP_AGENT_ID`: default authoring agent id for memories created via `memory_store`. When set, each stored memory records `agent_id` in its metadata unless an explicit `agent_id` argument is passed (the explicit argument wins). When unset and no argument is given, no `agent_id` is written (`null` = unknown), so behavior is unchanged. Useful in a shared multi-agent database to attribute who wrote each memory.

## Logging and Performance

- `LOG_LEVEL`: Root logging level (default `WARNING`).
- `DEBUG_MODE`: When unset, the service raises log levels for performance-critical libs (`sentence_transformers`, `transformers`, `torch`, `numpy`, `onnxruntime`).

## Recommended Defaults by Backend

- SQLite-vec:
  - Defaults enable WAL, busy timeout, optimized cache; customize with `MCP_MEMORY_SQLITE_PRAGMAS`.
  - For multi-client setups, the service auto-detects and may start/use an HTTP coordinator.
- Cloudflare:
  - Ensure required variables are set or the process exits with a clear error and checklist.
- Hybrid (recommended for production):
  - Uses SQLite-vec for 5 ms local reads with background Cloudflare sync. Requires all `CLOUDFLARE_*` variables. Set `MCP_HYBRID_SYNC_OWNER=http` when running alongside an MCP server so only the HTTP server syncs.
  - **HTTP Secondary Backend** (alternative to Cloudflare): Use `MCP_HYBRID_SECONDARY_BACKEND=http` with `MCP_HYBRID_SECONDARY_URL` and optional `MCP_HYBRID_SECONDARY_API_KEY` to sync to a remote MCP Memory Service instance instead of Cloudflare.

### Hybrid HTTP Secondary Backend

For self-hosted or on-premise deployments, the hybrid backend can sync to a remote HTTP MCP Memory Service instead of Cloudflare:

- `MCP_HYBRID_SECONDARY_BACKEND`: Set to `http` to use HTTP secondary backend (default `cloudflare`).
- `MCP_HYBRID_SECONDARY_URL`: Base URL of the remote MCP Memory Service (required when using HTTP backend).
- `MCP_HYBRID_SECONDARY_API_KEY`: Optional API key for authentication with the remote service.
- `MCP_HYBRID_SECONDARY_AUTH_STYLE`: Authentication style - `bearer` (default) or `x-api-key`. Use `x-api-key` when the remote service is behind nginx with basic auth.
- `MCP_HYBRID_SECONDARY_BASIC_USER`: Optional basic auth username for nginx layer authentication.
- `MCP_HYBRID_SECONDARY_BASIC_PASS`: Optional basic auth password for nginx layer authentication.

Example HTTP backend configuration:
```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_HYBRID_SECONDARY_BACKEND=http
export MCP_HYBRID_SECONDARY_URL=https://hub.example.com:8443
export MCP_HYBRID_SECONDARY_API_KEY=your-api-key-here
# For service behind nginx with basic auth:
export MCP_HYBRID_SECONDARY_AUTH_STYLE=x-api-key
export MCP_HYBRID_SECONDARY_BASIC_USER=nginx-user
export MCP_HYBRID_SECONDARY_BASIC_PASS=nginx-password
```

This configuration provides the same local SQLite-vec performance with background sync to your own HTTP-accessible MCP Memory Service hub, avoiding the need for Cloudflare accounts and credentials.

#### Embedding model match (startup safety check)

When the HTTP secondary is active, the client verifies at startup that the hub's
configured embedding model matches its own. Both nodes **must** use the same model
(`MCP_EMBEDDING_MODEL`): memories are re-embedded on store, so a mismatch produces
vectors that are not comparable and silently degrades recall.

- The hub exposes its configured model via an authenticated endpoint,
  `GET /api/health/model` (requires read access), returning `embedding_model`,
  `embedding_dimension` and `backend`.
- On `initialize()`, the client fetches that value and compares it to its own
  `MCP_EMBEDDING_MODEL`. The check is **fail-closed**: a mismatch, a missing model
  field, or an unreachable hub all raise `EmbeddingModelMismatchError` and the
  service refuses to start, naming both the local and the remote model.
- The check only runs for the HTTP secondary. It is a no-op for the Cloudflare
  secondary and for a SQLite-only (non-hybrid) deployment.

If you intentionally change the embedding model, update `MCP_EMBEDDING_MODEL` on
**every** node *and* re-embed the hub's existing memories — restarting with a new
model name alone leaves the stored vectors incompatible. See the troubleshooting
guide entry "HTTP secondary refuses to start: embedding model mismatch".
