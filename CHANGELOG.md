# Changelog

**Recent releases for MCP Memory Service (v10.36.4 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

**Versions v10.36.3 and earlier** – See [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [10.45.1] - 2026-04-30

### Fixed

- **[#794] Remove redundant `import json` inside `mistake_note_add()`**: The `json` module was already imported at the top of `memory_service.py`; the duplicate import inside the function body was flagged by CodeQL alert #393. One-line cleanup with no behavioral change. Thanks to @filhocf. (PR #794)

### Tests

- **[#795] Regression coverage for soft-delete UPDATE guards** (`tests/storage/test_soft_delete_guards.py`): 6 new tests verifying that the `AND deleted_at IS NULL` guards added in PR #783 silently skip tombstoned rows. Covers `_persist_access_metadata_batch`, `_record_conflicts`, `resolve_conflict` (deleted winner and deleted loser), `_touch`, and `update_memory_versioned`. Closes #791. Thanks to @filhocf. (PR #795)

## [10.45.0] - 2026-04-30

### Added

- **[#790] OpenAI-compatible quality scoring provider (LiteLLM / Ollama / MLX / vLLM)**: Adds `openai-compatible` as a new `MCP_QUALITY_AI_PROVIDER` value so homelab and self-hosted users can point quality scoring at any OpenAI `/v1/chat/completions`-compatible endpoint without a cloud API key or the ONNX model. Three new env vars: `MCP_QUALITY_AI_BASE_URL` (required), `MCP_QUALITY_AI_MODEL` (required), `MCP_QUALITY_AI_API_KEY` (optional). Config validation raises `ValueError` if the provider is set without the required vars. New Tier 2 in the fallback chain: local ONNX → openai-compatible → Groq → Gemini → implicit signals. Endpoint failures fall through silently — no exception bubbles to the storage path. 18 new tests in `tests/test_openai_compat_quality.py`. (PR #790)

### Fixed

- **[#783] Soft-delete UPDATE guards — 7 remaining UPDATE statements in `sqlite_vec.py`**: Seven `UPDATE memories SET ...` statements were missing the `AND deleted_at IS NULL` guard, meaning they could operate on soft-deleted (tombstoned) rows. All seven have been patched. No behavioral change for live rows. Continues the series from PRs #557, #558, #562. Follow-up testing tracked in #791. Thanks to @filhocf for the contribution. (PR #783)

## [10.44.0] - 2026-04-29

### Added

- **[#786] Mistake Notes — structured error replay for learning from failures**: Two new MCP tools (`mistake_note_add`, `mistake_note_search`) that store mistake patterns as regular memories with `memory_type='mistake'`. Reuses the existing memory store — no new tables, works on all backends. `mistake_note_add` auto-deduplicates: if a similar pattern exists above the configurable similarity threshold (`MCP_MISTAKE_NOTE_DEDUP_THRESHOLD`, default 0.85), it increments `failure_count` in metadata instead of creating a duplicate. `mistake_note_search` retrieves mistake notes by semantic similarity, filtered to `memory_type='mistake'`. Inspired by [Mistake Notebook Learning](https://arxiv.org/abs/2512.11485). 5 new tests in `tests/services/test_mistake_notes.py`. `models/ontology.py` updated to register `mistake` as subtype of `error`. Thanks to @filhocf for the contribution. (PR #786)

### Changed

- **[#789] CI: ignore glama.ai in markdown link-check** — glama.ai was causing intermittent link-check failures. Added to the ignore list in `.github/workflows/docs-link-check.yml`. (PR #789)

## [10.43.0] - 2026-04-29

### Added

- **[#773] Reciprocal Rank Fusion (RRF) for SQLite-vec hybrid search**: The SQLite-vec backend now supports RRF as an alternative fusion method for hybrid search (vector + keyword). Set `MCP_HYBRID_FUSION_METHOD=rrf` to activate; the default remains `weighted_average` for full backward compatibility. Two additional env vars control RRF behaviour: `MCP_HYBRID_RRF_K` (smoothing constant, default `60` per Cormack, Clarke & Buettcher 2009) and `MCP_HYBRID_RRF_CONSENSUS_BOOST` (score bonus when both retrieval paths rank the same document, default `0.1`). 10 new tests in `tests/storage/test_rrf_fusion.py`. Thanks to @filhocf for the contribution. (PR #773)

### Changed

- **Dependency bumps (Dependabot)**: `actions/checkout` 3 → 6 (PR #777), `docker/login-action` 3 → 4 (PR #778), `actions/upload-artifact` 4 → 7 (PR #779), uv group bump (PR #780): `authlib` 1.6.11 → 1.7.0, `cryptography` 46.0.7 → 47.0.0, `fastapi` 0.135.3 → 0.136.1, `uvicorn` 0.44.0 → 0.46.0, `sse-starlette` 3.3.4 → 3.4.1, `setuptools` 80.10.2 → 82.0.1 (constraint bumped to `<83`), plus `click`, `python-multipart`, `ruff`, `wandb` patches.

## [10.42.1] - 2026-04-29

### Fixed

- **[#775] Milvus: missing `anns_field` in `_check_semantic_duplicate` and `_run_search` causes silent failures on BM25-enabled collections**: Milvus collections with BM25 full-text search (pymilvus ≥ 2.5) contain two vector fields (`vector` dense + `sparse_vector` BM25-generated). Milvus rejects `search()` calls without an explicit `anns_field` when multiple vector fields exist. Two call sites swallowed the error in `except` blocks, causing completely silent failures: semantic deduplication was bypassed (duplicate memories stored silently), and pure vector-search fallback returned empty results on collections where `_has_bm25=False`, `_HYBRID_SEARCH_AVAILABLE=False`, or the hybrid search error-fallback path was taken. The hybrid search happy path (which already specified `anns_field` per `AnnSearchRequest`) was not affected. Does not manifest with Milvus Lite or pre-BM25 collections. Thanks to @henry201605 for the report and fix. (PR #775)

## [10.42.0] - 2026-04-26

### Added

- **[#762] MilvusGraphStorage — knowledge graph for Milvus backend**: New `MilvusGraphStorage` class (~760 lines) stored in a dedicated `{collection}_graph` Milvus scalar collection. Implements the full graph interface: `add_association`, `remove_association`, `find_connected` (application-layer BFS), `shortest_path`, and `get_subgraph`. Edge IDs use `sha256(f"{src}:{tgt}")` — 64-char fixed length, deterministic, and Zilliz Cloud-compatible. The graph collection includes a `_dummy_vec` field (dim=2, `[0.0, 0.0]`) to satisfy Zilliz Cloud's requirement of at least one vector field per collection. (PR #762, @henry201605, verification by @zc277584121)
- **[#762] BM25 hybrid search for Milvus 2.5+**: Milvus storage backend now creates a BM25 function index on the `content` field with `enable_analyzer=True` (required for Zilliz Cloud) and uses `RRFRanker` for combined vector + keyword search. Pre-existing collections without a `sparse_vector` field automatically fall back to vector-only search, so the upgrade is backward-compatible. Schema-level regression test (`test_bm25_content_field_has_enable_analyzer`) guards the `enable_analyzer` flag going forward. (PR #762, @henry201605)
- **[#762] Consolidation integration for Milvus**: `DreamInspiredConsolidator` now detects a Milvus storage backend at consolidation time and initializes a `MilvusGraphStorage` instance via lazy async init guarded by `asyncio.Lock`, preventing races when multiple consolidation cycles overlap. Relationship inference during consolidation cycles now works end-to-end for Milvus deployments. (PR #762, @henry201605)
- **[#762] Zilliz Cloud remote-compatibility test suite**: `TestRemoteMilvusCompat` (env-gated via `MILVUS_TEST_URI`) validates schema correctness against a real remote Milvus / Zilliz Cloud instance. `TestRealContentHashes` (5 tests) covers store/retrieve/delete with 64-char SHA-256 content hashes as used by the graph edge-ID scheme. 25 unit tests in `tests/test_milvus_graph.py` cover `MilvusGraphStorage` in isolation. (PR #762, @henry201605, @zc277584121)

## [10.41.0] - 2026-04-28

### Added

- **[#766] OAuth 2.1 `refresh_token` grant with rotation (MCP SEP-2207)**: Clients that include the `offline_access` scope in their authorization request now receive a refresh token alongside the access token (RFC 6749 §6, OAuth 2.1 §4.3.1). Every successful refresh issues a new access token AND a rotated refresh token while atomically revoking the presented one, preventing replay attacks. Replay detection walks the full `parent_token` chain to the root and bulk-revokes all descendant tokens in a single `UPDATE`, ensuring a stolen token cannot be reused even after the legitimate client has already rotated past it. Discovery (`/.well-known/oauth-authorization-server`) now advertises `refresh_token` in `grant_types_supported` and `offline_access` in `scopes_supported`. Both the Memory and SQLite OAuth storage backends implement the new contract; the SQLite backend uses additive schema changes only (no destructive `ALTER TABLE`). New env vars: `MCP_OAUTH_REFRESH_TOKEN_EXPIRE_DAYS` (default 30, range 1–365). Clients that do not request `offline_access` receive the same response shape as before — zero breaking changes. 17 new unit tests in `tests/unit/test_oauth_refresh.py`; storage parity tests extended in `tests/unit/test_oauth_storage_backends.py`. Documentation updated: `docs/oauth-setup.md`, `README.md`. Thanks to @netizen1119 for the contribution. (PR #766)
- **[#759] `memory_graph` tool for streamable-http MCP server**: Knowledge graph operations (find connected memories, shortest path, subgraph extraction) are now available in the FastMCP streamable-http server, matching the capabilities already present in stdio mode. Introduces a shared `GraphService` business-logic layer under `src/mcp_memory_service/services/graph_service.py` so both server variants reuse the same traversal + error-handling code paths. Graph operations require `sqlite_vec` or `hybrid` storage backends; `milvus` and `cloudflare` backends return a structured unavailability error instead of crashing. 14 unit tests for `GraphService`. Thanks to @henry201605 for the contribution. (PR #759)

### Fixed

- **[#759] Test isolation: `test_graph_service.py` no longer pollutes `sys.modules`**: The lightweight stub used to import `GraphService` without heavy dependencies previously replaced `mcp_memory_service.storage.graph` unconditionally at module-import time. This caused cascading `TypeError: _StubGraphStorage() takes no arguments` failures in `tests/test_graph_traversal.py` and `tests/web/api/test_analytics_graph.py` whenever `test_graph_service.py` was collected first. The stub is now only installed if the real module fails to import, preserving isolation in CI where dependencies are available.
- **[#759] Removed unused `List` import in `graph_service.py`**: CodeQL alert #391 (unused import).

## [10.40.4] - 2026-04-28

### Fixed

- **[#764] quality: ONNX cross-encoder scalar logits no longer silently return 0.5 placeholder score**: The cross-encoder scoring path in `ONNXRankerModel.rerank()` assumed logits always had shape `(N,)`, but the ONNX model can output shape `(1, 1)` for a single-pair input, causing a `TypeError` when indexing with `[i]`. The outer `except Exception` handler swallowed the error and fell back to a neutral 0.5, making quality-boosted search silently rank all results equal. The fix squeezes the logit tensor to 1-D before indexing, making the scorer shape-agnostic. Thanks to @thewusman2025 for the root-cause analysis and patch. (PR #765)

## [10.40.3] - 2026-04-24

### Fixed

- **claude-hooks: socket hang-up on multi-phase retrieval eliminated** (`memory-client.js`): Node.js HTTPS agent defaults to `keepAlive: true`, which causes Uvicorn to close idle sockets after ~5 s. The hook is a one-shot CLI process — keepAlive provides zero benefit and caused subsequent phase requests to reuse dead sockets, producing `ECONNRESET` ("socket hang up"). Added `agent: false` and `Connection: close` header to `_attemptHealthCheck`, `storeMemoryHTTP`, and `_performApiPost`. Also added a 10 s request timeout + timeout handler to `_performApiPost` (previously unset) for consistency with the other two paths and to ensure slow semantic-search queries fail fast rather than starving the overall HOOK_TIMEOUT. Intermittent silent partial-injection failures are resolved.
- **claude-hooks: HOOK_TIMEOUT raised from 9.5 s to 28 s** (`session-start.js`): Phase 0 (git query) + Phase 1 (recent memories) + Phase 2 (tagged memories) with a cold Python cache takes 12–15 s total. The 9.5 s budget expired before `formatMemoriesForContext()` ran, so memories were fetched but the injection block was never written — the hook appeared to "not work" in the Claude Code VSCode extension. Internal constant raised to 28 000 ms.
- **claude-hooks installer: outer process timeout raised from 10 s to 30 s** (`install_hooks.py`): The `timeout` field written into `~/.claude/settings.json` is Claude Code's hard kill limit for the hook process. Must be ≥ internal HOOK_TIMEOUT plus cleanup buffer; was 10 s (less than the old internal limit of 9.5 s leaving no headroom). Updated to 30 s.

## [10.40.2] - 2026-04-23

### Fixed

- **[#756] Docker: ONNX model pre-download now actually executes at build time**: The `python -c "..."` one-liner in `tools/docker/Dockerfile.slim` used `try/except` compound statements with backslash continuations — a construct Python rejects with `SyntaxError`. The shell `|| echo` fallback was silently swallowing the error, so the model cache was never populated. Replaced with a simple expression chain (`import; call; print`) and let the shell `||` fallback handle genuine download failures as originally intended. Cold-start time on `Dockerfile.slim` drops from ~30s to ~3s; prevents Fly.io 40s health-check grace-period timeouts. `Dockerfile` (non-slim) gets the same fix for its `onnxruntime` availability check. Thanks to @netizen1119 for the report, root-cause analysis, and verified fix. (PR #757)

## [10.40.1] - 2026-04-21

### Fixed

- **[#750] CF hybrid sync: `POST /api/sync/force` now reliably completes**: Deduplication logic in the force-sync path now compares against secondary-store hashes before embedding, so already-synced memories are skipped cheaply rather than consuming Cloudflare Workers AI quota. This eliminates the "0 synced / N failed" result that was caused by exhausting the embed rate limit on redundant re-submissions. (PR #753)
- **[#750] CF hybrid sync: sync status flag reflects current health, not lifetime-cumulative failures**: `status.sync_ok` was latching `False` on any historical error and never recovering. It now reflects whether the most-recent sync attempt succeeded, so dashboards and health probes show accurate state after a transient failure is resolved. (PR #751)
- **[#750] CF stats: totals no longer inflated by soft-deleted tombstones**: The Cloudflare statistics endpoint was counting soft-deleted (tombstoned) records in memory totals, making the remote count appear larger than the live dataset. Tombstones are now excluded from count queries. (PR #751)
- **[#750] Reduced timezone-mismatch log noise**: Spurious drift warnings caused by comparing UTC timestamps from Cloudflare against local naive datetimes have been suppressed. (PR #751)

### Changed

- **Dependency bumps (Dependabot)**: `python-semantic-release/python-semantic-release` (PR #748), `actions/setup-python` 5 → 6 (PR #749), `actions/setup-node` 4 → 6 (PR #747).

## [10.40.0] - 2026-04-22

### Added

- **[#721] Milvus storage backend (Lite / self-hosted / Zilliz Cloud)**: New fourth storage backend implementing the full `MemoryStorage` interface against Milvus. Supports three deployment modes from the same code path — Milvus Lite (zero-dep local `.db` file, ideal for scripts and tests), self-hosted Milvus via Docker (recommended for MCP servers and single-tenant deployments), and Zilliz Cloud (managed service for team/production use). ~1,750 lines of new code, 39 Milvus-specific tests. Activate with `MCP_MEMORY_STORAGE_BACKEND=milvus`. See `docs/milvus-backend.md` for full deployment guide. (PR #721, @zc277584121)
- **[#721] `backend:milvus` label + `.github/CODEOWNERS` + `test-milvus-docker` CI job**: Issue tracker label for routing Milvus bug reports; `@zc277584121` added to CODEOWNERS for `src/mcp_memory_service/storage/milvus.py` with a 6-month SLA commitment; dedicated Docker-based Milvus smoke-test job in Main CI/CD Pipeline. (PR #721)
- **[#740] Claude Code plugin manifest shape validation**: CI smoke test now validates `plugin.json` against the full Claude Code plugin spec (author object, tools array, schema fields) using structured JSON shape checks — catches regressions that `JSON.parse` alone misses. (PR #740)

### Security

- **[#745] oauth**: Harden the authorization-code redirect response against CodeQL
  alerts `py/reflective-xss` (#385) and `py/url-redirection` (#382).
  `_build_redirect_url` now rejects `javascript:`, `data:`, `vbscript:`,
  `file:`, `about:`, and `blob:` schemes (RFC 8252 custom schemes like
  `myapp://callback` remain supported). The meta-refresh URL is
  HTML-attribute-escaped and the JS redirect string has `</` escaped to
  `<\/` so it cannot break out of the `<script>` element.
  `validate_redirect_uri` already allowlists the URI against the registered
  client; these are defense-in-depth guards for the code-scanning findings. (PR #745)

### CI

- **[#741] Docs link-check: ignore milvus.io and docs.zilliz.com**: Unblocks the link-checker on all Milvus documentation. (PR #741)
- **[#721] Milvus CI hardening**: Docker image tag pinned for reproducibility, `docker-compose` standalone manifest added for older Docker versions, segment-sealing wait added to smoke test to prevent intermittent failures. (PR #721)

## [10.39.1] - 2026-04-19

### Fixed

- **plugin**: `plugin.json` `author` field now uses the Claude Code plugin spec's required object format (`{"name": "..."}`) instead of the pre-spec string form. Unblocks `/plugin install mcp-memory-service` — thanks @yingzhi0808 for the report (#738) and the fix (#739).

## [10.39.0] - 2026-04-19

### Added
- **plugin**: Claude Code plugin packaging for the claude-hooks suite. Install via `/plugin marketplace add doobidoo/mcp-memory-service` + `/plugin install mcp-memory-service`. Ships with `.mcp.json`, hook wiring, and self-healing `ensure-server.js`. Coexists with the legacy `install_hooks.py` installer — see [claude-hooks/PLUGIN.md](claude-hooks/PLUGIN.md). Closes #530 (plugin packaging track). (PR #736)

### Changed
- **hooks**: Route memory writes through `MemoryClient.storeMemory()` — enables HTTP-primary + MCP-fallback for `session-end` and `auto-capture` hooks. Closes silent write-failure path documented in #530 (Option B). (PR #735)

## [10.38.4] - 2026-04-19

### Fixed

- **[#733] MCP: return HTTP 202 for JSON-RPC notifications on `/mcp`**: JSON-RPC 2.0 §4.1 forbids servers from replying to notifications (messages without `id`), and MCP Streamable HTTP requires HTTP 202 Accepted with an empty body in that case. The `/mcp` handler previously fell through to method dispatch and returned a `-32601 Method not found` error for `notifications/initialized`. Tolerant clients (Claude Code) ignored it; strict clients (Codex's `rmcp`) treated the response as a handshake failure and refused to start the MCP server. Fixed by short-circuiting to `Response(status_code=202)` at the top of `mcp_endpoint` whenever `request.id is None`. Added regression tests for the 202/empty-body path and the `initialize` happy path. (PR #733)

## [10.38.3] - 2026-04-17

### Fixed

- **[#728] Dashboard: auto-check updates on Server tab open + accurate initial label**: The Server tab now automatically triggers an update check when opened, and displays an accurate initial label before the first check completes, eliminating stale/misleading status on first render. (PR #728)
- **[#731] API: add `total_pages` to `list_memories` return**: The `list_memories` REST API response now includes a `total_pages` field alongside `total_count` and `page`, enabling correct client-side pagination without extra requests. (PR #731)
- **[#730] Dashboard: render knowledge-graph edges for non-canonical relationship types**: Edges whose `type` had no matching CSS custom property were rendered invisible. Added fallback color resolution so all relationship types display correctly in the graph view. (PR #730)

### Changed

- **[#725] Dependency bump**: `pypdf` 6.10.1 → 6.10.2 (Dependabot)
- **[#726] Dependency bump**: `authlib` 1.6.10 → 1.6.11 (Dependabot)
- **[#727] CI: skip full test suite on docs-only changes**: Added `paths-ignore` for `docs/**`, `*.md`, `.claude/**`, `LICENSE`, `.gitignore` to `main.yml`. Docs-only PRs no longer trigger the 1587-test pytest suite, Docker build, or CodeQL scan.

## [10.38.2] - 2026-04-16

### Fixed

- **[#723] Windows PS7+: replace removed `ICertificatePolicy` with `ServerCertificateValidationCallback`**: `lib/server-config.ps1` used `Add-Type` to implement the `System.Net.ICertificatePolicy` interface, which was removed in .NET Core / .NET 5+. On PowerShell 7+ this caused "Cannot add type. Compilation errors occurred" at script load time, breaking all Windows update/service scripts. Replaced with a `[System.Net.ServicePointManager]::ServerCertificateValidationCallback` assignment, scoped to PS 5.1 only (`$PSVersionTable.PSVersion.Major -lt 6`) to avoid a global process-wide callback leak on PS 7+. (PR #723)
- **[#723] Windows PS7+: add `Get-McpWebRequestExtraParams` helper for HTTPS cert bypass**: PS 7+ `Invoke-WebRequest` / `Invoke-RestMethod` use `HttpClient` internally and ignore `ServicePointManager` entirely. Added `Get-McpWebRequestExtraParams` to `lib/server-config.ps1` that returns `@{ SkipCertificateCheck = $true }` on PS 7+ HTTPS targets. All 7 web-request call sites across `update_and_restart.ps1`, `manage_service.ps1`, and `run_http_server_background.ps1` now splat these extra params. (PR #723)
- **[#723] Windows: defer `lib/server-config.ps1` sourcing in `update_and_restart.ps1` until after `git pull` + `pip install`**: Previously the lib was sourced at script load time, so a buggy checked-out version of the lib would fail immediately, preventing the pull from delivering the fix. The script now sources the lib after the update steps complete, enabling self-healing on the next run. (PR #723)

### Changed

- **`scripts/service/mcp-memory.service` portability cleanup**: Replaced hardcoded `/home/hkr/` paths with the `%h` systemd specifier across `WorkingDirectory`, `PATH`, `PYTHONPATH`, and `ExecStart`, and removed the `User=hkr` / `Group=hkr` directives (user services run as the invoking user by default). Matches the convention already used in `scripts/service/mcp-memory-http.service`. Closes the follow-up flagged in PR #706 / #719.

### Documentation

- **[#662] Dead-reference cleanup across active docs**: Removed stale references to port 8443 (current default is 8000), the removed `python install.py` bootstrap, and the retired ChromaDB backend from 15 active documentation files. Rewrote the Homebrew and multi-client integration guides around the current `memory server --http` CLI entry point and `MCP_HTTP_ENABLED=true` + `MCP_MEMORY_USE_HOMEBREW_PYTORCH=1` env patterns (old installer flags no longer exist). Switched `scripts/ci/check_dead_refs.sh` from warning-only to `exit 1` on findings. (PR #702)
- **[#703] Rewrote `docs/guides/STORAGE_BACKENDS.md` for the current 3-backend model**: Previous guide compared SQLite-vec vs ChromaDB throughout — actively misleading since ChromaDB was removed from `SUPPORTED_BACKENDS` in v7.x. New guide covers SQLite-vec / Cloudflare / Hybrid with a 3-column comparison, backend-specific "When to Choose X" sections, a deployment matrix reframed around connectivity/privacy/scale, performance numbers sourced from CLAUDE.md, and per-backend configuration blocks sourced from `.env.example`. Net −80 lines. Migration section now points to the real scripts in `scripts/migration/` (with correct subcommand/argument syntax) and flags Cloudflare → SQLite-vec as the one direction without a dedicated script (hybrid-mode workaround documented). (PR #712)
- **[#713] Eliminated all current-tense ChromaDB references from active docs**: Swept 31 files to remove ChromaDB as a current backend option — config examples (`MCP_MEMORY_STORAGE_BACKEND=chromadb`), docker compose/run templates (`chromadb` → `sqlite_vec`, `chroma_db` → `sqlite_data`), `.[chromadb]` pip extras, `--with-chromadb` installer flag, comparison tables, architecture diagrams, and visible SVG text. Historical references preserved where they link users to `docs/guides/chromadb-migration.md`. (PR #714, net −159 lines)
- **[#713] CI hardening against ChromaDB regressions**: `scripts/ci/check_dead_refs.sh` now also blocks `MCP_MEMORY_CHROMA_PATH` and `MCP_MEMORY_CHROMADB_{HOST,PORT,SSL,API_KEY}` env vars as hard dead refs, plus `chromadb` as a soft dead ref with an explicit `SOFT_REF_ALLOWLIST` for the 13 files that legitimately carry historical/migration pointers or external-project (MemPalace) benchmark context. Script refactored to support per-ref exclusions. (PR #714)
- **[#706] Hardened `docs/deployment/systemd-service.md` for LAN/network exposure**: Added a "Network exposure hardening" subsection with five concrete recommendations — bind to a specific LAN interface instead of `0.0.0.0`, apply per-source-IP firewall rules (`ufw allow from <IP>`), restrict the database parent directory with `chmod 700` to cover SQLite sidecar files (`*-wal`, `*-shm`), guidance on TLS termination (reverse proxy) or WireGuard/Tailscale overlays for untrusted networks, and a warning about shared client config files (e.g. `~/.claude.json` symlinked across hosts) that cause every reader to hit the same service URL. Also swapped the hardcoded `/home/hkr/` path in the in-guide "Service File Structure" and LAN examples for the portable `%h` systemd specifier (matches the shipped `scripts/service/mcp-memory-http.service` template). The corresponding `scripts/service/mcp-memory.service` template cleanup is tracked separately — see the `### Changed` entry above. (PR #706)

## [10.38.1] - 2026-04-15

### Fixed

- **[#697] OAuth: accept native loopback redirect ports (RFC 8252)**: Native apps like OpenCode register a loopback redirect URI (e.g. `http://127.0.0.1`) without a port, then listen on an ephemeral port chosen at runtime. The authorization server now matches loopback URIs by scheme and host only, ignoring the port, in conformance with RFC 8252 §7.3. Previously, the port mismatch caused authorization to fail for native app clients. (PR #697, +109 test lines)
- **[#704] CLI: import missing `generate_content_hash` in ingestion**: `memory ingest-document` silently stored 0 chunks due to a `NameError` on `generate_content_hash` which was used but never imported in `src/mcp_memory_service/cli/ingestion.py`. (PR #704)
- **[#705] Server: `--sse-host` / `--sse-port` CLI flags now take effect**: Config module constants were frozen at import time, so the CLI flags had no effect on the transport's bind address. The transport now re-reads the environment at startup instead of using the cached constants. (PR #705)

### Changed

- **[#707] CI: bump `docker/metadata-action` 5 → 6** (PR #707)
- **[#708] CI: bump `docker/build-push-action` 5 → 7** (PR #708)
- **[#709] CI: bump `docker/setup-buildx-action` 3 → 4** (PR #709)

## [10.38.0] - 2026-04-14

### Added

- **[#631] Claude Code SessionEnd auto-harvest hook**: New opt-in hook `claude-hooks/core/session-end-harvest.js` that automatically calls `POST /api/harvest` at the end of every Claude Code session. Safe-by-default design: disabled by default (`sessionHarvest.enabled: false`), forces `dry_run: true` on first run (sentinel file `~/.claude/mcp-memory-harvest-first-run.done`), and enforces a minimum session message threshold (`minSessionMessages`, default 10) to skip trivially short sessions. (PR #711, issue #631)
- **[#631] Graceful failure guarantees**: The hook enforces a 5-second timeout and catches all exceptions — it never throws and never blocks session end. HTTP failures and timeouts are logged to stderr and silently ignored. (PR #711)
- **[#631] Security: TLS certificate validation opt-in only**: Self-signed certificate acceptance (`allowSelfSignedCerts`) is disabled by default and logs a warning when enabled, preventing silent MITM exposure for users who copy config templates. (PR #711)
- **[#631] Standalone CLI entry point**: The hook reads `transcript_path` and `cwd` from Claude Code's stdin JSON, making it usable as a direct `command:` entry in `.claude/settings.json` without any wrapper script. (PR #711)
- **[#631] Supporting files**: `claude-hooks/tests/session-end-harvest.test.js` (9 tests), `claude-hooks/README-SESSION-HARVEST.md` (user documentation), `claude-hooks/config.template.json` (`sessionHarvest` + `hooks.sessionEndHarvest` config sections). (PR #711)

### Tests

- **[#631] 9 new Node.js hook tests** in `claude-hooks/tests/session-end-harvest.test.js` covering: disabled-by-default, short-session skip, first-run dry-run force, subsequent runs honor config, timeout non-fatal, HTTP failure non-fatal, API key precedence, TLS opt-in, and transcript message counting. (PR #711)

## [10.37.0] - 2026-04-14

### Added

- **[#630] `POST /api/harvest` HTTP endpoint**: New REST endpoint that exposes the existing `memory_harvest` MCP tool over HTTP, enabling Session Harvest to be triggered from scripts, cron jobs, CI pipelines, or the dashboard without an active MCP session. Request fields mirror the MCP tool: `sessions`, `session_ids`, `use_llm`, `dry_run`, `min_confidence`, `types`, `project_path`. Auth via existing `require_write_access` dependency. New router: `src/mcp_memory_service/web/api/harvest.py` with Pydantic request/response models. (PR #710)
- **[#630] Security hardening for `project_path`**: The `project_path` parameter in `/api/harvest` accepts only relative names under `~/.claude/projects/`. Absolute paths, `..` path-traversal components, and symlink escapes all return HTTP 400. Addresses CodeQL path-injection findings #383 and #384. (PR #710)
- **[#630] Async hygiene in `harvester.py`**: `harvest_and_store` now offloads synchronous `_harvest_file` reads via `asyncio.to_thread`, keeping the event loop unblocked during file I/O. Benefits both MCP and HTTP callers. (PR #710)

### Tests

- **[#630] 10 new tests** in `tests/web/api/test_harvest_api.py` covering endpoint authentication, dry-run mode, path-traversal rejection, and symlink escape prevention. (PR #710)

## [10.36.8] - 2026-04-14

### Fixed

- **[#664] Event-loop blocking paths in `SqliteVecMemoryStorage.initialize()`**: Pragma application in `_connect_and_load_extension` now runs in a worker thread under `_conn_lock` via `_run_in_thread` instead of executing synchronously on the event loop. `_initialize_hash_embedding_fallback` is now async and wraps `_get_existing_db_embedding_dimension` in `_run_in_thread`. The sqlite-vec extension is not thread-safe so `asyncio.to_thread` (used in an earlier iteration) was replaced with `_run_in_thread` to ensure proper `_conn_lock` protection. (PR #700)

## [10.36.7] - 2026-04-14

### Security

- **[#698] Bumped pygments to 2.20.0**: Resolves CVE-2026-4539 (GHSA-5239-wwwm-4pmq, ReDoS via inefficient regex for GUID matching). Transitive dependency via rich. (PR #698)

## [10.36.6] - 2026-04-14

### Security

- **[#690] Bumped cryptography to 46.0.7**: Resolves CVE-2026-39892 (buffer overflow in non-contiguous buffer handling). (PR #690)

## [10.36.5] - 2026-04-14

### Fixed

- **[#689] Cloudflare Vectorize API v1→v2**: Updated `scripts/installation/setup_cloudflare_resources.py` to use the v2 Vectorize API endpoint, fixing error 1010 "incorrect_api_version" during Cloudflare resource setup. (PR #689, @mychaelgo)
- **[#689] `test_cloudflare_backend.py` test script fixes**: Added required `content_hash` argument to `Memory()` constructor and fixed `sys.path` to correctly locate the `src/` package directory. (PR #689, @mychaelgo)

## [10.36.4] - 2026-04-10

### Fixed

- **[#687] `Get-McpApiKey` returned first character of API key instead of full key**: A Gemini-suggested refactor in v10.36.3 replaced a working implementation with `($matches[1], $matches[2], $matches[3] | Where-Object { $_ -ne $null })[0]`. Unmatched regex capture groups are absent from `$matches` (not `$null`), so when only one group matched the comma expression produced a single-element string, which PowerShell enumerated to its `Char` array — making `[0]` return `'b'` instead of `bxvWZwrI...`. This broke `manage_service.ps1 status` for all Windows users: Version and Backend showed `(unavailable - set MCP_API_KEY in .env for details)` even when the key was correctly configured. Fixed by replacing the comma expression with an explicit `if/elseif` chain using `$matches.ContainsKey(N)` and `[string]` casts. Verified live: returns full 43-character key string, `manage_service.ps1 status` correctly displays Version and Backend.

