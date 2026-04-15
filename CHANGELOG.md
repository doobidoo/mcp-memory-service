# Changelog

**Recent releases for MCP Memory Service (v10.25.0 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

**Versions v10.24.0 and earlier** – See [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- **[#662] Dead-reference cleanup across active docs**: Removed stale references to port 8443 (current default is 8000), the removed `python install.py` bootstrap, and the retired ChromaDB backend from 15 active documentation files. Rewrote the Homebrew and multi-client integration guides around the current `memory server --http` CLI entry point and `MCP_HTTP_ENABLED=true` + `MCP_MEMORY_USE_HOMEBREW_PYTORCH=1` env patterns (old installer flags no longer exist). Switched `scripts/ci/check_dead_refs.sh` from warning-only to `exit 1` on findings. (PR #702)
- **[#703] Rewrote `docs/guides/STORAGE_BACKENDS.md` for the current 3-backend model**: Previous guide compared SQLite-vec vs ChromaDB throughout — actively misleading since ChromaDB was removed from `SUPPORTED_BACKENDS` in v7.x. New guide covers SQLite-vec / Cloudflare / Hybrid with a 3-column comparison, backend-specific "When to Choose X" sections, a deployment matrix reframed around connectivity/privacy/scale, performance numbers sourced from CLAUDE.md, and per-backend configuration blocks sourced from `.env.example`. Net −80 lines. Migration section now points to the real scripts in `scripts/migration/` (with correct subcommand/argument syntax) and flags Cloudflare → SQLite-vec as the one direction without a dedicated script (hybrid-mode workaround documented). (PR #712)
- **[#713] Eliminated all current-tense ChromaDB references from active docs**: Swept 31 files to remove ChromaDB as a current backend option — config examples (`MCP_MEMORY_STORAGE_BACKEND=chromadb`), docker compose/run templates (`chromadb` → `sqlite_vec`, `chroma_db` → `sqlite_data`), `.[chromadb]` pip extras, `--with-chromadb` installer flag, comparison tables, architecture diagrams, and visible SVG text. Historical references preserved where they link users to `docs/guides/chromadb-migration.md`. (PR #714, net −159 lines)
- **[#713] CI hardening against ChromaDB regressions**: `scripts/ci/check_dead_refs.sh` now also blocks `MCP_MEMORY_CHROMA_PATH` and `MCP_MEMORY_CHROMADB_{HOST,PORT,SSL,API_KEY}` env vars as hard dead refs, plus `chromadb` as a soft dead ref with an explicit `SOFT_REF_ALLOWLIST` for the 13 files that legitimately carry historical/migration pointers or external-project (MemPalace) benchmark context. Script refactored to support per-ref exclusions. (PR #714)

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

## [10.36.3] - 2026-04-10

### Fixed

- **[#685] Dashboard Settings modal version row showed N/A permanently**: `SYSTEM_INFO_CONFIG.settingsVersion` in `app.js` was still configured to fetch from `api: 'health'` (the public endpoint). The v10.21.0 security hardening (GHSA-73hc-m4hx-79pj) removed `version`, `timestamp`, and `uptime_seconds` from `/api/health`, so the Settings modal Version row had been displaying `N/A` for ~4 months. Fixed by pointing the config entry to `'detailedHealth'`, consistent with the header version badge which was already migrated. (PR #685)
- **[#685] `manage_service.ps1 status` showed blank Version and Backend**: `Get-ServerStatus` parsed `version` and `storage_backend` from the public `/api/health` response, fields that no longer exist since v10.21.0. Migrated to `/api/health/detailed` with Bearer auth when `MCP_API_KEY` is available; `Show-Status` now displays real version/backend values or a clear hint when the key is not configured. Introduced `Get-McpApiKey` helper in `lib/server-config.ps1` that parses `MCP_API_KEY` from `.env` with support for trailing comments, quoting, and whitespace — keys containing `#` inside quoted values are handled correctly. Falls back to `$null` gracefully when key is absent. (PR #685, regression introduced by GHSA-73hc-m4hx-79pj)

### Documentation

- Added English-language policy to issue templates (bug, feature, performance) and CONTRIBUTING.md (PR #683)

## [10.36.2] - 2026-04-10

### Fixed

- **[#682] Hardcoded server URLs in Windows management scripts**: All 5 Windows PowerShell management scripts (`manage_service.ps1`, `run_http_server_background.ps1`, `install_scheduled_task.ps1`, `uninstall_scheduled_task.ps1`, `update_and_restart.ps1`) used a hardcoded `http://127.0.0.1:8000` URL regardless of `.env` configuration. Health checks failed silently as soon as a user enabled HTTPS (`MCP_HTTPS_ENABLED=true`) or changed the port (`MCP_HTTP_PORT`). Introduced `scripts/service/windows/lib/server-config.ps1` as a shared helper that parses `.env` for `MCP_HTTP_HOST`, `MCP_HTTP_PORT`, and `MCP_HTTPS_ENABLED` and returns a `BaseUrl`/`HealthUrl` hashtable. All scripts now dot-source this helper. Falls back to `http://127.0.0.1:8000` if `.env` is absent or variables are unset — fully backward-compatible. (PR #682)
- **[#682] Silent Python stdout/stderr dropout in background server wrapper**: `run_http_server_background.ps1` used .NET `OutputDataReceived` event handlers that referenced `$script:LogFile`, a variable not captured in the event handler runspace. Every line of Python stdout and stderr was silently discarded. When Python crashed during initialization, no error was ever surfaced in the log. Replaced with `Start-Process -RedirectStandardOutput`/`-RedirectStandardError` which writes directly to `http-server-python.log` and a `.err` sidecar, eliminating the runspace capture problem entirely. (PR #682)
- **[#682] Log overwrite destroys crash evidence in restart loop**: `Start-Process` overwrites the redirect target file on each invocation. The previous size-based rotation (`> 10MB`) meant that on a crash-restart cycle, the crash output from attempt N was silently destroyed by attempt N+1 before it could be inspected. Rotation is now unconditional at the start of each loop iteration: the current log is renamed to `.old` before `Start-Process` is called, preserving the most recent crash output. (PR #682, Gemini review feedback)

### Changed

- **[#682] `.env` regex strips trailing inline comments**: The `.env` parser in `server-config.ps1` now uses `([^#\r\n]*?)` to capture values, so `MCP_HTTP_PORT=8001 # my custom port` parses correctly as `8001` rather than failing `[int]::TryParse`. (PR #682, Gemini review feedback)
- **[#682] TLS protocol uses additive `-bor` instead of replacement**: `Enable-McpSelfSignedCertBypass` previously replaced the session's `SecurityProtocol` with `Tls12` outright, which would have disabled TLS 1.3 if already enabled. Now uses `-bor` to add `Tls12` to the existing protocol set without removing newer protocols. (PR #682, Gemini review feedback)

## [10.36.1] - 2026-04-10

### Fixed

- **[#678] SQLite-vec segfault under concurrent worker-thread access**: The `sqlite-vec` extension is not thread-safe; `check_same_thread=False` only disables Python's safety check. Concurrent calls to `self.conn` from multiple `asyncio.to_thread()` workers could crash inside the C extension. Added a per-storage `threading.Lock` (`_conn_lock`) and a `_run_in_thread()` helper that serializes every worker-thread DB call. Routed `_execute_with_retry`, migrations, FTS5 init, embedding-dimension probe, and conflict detection through the new helper. Observed as `Fatal Python error: Segmentation fault` in `_get_stats` on Ubuntu CI. (PR #678)
- **[#678] SQLite-vec use-after-close segfault on hybrid shutdown**: `HybridStorage.close()` cancels background sync tasks, but cancellation only delivers `CancelledError` to the outer coroutine — an `asyncio.to_thread()` worker mid-query keeps running. The subsequent `primary.close()` then freed the connection underneath the worker, crashing sqlite3/sqlite-vec. `SqliteVecMemoryStorage.close()` now acquires `_conn_lock` before closing, so any in-flight worker finishes first. (PR #678)
- **[#678] Corrupt `connection_types` in conflict graph edges**: `_record_conflicts` stored `connection_types` as the bare string `"semantic"` instead of a JSON-encoded list, which then crashed downstream graph readers with `JSONDecodeError`. Encoded as `json.dumps(["semantic"])` and hardened `GraphStorage.get_subgraph()` to tolerate legacy corrupt rows instead of crashing the request. (PR #678)

### Tests

- **[#678] Thread-safety test rewrite**: `test_execute_with_retry_does_not_block_loop` previously asserted that two DB operations run truly in parallel on a shared connection — exactly the pattern that caused the segfault. Rewritten to verify the actual property (the asyncio event loop stays responsive while a slow DB op runs in a worker thread), using a 5×50 ms `asyncio.sleep` probe. (PR #678)
- **[#678] Stale ontology type count**: Bumped `test_total_type_count` from 75 → 77 after new burst types were added without updating the assertion. (PR #678)
- **[#678] Semantic dedup collision in graph edge cleanup test**: `test_delete_by_tag_removes_graph_edges` stored near-duplicate content ("Tagged mem 1"/"Tagged mem 2") which dedup dropped when the embedding model was loaded by prior tests in the suite. Passed `skip_semantic_dedup=True` since the test's subject is edge cleanup, not dedup semantics. (PR #678)

## [10.36.0] - 2026-04-09

### Added

- **[#673] OpenCode memory awareness integration**: Added an official `opencode/` integration with a minimal read-only plugin for session-start retrieval, system-context injection, and compaction-context injection via the documented HTTP API. Includes setup docs and example config for local OpenCode plugin installation. (PR #673, contributor: @irizzant, closes #671)

### Fixed

- **[#675] Lite package version sync**: `mcp-memory-service-lite` was stuck at 8.76.0 on PyPI because `pyproject-lite.toml` was never updated by the release automation. Synced to 10.35.0, added `pyproject-lite.toml` to semantic-release `version_variable` list, and added fallback sync step in CI workflow. (PR #675, closes #672)

## [10.35.0] - 2026-04-08

### Added

- **Session-level memory ingestion (`memory_store_session` MCP tool)**: New MCP tool that stores a full conversation as a single memory unit. All turns are concatenated as `[role] content` lines and stored with `memory_type=session` and auto-tagged `session:<id>`. Complements turn-level storage: turn-level is best for precise fact retrieval; session-level improves session-recall benchmarks. 10 new handler tests added.
- **`POST /api/sessions` HTTP endpoint**: REST endpoint for session-level ingestion, mirroring the MCP tool for HTTP API consumers. 7 new endpoint tests added (total: +17 tests, now 1,537).
- **`--ingestion-mode session|turn|both` flag for LongMemEval benchmark**: Allows apples-to-apples comparison of ingestion strategies. With session mode: R@5 improves from 80.4% to **86.0%** (+5.6%), with largest gains in multi-session (+15.2%) and temporal-reasoning (+10.6%) categories.
- **`session` and `conversation_turn` memory types**: Added to the memory type ontology so session-level memories are correctly classified and queryable by type.
- **MemPalace comparison in README**: New section documenting honest benchmark gap context and explaining why session-level ingestion narrows the gap to MemPalace-style approaches.

## [10.34.0] - 2026-04-08

### Added

- **[#665] LongMemEval benchmark**: New benchmark implementation evaluating semantic memory retrieval against the LongMemEval dataset (500 single-session questions, zero LLM API calls). Results: R@5 80.4%, R@10 90.4%, NDCG@10 82.2%, MRR 89.1%. Includes HuggingFace dataset loader (`scripts/benchmarks/longmemeval_dataset.py`), CLI orchestrator with ablation support (`scripts/benchmarks/benchmark_longmemeval.py`), and `ndcg_at_k` metric added to `locomo_evaluator.py`. (PR #665)
- **[#665] `docs/BENCHMARKS.md`**: New benchmark results documentation covering LongMemEval methodology, results table, and comparison context. (PR #665)

## [10.33.0] - 2026-04-06

### Changed

- **[#637/#663] Wrap all remaining direct SQLite calls in `_execute_with_retry`**: Eliminated ~119 direct `self.conn.execute()` / `executemany()` / `commit()` calls in async methods of `SqliteVecMemoryStorage`. All DB operations now run via `asyncio.to_thread()` through `_execute_with_retry`, preventing up to 15-second event-loop freezes under concurrent load (`busy_timeout=15000` WAL mode). Key correctness guarantees: cursor thread-safety (all `fetchall()`/`fetchone()`/`rowcount` accesses moved inside closures), DDL idempotency (`ALTER TABLE ADD COLUMN` guarded by `PRAGMA table_info()` check), `_savepoint_lock` (`asyncio.Lock`) added to serialize SAVEPOINT-based writes (`store`/`store_batch`/`evolve_memory`) at the coroutine level without blocking the event loop, and unique SAVEPOINT names (`store_<hex>`, `batch_<hex>`, `evolve_<hex>`) to prevent name collision. Removes the `TODO(#637)` comment added in v10.31.0. (PR #663)

### Fixed

- **[#637/#663] `_record_conflicts` silent data loss**: `_record_conflicts` was writing conflict tags and graph edges inside `_execute_with_retry` threads but never called `self.conn.commit()`, causing all conflict data to be silently discarded. Fixed by adding `self.conn.commit()` inside the `_record_all_conflicts` closure. (PR #663)
- **[#637/#663] `store_batch` outer-scope mutation**: `batch_insert` closure now returns its results list instead of mutating an outer-scope variable, making the pattern safe against future concurrency model changes. (PR #663)

## [10.32.0] - 2026-04-06

### Added
- **[#656] `/health` endpoint on SSE and Streamable HTTP transports**: Added a `/health` endpoint to the SSE and Streamable HTTP transport servers (port `MCP_SSE_PORT`, default 8765) for external monitoring — load balancers, Docker healthchecks, and Kubernetes liveness/readiness probes can now query transport health independently of the main HTTP API. (PR #656, contributor: @Lobster-Armlock)
- **[#656] Configurable transport timeouts**: Added `MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE` (default 5s) and `MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN` (default 30s) environment variables to control uvicorn timeouts for MCP transport instances, enabling tuning for different deployment environments. (PR #656, contributor: @Lobster-Armlock)
- **[#657] Optional DCR registration key protection**: Added `MCP_DCR_REGISTRATION_KEY` environment variable to optionally protect the `/oauth/register` Dynamic Client Registration endpoint. When set, requests must include `Authorization: Bearer <key>` (timing-safe comparison via `secrets.compare_digest`). Backward-compatible — when unset, DCR remains open per RFC 7591. (PR #657, contributor: @irizzant)

## [10.31.2] - 2026-04-03

### Fixed

- **[#648] Use `_safe_json_loads` consistently for metadata parsing**: Replaced remaining bare `json.loads` calls in `get_largest_memories()` and `get_graph_visualization_data()` with `_safe_json_loads` helper, ensuring consistent error handling for malformed metadata across all SQLite-Vec storage methods. Also corrected the context string from `"get_graph_data"` to `"get_graph_visualization_data"` for accurate error logs. (PR #648, contributor: @lawrence3699)
- **[#649] Handle non-JSON error responses in HTTP client and embedding API**: Wrapped `response.json()` calls on error paths in `http_client.py` (`store()`, `delete()`) with `try/except (ContentTypeError, ValueError)` to handle HTML/empty responses from reverse proxies. Added similar guards in `external_api.py` for `_verify_connection()` and `encode()`. (PR #649, contributor: @lawrence3699)
- **[#650] Correct upload progress tracking for single and batch uploads**: Removed broken single-file progress formula that always evaluated to 100%. Added per-file `session.progress` updates during batch processing so the polling endpoint returns smooth 0→100% progress instead of jumping at completion. (PR #650, contributor: @lawrence3699)

### Changed

- **Repo root cleanup**: Moved 8 legacy documentation files (`FIXES_COMPLETE.md`, `IMPLEMENTATION_SUMMARY.md`, `TEST_ADDITIONS_SUMMARY.md`, `TEST_VALIDATION_REPORT.md`, `AUTH_FLOW_DIAGRAM.md`, `test_auth_implementation.md`, `test_fixes.py`, `.commit-message`) to `archive/docs-root-cleanup-2026-04-02/`. Removed redundant `venv/` directory (keeping `.venv/` as the active Python 3.11 environment).
- **`.claude/` cleanup**: Archived obsolete consolidation handoff docs (v8.47.1, Dec 2025), completed tool-optimization task plans (PR #373), and removed tracked config backup duplicates (`settings.local.json.backup`, `.local`).
- **Agent consolidation (84% reduction)**: Merged `amp-bridge` + `amp-pr-automator` into single `amp-automation` agent (889 → 120 lines). Trimmed `github-release-manager` (640 → 144 lines) and `gemini-pr-automator` (881 → 122 lines) by removing duplicate sections and referencing existing scripts instead of embedding them. Slimmed `/release` command to thin wrapper (97 → 26 lines). Total: 2,507 → 412 lines.
- **Docs**: Added pending v10.31.0 blog post, LoCoMo benchmark analysis, DevBench/LoCoMo plans and specs, new learned instincts, and external data parser guideline to `CLAUDE.md`.

## [10.31.1] - 2026-03-31

### Fixed

- **[#644] `store()` fails with UNIQUE constraint after `delete()` of same content (tombstone blocks re-insertion)**: Soft-deleted memories leave a tombstone row (with `deleted_at` set) that caused `INSERT OR IGNORE` to silently drop re-insertions of the same content hash, and `INSERT` to raise a `UNIQUE constraint failed` error. Fixed by adding `_purge_tombstone(content_hash)` helper to `SqliteVecMemoryStorage` that removes the tombstone row before any INSERT. Applied to `store()`, `store_batch()`, and `update_memory_versioned()`. Added test `test_store_after_delete_same_content` covering the full delete → re-store roundtrip.

## [10.31.0] - 2026-03-30

### Added

- **[#641] Harvest evolution (P4): evolve similar memories instead of duplicating**: Before storing a harvested memory, `memory_harvest` now checks semantic similarity against all active memories. If the best match exceeds the configurable threshold (default 0.85), the harvested content is routed through `update_memory_versioned()` to evolve the existing memory rather than creating a duplicate. This keeps the memory store clean and preserves lineage history automatically.
- **[#641] `HarvestConfig` evolution fields**: Two new configuration fields added to `HarvestConfig`: `similarity_threshold` (float, default 0.85) and `min_confidence_to_evolve` (float, default 0.3). Both are overridable via environment variables `MCP_HARVEST_SIMILARITY_THRESHOLD` and `MCP_HARVEST_MIN_CONFIDENCE_TO_EVOLVE`.
- **[#641] `harvest_config_from_env()` factory function**: New factory that reads `HarvestConfig` from environment variables, providing a clean interface for deployment-time configuration without code changes.
- **[#641] 9 new harvest evolution tests**: 3 config tests (env-var overrides, defaults, factory function) and 6 evolution scenario tests (novel memory stored as new, stale memory evolved, threshold boundary, superseded memory skipped, fallback on evolve error, no candidates).
- **[#637] `asyncio.to_thread()` wrapping in `_execute_with_retry`**: All SQLite-Vec DB operations routed through `_execute_with_retry` now run in a thread pool via `asyncio.to_thread()`, preventing event-loop blocking under concurrent async load. A TODO comment documents the ~122 remaining direct `self.conn.execute()` calls that are candidates for the same treatment in a follow-up.
- **[#637] 3 new async threading tests**: Verify that `_execute_with_retry` does not block the event loop, that concurrent calls complete without serialization deadlocks, and that the `check_same_thread=False` connection flag is set correctly.

### Fixed

- **[#641] Tag computation moved inside `else` branch**: Avoided redundant tag work on the evolve path — tags are now computed only when a new memory is actually being stored.
- **[#641] Strengthened test assertions**: Replaced permissive `if result.found > 0` guards with hard `assert result.found > 0` so evolution test failures surface immediately rather than passing silently.
- **[#637] Removed unused import** in `sqlite_vec.py` introduced during the async refactor.
- **[#637] `check_same_thread` docstring note** added to the connection initialisation comment, explaining why `False` is required for the thread-pool pattern.

## [10.30.0] - 2026-03-30

### Added

- **[#635, #636] Memory Evolution — P1: Non-destructive versioned updates**: New `update_memory_versioned()` operation creates a child node from an existing memory via a SAVEPOINT-atomic operation, marking the parent as `superseded_by` the new version. Enables full lineage tracking while preserving historical context. Schema migration `011_memory_evolution_p1.sql` adds `parent_id`, `version`, `confidence`, `last_accessed`, and `superseded_by` columns to the memories table. Active memories are filtered with `WHERE superseded_by IS NULL` in all search queries so superseded versions are transparent to normal retrieval.
- **[#635, #636] Memory Evolution — P2: Staleness scoring with time-decayed confidence**: `_effective_confidence()` computes time-decayed confidence as `confidence × max(0, 1 − staleness × decay_rate)`. New `retrieve_with_staleness()` uses an overfetch strategy (n_results × 3) combined with confidence filtering to return the most relevant non-stale memories. `min_confidence` parameter added to the base `retrieve()` interface (propagated to all backends: SQLite-Vec, Cloudflare, Hybrid). Decay window configurable via `MEMORY_DECAY_WINDOW_DAYS` env var (default: 30 days).
- **[#635, #636] Memory Evolution — P3: Automatic conflict detection and resolution**: Conflict detection runs automatically on `memory_store()` — memories with cosine similarity > 0.95 and Levenshtein divergence > 20% are flagged as contradictions and linked via `contradicts` graph edges. Two new MCP tools: `memory_conflicts` (list unresolved contradictions) and `memory_resolve` (supersede the loser, boost winner confidence to 1.0). New REST endpoints: `GET /api/conflicts` and `POST /api/conflicts/resolve`. No new dependencies — uses stdlib `difflib.SequenceMatcher` for divergence scoring.
- **30 new tests**: 21 tests covering P1/P2 (versioned updates, lineage tracking, staleness decay, min_confidence filtering) and 9 tests covering P3 (conflict detection on store, `get_conflicts()`, `resolve_conflict()`, MCP tool handlers, REST endpoints).

### Fixed

- **Storage interface consistency**: `min_confidence` parameter added to `base.py`, `cloudflare.py`, and `hybrid.py` `retrieve()` signatures, ensuring all backends accept the new filtering parameter without raising `TypeError`.

## [10.29.1] - 2026-03-29

### Fixed

- **[#632] Clean up orphaned graph edges on memory deletion**: When memories were deleted, their associated edges in the `memory_graph` table were not removed, causing dead references to accumulate over time and pollute graph queries. The fix adds explicit edge removal to `delete()`, `delete_by_tag()`, and `delete_by_tags()` in `sqlite_vec.py`. The consolidation forgetting phase now also performs a periodic orphan-pruning pass after archival, ensuring edges referencing non-existent memories are swept up even for memories removed by other means.

### Documentation

- **Troubleshooting additions**: Pre-commit hook PATH workaround (`.venv/bin`), editable install vs PyPI version switching, Cloudflare 401 memory-first diagnosis, dashboard testing guidelines, uv.lock revision downgrade
- **Instincts bootstrap**: Added `instincts/learned.instincts.yaml` with 4 session-derived instincts (PR workflow, memory-first debugging, env/dashboard token sync, release agent usage)

## [10.29.0] - 2026-03-29

### Added

- **[#628] LLM-based classification layer for session harvest (Phase 2)**: `memory_harvest` now accepts an optional `use_llm` boolean parameter. When `true`, extracted memories are routed through a new `_GroqClassifierBridge` (in `harvest/classifier.py`) that calls the Groq API to produce higher-precision category labels. Falls back transparently to the existing rule-based classifier when `use_llm=false` (the default) or when the Groq API is unavailable. Closes #618.
- **[#628] `harvest/classifier.py`**: New module implementing `_GroqClassifierBridge` — a lightweight, async-compatible adapter that wraps Groq's chat-completion API for memory classification. Includes rate-limit handling, structured output parsing, and a `classify_batch()` method for efficient bulk classification.
- **[#628] 14 new tests**: Full unit and integration coverage for `_GroqClassifierBridge` (mock API responses, fallback on 429/503, batch classification), the `use_llm` path in the harvest MCP handler, and end-to-end harvest flows with and without LLM classification enabled.

## [10.28.5] - 2026-03-29

### Fixed

- **[#621] Anonymous access flag ignored in dashboard**: `MCP_ALLOW_ANONYMOUS_ACCESS=true` had no effect on the dashboard — unauthenticated users were still redirected to a login prompt regardless of the flag value. The OAuth middleware now grants anonymous users full `read write` scope when `MCP_ALLOW_ANONYMOUS_ACCESS=true`, matching the server's intended behavior. Users behind a firewall or using external auth (e.g. Nginx Basic Auth) who rely on this flag no longer need to provide credentials in the dashboard.

### Documentation

- **Anonymous access scope clarification**: Updated `.env.example`, dashboard auth modal, and test docstrings to explicitly state that `MCP_ALLOW_ANONYMOUS_ACCESS=true` grants read+write access (not read-only). Addresses Gemini review feedback on PR #626.

## [10.28.4] - 2026-03-29

### Security

- **[#622] bump cryptography from 46.0.5 to 46.0.6**: Fixes CVE-2026-34073 (incomplete DNS name constraint enforcement). Dependabot alert #68 (low severity). Automated Dependabot bump.
- **[#623] bump serialize-javascript to >=7.0.5**: Fixes CVE-2026-34043 (CPU exhaustion DoS via crafted array-like objects in serialized output). Dependabot alerts #66 and #67 (medium severity). Applied to `tests/integration/` and `tests/web/` npm packages.

### Fixed

- **[#623] Remove unused `Optional` import in `harvester.py`**: CodeQL alert #379 (note). Import was a leftover from an earlier implementation; removal has no functional impact.

### Maintenance

- **[#623] Sort dependencies alphabetically in `pyproject.toml`**: Improves readability and prevents merge conflicts on future dependency updates.

## [10.28.3] - 2026-03-26

### Fixed

- **[#619] Accept 'content' as alias for 'query' in HTTP MCP endpoint**: Claude Code sends `{content: "search terms"}` via HTTP transport, but the handler only read `arguments.get("query")`, causing `retrieve_memory` and `recall_memory` to always return empty results when invoked over HTTP. The endpoint now accepts both `content` and `query` as parameter names, with `query` taking precedence.

## [10.28.2] - 2026-03-26

### Fixed

- **Tune relationship inference thresholds for real-world memory distribution**: Lowered `min_typed_confidence` from 0.75 to 0.50, `min_typed_similarity` from 0.65 to 0.45, and `min_confidence` from 0.6 to 0.5 so that relationships are inferred for a much broader range of real-world memories. Expanded `type_patterns` to cover `note`, `reference`, `document`, and `configuration` memory types (previously only `decision`, `learning`, `error`, and `pattern` were mapped, missing 85%+ of memories). Result: 93.5% typed relationship labels vs 0.5% before tuning.
- **Add German language patterns for relationship inference**: Added German patterns for causation, resolution, support, and contradiction relationships, along with German stopwords. Shared tags are now accepted as an alternative to keyword overlap for domain affinity, improving inference on short or terse memories.

## [10.28.1] - 2026-03-26

### Fixed

- **[harvest] Filter system prompts, skill outputs, and long injected content**: The JSONL parser now skips blocks tagged `system-reminder`, `command-name`, and `ide_opened_file`, and drops text blocks exceeding 2000 characters. This eliminates false-positive learnings extracted from injected system context rather than genuine session content. 3 new tests added.

## [10.28.0] - 2026-03-26

### Added

- **[#615] Session harvest — extract learnings from Claude Code transcripts (closes #596)**: New `memory_harvest` MCP tool that parses Claude Code JSONL transcript files and extracts structured learnings using pattern-based extraction with confidence scoring. Dry-run mode is enabled by default for safe preview before committing any memories. 27 new tests cover the JSONL parser, extractor patterns, and dry-run behaviour.

### Dependencies

- **[#614] bump requests from 2.32.5 to 2.33.0 (security fix CVE-2026-25645)**: Addresses a security vulnerability in the `requests` library; upgrade is recommended for all deployments that use outbound HTTP (e.g. Cloudflare sync, external embedding APIs).
- **[#616] bump pypdf from 6.9.1 to 6.9.2**: Routine patch update to `pypdf`; no functional changes affecting this project.

## [10.27.0] - 2026-03-25

### Fixed

- **[#612] Tolerate missing index in external embedding responses (community contribution by [@qq540491950](https://github.com/qq540491950))**: The external embedding client raised a `KeyError` when the upstream API returned responses without an `index` field (non-standard but valid for single-item batches). The fix falls back to enumerate-based ordering when `index` is absent, making the client compatible with a broader range of self-hosted embedding providers.

### Documentation

- **[e9b5db0] Add real-world self-hosted Docker + Cloudflare deployment example**: Added a complete end-to-end deployment walkthrough covering Docker Compose setup, Cloudflare D1 + Vectorize configuration, and hybrid storage mode for production self-hosted deployments.

## [10.26.9] - 2026-03-24

### Refactored

- **[#610] Fix N+1 query in update_memories_batch**: The batch update method issued a separate SELECT per memory to fetch `updated_at`; it now includes `updated_at` in the initial bulk SELECT, eliminating the N+1 query pattern and reducing database round-trips proportionally to batch size.
- **[#610] Simplify _get_memory_age_days with max(filter(None, ...))**: Replaced explicit conditional logic with a concise `max(filter(None, ...))` expression, improving readability while preserving correct behaviour when either timestamp is absent.
- **[#610] Extract _initialize_hash_embedding_fallback() helper**: Duplicated hash-embedding fallback initialization logic has been consolidated into a single private helper method, eliminating code duplication and making the fallback path easier to maintain.

## [10.26.8] - 2026-03-24

### Fixed

- **[#603] Fix invalid memory_type "learning_note" in learning_session prompt**: The `create_learning_session` prompt handler was emitting `memory_type: "learning_note"`, which is not a valid type in the memory schema. Changed to `"learning"` so sessions are correctly classified and retrievable by type filter.
- **[#604] Remove memory.touch() call from update_memory_relevance_metadata**: `update_memory_relevance_metadata` was calling `memory.touch()` as a side-effect, which silently overwrote `updated_at` on every relevance update. Relevance metadata updates (access count, quality score) no longer corrupt the `updated_at` timestamp.
- **[#605] Add preserve_timestamps option to update_memories_batch; consolidation callers opt in**: `update_memories_batch` now accepts a `preserve_timestamps` flag (default `False` for backward compatibility). The consolidation pipeline passes `preserve_timestamps=True` so that merging and compressing memories does not reset their original creation/update times.
- **[#606] Use max(created_at, updated_at) for memory age in _get_memory_age_days**: The age calculation previously used only `created_at`, so a memory that was meaningfully updated still decayed as if it had never been touched. The fix uses `max(created_at, updated_at)` so that a substantive update resets the effective age and prevents premature forgetting.
- **[#607] Add dimension fallback when _DIMENSION_CACHE misses on _MODEL_CACHE hit**: If the embedding model was already cached but the dimension entry was absent (e.g. after a cache-partial warm-up), the encoder would raise a KeyError. The fix re-derives and caches the dimension from the loaded model so subsequent calls succeed without re-loading the model.
- **[#608] Detect existing DB schema dimension for _HashEmbeddingModel fallback**: When the real embedding model cannot be loaded and the `_HashEmbeddingModel` fallback is used, the code now inspects the existing database schema to determine the vector dimension already in use rather than defaulting to a hard-coded value. This prevents dimension mismatch errors when re-opening an existing database with the fallback encoder.

## [10.26.7] - 2026-03-23

### Fixed

- **[#601] Cloudflare D1 schema initialization fails on fresh database (issue #600)**: On a brand-new Cloudflare D1 database, `PRAGMA table_list` returns a success response (`success: true`) with an empty `results` array rather than an error. The schema migration logic incorrectly treated this as a failure and aborted initialization, leaving the database in an unusable state. The fix explicitly checks for an empty-results success response and proceeds with full schema creation. Contributed by [@Lyt060814](https://github.com/Lyt060814).

## [10.26.6] - 2026-03-20

### Security

- **[#597] bump authlib>=1.6.9 — JWS JWK header injection, JWE Bleichenbacher padding oracle, fail-open OIDC hash binding (Critical + 2 High)**: `authlib` minimum version raised from `>=1.6.5` to `>=1.6.9`. Three vulnerabilities addressed: (1) JWS JWK header injection allowed an attacker to inject their own public key into the header and bypass signature verification (Critical); (2) JWE RSA1_5 algorithm was susceptible to a Bleichenbacher padding oracle attack allowing ciphertext decryption (High); (3) OIDC hash binding (`c_hash`/`at_hash`) validation was fail-open — invalid hash values were silently accepted rather than rejected (High). These affect deployments using the OAuth 2.1 endpoints.
- **[#597] bump PyJWT[crypto]>=2.12.0 — unknown `crit` header extension acceptance (High)**: `PyJWT` minimum version raised from `>=2.8.0` to `>=2.12.0`. Prior versions accepted JWTs with unknown `crit` header extensions instead of rejecting them, which could allow crafted tokens to bypass validation checks relying on extension semantics.
- **[#597] bump pypdf>=6.9.1 — inefficient array-stream decoding (DoS) (Medium)**: `pypdf` minimum version raised from `>=3.0.0` to `>=6.9.1`. Processing attacker-controlled PDF files with large array-based content streams could cause excessive CPU/memory usage. Fix limits stream length and improves decoding performance.
- **[#598] uv.lock updated**: `pypdf` 6.8.0 -> 6.9.1, `authlib` 1.6.8 -> 1.6.9 (Dependabot lock-file sync).

## [10.26.5] - 2026-03-13

### Security

- **bump black dev dependency to >=26.3.1 (GHSA-3936-cmfr-pm3m, CVE-2026-32274, High)**: The `black` code formatter contained a path traversal vulnerability via the `--python-cell-magics` option that could allow an attacker to write files outside the intended directory. The minimum required version has been updated from `>=24.0.0` to `>=26.3.1`. This vulnerability affects development and CI environments only — `black` is not a runtime dependency and is never included in installed packages. `uv.lock` updated from black 26.1.0 to 26.3.1.

## [10.26.4] - 2026-03-12

### Fixed

- **[#589] FTS5 table not created for existing databases (hybrid search broken on upgrade)** (`sqlite_vec.py`, contributed by @xXGeminiXx): Databases created before v10.8.0 never had the `memory_content_fts` FTS5 virtual table initialised because `initialize()` returned early after running graph migrations for existing DBs, bypassing the FTS5 creation block entirely. This caused hybrid BM25+vector search to silently fall back to vector-only on any pre-v10.8.0 upgrade. Added `_ensure_fts5_initialized()` idempotent method that checks `sqlite_master` for the table's existence before creating it and running the `rebuild` backfill command. The method is now called on both the new-DB and existing-DB paths, eliminating 57 lines of duplicated inline DDL.
- **[#592] Dashboard auth detection and credential persistence** (`web/static/app.js`, contributed by @jeremykoerber, fixes #591): Fixed 9 bugs in the dashboard authentication lifecycle: API key no longer lost on page refresh; `detectAuthRequirement()` and `authenticateWithApiKey()` now probe `/health/detailed` instead of `/health` (which is always public since v10.21.0); `setupServerManagement()` and `setupSSE()` deferred until after auth resolves (race condition); `handleAuthFailure()` respects `initComplete` guard — credentials no longer wiped during startup 401s; SSE reconnect closes existing connection before opening a new one (leak fix); `startSyncStatusMonitoring()` now called in modal auth path; sync monitor interval ID stored and cleaned up in `destroy()`; auth failure toasts debounced to 30 s.

## [10.26.3] - 2026-03-10

### Fixed

- **Dashboard: metadata object values now rendered as JSON** (`app.js`, #582): Metadata values that are objects (or arrays of objects) were previously rendered as `[object Object]`. They are now serialised with `JSON.stringify` and HTML-escaped via `escapeHtml`, fixing both the display issue and a potential XSS vector introduced by the naive object-to-string coercion.
- **Dashboard: long memory content collapsed by default in detail modal; quality tab fetches full object** (`app.js`, `style.css`, #583): Memory content longer than 500 characters is now collapsed with a "Show more / Show less" toggle in the detail modal. When opening a memory from the quality tab, the full memory object is fetched via `GET /api/memories/{hash}` before the modal opens, preventing incomplete data display.
- **Quality scorer: empty query during `store_memory` no longer yields 0.0 score** (`ai_evaluator.py`, #584): The Groq scorer previously used a relevance-based prompt even when the query was empty (as is the case for `store_memory` calls). An empty query produced a semantically meaningless prompt and a near-zero score. The scorer now detects an empty query and switches to an absolute quality prompt that evaluates content quality independently of any query.
- **Quality scorer: Groq 429 rate limit triggers model fallback chain** (`ai_evaluator.py`, #585): The Groq quality scorer now attempts a sequence of models (`llama-3.1-8b-instant`, `llama3-8b-8192`, `gemma2-9b-it`) in order when a `429 Too Many Requests` response is received, instead of failing hard. This prevents quality scoring from failing silently under rate pressure and improves resilience for high-throughput deployments.

## [10.26.2] - 2026-03-08

### Fixed

- **[#576] OAuth token exchange fails with 500 for public PKCE clients** (`authorization.py`): claude.ai and other MCP clients that use OAuth 2.1 public-client flow (PKCE without `client_secret`) received a `500 Internal Server Error` during token exchange because the endpoint unconditionally called `authenticate_client()`, which requires a secret. The endpoint now detects public PKCE clients — requests that supply a `code_verifier` but no `client_secret` — and skips secret authentication, using the PKCE verifier as the sole identity proof instead, in accordance with OAuth 2.1 §2.1. Confidential clients (with `client_secret`) are unaffected. Closes #576.
- **Missing `/.well-known/oauth-protected-resource` endpoint** (`discovery.py`): The endpoint required by RFC 9728 and the MCP OAuth spec was returning 404, breaking OAuth discovery for compliant clients. Added `OAuthProtectedResourceMetadata` Pydantic model and the corresponding route, which advertises the resource identifier and authorization server URLs with `token_endpoint_auth_methods_supported: ["none"]`.
- **Opaque OAuth error logging**: Added `exc_info=True` to exception handlers in the token and authorization endpoints so that full tracebacks are recorded in logs instead of generic error messages, making future debugging significantly easier.

### Added

- **Automated CHANGELOG housekeeping workflow** (`.github/workflows/changelog-housekeeping.yml`): Monthly GitHub Actions workflow (runs on the 1st of each month, also triggerable via `workflow_dispatch`) that automatically archives CHANGELOG entries older than the 8 most recent versions into `docs/archive/CHANGELOG-HISTORIC.md`. Keeps the main CHANGELOG lean for faster reads while preserving full history. Validates that no version entries are lost during archival.
- **Changelog housekeeping script** (`scripts/maintenance/changelog_housekeeping.py`): Python script backing the workflow. Keeps the 8 most recent versions in `CHANGELOG.md`, moves older entries to the historic archive, and trims the README "Previous Releases" section to a maximum of 7 entries. Supports `--dry-run` for safe preview before applying changes.

## [10.26.1] - 2026-03-08

### Fixed

- **[#570] Hybrid backend misidentified as sqlite-vec in MCP health checks** (`memory_health` tool): `HealthCheckFactory` relied solely on the storage object's class name to select the health-check strategy. When the hybrid backend's storage is accessed through a delegation or wrapper layer the class name is not `HybridMemoryStorage`, so the factory fell back to the sqlite-vec strategy and reported `"sqlite-vec"` instead of `"hybrid"`, hiding Cloudflare sync status from users. The factory now performs structural detection — if the storage object exposes both a `primary` attribute and either a `secondary` or `sync_service` attribute it is classified as hybrid regardless of class name. The existing SQLite and Cloudflare strategy paths are unchanged. Adds three focused unit tests for strategy selection (sqlite class-name path, wrapped/delegated hybrid structural path, unknown fallback). Fixes #570.

## [10.26.0] - 2026-03-07

### Added

- **Credentials tab in Settings modal** (`GET /api/config/credentials`, `POST /api/config/credentials/test`, `POST /api/config/credentials`): Manage Cloudflare API token, Account ID, D1 Database ID, and Vectorize Index directly from the dashboard. Credentials shown with partial-reveal (masked) display and an eye-toggle for full reveal.
- **Connection test gate (test-gate pattern)**: Credentials must pass a live connection test before they can be saved, preventing accidental misconfiguration.
- **Sync Owner selector** (`MCP_HYBRID_SYNC_OWNER`: `http` / `both` / `mcp`): New setting to control which server handles Cloudflare sync in hybrid mode. Default is `http` (recommended) — HTTP server owns all sync; MCP server (Claude Desktop) uses SQLite-Vec only, removing the need for a Cloudflare token in the MCP config.
- **Settings tabs restructured**: The Backup tab is split into three focused tabs — Quality, Backup, and Server — bringing the total to 7 tabs for clearer organisation.

### Security

- **SSRF protection on credentials endpoint**: `account_id` validated against `[a-f0-9]{32}` regex before use in Cloudflare API calls, blocking Server-Side Request Forgery via crafted account IDs.
- **Newline injection prevention**: Credential values sanitised to reject embedded newlines, preventing HTTP header injection.
- **`sync_owner` allowlist**: Only `http`, `both`, and `mcp` are accepted; unknown values are rejected with a 422 error.

### Documentation

- `CLAUDE.md` and `README.md` updated: `MCP_HYBRID_SYNC_OWNER=http` documented as the recommended configuration for hybrid mode, along with the rationale (HTTP server as sole sync owner, MCP server token-free).

### CI

- Claude Code Review GitHub Actions workflow disabled due to OAuth token expiry issues (will be migrated to API key auth).

### Tests

- No new tests added in this release (1,420 total).

## [10.25.3] - 2026-03-07

### Fixed

- **[#561] Strict stdio MCP clients (e.g. Codex CLI) timing out during startup handshake**: Non-LM-Studio stdio clients performing eager initialization could exceed the client's fixed handshake budget (Codex uses ~10 s). The eager-init timeout is now capped at 5.0 s for these clients, ensuring the MCP handshake completes within budget. Co-authored-by SergioChan.
- **Syntax errors in eager-init timeout cap (follow-up to PR #569)**: Resolved a duplicate function call, an orphaned closing parenthesis, and a duplicate `return` statement introduced in the initial fix. Named constants replace magic numbers, a dead-code guard was corrected, and warning messages were clarified.
- **Hybrid sync premature termination**: The cloud-to-local sync aborted early when `synced_count` was 0 at the `HYBRID_MIN_CHECK_COUNT` (1,000) threshold, even though thousands of memories remained unchecked. The early-exit condition now only triggers after all `secondary_count` memories have been inspected, ensuring a complete Cloudflare-to-local sync.
- **Dashboard version badge always blank**: `loadVersion()` called `/health` which returns only `{"status":"healthy"}` since the v10.21.0 GHSA-73hc security hardening. Changed to `/health/detailed` which includes the `version` field.

### Chores

- `.gitignore` updated to exclude TLS certificate files (`*.pem`, `certs/` directory) to prevent accidental credential commits.

## [10.25.2] - 2026-03-07

### Fixed

- **Health check in `update_and_restart.sh` always reported "unknown" version**: The `/api/health` endpoint was stripped of its `version` field in v10.21.0 (security hardening GHSA-73hc-m4hx-79pj). The update script still tried to read `data.get('version')`, causing it to always fall back to "unknown" and wait the full 15-second timeout before giving up. The check now reads the `status` field (`"healthy"`) to confirm the server is up, and reports the already-known pip-installed version instead.

## [10.25.1] - 2026-03-06

### Security

- **[GHSA-g9rg-8vq5-mpwm] Wildcard CORS default enables cross-origin memory theft**: `MCP_CORS_ORIGINS` defaulted to `'*'`, allowing any website to read API responses cross-origin when combined with anonymous access. Default changed to `'http://localhost:8000,http://127.0.0.1:8000'`. `allow_credentials` is now automatically set to `False` when wildcard origins are configured (the two cannot be combined securely). A startup warning is logged if wildcard is explicitly set via environment variable.
- **[GHSA-x9r8-q2qj-cgvw] TLS certificate verification disabled in peer discovery**: Already fixed in v10.25.0 (config already contained `PEER_VERIFY_SSL=True` default). Advisory formally closed.

### Fixed

- **Soft-delete leak in `search_by_tag_chronological()`**: Missing `AND deleted_at IS NULL` filter caused soft-deleted memories to appear in chronological tag search results. All soft-delete filters are now consistently applied across all query paths.

## [10.25.0] - 2026-03-06

### Added

- **Embedding model migration script** (`scripts/maintenance/migrate_embeddings.py`): Standalone script for migrating embeddings between models, including across different dimensions (e.g., 384-dim `all-MiniLM-L6-v2` to 768-dim `nomic-embed-text`). Works with any OpenAI-compatible API (Ollama, vLLM, OpenAI, TEI). Features: `--dry-run`, auto-detect dimension, timestamped backup, service detection (macOS launchd, Linux systemd), cross-platform DB path detection, `--keep-graph` option, batched embedding with progress, post-migration integrity verification. Closes #552.

### Fixed

- **[#557] Soft-delete leaks in recall, time-range, and statistics methods**: `recall()` (both semantic and time-based paths), `get_memories_by_time_range()`, `get_largest_memories()`, and `get_memory_timestamps()` (both branches) were missing `deleted_at IS NULL` filters, causing soft-deleted memories to appear in results.
- **[#557] Incorrect cosine distance score formula in recall()**: Used `1.0 - distance` but cosine distance ranges [0, 2], producing negative scores for dissimilar pairs. Corrected to `max(0.0, 1.0 - (distance / 2.0))` to map to [0, 1].
- **[#557] Tag parsing in get_largest_memories()**: Used `json.loads()` to parse tags, but tags are stored as comma-separated strings. Changed to `split(",")` to match all other methods.
- **[#558] Substring tag matching bug**: `get_all_memories(tags=["test"])`, `count_all_memories(tags=["test"])`, and `retrieve()` used `LIKE '%test%'`, incorrectly matching `"testing"`, `"my-test-tag"`, etc. All now use the canonical GLOB exact-match pattern: `(',' || REPLACE(tags, ' ', '') || ',') GLOB '*,tag,*'`. Added `_escape_glob()` helper to prevent GLOB wildcard injection (`*`, `?`, `[`) from user-supplied tag values. `search_by_tag_chronological()` LIMIT/OFFSET now parameterized instead of f-string interpolated.
- **[#559] O(n²) memory in association sampling**: `_sample_memory_pairs()` materialized all `combinations(memories, 2)` into a list (50M pairs for 10k memories) just to pick 100. Now uses random index pair generation, O(max_pairs) in the sparse case.
- **[#559] Broken duplicate detection in consolidation**: `_get_existing_associations()` loaded all memories and filtered by `memory_type=="association"`, but new associations are stored with `memory_type="observation"` and tag `"association"`. The filter never matched, so duplicate associations were never prevented. Now uses `search_by_tag(["association"])`.
- **[#560] Soft-delete gaps in write and statistics methods**: `get_memory_connections()`, `get_access_patterns()`, `update_memory_metadata()`, and `update_memories_batch()` could return or modify tombstoned memories. All now include `deleted_at IS NULL` filters. `delete()` error handler now explicitly rolls back the transaction with `sqlite3.OperationalError` narrowing to prevent dangling embedding DELETEs.

### Performance

- **Batch access metadata persistence** (`retrieve()`): Access metadata now persisted in one `executemany` call per query instead of N individual `UPDATE+COMMIT` round-trips (new `_persist_access_metadata_batch()` method).
- **Hybrid search O(n+m) deduplication** (`retrieve_hybrid()`): Replaced O(n×m) nested-loop deduplication with O(n+m) dict-based merging. BM25-only memories now batch-fetched in one SQL query (capped at 999 to respect `SQLITE_MAX_VARIABLE_NUMBER`) instead of N+1 individual `get_by_hash()` calls.

### Tests

- 23 new regression tests covering all fixed methods
- Total: 1,420 tests
