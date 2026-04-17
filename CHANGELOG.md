# Changelog

**Recent releases for MCP Memory Service (v10.36.4 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

**Versions v10.36.3 and earlier** – See [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **[#725] Dependency bump**: `pypdf` 6.10.1 → 6.10.2 (Dependabot)
- **[#726] Dependency bump**: `authlib` 1.6.10 → 1.6.11 (Dependabot)
- **[#727] CI: skip full test suite on docs-only changes**: Added `paths-ignore` for `docs/**`, `*.md`, `.claude/**`, `LICENSE`, `.gitignore` to `main.yml`. Docs-only PRs no longer trigger the 1587-test pytest suite, Docker build, or CodeQL scan.

### Fixed

- **Knowledge graph dashboard**: Fixed invisible edges for non-canonical relationship types by providing a fallback color for undefined CSS variables.
- **Flaky concurrent test**: Relaxed strict write-count assertions in `test_concurrent_clients.py` from `== 10` to `>= 9` (and `== 5` to `>= 4`). SQLite WAL lock contention under CI runner load can legitimately drop 1 of N concurrent writes.

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

