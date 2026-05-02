# Housekeeping Audit — 2026-05-02

**Scope**: docs/, README.md, CLAUDE.md, docs/index.html, mcp-memory-service.wiki/
**Reference state**: v10.47.2, ~1,780 tests (CLAUDE.md authoritative — counts parameterized + integration variants; raw `def test_` count is 787)
**Categories scanned**: version, feature, wiki↔repo, cross-doc, links, structural

## Summary

- 🔴 Critical: 4 findings (F-007, F-008, F-015, F-016)
- 🟡 Major:    18 findings (F-020 verified false → reclassified Noise)
- 🟢 Minor:    13 findings (F-031 groups 46 repo orphans; F-032 groups 21 wiki orphans)
- ⚪ Noise:    excluded (historical "introduced in vX" annotations, per-release test counts in old changelog entries, PR template examples, HTTPS port 8443 config examples in wiki)

**Total findings**: 36 (after grouping 67 individual orphan findings into 2 cluster findings)

**How to triage**: For each finding below, mark exactly one disposition checkbox: `ship-now` / `defer-issue` / `skip`. When done, tell the controller to proceed.

---

## Findings

### F-001 🟡 Major · version drift · docs/index.html:245
**Current**: `<a href="https://github.com/doobidoo/mcp-memory-service/releases/tag/v10.47.0">`
**Expected**: `v10.47.2`
**Fix**: Change release URL to `.../releases/tag/v10.47.2` so "Release Notes" button on landing page links to latest patch release.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-002 🟢 Minor · version drift · docs/index.html:6,8
**Current**: `v10.47` (title tag and og:title — no patch version)
**Expected**: `v10.47.2`
**Fix**: Update `<title>` and `<meta property="og:title">` to `v10.47.2` for consistency. Per CLAUDE.md "MINOR/MAJOR releases only" policy, the `v10.47` form is technically allowed; this is cosmetic.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-003 🔴 Critical · cross-doc · test count drift (multiple files)
**Current** (3 contradictory sources):
  - `CLAUDE.md:73` → `# Run all tests (968 tests total)`
  - `CLAUDE.md:233` → `### Structure (968 tests)`
  - `CLAUDE.md:48` → `~1,780 tests` (same file contradicts itself)
  - `README.md:493–499` → per-release counts show `1,537–1,547` tests
  - `docs/index.html:140` → `~1,780 tests`
  - `wiki/13-Development-Roadmap.md:11` → `~1,780 tests passing`
**Expected**: `~1,780 tests` (current canonical per CLAUDE.md header)
**Fix**: Update CLAUDE.md:73 and CLAUDE.md:233 from `968 tests` → `~1,780 tests`. Per-release historical counts in README are correct as historical data and stay.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-004 🟡 Major · version drift · wiki/13-Development-Roadmap.md:9
**Current**: `**Current Version**: v10.47.0 (Production-ready)`
**Expected**: `v10.47.2`
**Fix**: Bump to `v10.47.2` so the wiki roadmap header matches latest release.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-005 🟢 Minor · version drift · wiki/13-Development-Roadmap.md:289
**Current**: `- **Version at Review**: v10.47.0`
**Expected**: `v10.47.2`
**Fix**: Update Q1 2026 review version marker to `v10.47.2`.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #826

### F-006 🟢 Minor · version drift · docs/BENCHMARKS.md:13
**Current**: `**Date:** 2026-04-08 · **Version:** v10.34.0`
**Expected**: Note reflects benchmark run version, not current.
**Fix**: Add parenthetical `(benchmark run version; latest release: v10.47.2)` or link to updated benchmark. Existing value is historically correct for that run but may mislead.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #826

---

### F-007 🔴 Critical · feature drift · docs/mastery/api-reference.md:9–37
**Current**: Entire "MCP (FastMCP HTTP) Tools" section documents pre-v10 deprecated tool names as canonical: `store_memory`, `retrieve_memory(query, n_results=5)`, `search_by_tag`, `delete_memory`, `check_database_health`.
**Replacement**: v10 unified names — `memory_store`, `memory_search`, `memory_list`, `memory_delete`, `memory_health`. Parameter `n_results` → `limit`.
**Fix**: Rewrite the "MCP Tools" table to list the 12 unified `memory_*` tools. Remove the `mcp_server.py` reference (still exists but is a compat shim, not primary).
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-008 🔴 Critical · feature drift · wiki/03-Integration-Guide.md:44–48
**Current**: "Verify Connection" step (the actual install/setup verification path) tells users to look for `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `check_database_health` in tool list.
**Replacement**: `memory_store`, `memory_search`, `memory_list`, `memory_delete`, `memory_health`.
**Fix**: Update bullet list to v10 names. Critical because this is what users follow to confirm setup worked.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-009 🟡 Major · feature drift · docs/mastery/api-reference.md:66–73 (Examples)
**Current**: Example snippets use `tool: store_memory`, `tool: retrieve_memory` with `n_results: 5`, `tool: search_by_tag`.
**Replacement**: `memory_store`, `memory_search` (with `limit` param), `memory_list` for tag search.
**Fix**: Update all three example blocks.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-010 🟡 Major · feature drift · wiki/14-Memory-Quality-System-Guide.md:235,273,298,318,708,792,809,812
**Current**: Multiple code examples use deprecated quality tool names — `rate_memory`, `retrieve_with_quality_boost`, `get_memory_quality`, `analyze_quality_distribution`.
**Replacement**: `memory_quality(action="rate"|"get"|"analyze")` and `memory_search(mode="hybrid", quality_boost=...)`.
**Fix**: Update all code blocks; add migration note at top.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-011 🟡 Major · feature drift · wiki/10-Complete-Feature-List.md:560–563
**Current**: Quality System "MCP Tools" subsection lists 4 deprecated names as the current inventory.
**Replacement**: `memory_quality` actions + `memory_search` with quality_boost.
**Fix**: Replace the 4 lines with correct unified signatures.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-012 🟡 Major · feature drift · docs/mastery/api-reference.md:37 + docs/mastery/architecture-overview.md:48,53
**Current**: Architecture/overview docs describe tool surface using deprecated names: `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `check_database_health`, `cleanup_duplicates`, `update_memory_metadata`.
**Replacement**: v10 unified names. Note `cleanup_duplicates` → `memory_cleanup`, `update_memory_metadata` → `memory_update`.
**Fix**: Rewrite prose descriptions.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-013 🟡 Major · feature drift · `n_results` parameter (4 locations)
**Current**: `n_results` parameter used in user-facing examples — deprecated since v10.0.0.
  - `wiki/03-Integration-Guide.md:475`: `'{"query": "integration test", "n_results": 5}'`
  - `wiki/03-Integration-Guide.md:586`: `body: JSON.stringify({ query, n_results: nResults })`
  - `docs/mastery/api-reference.md:13`: `retrieve_memory(query, n_results=5, min_similarity=0.0)`
  - `docs/mastery/api-reference.md:67`: `args: { "query": "OAuth refactor", "n_results": 5 }`
**Replacement**: `limit`.
**Fix**: Replace `n_results` → `limit` in all 4 locations.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-014 🟡 Major · feature drift · docs/guides/memory-consolidation-guide.md:96 + wiki/Memory-Consolidation-System-Guide.md:203
**Current**: Both guides show `trigger_consolidation` as the recommended MCP tool.
**Replacement**: `memory_consolidate(action="run", ...)`.
**Fix**: Update invocation in both files. Also update `consolidate_memories`, `scheduler_status`, `pause_consolidation` references to `memory_consolidate` action equivalents.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

---

### F-015 🔴 Critical · wiki↔repo · `chroma` backend referenced as valid (4 wiki locations)
**Current** (4 locations document `chroma` as a valid `MCP_MEMORY_STORAGE_BACKEND` value):
  - `wiki/03-Integration-Guide.md:68`
  - `wiki/03-Integration-Guide.md:627` (full "ChromaDB Backend for Team Sharing" code block)
  - `wiki/01-Installation-Guide.md:271`
  - `wiki/01-Installation-Guide.md:335`
**Expected** (per `src/mcp_memory_service/config.py:306`): `SUPPORTED_BACKENDS = ['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid', 'milvus']`. Setting `chroma` falls through to fallback warning + reverts to `sqlite_vec`.
**Fix**: Remove all `chroma` references from wiki; for "team sharing" use case, replace with `milvus` or note feature was retired.
**VERIFY BEFORE FIX**: Re-grep `src/mcp_memory_service/config.py` to confirm `chroma` truly absent from `SUPPORTED_BACKENDS`. If subagent's read was stale, downgrade to Major.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-016 🔴 Critical · wiki↔repo · OAuth env vars don't exist in code
**Current** (5 env vars documented in `wiki/04-Advanced-Configuration.md:466,469,472,546,547`): `MCP_OAUTH_CLIENT_STORAGE`, `MCP_OAUTH_REGISTRATION_RATE_LIMIT`, `MCP_OAUTH_AUDIT_LOG`, `MCP_OAUTH_CLEANUP_EXPIRED_TOKENS`, `MCP_OAUTH_TOKEN_CLEANUP_INTERVAL`.
**Expected**: None exist in `src/mcp_memory_service/config.py` per subagent grep — would be silently ignored if set.
**Fix**: Audit which OAuth env vars are actually implemented; remove or correct the rest. Note: `MCP_OAUTH_STORAGE_BACKEND` and `MCP_OAUTH_SQLITE_PATH` ARE real (per CLAUDE.md) — don't remove those.
**VERIFY BEFORE FIX**: Re-grep `src/` for each env var name. False positive risk if vars live in `web/oauth/` modules outside `config.py`.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-017 🟡 Major · wiki↔repo · wiki/01-Installation-Guide.md:9
**Current**: Banner says "Hybrid backend as default" (under v8.9.0 Highlights).
**Expected**: Default is `sqlite_vec` per `config.py:307`. Hybrid is recommended for production but requires explicit configuration.
**Fix**: Update banner to "SQLite-Vec is the default; Hybrid is recommended for production with Cloudflare credentials".
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-018 🟡 Major · wiki↔repo · MCP tool count "12" vs actual 18 in code
**Current**: Wiki and `server_impl.py` docstring both claim "12 total" tools.
**Expected** (per `grep -c 'types\.Tool(' src/mcp_memory_service/server_impl.py`): 18 distinct registrations — `memory_store`, `memory_store_session`, `memory_search`, `memory_list`, `memory_delete`, `memory_cleanup`, `memory_health`, `memory_stats`, `memory_update`, `memory_consolidate`, `memory_ingest`, `memory_harvest`, `memory_quality`, `memory_graph`, `memory_conflicts`, `memory_resolve`, `mistake_note_add`, `mistake_note_search`.
**Fix**: Update `handle_list_tools` docstring + wiki references to "18 total" (or reconcile if some are deprecated/conditional).
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-019 🟡 Major · wiki↔repo · wiki/04-Advanced-Configuration.md missing `MCP_SCHEDULE_*` env vars
**Current**: Wiki Advanced Configuration doesn't document `MCP_SCHEDULE_DAILY/WEEKLY/MONTHLY/QUARTERLY/YEARLY` env vars at all. Existing consolidation section (line 104) only covers manual triggering.
**Expected**: All 5 schedule env vars exist per `config.py`, default `'disabled'` (per v10.47.2 PR #821). These are the primary opt-in mechanism for automatic consolidation.
**Fix**: Add a schedule configuration table documenting all 5 vars with format examples (e.g. `MCP_SCHEDULE_DAILY=03:00`).
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-020 ⚪ Noise (FALSE FINDING — verified) · wiki↔repo · `MCP_MEMORY_HTTP_ENDPOINT` undocumented
**Status**: VERIFIED FALSE — var IS implemented in `examples/http-mcp-bridge.js:157` (`process.env.MCP_MEMORY_HTTP_ENDPOINT`) and documented in `examples/claude-desktop-http-config.json:7`. Wiki references in `03-Integration-Guide.md` correctly document the client-side bridge env var. No fix needed.
**Disposition**: [ ] ship-now  [ ] defer-issue  [x] skip

### F-021 🟡 Major · cross-doc · default storage backend (FAQ vs Installation Guide vs CLAUDE.md)
**Locations**:
  - `wiki/01-Installation-Guide.md:9` → "Hybrid backend as default" (overlaps F-017)
  - `wiki/08-FAQ.md:29` → "stored locally by default (SQLite-vec backend)"
  - `docs/glama-deployment.md:45` → table shows default `sqlite_vec`
  - `CLAUDE.md` Claude Desktop config example → recommends `"hybrid"`
**Expected** (per code): Effective runtime default is `sqlite_vec`. `hybrid` requires Cloudflare credentials and is recommended-for-production but not out-of-box.
**Fix**: Align all three wiki/docs to: "Default is SQLite-Vec. Hybrid is recommended for production deployment when Cloudflare credentials are configured." (Fix bundled with F-017.)
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

---

### F-022 🟡 Major · link rot · README.md:185,205 (MemPalace org rename)
**Broken URLs** (3 occurrences in README.md):
  - `https://github.com/milla-jovovich/mempalace` (301 → moved)
  - `https://github.com/milla-jovovich/mempalace/issues/27` (404 — issue links don't redirect)
  - line 205: another `issues/27` reference
**Context**: Repo transferred from user `milla-jovovich` to org `MemPalace`. Issue is live at `https://github.com/MemPalace/mempalace/issues/27`.
**Fix**: Replace all 3 `milla-jovovich/mempalace` → `MemPalace/mempalace` in README.md.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-023 🟡 Major · link rot · wiki/01-Installation-Guide.md:118,445 → `./Claude-Code-Memory-Awareness-Guide`
**Broken URL**: `./Claude-Code-Memory-Awareness-Guide` (page does not exist)
**Context**: Referenced as the primary "complete setup" guide and in "Next Steps".
**Fix**: Create the wiki page OR redirect to `./Claude-Code-Commands-Wiki` (exists) or `./06-Development-Reference`.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-024 🟡 Major · link rot · wiki/01-Installation-Guide.md:446 + 03-Integration-Guide.md:809 → `./Claude-Code-Quick-Reference`
**Broken URL**: `./Claude-Code-Quick-Reference` (page does not exist)
**Fix**: Point to `./Claude-Code-Commands-Wiki` (exists, covers same ground) OR create the page.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-025 🟢 Minor · link rot · wiki/01-Installation-Guide.md:433 → `./macOS-Intel-Legacy-Guide`
**Broken URL**: `./macOS-Intel-Legacy-Guide` (page does not exist).
**Fix**: Remove link (keep text plain) or fold into `./07-TROUBLESHOOTING`.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #825

### F-026 🟢 Minor · link rot · wiki/01-Installation-Guide.md:447 → `./Tag-System-Migration-and-Management`
**Broken URL**: `./Tag-System-Migration-and-Management` (page does not exist).
**Fix**: Remove link or redirect to `./04-Advanced-Configuration`.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #825

### F-027 🟡 Major · link rot · 3× `./TROUBLESHOOTING` should be `./07-TROUBLESHOOTING`
**Locations**:
  - `wiki/01-Installation-Guide.md:448`
  - `wiki/01-Installation-Guide.md:458`
  - `wiki/03-Integration-Guide.md:809`
**Fix**: Rename all 3 occurrences `./TROUBLESHOOTING` → `./07-TROUBLESHOOTING`.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-028 🟢 Minor · anchor rot · wiki/06-Development-Reference.md:10 → `#debugging--testing`
**Broken anchor**: TOC links `#debugging--testing` but heading at line 216 generates `#debugging--troubleshooting` (renamed).
**Fix**: Change TOC link to `#debugging--troubleshooting`.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #824

### F-029 🟢 Minor · anchor rot · wiki/05-Performance-Optimization.md:8 → emoji variation selector
**Broken anchor**: TOC links `#️-critical-cloudflare-service-limitations`. GitHub Wiki strips U+FE0F variation selector → actual anchor is `#-critical-cloudflare-service-limitations`.
**Fix**: Update TOC anchor.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #824

---

### F-030 🟡 Major · structural · Wiki Home.md navigation broken — entire numbered series invisible
**Issue**: Wiki Home.md has only 1 outbound link (to CONTRIBUTING.md on GitHub). The numbered wiki pages (01-Installation-Guide through 19-Graph-Database-Architecture) are referenced only from `Backend-Synchronization-Guide.md`'s docs table — and that file is itself orphaned. **Result**: the entire numbered wiki series is effectively invisible from Home.md navigation.
**Fix**: Rewrite `wiki/Home.md` to include direct links to all 19 numbered guides + key non-numbered pages (OAuth setup, Web Dashboard, Claude AI Remote MCP, etc.). This is the highest-leverage wiki fix — single page change, unblocks navigation for all numbered guides.
**Disposition**: [x] ship-now  [ ] defer-issue  [ ] skip

### F-031 🟢 Minor · structural · 46 orphaned docs in repo
**Issue**: 46 `.md` files under `docs/` have no incoming links from any active doc/wiki. Examples (full list — see audit transcript):
- `docs/ROADMAP.md` (superseded by wiki Development-Roadmap)
- `docs/HOOK_IMPROVEMENTS.md`, `docs/IMAGE_RETENTION_POLICY.md`, `docs/HARK_*` planning docs
- `docs/development/code-quality/phase-2*.md` (5 phase artifacts)
- `docs/releases/release-notes-v8.61.0.md`, `docs/releases/v8.72.0-testing.md`
- `docs/blog/2026-03-30-memory-evolution-v10-31-0.md`
- `docs/integrations/gemini.md` (orphaned despite being a real integration guide)
- `docs/natural-memory-triggers/installation-guide.md`, `cli-reference.md`
- `docs/architecture/search-enhancement-spec.md`, `docs/architecture/search-examples.md`
- `docs/deployment/dual-service.md`, `docs/deployment/production-guide.md`
- `docs/maintenance/changelog-housekeeping-prompt.md`
- `docs/wiki-Graph-Database-Architecture.md`, `docs/wiki-documentation-plan.md`
- `docs/api/PHASE2_REPORT.md`, `docs/DOCUMENTATION_AUDIT.md`
- `docs/development/COMMIT_MESSAGE.md`, `docs/development/refactoring-notes.md`, `docs/development/pr-280-post-mortem.md`
- `docs/research/locomo-benchmark-analysis.md`, `docs/statistics/REPOSITORY_STATISTICS.md`
- `docs/migrations/010-asymmetric-relationships.md`
- `docs/quality-system-ui-implementation.md`, `docs/demo-recording-script.md`, `docs/hybrid-graph-sync-plan.md`, `docs/CLAUDE_CODE_QUICK_REFERENCE.md`
- `docs/guides/advanced-command-examples.md`, `docs/guides/mdns-service-discovery.md`, `docs/guides/commands-vs-mcp-server.md`
- `docs/integrations.md`, `docs/mastery/local-setup-and-run.md`
- `docs/testing/integrity-monitoring-test-report.md`, `docs/verification/v9.0.0-knowledge-graph-verification.md`
- `docs/reviews/bug-fixes-issues-441-447.md`
- `docs/testing-cloudflare-backend.md`, `docs/technical/migration-log.md`, `docs/technical/sqlite-vec-embedding-fixes.md`
- `docs/images/dashboard-placeholder.md`
- `docs/remote-configuration-wiki-section.md`
**Fix**: Triage each — archive (move to `docs/archive/`), delete (placeholder/artifact files), or add inbound link if still relevant. Recommend filing as single umbrella issue for follow-up.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

### F-032 🟢 Minor · structural · 21 orphaned wiki pages
**Issue**: 21 wiki pages have no incoming links from Home.md or any other wiki page. Examples:
- `wiki/08-FAQ.md`, `wiki/13-Development-Roadmap.md`, `wiki/14-Contributor-Promotion.md`
- `wiki/15-ONNX-Quality-Evaluation.md`, `wiki/16-Metadata-Compression-System.md`, `wiki/17-Quality-Hooks-Integration.md`
- `wiki/18-Dashboard-UI-Guide.md`, `wiki/19-Graph-Database-Architecture.md`
- `wiki/AppleScript-Integration-with-Memory.md`, `wiki/Claude-AI-Remote-MCP-Integration.md`
- `wiki/Claude-Code-Commands-Wiki.md`
- `wiki/Cloudflare-Based-Multi-Machine-Sync.md`, `wiki/Hybrid_Setup_Configuration.md`
- `wiki/Memory-Consolidation-System-Guide.md`, `wiki/Memory-Quality-System-Evaluation.md`
- `wiki/Pull-Request-Templates.md`, `wiki/Issue-Templates.md`, `wiki/Screenshot-Automation-Guide.md`
- `wiki/Windows-Hybrid-Backend-Setup-Example.md`, `wiki/Windows-Setup-Summary-Example.md`
- `wiki/Development-Sprint-‐-November-2025.md`
**Fix**: Largely solved by F-030 (Home.md rewrite). Remaining: archive sprint planning artifact, dedupe the dashboard guides.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

### F-033 🟢 Minor · structural · Cloudflare guide cluster (3-way duplicate)
**Issue**: Three wiki guides cover the same Cloudflare hybrid setup territory:
- `wiki/Cloudflare-Backup-Sync-Setup.md` (step-by-step)
- `wiki/Cloudflare-Based-Multi-Machine-Sync.md` (problem/solution narrative)
- `wiki/Hybrid_Setup_Configuration.md` (v8.9.0 reference config dump)
**Fix**: Consolidate into single canonical guide (Cloudflare-Backup-Sync-Setup.md is best base — has step-by-step structure). Absorb multi-machine narrative + full config example. Retire other two with redirects or archive notes.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

### F-034 🟢 Minor · structural · Windows guide cluster (2-way duplicate)
**Issue**: Two wiki files split a single Windows hybrid setup guide:
- `wiki/Windows-Setup-Summary-Example.md` (status/summary)
- `wiki/Windows-Hybrid-Backend-Setup-Example.md` (JSON config blocks)
**Fix**: Merge into single Windows Hybrid Setup guide; link from `wiki/02-Platform-Setup-Guide.md`.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

### F-035 🟢 Minor · structural · `wiki/Backend-Synchronization-Guide.md` is misnamed
**Issue**: Filename suggests sync documentation but content is actually a near-full project README from ~v7.1.3 era (features overview, what's new, docs table). Competes with `Cloudflare-Backup-Sync-Setup.md` for the "backend sync" search term.
**Fix**: Archive or rename. `Cloudflare-Backup-Sync-Setup.md` is the real sync guide.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

### F-036 🟢 Minor · structural · `docs/deployment/` duplicate guides
**Issue**: Both orphaned, both cover production deployment with overlapping audience but no cross-reference:
- `docs/deployment/dual-service.md` (FastMCP+HTTP two-service setup)
- `docs/deployment/production-guide.md` (single consolidated service with systemd, mDNS, HTTPS, consolidation)
**Fix**: Add cross-references and link both from README deployment section, OR merge into one "Production Deployment Guide" with sections for each topology.
**Disposition**: [ ] ship-now  [x] defer-issue  [ ] skip
**Issue**: #823

---

## Triage Notes

**Quick-pick recommendations** (override freely):
- All 🔴 Critical (F-003, F-007, F-008, F-015, F-016) → `ship-now` mandatory.
  - F-015 + F-016: **verify code first** before fixing — re-grep `src/` to confirm subagent's claims.
- All 🟡 Major content fixes (F-001, F-004, F-009 through F-014, F-017 through F-024, F-027) → `ship-now`.
- 🟡 Major structural (F-030 wiki Home navigation) → `ship-now` (single high-leverage fix).
- All 🟢 Minor cosmetic (F-002, F-005, F-006, F-025, F-026, F-028, F-029) → `defer-issue` (cheap to fix later).
- All 🟢 Minor structural (F-031 through F-036) → `defer-issue` (umbrella issues for cleanup waves).

**Categories with cleanest fixes (low risk):**
- F-013 (`n_results` → `limit`): mechanical sed-style replace.
- F-027 (`./TROUBLESHOOTING` → `./07-TROUBLESHOOTING`): mechanical rename.
- F-022 (MemPalace org): mechanical org-name replace.
- F-001, F-002, F-004, F-005 (version bumps): mechanical.

**Categories needing judgment:**
- F-007, F-008, F-018: requires confirming the v10 unified tool surface count (12 vs 18 vs claimed) by reading current `server_impl.py`.
- F-015, F-016, F-020: require verifying the env vars/backend names truly absent from code before deleting from wiki.
- F-031–F-036: structural cleanup — defer to follow-up work.

**Estimated PR size if all 🔴 + 🟡 marked ship-now:**
- ~12 main-repo files touched (CLAUDE.md, README.md, docs/index.html, docs/mastery/api-reference.md, docs/mastery/architecture-overview.md, docs/guides/memory-consolidation-guide.md, docs/BENCHMARKS.md if expanded, etc.)
- ~6 wiki files touched (01, 03, 04, 06, 10, 13, 14, Memory-Consolidation, Home)
- 1 new CI script + workflow change
- Comfortable single-PR scope (well under the ~30-file threshold).

---

## Resolution

**Status**: Complete (2026-05-02)

### Shipped in PR-1: [#827](https://github.com/doobidoo/mcp-memory-service/pull/827)

**Branch**: `chore/housekeeping-fixes` → `main` (15 commits, 16 files changed)

**Critical fixes** (4 of 4):
- F-003 → `3ca8595` (CLAUDE.md test count)
- F-007 → `36917f3` (api-reference.md MCP Tools rewrite)
- F-008 → wiki commit `2742b42` (Verify Connection tool names)
- F-015 → wiki commit `7c376af` (chroma backend removed)
- F-016 → wiki commit `7c376af` (5 non-existent OAuth env vars removed)

**Major fixes** (18 of 18):
- Main-repo: F-001, F-002 → `3ca8595` (landing page); F-009, F-012, F-013 → `36917f3`; F-014 → `5990371` (consolidation guide); F-022 → `3ca8595` (MemPalace org); F-018 → `177de87` (server docstring)
- Wiki: F-004, F-010, F-011, F-013-wiki, F-014-wiki, F-017, F-018-wiki, F-019, F-021, F-023, F-024, F-027, F-030 → wiki commits `9ccd9d8`, `2742b42`, `7c376af`, `9f7ca65`

### Wiki direct commits (`mcp-memory-service.wiki` master)

- `9ccd9d8` — Home navigation rewrite (F-030, highest leverage)
- `2742b42` — v10.0.0 unified tool surface alignment (F-008, F-010, F-011, F-013, F-014, F-018)
- `7c376af` — config + storage backend defaults (F-015, F-016, F-017, F-019, F-021)
- `9f7ca65` — version bump + broken page links (F-004, F-023, F-024, F-027)

### Tooling shipped

- New: `scripts/ci/check_versions.sh` + `tests/ci/test_check_versions.sh` (TDD, 3/3 pass)
- New: `version-drift-check` job in `.github/workflows/docs-check.yml` (GREEN on first PR run)
- Extended: `scripts/ci/check_dead_refs.sh` — added `docs/superpowers/` to EXCLUDE_PATHS

### Deferred to issues

- [#823](https://github.com/doobidoo/mcp-memory-service/issues/823) — Doc structural cleanup: orphans + duplicates (F-031, F-032, F-033, F-034, F-035, F-036)
- [#824](https://github.com/doobidoo/mcp-memory-service/issues/824) — Wiki TOC anchor rot (F-028, F-029) [`good first issue`]
- [#825](https://github.com/doobidoo/mcp-memory-service/issues/825) — Wiki broken page-link redirects (F-025, F-026)
- [#826](https://github.com/doobidoo/mcp-memory-service/issues/826) — Cosmetic version-ref bumps (F-005, F-006) [`good first issue`]

### Skipped (intentional)

- F-020 (`MCP_MEMORY_HTTP_ENDPOINT`): verified FALSE — var IS implemented in `examples/http-mcp-bridge.js:157`. Wiki references are correct documentation of the client-side bridge env var.

### Manual follow-up

- Landing page (`docs/index.html`) re-publish to here.now (`merry-realm-j835`): `~/.agents/skills/here-now/scripts/publish.sh` not present at the documented path. Owner to republish manually after merge per CLAUDE.md instructions.

### Future tooling work (intentionally deferred from PR-1)

- Adding v10.0.0 deprecated MCP tool names (`store_memory(`, `retrieve_memory(`, etc.) to `check_dead_refs.sh` — deferred because 25+ active docs still use them as primary examples (mostly tutorials/agents/examples). Adding term-by-term as docs are migrated to the unified `memory_*` surface (tracked as part of #823 cleanup waves).
- Wiki CI link-check (cross-repo handling) — non-trivial; deferred per design spec.
