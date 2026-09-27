# Codeberg-Nummern-Mapping

Das Projekt lag von Juni bis September 2026 auf Codeberg. Jede Issue- und
PR-Nummer **unter 341**, die in CHANGELOG, README, Commit-Messages oder
Kommentaren auftaucht, ist eine Codeberg-Nummer -- GitHub verlinkt sie
automatisch auf eine gleichnamige eigene Nummer, und das ist jedes Mal ein
anderer Vorgang. Diese Tabelle loest sie auf.

Das Archiv unter <https://codeberg.org/doobidoo/mcp-memory-service> bleibt lesbar; die Links unten fuehren dorthin.

Beim Umzug am 2026-09-05 wurden die damals **offenen** Issues auf GitHub neu
angelegt (Label `migrated:codeberg`) und die Codeberg-Vorgaenger mit einem
Verweis geschlossen. Geschlossene Issues und alle PRs wurden nicht kopiert --
sie sind erledigt, und das Archiv traegt sie weiter.

Stand des Exports: 2026-09-05. 109 Issues, 231 PRs.

## Issues

| Codeberg | Titel | Zustand dort | Weitergefuehrt auf GitHub |
|---|---|---|---|
| [1](https://codeberg.org/doobidoo/mcp-memory-service/issues/1) | RFC: Self-Service Memory Intelligence for Ephemeral AI Agents | closed | - |
| [7](https://codeberg.org/doobidoo/mcp-memory-service/issues/7) | RFC: Structural Improvements (§8–§12) | closed | - |
| [8](https://codeberg.org/doobidoo/mcp-memory-service/issues/8) | Discussion: Unify storage backend maintenance burden | closed | - |
| [9](https://codeberg.org/doobidoo/mcp-memory-service/issues/9) | Discussion: Web/OAuth layer refactoring | closed | - |
| [10](https://codeberg.org/doobidoo/mcp-memory-service/issues/10) | Discussion: Deprecation plan for legacy tool names | closed | - |
| [11](https://codeberg.org/doobidoo/mcp-memory-service/issues/11) | RFC: Formal schema versioning and migration tool | closed | - |
| [19](https://codeberg.org/doobidoo/mcp-memory-service/issues/19) | feat: harvest support for OpenClaw gateway session transcripts | closed | - |
| [21](https://codeberg.org/doobidoo/mcp-memory-service/issues/21) | feat: multilingual embedding model documentation + re-embed maintenance tool | closed | - |
| [27](https://codeberg.org/doobidoo/mcp-memory-service/issues/27) | Make the Web UI update-check git remote/branch configurable via env | closed | - |
| [34](https://codeberg.org/doobidoo/mcp-memory-service/issues/34) | bug: graph.py INSERTs reference non-existent columns — all graph writes silently fail | closed | - |
| [43](https://codeberg.org/doobidoo/mcp-memory-service/issues/43) | fix(harvest): OpenClaw trajectory support (file discovery + role filter + noise) | closed | - |
| [53](https://codeberg.org/doobidoo/mcp-memory-service/issues/53) | v11.0.0 — Phase 2: Legacy Removal + Optional ML Dependencies | closed | - |
| [54](https://codeberg.org/doobidoo/mcp-memory-service/issues/54) | feat(ingestion): pluggable domain-specific NER pipeline as a hook | closed | - |
| [55](https://codeberg.org/doobidoo/mcp-memory-service/issues/55) | feat(graph): composite scoring — weigh hop distance + entity centrality in retrieval | closed | - |
| [56](https://codeberg.org/doobidoo/mcp-memory-service/issues/56) | RFC: two-phase query API (knowledge map -> entity detail) | closed | - |
| [57](https://codeberg.org/doobidoo/mcp-memory-service/issues/57) | RFC: Multi-store architecture with federated retrieval | closed | - |
| [59](https://codeberg.org/doobidoo/mcp-memory-service/issues/59) | ci: optional ML coverage lane (semantic tests skipped on 3.7GB runner) | open | [#1093](https://github.com/doobidoo/mcp-memory-service/issues/1093) |
| [61](https://codeberg.org/doobidoo/mcp-memory-service/issues/61) | impl: two-phase query API (memory_explore / memory_detail) — tracks #56 | closed | - |
| [67](https://codeberg.org/doobidoo/mcp-memory-service/issues/67) | post-V11 backlog: parked ideas / RFCs (reviewed in the consolidation window) | open | Sammel-Tracker, nicht migriert |
| [68](https://codeberg.org/doobidoo/mcp-memory-service/issues/68) | bug: multi-store vec0 migration aborts on existing DB — memory writes blocked (regression from #62) | closed | - |
| [80](https://codeberg.org/doobidoo/mcp-memory-service/issues/80) | feat(harvest): run LLM classification as background task | closed | - |
| [81](https://codeberg.org/doobidoo/mcp-memory-service/issues/81) | feat(mistake_notes): add mistake_note_update and mistake_note_delete tools | closed | - |
| [98](https://codeberg.org/doobidoo/mcp-memory-service/issues/98) | [Bug]: bug: No module named 'numpy' on bare install after §11 optional-ML refactor | closed | - |
| [100](https://codeberg.org/doobidoo/mcp-memory-service/issues/100) | feat: memory_consolidate — merge multiple memories into one (MCP tool) | closed | - |
| [101](https://codeberg.org/doobidoo/mcp-memory-service/issues/101) | [Feature]: Optional YAML-LD frontmatter + RDF/SHACL layer for linked-data memory (Vault-LD alignment) | open | [#1094](https://github.com/doobidoo/mcp-memory-service/issues/1094) |
| [102](https://codeberg.org/doobidoo/mcp-memory-service/issues/102) | [Feature]: Hermes Agent state.db data source adapter (auto-harvest) | open | [#1095](https://github.com/doobidoo/mcp-memory-service/issues/1095) |
| [104](https://codeberg.org/doobidoo/mcp-memory-service/issues/104) | RFC: Session transcript mining — retroactive knowledge extraction from agent sessions | closed | - |
| [112](https://codeberg.org/doobidoo/mcp-memory-service/issues/112) | merge action response: content_hash field contains success message instead of new memory hash | closed | - |
| [113](https://codeberg.org/doobidoo/mcp-memory-service/issues/113) | [Feature]: Per-agent rate limiting for MCP tool calls | open | [#1096](https://github.com/doobidoo/mcp-memory-service/issues/1096) |
| [114](https://codeberg.org/doobidoo/mcp-memory-service/issues/114) | [Feature]: Export internal metrics via Prometheus endpoint (optional OTel) | open | [#1097](https://github.com/doobidoo/mcp-memory-service/issues/1097) |
| [115](https://codeberg.org/doobidoo/mcp-memory-service/issues/115) | [Feature]: Export internal metrics via Prometheus endpoint (optional OTel) | closed | - |
| [116](https://codeberg.org/doobidoo/mcp-memory-service/issues/116) | [Feature]: Optional LLM fallback for auto_capture when heuristic confidence is low | open | [#1098](https://github.com/doobidoo/mcp-memory-service/issues/1098) |
| [117](https://codeberg.org/doobidoo/mcp-memory-service/issues/117) | [Feature]: Layered embedding cache (query LRU + content-addressed skip) | open | [#1099](https://github.com/doobidoo/mcp-memory-service/issues/1099) |
| [118](https://codeberg.org/doobidoo/mcp-memory-service/issues/118) | [Feature]: Expand agent_id from commit_session to all memories + cross-agent conflict resolution | open | [#1100](https://github.com/doobidoo/mcp-memory-service/issues/1100) |
| [119](https://codeberg.org/doobidoo/mcp-memory-service/issues/119) | [Feature]: Temporal decay for search relevance (recent memories rank higher) | closed | - |
| [120](https://codeberg.org/doobidoo/mcp-memory-service/issues/120) | [Bug]: memory_context handler uses hasattr() on dict — always returns empty | closed | - |
| [121](https://codeberg.org/doobidoo/mcp-memory-service/issues/121) | [RFC]: Belief derivation pipeline — promote repeated observations to active beliefs | closed | - |
| [133](https://codeberg.org/doobidoo/mcp-memory-service/issues/133) | [Bug]: Milvus signature mistmatch | closed | - |
| [134](https://codeberg.org/doobidoo/mcp-memory-service/issues/134) | [Bug]: multi-store vec0 migration reads embedding dimension before model init and DROPs memory_embeddings in autocommit — permanent embedding loss on non-384-dim databases | closed | - |
| [135](https://codeberg.org/doobidoo/mcp-memory-service/issues/135) | [Bug]: hash-embedding fallback silently writes SHA256 pseudo-vectors into the same vec0 table as real embeddings | closed | - |
| [136](https://codeberg.org/doobidoo/mcp-memory-service/issues/136) | [Bug]: `memory status` / `check-db` report "Service is healthy" while the embedding stack is missing or dimension-mismatched | closed | - |
| [143](https://codeberg.org/doobidoo/mcp-memory-service/issues/143) | Silent MiniLM-384 fallback: custom embedding model ignored + no dimension guard (sqlite_vec) | closed | - |
| [150](https://codeberg.org/doobidoo/mcp-memory-service/issues/150) | bug(graph): _prune_orphaned_graph_edges deletes all has_entity edges every consolidation cycle | closed | - |
| [155](https://codeberg.org/doobidoo/mcp-memory-service/issues/155) | [Bug]: Plugin installed from Marketplace doesn't always use ~/.claude/hooks/config.json | closed | - |
| [158](https://codeberg.org/doobidoo/mcp-memory-service/issues/158) | [Bug]: Outdated github.com references | closed | - |
| [162](https://codeberg.org/doobidoo/mcp-memory-service/issues/162) | [Bug]: Docker :latest ships onnxruntime without tokenizers — published standard image cannot load any real embedding backend | closed | - |
| [163](https://codeberg.org/doobidoo/mcp-memory-service/issues/163) | [Bug]: Docker latest image lacks tokenizers, causing ONNX initialization failure | closed | - |
| [170](https://codeberg.org/doobidoo/mcp-memory-service/issues/170) | [Bug]: Cannot run the local quality classifier in the official docker image | closed | - |
| [171](https://codeberg.org/doobidoo/mcp-memory-service/issues/171) | docs+config: retire the :quality-cpu tag, make the ONNX quality model dir configurable | closed | - |
| [172](https://codeberg.org/doobidoo/mcp-memory-service/issues/172) | docker: standard image installs CPU torch it cannot use | closed | - |
| [173](https://codeberg.org/doobidoo/mcp-memory-service/issues/173) | quality: torch.onnx.export fails on DeBERTa since torch 2.9 flipped dynamo to True | open | [#1101](https://github.com/doobidoo/mcp-memory-service/issues/1101) |
| [174](https://codeberg.org/doobidoo/mcp-memory-service/issues/174) | quality: openai-compatible scorer is not a like-for-like substitute for the local classifier | open | [#1102](https://github.com/doobidoo/mcp-memory-service/issues/1102) |
| [175](https://codeberg.org/doobidoo/mcp-memory-service/issues/175) | docker: :slim image is still on Python 3.10 (EOL Oct 2026) while the standard image is 3.12 | closed | - |
| [176](https://codeberg.org/doobidoo/mcp-memory-service/issues/176) | ontology: memory type validation is case-sensitive and custom type names cannot contain hyphens | closed | - |
| [177](https://codeberg.org/doobidoo/mcp-memory-service/issues/177) | claude-hooks: auto-capture and session-end send memory types the server ontology rejects | closed | - |
| [178](https://codeberg.org/doobidoo/mcp-memory-service/issues/178) | harvest: rewriter stays disabled unless GROQ_API_KEY is set, even with HARVEST_LLM_PROVIDERS configured | closed | - |
| [179](https://codeberg.org/doobidoo/mcp-memory-service/issues/179) | quality: MCP_QUALITY_BOOST_ENABLED/WEIGHT control two unrelated things (search reranking vs stored score) | closed | - |
| [188](https://codeberg.org/doobidoo/mcp-memory-service/issues/188) | [Feature]: Include `scripts/maintenance` and `scripts/migration` in the docker image | closed | - |
| [189](https://codeberg.org/doobidoo/mcp-memory-service/issues/189) | [Bug]: Running migrate_sqlite_vec_embeddings.py silently drops memory_graph | closed | - |
| [197](https://codeberg.org/doobidoo/mcp-memory-service/issues/197) | claude-hooks/config.json is tracked in git and appears to contain a live credential | closed | - |
| [202](https://codeberg.org/doobidoo/mcp-memory-service/issues/202) | ci: the claude-hooks node test suite never runs in CI | closed | - |
| [203](https://codeberg.org/doobidoo/mcp-memory-service/issues/203) | config.template.json has drifted out of sync with the shipped hooks config | closed | - |
| [213](https://codeberg.org/doobidoo/mcp-memory-service/issues/213) | [Bug]: Milvus throws 501 Not Implemented for /api/tags | closed | - |
| [215](https://codeberg.org/doobidoo/mcp-memory-service/issues/215) | [Feature]: LLM-internal summerization of memories | open | [#1103](https://github.com/doobidoo/mcp-memory-service/issues/1103) |
| [217](https://codeberg.org/doobidoo/mcp-memory-service/issues/217) | docs: memory_explore / memory_detail are undocumented outside an internal handoff note | closed | - |
| [218](https://codeberg.org/doobidoo/mcp-memory-service/issues/218) | entities: maintain's batch extraction drops every tag (metadata vs. attribute, 4th instance) | closed | - |
| [219](https://codeberg.org/doobidoo/mcp-memory-service/issues/219) | graph: extract_entities and memory_search's entity filter are gated on storage.graph, which nothing sets | closed | - |
| [220](https://codeberg.org/doobidoo/mcp-memory-service/issues/220) | memory_explore: entity selection ignores the query, and unmatched entities inherit the query's chunks | open | [#1104](https://github.com/doobidoo/mcp-memory-service/issues/1104) |
| [228](https://codeberg.org/doobidoo/mcp-memory-service/issues/228) | sqlite_vec: the hash-fallback refusal reports memories + embedding rows as "existing memories" | closed | - |
| [233](https://codeberg.org/doobidoo/mcp-memory-service/issues/233) | cli: memory status --storage-backend hybrid raises ValueError (get_storage has no hybrid branch) | closed | - |
| [234](https://codeberg.org/doobidoo/mcp-memory-service/issues/234) | tests: conftest does not scrub MCP_EMBEDDING_MODEL, so host env decides the embedding dimension | closed | - |
| [235](https://codeberg.org/doobidoo/mcp-memory-service/issues/235) | CoreML ONNX provider fails on dynamic sequence lengths | open | [#1105](https://github.com/doobidoo/mcp-memory-service/issues/1105) |
| [240](https://codeberg.org/doobidoo/mcp-memory-service/issues/240) | HTTP API and web dashboard have no store/partition support: analytics read all stores, CRUD and search read only "default" | open | [#1106](https://github.com/doobidoo/mcp-memory-service/issues/1106) |
| [243](https://codeberg.org/doobidoo/mcp-memory-service/issues/243) | time_parser builds recall windows in local time while timeframe-delete uses UTC (post-#237 split) | closed | - |
| [244](https://codeberg.org/doobidoo/mcp-memory-service/issues/244) | milvus: filter-expression escaping misses the backslash, so a trailing backslash breaks the query | open | [#1107](https://github.com/doobidoo/mcp-memory-service/issues/1107) |
| [250](https://codeberg.org/doobidoo/mcp-memory-service/issues/250) | harvest: LLM classification runs inline, so a large session can outlive the client timeout and hold the loop | open | [#1108](https://github.com/doobidoo/mcp-memory-service/issues/1108) |
| [253](https://codeberg.org/doobidoo/mcp-memory-service/issues/253) | maintain: metadata["tags"] as a comma-string explodes into single-character entities | closed | - |
| [254](https://codeberg.org/doobidoo/mcp-memory-service/issues/254) | memory_health / /api/health/detailed report hardcoded "all-MiniLM-L6-v2" when external embeddings are active | closed | - |
| [255](https://codeberg.org/doobidoo/mcp-memory-service/issues/255) | plugins: memory_search bypasses fire('on_retrieve') — the hook never sees the primary retrieval path | open | [#1109](https://github.com/doobidoo/mcp-memory-service/issues/1109) |
| [256](https://codeberg.org/doobidoo/mcp-memory-service/issues/256) | dashboard graph visualization: nodes selected by connection_count including has_entity edges, but only memory<->memory edges are drawn -> screen full of isolated dots | closed | - |
| [257](https://codeberg.org/doobidoo/mcp-memory-service/issues/257) | dashboard: fullscreen 3D graph — legend and controls are near-invisible (transparent overlay on dark scene) | open | [#1110](https://github.com/doobidoo/mcp-memory-service/issues/1110) |
| [260](https://codeberg.org/doobidoo/mcp-memory-service/issues/260) | harvest: the LLM classifier has no pacing, batching or backoff, so a large candidate set throttles itself | open | [#1111](https://github.com/doobidoo/mcp-memory-service/issues/1111) |
| [261](https://codeberg.org/doobidoo/mcp-memory-service/issues/261) | milvus: the <3.0.0 pin was widened back by a dependency bump, and no CI job here tests Milvus at all | open | [#1112](https://github.com/doobidoo/mcp-memory-service/issues/1112) |
| [262](https://codeberg.org/doobidoo/mcp-memory-service/issues/262) | cloudflare/hybrid: external embedding APIs are refused outright, so the recommended backend is locked to Workers AI | open | [#1113](https://github.com/doobidoo/mcp-memory-service/issues/1113) |
| [263](https://codeberg.org/doobidoo/mcp-memory-service/issues/263) | Daily Triage Digest | open | entspricht [#805](https://github.com/doobidoo/mcp-memory-service/issues/805) |
| [264](https://codeberg.org/doobidoo/mcp-memory-service/issues/264) | dashboard: concentric ring layout for the graph, with animated re-center on click | open | [#1114](https://github.com/doobidoo/mcp-memory-service/issues/1114) |
| [274](https://codeberg.org/doobidoo/mcp-memory-service/issues/274) | tests: test_write_performance asserts a 100ms wall-clock budget it misses about half the time locally | closed | - |
| [275](https://codeberg.org/doobidoo/mcp-memory-service/issues/275) | stdio: with LM Studio detected, 7 non-JSON lines land on the JSON-RPC channel before the initialize response | closed | - |
| [278](https://codeberg.org/doobidoo/mcp-memory-service/issues/278) | transformers 5.x migration: two advisories have no 4.x fix | closed | - |
| [280](https://codeberg.org/doobidoo/mcp-memory-service/issues/280) | lifecycle.launch() is complexity 26, so the quality gate blocks any change to the file | open | [#1115](https://github.com/doobidoo/mcp-memory-service/issues/1115) |
| [281](https://codeberg.org/doobidoo/mcp-memory-service/issues/281) | memory launch and run_http_server.py still differ: cert generation, and the shipped plist runs the legacy script | open | [#1116](https://github.com/doobidoo/mcp-memory-service/issues/1116) |
| [282](https://codeberg.org/doobidoo/mcp-memory-service/issues/282) | SSE auth puts the API key in the URL, so it lands in the access log in cleartext | open | [#1117](https://github.com/doobidoo/mcp-memory-service/issues/1117) |
| [283](https://codeberg.org/doobidoo/mcp-memory-service/issues/283) | lm_studio_compat Patch 2 has been inert: BaseSession._handle_notification no longer exists | closed | - |
| [292](https://codeberg.org/doobidoo/mcp-memory-service/issues/292) | quality_gate.sh blocks on complexity findings it labels non-blocking | open | [#1118](https://github.com/doobidoo/mcp-memory-service/issues/1118) |
| [293](https://codeberg.org/doobidoo/mcp-memory-service/issues/293) | 163 pre-existing unsanitised f-string logger calls make check 6.5 unpassable for three files | open | [#1119](https://github.com/doobidoo/mcp-memory-service/issues/1119) |
| [296](https://codeberg.org/doobidoo/mcp-memory-service/issues/296) | docker: published images set MCP_MEMORY_SQLITE_PATH to a directory, so a bare docker run cannot start | closed | - |
| [301](https://codeberg.org/doobidoo/mcp-memory-service/issues/301) | Migrate to transformers 5.x -- the <5.0.0 bound is holding two high-severity advisories open | closed | - |
| [304](https://codeberg.org/doobidoo/mcp-memory-service/issues/304) | onnx_ranker picks an arbitrary HF snapshot directory and breaks on an interrupted download | closed | - |
| [306](https://codeberg.org/doobidoo/mcp-memory-service/issues/306) | Two quality tests fail once an ONNX model is actually available | closed | - |
| [308](https://codeberg.org/doobidoo/mcp-memory-service/issues/308) | 19 test files import the package through a src. prefix, creating a shadow copy | open | [#1120](https://github.com/doobidoo/mcp-memory-service/issues/1120) |
| [312](https://codeberg.org/doobidoo/mcp-memory-service/issues/312) | ci: check_dead_refs.sh is wired nowhere, so the docs gate CLAUDE.md documents does not exist | open | [#1121](https://github.com/doobidoo/mcp-memory-service/issues/1121) |
| [313](https://codeberg.org/doobidoo/mcp-memory-service/issues/313) | .gitignore: .venv/ misses the symlinked .venv in a worktree, so git add -A stages it | closed | - |
| [315](https://codeberg.org/doobidoo/mcp-memory-service/issues/315) | quality: a disabled quality system still loads (and exports) the ONNX ranker | closed | - |
| [317](https://codeberg.org/doobidoo/mcp-memory-service/issues/317) | time_parser: naive datetimes make every time_expr query off by the host UTC offset (#237 fixed deletion only) | open | [#1122](https://github.com/doobidoo/mcp-memory-service/issues/1122) |
| [318](https://codeberg.org/doobidoo/mcp-memory-service/issues/318) | quality: the async scorer runs a synchronous ONNX export on the event loop | open | [#1123](https://github.com/doobidoo/mcp-memory-service/issues/1123) |
| [324](https://codeberg.org/doobidoo/mcp-memory-service/issues/324) | consolidate recommend returns identical results for weekly and monthly | closed | - |
| [326](https://codeberg.org/doobidoo/mcp-memory-service/issues/326) | consolidation: scikit-learn is declared in no extra, so the default clustering algorithm can never run | closed | - |
| [327](https://codeberg.org/doobidoo/mcp-memory-service/issues/327) | consolidation: after #325 no horizon reaches past 365 days, so forgetting can never see the stale tail | open | [#1124](https://github.com/doobidoo/mcp-memory-service/issues/1124) |
| [328](https://codeberg.org/doobidoo/mcp-memory-service/issues/328) | consolidation health checks return hardcoded literals, so every engine reports HEALTHY unconditionally | open | [#1125](https://github.com/doobidoo/mcp-memory-service/issues/1125) |

## Pull Requests

Nicht migriert. Der Code ist ueber `main` mitgekommen; die Tabelle existiert,
damit eine Referenz wie "PR #289" nachschlagbar bleibt.

| Codeberg | Titel | Zustand |
|---|---|---|
| [2](https://codeberg.org/doobidoo/mcp-memory-service/pulls/2) | feat(harvest): §0 Quality Pipeline — sentence extraction, role filter, LLM rewriter | merged |
| [3](https://codeberg.org/doobidoo/mcp-memory-service/pulls/3) | feat(ontology): §1 Observation Store — new observation subtypes | merged |
| [4](https://codeberg.org/doobidoo/mcp-memory-service/pulls/4) | feat(consolidation): §3 Consolidation Engine — memory_distill, schedulers, onboarding guide | closed |
| [5](https://codeberg.org/doobidoo/mcp-memory-service/pulls/5) | feat(bootstrap): §4 Bootstrap Profile + §5 Session Legacy | closed |
| [6](https://codeberg.org/doobidoo/mcp-memory-service/pulls/6) | feat(mcp): §7 MCP Protocol Integration — resource URI, onboarding, docs | closed |
| [12](https://codeberg.org/doobidoo/mcp-memory-service/pulls/12) | ci(codeberg): Forgejo release workflow + runner runbook | merged |
| [13](https://codeberg.org/doobidoo/mcp-memory-service/pulls/13) | feat: #11 Schema Versioning — migration registry + CLI | merged |
| [14](https://codeberg.org/doobidoo/mcp-memory-service/pulls/14) | ci(coverage): establish 55% baseline + enforce in CI | merged |
| [15](https://codeberg.org/doobidoo/mcp-memory-service/pulls/15) | refactor: extract inline handlers from server_impl.py (#7) | merged |
| [16](https://codeberg.org/doobidoo/mcp-memory-service/pulls/16) | ci(codeberg): port Docker Hub image cleanup to Forgejo | merged |
| [17](https://codeberg.org/doobidoo/mcp-memory-service/pulls/17) | ci(codeberg): cleanup fails loudly on delete errors | merged |
| [18](https://codeberg.org/doobidoo/mcp-memory-service/pulls/18) | docs(changelog): §0 harvest + Codeberg CI under Unreleased | merged |
| [20](https://codeberg.org/doobidoo/mcp-memory-service/pulls/20) | feat(harvest): OpenClaw trajectory parser + pt_BR v2 patterns (#19) | merged |
| [22](https://codeberg.org/doobidoo/mcp-memory-service/pulls/22) | docs: multilingual embedding model selection + re-embed guide (#21) | merged |
| [23](https://codeberg.org/doobidoo/mcp-memory-service/pulls/23) | fix(ci): repair stale §0 test + exclude benchmark tests | merged |
| [24](https://codeberg.org/doobidoo/mcp-memory-service/pulls/24) | ci diagnostic (do not merge) | closed |
| [25](https://codeberg.org/doobidoo/mcp-memory-service/pulls/25) | ci: scope to deterministic subset + report-only coverage | merged |
| [26](https://codeberg.org/doobidoo/mcp-memory-service/pulls/26) | chore(ci): remove leftover GitHub Actions workflows (Codeberg migration cleanup) | merged |
| [28](https://codeberg.org/doobidoo/mcp-memory-service/pulls/28) | chore(agents): add codeberg-release-manager, deprecate github-release-manager | merged |
| [29](https://codeberg.org/doobidoo/mcp-memory-service/pulls/29) | fix(agents): codeberg-release-manager tag detection (dry-run finding) | merged |
| [30](https://codeberg.org/doobidoo/mcp-memory-service/pulls/30) | chore: release v10.71.0 | merged |
| [31](https://codeberg.org/doobidoo/mcp-memory-service/pulls/31) | fix(ci): publish lite package with the release version (unstick from 10.39.1) | merged |
| [32](https://codeberg.org/doobidoo/mcp-memory-service/pulls/32) | fix(milvus): support ranked mode and ranking_weights in search_memories | merged |
| [33](https://codeberg.org/doobidoo/mcp-memory-service/pulls/33) | feat(beliefs): §2 Belief Store — observation-to-belief derivation pipeline | merged |
| [35](https://codeberg.org/doobidoo/mcp-memory-service/pulls/35) | chore(release): v10.72.0 | merged |
| [36](https://codeberg.org/doobidoo/mcp-memory-service/pulls/36) | feat(anti-hallucination): §6 — belief-aware quarantine pipeline | merged |
| [37](https://codeberg.org/doobidoo/mcp-memory-service/pulls/37) | refactor(dispatch): §13 — wire TOOL_REGISTRY + routing table into server_impl.py | merged |
| [38](https://codeberg.org/doobidoo/mcp-memory-service/pulls/38) | refactor(config): §9 — split config.py into domain modules | merged |
| [39](https://codeberg.org/doobidoo/mcp-memory-service/pulls/39) | feat(consolidation): §3 Consolidation Engine — memory_distill, schedulers, onboarding guide (rebase of #4) | merged |
| [40](https://codeberg.org/doobidoo/mcp-memory-service/pulls/40) | feat(bootstrap): §4 Bootstrap Profile + §5 Session Legacy (rebase of #5) | merged |
| [41](https://codeberg.org/doobidoo/mcp-memory-service/pulls/41) | chore(release): v10.73.0 | merged |
| [42](https://codeberg.org/doobidoo/mcp-memory-service/pulls/42) | refactor(storage): §10 — decompose sqlite_vec.py into focused mixins | merged |
| [44](https://codeberg.org/doobidoo/mcp-memory-service/pulls/44) | fix(harvest): OpenClaw trajectory support (file discovery + role filter) | closed |
| [45](https://codeberg.org/doobidoo/mcp-memory-service/pulls/45) | chore(release): v10.74.0 | merged |
| [46](https://codeberg.org/doobidoo/mcp-memory-service/pulls/46) | fix(harvest): reject OpenClaw prompt preamble noise in PatternExtractor | merged |
| [47](https://codeberg.org/doobidoo/mcp-memory-service/pulls/47) | chore(release): v10.74.1 | merged |
| [48](https://codeberg.org/doobidoo/mcp-memory-service/pulls/48) | docs(readme): add v10.74.1 patch release entry | merged |
| [49](https://codeberg.org/doobidoo/mcp-memory-service/pulls/49) | feat(deps): §11 — make torch/transformers optional, ONNX-first fallback | merged |
| [50](https://codeberg.org/doobidoo/mcp-memory-service/pulls/50) | refactor(storage): Phase 1 — extract shared utilities into storage/shared.py | merged |
| [51](https://codeberg.org/doobidoo/mcp-memory-service/pulls/51) | docs(oauth): add quickstart guide with 3 minimal recipes | closed |
| [52](https://codeberg.org/doobidoo/mcp-memory-service/pulls/52) | feat/legacy-deprecation-phase1 | merged |
| [58](https://codeberg.org/doobidoo/mcp-memory-service/pulls/58) | docs(readme): add In the Media section with interview + testimonial | merged |
| [60](https://codeberg.org/doobidoo/mcp-memory-service/pulls/60) | feat(v11): remove legacy tool aliases (Block A, Issue #53 Step 3) | closed |
| [62](https://codeberg.org/doobidoo/mcp-memory-service/pulls/62) | feat(storage): multi-store with partition key — Phase 1 schema (#57) | merged |
| [63](https://codeberg.org/doobidoo/mcp-memory-service/pulls/63) | chore(hooks): scope PostToolUse tests to edited file | merged |
| [64](https://codeberg.org/doobidoo/mcp-memory-service/pulls/64) | feat(tools): add memory_explore + memory_detail two-phase query API (#56/#61) | closed |
| [65](https://codeberg.org/doobidoo/mcp-memory-service/pulls/65) | feat(graph): two-phase query handler skeletons (#56/#61) | merged |
| [66](https://codeberg.org/doobidoo/mcp-memory-service/pulls/66) | feat(hooks): SessionStart guard against shared-worktree collisions | merged |
| [69](https://codeberg.org/doobidoo/mcp-memory-service/pulls/69) | fix(storage): multi-store vec0 migration aborts on existing DB (#68) | merged |
| [70](https://codeberg.org/doobidoo/mcp-memory-service/pulls/70) | chore(ci): quality gate skips AI checks gracefully when Gemini CLI absent | merged |
| [71](https://codeberg.org/doobidoo/mcp-memory-service/pulls/71) | docs: migrate legacy tool-name examples to current registry names (pre-V11) | merged |
| [72](https://codeberg.org/doobidoo/mcp-memory-service/pulls/72) | feat(v11)!: remove legacy tool-name aliases (Issue #53 Step 3, supersedes #60) | merged |
| [73](https://codeberg.org/doobidoo/mcp-memory-service/pulls/73) | chore: release v11.0.0 -- MAJOR: legacy alias removal + optional ML deps | merged |
| [74](https://codeberg.org/doobidoo/mcp-memory-service/pulls/74) | docs(landing): repoint dead GitHub links to Codeberg + fix blog canonical | merged |
| [75](https://codeberg.org/doobidoo/mcp-memory-service/pulls/75) | docs(landing): swap GitHub octocat logo + label for Codeberg on buttons | merged |
| [76](https://codeberg.org/doobidoo/mcp-memory-service/pulls/76) | fix(maintenance): make ontology + dedup scripts API-key aware | merged |
| [77](https://codeberg.org/doobidoo/mcp-memory-service/pulls/77) | feat(graph): composite scoring — relevance + proximity + centrality (#55) | merged |
| [78](https://codeberg.org/doobidoo/mcp-memory-service/pulls/78) | feat(graph): two-phase aggregation — entity-specific chunks + extractive summary (#56/#61) | merged |
| [79](https://codeberg.org/doobidoo/mcp-memory-service/pulls/79) | feat(extraction): multi-locale heuristic NER — PT-BR + EN (#54) | closed |
| [82](https://codeberg.org/doobidoo/mcp-memory-service/pulls/82) | fix(maintenance): harden find_duplicates load_config (PR #76 review minors) | merged |
| [83](https://codeberg.org/doobidoo/mcp-memory-service/pulls/83) | docs(changelog): record unreleased maintenance-script fixes (#76, #82) | merged |
| [84](https://codeberg.org/doobidoo/mcp-memory-service/pulls/84) | fix(opencode): move status file to XDG state directory | merged |
| [85](https://codeberg.org/doobidoo/mcp-memory-service/pulls/85) | chore: release v11.1.0 | merged |
| [86](https://codeberg.org/doobidoo/mcp-memory-service/pulls/86) | chore(gitignore): ignore .env.bak* token/secret backups | merged |
| [87](https://codeberg.org/doobidoo/mcp-memory-service/pulls/87) | docs(changelog): move opencode XDG fix (#84) to [Unreleased] | merged |
| [88](https://codeberg.org/doobidoo/mcp-memory-service/pulls/88) | docs: document auth rate-limit and body-size env vars | closed |
| [89](https://codeberg.org/doobidoo/mcp-memory-service/pulls/89) | docs(changelog): add opt-in composite scoring (#55/#77) to [Unreleased] | merged |
| [90](https://codeberg.org/doobidoo/mcp-memory-service/pulls/90) | fix(storage): prevent orphaned-embedding rowid collisions in sqlite-vec | merged |
| [91](https://codeberg.org/doobidoo/mcp-memory-service/pulls/91) | fix(oauth): security hardening + proxy-aware rate limiting (supersedes #88) | merged |
| [92](https://codeberg.org/doobidoo/mcp-memory-service/pulls/92) | chore: release v11.2.0 | merged |
| [93](https://codeberg.org/doobidoo/mcp-memory-service/pulls/93) | feat(web): make update-check git remote/branch configurable via env (#27) | merged |
| [94](https://codeberg.org/doobidoo/mcp-memory-service/pulls/94) | feat(web): cinematic 3D knowledge graph (Orrery-style) — PoC | merged |
| [95](https://codeberg.org/doobidoo/mcp-memory-service/pulls/95) | docs: feature the 3D knowledge graph screenshot | merged |
| [96](https://codeberg.org/doobidoo/mcp-memory-service/pulls/96) | docs: link the 3D knowledge graph promo video | merged |
| [97](https://codeberg.org/doobidoo/mcp-memory-service/pulls/97) | fix(hooks): auto-capture once per turn on Stop + word-boundary patterns | merged |
| [99](https://codeberg.org/doobidoo/mcp-memory-service/pulls/99) | chore: release v11.3.2 | merged |
| [103](https://codeberg.org/doobidoo/mcp-memory-service/pulls/103) | feat(consolidate): add merge action — atomic multi-memory merge | closed |
| [105](https://codeberg.org/doobidoo/mcp-memory-service/pulls/105) | feat: add merge action to memory_consolidate tool (#100) | merged |
| [106](https://codeberg.org/doobidoo/mcp-memory-service/pulls/106) | feat(site): mcpmemory.services landing page, galaxy hero video, in-browser demo | merged |
| [107](https://codeberg.org/doobidoo/mcp-memory-service/pulls/107) | feat(ner): pluggable domain-specific entity extractors (#54) | merged |
| [108](https://codeberg.org/doobidoo/mcp-memory-service/pulls/108) | chore: release v11.4.0 | merged |
| [109](https://codeberg.org/doobidoo/mcp-memory-service/pulls/109) | chore: release v11.4.0 | closed |
| [110](https://codeberg.org/doobidoo/mcp-memory-service/pulls/110) | docs(site): correct test count on landing page (2,602 collected) | merged |
| [111](https://codeberg.org/doobidoo/mcp-memory-service/pulls/111) | feat(ci): auto-deploy mcpmemory.services, retire GitHub Pages landing page | merged |
| [122](https://codeberg.org/doobidoo/mcp-memory-service/pulls/122) | docs(readme): link mcpmemory.services, embed landing-page video, fix dead blog link | merged |
| [123](https://codeberg.org/doobidoo/mcp-memory-service/pulls/123) | feat(retrieve): conditional temporal decay, default OFF (#119) | merged |
| [124](https://codeberg.org/doobidoo/mcp-memory-service/pulls/124) | fix(beliefs): semantic grouping + noise filter — make derive_beliefs functional (#121) | merged |
| [125](https://codeberg.org/doobidoo/mcp-memory-service/pulls/125) | docs: link AI Tinkerers Zürich talk in README and landing page | merged |
| [126](https://codeberg.org/doobidoo/mcp-memory-service/pulls/126) | fix(consolidation): restore include_embeddings=True for clustering/associations | merged |
| [127](https://codeberg.org/doobidoo/mcp-memory-service/pulls/127) | feat(bootstrap): inject beliefs + activate task_summary (#120, #121) | merged |
| [128](https://codeberg.org/doobidoo/mcp-memory-service/pulls/128) | chore: release v11.5.0 | merged |
| [129](https://codeberg.org/doobidoo/mcp-memory-service/pulls/129) | docs(readme): fix dead GitHub refs after Codeberg migration | merged |
| [130](https://codeberg.org/doobidoo/mcp-memory-service/pulls/130) | docs(claude): restructure dev guide (Non-Negotiables, Definition of Done, Codeberg-native) | merged |
| [131](https://codeberg.org/doobidoo/mcp-memory-service/pulls/131) | docs: trim README release list to v11 series; refresh CLAUDE.md to v11.5.0 | merged |
| [132](https://codeberg.org/doobidoo/mcp-memory-service/pulls/132) | docs(readme): restructure for clarity — one pitch, one quick start, logical flow | merged |
| [137](https://codeberg.org/doobidoo/mcp-memory-service/pulls/137) | fix(storage): derive multi-store migration dimension from existing vec0 DDL + crash-safe backup (#134) | merged |
| [138](https://codeberg.org/doobidoo/mcp-memory-service/pulls/138) | fix(storage): refuse hash-embedding fallback on non-empty databases (#135) | closed |
| [139](https://codeberg.org/doobidoo/mcp-memory-service/pulls/139) | fix(cli): status verifies embedding backend + dimension, adds --deep smoke test (#136) | closed |
| [140](https://codeberg.org/doobidoo/mcp-memory-service/pulls/140) | fix(storage): add delete_memory proxy to sqlite_vec DeleteMixin | merged |
| [141](https://codeberg.org/doobidoo/mcp-memory-service/pulls/141) | chore: release v11.5.1 | merged |
| [142](https://codeberg.org/doobidoo/mcp-memory-service/pulls/142) | docs+test: Mingjian Shao testimonial + hermetic ontology/harvest tests | merged |
| [144](https://codeberg.org/doobidoo/mcp-memory-service/pulls/144) | docs(changelog): log PR #140 delete_memory proxy fix under Unreleased | merged |
| [145](https://codeberg.org/doobidoo/mcp-memory-service/pulls/145) | fix(storage): refuse hash-embedding fallback on non-empty DBs + CI fix (#135, supersedes #138) | merged |
| [146](https://codeberg.org/doobidoo/mcp-memory-service/pulls/146) | fix(storage): guard against silent embedding-dimension mismatch on sqlite_vec (#143) | merged |
| [147](https://codeberg.org/doobidoo/mcp-memory-service/pulls/147) | chore: release v11.5.2 | merged |
| [148](https://codeberg.org/doobidoo/mcp-memory-service/pulls/148) | fix(milvus): restore multi-store store kwarg on 3 backend methods (#133) | merged |
| [149](https://codeberg.org/doobidoo/mcp-memory-service/pulls/149) | fix(bootstrap): deduplicate beliefs section + filter session-legacy noise (#121) | closed |
| [151](https://codeberg.org/doobidoo/mcp-memory-service/pulls/151) | fix(graph): exclude has_entity from target-orphan prune (#150) | merged |
| [152](https://codeberg.org/doobidoo/mcp-memory-service/pulls/152) | fix(beliefs): filter session-legacy noise from belief derivation (#121) | merged |
| [153](https://codeberg.org/doobidoo/mcp-memory-service/pulls/153) | docs(onboarding): rewrite Kiro CLI guide with validated real-world workflows | merged |
| [154](https://codeberg.org/doobidoo/mcp-memory-service/pulls/154) | feat(ner): PT-BR + EN DomainExtractor plugins with benchmark (#54) | merged |
| [156](https://codeberg.org/doobidoo/mcp-memory-service/pulls/156) | fix(hooks): read ~/.claude/hooks/config.json under Marketplace install (#155) | merged |
| [157](https://codeberg.org/doobidoo/mcp-memory-service/pulls/157) | chore: release v11.5.3 | merged |
| [159](https://codeberg.org/doobidoo/mcp-memory-service/pulls/159) | fix(dashboard): replace stale github.com references with Codeberg (#158) | merged |
| [160](https://codeberg.org/doobidoo/mcp-memory-service/pulls/160) | chore: release v11.5.4 | merged |
| [161](https://codeberg.org/doobidoo/mcp-memory-service/pulls/161) | fix(plugin): bump marketplace plugin version to 1.0.1 (#155) | merged |
| [164](https://codeberg.org/doobidoo/mcp-memory-service/pulls/164) | fix(docker): ship tokenizers in the standard image so the ONNX backend works | merged |
| [165](https://codeberg.org/doobidoo/mcp-memory-service/pulls/165) | chore: release v11.5.5 | merged |
| [166](https://codeberg.org/doobidoo/mcp-memory-service/pulls/166) | chore: repo housekeeping (superseded by #167) | closed |
| [167](https://codeberg.org/doobidoo/mcp-memory-service/pulls/167) | chore: untrack local developer config; tidy CLAUDE.md wording | merged |
| [168](https://codeberg.org/doobidoo/mcp-memory-service/pulls/168) | docs(claude-md): drop duplicated rules and dead context-mode block | merged |
| [169](https://codeberg.org/doobidoo/mcp-memory-service/pulls/169) | chore(ci): pin checkout action to a commit SHA; add authorship statement | merged |
| [180](https://codeberg.org/doobidoo/mcp-memory-service/pulls/180) | fix(harvest): classifier honors HARVEST_LLM_PROVIDERS, not just GROQ_API_KEY | merged |
| [181](https://codeberg.org/doobidoo/mcp-memory-service/pulls/181) | feat(quality): decouple store-time implicit weight from search reranking weight | closed |
| [182](https://codeberg.org/doobidoo/mcp-memory-service/pulls/182) | fix: canonicalize memory types so capitalized and hyphenated types resolve | merged |
| [183](https://codeberg.org/doobidoo/mcp-memory-service/pulls/183) | fix(harvest): treat a configured provider chain as a configured rewriter | merged |
| [184](https://codeberg.org/doobidoo/mcp-memory-service/pulls/184) | fix(docker): stop installing a PyTorch the standard image cannot use | merged |
| [185](https://codeberg.org/doobidoo/mcp-memory-service/pulls/185) | fix(ci): make the pre-PR gate report what it actually checked | merged |
| [186](https://codeberg.org/doobidoo/mcp-memory-service/pulls/186) | fix(hooks): send memory types the server ontology accepts | merged |
| [187](https://codeberg.org/doobidoo/mcp-memory-service/pulls/187) | fix(quality): separate the store-time blend from search reranking | merged |
| [190](https://codeberg.org/doobidoo/mcp-memory-service/pulls/190) | fix(migration): keep graph edges and beliefs when re-embedding; ship admin scripts in the image | merged |
| [191](https://codeberg.org/doobidoo/mcp-memory-service/pulls/191) | chore: release v11.6.0 | merged |
| [192](https://codeberg.org/doobidoo/mcp-memory-service/pulls/192) | docs: trim CLAUDE.md to what a session cannot derive; surface AUTHORSHIP.md | merged |
| [193](https://codeberg.org/doobidoo/mcp-memory-service/pulls/193) | fix(hooks): stop the worktree guard from minting a branch per session start | merged |
| [194](https://codeberg.org/doobidoo/mcp-memory-service/pulls/194) | test(harvest): pin the classifier provider chain; sanitize its log values | merged |
| [195](https://codeberg.org/doobidoo/mcp-memory-service/pulls/195) | ci(plugin): gate releases on the Claude Code plugin manifest version | merged |
| [196](https://codeberg.org/doobidoo/mcp-memory-service/pulls/196) | chore: release v11.6.1 | merged |
| [198](https://codeberg.org/doobidoo/mcp-memory-service/pulls/198) | fix(hooks): gate TLS verification bypass behind explicit opt-in | merged |
| [199](https://codeberg.org/doobidoo/mcp-memory-service/pulls/199) | fix(harvest): pin the rewriter provider chain; sanitize its log values | merged |
| [200](https://codeberg.org/doobidoo/mcp-memory-service/pulls/200) | fix(hooks): strip the credential and local path from the bundled hooks config (#197) | merged |
| [201](https://codeberg.org/doobidoo/mcp-memory-service/pulls/201) | fix(hooks): bring config.template.json back in sync with config.json | merged |
| [204](https://codeberg.org/doobidoo/mcp-memory-service/pulls/204) | fix(hooks): land the config.template.json sync on main (#201) | merged |
| [205](https://codeberg.org/doobidoo/mcp-memory-service/pulls/205) | ci: run the claude-hooks test suite (#202) | merged |
| [206](https://codeberg.org/doobidoo/mcp-memory-service/pulls/206) | fix(hooks): land #201 on main, add drift-gate tests, close its two review requests | closed |
| [207](https://codeberg.org/doobidoo/mcp-memory-service/pulls/207) | chore(harvest): hoist the stdlib imports rewriter.py imported inline | merged |
| [208](https://codeberg.org/doobidoo/mcp-memory-service/pulls/208) | fix(consolidate): return the merged memory's hash, not store()'s message (#112) | merged |
| [209](https://codeberg.org/doobidoo/mcp-memory-service/pulls/209) | feat(quality): configurable ONNX model dir; retire the :quality-cpu pull instructions (#171) | merged |
| [210](https://codeberg.org/doobidoo/mcp-memory-service/pulls/210) | fix(opencode): gate TLS verification bypass behind explicit opt-in | merged |
| [211](https://codeberg.org/doobidoo/mcp-memory-service/pulls/211) | fix(examples): gate the bridge's TLS bypass; run opencode tests in CI | merged |
| [212](https://codeberg.org/doobidoo/mcp-memory-service/pulls/212) | chore: release v11.7.0 | merged |
| [214](https://codeberg.org/doobidoo/mcp-memory-service/pulls/214) | fix(milvus): implement the storage methods the web API calls unguarded | merged |
| [216](https://codeberg.org/doobidoo/mcp-memory-service/pulls/216) | fix(cli): gate the lifecycle health-check's TLS bypass behind explicit opt-in | merged |
| [221](https://codeberg.org/doobidoo/mcp-memory-service/pulls/221) | fix(entities): populate the entity graph — tags dropped in maintain, dead storage.graph guards | merged |
| [222](https://codeberg.org/doobidoo/mcp-memory-service/pulls/222) | docs(retrieval): document token control and the two-phase entity tools | merged |
| [223](https://codeberg.org/doobidoo/mcp-memory-service/pulls/223) | chore(deps): sync uv.lock with the milvus extra | merged |
| [224](https://codeberg.org/doobidoo/mcp-memory-service/pulls/224) | fix(cli): make _is_https_enabled() env-var-only, matching the self-signed-certs opt-in's discipline | merged |
| [225](https://codeberg.org/doobidoo/mcp-memory-service/pulls/225) | docs(env): warn that .env MCP_HTTPS_ENABLED does not reach the CLI | merged |
| [226](https://codeberg.org/doobidoo/mcp-memory-service/pulls/226) | chore: release v11.8.0 | merged |
| [227](https://codeberg.org/doobidoo/mcp-memory-service/pulls/227) | docs: drop a personal store size from the v11.8.0 notes | merged |
| [229](https://codeberg.org/doobidoo/mcp-memory-service/pulls/229) | docs(cli): note config-import hazard in _is_https_enabled docstring | merged |
| [230](https://codeberg.org/doobidoo/mcp-memory-service/pulls/230) | fix(ci): resolve venv Python in validate_imports.sh instead of bare python3 | merged |
| [231](https://codeberg.org/doobidoo/mcp-memory-service/pulls/231) | fix(config): standardize truthy env-var parsing for MCP_HTTPS_ENABLED | merged |
| [232](https://codeberg.org/doobidoo/mcp-memory-service/pulls/232) | fix(cli): honor configured sqlite-vec embedding model | merged |
| [236](https://codeberg.org/doobidoo/mcp-memory-service/pulls/236) | Milvus storage and graph parity | open |
| [237](https://codeberg.org/doobidoo/mcp-memory-service/pulls/237) | fix(storage): use UTC midnight for timeframe-delete date boundaries | merged |
| [238](https://codeberg.org/doobidoo/mcp-memory-service/pulls/238) | fix(cli): accept on/enabled for MCP_HTTPS_ENABLED, matching safe_get_bool_env | merged |
| [239](https://codeberg.org/doobidoo/mcp-memory-service/pulls/239) | fix(oauth): avoid double slashes in OAuth discovery endpoint URLs | merged |
| [241](https://codeberg.org/doobidoo/mcp-memory-service/pulls/241) | fix(consolidation): exclude association records from meta-association pool | merged |
| [242](https://codeberg.org/doobidoo/mcp-memory-service/pulls/242) | fix(consolidation): merge real run counters into health_check statistics | merged |
| [245](https://codeberg.org/doobidoo/mcp-memory-service/pulls/245) | fix(milvus): escape backslashes in filter expressions, not just quotes | merged |
| [246](https://codeberg.org/doobidoo/mcp-memory-service/pulls/246) | chore: release v11.8.1 | merged |
| [247](https://codeberg.org/doobidoo/mcp-memory-service/pulls/247) | chore: publish repository URLs and align the PR and release directives | merged |
| [248](https://codeberg.org/doobidoo/mcp-memory-service/pulls/248) | docs: describe the GitHub mirror instead of a suspended account | merged |
| [249](https://codeberg.org/doobidoo/mcp-memory-service/pulls/249) | chore(deps): move the locked versions past the open advisories | merged |
| [251](https://codeberg.org/doobidoo/mcp-memory-service/pulls/251) | chore: point the mirror's issue links at Codeberg | merged |
| [252](https://codeberg.org/doobidoo/mcp-memory-service/pulls/252) | ci: put CodeQL back, on the GitHub mirror | merged |
| [258](https://codeberg.org/doobidoo/mcp-memory-service/pulls/258) | chore: release v11.8.2 | merged |
| [259](https://codeberg.org/doobidoo/mcp-memory-service/pulls/259) | docs: record the tag-push rule, the artifact check, and what Pages actually does | merged |
| [265](https://codeberg.org/doobidoo/mcp-memory-service/pulls/265) | fix(deps): pin pymilvus back below 3.0.0, and drop the dead triage script | merged |
| [266](https://codeberg.org/doobidoo/mcp-memory-service/pulls/266) | chore: gitignore the agent tooling other assistants drop in the checkout | merged |
| [267](https://codeberg.org/doobidoo/mcp-memory-service/pulls/267) | fix(quality): stop a comma-separated tag string becoming one entity per character | merged |
| [268](https://codeberg.org/doobidoo/mcp-memory-service/pulls/268) | chore(logging): sanitize logged values in retrieve and consolidator | merged |
| [269](https://codeberg.org/doobidoo/mcp-memory-service/pulls/269) | fix(graph): stop selecting graph nodes on edges the renderer cannot draw | merged |
| [270](https://codeberg.org/doobidoo/mcp-memory-service/pulls/270) | chore(logging): sanitize logged values in server_impl | merged |
| [271](https://codeberg.org/doobidoo/mcp-memory-service/pulls/271) | fix(logging): sanitize the three multi-line consolidator log calls CodeQL still flags | merged |
| [272](https://codeberg.org/doobidoo/mcp-memory-service/pulls/272) | feat(maintenance): Codeberg triage digest, replacing the GitHub-only one | merged |
| [273](https://codeberg.org/doobidoo/mcp-memory-service/pulls/273) | docs: credit the OAuth reporter by his full name | merged |
| [276](https://codeberg.org/doobidoo/mcp-memory-service/pulls/276) | chore(pr): run the quality gate against a local model instead of skipping it | merged |
| [277](https://codeberg.org/doobidoo/mcp-memory-service/pulls/277) | fix(deps): move mcp and the test lockfiles past their advisories | merged |
| [279](https://codeberg.org/doobidoo/mcp-memory-service/pulls/279) | fix(cli): make memory launch honour the configured TLS settings | merged |
| [284](https://codeberg.org/doobidoo/mcp-memory-service/pulls/284) | fix(compat): drop the LM Studio patch that stopped applying, and add the alarm | merged |
| [285](https://codeberg.org/doobidoo/mcp-memory-service/pulls/285) | fix(web): keep the API key out of the HTTP access log | merged |
| [286](https://codeberg.org/doobidoo/mcp-memory-service/pulls/286) | chore(release): bump version to 11.8.3 | merged |
| [287](https://codeberg.org/doobidoo/mcp-memory-service/pulls/287) | fix: keep startup diagnostics off the stdio channel, and let the CLI build hybrid | merged |
| [288](https://codeberg.org/doobidoo/mcp-memory-service/pulls/288) | chore: merge local permission grants, ignore local agent artifacts | merged |
| [289](https://codeberg.org/doobidoo/mcp-memory-service/pulls/289) | fix(hybrid): stop the sync loop from scanning D1 on every cycle | merged |
| [290](https://codeberg.org/doobidoo/mcp-memory-service/pulls/290) | fix: report the embedding model in use, and stop doubling the row count in refusal messages | merged |
| [291](https://codeberg.org/doobidoo/mcp-memory-service/pulls/291) | test: make the hybrid perf assertions mean something, and stop the host picking the embedding dimension | merged |
| [294](https://codeberg.org/doobidoo/mcp-memory-service/pulls/294) | chore(release): bump version to 11.8.4 | merged |
| [295](https://codeberg.org/doobidoo/mcp-memory-service/pulls/295) | fix(docker): move the remaining images off Python 3.10 | merged |
| [297](https://codeberg.org/doobidoo/mcp-memory-service/pulls/297) | fix(docker): make the images start without a path override | merged |
| [298](https://codeberg.org/doobidoo/mcp-memory-service/pulls/298) | test(tz): give the timezone back after a test borrows it | merged |
| [299](https://codeberg.org/doobidoo/mcp-memory-service/pulls/299) | chore: release v11.8.5 | merged |
| [300](https://codeberg.org/doobidoo/mcp-memory-service/pulls/300) | fix(deps): correct the setuptools bound that keeps milvus-lite importable | merged |
| [302](https://codeberg.org/doobidoo/mcp-memory-service/pulls/302) | docs: correct the venv Python version and note the missing dev tooling | merged |
| [303](https://codeberg.org/doobidoo/mcp-memory-service/pulls/303) | ci: run the suite once with the ml and nli extras installed | merged |
| [305](https://codeberg.org/doobidoo/mcp-memory-service/pulls/305) | deps: move to transformers 5.x, closing two high-severity advisories | merged |
| [307](https://codeberg.org/doobidoo/mcp-memory-service/pulls/307) | fix(quality): let huggingface_hub resolve its own cache | merged |
| [309](https://codeberg.org/doobidoo/mcp-memory-service/pulls/309) | test(quality): make the two ONNX tests test what they claim | merged |
| [310](https://codeberg.org/doobidoo/mcp-memory-service/pulls/310) | deps: put onnxscript in [ml] so the quality export can actually run | merged |
| [311](https://codeberg.org/doobidoo/mcp-memory-service/pulls/311) | docs: point contributors at Codeberg, keep security reports on GitHub | merged |
| [314](https://codeberg.org/doobidoo/mcp-memory-service/pulls/314) | fix: run the dead-ref check that was never wired up, and ignore a symlinked .venv | merged |
| [316](https://codeberg.org/doobidoo/mcp-memory-service/pulls/316) | fix(quality): stop a disabled quality system from loading the ONNX ranker | merged |
| [319](https://codeberg.org/doobidoo/mcp-memory-service/pulls/319) | test(quality): stop the async-scorer test from exporting a real ONNX model | merged |
| [320](https://codeberg.org/doobidoo/mcp-memory-service/pulls/320) | chore: release v11.9.0 | merged |
| [321](https://codeberg.org/doobidoo/mcp-memory-service/pulls/321) | fix(harvest): stop hardcoded 30s/60s wrapper timeouts overriding HARVEST_LLM_TIMEOUT | merged |
| [322](https://codeberg.org/doobidoo/mcp-memory-service/pulls/322) | fix(harvest): meta-filter escape hatch checks any pattern, not just convention | open |
| [323](https://codeberg.org/doobidoo/mcp-memory-service/pulls/323) | fix(hooks): settings.json merge in configure_claude_settings | merged |
| [325](https://codeberg.org/doobidoo/mcp-memory-service/pulls/325) | fix(consolidation): make every time horizon mean the window it documents | merged |
| [329](https://codeberg.org/doobidoo/mcp-memory-service/pulls/329) | fix(consolidation): make the configured clustering algorithm the one that runs | merged |
| [330](https://codeberg.org/doobidoo/mcp-memory-service/pulls/330) | fix(hooks): pair the last assistant turn with the prompt that preceded it | merged |
| [331](https://codeberg.org/doobidoo/mcp-memory-service/pulls/331) | chore: release v11.10.0 | merged |
| [332](https://codeberg.org/doobidoo/mcp-memory-service/pulls/332) | docs(site): publish the architecture and deployment diagrams on mcpmemory.services | merged |
| [333](https://codeberg.org/doobidoo/mcp-memory-service/pulls/333) | docs: give the mirror tag rule its real reason | merged |
| [334](https://codeberg.org/doobidoo/mcp-memory-service/pulls/334) | chore(deps): stop Dependabot re-proposing setuptools, refresh the lock | merged |
| [335](https://codeberg.org/doobidoo/mcp-memory-service/pulls/335) | docs(site): put both diagrams in English and say what the architecture one shows | merged |
| [336](https://codeberg.org/doobidoo/mcp-memory-service/pulls/336) | feat(site): give mcpmemory.services a favicon from the brain logo | merged |
| [337](https://codeberg.org/doobidoo/mcp-memory-service/pulls/337) | docs: trim CLAUDE.md and flatten the gitnexus skill paths | merged |
| [338](https://codeberg.org/doobidoo/mcp-memory-service/pulls/338) | fix(tests): stop TestBaseUrl from reading the real server PID file | merged |
| [339](https://codeberg.org/doobidoo/mcp-memory-service/pulls/339) | fix(consolidation): map recommendation values onto the API's public contract | open |
| [340](https://codeberg.org/doobidoo/mcp-memory-service/pulls/340) | fix(consolidation): derive DBSCAN eps from the data's own k-distance curve | open |
