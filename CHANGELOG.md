# Changelog

**Recent releases for MCP Memory Service (v10.36.4 and later)**

All notable changes to the MCP Memory Service project will be documented in this file.

> **A note on issue and PR numbers.** This project was hosted on Codeberg from June to
> September 2026. Entries from that period cite Codeberg numbers, and GitHub will
> auto-link a bare `#123` to its own issue of that number, which is a different thing.
> Roughly: numbers under #341 in v10.71.0 through v11.10.0 entries are Codeberg. The
> originals stay readable at https://codeberg.org/doobidoo/mcp-memory-service. Older
> entries predate the move and their numbers are GitHub numbers again.

**Versions v10.36.3 and earlier** – See [docs/archive/CHANGELOG-HISTORIC.md](./docs/archive/CHANGELOG-HISTORIC.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Thanks to eunwoo song for the retrieval fix below (#1128) and to timkjr for the two consolidation fixes (#1087, #1088), both carried over from pull requests opened on Codeberg before the move.

### Removed

- **HTTPClientStorage and the HTTP coordination mode (#1155, closes #1078).** `storage/http_client.py`, `utils/http_server_manager.py` and the `ServerCoordinator` auto-detection are gone. The mode had not been able to start since v7.5.0: the class lacked four abstract methods, and the detection probe asked `/health` while the server answers under `/api/health`, so every process fell back to direct SQLite anyway. It also carried no authentication and no TLS, so it could not have reached an `MCP_API_KEY`-gated server even if it had run. Multi-client access is WAL mode for several local clients and one shared HTTP server for everything else; the docs that still described the auto-detection now say so, and `docs/architecture.md` no longer claims Bearer-token support for a backend that had none.

### Fixed

- **fix(storage): apply eligibility filters before the nearest-neighbour limit (#1128, eunwoo song, closes #1077).** `recall()` and `retrieve()` asked sqlite-vec for the k nearest embeddings first and filtered afterwards, so soft-deleted rows, rows outside the requested time window, and rows excluded by tag or supersession consumed the candidate budget before a valid match could be seen. With enough excluded neighbours the result set came back short or empty even though matching memories existed. Both queries now restrict the KNN scan to eligible rowids, so k counts only memories that can actually be returned. The tests run against real sqlite-vec with 4,100 excluded neighbours crowding five valid ones, which is what distinguishes a fix here from a fix that merely reorders the same failure.
- **fix(consolidation): map recommendation values onto the API's public contract (#1087, timkjr).** `/api/consolidation/recommendations/{time_horizon}` sanitized its output against an allowlist of uppercase values, but `DreamInspiredConsolidator.get_consolidation_recommendations()` has always returned lowercase snake_case ones (`consolidation_beneficial`, `optional`, `no_action`, `error`). Nothing ever matched, so every call returned `"recommendation": "UNKNOWN"` regardless of the consolidator's actual assessment. The values are now mapped explicitly, with the CWE-209 fallback kept for anything unrecognised. Traces back to Codeberg #328, ported here from Codeberg PR #339.
- **fix(consolidation): derive DBSCAN's eps from the data's own k-distance curve (#1088, timkjr).** `eps` was computed from dataset size alone (`0.5 - n/10000`, clamped), which bakes in an assumption about how far apart an embedding model puts unrelated text. all-MiniLM-L6-v2 keeps unrelated content at a narrow, elevated baseline similarity, so density-reachability chained memories into one giant cluster — observed at 989 of 1,104 memories in a single cluster with coherence 0.461. `eps` now comes from the knee of the sorted k-distance curve (Ester et al., 1996), computed with `NearestNeighbors` rather than a full n×n distance matrix. Ported from Codeberg PR #340.

### Internal

No user-visible effect; recorded because they change how the repository is maintained.

- **A harvest regression test that was missing (#1089, timkjr).** The meta-discussion filter's escape hatch is convention-only, and nothing pinned that. Two tests now do. Behaviour is unchanged: the widening originally proposed in Codeberg PR #322 was reverted before the merge.
- **Dependabot no longer widens load-bearing version bounds (#1130).** A grouped bump rewrites the constraint in `pyproject.toml` rather than respecting it, and #1086 used that to lift `mcp` to `<3.0.0` and `pymilvus` to `<4.0.0`. mcp 2.x removes `mcp.shared.session`, which broke test collection outright; pymilvus 3.x has no CI coverage here at all (#1112), so it would have gone in silent. Major updates for both are now ignored. The `github-actions` ecosystem gained a group as well, because `codeql-action`'s `init` and `analyze` steps must move in the same commit — split across #1080 and #1081 they deadlocked, with analyze refusing to run against a config written by a different version. Both are now on v4.37.9 (#1129).

## [11.11.0] - 2026-09-05

MINOR release. Closes three critical advisories reported by alivirgo, and moves development back to GitHub.

The three advisories share a shape worth naming: in each case the guard existed and was simply not reachable from where it was needed. `local_only_tools()` was applied by one transport out of three. The SSE auth check was a closure inside another transport's function. The `client_credentials` hardening from GHSA-5p27-64mv-pr73 checked how a client authenticates but not how it came to exist. Each fix moves the guard to where every caller passes, rather than adding a second copy.

### Security

- **`memory_harvest` and `memory_ingest` were reachable over remote transports** (GHSA-7crr-2r7w-cpfm). Both take a caller-controlled filesystem path. The `local_only_tools()` filter ran only in the FastAPI JSON-RPC shim, so Streamable HTTP and SSE served them to remote callers. The filter now lives in `MemoryServer.list_tools()` and `call_tool()`, which every transport goes through. `call_tool()` refuses before the handler resolves, since hiding a tool from `tools/list` does nothing about a caller that names it directly.
- **The SSE transport had no authentication at all** (GHSA-2hh8-qjxc-43x3). `/sse` and `/messages/` both reach the full MCP tool surface and neither was gated. The check now lives at module level and both transports call it.
- **Open Dynamic Client Registration handed out read-write tokens** (GHSA-6mvm-q4j3-27qg). With `MCP_DCR_REGISTRATION_KEY` unset, a caller could register itself as a confidential client, exchange its own credentials, and hold a `read write` token the owner never granted.

### Behaviour changes

Both are deliberate, and both fail loudly rather than degrading quietly.

- **`client_credentials` is refused while Dynamic Client Registration is open.** If you use that grant, set `MCP_DCR_REGISTRATION_KEY` and register with it. `authorization_code` with PKCE is unaffected, which is the flow Claude.ai Remote MCP uses.
- **An MCP transport refuses to start on a non-loopback bind with no authentication configured.** Set `MCP_API_KEY`, enable OAuth with `MCP_OAUTH_ENABLED=true`, or bind to `127.0.0.1`. This covers Streamable HTTP as well as SSE: its `/mcp` gate is conditional on `OAUTH_ENABLED or API_KEY`, so with neither set it was equally open.

### Infrastructure

- **Development moved back to GitHub.** Issues, pull requests, CI, releases and the wiki are at github.com/doobidoo/mcp-memory-service. The four Forgejo workflows were ported to GitHub Actions and `.forgejo/` removed. Codeberg remains readable as an archive so existing links resolve.
- Issue and PR numbers below #341 in entries from June to September 2026 are Codeberg numbers. See the note at the top of this file.
- `claude-hooks` plugin manifest bumped to 1.0.5: its marketplace homepage pointed at the old forge.


## [11.10.0] - 2026-08-28

Special thanks to timkjr for two of the fixes below (#321, #323) and to alivirgo for porting a Stop-hook fix over from the GitHub mirror (#330), which cannot take pull requests directly.

MINOR release. Consolidation gets two correctness fixes this cycle, and one of them changes behavior under an existing config rather than just fixing a bug: `MCP_CLUSTERING_ALGORITHM=dbscan` without scikit-learn installed now raises `ConsolidationError` at consolidation time instead of silently falling back to `simple` clustering (#329) — see the Upgrade note below. The other closes the gap between what the five `memory_consolidate` time horizons document and what they actually did (#325). Three hook fixes round out the release: two from external contributor timkjr on hardcoded harvest timeouts and a settings-merge bug that could wipe a user's own hooks (#321, #323), and one authored by alivirgo porting a Stop-hook transcript-pairing fix from the GitHub mirror, which takes no PRs of its own (#330).

**Upgrade note:** if you have `MCP_CLUSTERING_ALGORITHM` set explicitly to `dbscan` or `hierarchical` and do not have scikit-learn installed, consolidation will now fail loudly with `ConsolidationError` instead of silently running `simple` clustering. Fix with `pip install 'mcp-memory-service[clustering]'`, or set `MCP_CLUSTERING_ALGORITHM=auto` to let the engine pick what's actually available.

### Fixed

- **fix(consolidation): make every time horizon mean the window it documents (#325, closes #324).** The `memory_consolidate` tool description promises one window per horizon — 24 hours / 7 days / 30 days / 90 days / 365 days — and four of the five did something else: daily reached 2 days instead of 1, weekly and monthly applied no age filter at all and scanned the entire store, and quarterly/yearly kept only memories OLDER than their window, which is inverted. All five now filter to the window they document. The dead `delta` config field and `filter_memories_by_age` were removed, and the batch cap moved into `_take_oldest_batch` so it applies inside the window instead of across the whole store. Known follow-up, not fixed here: no horizon reaches past 365 days any more, so controlled forgetting can't see anything older than a year (#327).
- **fix(consolidation): make the configured clustering algorithm the one that runs (#329, closes #326).** `MCP_CLUSTERING_ALGORITHM` defaults to `dbscan`, but scikit-learn — which `SemanticClusteringEngine` imports `DBSCAN` and `AgglomerativeClustering` from — was declared in no extra anywhere, so the default was unsatisfiable and clustering silently ran the `simple` fallback behind a single WARNING log line. Adds a `[clustering]` extra (also folded into `[ml]`), changes the default to `auto`, and naming `dbscan`/`hierarchical` explicitly without scikit-learn installed now raises `ConsolidationError` with an install hint instead of degrading silently. The clustering health check now reports the algorithm actually in use rather than a hardcoded string; the other five engine health checks are still hardcoded literals, tracked separately as #328.
- **fix(harvest): stop hardcoded 30s/60s wrapper timeouts overriding `HARVEST_LLM_TIMEOUT` (#321, timkjr).** `rewrite_sync()` and `rewrite_batch_sync()` wrapped the async LLM call in a `ThreadPoolExecutor` and capped `future.result()` at a hardcoded 30s/60s, completely independent of `HARVEST_LLM_TIMEOUT`, which only controlled the inner httpx client's own timeout. Raising the env var did nothing for either wrapper — a real LLM call slower than the hardcoded cap silently returned `None`, indistinguishable from a legitimate SKIP verdict. Both wrappers now derive their timeout from the configured budget, with a margin so the outer wrapper never fires before the inner client's own timeout would, and the wrapper timeout scales with provider-chain length.
- **fix(hooks): settings.json merge in `configure_claude_settings` (#323, timkjr).** Two bugs in how a plugin reinstall handled an existing `~/.claude/settings.json`: declining the permission-request hook deleted the entire `PreToolUse` key instead of just that hook's own group, wiping out any other `PreToolUse` hook a user had registered themselves (live on every unattended install, since the opt-out prompt defaults to declined when stdin isn't a TTY); and the merge check for whether to append new groups compared the combined set of an entire batch rather than each group individually, so one new group in a batch caused every group in it to be re-appended, duplicating ones already present. Both are now scoped to the individual group.
- **fix(hooks): pair the last assistant turn with the prompt that preceded it (#330, authored by alivirgo, reported by AndreLiotta).** Ports a fix opened against the GitHub mirror, which cannot take merges directly. `parseTranscript` in the auto-capture Stop hook walked the transcript backwards twice, independently, to find the last user prompt and the last assistant answer — so the hook could pair the user's *next* prompt with the *previous* answer. Tool_result-only user envelopes are now also skipped during the scan.

### Changed

- **claude-hooks: bump plugin manifest to 1.0.4.** Two of the fixes above (#323, #330) touch `claude-hooks/`, and the Claude Code Marketplace caches the plugin by this version, so a hook fix shipped without a manifest bump never reaches an already-installed user.

## [11.9.0] - 2026-08-27

MINOR release. The headline is a dependency bump that closes two high-severity transformers advisories with no fix anywhere in the 4.x line (#305) — the reason to upgrade if you install the `[ml]` or `[nli]` extras. Verifying it needed a CI job that actually installs those extras (#303), and that job promptly earned its keep: it turned up a real quality-system bug where a *disabled* quality system still built and exported a full ONNX model on its first call, found when it blew pytest's 120s timeout in CI (#316). Everything else this cycle is dependency and test-hygiene follow-through from those two changes, a docs fix, and finally wiring in a dead-reference check that had existed since #702 without ever running once (#314).

### Security

- **deps: move to transformers 5.x, closing two high-severity advisories (#305).** GHSA-29pf-2h5f-8g72 (RCE via `_attn_implementation_internal` in a crafted `config.json`, fixed in 5.3.0) and GHSA-fgcw-684q-jj6r (`trust_remote_code=False` overridden by a nested config on the LightGlue load path, fixed in 5.5.0) — neither advisory has a 4.x patch. `[ml]` and `[nli]` now both declare `transformers>=5.5.0,<6.0.0`; before this the `[ml]` extra capped at `<5.0.0` while `[nli]` carried a bare `>=4.30.0`, so an `[nli]`-only install was already resolving 5.x. Verified against the same selection the new ml-extras job (#303) runs, on both transformers 4.57.6 and 5.15.1: 2347 passed, 63 skipped, 8 deselected, 13 xfailed, 2 xpassed — identical on both. The lock moves `huggingface-hub` 0.36.0 to 1.28.0 and `safetensors` 0.4.5 to 0.8.0; torch does not move.

### Fixed

- **fix(quality): stop a disabled quality system from loading the ONNX ranker (#316, closes #315).** `evaluate_quality()` and `evaluate_quality_batch()` return a neutral 0.5 when the quality system is off, but both called `_ensure_initialized()` before checking `config.enabled`, so `MCP_MEMORY_QUALITY_ENABLED=false` still paid for a full `torch.onnx.export` of DeBERTa on the first call. Surfaced as CI noise on an unrelated docs PR (#314), which failed twice on identical content with exactly one test timing out — the twelve green runs before it were the same code getting lucky on timing, since the export path only became reachable in CI once #310 added onnxscript. Three regression tests in `tests/quality/test_disabled_skips_model_load.py` assert the loader is never invoked while the system is disabled.
- **fix(quality): let huggingface_hub resolve its own cache (#307, closes #304).** `onnx_ranker.py` built the model cache path by hand and took `snapshots[0]` from a glob with no ordering or completeness guarantee, so it ignored `HF_HOME`/`HF_HUB_CACHE` (which this project sets itself in `offline_mode.py`) and could pick an empty snapshot left behind by an interrupted download. Addressing models by repository id through `huggingface_hub` fixes both; `get_onnx_ranker_model` now resolves through the same `onnx_model_dir()` helper `ONNXRankerModel` already used instead of checking `Path.home()` directly.
- **fix(deps): correct the setuptools bound that keeps milvus-lite importable (#300).** The `[milvus]` extra's `setuptools<83; python_version >= '3.13'` bound was wrong on both counts: `pkg_resources` was removed in setuptools 82.0.0, not 81 as the comment claimed, and Python 3.12 already ships without setuptools in a fresh venv, so the marker left that case uncovered entirely. One unconditional `setuptools<82` covers both; `uv.lock` moves 82.0.1 down to 81.0.0.
- **deps: put onnxscript in the `[ml]` extra (#310).** `torch.onnx.export` routes through the dynamo exporter, which imports `onnxscript`. It was declared only in `tools/docker/Dockerfile.quality-cpu`, so a plain `pip install mcp-memory-service[ml]` could never perform the export itself and failed with `ModuleNotFoundError: No module named 'onnxscript'` partway through loading the DeBERTa ranker.

### Changed

- **ci: run the suite once with the `ml` and `nli` extras installed (#303).** `transformers` and `sentence-transformers` had never been imported in CI, so the quality ranker, the SentenceTransformer embedding path, and anything branching on a real embedding model being present rather than the hash fallback were uncovered by a green build. The job found three pre-existing failures on its first run (next entry) and is what made the transformers 5.x move above verifiable at all.
- **test(storage): stop the 1024-dim migration tests from passing by accident.** Three fixtures ran with the default `strict_dimension_check=True` and only worked because CI had no embedding model installed; with the ml extras added, a real 384-dim `all-MiniLM-L6-v2` load trips the dimension guard these tests were never meant to exercise. Set to `strict_dimension_check=False`, which is what they actually cover.
- **test(quality): make the two ONNX tests test what they claim (#309, closes #306).** One test's tokenizer call passed a query/document pair as a single tuple where the API expects two separate arguments. The other's fallback-path mock patched `onnx_ranker.get_onnx_ranker_model` through the `src.` import prefix, which resolves to a different module object than the one `ai_evaluator` actually imports, so the patch never reached the code under test. Also drops six long-`xfail`ed tests in `test_lightweight_onnx.py` that mocked an internal code path removed since #307.
- **test(quality): stop the async-scorer test from exporting a real ONNX model (#319).** `test_async_background_scoring` used the default evaluator, so with onnxscript installed the worker's first queued item triggered a synchronous, multi-minute `torch.onnx.export` inside an async worker — blocking the event loop badly enough that even `stop()`'s own 5s `wait_for` could not fire, which showed up as another 120s CI timeout. Switched to a disabled evaluator, which since #316 returns a neutral score without loading anything; every assertion in the test is unchanged.
- **docs: correct the venv Python version and note the missing dev tooling (#302).** CLAUDE.md said 3.11 in both places it mentions the venv; it's 3.12. Also documents that a `uv`-created `.venv` carries runtime dependencies only, with no `pip` and no `pytest` — the repair command is now in the same section as the trap.
- **docs: point contributors at Codeberg, keep security reports on GitHub (#311, reported by FBISiri).** CONTRIBUTING.md never mentioned Codeberg once, so a visitor landing on the read-only GitHub mirror had no way to learn where development actually happens; it now opens with that, and a new `.github/PULL_REQUEST_TEMPLATE.md` says the same in the one place a would-be contributor can't miss it. SECURITY.md's supported-version table still claimed 8.x and 7.x and now reads 11.x only, and both files explain why security reports stay on the GitHub mirror instead of moving to Codeberg too — Forgejo has no confidential-issue or security-advisory feature, so a Codeberg issue is always public.
- **fix: run the dead-ref check that was never wired up (#314, closes #313).** `scripts/ci/check_dead_refs.sh` has existed since #702 and was extended twice since, but nothing ever ran it — no Forgejo workflow, no git hook, not `pre_pr_check.sh` — despite CLAUDE.md telling contributors it catches dead refs in CI. It is now check 6.8 of the pre-PR gate, scoped to `docs/` and `README.md`. Wiring it into CI itself is tracked separately as #312.

## [11.8.5] - 2026-08-25

PATCH release. The two Docker fixes are the reason to upgrade if you run the published images — both were reproduced against `doobidoo/mcp-memory-service:slim` and `:latest` as published, not just against a local build. A timezone-boundary bug in timeframe deletion, from external contributor timkjr, rides along.

### Fixed

- **fix(docker): move the remaining images off Python 3.10 (#295).** The `:slim` image is the multi-arch one and was the only image left on an interpreter EOL in October 2026 while the standard image, CI, and local development had all moved to 3.12. Past that EOL, onnxruntime stopped publishing cp310 wheels after 1.23.2, so `:slim` was resolving six minor versions behind the 1.29.0 that everything else runs — on the embedding runtime, that is the whole point of the image. Verified by building for both `linux/amd64` and `linux/arm64` exactly as `release.yml` does: both come up on Python 3.12.14 with onnxruntime 1.29.0, tokenizers 0.23.1, and sqlite-vec v0.1.9; the arm64 image is 636MB against 656MB before; an HTTP container serves `{"status":"healthy"}`, and a store followed by a semantic search returns the memory it just wrote. `Dockerfile.glama` and three deprecated compose files moved to the same interpreter in the same change.
- **fix(docker): make the images start without a path override (#297).** Every published image set `MCP_MEMORY_SQLITE_PATH` to `/app/sqlite_db`, which is the volume mount point and therefore always a directory — a bare `docker run` exited on startup with "unable to open database file". Reproduced against `doobidoo/mcp-memory-service:slim` and `:latest` as published. The compose files papered over it with a real file path, including one of the documented `docker run` recipes, so the bug only showed up for anyone running the image the way the README's plain `docker run` example shows it. The default is now `/app/sqlite_db/memory.db`, and a directory is rejected before the connect attempt with a message that names both the problem and the fix, which is what anyone still carrying the old value in their own config will see on upgrade. The docs recipes that mounted `/app/sqlite_data` while the images declared `/app/sqlite_db` are corrected too — following them as written never persisted anything. Closes #296.
- **fix(storage): use UTC midnight for timeframe-delete date boundaries (#237, thanks timkjr).** `delete_by_timeframe` and `delete_before_date` built their SQL date-range boundaries with `datetime.combine(date, time()).timestamp()`, which interprets a naive datetime as host-local time, while `created_at` is stored as a UTC epoch everywhere else in the codebase. On any host away from UTC this silently missed memories near day boundaries. Fixed across the sqlite-vec mixin, Cloudflare, and Milvus, plus the `delete_memories` time-only path in `storage/base.py`. Verified with `base.py` reverted and the contributor's regression test in place: it fails with `deleted_count` 1 instead of 2, the memory sitting in the nine-hour gap survives.
- **test(tz): give the timezone back after a test borrows it (#298).** Follow-up to #237. The regression test set `TZ=Asia/Tokyo` and undid it with a `tzset()` call in a `finally` block, which runs before `monkeypatch` restores the environment variable, so the C library kept JST for every test that ran afterward in the same worker. Restoring in the right order now lives in a reusable `local_timezone` fixture.

## [11.8.4] - 2026-08-25

PATCH release. The headline fix stops a hybrid deployment from burning through Cloudflare's D1 free-tier read allowance; three smaller correctness fixes ride along.

### Fixed

- **fix(hybrid): stop the sync loop from scanning D1 on every cycle (#289).** Cloudflare mailed that this account regularly exceeds the D1 free-tier daily limits it starts enforcing on 1 September 2026 (5 million rows read, 100,000 rows written), and it did: the background sync called `get_stats()` twice per cycle, once as the health check and again immediately after inside the capacity check, and that statement scanned the memories table four times over. At the default 300s interval that came to roughly 19 million rows read per day against the 5 million allowance — the budget was gone after about six hours of uptime. Writes were never the problem, 64 new memories in seven days. Measured against the live database via the D1 REST API's `meta.rows_read` (17,658 rows, 9,985 of them tombstones): `SELECT 1` reads zero, the old `get_stats()` read 33,669, the new one reads 25,994, and both return identical values for all seven fields. Four changes carry the fix: a `health_probe()` that is `SELECT 1` and reads nothing, with a base-class default and a `getattr` fallback so other backends are untouched; capacity monitoring moved off the sync cadence onto its own interval, `MCP_HYBRID_CAPACITY_CHECK_INTERVAL` (default 3600s); `get_stats()` rewritten to conditional aggregates so it counts in one pass instead of one subquery per figure; and `CloudflareStorage.purge_deleted()`, because the existing tombstone purge only ever touched the primary, so D1 kept every tombstone forever and scanned them on every stats call — they had reached 57% of the table. Together the sync loop drops from roughly 19M to well under 1M rows read per day. Two things worth knowing before upgrading a hybrid deployment: the first run after this ships hard-deletes D1 tombstones already past the 30-day retention window (Vectorize entries and R2 objects were already gone at soft-delete time, so this only reclaims D1 rows, and the one-off write stays well inside the daily budget), and the new interval is documented in `.env.example`.
- **fix(health): report the embedding model actually in use (#290).** With an external embedding endpoint configured, `memory_health` and `/api/health/detailed` kept reporting `all-MiniLM-L6-v2` while a differently-sized model was doing the work — the external initialisation path set the embedding function and its dimension but never the name, so the attribute kept the constructor default. Closes #254.
- **fix(sqlite_vec): stop reporting memories plus embedding rows as a memory count (#290).** The hash-fallback refusal message summed the memories and memory_embeddings tables and presented that sum as a memory count, roughly double the true figure — exactly the kind of number that ends up quoted in an incident writeup. The two counts are now reported separately. Closes #228.
- **fix(stdio): keep startup diagnostics off the JSON-RPC channel (#287).** With LM Studio detected, the server wrote human-readable diagnostics to stdout, which on a stdio MCP server is the protocol channel itself — a strict client sees a corrupt stream before the handshake completes. The existing guard had the diagnosis right and the remedy backwards: LM Studio speaks stdio too, so restricting the output to that client protected the one client that was fine while still breaking it. Moved to stderr instead of deleted, so the warnings still reach the user. Closes #275.
- **fix(cli): give get_storage a hybrid branch (#287).** `memory status --storage-backend hybrid` raised `ValueError: Unsupported storage backend: hybrid` outright — hybrid was offered on the command line and documented as the accepted name, but the function never branched on it. Closes #233.

## [11.8.3] - 2026-08-24

PATCH release, security and correctness. Three things here are worth upgrading for, and all three share a shape: something was wrong in a way that looked fine. The MCP Python SDK shipped an advisory that is genuinely reachable in this project — HTTP transports served session requests without verifying the authenticated principal, and both affected transports are wired up here. `memory launch` never passed its TLS settings to uvicorn, so a deployment configured for HTTPS came back up on plain HTTP after every restart, with `memory info` reporting `http://` in agreement. And the dashboard's SSE connection put the API key in the query string, which uvicorn's access log then wrote to disk in cleartext, where TLS could not help it.

### Security

- **fix(deps): move the MCP SDK past its transport advisories (#277).** `mcp` 1.27.1 to 1.29.0. Three advisories applied; only one is reachable here. **CVE-2026-52869** — HTTP transports serve session requests without verifying the authenticated principal — matters, because `SseServerTransport` and `StreamableHTTPSessionManager` are both wired up in `utils/startup_orchestrator.py`. CVE-2026-52870 covers the experimental task handlers, which this project does not use, and CVE-2026-59950 is the WebSocket transport, which it does not ship at all; the bump clears both regardless. The full import surface was checked against 1.29.0 before the bump, since it crosses two minors. Also in the same change: `brace-expansion` to 5.0.9 in the two Node test harnesses, closing three DoS advisories that only ever affected the test tooling.
- **fix(cli): make `memory launch` honour the configured TLS settings (#279).** The uvicorn command line was a fixed list — host, port, log level — and never carried `--ssl-certfile`/`--ssl-keyfile`. uvicorn's CLI knows nothing about `MCP_HTTPS_ENABLED`, so the translation step simply did not exist, in either the background or the foreground path, while `scripts/server/run_http_server.py` did honour the same settings. Anyone running the CLI that the docs recommend was served plain HTTP no matter what their configuration said. `memory info` agreed, because `_is_https_enabled()` reads `os.environ` only — deliberately, since v11.7.0 — while the server's own config comes from `.env`; both halves were wrong in the same direction, so the state looked consistent. TLS that is requested but unusable is now a hard failure to start rather than a silent fallback, and the launched scheme is recorded in the PID file so later health checks probe what actually runs. Certificate auto-generation remains available only in `run_http_server.py`; see #281.
- **fix(web): keep the API key out of the HTTP access log (#285).** The dashboard authenticates its SSE connection through the query string because `EventSource` cannot send an `Authorization` header. uvicorn logs the full request line, so every SSE connection wrote the key in cleartext into `~/.local/share/mcp-memory/logs/server.log` — a file that outlives the session and is exactly what gets pasted into bug reports. TLS does not cover this: it protects the wire, not the file. Sensitive query parameter values are now redacted before the record is formatted. This is containment, not a cure — the key still travels in a URL, and the short-lived SSE ticket that would fix that properly is tracked in #282.
- **chore(logging): sanitize logged values across the consolidator, retrieve path and server_impl (#268, #270, #271).** User-provided values in f-string log calls now go through `_sanitize_log_value()`, which strips newline, carriage-return and escape characters and prevents log forging and ANSI injection. #271 covers the multi-line calls specifically: the pre-PR guard is line-based and could not see a `logger.info(` whose interpolation continued on the next line, so both the fix and the check had the same blind spot.

### Fixed

- **fix(compat): drop the LM Studio patch that had stopped applying, and add an alarm (#284).** One of the three patches wrapped `BaseSession._handle_notification`, a method the MCP SDK no longer has — and did not have at 1.27.1 either, so this predates the bump above. The `hasattr` guard meant it never raised and never logged; a third of the compatibility surface quietly went missing and stayed missing. Behaviour is unchanged, since the receive-loop patch covers the same failure, but the module now has tests asserting that at least one patch actually applies and that the removed target is still absent, so the next drift is noticed rather than rediscovered.
- **fix(graph): stop selecting graph nodes on edges the renderer cannot draw (#269).**
- **fix(quality): stop a comma-separated tag string becoming one entity per character (#267).** `metadata["tags"]` given as `"a,b,c"` rather than a list was iterated character by character during entity extraction.
- **fix(deps): pin `pymilvus` back below 3.0.0, and drop the dead triage script (#265).** A bulk dependency bump had widened the bound to `<4.0.0` by accident. 3.x has breaking API changes and no CI job exercises Milvus, so a break would have been silent.

### Changed

- **chore(pr): run the quality gate against a local model instead of skipping it (#276).** Without the Gemini CLI installed, `pre_pr_check.sh` was skipping its complexity and security checks and still reporting a pass. It now runs against a local endpoint, and four defects surfaced the moment the checks actually executed.
- **feat(maintenance): Codeberg triage digest, replacing the GitHub-only one (#272).** A maintenance script under `scripts/`, not part of the published package.
- **chore: gitignore the agent tooling other assistants drop in the checkout (#266).**
- **docs: credit the OAuth reporter by his full name (#273)** — Sergio Rodríguez Jové, who reported GHSA-5p27-64mv-pr73.
- **docs: record the tag-push rule, the artifact check, and what Pages actually does (#259).**

## [11.8.2] - 2026-08-23

PATCH release, security. GHSA-5p27-64mv-pr73, CVSS 3.1 9.1 (Critical): an unauthenticated caller could obtain a read+write bearer token and bypass the owner API key entirely. Affected only when `MCP_OAUTH_ENABLED=true` and Dynamic Client Registration is open (`MCP_DCR_REGISTRATION_KEY` unset, which is the default when OAuth is on) — `MCP_OAUTH_ENABLED` itself defaults to false, so a default install was never exposed. If you run OAuth with open DCR, upgrade now. The rest of this release is repository housekeeping left over from the GitHub-mirror reinstatement: dependency updates past their open advisories and a PyPI metadata fix.

### Security

- **fix(oauth): reject `client_credentials` from clients that never registered for it (GHSA-5p27-64mv-pr73, CVE requested).** Registering a public client via `/oauth/register` and replaying its returned credentials against `/oauth/token` with `grant_type=client_credentials` bypassed the owner API key entirely, handing an unauthenticated caller a read+write bearer token. Four defects lined up: `registration.py` issued a `client_secret` even to clients that registered with `token_endpoint_auth_method=none` (an RFC 7591 public client authenticates with PKCE, not a secret); `_handle_client_credentials_grant` never checked the client's registered `grant_types` (RFC 6749 section 4.4) or its auth method, so a public client could authenticate as if it were confidential; and `store_client()` hashed unconditionally, persisting an absent secret as the SHA-256 of the empty string — not exploitable, since the constant-time comparison rejects empty input before the hash is consulted, but indistinguishable from a real one in storage. The `authorization_code` grant was never affected: it already read the auth method off the stored client, and this fix copies that pattern. Affected v10.20.0 through v11.8.1. Operators who cannot upgrade immediately: set `MCP_DCR_REGISTRATION_KEY`, or `MCP_OAUTH_ENABLED=false`. Reported privately by Sergio Rodríguez Jové, with a working proof of concept.

### Changed

- **A public client's registration response no longer contains a `client_secret`.** This is the intended consequence of the fix above, and the one behaviour change to be aware of: register with `token_endpoint_auth_method=none` and the `client_secret` field is now absent rather than populated. That is what RFC 7591 prescribes — a public client authenticates with PKCE — and it is what the `authorization_code` flow already assumed. A client that was reading the secret out of its own registration response and using it was relying on the vulnerability.
- **chore(deps): move the locked Python and JS dependencies past their open advisories (#249).** `uv.lock`: aiohttp, cryptography, gitpython, joserfc, pydantic-settings, pyjwt, pypdf, python-multipart, starlette. `tests/bridge/package-lock.json` and `tests/integration/package-lock.json`: js-yaml.
- **chore: publish repository URLs on PyPI (#247).** `pyproject.toml` had no `[project.urls]` section at all, so the PyPI project page for the main distribution showed no Homepage, Repository, Documentation or Issues link. `pyproject-lite.toml` already carried one; both now note that they are kept in sync.
- **ci: restore CodeQL on the GitHub mirror (#252)** — the one workflow the mirror is allowed to carry, since Forgejo has no equivalent security-analysis job.
- **chore: point the mirror's issue template config at Codeberg (#251).**
- **docs: describe the GitHub mirror instead of a suspended account (#248).**

## [11.8.1] - 2026-08-22

PATCH release: eight fixes on top of v11.8.0, six of them from timkjr. Two are worth reading before you upgrade if you run this in production: OAuth issuer validation got stricter (#239) and the Docker image now actually starts HTTPS where it used to silently fall through to HTTP (#231).

### Fixed

- **fix(cli): make sqlite-vec CLI commands honor `MCP_EMBEDDING_MODEL`.** The CLI storage helper constructed `SqliteVecMemoryStorage` without the configured model, so commands such as `memory status` silently used the constructor default (`all-MiniLM-L6-v2`, 384 dimensions) even when the server correctly used a custom model. The CLI now passes `EMBEDDING_MODEL_NAME`, matching the server and storage factory paths.
- **`validate_imports.sh` resolves the venv Python instead of a bare `python3`, and fails loudly instead of silently falling back to the system interpreter** (#230, thanks timkjr). A bare `python3` can validate imports against whatever happens to be first on `PATH`, which is not necessarily this project's venv — a false pass in CI tells you nothing about whether the package actually imports in its own environment.
- **`_is_https_enabled()` accepts `on`/`enabled`** (#238, thanks timkjr), matching the truthy set `config.base.safe_get_bool_env` already accepts everywhere else. Before this, `MCP_HTTPS_ENABLED=on` was silently treated as false by the CLI while the rest of the config layer treated it as true.
- **Consolidation excludes association records from the candidate pool for meta-association inference** (#241, thanks timkjr). Without this, an association record could itself become the input to a later association-inference pass, and each run compounded on the last run's synthetic associations instead of just the real memories.
- **`health_check` reports the consolidator's real run counters** (#242, thanks timkjr). The statistics dict `health_check` read from was never populated, so it always reported empty statistics regardless of how many consolidation runs had actually happened.
- **Milvus filter expressions escape backslashes, not just double quotes** (#244, PR #245). Fifteen sites built expression strings with `value.replace('"', '\"')`, so a value ending in a backslash escaped the closing quote of its own literal and Milvus rejected the whole expression. Entity names come from memory content, so a Windows path or a regex fragment was enough to break graph queries. This also fixes tag search, where `_escape_like()` doubles backslashes for LIKE and the old escaping then handed a single one to the parser.

### Changed

- **Truthy env-var parsing for `MCP_HTTPS_ENABLED` is standardized across `config/transport.py`, `scripts/server/run_server.py`, `scripts/server/check_http_server.py`, and the repo-root `run_server.py`** (#231, thanks timkjr) — the last of those is what the Docker image actually executes. If you run the Docker image with `MCP_HTTPS_ENABLED=1`, `yes`, `on`, or `enabled`, it now starts HTTPS; before this fix, those values fell through to plain HTTP because the root `run_server.py` used a narrower truthy check than the rest of the config layer. If you were relying on that fallthrough, set `MCP_HTTPS_ENABLED=false` explicitly.
- **OAuth discovery endpoint URLs no longer contain double slashes** (#239, thanks timkjr). `OAUTH_ISSUER` is now rstripped and endpoints are built through a shared `join_url` helper, and the `iss` claim is validated exactly against the normalized issuer. If you set `MCP_OAUTH_ISSUER` with a trailing slash, tokens minted before this upgrade will fail validation until they expire (60 minutes by default) — reissue them, or drop the trailing slash from your existing tokens' issuer if you mint them yourself. Deployments without a trailing slash on `MCP_OAUTH_ISSUER` are unaffected.

## [11.8.0] - 2026-08-09

MINOR release: the knowledge-graph layer actually works now. `memory_explore` returned nothing on any store because entity extraction discarded every tag (#218) and two features were gated on a storage attribute nothing ever set (#219). Also in this cycle: the `memory` CLI's TLS handling catches up with the self-signed-cert opt-in from v11.7.0 (#216), and `.env`'s `MCP_HTTPS_ENABLED` no longer silently decides the CLI's scheme behind your back (#224) — both from timkjr, his fourth and fifth merged contribution this cycle. Milvus gets the storage methods the web API was calling unguarded (#214, reported by @ghulands). Two new docs cover a set of tools that were previously undocumented: [`docs/guides/token-efficient-retrieval.md`](docs/guides/token-efficient-retrieval.md) and wiki page 20-Token-Efficient-Retrieval.

### Added

- **docs(retrieval): document token control and the two-phase entity tools** (#215, #217, #220, #222). `memory_explore` and `memory_detail` shipped in v11.1.0 and appeared in no user-facing documentation, only an internal handoff note — a user with 11k memories filed #215 asking for a feature that is most of what they already do. The new [`docs/guides/token-efficient-retrieval.md`](docs/guides/token-efficient-retrieval.md) (also wiki page 20-Token-Efficient-Retrieval) covers bounding a search response (`limit`, `max_response_chars`, works everywhere with no setup) and the two-phase entity map separately, leading with the entity-graph prerequisite since `memory_explore` returns `count=0` for every query without one. Also documents two traps that cost real time to discover: `MCP_MAINTAIN_SCAN_LIMIT` defaults to 2000 and takes the first N in storage order, so repeated runs re-scan the same N and never reach the tail of a large store.

### Fixed

- **Entity graph was never populated** (#218, #219). Three separate faults kept the knowledge-graph layer empty, which in turn made `memory_explore` return nothing on any store. `maintain`'s batch extraction passed a memory's `metadata` into the entity extractor while the extractor looks for `metadata["tags"]` — but `tags` is a top-level attribute of `Memory` and `metadata` is empty on a normally-stored memory, so every tag entity was silently discarded. Since the extractor is pattern-based by design (`@mentions`, `#hashtags`, URLs, file paths — not semantic NER), tags are nearly the only signal on a heavily-tagged store, which is how a graph ends up with no entities at all. Separately, `memory_graph action=extract_entities` and `memory_search`'s `entity` filter were both gated on `storage.graph`, an attribute no storage class assigns, so the first errored out for every hash on every backend and the second silently never applied. Both now resolve graph storage the way the rest of the module does. Measured on three tagged memories, `maintain` goes from 4 entities found to 10.
- `memory_search`'s `entity` filter no longer returns unfiltered results when it cannot apply the filter — it reports the problem, and returns an empty result set when the entity has no linked memories (#219).
- `memory launch/stop/restart/info/health` verify TLS certificates by default; the previous unconditional bypass is now behind `MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS`, completing the work from v11.7.0 (#216, thanks timkjr). Self-signed HTTPS deployments need that variable set. The five commands also distinguish "not running" from "running, certificate unverifiable" instead of treating any failed health check as a stopped server, and stale-process cleanup no longer risks killing a client connection instead of the listener (`lsof -sTCP:LISTEN`).
- Milvus backend implements the storage methods the web API called unguarded, fixing 501s on `/api/tags` and friends (#214, reported by @ghulands).

### Changed

- **`MCP_HTTPS_ENABLED` in a `.env` file no longer affects the `memory` CLI** (#224, thanks timkjr). `launch/stop/restart/info/health` read it from the environment only, the same discipline `MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS` already followed — a stray `.env` in whatever directory happens to be the current one should not decide which scheme a security-relevant health probe uses. The server still honours `.env`, so if you configured HTTPS only there, export it for the CLI too (`export MCP_HTTPS_ENABLED=true`); otherwise the CLI probes `http://` against your HTTPS server and reads it as "not running" — `info` shows INACTIVE, `stop` finds nothing to stop, and `launch` without a PID file frees the port by killing the healthy server. Also note the CLI accepts `1`/`yes`/`true` while the server accepts only `true`; use `true` and both agree.

## [11.7.0] - 2026-08-05

MINOR release, though in substance a security release: three unconditional TLS certificate-verification bypasses — claude-hooks, the opencode plugin, and `examples/http-mcp-bridge.js` — are now gated behind an explicit `allowSelfSignedCerts` opt-in instead of running by default, and a real credential committed in the bundled hooks config is gone, with a CI gate that keeps it from coming back. Default endpoints for all three are `http://127.0.0.1:8000`, so most installs are unaffected, but if you point any of them at `https://` with a self-signed certificate you now need to set the opt-in (`MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS`, `OPENCODE_MEMORY_ALLOW_SELF_SIGNED_CERTS`, or `allowSelfSignedCerts` in the hooks config) — the old silent bypass is gone. Also in this release: a configurable ONNX model directory for containers that don't run as root, the Milvus backend's `store()` signature brought back in line with every other backend, and the consolidation merge action finally returning the hash it always should have. The Claude Code plugin manifest goes to 1.0.3 to ship the eight hook files that changed since 1.0.2.

Special thanks to timkjr for five PRs this cycle — the biggest single contribution to this release — and to @tecnobrat for the reports behind the whole quality/docker cluster (#170 and its follow-ups) plus production measurements that corrected two false claims in the Docker docs, and to @ghulands for the precise Milvus report.

### Security

- **fix(hooks): gate claude-hooks' TLS verification bypass behind explicit opt-in (#198, timkjr).** `rejectUnauthorized: false` was hardcoded and unconditional across seven call sites in five files — `session-end.js`, `memory-retrieval.js`, `topic-change.js`, `dynamic-context-updater.js`, and three in `memory-client.js` — so every HTTPS request these hooks made skipped certificate verification regardless of whether the endpoint needed it. `session-end-harvest.js` was the one file already doing this right. The bypass is now `allowSelfSignedCerts`, off by default in `config.template.json`, threaded through a shared `utilities/tls-options.js` helper instead of five copies of the same check, and it logs a warning on every use. Anyone pointed at a properly issued certificate (or an internal CA via `NODE_EXTRA_CA_CERTS`) now gets real verification; self-signed setups opt in on purpose.

- **fix(opencode): gate the opencode plugin's TLS bypass behind explicit opt-in (#210, timkjr).** Same shape as #198, in a different plugin: `process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"` disabled certificate verification for every TLS connection the opencode process made, not only plugin requests — worse than a per-request bypass, since it reached anything else running in that process — and a paired `https.Agent({ rejectUnauthorized: false })` applied unconditionally to every HTTPS request with no way back. Both are gone. `allowSelfSignedCerts` (config field, or `OPENCODE_MEMORY_ALLOW_SELF_SIGNED_CERTS`) is off by default and threaded through `httpsFetch` -> `requestJson`, matching the claude-hooks pattern and the existing correctly-gated opt-in in `session-end-harvest.js`.

- **fix(examples): gate the HTTP bridge's TLS bypass behind an explicit opt-in.** `examples/http-mcp-bridge.js` built an unconditional insecure `https.Agent` for every HTTPS request, and went further than the cases #198 and #210 fixed: alongside `rejectUnauthorized: false` it set `requestCert: false` (a server-side option that did nothing here) and `checkServerIdentity: () => undefined`, disabling hostname checking as well — redundant while `rejectUnauthorized` is false, but it would have survived someone re-enabling verification. Both extras are gone and the bypass now requires `MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true|1`, warning on every request that uses it. The agent is still constructed either way, so connection behaviour (`keepAlive: false`) is unchanged; only verification is. This is example code rather than a shipped path, but examples get copied, and this one was missed by the repo-wide TLS audit in #210.

- **fix(hooks): remove the credential and local path committed in the bundled hooks config (#197, reported by timkjr).** `claude-hooks/config.json` is a shipped package asset, not a personal file: `utilities/config-loader.js` reads it when `~/.claude/hooks/config.json` is absent, and `install_hooks.py` copies it into the user's home as the base config when no existing MCP install is detected. A real 40-character `memoryService.http.apiKey` and a maintainer-local `serverWorkingDir` had been in it since 24af496c. The installer's key generation (#531) only mints a fresh key when the value is empty or `auto-detect`, so every install that took that path adopted the committed key verbatim rather than getting its own. The two values are now `auto-detect` and `null`. The committed key was already out of use on the maintainer's side and only ever authenticated against a loopback-bound server, so nothing needed revoking — but that is not the point: the shipped default meant an unknown number of installs share one publicly readable token as their memory service credential. A new CI gate, `scripts/ci/check_hooks_config_secrets.sh`, fails the build when the bundled config carries anything that looks like a credential or an absolute local path, so this cannot come back. If you installed the hooks before this release, check `~/.claude/hooks/config.json`: an `apiKey` you did not choose yourself should be replaced with one you generate, matching `MCP_API_KEY` on your server. That matters most if your server binds anything wider than `127.0.0.1`.

### Added

- **ci: the JS test job now covers `opencode/` as well as `claude-hooks/`.** `scripts/ci/run_hooks_tests.sh` (#202) globbed `claude-hooks/tests/*.test.js` only, so #210's TLS gate under `opencode/` shipped untested — asking for a test there would have produced a file that never executed. The runner takes a list of directories, checks each one separately so emptying a suite cannot hide behind another still having tests, and prints paths rather than basenames now that two directories both hold a `tls-verification-opt-in.test.js`. `opencode/memory-plugin.js` gains an `_internal` test export mirroring the `claude-hooks` convention, and thirteen tests pin the gate — including that only a real `true` opts in (not `"true"`, `"1"`, or any other truthy value) and that it never touches the process-wide `NODE_TLS_REJECT_UNAUTHORIZED` switch #210 removed.

- **ci(hooks): gate `config.template.json` drift from `config.json` (#201, #204, timkjr).** The template lists every config key a fresh install should have; nothing enforced that it stayed in sync with the real `config.json`, and `allowSelfSignedCerts` (added in #198) was one of several keys the template hadn't caught up on. #201's original PR landed against a feature branch instead of main, so the sync sat unmerged until #204 re-targeted it at main and added the test coverage. `scripts/ci/check_hooks_config_drift.sh` now diffs the two files' key sets in CI and in `pre_pr_check.sh`, so a new hook config key without a matching template entry fails the build instead of surfacing later as a support question.

- **feat(quality): `MCP_QUALITY_ONNX_MODEL_DIR` makes the ONNX model directory explicit (#171, from #170 reported by @tecnobrat).** `ONNXRankerModel` hardcoded its cache to `Path.home() / ".cache" / "mcp_memory" / "onnx_models"`. In a container that resolved to `/root/.cache/...` only because the images run as root and set neither `USER` nor `HOME`, so mounting pre-exported models depended on an implementation accident — a non-root or read-only-rootfs deployment had to redirect `HOME` at a writable volume just to place them. The new variable replaces that parent directory (the loader still appends `/<model_name>`, so one setting covers every model) and defaults to the previous path, so nothing moves for existing installs.

### Changed

- **docs(docker): retire the `:quality-cpu` pull instructions, document the golden path (#171).** `README.md` and `tools/docker/README.md` both told users to `docker pull doobidoo/mcp-memory-service:quality-cpu`; that tag was last published as `10-quality-cpu` on 2026-05-29 and its build job lived in the GitHub workflow removed in `9c7f8b89`. It is not returning as a per-release build — the ONNX models are version-independent, so rebuilding them every patch is waste. `Dockerfile.quality-cpu` stays as a build-it-yourself option, and the docs now cover the three real paths: export once and mount with `MCP_QUALITY_ONNX_MODEL_DIR`, build the image yourself, or point `MCP_QUALITY_AI_PROVIDER=openai-compatible` at an endpoint you already run. Includes @tecnobrat's verified non-root read-only-rootfs Kubernetes recipe, in both the new form and the `HOME`-redirect form that works on current releases. Two things the old docs got wrong, both caught by @tecnobrat's production measurements: the quantization note no longer implies a size win — int8 measured 0.28-0.39 correlation against fp32 for `nvidia-quality-classifier-deberta`, so the 0.98 gate rejects it and the build keeps fp32 at roughly 736 MB — and the tags table's "~600 MB larger than `:slim`" was wrong; the last published `10-quality-cpu` was 1.26 GB compressed against 164 MB for `10-slim`.

- **chore(harvest): hoist `rewriter.py`'s inline stdlib imports (#207).** `pre_pr_check.sh` check 5 flagged three indented imports on every PR that merged main in, because a merge commit puts every merged file back in the index. `asyncio` and `concurrent.futures` move to module level — both are already resident in `sys.modules` once `mcp_memory_service.harvest` has imported, so hoisting costs nothing measured. `httpx` stays lazy: it costs about 36ms to import, nothing else on that path needs it, and `classifier.py` imports this module at top level while `harvest/__init__.py` imports the classifier eagerly, so hoisting `httpx` would add that cost to every harvest import, including ones that never call an LLM.

### Fixed

- **fix(harvest): pin the rewriter's provider chain the same way #194 pinned the classifier's (#199, timkjr).** `HarvestRewriter.__init__` stored `load_llm_providers()` unfiltered, so a credential-less legacy Groq entry — synthesized whenever `HARVEST_LLM_PROVIDERS` is unset — sat in the list, and `_call_llm`'s `if self._providers:` check treated it as usable. Every rewrite on an unconfigured chain made a doomed call to Groq's real endpoint instead of falling through correctly. The list is now filtered through `is_usable_provider` at construction, mirroring `classifier.py` exactly. The six `logger.warning` calls in the file also get the same log-injection sanitizing #194 gave the classifier — provider names and exception text (which can carry a remote response body) now go through lazy `%`-formatting with `_sanitize_log_value` instead of landing in an f-string.

- **fix(consolidate): return the merged memory's real content hash from the merge action, not `store()`'s status message (#208, closes #112).** `BaseStorage.store()` returns `(success, message)` — the second element is prose, not a hash. The merge handler unpacked it into a variable named `result_hash` and put that string in the response, so every caller's `content_hash` field read `"Memory stored successfully"` and had to re-search by content to find the memory it had just created. The correct value, `new_hash` from `generate_content_hash()`, was already computed two lines above; it's what the response now carries, and the misleading variable is renamed to `store_message` so the mistake can't recur by copy-paste. The regression test's mock had matched the bug — it returned a hash-shaped string where real backends return prose — and now returns what sqlite_vec, Cloudflare, and hybrid actually return.

- fix(milvus): restore the multi-store `store` keyword on the Milvus backend's `store()`, `count_all_memories()`, and `search_memories()` (#148, closes #133). The #57 Phase 1 partition-key change added `store` to the concrete backends and to every call site but missed these three Milvus overrides, so `memory status`/`list_memories` and store/search paths crashed with `count_all_memories() got an unexpected keyword argument 'store'` on the Milvus backend. The Milvus backend is single-collection and does not partition by store, so the argument is accepted for interface parity and ignored (matching the existing `get_all_memories`).

  The abstract declaration had drifted the other way: `MemoryStorage.store()` never got the parameter either, while all four concrete backends accepted it and four call sites passed it (`services/memory_service.py`, `server/handlers/documents.py`, `utils/document_processing.py`), so a backend written against the interface would have reproduced this bug on day one. The interface now declares it. The new backend signature-conformance test covers the abstract class as well as the implementations — checking only the implementations is why the drift survived — so the multi-store contract cannot silently diverge per backend again. Also adds `milvus` to the bug-report storage-backend dropdown. Thanks to @ghulands for the precise report.

## [11.6.1] - 2026-08-03

PATCH release: the harvest classifier now honors `HARVEST_LLM_PROVIDERS` instead of demanding `GROQ_API_KEY`, which completes #178 — the rewriter half shipped in v11.6.0 and the classifier did not. The Claude Code plugin manifest goes to 1.0.2, and that bump is what actually delivers the #177 hook fix to anyone who already installed the plugin: v11.6.0 shipped that fix without moving the manifest version, and the Marketplace cache is keyed on it. A CI gate now makes that mistake impossible to repeat. Special thanks to timkjr for the classifier fix and to @tecnobrat for the reports behind most of this.

### Fixed

- **fix(harvest): classifier honors `HARVEST_LLM_PROVIDERS`, not just `GROQ_API_KEY` (#180, timkjr; #178, reported by @tecnobrat in #170).** `HarvestClassifier._ensure_initialized` gated all LLM classification on `GROQ_API_KEY`, so a deployment pointed at LiteLLM, Ollama, or any other OpenAI-compatible endpoint via `HARVEST_LLM_PROVIDERS` skipped classification silently, with no warning. #183 fixed the same bug in `HarvestRewriter` for v11.6.0 but left the classifier alone, so the workaround — setting a placeholder `GROQ_API_KEY` — was still needed on the classifier path. The rewriter's provider loading and usability check are now module-level functions in `rewriter.py` (`load_llm_providers`, `is_usable_provider`) that the classifier reuses, so the two cannot drift apart again; the classifier tries the provider chain first through a new OpenAI-compatible call path and falls back to the legacy Groq SDK bridge only when nothing usable is configured, logging which provider resolved.
- **chore(plugin): bump the Claude Code plugin manifest to 1.0.2.** This is the release that actually ships #177 (`fix(hooks): send memory types the server ontology accepts`) to installed plugin users. That fix landed in v11.6.0 with the manifest still at 1.0.1, and the Marketplace caches the plugin under a version-keyed directory — without a version change, an existing install never re-fetches the hooks. If you installed the plugin before v11.6.1, your auto-capture was still tagging everything `observation`; updating to plugin 1.0.2 is what corrects it.
- fix(hooks): stop the worktree guard from minting a branch per session start (#193). `.claude/hooks/worktree_guard.sh` — the repo-local dev hook, not the Claude Code plugin — called `prepare` unconditionally on every `SessionStart` firing and never reclaimed anything, which had accumulated 85 orphaned `wt/*` branches and 9 abandoned worktrees in one checkout. It now reuses an already-prepared, unused worktree instead of creating another, and garbage-collects `wt/*` branches that back no worktree and are already merged into the default branch. `tests/ci/test_worktree_guard.sh` covers both the reuse path and, more importantly, what `gc` must never delete.

### Added

- **ci(plugin): gate releases on the Claude Code plugin manifest version (#195; #170, reported by @tecnobrat).** The plugin in `claude-hooks/` carries its own version and the Marketplace caches it under a version-keyed directory, so a hook fix released without bumping `claude-hooks/.claude-plugin/plugin.json` never reaches anyone who already installed the plugin. That happened in v11.6.0: the release notes said there were no `claude-hooks/` changes since the last bump, while `5d40d8e4` (`fix(hooks): send memory types the server ontology accepts`) sat inside the range. `scripts/ci/check_plugin_version.sh` now runs as a CI job on release changes and fails when `claude-hooks/` changed since the last commit that moved the manifest version. It anchors on the version line rather than on the file, because a metadata-only edit to the manifest would otherwise move the window forward and hide the hook changes before it.

### Changed

- test(harvest): pin the classifier provider chain and sanitize its log values (#194). Nothing in `tests/harvest/test_classifier.py` configured `HARVEST_LLM_PROVIDERS`, so the exact case reported in #178 — a provider chain configured, no `GROQ_API_KEY` — could regress unnoticed. Five cases now hold it: the chain makes the classifier usable without Groq credentials, an unset chain with no key still reports unavailable, the keyless Groq entry the legacy path synthesizes does not count as configured, classification really goes out through the provider endpoint, and a failing provider falls through to the next. The new provider logging also interpolated provider names and remote response bodies straight into f-strings; all logger calls in the file now use lazy `%`-style formatting with `_sanitize_log_value` on anything from outside.
- docs(claude-md): trim `CLAUDE.md` to what a session cannot derive from the repo, and surface `AUTHORSHIP.md` (#192). Content a model can read off the tree — command lists, file inventories, test counts — was replaced with the command that produces it, and two long sections moved into `.claude/directives/`. `AUTHORSHIP.md` is now linked from `README.md` and `CONTRIBUTING.md` instead of only existing.

## [11.6.0] - 2026-08-02

MINOR release: the migration script was silently dropping your knowledge graph. If you ever ran `migrate_sqlite_vec_embeddings.py` to switch embedding models, your `memory_graph` edges and derived beliefs did not survive the swap — this release fixes that and gives you a recovery path if you already hit it. Also ships locale-aware NER/NLI as YAML plugins, bundles the maintenance and migration scripts inside the Docker images, and closes out a run of quality, ontology, and harvest fixes reported by @tecnobrat. Special thanks to @tecnobrat for the detailed reports that shaped most of this release, and to filhocf for the Kiro CLI onboarding rewrite.

### Fixed

- **fix(migration): keep graph edges and beliefs when re-embedding (#189, reported by @tecnobrat).** `migrate_sqlite_vec_embeddings.py` rebuilds the database from the `memories` table only: it re-stores every memory to regenerate embeddings, then swaps the file in. Anything that doesn't ride along on the `Memory` object — `memory_graph` edges, derived beliefs — was gone afterward, which surfaces later as "No connected memories found." Both tables key on content hashes that don't change on re-embedding, so the data was recoverable the whole time; the migration now copies `memory_graph` and `beliefs` across after the memories are restored, dropping any edge whose endpoint failed to restore instead of inserting it dangling. **If you already ran an affected version and lost your graph:** the migration writes a `.backup_*` copy of your pre-migration database before it changes anything — restore `memory_graph` and `beliefs` from that file rather than re-deriving from scratch.
- fix(harvest): treat a configured provider chain as a configured rewriter (#178, reported by @tecnobrat in #170). `_get_rewriter` decided whether a rewriter existed at all by checking for `GROQ_API_KEY`, so a deployment pointed at LiteLLM, Ollama, or any other OpenAI-compatible endpoint got silently unrewritten harvests with no error. The workaround in the field was setting a placeholder `GROQ_API_KEY`; that's no longer necessary.
- fix(quality): separate the store-time blend from search reranking (#179, reported by @tecnobrat in #170). `MCP_QUALITY_BOOST_WEIGHT` controlled both quality-boosted search and how strongly `QualityScorer` blends implicit signals into the score written to the database, with opposite meanings in the two places — and at the 0.3 default, every stored score put a ceiling on fresh memories no matter how good the content was. New `MCP_QUALITY_IMPLICIT_BLEND_ENABLED` / `MCP_QUALITY_IMPLICIT_WEIGHT` settings decouple the two; both fall back to the boost values when unset, so no existing deployment's stored scores move because of this change.
- fix(ontology): canonicalize memory types instead of comparing them raw (#176). Custom type registration lowercased the base type name while validation compared the incoming string as-is, so a capitalized custom type could never resolve and was silently coerced to `observation`; a `str.isidentifier()` check also rejected hyphenated names outright. Both paths now go through `canonicalize_memory_type()`.
- fix(hooks): send memory types the server ontology accepts (#177). Auto-capture classified every captured memory as `Decision`, `Error`, `Learning`, or `Context` and then threw the classification away — none of those match the ontology's lowercase canonical names, so everything landed as `observation` regardless of what the pattern matcher decided. The hooks now send the canonical values; `scripts/maintenance/improve_memory_ontology.py` re-classifies existing history for anyone who wants it corrected.
- fix(docker): stop installing a PyTorch the standard image cannot use (#170, #172). The default build (`INSTALL_EXTRA="[sqlite]"`, what the release workflow builds) pulled in the CPU torch wheel even though nothing in that image can use it — embeddings go through ONNX, and the only torch import (the quality-model ONNX export) also needs `transformers` from the `ml` extra. torch now only installs under `FORCE_CPU_PYTORCH=true`. Shrinks the standard image from 356 MB to 166 MB.
- fix(onboarding): `scoring="composite"` to `mode="ranked"` in the onboarding docs — `memory_search` has no `scoring` parameter.
- fix(ci): make the pre-PR gate report what it actually checked. The coverage step aborted silently the moment one test failed, and the complexity/security step reported PASS even when it had skipped every check because the Gemini CLI was absent. Both now report accurately, coverage is advisory rather than blocking (matching CI), and the gate's own test selection now mirrors CI's.

### Added

- feat(i18n): locale-aware NER + NLI via YAML plugins (#54). Entity extraction and contradiction detection are now driven by per-locale YAML pattern files instead of hardcoded regex — adding a language is two YAML files and zero Python code. Ships with `en` and `pt_BR`; a single `MCP_LOCALE` env var governs every locale-aware subsystem. On a 1000-memory multilingual benchmark: entity yield up 122%, and PT-BR contradiction detection that previously only worked in English.
- feat(docker): ship the maintenance and migration scripts in the image (#188). `scripts/maintenance` and `scripts/migration` — re-embedding after a model change, tag repair, test-data cleanup — were never in the published images; the workaround was `docker cp` into a running container first. Both directories now ship in both images and run as-is from the container's working directory; the Docker deployment guide gains a section showing the re-embed and cleanup calls.

### Changed

- chore(docker): exclude nested `__pycache__` from the build context. The `.dockerignore` patterns only matched at the context root, so 34 stale bytecode directories under `src/` and `scripts/` were being copied into a slim build.
- chore(pr-gate): stop treating `print()` in CLI scripts as debug code. Check 6 flagged `print(` in every staged Python file, which failed the gate on any change to a `scripts/` file that talks to the operator over stdout. `print(` is now only flagged under `src/`, where it really is a leftover.
- docs(onboarding): rewrite the Kiro CLI guide with validated real-world workflows (#153, filhocf) — startup/pre-task/topic-exploration/post-task patterns tested across three machines over four weeks.
- docs(claude-md): drop duplicated rules and dead context-mode block.
- chore: untrack local developer config (`.claude/agents/`); tidy CLAUDE.md wording.
- chore(ci): pin the checkout action to a commit SHA; add an authorship statement.

## [11.5.5] - 2026-07-24

PATCH release: standard Docker image now ships tokenizers so the ONNX embedding backend actually loads. Special thanks to @peanutlasko and @stanthewizzard for the reports.

### Fixed

- fix(docker): add tokenizers to the `[sqlite]` extra so the ONNX backend loads (#162, #164). The published `standard`/`:latest` Docker image had onnxruntime but not tokenizers, so the ONNX embedding model could never initialize. On an empty database this degraded silently to SHA256 hash pseudo-vectors; on a database with existing data the v11.5.2 embedding-integrity guard correctly refused that silent fallback, which meant the container crash-looped at startup instead. Reported by @peanutlasko (#162); @stanthewizzard filed a duplicate (#163). `tokenizers>=0.22.2` is now a declared dependency of the `[sqlite]` extra, which also fixes a plain `pip install ".[sqlite]"`, and the Docker build's ONNX check is now a hard gate that imports both onnxruntime and tokenizers, so a missing dependency fails the build instead of shipping broken.
- fix(tests): keep ONNX offline in the test harness after tokenizers landed. With tokenizers present, `ONNXEmbeddingModel` gained an `MCP_MEMORY_ONNX_ALLOW_DOWNLOAD` guard (enabled by default) to stop it from reaching out for the ~80MB model; the test harness sets it to `0` so CI stays deterministic and offline.

## [11.5.4] - 2026-07-22

PATCH release: web dashboard GitHub references replaced with Codeberg. Special thanks to @sunnyagain.

### Fixed

- fix(dashboard): replace stale github.com references with Codeberg (#158, #159, reported by @sunnyagain). The web dashboard footer still linked to `github.com` after the Codeberg migration (missed in #74). Wiki/repo/LICENSE links now point to `codeberg.org` (LICENSE uses the Forgejo `/src/branch/main/` path form), the footer and all 7 i18n locales (en/de/es/fr/ja/ko/zh) now read "Codeberg Repository" instead of "GitHub Repository", and the GitHub octocat icon was swapped for a brand-neutral git icon. `app.py`'s server-rendered fallback page link was updated to match.

## [11.5.3] - 2026-07-22

PATCH release: Claude Code hooks config resolution under Marketplace install, graph orphan-prune fix, and belief-derivation noise filter. Special thanks to filhocf and @tecnobrat.

### Fixed

- fix(hooks): read `~/.claude/hooks/config.json` under Marketplace install (#155, reported by @tecnobrat; #156). The six core Claude Code memory hooks (`session-start`, `auto-capture-hook`, `memory-retrieval`, `session-end`, `session-end-harvest`, `topic-change`) resolved their config via `path.join(__dirname, '../config.json')`, which under a Marketplace plugin install resolves to the version-pinned plugin cache directory instead of the user-owned `~/.claude/hooks/config.json` — user edits were silently ignored and reverted on every plugin upgrade. New `claude-hooks/utilities/config-loader.js` `resolveConfigPath()` prefers the home-directory config, falling back to the bundled one.
- fix(graph): exclude `has_entity` edges from target-orphan pruning (#150, #151, @filhocf). `_prune_orphaned_graph_edges()` treats `has_entity.target_hash` as an entity *name*, not a content hash; the target-existence check was deleting valid entity links because no `memories` row ever matches an entity name. Source-orphan pruning is unaffected — `has_entity` edges are still removed when the owning memory is deleted. Includes `scripts/maintenance/backfill_has_entity.py` to restore edges dropped by the bug on existing databases.
- fix(beliefs): filter session-legacy noise from belief derivation (#121, #152, @filhocf). Completes the fix for #121 — the duplicate-belief half was already resolved by #127; this adds the missing noise filter in `belief_service.py` so session-legacy content no longer produces spurious derived beliefs.

## [11.5.2] - 2026-07-15

PATCH release: sqlite_vec `delete_memory` proxy fix, hash-embedding fallback guard, and embedding-dimension mismatch guard. Special thanks to @jonatanbellido (first-time contributor) and @nxxxsooo.

### Fixed

- fix(storage): add a `delete_memory` proxy to the sqlite_vec `DeleteMixin`, delegating to `delete()` and unpacking its `(bool, str)` result (#140, @jonatanbellido). `DreamInspiredConsolidator` calls `storage.delete_memory()` during forgetting/archival; `HybridMemoryStorage` and `MilvusMemoryStorage` already implemented it, but the sqlite_vec backend did not, so consolidation forgetting/archival raised `AttributeError` and failed silently on sqlite_vec. First-time contribution — thanks!
- fix(storage): refuse the hash-embedding fallback on non-empty databases (#135, @nxxxsooo). When no ML backend is available, `_initialize_hash_embedding_fallback` previously installed `_HashEmbeddingModel` unconditionally, writing SHA256 pseudo-vectors into the same vec0 table as real embeddings and silently poisoning semantic search on databases that already held real memories. The fallback is now allowed only on an effectively-empty database (no `memories` or `memory_embeddings` rows); on a database with data it raises an actionable `RuntimeError`. Set `MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1/true/yes` to force-allow (e.g. for testing) or `0/false/no` to force-refuse. When the hash model is installed, `embedding_backend_degraded` is set to `True` so health tooling can detect degraded mode. Fresh/CI databases (ML-free) keep working unchanged.
- fix(storage): guard against a silent embedding-dimension mismatch on the sqlite_vec backend (#143). Two gaps combined so that configuring a non-default embedding model whose dimension differed from an existing database degraded silently to MiniLM-384 and only failed later at INSERT time, reading as data corruption. Now: (1) opening an existing database whose vec0 `memory_embeddings` table is `FLOAT[N]` with an embedding model that produces a different dimension raises a clear `RuntimeError` at init naming both dimensions and the configured model (the hash fallback still adapts to the existing dimension, so this only trips fixed-dimension ONNX/SentenceTransformer models); (2) the ONNX path — which only serves `all-MiniLM-L6-v2` — now logs a prominent warning when a different model is requested instead of logging plain success, so the requested model is no longer silently ignored.

## [11.5.1] - 2026-07-15

PATCH release: multi-store migration dimension safety + embedding-backend verification in `memory status`. Special thanks to @nxxxsooo for both contributions.

### Fixed

- fix(storage): derive the multi-store vec0 migration dimension from the existing table's DDL instead of the hardcoded 384 default, and add a crash-safe durable backup with automatic recovery, so migrating a non-384 database no longer destroys all embeddings (#134, @nxxxsooo).
- fix(cli): `memory status` now verifies the embedding backend instead of claiming "healthy" after only reading storage stats (#136, @nxxxsooo). It reports the loaded embedding backend class, configured model, model dimension, and the vec0 table's declared dimension; flags a degraded hash-pseudo-embedding fallback (semantic search non-functional) and a model↔database dimension mismatch (writes will fail); and exits non-zero when either problem is present. New `--deep` flag runs a real store→search→delete round trip to prove memories can be stored and semantically retrieved.

## [11.5.0] - 2026-07-10

MINOR release: conditional temporal decay for search relevance, functional belief derivation pipeline, consolidation clustering fix, and bootstrap profile belief injection with task-aware retrieval. Special thanks to @filhocf for all four contributions.

### Added

- feat(retrieve): conditional temporal decay for search relevance (#123, @filhocf). Opt-in via `MEMORY_DECAY_WINDOW_DAYS` (default `0` = disabled, zero behavior change for existing users). When enabled, recent memories rank above older ones with equivalent similarity.
- feat(bootstrap): inject derived beliefs into the bootstrap profile and support task-aware retrieval (#120, #121 patch 4, PR #127, @filhocf). New `task_summary` and `budget_tokens` params on `get_bootstrap_profile`. Belief derivation is gated behind `MCP_BELIEFS_ENABLED` (default `false`).

### Fixed

- fix(beliefs): semantic grouping + noise filter make `derive_beliefs` functional (#121 patches 1+2, PR #124, @filhocf). Previously grouped by exact content equality, resulting in zero belief promotions ever on real data.
- fix(consolidation): restore `include_embeddings=True` at the three `_get_memories_for_horizon()` call sites in the consolidation scheduler (PR #126, @filhocf). Regression from #985 caused zero clusters and zero associations on real databases because clustering filtered out all memories when embeddings were `None`.

## [11.4.0] - 2026-07-04

MINOR release: memory merge action, pluggable domain NER extractors, mcpmemory.services landing page. Special thanks to @filhocf for both feature contributions.

### Added

- feat(consolidate): `merge` action for the `memory_consolidate` tool (#100, PR #105, @filhocf). Merges N source memories into a single memory with store-first/delete-after ordering and per-delete result checking. When not explicitly provided, `tags` default to the sorted union of all source tags and `memory_type` is inherited from the first source. The response's `ok` flag is `true` only when every source deletion succeeded (`len(failed) == 0`), so partial failures are reported correctly. 11 tests including failure paths.
- feat(ner): pluggable domain-specific entity extractors (#54, PR #107, @filhocf). New `DomainExtractor` protocol in `reasoning/entities.py`; register implementations via the `MCP_ENTITY_EXTRACTOR_MODULES` env var (`module.path:ClassName`, comma-separated). Extractors are lazily loaded once and cached, failures are isolated per extractor (built-in extraction never breaks), and domain entities are deduplicated against built-in ones. Wired into the graph handler, the quality maintain cycle, and the memory store path. See `docs/DOMAIN_EXTRACTORS.md`.
- feat(site): mcpmemory.services landing page (PR #106) — galaxy hero video rendered from 3,134 connected memories, blog tutorial, and an in-browser semantic memory demo running entirely client-side (sqlite-vec WASM + transformers.js embeddings).
- feat(demo): AI Tinkerers Zürich demo scripts (local-agent, concurrency, standalone).

## [11.3.3] - 2026-07-01

### Fixed
- fix(cli): `memory health/info/status` commands now respect `MCP_HTTPS_ENABLED` -- previously hardcoded `http://` caused silent failures when the server runs with TLS enabled. `_is_https_enabled()` reads the env var or `.env` file; `_base_url()` returns the correct scheme; `_http_get_json` skips cert verification for self-signed certs.

## [11.3.2] - 2026-06-30

### Fixed
- Declare `numpy>=1.24.0` as a core dependency — bare install via `uvx` crashed with `No module named 'numpy'` after the v11 optional-ML refactor dropped the implicit transitive numpy dep from torch (closes #98)

## [11.3.1] - 2026-06-22

PATCH release: claude-hooks noise reduction - auto-capture moved to Stop event and gated on substantive content.

### Fixed
- Auto-capture now fires once per turn at the Stop event instead of on every PostToolUse call, eliminating duplicate low-value memory entries per turn.
- Word-boundary anchors added to trigger regex patterns so substrings of ordinary words (e.g. "error" inside "errored") no longer produce false-positive captures.
- Session-end consolidation (`session-end.js`) now gates on `isSessionMeaningful()`: sessions that only match generic topic/next-step keywords but contain no real decisions, insights, or code changes are skipped, preventing floods of 0.0-quality "Session Summary" memories.

## [11.3.0] - 2026-06-21

### Added
- Interactive 3D knowledge-graph visualization in the dashboard Analytics view, inspired by the Orrery Obsidian plugin. Renders memories as a rotating 3D galaxy via 3d-force-graph (three.js), loaded as ESM with an importmap so three is shared (required for bloom).
  - Soft glowing sprite nodes, UnrealBloomPass glow, and a starfield and nebula backdrop.
  - Curved edges, hover neighbour-highlight with directional flow particles, and click-to-open the full memory detail modal.
  - Dynamic per-memory-type colours and filter pills (every type present is individually filterable; hover a pill for its node count).
  - Pause/resume auto-rotation.
  - True OS fullscreen (Fullscreen API) with in-view filter pills, a legend show/hide toggle, and a working detail modal.
  - The legacy 2D D3 SVG renderer is retained as an automatic fallback and via a 2D/3D toggle.

### Changed
- Graph-visualization node cap lifted now that WebGL handles large graphs: /analytics/graph-visualization default 100->1000, max 500->10000.
- Symmetric graph relationships are deduplicated for rendering (one curve per node pair instead of two mirrored ones).
- Storage graph-edges query binds only the source side (N placeholders instead of 2N) so large node limits stay well under SQLite bound-variable cap.

### Notes
- The dashboard now loads three and 3d-force-graph from the esm.sh CDN. Self-hosting/vendoring these assets is a possible follow-up for offline deployments.

## [11.2.0] - 2026-06-20

MINOR release: OAuth security hardening, sqlite-vec rowid collision fix, composite graph scoring, OpenCode XDG state dir fix.

### Added

- feat(oauth): OAuth security hardening (PR #91, supersedes #88, @mzu + follow-up). client_secret hashed at rest (SHA-256, constant-time verify, transparent legacy-plaintext upgrade-on-auth); request body-size caps via `BodySizeLimitMiddleware` (enforces declared Content-Length AND streamed byte count; `/oauth/*` tight cap, global cap elsewhere, document ingestion exempt); in-process per-IP auth rate limiting + concurrency caps on `/oauth/authorize|token|register`; `/authorize` XSS fix (query re-serialized via urlencode + html.escape, grant page double-escapes attribute + JS-string context). New env vars: `MCP_OAUTH_RATE_LIMIT_PER_MINUTE` (default 60), `MCP_OAUTH_RATE_LIMIT_MAX_CONCURRENT` (default 20), `MCP_HTTP_MAX_BODY_BYTES` (default 1 MiB), `MCP_OAUTH_MAX_BODY_BYTES` (default 64 KiB), and opt-in `MCP_OAUTH_TRUST_PROXY_HEADER` (default empty = key on socket peer; set e.g. to `X-Forwarded-For` when behind a trusted reverse proxy so the rate limiter keys on the real client IP instead of collapsing all clients into one bucket).
- feat(graph): opt-in composite scoring for `memory_explore` and `memory_detail` via the new `scoring="composite"` parameter (#55, @filhocf). Re-ranks chunks by semantic relevance + graph proximity (hop-distance decay) + entity centrality, adding `composite_score` and `score_components` fields. Default ranking is byte-for-byte unchanged when the parameter is omitted; falls back to relevance-only when graph data is unavailable. Batches graph proximity (one `find_connected()` per entity, no N+1). The `scoring` param is declared in both tool schemas for discoverability (PR #77).

### Fixed

- fix(storage): orphaned-embedding rowid collisions in sqlite-vec (PR #90). Hard-delete paths (`purge_deleted`, `_purge_tombstone`) removed `memories` rows without deleting their `memory_embeddings`; since `memories` has no AUTOINCREMENT, SQLite reused the freed rowid and the next store hit "UNIQUE constraint failed: memory_embeddings" on every write. store() now pre-deletes any embedding at the target rowid before INSERT (single + batch); purge paths delete embeddings for rows they hard-delete; health_check surfaces orphaned/missing embeddings + rowid collision risk (status=degraded) for sqlite and hybrid backends. Includes regression tests.
- fix(opencode): move plugin status file from `~/.config/opencode/` to `~/.local/state/opencode/` (XDG state dir), configurable via `OPENCODE_MEMORY_STATUS_FILE`

## [11.1.0] - 2026-06-18

MINOR release. Two-phase query API aggregation (#78, @filhocf) implementing RFC #56/#61: graph-aware composite scoring integrates semantic relevance, graph proximity, and entity centrality into memory retrieval. Maintenance script hardening (#76, #82). Landing page Codeberg migration (#74, #75).

### Added

- feat(graph): two-phase query API aggregation — `memory_explore` discovers related entities and builds a knowledge map; `memory_detail` hydrates chunks with graph context. New `src/mcp_memory_service/scoring/composite.py` implements composite scoring: semantic relevance + graph proximity + entity centrality. `storage/graph.py` gains `get_entities_for_memory()` resolving `has_entity` edges to entity names. Implements RFC #56/#61. Note: `related_entities` is populated only after entity extraction (`maintain`) seeds `has_entity` edges (PR #78, @filhocf).

### Fixed

- fix(maintenance): `improve_memory_ontology.py` and `find_duplicates.py --use-api` are now API-key aware. The ontology script sent no `Authorization` header and silently fetched 0 memories against an API-key-protected server; `find_duplicates.py --use-api` only read credentials from `~/.claude/hooks/config.json`. Both now fall back to the `MCP_API_KEY` environment variable (and `MCP_MEMORY_HTTP_ENDPOINT` / `MCP_HTTPS_ENABLED` / `MCP_HTTP_PORT` for the endpoint), so they work against a protected HTTP server without a hooks config (PR #76). These are developer maintenance scripts under `scripts/` — not part of the shipped package.
- fix(maintenance): harden `find_duplicates.py` `load_config()` — type-guard a malformed, non-dict `memoryService` config value (avoids `AttributeError`), and derive the fallback endpoint scheme from `MCP_HTTPS_ENABLED` instead of hardcoding `https://`, so HTTP-only deployments resolve correctly (PR #82).

### Docs

- docs(landing): repoint dead GitHub links to Codeberg, fix blog canonical URL (PR #74).
- docs(landing): swap GitHub octocat logo and label for Codeberg on landing page buttons (PR #75).

## [11.0.0] - 2026-06-13

MAJOR release. The default install is now dramatically smaller and faster: torch and transformers are optional extras, ONNX is the primary embedding backend. The tool surface is cleaner too — 28 well-named tools, no legacy aliases cluttering clients that support tool discovery.

BREAKING CHANGE: the 34 legacy tool-name aliases that were deprecated in v10.x are removed. Clients still using old names (e.g. retrieve_memory, store_memory, delete_by_tag, search_by_tag, rate_memory, and the consolidation/graph/ingest aliases) must update to the current names before upgrading. See docs/MIGRATION.md for the complete mapping.

### Added

- feat(deps): torch and transformers are now optional extras (install extras: [ml] or [full]). The default install uses ONNX Runtime for embeddings, which requires no GPU, has a much smaller footprint, and starts faster. Existing deployments that rely on sentence-transformers can add [ml] to their install command and nothing else changes (PR #49, @filhocf).

### Changed

- feat(v11)!: remove the full deprecation layer — DEPRECATED_TOOLS dict, transform_deprecated_call(), is_deprecated(), get_new_tool_name(), get_deprecated_tool_defs(), get_deprecation_message(), MCP_SHOW_LEGACY_TOOLS env var, and ToolDef.deprecated field are all gone. The 34 legacy aliases they backed (retrieve_memory, store_memory, delete_by_tag, search_by_tag, rate_memory, and all consolidation/graph/ingest shims) no longer exist as callable MCP tools. The module mcp_memory_service.compat is retained for _sanitize_log_value() which is still used by the log-injection guard throughout the codebase (PR #72, supersedes #60).
- docs: README, architecture guide, quality guide, and wiki examples migrated to the current 28-tool registry names. No prose references to legacy aliases remain (PR #71).

### Migration

Clients must rename calls before upgrading to v11. Full mapping: docs/MIGRATION.md. The canonical tool surface is the 28 tools in src/mcp_memory_service/tools/registry.py. Quick reference for the most common renames:

- retrieve_memory -> memory_search
- store_memory -> memory_store
- delete_by_tag -> memory_delete
- search_by_tag -> memory_list
- rate_memory -> memory_quality

## [10.74.1] - 2026-06-06

### Fixed

- fix(harvest): reject OpenClaw prompt preamble noise in PatternExtractor — adds `_is_openclaw_preamble()` filter that inspects the first 200 chars for gateway markers ("Sender (untrusted metadata):", "Conversation context:", "(untrusted metadata):") and rejects the input before pattern matching. Eliminates false-positive harvest candidates caused by OpenClaw prepending metadata to user prompts. Fixes the third issue from #43 (PR #46, @filhocf)

## [10.74.0] - 2026-06-05

### Changed

- refactor(dispatch): §13 — decompose monolithic if/elif tool dispatch in `server_impl.py` into a declarative `TOOL_REGISTRY` (in `tools/registry.py`) + `ROUTING_TABLE` (in `tools/routing.py`) lookup. Removes 49 inline `handle_*` wrapper methods; uses `__getattr__` lazy delegation via `_HANDLER_METHOD_MAP` with `setattr` caching for first-call performance. All 26 active tools registered including the 7 new tools from v10.73.0. 81 tests pass (+1427/-1407 across 7 files, PR #37, @filhocf)
- refactor(storage): §10 — decompose monolithic 5800-line `storage/sqlite_vec.py` into 8 focused mixins under `storage/mixins/`: `BaseMixin`, `MigrationsMixin`, `EmbeddingsMixin`, `StoreMixin`, `RetrieveMixin`, `HybridMixin`, `DeleteMixin`, `MetadataMixin`. No public API changes. 265 tests pass (+5464/-5861 across 21 files, PR #42, @filhocf)

### Fixed

- fix(harvest): OpenClaw trajectory harvest fixes — file discovery deduplication prevents double-ingestion of trajectory files; `role_filter` disabled for trajectory files to avoid silently dropping assistant turns (issues from #43, merged via PR #42, @filhocf)

## [10.73.0] - 2026-06-05

### Added

- feat(anti-hallucination): §6 belief-aware quarantine pipeline — new `consolidation/quarantine.py` (117 LOC) triggers quarantine when a memory contradicts active beliefs above CONTRADICTION_THRESHOLD. Two new MCP tools: `get_quarantined_memories` and `unquarantine_memory`. 290 LOC of new tests (RFC-1047 §6, PR #36, @filhocf)
- feat(consolidation): §3 Consolidation Engine v2 — three new MCP tools: `memory_distill` (batch LLM rewriter for undistilled memories), `get_onboarding_guide`, and `commit_session_legacy`. Adds scheduled distill job (6h interval), `_consolidation_counter` post-store trigger, and harvest pipeline v2 (RFC-1047 §3, PR #39, @filhocf)
- feat(bootstrap): §4 Bootstrap Profile + §5 Session Legacy — new `bootstrap/` package with Kiro/Claude/Generic formatters. Adds `get_bootstrap_profile` MCP tool. Confidence-weighted dedup (cosine threshold 0.70). Fixes deprecated `datetime.utcnow()` calls (RFC-1047 §4+§5, PR #40, @filhocf)

### Changed

- refactor(config): §9 — split monolithic 1327-line `config.py` into 12 domain modules (`base`, `backup`, `consolidation`, `documents`, `embedding`, `graph`, `oauth`, `ontology`, `quality`, `search`, `storage`, `transport`, `validation`) with full backward-compatible re-exports in `config/__init__.py`. 61 tests pass unchanged (PR #38, @filhocf)

## [10.72.0] - 2026-06-03

### Added

- fix(milvus): support `mode="ranked"` and `ranking_weights` in `search_memories` — the Milvus backend was missing the ranked-mode contract introduced in v10.70.0. Every `memory_search` call on Milvus raised `TypeError: search_memories() got an unexpected keyword argument 'ranking_weights'`. The override now accepts `ranking_weights`, whitelists `ranked` mode, over-fetches 3x and applies the shared `apply_ranked_rerank` multi-signal reranker (semantic + time decay + access frequency + quality), reaching parity with the base/SQLite-Vec implementation (PR #32, @henry201605)
- feat(schema-versioning): migration registry + CLI (#11) — checksum-verified migration registry, `check()` read-only safety probe, per-migration transactions, `_stamp_baseline()` for existing databases, and a `memory schema` CLI subcommand (`check`, `migrate`, `status`). Backward-compat `run_migrations_sync` wrapper preserved; `sqlite_vec.py` now routes all schema work through `runner.run_pending()` (RFC-1047 §6, PR #13, @filhocf)
- feat(beliefs): §2 Belief Store — observation-to-belief derivation pipeline — new `belief` memory subtype, `belief.py` derivation engine (confidence scoring, contradiction detection, evidence linking), `belief_service.py` orchestrator, SQL migration `012_add_belief_store.sql`, and 417-test coverage in `tests/test_belief_store.py` (RFC-1047 §2, PR #33, @filhocf)

## [10.71.0] - 2026-06-03

First release published from Codeberg (Forgejo). It bundles the post-migration body of work: the §0/§1 memory-intelligence groundwork, the OpenClaw harvest parser, the §8 handler refactor, and the full Codeberg CI/CD + release tooling that replaces the retired GitHub Actions pipeline.

### Added

- feat(harvest): §0 harvest-quality pipeline — sentence-level extraction (captures the sentence(s) around a pattern match instead of the first 500 chars), an assistant-only role filter (user messages contribute conventions only), the confidence gate raised 0.6 → 0.75, and an optional multi-provider LLM rewriter (Groq/DeepSeek/Ollama, any OpenAI-compatible endpoint) that rewrites conversational text into standalone, timeless insights. Meta/temporal/generic noise filters are locale-aware via `harvest/patterns/{en,de,pt_BR}.yaml` (no project-specific strings in core); bootstrap dedup reuses the ONNX embedding singleton (RFC-1047 §0, Codeberg PR #2, @filhocf)
- feat(ontology): §1 Observation Store — new `observation` subtypes (`user_correction`, `tool_outcome`, `preference_signal`) with `derived_from` preserved, so harvested signals classify into meaningful buckets instead of a generic catch-all (RFC-1047 §1, Codeberg PR #3, @filhocf)
- feat(harvest): OpenClaw trajectory parser plus refreshed pt_BR v2 locale patterns — harvest now ingests OpenClaw session trajectories and recognises Brazilian-Portuguese decision/bug/convention cues (Codeberg PR #20, @filhocf)

### Changed

- ci(codeberg): add Forgejo Actions pipelines for the Codeberg mirror — `.forgejo/workflows/release.yml` (test → PyPI main+lite via API token → Docker Hub: full `amd64` + slim multi-arch `amd64`/`arm64`) and `cleanup-images.yml` (version-aware Docker Hub tag retention, ported from the GHCR cleanup). Both run on a self-hosted `forgejo-runner`, resolve actions via Codeberg's mirror, and use raw `docker buildx` so the release path is independent of GitHub Actions / PyPI OIDC (Codeberg PRs #12, #16, #17)
- ci(codeberg): completed the GitHub→Codeberg CI/CD migration — removed the 21 leftover `.github/workflows/*.yml` (GitHub-only; `runs-on: ubuntu-latest` never matched the self-hosted Forgejo runner and left permanent "pending" phantom checks on PRs), scoped the Forgejo test job to a deterministic subset with report-only coverage, and archived the old workflows at git tag `archive/github-workflows-pre-codeberg` in case the project ever returns to GitHub (Codeberg PRs #14, #23, #25, #26)
- refactor: extract inline MCP handlers out of `server_impl.py` into `server/handlers/*` for the §8 structural-improvements track — smaller, testable handler modules with no behavioral change (RFC #7 §8, Codeberg PR #15, @filhocf)
- chore(agents): replace the GitHub-bound `github-release-manager` with a Forgejo-native `codeberg-release-manager` (Forgejo REST API, token from `.env`, never `gh`; Forgejo releases; PyPI/Docker publish via `.forgejo/workflows/release.yml`). The old agent is kept as a deprecated historical reference (Codeberg PRs #28, #29)
- docs: multilingual embedding model selection + re-embedding guide (Codeberg PR #22)

### Fixed

- fix(milvus): support `mode="ranked"` and the `ranking_weights` parameter in the Milvus backend. The ranked search feature (v10.70.0, #1028) extended the `MemoryStorage.search_memories` base signature and the MCP tool handler now always forwards `ranking_weights=...`, but the Milvus override was never updated — so every `memory_search` call on a Milvus backend raised `TypeError: search_memories() got an unexpected keyword argument 'ranking_weights'`. The override now accepts `ranking_weights`, whitelists `ranked` mode, over-fetches 3× and applies the shared `apply_ranked_rerank` multi-signal reranker (semantic + time decay + access frequency + quality), keeping parity with the base/sqlite-vec implementation
- fix(mistake-notes): reject empty/whitespace-only `correct_action` in `mistake_note_add` and `mistake_note_update`. JSON-schema `required` enforces presence, not non-emptiness, so a blank `correct_action` previously passed validation and stored a mistake note with an error pattern but no remediation. Both add and update now return a clear validation error instead (issue #1055, PR #1057, @Aigen-Protocol)

## [10.70.3] - 2026-05-29

### Fixed

- fix(ci): multi-arch-safe GHCR cleanup — replace `actions/delete-package-versions` with `dataaxiom/ghcr-cleanup-action` and remove the `workflow_run`-after-release trigger. The old cleanup deleted per-platform manifests (untagged children of the multi-arch index) minutes after each publish, causing `docker pull` 404s for all multi-arch tags since v10.66. This release re-publishes fresh platform manifests that now survive cleanup (issue #1044, diagnosis by @jonatanbellido) (PR #1052)

## [10.70.2] - 2026-05-29

### Fixed

- fix(security): wrap log f-strings in `storage/graph.py` with `_sanitize_log_value()` — resolves CodeQL `py/log-injection` alerts #483–#486 (`source_hash`, `target_hash`, `relationship_type`); alert #467 (`py/unused-global-variable` for `MCP_AUTO_EXTRACT_DEFAULT`) dismissed as false positive — imported via lazy cross-module import in `server/handlers/memory.py` (PR #1048)

## [10.70.1] - 2026-05-29

### Added

- feat(auto-capture): `memory_observe`, `auto_extract`, and harvest pipeline — passive background capture, intent-driven extraction, and batch harvest ingestion (RFC #1008 §3, closes #1032, PR #1047, @filhocf)

### Fixed

- fix(ci): remove GHA build cache from `publish-docker` step — cache hits caused buildx to push only the index manifest while skipping platform layer uploads to GHCR, resulting in 404 on `docker pull` for all multi-arch tags since v10.66 (issue #1044, @jonatanbellido)

## [10.70.0] - 2026-05-29

### Added

- feat(search): multi-signal ranked search mode (`mode="ranked"`) combining semantic similarity, time decay, access frequency, and quality scores via configurable weights; tag/time filtering delegated to shared post-retrieve tail with 5× oversample when filters active (RFC #1008 §2, closes #1028, PR #1046, @filhocf)

### Fixed

- fix(ci): update `docs/index.html` version badge to v10.69.0 — resolves version-drift-check CI failure
- fix(security): dismiss 9 CodeQL `py/path-injection` false positives via API — paths are user-specified by design in authenticated MCP tool calls; remove unused `import tempfile` from `handlers/documents.py`

### Changed

- docs(claude): clarify landing page update is MANDATORY on every MINOR/MAJOR release — version-drift-check CI gate enforces it

## [10.69.0] - 2026-05-28

### Added

- feat(mistake_notes): add `mistake_note_update` and `mistake_note_delete` MCP tools — update failure_count or content fields; delete by content_hash; both validate `memory_type='mistake'` before operating (closes #1035, PR #1045, @filhocf)

### Fixed

- fix(ci): disable buildx provenance+sbom attestations in publish-docker job — ghcr.io multi-arch index referenced attestation manifests instead of platform layers, causing `manifest unknown` on pull (fixes #1044)

### Changed

- chore(ci): pre_pr_check.sh now flags f-string logger calls missing `_sanitize_log_value()` (check 6.5) — catches py/log-injection locally before CodeQL runs; CLAUDE.md documents the Log Injection Guard pattern

## [10.68.0] - 2026-05-28

### Security

- fix(security): sanitize user input in all log messages and path expressions — 32 CodeQL alerts resolved (py/log-injection x26, py/path-injection x6) across compat.py, reasoning/inference.py, server_impl.py, and 7 handler files

### Added

- feat(reasoning): temporal edges with valid_from/valid_until for point-in-time graph queries (RFC #1008 §4, PR #1041, @filhocf)
- feat(reasoning): fact mutability classification — stable/volatile/ephemeral with contradiction_action() (RFC #1008 §5, PR #1042, @filhocf)
- feat(reasoning): multi-strategy retrieval with RRF fusion — concurrent semantic+tag strategies via asyncio.gather (RFC #1008 §6, PR #1043, @filhocf)

### Fixed

- fix(storage): graceful fallback when sqlite-vec DELETE crashes on corrupted blob — all three delete paths now catch vec exceptions and proceed with soft-delete (closes #1037)
- fix(reasoning): log warning when NLI backend is unimplemented — once-per-instance warning, batch-safe (PR #1036, closes #1033, @filhocf)
- fix(scripts): handle corrupted UTF-8 in regenerate_embeddings.py via text_factory fallback (PR #1039, closes #1038, @filhocf)
- fix(mistake_notes): increment existing note on semantic dedup rejection instead of returning error (PR #1040, closes #1034, @filhocf)

## [10.67.1] - 2026-05-28

### Security

- **fix(security): enforce authentication on all `/api/documents/*` routes** (GHSA-84hp-mqvj-3p8h, CVSSv3.1 9.8 CRITICAL, commit 907bac72): All 7 document endpoints (upload, batch-upload, status, history, remove, remove-by-tags, search-content) were served without any auth check, allowing unauthenticated access even when `MCP_API_KEY` or OAuth 2.1 was configured.

## [10.67.0] - 2026-05-28

### Security

- **fix(storage): sanitize query in BM25 log statement** ([CodeQL #440](https://github.com/doobidoo/mcp-memory-service/security/code-scanning/440)): User-supplied query string was interpolated directly into a `logger.debug` call in `sqlite_vec.py`'s BM25 search path. Now passes through the existing `_sanitize_log_value()` helper (strips `\n`, `\r`, ESC) to prevent log injection.

### Added

- **feat(reasoning): NLI-based contradiction detection — RFC #732 Phase 3** ([PR #1027](https://github.com/doobidoo/mcp-memory-service/pull/1027), @filhocf): Introduces `reasoning/nli.py` with a 4-stage pipeline — entity gate → embedding similarity pre-filter → heuristic NLI classifier → `contradicts` graph edge storage. `detect_contradictions_nli()` is called on every `memory_store` to check for conflicts with semantically similar memories. Kill-switch via `MCP_NLI_ENABLED` (default off); confidence threshold via `MCP_NLI_CONFIDENCE_THRESHOLD` (default 0.4). `memory_resolve` extended to accept a list of hashes for batch conflict resolution. `transformers` backend deferred to follow-up (tracked in issue #1033 — will emit a warning log when attempted).

- **fix(mcp): expose full v10 tool surface over HTTP** ([PR #1017](https://github.com/doobidoo/mcp-memory-service/pull/1017), @laanwj): `/mcp tools/list` previously advertised only 7 pre-v10 names (forked from stdio around v4, never resynced through the v10 consolidation). Now matches stdio's full v10 surface: `memory_graph`, `memory_quality`, `memory_harvest`, `memory_conflicts`, `memory_resolve`, `memory_consolidate`, `memory_ingest`, `memory_update`, `memory_stats`, `memory_store_session`, `mistake_note_add`, `mistake_note_search` are now reachable over HTTP. Pre-v10 names remain callable via the deprecation compat layer but are no longer advertised. `serverInfo.version` now reports the running package version instead of the stale `4.1.1` literal. Write-scope enforcement derived dynamically from `readOnlyHint` annotations — new tools automatically get correct scope gating without manual list maintenance. `memory_harvest` and `memory_ingest` blocked over HTTP (filesystem-path tools, stdio-only for security). Also subsumes the per-tool `recall_memory` time-expression fix from #1029 — HTTP transport now inherits stdio behavior by routing through the shared dispatcher.

## [10.66.1] - 2026-05-27

### Changed

- **docs(contributing): tighten contributor guidelines + add agent disclosure policy**: Added `## Security-Sensitive Changes` section (protected paths, required scope-enforcement checklist for new MCP tools) and `## Autonomous Agents & AI-Generated PRs` section (mandatory disclosure, 7-day clarification window). PR template extended with `## Security` and `## Agent Disclosure` sections. CODEOWNERS updated to require `@doobidoo` approval on `web/api/mcp.py`, `web/oauth/`, and `.github/workflows/`.

- **chore(governance): filhocf collaborator access + CODEOWNERS expanded to reasoning/**: Added filhocf as collaborator; CODEOWNERS updated to include `src/mcp_memory_service/reasoning/` under filhocf review.

- **chore(deps): bump actions/checkout 4.3.1→6.0.2, snok/container-retention-policy 2.2.1→3.0.1, actions/delete-package-versions 4.1.1→5.0.0, uv group (8 packages)**: Routine dependency maintenance.

### Fixed

- **fix(storage): align `HttpClientStorage.retrieve` signature with `BaseStorage`** ([CodeQL #428](https://github.com/doobidoo/mcp-memory-service/security/code-scanning/428)): `http_client.py` override was missing `min_confidence`, `start_time`, and `end_time` parameters present in the abstract base. All three are now forwarded to the HTTP API payload when set (`py/inheritance/signature-mismatch`).

- **fix(harvest): multi-CLI session directory resolution** ([PR #1025](https://github.com/doobidoo/mcp-memory-service/pull/1025), @filhocf): 3-level fallback: `MCP_HARVEST_SESSION_DIR` env var → `~/.claude/projects/{cwd}` → `~/.kiro/sessions/cli/`. Adds Kiro CLI support alongside existing Claude CLI support.

- **fix(harvest): expanduser() on MCP_HARVEST_SESSION_DIR**: Paths specified as `~/my-sessions` were not expanded to the full home directory path, causing session directory resolution to fail silently.

- **fix(mcp): expose full v10 tool surface over HTTP** — `/mcp tools/list` previously advertised only 7 pre-v10 names (`store_memory`, `retrieve_memory`, `recall_memory`, `search_by_tag`, `delete_memory`, `check_database_health`, `list_memories`), forked from stdio around v4 and never resynced through the v10 consolidation. It now matches stdio's v10 surface: `memory_graph`, `memory_quality`, `memory_harvest`, `memory_conflicts`, `memory_resolve`, `memory_consolidate`, `memory_ingest`, `memory_update`, `memory_stats`, `memory_store_session`, `mistake_note_add`, and `mistake_note_search` are now reachable over HTTP. Pre-v10 names are no longer advertised but remain callable via the deprecation compat layer. `serverInfo.version` now reports the running package version instead of the stale `4.1.1` literal.

- **fix(opencode): don't use https for http access**: Fix connection failure while using local `http` endpoint with the opencode plugin.

### CI

- **fix(ci): restore workflow YAML corrupted by SHA-pinning regex**: The SHA-pinning script (PR #1005) matched too broadly, splitting `with:` into `wit\nh:` and collapsing step separators across 18 workflow files (62 occurrences). Restored pre-pinning YAML structure and re-applied all 20 SHA pin mappings correctly.

- **chore(ci): pin all GitHub Actions to full SHA hashes** ([#1005](https://github.com/doobidoo/mcp-memory-service/pull/1005)): Supply chain hardening against TeamPCP-style tag-mutation attacks. All 20 workflow files now use full 40-character commit SHA hashes instead of floating version tags (`@v1`, `@v4`, `@v6`, `@v7` etc.) — 109 `uses:` entries across 18 unique action refs. Highest-risk refs: `gaurav-nelson/github-action-markdown-link-check` (community maintainer), `snok/container-retention-policy` (community), and `anthropics/claude-code-action` (runs with `CLAUDE_CODE_OAUTH_TOKEN`). Human-readable `# vtag` comments preserved. `peter-evans/create-pull-request` was already SHA-pinned.

## [10.66.0] - 2026-05-26

### Added

- **feat(reasoning): Phase 1a — transitive closure + abductive inference** ([#1009](https://github.com/doobidoo/mcp-memory-service/pull/1009), RFC [#732](https://github.com/doobidoo/mcp-memory-service/issues/732)): Implements the first phase of the reasoning engine. Transitive closure walks multi-hop graph edges to surface indirect memory connections. Abductive inference proposes the most plausible explanations for observed memory patterns using NLI-backed scoring. Lays the foundation for higher-level reasoning layers in subsequent phases.

- **feat(reasoning): Phase 1b — entity grouping** ([#1010](https://github.com/doobidoo/mcp-memory-service/pull/1010)): Groups co-referenced entities across memories into canonical clusters, enabling cross-memory entity resolution. Entity groups feed into the transitive closure engine so that edges between aliased entities are correctly traversed.

- **feat(reasoning): Phase 3 — insight cards** ([#1011](https://github.com/doobidoo/mcp-memory-service/pull/1011)): Introduces insight cards — structured summaries synthesised from related memory clusters and inferred relationships. Cards surface via the REST API and memory_graph tool, giving clients a compact, human-readable view of what the reasoning engine has discovered.

### Fixed

- **fix(time-filter): enforce time filter at SQL level** ([#1014](https://github.com/doobidoo/mcp-memory-service/pull/1014)): Time-range filters (`start_date`, `end_date`, `stale_days`) were previously applied in Python after retrieval, allowing the vector search to return results outside the requested window when the result set was small. Filters are now pushed down into the SQL `WHERE` clause so the database enforces the constraint before any vector scoring occurs.

## [10.65.3] - 2026-05-25

### Security

- **fix(security): enforce write scope on MCP tools/call — [GHSA-2r68-g678-7qr3](https://github.com/doobidoo/mcp-memory-service/security/advisories/GHSA-2r68-g678-7qr3)** ([#1004](https://github.com/doobidoo/mcp-memory-service/pull/1004)): OAuth read-only clients could invoke mutating MCP tools (`store_memory`, `delete_memory`, and related write operations) via the `/mcp/tools/call` endpoint despite holding only a read-scope token. Fixed by adding a `_WRITE_TOOLS` set and checking OAuth scope before dispatch; unauthorized calls now return JSON-RPC error `-32003` and HTTP 403. CVSS 8.1, CWE-862 (Missing Authorization). 4 regression tests added.

### CI

- **ci: restrict quality-cpu Docker build to linux/amd64 only** ([#1003](https://github.com/doobidoo/mcp-memory-service/pull/1003), closes [#1002](https://github.com/doobidoo/mcp-memory-service/issues/1002)): The `quality-cpu` Docker image build job was timing out at the 6-hour GitHub Actions limit on every release since v10.64.0 due to QEMU-emulated arm64 cross-compilation. Platforms restricted to `linux/amd64`; arm64 users should use the `:slim` or `:latest` images which are multi-arch.

## [10.65.1] - 2026-05-24

### Fixed

- **fix(prompts): guard `learning_session` against unresolved CLI `$N` placeholders** ([#1000](https://github.com/doobidoo/mcp-memory-service/pull/1000)): Adds regex-based detection of unresolved CLI positional placeholders (`$1`, `$2`, etc.) in `_prompt_learning_session` to prevent storing them as real memories when the prompt template is invoked without required arguments. Closes [#998](https://github.com/doobidoo/mcp-memory-service/issues/998).

### Changed

- **docs: make audit log plugin privacy-safe by default** ([#999](https://github.com/doobidoo/mcp-memory-service/pull/999)): Example audit-log plugin now defaults to `MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE=safe`, which strips or hashes sensitive fields. Raw mode (`MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE=raw`) is available for debugging. Optional HMAC key (`MCP_PLUGIN_AUDIT_LOG_HMAC_KEY`) enables deterministic pseudonymisation of user identifiers.

## [10.65.0] - 2026-05-24

### Added

- **feat(opencode): `/memory` slash command, TUI toasts, status file bridge** ([#997](https://github.com/doobidoo/mcp-memory-service/pull/997)): Adds three slash commands — `/memory` (current session status), `/memory search <query>` (top 5 semantic matches), `/memory health` (backend type, status, total memory count). Implemented via `command.execute.before` hook that fetches data from the memory service and replaces the user message with a formatted block plus a "reply verbatim" instruction. TUI toasts now fire for memory load / auto-capture / session-summary events — the previous blocker was a missing `variant` field on `/tui/show-toast` that returned 400 silently. Status file at `~/.config/opencode/.memory-status.json` keeps a live snapshot for the slash command and a future TUI sidebar consumer. Session-summary upsert on `session.idle` now deletes the previous per-session summary via `DELETE /api/memories/<hash>` before storing the new one, eliminating duplicate session-summary memories in the vector DB. `/api/health` no longer exposes backend metadata after [GHSA-73hc-m4hx-79pj](https://github.com/doobidoo/mcp-memory-service/security/advisories/GHSA-73hc-m4hx-79pj), so `/memory health` now hits `/api/health/detailed`. Ships a working Solid TUI sidebar widget (`opencode/memory-status-tui.tsx` + `opencode/build-tui-plugin.mjs` compiled via `babel-preset-solid`) that renders a "Memory" panel in OpenCode's sidebar. TUI plugins live in `~/.config/opencode/tui.json` (separate from server plugins in `opencode.json`) — see `opencode/README.md` for the install + config steps. Multi-project status corruption fix (per Gemini code review).

## [10.64.2] - 2026-05-23

### Fixed

- **fix(opencode): replace dead chat.message hook with event-based message.part.updated** ([#995](https://github.com/doobidoo/mcp-memory-service/pull/995)): OpenCode's plugin system never triggers `chat.message` — the hook type exists in `@opencode-ai/plugin` but there is no `trigger()` call site in the OpenCode server source. Replaced with `event` hook listening to `message.part.updated` bus events which carry full text content. Also fixes TLS: uses `node:https.Agent` with `rejectUnauthorized: false` because Bun's `fetch()` ignores `NODE_TLS_REJECT_UNAUTHORIZED` at runtime against self-signed certs. Dual export format (V1 `export default {id, server}` + legacy `export const`) for maximum loader compatibility.

### Added

- **feat(harvest): LLM-based pattern discovery script for locale plugins** ([#992](https://github.com/doobidoo/mcp-memory-service/pull/992), closes [#909](https://github.com/doobidoo/mcp-memory-service/issues/909)): New `scripts/maintenance/discover_harvest_patterns.py` — a one-shot CLI tool that analyzes low-yield harvest sessions (<3 matches from ≥50 messages) and proposes new regex patterns via an optional LLM (Groq or OpenAI-compatible API). Outputs candidate patterns as YAML to `patterns/auto_generated/{locale}.yaml`, matching the existing locale plugin schema. Includes regex validation and plain-text rejection. See `scripts/maintenance/README.md` for usage.

### Changed

- **feat(opencode): port Claude Code hooks — auto-capture, session-end write-back, harvest** — Ported Claude Code hook capabilities into the OpenCode in-process plugin. Memory retrieval on session start, auto-capture via pattern detection (#skip/#remember overrides), session-end analysis + summary storage, and optional harvest (POST /api/harvest). Hook architecture: `event` (bus events) + `experimental.chat.system.transform` (memory injection) + `experimental.session.compacting` (context preservation). Config sections: `autoCapture`, `sessionEnd`, `harvest`. (Follow-up in [#995](https://github.com/doobidoo/mcp-memory-service/pull/995) replaced `chat.message` with `message.part.updated` bus event.)

### Maintenance

- **chore(deps): upgrade transitive dependencies** — `uv sync --upgrade` updated attrs, propcache, yarl, starlette, scikit-learn, scipy, zeroconf, and others to latest compatible versions.
- **chore(docs): archive stale design plans** — Moved 7 design documents from `docs/plans/` to `docs/archive/` (Jan–Apr 2026). No content changes.
- **chore(git): install and configure Git LFS** — 29 video assets track via Git LFS. Run `git lfs pull` after cloning to retrieve binary content.
- **chore(git): prune 24 stale local and 5 remote branches** — Cleaned up branches from merged PRs that were squash-merged and not deleted.

## [10.64.1] - 2026-05-23

### Fixed

- **fix(consolidation): raise association confidence threshold to 0.5** ([#991](https://github.com/doobidoo/mcp-memory-service/pull/991)): Increases the minimum confidence score for association discovery from the previous default to 0.5, reducing false-positive graph edges during incremental consolidation runs.
- **fix(consolidation): advance last_run_at on incremental timeout** ([#989](https://github.com/doobidoo/mcp-memory-service/pull/989), closes [#986](https://github.com/doobidoo/mcp-memory-service/issues/986)): When the 10-second timeout fires during an incremental consolidation, `last_run_at` is now still advanced so the next run picks up where this one left off instead of re-processing the same window. Fixes a user-facing regression introduced in v10.64.0.
- **fix(oauth): remove offline_access from PRM scopes_supported per SEP-2207** ([#990](https://github.com/doobidoo/mcp-memory-service/pull/990)): Removes `offline_access` from the `scopes_supported` field in OAuth provider metadata (PRM endpoint) to comply with SEP-2207, which mandates that refresh tokens require explicit user authorization rather than being advertised as a default capability.
- **fix(consolidation): tighten temporal_proximity to 7-day window** ([#988](https://github.com/doobidoo/mcp-memory-service/pull/988)): Reduces the temporal proximity window for association discovery from 14 to 7 days, making the consolidation more conservative about linking temporally-distant memories.

### Maintenance

- **chore: gitignore graphify-out and scheduled_tasks.lock** (commit dcbfc4ee): Adds generated graph output directory and scheduled task lock file to `.gitignore`.

## [10.64.0] - 2026-05-22

### Added

- **feat(consolidation): incremental time horizon for `memory_consolidate`** ([#985](https://github.com/doobidoo/mcp-memory-service/pull/985), closes [#983](https://github.com/doobidoo/mcp-memory-service/issues/983), @filhocf): New `time_horizon="incremental"` mode processes only memories created since the last consolidation run, enabling safe invocation from session Stop hooks with bounded latency. Uses a DB-based atomic lock (`BEGIN IMMEDIATE` + `locked` column) to prevent concurrent runs across multiple processes. Enforces a 10-second timeout via `asyncio.wait_for` at the handler level. Skips decay and forgetting phases (those remain on monthly/yearly horizons); runs clustering, compression, and association discovery on the incremental window. Bootstraps with a 24-hour window on first run, then advances a `created_at > last_run_at` cursor. New `consolidation/run_tracker.py` module tracks run state. 15 new unit tests added. Relates to RFC [#732](https://github.com/doobidoo/mcp-memory-service/issues/732).
- **docs(research): contradiction resolution approaches reference** ([#984](https://github.com/doobidoo/mcp-memory-service/pull/984), @rudi193-cmd / Sean Campbell): New `docs/research/contradiction-resolution-approaches.md` — a system-neutral survey of invalidation models, detection mechanisms, and a decision guide for implementors, written in support of RFC #732.

### Fixed

- **fix(web): repair `GET /api/quality/trends` AttributeError** ([#982](https://github.com/doobidoo/mcp-memory-service/pull/982), closes [#981](https://github.com/doobidoo/mcp-memory-service/issues/981), reported by @TonbiLX): The endpoint raised a 500 on every storage backend due to two stacked bugs: `recall_by_timeframe` is a server-tool handler, not a storage method, and `search_all_memories` has never existed on the storage interface. Fixed by replacing both with `get_memories_by_time_range` using a DB-side BETWEEN filter. Quality trends endpoint now returns correct data.

### Maintenance

- **chore: migrate binaries to Git LFS + remove generated statistics** ([e08e606c](https://github.com/doobidoo/mcp-memory-service/commit/e08e606c)): Binary assets moved to Git LFS; generated statistics files removed from version control.
- **chore: remove stale archive directories** ([14c742ff](https://github.com/doobidoo/mcp-memory-service/commit/14c742ff)): Deleted obsolete archive directories that accumulated in the repository.
- **fix(docs): remove dead links to deleted archive files** ([bde77499](https://github.com/doobidoo/mcp-memory-service/commit/bde77499)): Documentation links pointing to the removed archive files cleaned up.

## [10.63.0] - 2026-05-20

### Added

- **feat(milvus): low-priority optional overrides — completes Issue #888** ([#978](https://github.com/doobidoo/mcp-memory-service/pull/978), closes [#888](https://github.com/doobidoo/mcp-memory-service/issues/888), @henry201605): Implements the final 4 native Milvus overrides to fully complete Issue #888. `search_by_tag_chronological` pushes tag filter + `sort_desc_key=created_at` to Milvus via `_query_memories` (replaces base-class fetch-all-then-sort fallback, supports pagination). `count_memories_by_tag` uses Milvus `count(*)` query with tag filter (replaces fetch-all-then-len fallback). `is_deleted` checks `metadata.deleted_at` field (returns False if memory not found). `purge_deleted` queries memories with `created_at <= cutoff`, filters those with `deleted_at` in metadata, and hard-deletes tombstones. 17 new mock-based unit tests across 4 test classes.

### Fixed

- **fix(harvest): support Kiro CLI `AssistantMessage` kind + raise system-content threshold to 10k** ([#979](https://github.com/doobidoo/mcp-memory-service/pull/979), closes [#972](https://github.com/doobidoo/mcp-memory-service/issues/972), @filhocf): Adds `"AssistantMessage": "assistant"` to `KIRO_KIND_MAP` so Kiro CLI assistant messages (which use this kind, not `Response`) are correctly parsed. Removes the redundant `len(text) > 2000` filter in `_is_system_content` (already capped at `MAX_CANDIDATE_CONTENT_LENGTH = 500` in extractor). Result: parse yield increases from 1 candidate per 71 messages to 36 candidates per 373 messages.

## [10.62.0] - 2026-05-20

### Added

- **feat(milvus): native `search_memories`, `retrieve_with_quality_boost`, `recall_memory`** ([#970](https://github.com/doobidoo/mcp-memory-service/pull/970), part of [#888](https://github.com/doobidoo/mcp-memory-service/issues/888), @henry201605): Completes the medium-priority Milvus native method set. All three methods push filters to the Milvus server side — `search_memories` applies semantic similarity + tag/type/date filters in one ANN call, `retrieve_with_quality_boost` re-ranks by blending embedding similarity with stored quality scores, and `recall_memory` adds temporal recency weighting. Eliminates N round-trip fallbacks to the base class. 18 new mock-based unit tests in `tests/storage/test_milvus_search_methods.py`.

### Fixed

- **fix(hooks): parse Claude Code transcripts as JSONL with nested message** ([#971](https://github.com/doobidoo/mcp-memory-service/pull/971)): Claude Code transcript format changed from bare JSON objects to JSONL with a `message` wrapper. The auto-capture hook now parses each line as `JSON.parse(line).message ?? line` before extracting tool calls, restoring transcript-based memory capture for current Claude Code versions.

### Dependencies

- **chore(deps): bump actions/github-script from 7 to 9** ([#973](https://github.com/doobidoo/mcp-memory-service/pull/973)): GitHub Actions dependency update.
- **chore(deps): bump snok/container-retention-policy from 1 to 2** ([#974](https://github.com/doobidoo/mcp-memory-service/pull/974)): GitHub Actions dependency update.
- **chore(deps): bump actions/setup-python from 4 to 6** ([#975](https://github.com/doobidoo/mcp-memory-service/pull/975)): GitHub Actions dependency update.
- **chore(deps): bump uv group with 13 updates** ([#976](https://github.com/doobidoo/mcp-memory-service/pull/976)): Dependency maintenance update.

### Documentation

- **docs: modernize server commands to `memory` CLI** ([#969](https://github.com/doobidoo/mcp-memory-service/pull/969)): Sweep across 23 docs files replacing outdated/dead startup commands with the modern `memory` lifecycle CLI (`launch`, `server`, `restart`). Fixes broken refs (`python scripts/run_http_server.py` — wrong path; `./start_all_servers.sh`, `./stop_all_servers.sh`, `./status_servers.sh`, `python run_server.py` — no longer exist; `python -m src.mcp_memory_service.server` — invalid module path). Modernizes legacy patterns (`uv run memory server` → `memory launch` for HTTP or `memory server` for MCP stdio/Inspector; `python scripts/server/run_http_server.py` → `memory launch`; `./scripts/update_and_restart.sh` → `memory restart`). systemd `ExecStart=` paths, the `claude-hooks/PLUGIN.md` spawn fallback chain, and `python -m mcp_memory_service.server` for MCP stdio are intentionally preserved.

## [10.61.0] - 2026-05-19

### Added

- **feat(milvus): native `update_memory` and `update_memories_batch`** ([#966](https://github.com/doobidoo/mcp-memory-service/pull/966), part of [#888](https://github.com/doobidoo/mcp-memory-service/issues/888), @henry201605): Milvus backend now overrides `update_memory` (delegates to `update_memory_metadata` for a single-record update) and `update_memories_batch` (single batch fetch + batch embedding + single upsert — 1 round-trip instead of N per record). 15 new unit tests added.
- **feat(sse): Last-Event-ID replay on `/api/events` reconnect** ([#953](https://github.com/doobidoo/mcp-memory-service/pull/953)): Bounded ring buffer of recently broadcast events lets SSE clients resume after a transient disconnect per the standard EventSource resume header. Buffer size configurable via `MCP_SSE_REPLAY_BUFFER_SIZE` (default 1000, 0 disables). Replay outcome (`status: resumed` or `status: id_not_in_buffer`) is surfaced in the `connection_established` welcome event so clients can detect overflow and fall back to their own catch-up strategy. Connection-scoped events (welcome, close) and heartbeats are not buffered; filtered broadcasts are excluded to avoid expanding the original audience on replay.

## [10.60.2] - 2026-05-19

### Fixed

- **fix(milvus): replace ANN `search()` with brute-force `query()` in semantic dedup to fix growing-segment visibility on Milvus Lite** ([#964](https://github.com/doobidoo/mcp-memory-service/pull/964), closes [#938](https://github.com/doobidoo/mcp-memory-service/issues/938), @henry201605): Milvus Lite's ANN `search()` cannot find freshly inserted records in unsealed (growing) segments, causing the semantic deduplication check to silently miss near-duplicates. The fix replaces `search()` with `query(consistency_level="Strong")` + client-side cosine similarity computed from the pre-stored normalized embedding and a pre-computed query norm, restoring correct deduplication behaviour on Milvus Lite.

## [10.60.1] - 2026-05-19

### Fixed

- **fix(milvus): add missing `tag_match` param to `get_all_memories`/`count_all_memories`** ([#958](https://github.com/doobidoo/mcp-memory-service/pull/958), @henry201605): Both methods were missing the `tag_match` parameter present in other backends, causing AND/OR tag filtering to be silently ignored in Milvus deployments.
- **fix(hooks): apply protocol-correct port fallback to `session-end.js` `triggerQualityEvaluation`** ([#960](https://github.com/doobidoo/mcp-memory-service/pull/960), fixes [#957](https://github.com/doobidoo/mcp-memory-service/issues/957)): `session-end.js` was not updated alongside the `memory-client.js`/`memory-retrieval.js` fix from PR #952. Applies the same protocol-correct port resolution (omit port for `https://` or `http://` standard ports) to `triggerQualityEvaluation`, restoring hook functionality for Cloudflare Tunnel and reverse proxy deployments.
- **fix(consolidation): repair broken contradiction detection** ([#961](https://github.com/doobidoo/mcp-memory-service/pull/961), fixes [#959](https://github.com/doobidoo/mcp-memory-service/issues/959)): Three bugs in `contradictions.py` caused the detection module to fail silently on every invocation: (1) `list_memories()` replaced by correct `get_all_memories()` call, (2) dataclass attribute access switched from `metadata.get()` dict-style to direct field access (`memory.tags`, `memory.memory_type`), (3) `search_memories()` parameter name and return-type handling corrected. Module now executes as designed.

## [10.60.0] - 2026-05-18

### Added

- **feat(consolidation): temporal contradiction detection via embedding similarity band** ([#949](https://github.com/doobidoo/mcp-memory-service/pull/949), @filhocf): New module `src/mcp_memory_service/consolidation/contradictions.py`. Detects contradictions using a similarity band of 0.4–0.75 (too similar to be independent facts, too different to be duplicates). Emits a `CONTRADICTED_BY` graph edge and sets `superseded_by` on the older memory. Opt-in via `MCP_CONTRADICTION_DETECTION_ENABLED=true` and `MCP_CONTRADICTION_ON_STORE=true`. Integrated as Step 7 in `handlers/quality.py` maintain flow. 8 new tests in `tests/consolidation/test_contradictions.py`.
- **feat(benchmarks): mem0 adapter — tested end-to-end with cloud API** ([#954](https://github.com/doobidoo/mcp-memory-service/pull/954), @filhocf): Adds `scripts/benchmarks/adapters/` with an abstract `BenchmarkAdapter` base class and a concrete `Mem0Adapter` implementation that wraps the mem0 cloud API. Validated end-to-end with the mem0 cloud service. Provides a foundation for systematic latency/quality comparisons between mcp-memory-service and alternative memory backends.

### Fixed

- **fix(milvus): instance-level graph cache + filter superseded in retrieve** ([#948](https://github.com/doobidoo/mcp-memory-service/pull/948), @henry201605): Replaces the class-variable `_graph_storage_cache` with an instance attribute protected by double-checked locking, preventing cross-instance contamination in tests. `retrieve()` now filters out `superseded_by` memories before trimming results to match the sqlite_vec behavior.
- **fix(hooks): use protocol-correct default port for standard HTTPS/HTTP URLs** ([#952](https://github.com/doobidoo/mcp-memory-service/pull/952), fixes [#950](https://github.com/doobidoo/mcp-memory-service/issues/950)): `memory-client.js` and `memory-retrieval.js` used `url.port || 8443` (or `|| 8080`) as the default port. For standard `https://` URLs (e.g. Cloudflare Tunnel, reverse proxy) `url.port` is empty string — causing the fallback to always trigger and producing `https://host:8443/...` instead of the correct portless URL. Fix: use the protocol's default port (`443` for https, `80` for http) when `url.port` is absent, and omit the port from the constructed URL if it matches the protocol default. Resolves broken hook connectivity for all Cloudflare Tunnel and reverse proxy deployments.

## [10.59.2] - 2026-05-17

### Fixed

- **fix(oauth): use AnyUrl for redirect_uri in AuthorizationRequest and TokenRequest** ([#942](https://github.com/doobidoo/mcp-memory-service/issues/942), reported by @tkislan): `HttpUrl` only accepts `http`/`https` — `cursor://`, `vscode://`, `vscode-insiders://` were silently rejected by Pydantic before reaching the `ALLOWED_SCHEMES` whitelist in `registration.py`, making the scheme addition in v10.59.0 a no-op in practice. Fixed: `redirect_uri` fields in `AuthorizationRequest` and `TokenRequest` changed from `Optional[HttpUrl]` to `Optional[AnyUrl]` in `src/mcp_memory_service/web/oauth/models.py`. `ErrorResponse.error_uri` keeps `HttpUrl`. 8 regression tests added in `tests/unit/test_oauth_native_clients.py`.

## [10.59.1] - 2026-05-17

### Fixed

- **fix(oauth): reflect OAuth state parameter verbatim per RFC 6749 §4.1.2** ([#944](https://github.com/doobidoo/mcp-memory-service/pull/944), @tkislan): `_sanitize_state()` stripped non-`[A-Za-z0-9-_.]` characters and truncated to 128 chars before reflecting `state` back to the client. RFC 6749 §4.1.2 requires returning `state` exactly as received. This broke Cursor OAuth (base64url padding `=`, JWTs, values >128 chars all got mangled). Fix: remove `_sanitize_state()` entirely and reflect `state` verbatim. 5 parametrized regression tests added in `tests/unit/test_oauth_native_clients.py`.

## [10.59.0] - 2026-05-16

### Added

- **feat(oauth): file-based PEM key loading via `MCP_OAUTH_PRIVATE_KEY_PATH` / `MCP_OAUTH_PUBLIC_KEY_PATH`** ([#926](https://github.com/doobidoo/mcp-memory-service/pull/926), co-authored by aria-inboxia): New `_load_pem_from_env()` helper in `config.py` reads PEM content from a file path when the corresponding `_PATH` env var is set. Inline env vars continue to take precedence. When a `_PATH` var is set but the file cannot be read, startup aborts with `ValueError` — fail-hard prevents silent JWT invalidation on restart. 6 unit tests added in `tests/test_config.py`.
- **feat(oauth): allow `cursor`, `vscode`, `vscode-insiders` as OAuth redirect URI schemes** ([#942](https://github.com/doobidoo/mcp-memory-service/pull/942), co-authored by tkislan): `src/mcp_memory_service/web/oauth/registration.py` now accepts IDE deep-link schemes (`cursor://`, `vscode://`, `vscode-insiders://`) as valid OAuth redirect URIs, enabling OAuth callbacks for Cursor and VS Code extensions.

### Fixed

- **fix(hooks): symmetric project-affinity check in `memory-scorer.js`** ([#941](https://github.com/doobidoo/mcp-memory-service/issues/941), reported by minecraft-mattsource): Added `projectName.includes(tag)` as the inverse check so short memory tags (e.g. `wing:hoi4`) are matched when the project name is a superset (e.g. `hoi4coach`). Previously all memories were zeroed by the affinity filter in this case.

## [10.58.0] - 2026-05-16

### Added

- **feat(insights): configurable tag exclusion, metadata heuristic, and acknowledgement flow** ([#939](https://github.com/doobidoo/mcp-memory-service/pull/939), discussion [#897](https://github.com/doobidoo/mcp-memory-service/discussions/897)): Three improvements to the InsightGenerator based on real-corpus observations. (1) **`MCP_INSIGHT_EXCLUDE_TAGS`** env var — comma-separated list of tags to exclude from gap detection in addition to the built-in set (`conflict:unresolved`, `automated`, `__test__`, `temporary`, `processed`, `auto-generated`, `insight-card`). Example: `MCP_INSIGHT_EXCLUDE_TAGS=ci,radar`. (2) **Automated-type heuristic** — gap detection is skipped for tags where >90% of memories have an automated `memory_type` (`session`, `auto-generated`, `temporary`), catching status markers not in the exclusion list. Normal types like `observation` are not treated as automated. (3) **Insight card acknowledgement** — tagging an insight card with `acknowledged` causes `store_insights` to materialise a stable sentinel (hash independent of source memories) so the card is never regenerated, even if the original card is later deleted. 8 new tests covering all three paths.
- **feat(harvest): locale-based pattern plugins for multilingual extraction** ([#935](https://github.com/doobidoo/mcp-memory-service/pull/935), @filhocf): Replaces hardcoded English regex in the harvest extractor with a YAML-based pattern system. Patterns are loaded at startup via `HARVEST_LOCALE` env var (default: `"en"`, backward compatible). New files: `harvest/patterns/__init__.py` (loader with additive locale merging), `harvest/patterns/en.yaml`, `harvest/patterns/pt_BR.yaml` (Portuguese), `harvest/patterns/de.yaml` (German). `PatternExtractor` now stores patterns as `self._patterns` (instance-level). Unknown locales log a warning and are skipped. Non-English users see up to 8x improvement in harvest candidates. Closes [#908](https://github.com/doobidoo/mcp-memory-service/issues/908).
- **feat(plugin): smart-tagger example plugin — auto-tagging + mistake-note boost** ([#932](https://github.com/doobidoo/mcp-memory-service/pull/932), @filhocf): Second reference plugin in `examples/plugin-smart-tagger/`, alongside `plugin-audit-log`. Uses the `mcp_memory_service.plugins` entry-point mechanism with `on_store` (auto-tags content by regex: decision, bug, convention, database, infra, frontend, backend) and `on_retrieve` hooks (boosts score for `mistake-note`/`error-replay` tagged memories). Configurable via `MCP_PLUGIN_SMART_TAGGER_ENABLED` (default: `true`) and `MCP_PLUGIN_SMART_TAGGER_BOOST` (default: `0.15`). No external dependencies.

### Fixed

- **fix(milvus): deduplicate `_drain_graph_edges` against Milvus Lite double-batch bug** (commit 6f7e1f82): `_drain_graph_edges` (used by `get_memory_connections`) had the same QueryIterator last-batch duplication issue as `_drain_query_iterator`. Added `seen_ids` set and `"id"` to `output_fields` so graph edges are correctly deduplicated. Fixes `test_get_memory_connections_with_graph_data`.
- **fix(milvus): use `consistency_level="Session"` in semantic-dedup ANN search** (commit 6f7e1f82): Milvus's default `Bounded` consistency does not guarantee that a memory stored earlier in the same session is visible to an immediately following `search()`. `_check_semantic_duplicate` now passes `consistency_level="Session"` so near-duplicates are reliably detected. Fixes `test_semantic_dedup_blocks_near_duplicate`.
- **fix(triage): use existing `daily-triage` label; create missing `automated` label** (commit 6f7e1f82): `triage_discussions.py` was passing `--label 'triage,automated'` to `gh issue create`, but neither label existed, causing the workflow to fail. Changed to `daily-triage` (already present) and created the `automated` label via API.
- **fix(milvus): deduplicate QueryIterator results to defend against Milvus Lite double-batch bug** (commit 96bfbd87): `_drain_query_iterator` now tracks seen primary-key `id` values and skips duplicate rows. Milvus Lite returns the last batch a second time before signalling end-of-data with an empty batch, causing `get_all_memories()`, `count_all_memories()`, and `query_memories()` to return inflated result sets. Fixes `test_get_all_memories_and_count`, `test_query_memories_returns_most_recent_first`, `test_query_memories_pagination`.
- **fix(milvus): remove server-side time filter from semantic-dedup ANN search** (commit 96bfbd87): Some Milvus Lite versions raise `Method not implemented` for filtered ANN searches. `_check_semantic_duplicate` now issues an unfiltered `limit=10` ANN search and applies the time-window cut-off on the client over the returned results.
- **fix(ci): repair YAML parse error in `pr-contributor-welcome` workflow** (commit 96bfbd87): The welcome message was built with a JS template literal whose body lines started at column 0, terminating the YAML block scalar prematurely and causing GitHub to report a workflow file issue on every push. Replaced with a JS array `.join("\n")` so all content stays within the YAML block indentation.
- **fix(test): raise performance threshold to 500ms for CI stability** ([#939](https://github.com/doobidoo/mcp-memory-service/pull/939)): `test_api_search_by_tag_time_filter_performance` was using a 200ms threshold that was too tight for slow GitHub Actions runners. Raised to 500ms. Unrelated to feature changes.

## [10.57.3] - 2026-05-14

### Added

- **feat(milvus): last_accessed tracking via `_access` side-collection** ([#925](https://github.com/doobidoo/mcp-memory-service/pull/925), @henry201605): Implements a lightweight `{collection_name}_access` side-collection that records retrieve-hit timestamps. Fixes the Forgetting engine's `access_boost` (was always falling back to `updated_at`), fixes `count_all_memories(stale_days=N)` where `stale_days` was silently ignored, and fixes `memory_quality(action="maintain")` stale detection. `_touch_access()` is fire-and-forget via `asyncio.create_task` (non-blocking) with graceful degradation if the collection is unavailable. Closes [#923](https://github.com/doobidoo/mcp-memory-service/issues/923).

## [10.57.2] - 2026-05-14

### Fixed

- **fix(deps): pin pymilvus<3.0.0 to restore Milvus Docker CI** ([#921](https://github.com/doobidoo/mcp-memory-service/pull/921)): Added `<3.0.0` upper bound to the `pymilvus` dependency in `pyproject.toml` and re-locked to `2.6.13` in `uv.lock`. pymilvus 3.0.0 introduced breaking API changes that silently broke the Milvus Docker CI job when the dependency was upgraded. Full pymilvus 3.x migration is tracked in [#922](https://github.com/doobidoo/mcp-memory-service/issues/922).

## [10.57.1] - 2026-05-14

### Fixed

- **fix(sqlite): replace GLOB with LIKE ESCAPE for tag matching** ([#916](https://github.com/doobidoo/mcp-memory-service/pull/916)): Replaced `_escape_glob` with `_escape_like` in `sqlite_vec.py`. All 13 tag-filtering query sites migrated from `GLOB ?` to `LIKE ? ESCAPE '\'`, fixing fragility when tags contain `%`, `_`, or `\` characters. Closes [#914](https://github.com/doobidoo/mcp-memory-service/issues/914).
- **fix(milvus): compare values in structural change detection for `preserve_timestamps`** ([#918](https://github.com/doobidoo/mcp-memory-service/pull/918), @henry201605): In `_compute_update_timestamps()`, replaced key-presence checks with value comparisons. Prevents consolidation runs from incorrectly bumping `updated_at` for all memories and fixes the Forgetting engine's `access_boost` fallback logic on Milvus.

## [10.57.0] - 2026-05-13

### Added

- **`tag_match` parameter for `memory_list` MCP tool** ([#904](https://github.com/doobidoo/mcp-memory-service/pull/904), @filhocf): Extends AND/OR tag filtering to `memory_list`, harmonizing it with `memory_search` and `memory_delete`. Accepts `tag_match: "any"` (OR, default — existing behavior unchanged) or `tag_match: "all"` (AND — only memories matching every supplied tag are returned). Implemented across `server_impl.py`, `server/handlers/memory.py`, REST API, and all three storage backends (sqlite_vec, cloudflare, hybrid).
- **Automatic chunking for `memory_store_session`** ([#912](https://github.com/doobidoo/mcp-memory-service/pull/912), @filhocf): Long sessions stored via `memory_store_session` are now automatically split at turn boundaries (lines starting with `User:`, `Assistant:`, `Human:`, `AI:`, `You:`, `Bot:`). Configure with `SESSION_CHUNK_SIZE` env var (default: 1500 chars, set to 0 to disable). Each chunk is tagged `chunk:N/M` for sequential retrieval. Fully backward compatible — short sessions are stored as a single memory with no change in behavior.

### Fixed

- **CI: guard `pr-contributor-welcome` against non-PR event triggers** (commit b42cf004): Added `github.event_name == 'pull_request_target'` guard to the contributor-welcome workflow to prevent crashes when triggered by push events.

## [10.56.3] - 2026-05-13

### Fixed

- **feat(milvus): implement `get_memory_connections()` via graph collection** ([#907](https://github.com/doobidoo/mcp-memory-service/pull/907), @henry201605): `MilvusMemoryStorage.get_memory_connections()` was a stub returning `{}`, meaning hub memories had no protection from archival in the Forgetting engine's connection-based retention boost. Implemented using `QueryIterator` + `asyncio.to_thread` to drain all edges from the `{collection_name}_graph` collection. Adds 2 unit tests.
- **fix(quality): apply Gemini review suggestion for `MAINTAIN_SCAN_LIMIT` fallback**: Replaced `__import__('os')` with a standard `import os`, added `try/except ValueError` guard for invalid `MCP_MAINTAIN_SCAN_LIMIT` env values, and documented the DoS risk of uncapped values in a comment. Addresses a Gemini code-assist review comment on PR #902 that was missed before merge.

## [10.56.2] - 2026-05-12

### Fixed

- **fix(milvus): add missing `stale_days` param to `count_all_memories`** ([#901](https://github.com/doobidoo/mcp-memory-service/pull/901), @henry201605): `MilvusMemoryStorage.count_all_memories()` was missing the `stale_days: Optional[int] = None` parameter present on all other backends, causing a `TypeError` when callers passed this argument. The parameter is now accepted and silently ignored (Milvus has no `last_accessed` field).
- **fix(quality): graceful fallback for `MAINTAIN_SCAN_LIMIT` on stale server process**: Wrapped the `from ...config import MAINTAIN_SCAN_LIMIT, MCP_INSIGHT_CARDS_ENABLED` import in a `try/except ImportError` block. On in-place upgrades where the server process has a stale `sys.modules` cache, the import can fail; the handler now falls back to reading `MCP_MAINTAIN_SCAN_LIMIT` from the environment (default: 2000) so maintain cycles continue working without a full server restart.

## [10.56.1] - 2026-05-12

### Fixed

- **fix(session): pass `session_id` as `conversation_id` to bypass semantic dedup** ([#898](https://github.com/doobidoo/mcp-memory-service/pull/898), @henry201605): `memory_store_session` calls were incorrectly blocked by semantic deduplication against topically-similar atomic memories — a category error (session logs vs atomic facts). Fixed by setting `skip_dedup = bool(conversation_id) or (memory_type == "session")` in `memory_service.store_memory()`.
- **fix(maintain): hoist `get_all_memories()` and `scan_slice` before Steps 5 & 6** (addresses code review on [#899](https://github.com/doobidoo/mcp-memory-service/pull/899)): Eliminates a duplicate DB call and a latent `NameError` where `MAINTAIN_SCAN_LIMIT` was only defined inside Step 5's try block and would have raised `NameError` in Step 6 if Step 5 failed before reaching the import.

## [10.56.0] - 2026-05-12

### Added

- **`MCP_MAINTAIN_SCAN_LIMIT` env var** (default: 2000, 0 = unlimited): Controls how many memories are scanned per maintain cycle for entity extraction (Step 5) and insight card generation (Step 6). Previously hardcoded to 500 — large deployments can now tune or remove the cap entirely.

### Fixed

- **InsightGenerator gap detector skips metadata/status tags**: The `_detect_gaps` method now ignores tags that are operational markers rather than knowledge domains (`conflict:unresolved`, `automated`, `__test__`, `temporary`, `processed`, `auto-generated`, `insight-card`), eliminating false-positive "Decision gap" insight cards for these system tags.

## [10.55.2] - 2026-05-12

### Fixed

- **fix(insights): handle None memory\_type and tags in InsightGenerator sort**: Fixed `TypeError: '<' not supported between instances of 'str' and 'NoneType'` in `InsightGenerator` when memories have `None` values for `memory_type` or `tags` fields. `dict.get(key, default)` does not fall back to the default when the key is present with a `None` value; fixed with `or ""` / `or []` idiom. Step 6 (Insight Cards) in the `maintain` cycle now runs without errors.

## [10.55.1] - 2026-05-11

### Fixed

- **`maintain` Step 5: entity links always 0 due to wrong graph accessor** ([#895](https://github.com/doobidoo/mcp-memory-service/pull/895)): `quality.py` Step 5 checked `storage.graph`, which is never set on storage objects, so `links_stored` was always `0` even when `entities_found > 0`. Fixed by using `get_graph_storage()` — the same accessor pattern used by all other graph handlers.

## [10.55.0] - 2026-05-11

### Added

- **Entity extraction and memory-entity linking** ([#868](https://github.com/doobidoo/mcp-memory-service/pull/868), @filhocf): Phase 2 of the #732 reasoning roadmap. Introduces `EntityExtractor` that detects @mentions, #tags, URLs, and file paths inside memory content. `memory_search` gains an `entity` filter parameter for targeted retrieval; `memory_graph` gains `action="extract_entities"` to surface entities from a memory. Entity extraction also runs as Step 5 in the `maintain` consolidation cycle, continuously indexing entities from new memories.

- **Insight Cards — automated pattern/trend/gap detection** ([#869](https://github.com/doobidoo/mcp-memory-service/pull/869), @filhocf): Phase 3 of the #732 reasoning roadmap. Adds `InsightGenerator` that analyses the memory corpus and produces three insight types: patterns (recurring knowledge clusters), trends (frequency changes over time), and gaps (under-represented topic areas). Runs as Step 6 in the `maintain` consolidation cycle. Opt-in via `MCP_INSIGHT_CARDS_ENABLED` (default: `false`) to avoid performance impact on existing deployments.

### Changed

- **Bump urllib3 2.6.3 → 2.7.0** ([#893](https://github.com/doobidoo/mcp-memory-service/pull/893)): Routine dependency update.

## [10.54.0] - 2026-05-10

### Added

- **`tag_match` parameter for `memory_search` MCP tool** ([#890](https://github.com/doobidoo/mcp-memory-service/pull/890), @filhocf, closes [#889](https://github.com/doobidoo/mcp-memory-service/issues/889)): Extends the AND/OR tag filtering already present in `memory_delete` to the `memory_search` tool. Accepts `tag_match: "any"` (OR, default — existing behavior unchanged) or `tag_match: "all"` (AND — only memories matching every supplied tag are returned). Implemented across `server_impl.py`, `server/handlers/memory.py`, and `storage/base.py`.

## [10.53.0] - 2026-05-09

### Added

- **Milvus consolidation embedding hydration end-to-end** ([#885](https://github.com/doobidoo/mcp-memory-service/pull/885), @henry201605): Completes a 4-PR series (#872, #878, #881, #885) that fixes a production failure on Milvus-backed deployments where consolidation produced 0 clusters and 0 associations. Root cause: the `vector` column was dropped during bulk reads, leaving every `Memory` with `embedding=None`. `consolidator._get_memories_for_horizon` now passes `include_embeddings=True` to both `get_all_memories` and `get_memories_by_time_range`. Supporting changes: `sqlite_vec.get_memories_by_time_range` gains a conditional LEFT JOIN on `memory_embeddings`; `hybrid.py` forwards the kwarg to the primary backend; `cloudflare.py` accepts the kwarg on both methods (ignores it — vectors live in Vectorize); `milvus._coerce_vector` now explicitly rejects `str`/`bytes`/`dict` types and `_log_hydration_stats` receives a pre-computed count for O(n) efficiency. Covered by 24 unit tests (`test_milvus_hydration.py`) and 5 Milvus Lite integration tests (`test_milvus_consolidation.py`).

### Security

- **Bump GitPython 3.1.47 → 3.1.50** ([#886](https://github.com/doobidoo/mcp-memory-service/pull/886)): Resolves 3 high-severity vulnerabilities in transitive dependency (`wandb → GitPython`): path traversal allowing arbitrary file write/delete outside the repository ([GHSA-7545-fcxq-7j24](https://github.com/advisories/GHSA-7545-fcxq-7j24)), newline injection in `config_writer().set_value()` enabling RCE via `core.hooksPath` ([GHSA-v87r-6q3f-2j67](https://github.com/advisories/GHSA-v87r-6q3f-2j67)), and newline injection in `config_writer()` section parameter bypassing the prior CVE patch ([GHSA-mv93-w799-cj2w](https://github.com/advisories/GHSA-mv93-w799-cj2w)).

## [10.52.0] - 2026-05-08

### Added

- **Cascading search fallback when semantic results are sparse** ([#883](https://github.com/doobidoo/mcp-memory-service/pull/883), @filhocf, closes [#873](https://github.com/doobidoo/mcp-memory-service/issues/873)): Adds a two-tier fallback to `retrieve_memories` for deployments where vector similarity produces fewer results than requested. When enabled (`fallback=True`, opt-in), the system first attempts a BM25 exact-match pass over stored content, then a tag-intersection pass, and merges de-duplicated results up to `n_results`. Default is `fallback=False` so existing callers are unaffected.

### Changed

- **`MemoryStorage` ABC — `include_embeddings` parameter on bulk-read methods** ([#881](https://github.com/doobidoo/mcp-memory-service/pull/881), @henry201605): `get_all_memories` and `get_memories_by_time_range` in the base class (and all concrete backends) now accept `include_embeddings: bool = False`. When `True`, raw embedding vectors are hydrated into the returned `Memory` objects, enabling consolidation pipelines that need embedding data without a separate fetch. Default preserves existing behaviour for all callers.

### Fixed

- **CI fork-PR label/comment automation** ([#882](https://github.com/doobidoo/mcp-memory-service/pull/882)): Workflow triggers that write labels or post comments now use `pull_request_target` instead of `pull_request`, resolving `403` read-only-token failures that broke automation for all fork-originated PRs.

## [10.51.3] - 2026-05-08

### Added

- **Versioned memory update via `memory_update` tool** ([#865](https://github.com/doobidoo/mcp-memory-service/pull/865), @filhocf): Adds `versioned: bool = False` parameter to the `memory_update` MCP tool. When `True`, routes through `update_memory_versioned()` in sqlite_vec, storing `superseded_by` in the replaced memory's metadata for a full audit trail. Returns an explicit error message on backends that do not support versioning.
- **Transitive inference and relationship suggestions in `memory_graph`** ([#866](https://github.com/doobidoo/mcp-memory-service/pull/866), @filhocf): Wires `infer` and `suggest` actions into the `memory_graph` MCP tool. `infer_transitive` computes the transitive closure of a starting node using a recursive CTE in `GraphStorage` (database-side traversal, no Python BFS), keeping large graph queries fast. `suggest_relationships` proposes new edges based on semantic proximity of existing associations.

## [10.51.2] - 2026-05-08

### Fixed

- **OAuth CORS preflight failures and missing resource_metadata** ([#877](https://github.com/doobidoo/mcp-memory-service/pull/877), @ghelleks): Resolves three bugs in the OAuth remote connector flow. CORS headers were missing on the `oauth_app` sub-application; `OPTIONS` requests to `/mcp` were not handled, blocking browser-based preflight checks; and `WWW-Authenticate` headers lacked the `resource_metadata` field required by RFC 9728, causing Remote MCP clients to fail authentication discovery. Fixes issue [#876](https://github.com/doobidoo/mcp-memory-service/issues/876).
- **Milvus consolidation returning 0 clusters/associations** ([#878](https://github.com/doobidoo/mcp-memory-service/pull/878), @henry201605): Adds `include_embedding: bool = False` opt-in parameter to Milvus read paths (`retrieve_memory`, `list_memories`). When `True`, raw embedding vectors are returned alongside memory data, enabling the consolidation pipeline to access embeddings during clustering and association discovery. Fixes consolidation silently returning 0 clusters and 0 associations on Milvus deployments.

## [10.51.1] - 2026-05-07

### Fixed

- **Milvus consolidation failure** ([#872](https://github.com/doobidoo/mcp-memory-service/pull/872), @henry201605): Adds `delete_memory(hash) -> bool` alias to `MilvusMemoryStorage`. Without this method, `memory_consolidate` silently failed on the Milvus backend during Compression (stage 4) and Controlled Forgetting (stage 5) with `AttributeError`. No behaviour change for other backends.

## [10.51.0] - 2026-05-07

### Added

- **Plugin fire points wired into MemoryService lifecycle** ([#864](https://github.com/doobidoo/mcp-memory-service/pull/864), @filhocf): Connects the plugin hook scaffolding (introduced in v10.50.0, PR #856) to actual lifecycle events in `MemoryService`. The four hooks — `on_store`, `on_delete`, `on_retrieve`, and `on_consolidate` — are now invoked at the appropriate call sites. Third-party plugins registered via `entry_points` will receive live events from this release onward.
- **`GET /api/types` endpoint + dynamic type dropdowns in dashboard** ([#863](https://github.com/doobidoo/mcp-memory-service/pull/863), @filhocf): New REST endpoint returns all valid memory types (built-in + custom types from `MCP_CUSTOM_MEMORY_TYPES`). The web dashboard type filter and store-form dropdowns are now populated dynamically from this endpoint instead of being hardcoded, so custom ontology entries appear automatically in the UI.
- **Audit-log example plugin** ([#867](https://github.com/doobidoo/mcp-memory-service/pull/867), @filhocf): Reference implementation in `examples/plugins/audit_log/` demonstrating all four lifecycle hooks. Shows how to write, package, and install a plugin using `entry_points` discovery. Serves as living documentation for the plugin API.

## [10.50.0] - 2026-05-06

### Added

- **Plugin hook scaffolding** ([#856](https://github.com/doobidoo/mcp-memory-service/pull/856), @filhocf, refs [#732](https://github.com/doobidoo/mcp-memory-service/issues/732)): Introduces the plugin extension API for `mcp-memory-service`. Four lifecycle hooks are defined — `on_store`, `on_delete`, `on_retrieve`, and `on_consolidate` — with `entry_points` discovery so third-party packages can register hooks without modifying core. This is pure scaffolding; fire points will be wired into `MemoryService` in a follow-up PR. Enables the ecosystem extensibility roadmap item from canonical issue #732.

### Changed

- **Dependency bumps** (PRs [#858](https://github.com/doobidoo/mcp-memory-service/pull/858), [#859](https://github.com/doobidoo/mcp-memory-service/pull/859), [#860](https://github.com/doobidoo/mcp-memory-service/pull/860), [#861](https://github.com/doobidoo/mcp-memory-service/pull/861)):
  - `actions/attest-build-provenance` 1 → 4
  - `github/codeql-action` 3 → 4
  - `hadolint/hadolint-action` 3.1.0 → 3.3.0
  - `authlib` 1.7.0 → 1.7.1, `cryptography` 47 → 48, `torch` 2.10 → 2.11, `setuptools` 82 → 81 (uv group bump, 905 tests validated)

## [10.49.4] - 2026-05-05

### Fixed

- **[#853] Mistake-notes survive consolidation**: `_is_protected_memory()` in [`consolidation/base.py`](src/mcp_memory_service/consolidation/base.py) now shields memories with `memory_type='mistake'` and `failure_count >= 3` from decay and forgetting passes. High-value error-replay records no longer vanish during scheduled consolidation. 10 new tests in `tests/consolidation/test_mistake_lifecycle.py`. Closes #853. (PR #854, @filhocf)

## [10.49.3] - 2026-05-05

### Fixed

- **[#847] OpenCode plugin API path and payload corrected**: The plugin was calling `/api/memories/search` (HTTP 405) instead of `/api/search`, and sending `limit` instead of `n_results` in the request body (per `SemanticSearchRequest` schema). Both issues caused all OpenCode memory searches to fail silently. Fix applied in PR #850. Closes #847.
- **[#847] OpenCode plugin tag filter now enforced client-side**: `/api/search` ignores the `tags` field server-side, so project-scoped searches were returning unfiltered results. The plugin now over-fetches (`max(limit * 4, 20)`) when tags are present and filters client-side by tag intersection before trimming to the requested limit. Fix applied in PR #849 (Gemini review follow-up).
- **CI version-drift detection**: `scripts/ci/check_versions.sh` updated to skip landing-page version checks for PATCH releases, matching the documented release protocol that landing-page updates are MINOR/MAJOR only. (PR #850)

## [10.49.2] - 2026-05-05

### Fixed

- **[#842] Custom base types with empty subtype lists were silently dropped**: `_load_custom_types_from_config` in [`models/ontology.py`](src/mcp_memory_service/models/ontology.py) guarded registration with `if valid_subtypes:`, so a type declared as `MCP_CUSTOM_MEMORY_TYPES='{"foo": []}'` — the exact form documented in the v10.49.1 coercion warning and in the `memory_store` tool description — was never added to the ontology. The validator never saw `foo`, so `Memory.__post_init__` kept coercing it to `"observation"` even after the user had correctly configured the env var. Fix: register the base type unconditionally; emit a warning only when subtypes were supplied but all failed validation. Two regression tests added. Closes #842. (PR #846)

## [10.49.1] - 2026-05-05

### Fixed

- **[#842 / #843] `memory_type` ontology coercion was invisible to callers**: When a user passed an unknown `memory_type` (e.g. `"foo"`), `Memory.__post_init__` silently rewrote it to `"observation"` and only logged a warning. The MCP/HTTP store responses still reported `success`, so the caller had no way to detect the rewrite — subsequent `memory_list` queries filtered on the original type returned 0 results and looked like a broken filter (#842). Fix: both the MCP `memory_store` handler ([`server/handlers/memory.py`](src/mcp_memory_service/server/handlers/memory.py)) and the HTTP `POST /memories` endpoint ([`web/api/memories.py`](src/mcp_memory_service/web/api/memories.py)) now compare the requested vs. effective `memory_type` and append a visible warning to the response when they diverge, including a hint to register the type via `MCP_CUSTOM_MEMORY_TYPES`. The default `"note"` (applied when the caller omits `type`) does not trigger a warning. Tool description for `memory_store` updated to enumerate the built-in base types and link to the new ontology guide. Closes #843.
- **[#797] `/api/quality/memories/{hash}/evaluate` self-relevance loop returned 1.0 for everything**: When no request body was supplied, the endpoint defaulted the relevance query to the memory's own first 200 chars, collapsing the relevance prompt to "rate how relevant X is to X" — which any LLM scores at the ceiling. All `openai-compatible` AI tier evaluations on v10.45.0+ deployments returned `ai_score: 1.0` regardless of memory quality. Fix: pass an empty query through when none is supplied, so `_create_scoring_prompt` takes the absolute-quality branch which produces a calibrated 0.0–1.0 score with proper discrimination. Reported with full reproducer in #797. (PR #839)
- **[#797] `gpt-5.x` family rejected from openai-compatible quality scorer**: `_score_with_openai_compatible` hardcoded `max_tokens=50` and `temperature=0.1`, both of which OpenAI's `gpt-5.x` family rejects with HTTP 400 ("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead"). The AI tier silently fell through to `implicit_signals`, so the provider _appeared_ to work but wasn't using the AI. Fix: branch on `model.startswith("gpt-5")` to use `max_completion_tokens=800` (sized for reasoning variants which spend ~200–400 tokens reasoning internally) and skip `temperature`. Non-gpt-5 models keep the original payload to preserve cost and determinism. Bug 2 reported and patched by @thewusman2025. 6 new tests in `tests/web/api/test_quality_evaluate.py` and `tests/test_openai_compat_quality.py`. Closes #797. (PR #839)

### Changed

- **`docs/deployment/production-guide.md` rewritten as a topology selector**: The previous version was a 56-line stub that hardcoded an example API key, referenced a non-existent `COMPLETE_SETUP_GUIDE.md`, and did not actually guide production deployments. The new version is a topology selector that routes readers to the correct concrete guide (Docker / dual-service / systemd / external-embeddings) plus a common production checklist (WAL pragma, API key generation, OAuth storage, health checks, backups, hybrid sync owner). Security: the example API key (also present in two `archive/` files where it is no longer load-bearing) was removed from the active doc. (PR #836, closes #835)

### Documentation

- **New: [`docs/memory-ontology.md`](docs/memory-ontology.md)**: Documents the built-in memory-type taxonomy (12 base types, ~60 subtypes), the rationale for ontology validation, the coercion warning surfaced in store responses, and the JSON format for `MCP_CUSTOM_MEMORY_TYPES`. Linked from the README documentation index and from the `memory_store` MCP tool description. `.env.example` expanded with concrete examples for the env var. (Issue #843)
- **Housekeeping audit follow-up — wave 1 + wave 2** (closes #823): Archived 19 historical/superseded `docs/` files to `docs/archive/` (5 phase-2 code-quality artifacts, 3 historical development docs, 3 phase reports, 2 superseded release notes, 4 planning artifacts, 2 one-time artifacts). Added 13 inbound links from `docs/README.md` for legitimate orphan guides (gemini integration, groq integrations, deployment guides, natural-memory-triggers, architecture & design). Two new sections added to the docs index: "Natural Memory Triggers" and "Architecture & Design". (PRs #831, #832)
- **Wiki cleanup**: TOC anchors fixed (commits `387db7f`, closes #824), macOS Intel link redirected to existing Platform Setup section (commit `80bdc17`, closes #825), Windows guide cluster consolidated into single `Windows-Hybrid-Setup.md` (closes #834), Cloudflare guide cluster consolidated by removing pre-hybrid `Cloudflare-Based-Multi-Machine-Sync.md` and adding deprecation guidance to `Cloudflare-Backup-Sync-Setup.md` (closes #833).
- **`docs/BENCHMARKS.md` clarification**: benchmark run version (`v10.34.0`) now annotated with "(benchmark run version; latest release: vX.Y.Z)" to prevent new readers from interpreting the historical run version as current. (PR #829, closes #826)

## [10.49.0] - 2026-05-04

### Added
- **CLI lifecycle commands**: Added `launch`, `stop`, `restart`, `info`, `health`, `logs` commands for background HTTP server management. These commands use the new `cli/lifecycle.py` module with cross-platform PID tracking and health polling.
- **Lazy CLI command loading**: Ingestion commands (`ingest-document`, `ingest-directory`, `list-formats`) are now lazy-loaded — imported only when invoked, not at CLI startup.
- **Lazy package imports**: Heavy dependencies (torch, transformers, sentence-transformers) are imported only when lazy-loaded attributes like `Memory`, `MemoryQueryResult`, or `MemoryStorage` are accessed.
- **Unit tests for lazy loading**: Added targeted unit tests covering package lazy imports, CLI lazy command availability, and lifecycle command registration.

### Changed
- **CLI startup performance**: `memory --help` and lifecycle commands now start in under 3 seconds (was ~22s) by avoiding eager ML imports at module load time.
- **HTTP host/port ergonomics**: `memory server --http` now supports `--http-host` and `--http-port`; lifecycle commands use `--host` and `--port` with environment fallbacks.
- **`restart` inherits flags**: `restart` threads `--storage-backend` and `--debug` flags from the running server process automatically.
- **Test conftest cleanup**: Removed Unicode emoji characters from `tests/conftest.py` to avoid `UnicodeEncodeError` on Windows cp1252 consoles.
- **Test architecture**: `tests/test_memory_ontology_integration.py` rewritten to drop `sys.modules` shadow gymnastics; subprocess isolation for lazy-import assertions; qualified conftest imports in csv/json/semtools loader tests restored (+18 recovered tests, total ~1,803).

### Security
- **Fixed command injection in `launch`**: Replaced unsafe `-c` command string with safe argument list using `sys.executable -m uvicorn` with separate args for `--host` and `--port`. User-controlled host values no longer get interpolated into code strings. (PR #740)
- **Fixed file handle leak in detached launch**: Parent process now explicitly closes stdout/stderr file handles immediately after spawning child process, preventing resource exhaustion. (PR #740)
- **`MCP_ALLOW_ANONYMOUS_ACCESS` pass-through**: The env var is no longer forced to `true` internally; the actual environment value is now passed through to the launched server, restoring the intended security boundary.
- **PR #438 test-safety protections preserved**: Windows test-safety positive allowlist is maintained alongside production indicators — the triple-safety system against production database deletion remains intact.

### Fixed
- **PID-reuse detection**: `stop` and `restart` commands detect PID reuse via `create_time` + `cmdline_hint` to avoid killing an unrelated process that inherited the same PID.
- **Version lookup**: CLI `info` command now reads version from `_version.py` directly instead of `importlib.metadata`, avoiding stale cached values.
- **Optimized log reading**: `logs` command uses streaming tail with `collections.deque(maxlen=lines)` instead of `read_text().splitlines()` to avoid loading the entire log file into memory. (PR #740)
- **Security warnings in docs**: Added explicit security notice in `launch` command help text and `README.md` lifecycle section warning that binding to non-loopback hosts exposes the API. (PR #740)

### Credits
- Lead author: @creativelaides (Jose Velaides) — original PR #743; all 6 commits co-authored on the merged squash in PR #841.


## [10.48.0] - 2026-05-02

### Added

- **[#732] `include_superseded` retrieval filter — opt in to see the full contradiction chain**: All retrieval paths (`memory_search`, `retrieve`, `retrieve_with_quality_boost`, `retrieve_hybrid`) and all storage backends (`sqlite_vec`, `cloudflare`, `hybrid`, `milvus`, `http_client`) now accept an `include_superseded: bool = False` parameter. Default behavior is unchanged — superseded memories stay filtered out — but callers can pass `include_superseded=True` to retrieve the complete contradiction chain. Partial implementation of RFC #732. Thanks to @filhocf. (PR #814)
- **[#732] Auto-mark `superseded_by` on high-confidence contradiction**: The consolidator now automatically marks the older memory with `superseded_by` pointing to the newer one whenever a `contradicts` relationship is detected with confidence ≥ 0.75. Executed in a single batched transaction via the new `mark_superseded_batch()` storage method (thread-safe via `_conn_lock` / `_execute_with_retry`). No DB migration needed — uses existing `superseded_by` column from migration 011. 5 new tests in `tests/storage/test_superseded_filter.py`. Thanks to @filhocf. (PR #814)


## [10.47.2] - 2026-05-02

### Fixed

- **[#808] Consolidation schedule defaults changed to `'disabled'` — operators must now opt in to automatic consolidation**: Previously, omitting `MCP_SCHEDULE_DAILY`, `MCP_SCHEDULE_WEEKLY`, and `MCP_SCHEDULE_MONTHLY` env vars silently activated automatic consolidation runs at `02:00` daily, `SUN 03:00` weekly, and `01 04:00` monthly. One affected deployment accumulated 1,369 unintended `compressed/` entries and had 76+ files silently archived in a monthly run they believed was disabled. The defaults are now `'disabled'` — automatic consolidation no longer runs unless the env vars are explicitly set. **If you relied on the prior automatic behavior**, add the following to your `.env` to restore it: `MCP_SCHEDULE_DAILY=02:00`, `MCP_SCHEDULE_WEEKLY=SUN 03:00`, `MCP_SCHEDULE_MONTHLY=01 04:00`. A new `CONSOLIDATION SCHEDULING` section in `.env.example` documents the syntax. Quarterly schedule format docstring corrected (caught by gemini-code-assist). Closes #808. (PR #821, commit `0d4a658`)

### Added

- **[#811] Docker `:slim` and `:quality-cpu` images missing `aiosqlite` and other core deps**: The hand-curated dependency lists in `tools/docker/Dockerfile.slim` and `tools/docker/Dockerfile.quality-cpu` had drifted from `pyproject.toml`'s `dependencies` block. Both Dockerfiles use `pip install -e . --no-deps` to skip the heavy ML stack (`torch`, `transformers`, `sentence-transformers`) but were also accidentally skipping `aiosqlite>=0.20.0` (the visible failure: `ModuleNotFoundError: No module named 'aiosqlite'` on first memory tool call with `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`), plus `apscheduler` (consolidation), `authlib`/`PyJWT[crypto]`/`cryptography` (OAuth), `httpx`/`requests`, `python-dotenv`, and `pypdf`. The list was also still installing the deprecated `PyPDF2` while the code imports `pypdf`. Both Dockerfiles now ship the full pyproject `dependencies` set minus the heavy ML deps, with `mcp` and `tokenizers` version specifiers re-aligned to pyproject. The `Dockerfile.slim` install step also strips uv/pip caches at the end of the RUN, mirroring `Dockerfile.quality-cpu`. Closes #811. (PR #815)

## [10.47.1] - 2026-05-01

### Fixed

- **[#729] `/server/update` no longer silently fails**: The dashboard's *Update & Restart* button used to show a generic success toast even when `git pull` aborted on a dirty working tree, when `pip install` failed, or when the process never actually restarted. Three changes fix this end-to-end: (1) `/api/server/update` now refuses to pull on a dirty working tree (HTTP 409 with the offending paths) unless the new `force=true` flag is set; (2) git/pip failures now return HTTP 500 with the real stderr in `detail` instead of HTTP 200 with `status: "error"` in the body that the frontend never read; (3) `/api/server/restart` and `/api/server/update` now return `pre_restart_pid` + `pre_restart_version`, and the dashboard polls `/api/server/status` after restart to verify the process actually rolled over (pid or version changed). The restart overlay surfaces a clear warning if the process never restarted instead of reloading to the same stale build. The dashboard also offers a force-retry confirmation dialog when a dirty tree blocks an update. 8 new tests in `tests/web/api/test_server_management.py`. Closes #729. (PR #807)
- **[Security] CodeQL log injection fix**: Inline CR/LF sanitization on all user-controlled strings written to `server.py` audit logs, preventing log-forging via malicious input.
- **[Frontend] `apiCall` now attaches `.status` to thrown errors**: `app.js` error objects surface the HTTP status code so callers can branch on specific failure codes (e.g. 409 dirty-tree vs 500 subprocess failure).
- **[Tests] `test_server_management.py` monkeypatch switched to module-object refs**: Fixes `uvx`/isolated-env CI failures where dotted-string patch targets resolved to the wrong module copy.

## [10.47.0] - 2026-05-01

### Added

- **[#799] `memory_quality(action='maintain')` — one-call maintenance orchestrator**: New `maintain` action runs cleanup → conflict detection → stale detection → quality snapshot in a single cycle. `dry_run=true` by default so the tool is always safe to call. New `maintain_status` action returns last-run stats. Auto-resolve is opt-in via `MCP_MAINTAIN_AUTO_RESOLVE` (default `false`); when enabled, conflict pairs are auto-resolved when three signals all pass: cosine similarity ≥ `MCP_MAINTAIN_AUTO_RESOLVE_THRESHOLD` (default `0.95`), same `memory_type`, and age delta > `MCP_MAINTAIN_AUTO_RESOLVE_AGE_DAYS` (default `7`). Winner is the newer memory (`created_at`). New env vars: `MCP_MAINTAIN_STALE_DAYS`, `MCP_MAINTAIN_AUTO_RESOLVE`, `MCP_MAINTAIN_AUTO_RESOLVE_THRESHOLD`, `MCP_MAINTAIN_AUTO_RESOLVE_AGE_DAYS`. 10 new tests. Closes #799. Thanks to @filhocf for the contribution. (PR #802)

### Changed

- **[#793] Quantize deberta quality classifier at Docker build time**: New `tools/docker/scripts/quantize_quality_models.py` runs after the existing ONNX export in `Dockerfile.quality-cpu`. Produces fp16 (`onnxconverter-common`) and dynamic int8 (`onnxruntime.quantization.quantize_dynamic` on MatMul + Gather) variants of `nvidia-quality-classifier-deberta`, benchmarks each against the fp32 baseline (file size, mean inference latency, and Pearson correlation on 100 sample texts), and replaces `model.onnx` in place with the smallest variant whose correlation is ≥ 0.98. The correlation reduction mirrors the production scoring path in `onnx_ranker.py::score_quality` (softmax + weighted sum `[High=1.0, Medium=0.5, Low=0.0]`) so the metric reflects the score that actually drives quality decisions at runtime. Cleanup deletes the fp32 external-weights sidecar (`model.onnx_data`, ~700 MB) along with rejected variants. Falls back to fp32 (no build failure) if no variant meets the gate; pass `--strict` to make a missed gate hard-fail. Strategy is overridable at build time via `--build-arg QUANTIZE_MODE={fp16|int8|best}` and `--build-arg QUANTIZE_MIN_CORR=<float>`. Expected size delta vs `:slim` drops from ~1.7 GB to ~600 MB on the next tagged build. `ms-marco-MiniLM-L-6-v2` is intentionally not quantized (already ~80 MB). Closes #793. (PR #803)

## [10.46.0] - 2026-04-30

### Added

- **[#784] `stale_days` filter for `memory_list`**: Adds an optional `stale_days` integer parameter to the `memory_list` tool and REST endpoint. Memories whose `COALESCE(last_accessed, created_at)` timestamp falls strictly before `now - stale_days * 86400 seconds` are considered stale. Memories accessed exactly at the threshold are NOT stale (strict `<` semantics). Memories that have never been read (`last_accessed IS NULL`) fall back to `created_at`, so never-accessed memories are included in stale results when their creation date is old enough. The filter composes freely with the existing `tags`, `memory_type`, and pagination parameters. Backend coverage: fully implemented in SQLite-vec; Cloudflare, Hybrid, and Milvus backends accept the parameter but ignore it (returning all memories as before — no silent wrong results). Refactors `_apply_stale_days_filter` as a static helper shared between `get_all_memories` and `count_all_memories` to avoid duplication (per code review). 8 new tests in `tests/storage/test_stale_days.py`. Closes #784. Thanks to @filhocf for the contribution. (PR #796)

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

