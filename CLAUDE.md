# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository. It is written to be self-sufficient: a model should be able to work correctly here from this file alone, without filling gaps from prior knowledge.

> **Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Information Lookup**: Files first, memory second, user last. See [`.claude/directives/memory-first.md`](.claude/directives/memory-first.md) for strategy. Comprehensive project context is stored in the MCP Memory Server with tag `claude-code-reference`.

## Non-Negotiables (hard rules)

Quick reference; each rule is expanded in the sections below. Violations cause real incidents.

1. **Development happens on GitHub.** `origin` is `github.com:doobidoo/mcp-memory-service`; CI is GitHub Actions (`.github/workflows/`); issues, PRs, releases and the tag push that starts a release all live there, and `gh` is the CLI for all of it. GitLab is a push mirror that receives fast-forwards of `main` and **never tags** (see "Source Control & Hosting").
2. **Never manually bump versions.** Follow the documented release workflow for every version bump and release.
3. **Run `bash scripts/pr/pre_pr_check.sh` before every PR.** It is the mandatory pre-PR gate and must pass.
4. **Use the project venv.** Run `.venv/bin/python` and `.venv/bin/pytest` (Python 3.12) — the system interpreters are not the project environment.
5. **Store context in the MCP Memory Server first, tagged `mcp-memory-service`.** Claude Code's file-based memory (`~/.claude/projects/<project>/memory/`) may run alongside it as a local cache — it is a supplement, not a substitute, and it lives outside the repo. Never commit memory files into the working tree.
6. **Access `Memory` fields by attribute** (`memory.tags`), never via `memory.metadata.get('tags')`. This has caused 3 production bugs.
7. **Sanitize user-provided values in logs** with `_sanitize_log_value()` (from `mcp_memory_service.compat`).
8. **MCP server configs go in `.mcp.json`**, not `settings.json`.
9. **Update `site/index.html` version strings on every MINOR/MAJOR release** (a CI gate enforces this).
10. **Never click "Always allow" on heredoc commands** — it corrupts `.claude/settings.local.json`.
11. **Before any SSH/network task, verify `hostname` and connection direction** (source → target).
12. **Report outcomes faithfully.** "Tests pass" means you ran them and saw them pass; if a step was skipped or failed, say so with the output.
13. **Write in the maintainer's or contributor's own voice.** Commit messages, PR descriptions, CHANGELOG entries, issue/PR comments, and release notes read as authored by the maintainer or contributor.

## Critical Directives

**IMPORTANT**: Before working with this project, read:
- **`.claude/directives/memory-tagging.md`** - MANDATORY: Always tag memories with `mcp-memory-service` as first tag
- **`.claude/directives/README.md`** - Additional topic-specific directives

## Operational Rules

### Auto-Save Learnings
- **After completing tasks**: automatically save key learnings, decisions, and patterns to the MCP Memory Server without being asked
- Include relevant tags: `mcp-memory-service`, task-specific tags, and `learnings`

### Source Control & Hosting

- **GitHub is where the work happens.** CI runs as GitHub Actions in `.github/workflows/`: `ci.yml`, `release.yml`, `deploy-site.yml`, `cleanup-images.yml`, `codeql.yml`. Issues, PRs and releases are there, and so is the tag push that starts a release.
- **Only GitHub-owned actions are allowed, and SHA pinning is enforced.** The repository is set to `allowed_actions: selected` with `github_owned_allowed: true`, an empty `patterns_allowed` list, and `sha_pinning_required: true`. A third-party action therefore does not fail at review, it fails at run time. That is why `release.yml` shells out to raw `docker buildx` instead of using the `docker/*` actions. Before adding any `uses:` that is not `actions/*` or `github/*`, the allowlist has to be widened deliberately.
- **A ruleset caps a single push at two refs.** `Pushes can not update more than 2 branches or tags`. Bulk tag pushes have to be batched in pairs; this is why the historical tag import ran as 15 pushes rather than one.
- **One publisher at a time.** `release.yml` owns PyPI and Docker Hub, and it is the only thing that may publish. Before wiring any second automation that could publish, disarm the first one. Codeberg lost its secrets on 2026-09-05 for exactly this reason.
- **PyPI auth is still a classic token.** Trusted Publishing is not configured. The pre-Codeberg workflows declared `id-token: write` but authenticated with `secrets.PYPI_TOKEN` via twine, so switching to OIDC is new work and needs a publisher configured on the PyPI side.
- **GitLab is a push mirror**, not a second publisher: fast-forwards of `main` only, no secrets, **never tags**. Prove the fast-forward instead of assuming it:
  ```bash
  git fetch origin main
  git merge-base --is-ancestor "$(git ls-remote gitlab main | cut -f1)" FETCH_HEAD
  git push gitlab FETCH_HEAD:refs/heads/main
  ```
  If the ancestor check fails, stop and investigate. Do not force. Note that some shells mangle `merge-base --is-ancestor "$A" "$B"`; if it reports "Not a valid object name" on a ref that clearly exists, run it through Python's `subprocess` rather than debugging the ref.
- **Codeberg is a frozen archive** at `codeberg.org/doobidoo/mcp-memory-service`. It holds no secrets, receives no pushes, and its CI is dead. It stays readable so old links resolve. Do not resume mirroring to it: Codeberg's ToU § 2 (1) 7 bars projects written mostly with generative AI tools, this account was warned under that clause on 2026-08-27, and a mirror is still sharing a project.
- **Issue and PR numbers below #341 are Codeberg numbers.** GitHub will happily auto-link `#123` in old CHANGELOG and README entries to a GitHub issue of that number, which is a different thing entirely. When citing history, say "Codeberg #123" or link the full URL. [`docs/codeberg-issue-map.md`](docs/codeberg-issue-map.md) resolves every one of them. The issues that were still open on 2026-09-05 were re-created on GitHub and carry the label `migrated:codeberg`.
- **GHSA identifiers** (e.g. `GHSA-2r68-g678-7qr3`) are just advisory IDs and remain valid references.
- **Authorship voice.** Commit messages, PR descriptions, CHANGELOG entries, issue/PR comments, and release notes are written in the maintainer's or contributor's own voice.

### Release Workflow Checklist
Before merging or releasing:
1. **Verify CI is green on the target branch** (GitHub Actions). `gh run list --branch <branch>` or the Actions tab.
2. **Update `site/index.html` version strings** whenever MAJOR.MINOR changes (i.e. every MINOR or MAJOR release — PATCH releases are exempt). The `version-drift-check` CI gate enforces this and will fail if skipped. Update ALL occurrences: `<title>`, `<meta og:title>`, hero badge, "What's New" section, release link `href`. Use `grep -n "v11\." site/index.html` to find them. This is MANDATORY — not optional for "incremental" releases. The site auto-deploys to Cloudflare Pages (mcpmemory.services) when the change lands on main (`.github/workflows/deploy-site.yml`).
3. **Bump `claude-hooks/.claude-plugin/plugin.json` if the hooks changed.** The Claude Code plugin carries its own version, and the Marketplace cache is keyed on it, so a hook fix released without a manifest bump never reaches installed users. The `plugin-version-check` CI gate enforces this on release changes (`scripts/ci/check_plugin_version.sh`): it fails when `claude-hooks/` changed since the last commit that moved the manifest version. v11.6.0 shipped a hook fix this way (#170).
4. Clean up merged branches after release (`git branch -d`, `git push origin --delete`).
5. Follow the release workflow — never manually bump versions.

## Overview

Current version and release history: [CHANGELOG.md](CHANGELOG.md). The forge-API tag trap that
left v11.8.1 unpublished is documented in [`.claude/directives/version-management.md`](.claude/directives/version-management.md).

## Essential Commands

### Python Environment

This repo uses a project virtualenv at `.venv` (Python 3.12). The system `python`/`pytest` are **not** the project interpreter. Always use the venv binaries:

```bash
.venv/bin/python -m mcp_memory_service.server   # run a module
.venv/bin/pytest                                # run the test suite
.venv/bin/python -m pip install -e .            # editable install
```

The venv is not guaranteed to carry the dev tooling: a `uv`-created one holds the
runtime dependencies only, with no `pip` and no `pytest`. `scripts/pr/pre_pr_check.sh`
resolves its interpreter from `.venv/bin/` before it looks at `$VIRTUAL_ENV`, so it
cannot be pointed elsewhere, and it discards the output of its own `pip install
pytest-cov`. The result is a bare `FAIL - Test suite` at step `[3/9]` with no test
names under it. That is an environment failure, not a broken change. Repair it with:

```bash
uv pip install --python .venv/bin/python pip pytest pytest-cov pytest-asyncio \
    pytest-subtests pytest-timeout pytest-mock
```

`pytest-timeout` is required, not optional -- the gate passes `--timeout=120`.

The commands below are written as bare `python`/`pytest` for brevity — run them via `.venv/bin/...`. For `git commit`, the pre-commit hook uses **system** Python and will fail with "Package not installed"; prefix commits with `PATH=".venv/bin:$PATH"` (see Troubleshooting). `uv` is also used (`uv.lock` present); `uv pip install -e .` and `uv run memory ...` work too.

### Development Server

The lifecycle CLI (`memory launch|stop|restart|info|logs`, see `memory --help`) is the
preferred way to manage the server. The legacy entry points
(`python -m mcp_memory_service.server`, `scripts/server/run_http_server.py`,
`scripts/update_and_restart.sh`) still work but may be deprecated in a future release;
`run_http_server.py` and `update_and_restart.sh` are not the path for new deployments.

### Testing

Standard `pytest` invocations apply; markers are declared in `pytest.ini`. Exact test
count: `.venv/bin/pytest --collect-only -q | tail -1`.

```bash
# Pre-PR validation (MANDATORY before submitting PR)
bash scripts/pr/pre_pr_check.sh
```

**Full command reference:** [scripts/README.md](scripts/README.md). Validation and
diagnostics live in `scripts/validation/`; the HTTP health endpoint is
`/api/health` on the configured port (default 8000).

## Code Architecture

Package layout: `ls src/mcp_memory_service/`. Non-obvious entry points:

- `server_impl.py` - the `MemoryServer` class. `list_tools()` builds the tool list from the declarative registry, `call_tool()` dispatches through the routing table, and the `handle_*` methods live here too.
- `tools/registry.py` - `TOOL_REGISTRY`, the declarative list of `ToolDef` objects, each mapping 1:1 to a `types.Tool`. Count: `grep -c 'ToolDef(' src/mcp_memory_service/tools/registry.py`.
- `tools/routing.py` - `ROUTING_TABLE` (name -> handler) plus `resolve_handler(name)` with lazy import.
- `server/handlers/*.py` - request handlers, signature `async def handle_X(server, arguments) -> List[types.TextContent]`.
- `compat.py` - compatibility shims and `_sanitize_log_value()`.
- `consolidation/` - dream-inspired memory maintenance, not a data-consolidation layer.

Global singleton caching in `server/cache_manager.py` prevents redundant storage
initialization across MCP tool calls. Do not instantiate storage per call.

### Storage backends (`storage/`)

Strategy Pattern over `BaseStorage` (`base.py`), three implementations:

| Backend | File | Performance | Use case |
|---------|------|-------------|----------|
| SQLite-Vec | `sqlite_vec.py` | 5ms reads | Development, single-user |
| Cloudflare | `cloudflare.py` | Network-dependent | Cloud-only, edge |
| Hybrid | `hybrid.py` | 5ms local plus cloud sync | **Production (recommended)** |

SQLite-Vec uses the sqlite-vec extension for KNN search, Cloudflare uses D1 plus
Vectorize, Hybrid reads locally and syncs to Cloudflare in the background. Graph
storage sits in `graph.py`. Embeddings: ONNX, `sentence-transformers/all-MiniLM-L6-v2`.

### Web layer (`web/`)

FastAPI dashboard plus REST API. Two things that are not visible from the file names:

- `api/mcp.py` is a thin MCP-over-HTTP protocol shim. Its tool surface and dispatch
  are inherited from the shared `MemoryServer`, so tools are not defined twice.
- Tools that read caller-supplied filesystem paths are filtered out of the remote
  transport by `local_only_tools()` in `server_impl.py`. This is the confused-deputy
  guard, not an optimization.

### Quality system (`quality/`)

Three tiers, local first:

| Tier | Provider | Latency | Cost | Role |
|------|----------|---------|------|------|
| 1 | Local ONNX | 80-150ms | 0 | Default |
| 2 | Groq / Llama 3 | 500-800ms | 0.0015 | Fallback if local fails |
| 3 | Gemini 1.5 Flash | 1-2s | 0.01 | High-accuracy scoring |

Scores are 0.0 to 1.0 and feed quality-boosted search and retention policy.

### Consolidation (`consolidation/`)

Runs via the HTTP API, not the MCP tools (roughly 90 percent fewer tokens), scheduled
with APScheduler. Retention by quality tier: high 365d, medium 180d, low 30-90d
(`forgetting.py`). Relationship inference classifies associations as causes, fixes,
contradicts, supports, follows or related, with a default confidence threshold of 0.6.
For existing associations, backfill with
`scripts/maintenance/update_graph_relationship_types.py`.

### Document ingestion (`ingestion/`)

Registry pattern, loaders selected by file extension. Chunker defaults: 1000 chars
with 200 overlap. `semtools_loader.py` is optional and adds LlamaParse for PDF, DOCX
and PPTX.

## Test Architecture

About 2,400 tests across roughly 216 files. `tests/` mirrors `src/`.

### Test safety (critical, PR #438)

On Feb 8, 2026 a test cleanup deleted 8,663 production memories. Recovery ran off an
emergency backup. Three guards now stand between the suite and a production database,
all in `conftest.py`:

1. An isolated temp directory with prefix `mcp-test-` is created at module import time.
2. `pytest_sessionstart` aborts the run if a production path is detected.
3. `pytest_sessionfinish` validates temp location, absence of production indicators,
   and presence of test markers before deleting anything.

Tests force `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec` unless
`MCP_TEST_ALLOW_CLOUD_BACKEND=true`. Do not weaken any of this to make a test pass.

### Fixtures and conventions (`conftest.py`)

`temp_db_path` gives an auto-cleaned temp directory, `unique_content` generates
non-duplicate test content, and `test_store` auto-tags everything with
`TEST_MEMORY_TAG = "__test__"`, which is reserved for automatic cleanup. Never use
that tag for real data.

Markers in `pytest.ini`: `unit`, `integration` (needs storage), `performance`,
`asyncio` (auto-detected).

## Configuration

Full variable list: `.env.example`. Precedence: environment variables, then `.env`,
then global Claude config, then defaults. After changing `.env`, restart with
`memory restart`.

Three settings that cause real incidents when wrong:

- `MCP_MEMORY_SQLITE_PRAGMAS` must include `journal_mode=WAL`. Without it, concurrent
  reads and writes are disabled and the HTTP server plus MCP server running together
  produce "database is locked". Working value:
  `journal_mode=WAL,busy_timeout=15000,cache_size=20000`.
- `MCP_HYBRID_SYNC_OWNER=http` for hybrid mode. Then only the HTTP server syncs to
  Cloudflare, the MCP server skips Cloudflare initialization entirely and talks to
  SQLite-Vec directly, and `claude_desktop_config.json` needs no Cloudflare
  credentials. Claude Desktop gets memory access, the HTTP server owns sync.
- `MCP_INIT_TIMEOUT` defaults to 30s on Windows and 15s elsewhere, auto-doubled on
  first run. Slow Windows machines need it raised.

### External embedding APIs

Only supported on the `sqlite_vec` backend, not on `hybrid` or `cloudflare`. Set
`MCP_EXTERNAL_EMBEDDING_URL` and `MCP_EXTERNAL_EMBEDDING_MODEL`, optionally
`MCP_EXTERNAL_EMBEDDING_API_KEY`. Works with vLLM, Ollama, TEI, OpenAI or any
OpenAI-compatible `/v1/embeddings` endpoint. Embedding dimensions must match the
database schema, and changing them requires re-embedding every memory. Details:
[`docs/deployment/external-embeddings.md`](docs/deployment/external-embeddings.md).

Claude Desktop wiring: see README.

## Development Guidelines

### Quality gates

Three layers: pre-commit (under 5s, Groq/Gemini complexity plus security, blocks on
complexity above 8 or a security finding), the PR gate
`bash scripts/pr/pre_pr_check.sh` (10-60s, blocks on security or health below 50), and
a weekly pyscn review with trend tracking. Health score below 50 blocks a release,
50-69 means refactor within two weeks, 70 and up is fine. Complexity target is grade
A-B, meaning 8 or lower.

### Log injection guard (v10.68.0, from CodeQL GHSA-84hp-mqvj-3p8h)

User-provided values in raw f-string log calls trigger CodeQL `py/log-injection`.
Wrap them:

```python
from mcp_memory_service.compat import _sanitize_log_value
logger.info(f"Stored: {_sanitize_log_value(content)}")
```

The helper strips newline, carriage return and escape characters, which prevents log
forging and ANSI injection. `pre_pr_check.sh` check 6.5 detects unwrapped f-string
logger calls. For paths, validate with `Path(user_input).resolve()` and verify the
result is under the expected base directory.

### Memory field access (cause of three production bugs)

`tags`, `memory_type`, `content_hash` and `created_at` are top-level attributes of the
`Memory` dataclass. `metadata` is a separate dict for custom key-value pairs only and
never holds standard fields.

```python
memory.tags               # correct
memory.memory_type or ''  # correct, with fallback

memory.metadata.get('tags', [])  # wrong, silently returns []
memory['tags']                   # wrong, raises AttributeError
```

The metadata form fails silently, which is why it reached production three times
(PRs #466, #467, #469 in v10.13.1): a broken REST filter returning zero results, tags
rendered character by character, and crashing prompt handlers. Flag any
`metadata.get('tags')` or `metadata.get('memory_type')` in review.

### External data parsers

Download and inspect a real sample before writing a parser or its tests. API docs and
project pages misdescribe structures often enough to matter, for example LoCoMo
observations are nested dicts rather than newline-separated strings.

### Directives to read before working

- [`.claude/directives/development-setup.md`](.claude/directives/development-setup.md) - editable install
- [`.claude/directives/pr-workflow.md`](.claude/directives/pr-workflow.md) - pre-PR checks
- [`.claude/directives/refactoring-checklist.md`](.claude/directives/refactoring-checklist.md) - refactoring safety
- [`.claude/directives/version-management.md`](.claude/directives/version-management.md) - release workflow (how)
- [`.claude/directives/release-cadence.md`](.claude/directives/release-cadence.md) - release batching (when)
- [`.claude/directives/extending.md`](.claude/directives/extending.md) - new MCP tool, storage backend, document loader
- [`.claude/directives/troubleshooting.md`](.claude/directives/troubleshooting.md) - symptom-to-fix table

Version bumps follow the documented release workflow, never by hand. It keeps
`pyproject.toml`, `_version.py`, CHANGELOG and the GitHub release in sync and writes
the release notes.

### Gotchas when changing things

- Dashboard (`web/static/`): the JS has no automated coverage. Verify in a browser and
  attach manual testing evidence or screenshots to the PR.
- Removing a feature, port or command: run `grep -r "<term>" docs/ README.md` and clean
  up references in the same PR. `scripts/ci/check_dead_refs.sh` catches new dead refs
  as check 6.8 of `pre_pr_check.sh` — **not** in CI, and it scans only `docs/` and
  `README.md`, so `CONTRIBUTING.md` and `SECURITY.md` are outside its reach. Wiring it
  into CI and widening the scan targets is issue #312.

### Extension points

Adding an MCP tool, a storage backend, or a document loader — including the
`readOnlyHint` OAuth-scope trap and the maintenance scripts:
[`.claude/directives/extending.md`](.claude/directives/extending.md). Renaming a tool is
a breaking change (the alias layer was removed in v11) — read that file first.

## Definition of Done

Work is not "done" until the relevant checks below have been run and pass. Report results plainly — paste the command output; if a check was skipped or failed, say so rather than claiming success.

**Every code change:**
- [ ] Relevant tests pass: `.venv/bin/pytest tests/<area>` (or `.venv/bin/pytest -k <pattern>`); run the full suite before a release.
- [ ] `bash scripts/pr/pre_pr_check.sh` passes — the MANDATORY pre-PR gate (blocks on security findings and health score <50, includes the log-injection check).
- [ ] New code stays within the complexity budget (A-B grade, complexity ≤8).
- [ ] User-provided values in `logger.*` calls are wrapped with `_sanitize_log_value()`.
- [ ] `Memory` fields are accessed by attribute (`memory.tags`), never `memory.metadata.get('tags')`.
- [ ] If a feature/port/command was removed: `grep -r "<term>" docs/ README.md` and clean up references in the same change.

**Dashboard changes (`web/static/`):** verified in a browser (no automated JS coverage) — include a screenshot or a note of what was exercised.

**Before a release / version bump:**
- [ ] CI is green on the target branch (GitHub Actions).
- [ ] Version bumped via the release workflow (never by hand) — keeps `pyproject.toml`, `_version.py`, CHANGELOG, and the GitHub release in sync.
- [ ] `site/index.html` version strings updated if MAJOR.MINOR changed (see the Release Workflow Checklist).

**After finishing a task:** save key learnings/decisions to the MCP Memory Server, tagged `mcp-memory-service` first (per the Auto-Save rule).

## Troubleshooting

Symptom-to-fix table (database locks, Cloudflare 401/403, MCP init timeouts, the venv
pre-commit trap) plus the heredoc permission-corruption warning:
[`.claude/directives/troubleshooting.md`](.claude/directives/troubleshooting.md).

## Performance Characteristics

**Key Metrics** (from production deployments):
- **5ms reads** - SQLite-Vec local storage
- **534,628x faster** - Global caching optimization (v8.26.0)
- **90% token reduction** - Consolidation via HTTP API vs MCP tools
- **85%+ trigger accuracy** - Natural memory triggers (v7.1.3+)
- **80-150ms** - Local ONNX quality scoring

## Documentation

The wiki is the comprehensive reference:
https://github.com/doobidoo/mcp-memory-service/wiki

**When to update each:**
- **CLAUDE.md** - Architecture changes, new patterns, development workflows
- **README.md** - New features, installation changes, user-facing updates
- **CHANGELOG.md** - Every version bump (via the release workflow)
- **site/index.html** - Landing page: MINOR/MAJOR releases only (title, og:title, hero badge, "What's New" cards, test count, release link). No manual publish step: merging to main triggers `.github/workflows/deploy-site.yml`, which deploys `site/` to Cloudflare Pages (mcpmemory.services). **GitHub Pages stays enabled and that is intentional**: it serves `docs/` from `main`, where `docs/index.html` is a 19-line stub redirecting to mcpmemory.services, so `doobidoo.github.io/mcp-memory-service/` keeps working for the links still pointing at it. Every push to main therefore fires a `pages build and deployment` run — expected, not a stray workflow. Never put version strings or content in `docs/index.html`; `site/index.html` is canonical.
- **Wiki** - Detailed guides, troubleshooting, tutorials

## Additional Resources

- **Storage Backends:** [`.claude/directives/storage-backends.md`](.claude/directives/storage-backends.md)
- **Hooks Configuration:** [`.claude/directives/hooks-configuration.md`](.claude/directives/hooks-configuration.md)
- **Quality System:** [`.claude/directives/quality-system-details.md`](.claude/directives/quality-system-details.md)
- **Consolidation:** [`.claude/directives/consolidation-details.md`](.claude/directives/consolidation-details.md)
- **Code Quality:** [`.claude/directives/code-quality-workflow.md`](.claude/directives/code-quality-workflow.md)

---

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
