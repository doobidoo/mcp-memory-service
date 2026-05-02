# Repository Housekeeping Audit & Fix — Design

**Date**: 2026-05-02
**Status**: Design approved, awaiting implementation plan
**Owner**: Henry
**Reference state**: v10.47.2, ~1,780 tests

## Goal

Detect and fix documentation drift across the repository (`docs/`, `README.md`, `CLAUDE.md`, `docs/index.html`) and the GitHub Wiki (separate `mcp-memory-service.wiki` repo), then add lightweight CI guards to prevent regression.

Drift categories in scope:

- **Version drift** — stale version refs, wrong test counts, outdated badges
- **Feature drift** — removed/renamed features still mentioned in active docs
- **Wiki↔repo drift** — wiki claims that contradict current code/config
- **Cross-doc inconsistency** — same fact stated differently across docs/wiki/CLAUDE.md
- **Link rot** — broken internal/external links (partial CI coverage today)
- **Structural drift** — orphaned docs, duplicate guides, archive candidates

## Non-Goals

- Wiki CI integration (separate-repo handling is deferred — file as follow-up issue)
- Cross-repo automated drift detection
- Spell-check / prose-style linting
- Adding new documentation pages (this work only fixes existing drift)
- Restructuring `docs/` hierarchy (orphan/dup cleanup is `defer-issue`, not inline)

## Approach

**Discovery-first, four-phase flow:**

1. **Audit** — scan all six categories, collect findings into single triage report. No code changes.
2. **User triage gate** — user marks each finding `ship-now` / `defer-issue` / `skip`.
3. **Execute fixes** — main-repo PR(s) for `ship-now` items, direct wiki commits for wiki items, GitHub issues for `defer-issue` items.
4. **Verify** — extended dead-ref check + new version-drift check pass; manual re-scan confirms zero remaining `ship-now` findings.

Discovery-first chosen over fix-as-you-go to enable batching related fixes (e.g. one PR for all version bumps across landing + README + wiki) instead of touching the same file multiple times.

## Audit Mechanics

| # | Category | How scanned | Output |
|---|----------|-------------|--------|
| a | **Version drift** | grep `docs/`, `README.md`, `docs/index.html`, `wiki/` for `v\d+\.\d+\.\d+` patterns + "test count" + badges; compare vs `_version.py` (10.47.2) and current test count | `file:line`, found version, expected |
| b | **Feature drift** | Cross-ref CHANGELOG `[Removed]` / `[Deprecated]` sections + recent rename PRs against current docs/wiki text | `file:line`, dead term, replacement |
| c | **Wiki↔repo drift** | Spot-check wiki claims against current code/config: ports, env vars, CLI commands, file paths, default values | wiki page:section, claim, current reality |
| d | **Cross-doc inconsistency** | Pick high-traffic facts (storage backends, ports, install commands, version count) and grep all docs/wiki — flag where they disagree | fact, locations + variants |
| e | **Link rot** | Run existing markdown-link-check on `docs/` + README locally; run separately on wiki dir | `file:line`, broken URL, status code |
| f | **Structural drift** | Find orphaned `.md` files (no incoming links from any other doc/README); flag duplicate guides (similar titles, similar content) | orphan files, dup-set candidates |

Each scan is a bash one-liner or short ad-hoc script (not committed — discovery-only). Findings collated into the report by hand.

## Severity Rubric

| Severity | Definition | Default disposition |
|----------|------------|---------------------|
| **🔴 Critical** | Wrong info that misleads users into broken state — wrong setup commands, removed features documented as available, broken install steps, broken links in install path | `ship-now` |
| **🟡 Major** | Stale but recoverable — wrong version on landing page, outdated test count, wiki claim contradicts current default but user can figure it out | `ship-now` |
| **🟢 Minor** | Cosmetic — orphaned doc with no harm, duplicate guide where both still accurate, broken external link to non-critical resource | `defer-issue` |
| **⚪ Noise** | Intentional historical refs (chromadb in `archive/`), version refs in CHANGELOG, migration guides | `skip` |

User can override per-finding during triage gate.

## Triage Report Format

Path: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

```markdown
# Housekeeping Audit — 2026-05-02

**Scope**: docs/, README.md, CLAUDE.md, docs/index.html, wiki/
**Reference state**: v10.47.2, ~1,780 tests
**Categories scanned**: version, feature, wiki↔repo, cross-doc, links, structural

## Summary
- 🔴 Critical: N findings
- 🟡 Major:    N findings
- 🟢 Minor:    N findings
- ⚪ Noise:    N findings (excluded)

## Findings

### F-001 🔴 Critical · version drift · docs/index.html:42
**Current**: `<span class="version">v10.13.0</span>`
**Expected**: `v10.47.2`
**Fix**: edit + re-publish to here.now (`merry-realm-j835`)
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip

### F-002 🟡 Major · wiki↔repo · wiki/04-Advanced-Configuration.md
**Current**: claims `MCP_HTTP_PORT` defaults to 8443
**Expected**: defaults to 8000 (per src/mcp_memory_service/config.py)
**Fix**: wiki direct commit
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
...
```

User edits checkboxes during triage gate. The implementer parses decisions and executes accordingly.

After Phase 4 ships, append a `## Resolution` section noting what shipped where (PR/commit/issue links).

## Fix Delivery

**Main-repo PRs:**

- **PR-1: `chore/housekeeping-fixes`** — all `ship-now` fixes for `docs/`, `README.md`, `CLAUDE.md`, `docs/index.html` plus tooling extensions (b: extended `check_dead_refs.sh`, c: new `check_versions.sh` + workflow wire-up).
- **PR-2 (only if PR-1 grows past ~30 files or mixes concerns badly)**: split off the structural/cosmetic batch.

No version bump → do **not** invoke `github-release-manager`. Use `commit-commands:commit-push-pr` (or equivalent) to ship the PR.

**Wiki direct commits:**

- One commit per logical group (e.g. "fix: port references in 04-Advanced-Configuration", "fix: stale version refs across guides").
- Push to `mcp-memory-service.wiki` `main` directly (gollum, no PR mechanism).
- After landing-page version bump in main repo, also re-publish to here.now per CLAUDE.md:
  ```bash
  cd docs && ~/.agents/skills/here-now/scripts/publish.sh index.html --slug merry-realm-j835
  ```

**Issues filed:**

- One GitHub issue per `defer-issue` finding (or grouped if related).
- Labels: `documentation`, `housekeeping`.
- Each issue links back to the audit report via permalink.

## Tooling Extensions

### Extension b — extend `scripts/ci/check_dead_refs.sh`

Add new `DEAD_REFS` entries for any feature/command/env-var renames discovered during audit. Add to `SOFT_REF_ALLOWLIST` only when a doc legitimately needs to retain a historical pointer. No structural change to the script — data-only updates.

### Extension c — new `scripts/ci/check_versions.sh`

Pseudocode (real script lives in PR-1):

```text
1. Read canonical version from src/mcp_memory_service/_version.py
   (use grep + sed, or `python -c "from mcp_memory_service._version import __version__; print(__version__)"`)
2. SCAN_TARGETS = docs/, README.md
3. EXCLUDE_PATHS = docs/archive, docs/legacy, docs/plans, docs/migrations, CHANGELOG
4. For each *.md and docs/index.html under SCAN_TARGETS:
   - grep for /v\d+\.\d+\.\d+/
   - filter out EXCLUDE_PATHS
   - compare each match against CANONICAL via semver compare
   - if any match < CANONICAL, record as drift finding
5. Exit 1 if any drift found, 0 otherwise
6. Print drift report in same format as check_dead_refs.sh
```

**Scope guards** (avoid noise):

- Skip CHANGELOG (intentionally references all historical versions)
- Skip `archive/`, `legacy/`, `plans/`, `migrations/`
- Skip code-block content tagged with explicit version (e.g. install snippets pinning a specific version are OK)
- Optional post-rollout tunable: only fail on `docs/index.html` and `README.md`; warn-only elsewhere

**Wire-up** in `.github/workflows/docs-check.yml`:

```yaml
  version-drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Check for version drift
        run: bash scripts/ci/check_versions.sh
```

**Important**: tooling ships in PR-1 alongside the fixes those tools enforce. Do not ship enforcement without first cleaning all violations — CI would fail immediately.

## Validation (Phase 4)

PR-1 description must include this checklist:

- [ ] `bash scripts/ci/check_dead_refs.sh` exits 0 locally
- [ ] `bash scripts/ci/check_versions.sh` exits 0 locally
- [ ] Markdown link-check (via `gaurav-nelson/github-action-markdown-link-check`) passes on PR CI
- [ ] Manual re-scan: re-run audit one-liners from Phase 1 → confirm no `ship-now` findings remain
- [ ] Wiki repo: each direct commit pushed; `git log` shows expected fixes
- [ ] Landing-page (`docs/index.html`) re-published to here.now (`merry-realm-j835`)
- [ ] Issues filed for each `defer-issue` finding; linked to audit report
- [ ] Audit report updated with `## Resolution` section

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Audit produces too many findings to triage in one session | Severity rubric biases toward `defer-issue` for minor; cap PR-1 at ~30 files, split if needed |
| Version-drift script false-positives on intentional pinned-version snippets | Exclude code-block-internal version pins; tunable warn-only mode for non-landing-page paths |
| Wiki commits land but landing page not republished | Phase 4 checklist explicit step; CLAUDE.md already documents the `here-now` republish command |
| New CI check fails immediately on existing repo state | Tooling ships in same PR as cleanup; verify locally before push |
| Subjective "wiki↔repo drift" findings (what's truly contradictory vs out-of-date phrasing) | Spot-check a fixed list of high-traffic facts (ports, env vars, install commands, defaults) — not exhaustive prose review |

## Open Questions

None at design stage. All Q1–Q4 from brainstorming resolved:

- Q1: Audit + fix (B), tooling extension where cheap
- Q2: All six drift categories
- Q3: Severity-gated, 1–2 main-repo PRs + direct wiki commits + issues for low-pri
- Q4: Extend `check_dead_refs.sh` + add `check_versions.sh`; defer wiki CI

## Next Step

User reviews this spec → invoke `superpowers:writing-plans` to produce phase-by-phase implementation plan.
