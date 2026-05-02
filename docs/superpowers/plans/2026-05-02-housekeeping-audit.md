# Repository Housekeeping Audit & Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect documentation drift across `docs/`, `README.md`, `CLAUDE.md`, `docs/index.html`, and the `mcp-memory-service.wiki` repo, then fix critical/major findings via 1-2 main-repo PRs + direct wiki commits, and add a new version-drift CI check to prevent regression.

**Architecture:** Discovery-first, four-phase flow — (1) audit all six drift categories into a single triage report, (2) user marks each finding `ship-now` / `defer-issue` / `skip`, (3) execute fixes with bundled tooling extensions, (4) verify via extended dead-ref check + new version-drift check + manual re-scan.

**Tech Stack:** bash scripts, GitHub Actions (`gaurav-nelson/github-action-markdown-link-check`), markdown, GitHub CLI (`gh`), git, Python (`_version.py` parsing).

**Reference state:** v10.47.2, ~1,780 tests, 40+ wiki pages, 60+ doc files.

**Spec:** [`docs/superpowers/specs/2026-05-02-housekeeping-audit-design.md`](../specs/2026-05-02-housekeeping-audit-design.md)

---

## Phase 1 — Audit

Goal: produce `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md` with all findings classified by severity. No code changes in this phase.

Each task in this phase is a discovery scan. Findings from all tasks accumulate into the single report file. The report file is created in Task 1.0 and appended to by each subsequent task.

### Task 1.0: Bootstrap audit report

**Files:**
- Create: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Determine canonical reference state**

```bash
# Canonical version
grep -E '__version__\s*=' src/mcp_memory_service/_version.py

# Test count (best-effort approximation)
grep -rE '^\s*def test_' tests/ --include='*.py' | wc -l
```

Record both values. Use the canonical version from `_version.py` for all version comparisons.

- [ ] **Step 2: Create report skeleton**

Write `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`:

```markdown
# Housekeeping Audit — 2026-05-02

**Scope**: docs/, README.md, CLAUDE.md, docs/index.html, mcp-memory-service.wiki/
**Reference state**: v<CANONICAL>, ~<TEST_COUNT> tests
**Categories scanned**: version, feature, wiki↔repo, cross-doc, links, structural

## Summary
- 🔴 Critical: 0 findings
- 🟡 Major:    0 findings
- 🟢 Minor:    0 findings
- ⚪ Noise:    0 findings (excluded)

## Findings

<!-- Findings appended below by audit tasks. ID format: F-NNN. -->
```

Replace `<CANONICAL>` and `<TEST_COUNT>` with the values from Step 1. Do not commit yet.

### Task 1.1: Version drift scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Grep for hardcoded version refs**

```bash
# Repo: docs/, README.md, CLAUDE.md, docs/index.html
grep -rEn 'v[0-9]+\.[0-9]+\.[0-9]+' \
  docs/ README.md CLAUDE.md \
  --include='*.md' --include='*.html' \
  | grep -vE '/(archive|legacy|plans|migrations)/|CHANGELOG'

# Wiki: separate repo
grep -rEn 'v[0-9]+\.[0-9]+\.[0-9]+' \
  /Users/hkr/Repositories/mcp-memory-service.wiki/ \
  --include='*.md' \
  | grep -vE '/archive/'
```

- [ ] **Step 2: Grep for test-count refs**

```bash
grep -rEn '[0-9],?[0-9]{3}\s*tests' docs/ README.md CLAUDE.md --include='*.md' \
  | grep -v CHANGELOG

grep -rEn '[0-9],?[0-9]{3}\s*tests' \
  /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
```

- [ ] **Step 3: Compare each match against canonical**

For each match where the version is older than canonical (semver comparison), record it. Skip matches inside CHANGELOG, archive/, legacy/, plans/, migrations/. Skip matches that are intentionally pinned (e.g. `pip install mcp-memory-service==X.Y.Z` where X.Y.Z is an old known-good version — judgment call).

- [ ] **Step 4: Append findings to report**

Append each finding under `## Findings` using this format:

```markdown
### F-001 🟡 Major · version drift · docs/index.html:42
**Current**: `v10.13.0` (in `<span class="version">`)
**Expected**: `v10.47.2`
**Fix**: edit + re-publish to here.now (`merry-realm-j835`)
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
```

Use F-NNN sequential IDs starting at F-001. Severity per rubric in spec.

### Task 1.2: Feature drift scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Build dead-feature term list**

```bash
# Pull all [Removed] / [Deprecated] sections from CHANGELOG (last 12 months)
grep -nE '^### \[(Removed|Deprecated)\]' CHANGELOG.md | head -40

# Inspect each section, extract feature names, env vars, ports, CLI flags
# that have been removed or deprecated
```

Build a list `DEAD_TERMS=(...)` of strings to grep for. Include obvious ones from the existing `check_dead_refs.sh` `DEAD_REFS` array as a seed.

- [ ] **Step 2: Scan active docs + wiki for dead terms**

For each term in `DEAD_TERMS`:

```bash
TERM='--storage-backend chromadb'  # example

# Repo
grep -rln "$TERM" docs/ README.md CLAUDE.md --include='*.md' --include='*.html' \
  | grep -vE '/(archive|legacy|plans|migrations)/|CHANGELOG'

# Wiki
grep -rln "$TERM" /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md' \
  | grep -v /archive/
```

- [ ] **Step 3: Classify each match**

Severity:
- 🔴 Critical: dead term in install/setup path (would break user)
- 🟡 Major: dead term in feature description (misleading but recoverable)
- 🟢 Minor: dead term in passing reference / older guide

- [ ] **Step 4: Append findings to report**

Same format as Task 1.1, with `feature drift` as category. For each finding, also include the **Replacement** field naming the current term.

### Task 1.3: Wiki↔repo drift scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Define spot-check fact list**

Fixed list to verify against current code/config:

| Fact | Source of truth |
|------|----------------|
| Default HTTP port | `src/mcp_memory_service/config.py` (`HTTP_PORT`) |
| Default storage backend | `src/mcp_memory_service/config.py` (`STORAGE_BACKEND`) |
| Default consolidation schedule | `src/mcp_memory_service/config.py` (consolidation section) |
| MCP tool count | `src/mcp_memory_service/server_impl.py` (count `@self.server.tool()` decorators) |
| Required Python version | `pyproject.toml` (`requires-python`) |
| Install command | `README.md` (canonical install section) |
| Cloudflare env var names | `src/mcp_memory_service/config.py` (`CLOUDFLARE_*`) |
| OAuth env var names | `src/mcp_memory_service/config.py` + `web/oauth/` |
| MCP server entrypoint | `src/mcp_memory_service/server.py` (`__main__`) |

- [ ] **Step 2: Read each source-of-truth value**

Use `grep` / `Read` tool to capture the current value for each fact above.

- [ ] **Step 3: Spot-check each wiki page for those facts**

```bash
# For each fact, grep wiki pages for old/wrong values
grep -rEn 'port\s*[:=]\s*[0-9]+' /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
grep -rEn 'MCP_HTTP_PORT' /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
# ... etc
```

- [ ] **Step 4: Append findings to report**

Format:

```markdown
### F-NNN 🟡 Major · wiki↔repo · 04-Advanced-Configuration.md:L
**Current**: claims `MCP_HTTP_PORT` defaults to 8443
**Expected**: defaults to 8000 (per src/mcp_memory_service/config.py)
**Fix**: wiki direct commit
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
```

### Task 1.4: Cross-doc inconsistency scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Define high-traffic facts**

Same fact list as Task 1.3, but the question is now: **does each doc/page state the same value?** Not: does each match the code (that's Task 1.3).

- [ ] **Step 2: Grep across ALL doc surfaces for each fact**

```bash
FACT='HTTP_PORT'
grep -rln "$FACT" docs/ README.md CLAUDE.md /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
# Read each match, record the stated value
```

- [ ] **Step 3: Flag where stated values disagree**

If `README.md` says port 8000 but `wiki/Hybrid_Setup_Configuration.md` says port 8443, that's a cross-doc inconsistency. Severity is usually 🟡 Major. (If both are wrong vs code, Task 1.3 already caught that — this catches "different docs say different things".)

- [ ] **Step 4: Append findings to report**

Format includes a `**Locations**` field listing all variants:

```markdown
### F-NNN 🟡 Major · cross-doc · MCP_HTTP_PORT default
**Locations**:
  - README.md:120 → `8000`
  - wiki/04-Advanced-Configuration.md:55 → `8443`
  - docs/http-server-management.md:30 → `8000`
**Expected** (per code): `8000`
**Fix**: align wiki to `8000` (covered by F-002 if this overlaps)
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
```

If a finding overlaps with Task 1.3, cross-reference and don't double-count toward summary.

### Task 1.5: Link rot scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Run markdown-link-check on repo docs**

Use the same tool the CI workflow uses, locally via npx:

```bash
npx markdown-link-check -c .mlc-config.json -q docs/**/*.md README.md 2>&1 | tee /tmp/linkcheck-repo.log
```

If `.mlc-config.json` doesn't exist, inspect `.github/workflows/docs-check.yml` for config.

- [ ] **Step 2: Run markdown-link-check on wiki**

```bash
cd /Users/hkr/Repositories/mcp-memory-service.wiki
npx markdown-link-check -q *.md 2>&1 | tee /tmp/linkcheck-wiki.log
cd -
```

- [ ] **Step 3: Parse results, classify**

For each broken link:
- 🔴 Critical: link in install/setup section, broken to internal repo file
- 🟡 Major: broken internal link (wiki→wiki, doc→doc)
- 🟢 Minor: broken external link to non-critical resource

Skip known-flaky external sites (e.g. `glama.ai` per existing `.mlc-config.json` ignore list — see [project_glama_link_ignore.md] reference in MEMORY.md).

- [ ] **Step 4: Append findings to report**

```markdown
### F-NNN 🟡 Major · link rot · docs/integration/claude-desktop.md:80
**Broken URL**: `./docs/old-setup.md` (404 — file does not exist)
**Fix**: replace with `./first-time-setup.md` or remove
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
```

### Task 1.6: Structural drift scan

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Find orphaned `.md` files in repo**

```bash
# List all docs
find docs/ -name '*.md' -not -path '*/archive/*' -not -path '*/legacy/*' > /tmp/all_docs.txt

# For each doc, grep all OTHER docs + README + CLAUDE.md for incoming links
# Flag any doc that has zero incoming links
```

Use a small bash loop, or a one-off Python script. Don't commit the script.

- [ ] **Step 2: Find orphaned `.md` files in wiki**

Wiki linking is via gollum: `[[Page Name]]` syntax + plain markdown links. Check both forms.

```bash
# Pages potentially referenced via gollum syntax
grep -rEn '\[\[[^]]+\]\]' /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
# Plus plain markdown links
grep -rEn '\]\([^)]+\.md\)' /Users/hkr/Repositories/mcp-memory-service.wiki/ --include='*.md'
```

Flag wiki pages with no inbound references from `Home.md` or any other page.

- [ ] **Step 3: Find duplicate guides**

Group `.md` files by similar titles or filenames (e.g. `Cloudflare-Backup-Sync-Setup.md` + `Cloudflare-Based-Multi-Machine-Sync.md` + `quick-setup-cloudflare-dual-environment.md` may overlap). Open each group and judge whether they're redundant.

- [ ] **Step 4: Append findings to report**

Always 🟢 Minor for orphans/dups (default → `defer-issue`).

```markdown
### F-NNN 🟢 Minor · structural · docs/enhancement-roadmap-issue-14.md
**Issue**: orphaned (no incoming links from any active doc)
**Fix**: archive (move to docs/archive/) OR add inbound link if still relevant
**Disposition**: [ ] ship-now  [ ] defer-issue  [ ] skip
```

### Task 1.7: Finalize report + commit

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Update Summary counts**

Recount findings by severity, update the `## Summary` block at top of report.

- [ ] **Step 2: Sort findings**

Sort `## Findings` section: Critical first, then Major, then Minor, then Noise. Within each severity, group by category for readability.

- [ ] **Step 3: Commit report**

```bash
git add docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md
git commit -m "docs(audit): housekeeping audit findings — $(date +%Y-%m-%d)"
```

- [ ] **Step 4: Push to feature branch**

```bash
git checkout -b chore/housekeeping-fixes
git push -u origin chore/housekeeping-fixes
```

(Or if main has been the working branch up to now, branch off here so report + fixes ship together in PR-1.)

---

## Phase 2 — User triage gate

### Task 2.0: User reviews + dispositions findings

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Present report to user**

Tell user: "Audit complete. Report at `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md` with N total findings. Please mark each finding's `**Disposition**` checkbox: `ship-now`, `defer-issue`, or `skip`. When done, tell me to proceed."

- [ ] **Step 2: Wait for user**

User edits checkboxes in the report file. Do not advance until user signals done.

- [ ] **Step 3: Parse decisions**

Re-read the report, extract:
- `ship_now_repo`: list of findings with disposition = ship-now AND location is `docs/`, `README.md`, `CLAUDE.md`, or `docs/index.html`
- `ship_now_wiki`: list of findings with disposition = ship-now AND location is in wiki repo
- `defer_issue`: list of findings to file as GitHub issues
- `skipped`: list of findings to ignore

- [ ] **Step 4: Commit dispositions**

```bash
git add docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md
git commit -m "docs(audit): user dispositions for housekeeping findings"
```

---

## Phase 3 — Execute fixes

Order: tooling first (TDD), then repo doc fixes, then wiki commits, then issues, then CI wire-up + landing page republish, then PR.

### Task 3.0: TDD `check_versions.sh` — fixture + failing test

**Files:**
- Create: `tests/ci/test_check_versions.bats`
- Create: `tests/ci/fixtures/version_drift_pos.md` (positive case — has stale version)
- Create: `tests/ci/fixtures/version_drift_neg.md` (negative case — only canonical version)

- [ ] **Step 1: Verify bats is available**

```bash
which bats || echo "bats not installed; install via: brew install bats-core"
```

If bats unavailable, use a plain bash test harness in `tests/ci/test_check_versions.sh` instead. Adapt the test format below accordingly. Document the chosen harness in a comment at the top of the test file.

- [ ] **Step 2: Create fixture files**

`tests/ci/fixtures/version_drift_pos.md`:

```markdown
# Test Doc — Stale Version

This doc references v9.0.0 which is way older than canonical.

Install: `pip install mcp-memory-service==v9.0.0`
```

`tests/ci/fixtures/version_drift_neg.md`:

```markdown
# Test Doc — Current Only

References to v999.999.999 (above canonical) and CHANGELOG-style historical
versions like v1.0.0 should be allowed when in excluded paths only.

This file in scan path mentions only the canonical version.
```

(The negative fixture file should NOT have any old version references when used with the script's default scan paths. Use it as a baseline that produces zero drift.)

- [ ] **Step 3: Write failing test**

`tests/ci/test_check_versions.bats`:

```bash
#!/usr/bin/env bats

setup() {
  REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
  SCRIPT="$REPO_ROOT/scripts/ci/check_versions.sh"
  FIXTURES="$BATS_TEST_DIRNAME/fixtures"
}

@test "exits 0 when no drift" {
  # Point script at a clean fixture dir (override SCAN_TARGETS via env)
  MCS_VERSION_SCAN_TARGETS="$FIXTURES/version_drift_neg.md" \
    bash "$SCRIPT"
  [ "$status" -eq 0 ]
}

@test "exits 1 when drift found" {
  run env MCS_VERSION_SCAN_TARGETS="$FIXTURES/version_drift_pos.md" \
    bash "$SCRIPT"
  [ "$status" -eq 1 ]
  [[ "$output" == *"v9.0.0"* ]]
}

@test "skips CHANGELOG even with old refs" {
  echo "This file references v1.0.0" > "$BATS_TMPDIR/CHANGELOG.md"
  run env MCS_VERSION_SCAN_TARGETS="$BATS_TMPDIR/CHANGELOG.md" \
    bash "$SCRIPT"
  [ "$status" -eq 0 ]
}
```

- [ ] **Step 4: Run test to verify it fails**

```bash
bats tests/ci/test_check_versions.bats
```

Expected: FAIL with "script not found: scripts/ci/check_versions.sh".

- [ ] **Step 5: Commit**

```bash
git add tests/ci/test_check_versions.bats tests/ci/fixtures/
git commit -m "test(ci): add failing tests for check_versions.sh"
```

### Task 3.1: Implement `check_versions.sh`

**Files:**
- Create: `scripts/ci/check_versions.sh`

- [ ] **Step 1: Write minimal implementation**

`scripts/ci/check_versions.sh`:

```bash
#!/bin/bash
#
# Version Drift Check
#
# Reads canonical version from src/mcp_memory_service/_version.py.
# Greps SCAN_TARGETS for hardcoded vN.N.N refs.
# Fails if any non-excluded match references a version older than canonical.
#
# Override SCAN_TARGETS for testing via MCS_VERSION_SCAN_TARGETS env var.
#
# Exit codes:
#   0 - No drift found
#   1 - Drift found

set -uo pipefail

# Locate canonical version. Use grep+sed (no Python import — script runs in CI
# without venv).
VERSION_FILE="${MCS_VERSION_FILE:-src/mcp_memory_service/_version.py}"
if [ ! -f "$VERSION_FILE" ]; then
  echo "❌ Version file not found: $VERSION_FILE"
  exit 1
fi

CANONICAL=$(grep -E '^__version__\s*=' "$VERSION_FILE" \
  | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
if [ -z "$CANONICAL" ]; then
  echo "❌ Could not parse __version__ from $VERSION_FILE"
  exit 1
fi

# Default scan targets; overridable for tests.
if [ -n "${MCS_VERSION_SCAN_TARGETS:-}" ]; then
  # Space-separated list of paths/files
  read -ra SCAN_TARGETS <<< "$MCS_VERSION_SCAN_TARGETS"
else
  SCAN_TARGETS=("docs/" "README.md" "CLAUDE.md")
fi

EXCLUDE_PATHS=(
  "docs/archive"
  "docs/legacy"
  "docs/plans"
  "docs/migrations"
  "CHANGELOG"
)

# Semver compare: returns 0 if $1 < $2, 1 otherwise.
older_than() {
  local a="$1" b="$2"
  [ "$a" = "$b" ] && return 1
  local lower
  lower=$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -1)
  [ "$lower" = "$a" ]
}

FOUND=0
declare -a DRIFT_LINES

for target in "${SCAN_TARGETS[@]}"; do
  [ -e "$target" ] || continue
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    file=$(echo "$line" | cut -d: -f1)
    skip=false
    for excl in "${EXCLUDE_PATHS[@]}"; do
      if [[ "$file" == *"$excl"* ]]; then
        skip=true
        break
      fi
    done
    [ "$skip" = true ] && continue

    # Extract version from line
    version=$(echo "$line" | grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/^v//')
    [ -z "$version" ] && continue

    if older_than "$version" "$CANONICAL"; then
      DRIFT_LINES+=("$line  →  expected v$CANONICAL (or excluded path)")
      FOUND=1
    fi
  done < <(grep -rEn 'v?[0-9]+\.[0-9]+\.[0-9]+' "$target" --include='*.md' --include='*.html' 2>/dev/null || true)
done

if [ $FOUND -eq 0 ]; then
  echo "✅ No version drift found (canonical: v$CANONICAL)"
  exit 0
fi

echo "❌ Version drift detected (canonical: v$CANONICAL):"
for line in "${DRIFT_LINES[@]}"; do
  echo "   $line"
done
echo ""
echo "Fix: update each occurrence to v$CANONICAL or move under an excluded path."
exit 1
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/ci/check_versions.sh
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
bats tests/ci/test_check_versions.bats
```

Expected: 3/3 PASS.

- [ ] **Step 4: Run script against real repo (sanity check)**

```bash
bash scripts/ci/check_versions.sh
```

Expected: depends on Phase 3.3 fixes — for now it likely reports drift on the real repo. That's expected; the next tasks fix it. Note the violations for cross-checking against Phase 1 findings.

- [ ] **Step 5: Commit**

```bash
git add scripts/ci/check_versions.sh
git commit -m "feat(ci): add check_versions.sh — version drift detection"
```

### Task 3.2: Extend `check_dead_refs.sh`

**Files:**
- Modify: `scripts/ci/check_dead_refs.sh`

- [ ] **Step 1: Identify new dead-term findings from audit**

From the audit report, list every `feature drift` finding where the disposition is `ship-now` AND the term should be permanently blocked from re-entering active docs. These go into `DEAD_REFS`.

- [ ] **Step 2: Edit `DEAD_REFS` array**

Add new entries to the `DEAD_REFS=(...)` block in `scripts/ci/check_dead_refs.sh`. Keep alphabetical or grouped-by-feature ordering. One entry per line. Comment if the term needs context.

Example:

```bash
DEAD_REFS=(
  "--storage-backend chromadb"
  "storage-backend chromadb"
  ...
  # Added 2026-05-02 housekeeping audit
  "MCP_OLD_FEATURE_FLAG"
  "--legacy-flag-name"
)
```

- [ ] **Step 3: Update `SOFT_REF_ALLOWLIST` only if needed**

Only add a path to `SOFT_REF_ALLOWLIST` if a doc legitimately needs to retain a historical pointer (per existing rule 5 from issue #713). Do not add new entries casually — every new entry weakens the script.

- [ ] **Step 4: Run script to verify**

```bash
bash scripts/ci/check_dead_refs.sh
```

Expected: still passes against current repo state (because Phase 3.3 hasn't yet fixed the docs that contain those terms). If it fails, that's expected — proceed to 3.3 to fix the violations, then re-run.

- [ ] **Step 5: Commit**

```bash
git add scripts/ci/check_dead_refs.sh
git commit -m "ci(dead-refs): add new dead terms from housekeeping audit"
```

### Task 3.3: Apply main-repo doc fixes

**Files:**
- Modify: any of `docs/`, `README.md`, `CLAUDE.md`, `docs/index.html` per audit findings

- [ ] **Step 1: Group findings by file**

From the audit report, collect all `ship_now_repo` findings grouped by target file. Each file should be edited exactly once (to avoid touching the same file in multiple commits).

- [ ] **Step 2: Apply fixes file-by-file**

For each file, use the `Edit` tool with the **Current** → **Expected** mapping from each finding. Verify edits match the finding's stated location.

- [ ] **Step 3: Verify CI checks pass locally**

After all fixes:

```bash
bash scripts/ci/check_dead_refs.sh
bash scripts/ci/check_versions.sh
```

Both must exit 0. If not, the fixes missed something — re-read findings, re-edit, re-run.

- [ ] **Step 4: Re-run audit one-liners (subset)**

Re-run the version-drift grep from Task 1.1 Step 1, and the feature-drift greps from Task 1.2 Step 2, against the repo. Compare against the audit report — every `ship-now` finding's match should no longer appear.

- [ ] **Step 5: Commit per logical group**

If multiple files were touched for the same finding category, group them in one commit. Example:

```bash
git add docs/index.html docs/README.md README.md CLAUDE.md
git commit -m "docs: bump stale version refs to v10.47.2"

git add docs/integration/*.md
git commit -m "docs: remove references to retired CLI flags (housekeeping audit)"
```

Use as many commits as makes sense — each commit should be one logical fix group.

### Task 3.4: Apply wiki direct commits

**Files:**
- Modify: files under `/Users/hkr/Repositories/mcp-memory-service.wiki/`

- [ ] **Step 1: Group wiki findings by file**

From the audit report, collect all `ship_now_wiki` findings grouped by wiki page.

- [ ] **Step 2: Apply fixes per wiki file**

```bash
cd /Users/hkr/Repositories/mcp-memory-service.wiki
git pull --ff-only  # ensure up-to-date
```

Use the `Edit` tool for each wiki file. Wiki uses gollum format — most files are plain markdown, page links use `[[Page Name]]` syntax.

- [ ] **Step 3: Commit per logical group**

```bash
cd /Users/hkr/Repositories/mcp-memory-service.wiki
git add 04-Advanced-Configuration.md
git commit -m "fix: correct MCP_HTTP_PORT default to 8000 (housekeeping audit)"

git add Hybrid_Setup_Configuration.md Cloudflare-Backup-Sync-Setup.md
git commit -m "docs: align Cloudflare guide version refs to v10.47.2"
```

- [ ] **Step 4: Push wiki commits**

```bash
cd /Users/hkr/Repositories/mcp-memory-service.wiki
git push origin main  # or master, check default branch
cd -
```

- [ ] **Step 5: Verify wiki online**

Open `https://github.com/doobidoo/mcp-memory-service/wiki` in browser, confirm one of the changed pages reflects the fix.

### Task 3.5: File GitHub issues for `defer-issue` findings

**Files:** none (GitHub issues, external state)

- [ ] **Step 1: Group `defer-issue` findings**

Group related deferred findings (e.g. all orphaned docs → one umbrella issue; all minor link rot in old guides → one issue). Standalone findings get their own issue.

- [ ] **Step 2: File issues via gh CLI**

For each issue group:

```bash
gh issue create \
  --title "Housekeeping: <short description>" \
  --label documentation,housekeeping \
  --body "$(cat <<'EOF'
Discovered during housekeeping audit on 2026-05-02.

**Source**: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md` finding F-NNN

**Issue**: <restate finding>

**Fix sketch**: <restate fix>

**Severity**: 🟢 Minor (deferred from audit)
EOF
)"
```

Record issue numbers as they're created.

- [ ] **Step 3: Update audit report with issue links**

For each `defer-issue` finding in the report, append `**Issue**: #NNN` to the finding block.

- [ ] **Step 4: Commit report update**

```bash
git add docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md
git commit -m "docs(audit): link deferred findings to GitHub issues"
```

### Task 3.6: Wire `check_versions.sh` into CI workflow

**Files:**
- Modify: `.github/workflows/docs-check.yml`

- [ ] **Step 1: Add new job**

Edit `.github/workflows/docs-check.yml`. After the existing `dead-ref-check` job, add:

```yaml
  version-drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Check for version drift
        run: bash scripts/ci/check_versions.sh
```

- [ ] **Step 2: Verify YAML syntax**

```bash
# Use yq if available, else cat to eyeball
yq eval '.jobs | keys' .github/workflows/docs-check.yml 2>/dev/null \
  || cat .github/workflows/docs-check.yml
```

Expected: `link-check`, `dead-ref-check`, `version-drift-check` listed.

- [ ] **Step 3: Run script one more time as final sanity**

```bash
bash scripts/ci/check_versions.sh
```

Expected: exit 0 (because Task 3.3 fixed all violations).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docs-check.yml
git commit -m "ci(docs): add version drift check to docs-check workflow"
```

### Task 3.7: Republish landing page to here.now

**Files:** none (external state — here.now publish)

- [ ] **Step 1: Confirm landing page version is current**

```bash
grep -E 'v[0-9]+\.[0-9]+\.[0-9]+' docs/index.html
```

Should show v10.47.2 (or current canonical) after Task 3.3.

- [ ] **Step 2: Republish**

```bash
cd docs && ~/.agents/skills/here-now/scripts/publish.sh index.html --slug merry-realm-j835
cd -
```

- [ ] **Step 3: Verify**

Open the here.now URL in browser, confirm version badge shows current canonical.

If `~/.agents/skills/here-now/scripts/publish.sh` is not present, ask user to republish manually and skip this task. Do NOT block on this — note in PR description.

### Task 3.8: Open PR-1

**Files:** none (PR creation)

- [ ] **Step 1: Push branch**

```bash
git push origin chore/housekeeping-fixes
```

- [ ] **Step 2: Create PR**

```bash
gh pr create \
  --title "chore: repository housekeeping audit + drift fixes" \
  --body "$(cat <<'EOF'
## Summary

Discovery-first housekeeping audit + execution. Fixes documentation drift across `docs/`, `README.md`, `CLAUDE.md`, `docs/index.html`. Adds `check_versions.sh` CI guard. Extends `check_dead_refs.sh` with new dead-term findings.

Wiki fixes shipped as direct commits (gollum, separate repo). Minor findings filed as GitHub issues for follow-up.

## Audit Report

See `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md` for full findings, dispositions, and resolution mapping.

## Changes

- 🔴 N critical drift fixes
- 🟡 N major drift fixes
- 🟢 N minor findings → GitHub issues #...
- ✅ New: `scripts/ci/check_versions.sh` + workflow wire-up
- ✅ Extended: `scripts/ci/check_dead_refs.sh` (data-only)

## Validation

- [ ] `bash scripts/ci/check_dead_refs.sh` exits 0
- [ ] `bash scripts/ci/check_versions.sh` exits 0
- [ ] Markdown link-check passes
- [ ] Wiki direct commits pushed to mcp-memory-service.wiki
- [ ] Landing page republished to here.now (`merry-realm-j835`)
- [ ] All `defer-issue` findings have linked GitHub issues
- [ ] Audit report updated with `## Resolution` section

## Test Plan

- [ ] Run `bash scripts/ci/check_versions.sh` locally → exits 0
- [ ] Run `bats tests/ci/test_check_versions.bats` → 3/3 pass
- [ ] CI green on docs-check.yml workflow

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Note PR URL**

Record the PR URL for Phase 4 verification.

---

## Phase 4 — Validation

### Task 4.0: Run all checks + finalize report

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`

- [ ] **Step 1: Run all CI scripts locally**

```bash
bash scripts/ci/check_dead_refs.sh && echo "✅ dead refs OK"
bash scripts/ci/check_versions.sh && echo "✅ versions OK"
bats tests/ci/test_check_versions.bats && echo "✅ version test OK"
```

All three must exit 0.

- [ ] **Step 2: Watch CI on PR**

```bash
gh pr checks <PR_NUMBER> --watch
```

Wait for `link-check`, `dead-ref-check`, `version-drift-check` jobs → all green.

- [ ] **Step 3: Manual re-scan**

Re-run the high-signal audit one-liners from Task 1.1 Step 1, Task 1.2 Step 2, and confirm zero `ship-now` findings remain in active docs.

- [ ] **Step 4: Verify wiki**

```bash
cd /Users/hkr/Repositories/mcp-memory-service.wiki
git log --oneline -10
cd -
```

Confirm expected commits are present.

- [ ] **Step 5: Verify landing page**

Open here.now URL (`merry-realm-j835` slug). Confirm version badge is canonical.

- [ ] **Step 6: Append Resolution section to audit report**

Append at end of `docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md`:

```markdown
---

## Resolution

**Status**: Complete (YYYY-MM-DD)

### Shipped (PR-1: #NNN)
- F-001, F-002, ... → docs/ + README.md fixes
- check_versions.sh + workflow wired
- check_dead_refs.sh extended

### Wiki commits (mcp-memory-service.wiki)
- F-NNN → 04-Advanced-Configuration.md (commit SHA)
- F-NNN → ...

### Deferred to issues
- F-NNN → #NNN
- F-NNN → #NNN

### Skipped (intentional)
- F-NNN, F-NNN, ...
```

- [ ] **Step 7: Commit resolution**

```bash
git add docs/superpowers/specs/2026-05-02-housekeeping-audit-report.md
git commit -m "docs(audit): add resolution mapping for housekeeping audit"
git push origin chore/housekeeping-fixes
```

### Task 4.1: Merge PR + cleanup

**Files:** none

- [ ] **Step 1: Merge PR**

Per CLAUDE.md release workflow: verify CI green first, then merge with `--admin` if needed (branch protection).

```bash
gh pr checks <PR_NUMBER>  # confirm all green
gh pr merge <PR_NUMBER> --squash --admin --delete-branch
```

- [ ] **Step 2: Pull main locally**

```bash
git checkout main
git pull origin main
```

- [ ] **Step 3: Delete local branch**

```bash
git branch -d chore/housekeeping-fixes
```

- [ ] **Step 4: Save learnings to MCP Memory**

Per CLAUDE.md auto-save directive, store key findings/patterns from this audit. Tag: `mcp-memory-service`, `housekeeping`, `documentation`, `learnings`. Include: notable drift patterns found, what tooling caught what, recommended cadence for next audit (suggest scheduling a routine via `/schedule`).

---

## Self-Review Notes

(Per writing-plans skill — verifying plan against spec)

**Spec coverage check:**
- ✅ Phase 1 (audit) → Tasks 1.0–1.7
- ✅ Phase 2 (triage gate) → Task 2.0
- ✅ Phase 3 (execute) → Tasks 3.0–3.8
- ✅ Phase 4 (verify) → Tasks 4.0–4.1
- ✅ All 6 drift categories scanned (Tasks 1.1–1.6)
- ✅ Severity rubric applied per finding-append step
- ✅ Triage report format matches spec
- ✅ Tooling: extend dead-refs (Task 3.2) + new check_versions.sh (Tasks 3.0, 3.1) + workflow wire-up (Task 3.6)
- ✅ Wiki direct commits (Task 3.4)
- ✅ Issues for defer-issue (Task 3.5)
- ✅ Landing-page republish (Task 3.7)
- ✅ Validation checklist (Task 4.0)

**Placeholder scan:** No TBD/TODO/etc. in steps. All code blocks complete. All commands exact.

**Type/name consistency:** `check_versions.sh`, `MCS_VERSION_SCAN_TARGETS`, `MCS_VERSION_FILE`, `chore/housekeeping-fixes`, `2026-05-02-housekeeping-audit-report.md` — used consistently throughout.

**Known soft spots (not blockers):**
- Task 1.6 orphan detection script is described but not provided as code — left as judgment call to keep plan size reasonable; agent can write the small loop ad-hoc.
- Task 3.0 assumes `bats` is available; fallback to plain bash test harness documented inline.
- Task 3.7 here.now publish step is best-effort (gracefully skips if script absent).
