#!/usr/bin/env bash
# Covers scripts/ci/check_changelog_entry.sh, the gate that requires a changelog
# fragment for a src/ change, and scripts/release/collect_changelog.py, which merges
# the fragments back into CHANGELOG.md.
#
# Each gate case builds a throwaway git repo, so the checks run against real diffs
# rather than a mocked one.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$REPO_ROOT/scripts/ci/check_changelog_entry.sh"
COLLECT="$REPO_ROOT/scripts/release/collect_changelog.py"
failures=0

report() {
    local name="$1" expected="$2" actual="$3"
    if [ "$actual" -eq "$expected" ]; then
        echo "PASS - $name"
    else
        echo "FAIL - $name (expected exit $expected, got $actual)"
        failures=$((failures + 1))
    fi
}

# Builds a repo with one base commit, runs $1 as the change, then the gate.
run_gate_case() {
    local name="$1" expected="$2" change="$3"
    local tmp
    tmp="$(mktemp -d)" || return 1
    (
        cd "$tmp" || exit 2
        # Name the branch explicitly: falling back between master and main would
        # turn a script error (exit 2) into a second attempt and hide it.
        git init -q -b main .
        git config user.email t@example.com
        git config user.name t
        mkdir -p src/mcp_memory_service changelog.d scripts/ci scripts/pr/lib scripts/release
        cp "$REPO_ROOT/scripts/pr/lib/is_release_bump.py" scripts/pr/lib/
        echo "# fragments" > changelog.d/README.md
        echo "x = 1" > src/mcp_memory_service/thing.py
        printf '__version__ = "1.0.0"\n' > src/mcp_memory_service/_version.py
        git add -A && git commit -qm base
        git checkout -qb feature
        eval "$change"
        git add -A && git commit -qm change
        bash "$GATE" main >/dev/null 2>&1
    )
    report "$name" "$expected" "$?"
    rm -rf "$tmp"
}

# --- the gate --------------------------------------------------------------

run_gate_case "src change without a fragment fails" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py'

run_gate_case "src change with a fragment passes" 0 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     echo "- **Something changed (#1).** Because of a reason." > changelog.d/1.fixed.md'

run_gate_case "no src change needs no fragment" 0 \
    'echo "notes" > NOTES.md'

run_gate_case "version bump is exempt" 0 \
    'printf "__version__ = \"1.1.0\"\n" > src/mcp_memory_service/_version.py'

run_gate_case "unknown category is rejected" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     echo "- entry" > changelog.d/1.changed.md'

run_gate_case "empty fragment is rejected" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     : > changelog.d/1.fixed.md'

run_gate_case "fragment without a list item is rejected" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     echo "just some prose" > changelog.d/1.fixed.md'

# A deletion-only diff is exempt from the prove-fix gate but NOT from this one:
# a removed feature is exactly what a changelog reader needs told.
run_gate_case "deletion-only change still needs a fragment" 1 \
    'rm src/mcp_memory_service/thing.py'

# A shipped YAML pattern file or SQL migration changes behaviour as much as a module
# does; src/ carries 24 non-Python files. Greptile caught the gate narrowing to .py.
run_gate_case "non-Python change under src/ still needs a fragment" 1 \
    'mkdir -p src/mcp_memory_service/harvest/patterns
     echo "patterns: []" > src/mcp_memory_service/harvest/patterns/en.yaml'

run_gate_case "non-Python change under src/ passes with a fragment" 0 \
    'mkdir -p src/mcp_memory_service/storage/migrations
     echo "CREATE TABLE t (id INT);" > src/mcp_memory_service/storage/migrations/009.sql
     echo "- **A migration (#1).**" > changelog.d/1.added.md'

# A fragment in a subdirectory is found by the diff but never read by the collector,
# so it would pass CI and vanish from the release notes.
run_gate_case "fragment in a subdirectory is rejected" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     mkdir -p changelog.d/sub
     echo "- entry" > changelog.d/sub/1.fixed.md'

# The collector requires a non-empty number; the gate must not accept what it drops.
run_gate_case "fragment with an empty number is rejected" 1 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     echo "- entry" > changelog.d/.fixed.md'

# Validation must not sit behind the early exits: a docs-only PR adding a broken
# fragment would otherwise be green here and silently dropped at release time.
run_gate_case "malformed fragment fails even with no src/ change" 1 \
    'echo "notes" > NOTES.md
     : > changelog.d/1.fixed.md'

# Unquoted expansion has produced false PASS results in this repo before. A name with
# a space must be handled as one path, not split into two unreadable ones.
run_gate_case "fragment name containing a space is handled as one path" 0 \
    'echo "x = 2" > src/mcp_memory_service/thing.py
     echo "- **Spaced out (#1).**" > "changelog.d/1 and 2.fixed.md"'

# Editing a fragment that is already on the base branch is not this PR's entry.
run_gate_case "modifying an existing fragment does not count" 1 \
    'git checkout -q main
     echo "- old entry" > changelog.d/9.fixed.md
     git add -A && git commit -qm "existing fragment"
     git checkout -q feature
     git merge -q --no-edit main
     echo "x = 2" > src/mcp_memory_service/thing.py
     echo "- edited entry" > changelog.d/9.fixed.md'

# --- the collector ---------------------------------------------------------

collect_case() {
    local name="$1" expected_grep="$2"
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    cat > "$tmp/CHANGELOG.md" <<'EOF'
# Changelog

## [Unreleased]

### Fixed

- **An entry that was already here (#0).**

## [1.0.0] - 2026-01-01

- older
EOF
    echo "- **A new fix (#2).**" > "$tmp/changelog.d/2.fixed.md"
    echo "- **A new feature (#3).**" > "$tmp/changelog.d/3.added.md"
    echo "- **Some tooling (#4).**" > "$tmp/changelog.d/4.internal.md"
    echo "# fragments" > "$tmp/changelog.d/README.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL - $name (collector exited $rc)"
        failures=$((failures + 1))
        rm -rf "$tmp"
        return
    fi
    local out="$tmp/CHANGELOG.md"
    local ok=0
    # The pre-existing entry survives, all three fragments land, the released
    # block is untouched, and every fragment file is gone.
    grep -q "An entry that was already here" "$out" || { echo "  missing: pre-existing entry"; ok=1; }
    grep -q "A new fix" "$out" || { echo "  missing: fixed fragment"; ok=1; }
    grep -q "A new feature" "$out" || { echo "  missing: added fragment"; ok=1; }
    grep -q "Some tooling" "$out" || { echo "  missing: internal fragment"; ok=1; }
    grep -q "^## \[1.0.0\]" "$out" || { echo "  missing: released block"; ok=1; }
    grep -q "^### Added" "$out" || { echo "  missing: created Added heading"; ok=1; }
    [ -f "$tmp/changelog.d/2.fixed.md" ] && { echo "  fragment not deleted"; ok=1; }
    [ -f "$tmp/changelog.d/README.md" ] || { echo "  README was deleted"; ok=1; }
    # The new fix must land under Fixed, above the [1.0.0] heading.
    local fixed_line new_fix released
    fixed_line=$(grep -n "^### Fixed" "$out" | head -1 | cut -d: -f1)
    new_fix=$(grep -n "A new fix" "$out" | head -1 | cut -d: -f1)
    released=$(grep -n "^## \[1.0.0\]" "$out" | head -1 | cut -d: -f1)
    [ "$new_fix" -gt "$fixed_line" ] && [ "$new_fix" -lt "$released" ] || {
        echo "  new fix landed outside the Fixed section of [Unreleased]"; ok=1; }
    report "$name" 0 "$ok"
    rm -rf "$tmp"
}

collect_case "collector merges fragments and deletes them" 0

# Running it twice must not duplicate anything: the fragments are gone.
rerun_case() {
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    printf '# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n' > "$tmp/CHANGELOG.md"
    echo "- **Only once (#5).**" > "$tmp/changelog.d/5.fixed.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 \
        && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local n
    n=$(grep -c "Only once" "$tmp/CHANGELOG.md")
    report "collector is idempotent (entry appears once)" 1 "$n"
    rm -rf "$tmp"
}

rerun_case

# A created section must land in SECTIONS order, not at the end: with only ### Fixed
# present, a new Added section belongs above it.
order_case() {
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    printf '# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **Existing (#0).**\n\n## [1.0.0] - 2026-01-01\n' > "$tmp/CHANGELOG.md"
    echo "- **New feature (#3).**" > "$tmp/changelog.d/3.added.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local added fixed
    added=$(grep -n "^### Added" "$tmp/CHANGELOG.md" | head -1 | cut -d: -f1)
    fixed=$(grep -n "^### Fixed" "$tmp/CHANGELOG.md" | head -1 | cut -d: -f1)
    if [ -n "$added" ] && [ -n "$fixed" ] && [ "$added" -lt "$fixed" ]; then
        echo "PASS - a created section keeps canonical order (Added before Fixed)"
    else
        echo "FAIL - created section out of order (Added at ${added:-none}, Fixed at ${fixed:-none})"
        failures=$((failures + 1))
    fi
    rm -rf "$tmp"
}

order_case

# If the unlink after the write is interrupted, the fragment survives with its entry
# already in the file. A rerun must not append it twice.
crash_case() {
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    printf '# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n' > "$tmp/CHANGELOG.md"
    echo "- **Survives a crash (#6).**" > "$tmp/changelog.d/6.fixed.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    # Simulate the interrupted run: put the fragment back, entry already merged.
    echo "- **Survives a crash (#6).**" > "$tmp/changelog.d/6.fixed.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local n
    n=$(grep -c "Survives a crash" "$tmp/CHANGELOG.md")
    report "collector skips an entry already in [Unreleased]" 1 "$n"
    rm -rf "$tmp"
}

crash_case

# First lines repeat across releases ("- **Dependency bumps.**"), so identity has to
# be the whole entry: a fragment sharing only its first line must still be merged,
# and must not be deleted as a duplicate.
same_first_line_case() {
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    printf '# Changelog\n\n## [Unreleased]\n\n### Internal\n\n- **Dependency bumps.**\n  anyio 4.14.2 (#1268).\n\n## [1.0.0] - 2026-01-01\n' > "$tmp/CHANGELOG.md"
    printf -- '- **Dependency bumps.**\n  ruff 0.16.7 and transformers 5.17.0 (#1257).\n' > "$tmp/changelog.d/7.internal.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local ok=0
    grep -q "anyio 4.14.2" "$tmp/CHANGELOG.md" || { echo "  lost: the pre-existing detail"; ok=1; }
    grep -q "ruff 0.16.7" "$tmp/CHANGELOG.md" || { echo "  lost: the new fragment's detail"; ok=1; }
    report "entries sharing a first line are both kept" 0 "$ok"
    rm -rf "$tmp"
}

same_first_line_case

# An entry may carry nested bullets. Only a bullet in column 0 starts an entry — an
# indented one belongs to the entry above, and splitting on it truncates that entry,
# which then never matches its fragment and gets merged a second time.
nested_bullets_case() {
    local tmp
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/changelog.d" "$tmp/scripts/release"
    cp "$COLLECT" "$tmp/scripts/release/"
    printf '# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n' > "$tmp/CHANGELOG.md"
    printf -- '- **Nested (#8).** It has parts:\n  - first part\n  - second part\n' > "$tmp/changelog.d/8.fixed.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    # The interrupted-unlink case: the fragment is back, its entry already merged.
    printf -- '- **Nested (#8).** It has parts:\n  - first part\n  - second part\n' > "$tmp/changelog.d/8.fixed.md"
    ( cd "$tmp" && python3 scripts/release/collect_changelog.py >/dev/null 2>&1 )
    local n
    n=$(grep -c "Nested (#8)" "$tmp/CHANGELOG.md")
    report "an entry with nested bullets is not merged twice" 1 "$n"
    rm -rf "$tmp"
}

nested_bullets_case

echo ""
if [ "$failures" -eq 0 ]; then
    echo "All changelog gate tests passed"
    exit 0
fi
echo "$failures test(s) failed"
exit 1
