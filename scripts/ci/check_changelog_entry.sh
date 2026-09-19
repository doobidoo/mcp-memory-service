#!/usr/bin/env bash
# Fails a pull request that changes src/ without leaving a changelog fragment.
#
# The entry goes to changelog.d/<number>.<category>.md, not into CHANGELOG.md, so
# that two pull requests never touch the same lines. scripts/release/collect_changelog.py
# merges the fragments at release time. changelog.d/README.md has the format.
#
# Why this exists: between v11.12.0 and v11.13.0, 1 of 19 merged pull requests left a
# CHANGELOG entry. The other 18 were reconstructed from commit messages at release
# time, by someone who had not written them (#1273).
#
# Usage: check_changelog_entry.sh <base-ref>
#
# Read-only: it inspects the diff and never touches the working tree.
#
# Exemptions:
#   - a release version bump (scripts/pr/lib/is_release_bump.py)
#   - the `skip-changelog` label, applied by the job's `if:` in ci.yml, not here.
#     A label needs a NEW event to take effect; re-running replays the old payload
#     (#1266), so push a commit or update the branch after labelling.
#
# A deletion-only change is deliberately NOT exempt: removing a feature is exactly
# what a reader needs told. That is the difference to check_tests_prove_fix.sh,
# which skips cleanup-only diffs because there is nothing new to prove.
#
# Every fragment the pull request adds is validated even when no fragment was
# required, because a malformed one is silently dropped at release time — the exact
# outcome this gate exists to prevent.
set -uo pipefail

BASE_REF="${1:?usage: $0 <base-ref>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_HELPER="$SCRIPT_DIR/../pr/lib/is_release_bump.py"
PYTHON="${PYTHON:-python3}"

# Must stay in step with FRAGMENT_RE in scripts/release/collect_changelog.py: a name
# this accepts and the collector rejects passes CI and vanishes from the changelog.
FRAGMENT_RE='^changelog\.d/[^./]+\.(added|fixed|removed|internal)\.md$'

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 2
cd "$REPO_ROOT" || exit 2

base="$(git merge-base "$BASE_REF" HEAD)" || {
    echo "FAIL - cannot resolve merge-base of $BASE_REF and HEAD (shallow clone?)"
    exit 2
}

# Added fragments only. A modified one belongs to a pull request that already
# landed, so editing it is not this change's entry.
added_under_dir="$(git diff --name-only --diff-filter=A "$base" HEAD -- 'changelog.d/' \
    | grep -vE '(^|/)README\.md$')"

status=0
fragments=""
while IFS= read -r path; do
    [ -n "$path" ] || continue
    if ! printf '%s' "$path" | grep -qE "$FRAGMENT_RE"; then
        echo "FAIL - $path is not a usable fragment"
        echo "       expected changelog.d/<number>.<category>.md directly in changelog.d/,"
        echo "       category one of: added, fixed, removed, internal"
        echo "       (the collector reads only that shape; anything else is dropped silently)"
        status=1
        continue
    fi
    if [ ! -s "$path" ]; then
        echo "FAIL - $path is empty"
        status=1
        continue
    fi
    if ! grep -qE '^[[:space:]]*-[[:space:]]+\S' "$path"; then
        echo "FAIL - $path has no entry line; write it as a markdown list item starting with '- '"
        status=1
        continue
    fi
    fragments="${fragments}${path}"$'\n'
done <<< "$added_under_dir"

[ "$status" -eq 0 ] || exit 1

# Anything shipped under src/ counts, not only Python: the NER, harvest and NLI
# pattern files are YAML and the schema lives in .sql, and a change to those is as
# visible to a user as a change to a module.
src_changed="$(git diff --name-only "$base" HEAD -- 'src/')"
if [ -z "$src_changed" ]; then
    echo "PASS - no change under src/"
    exit 0
fi

if git diff "$base" HEAD | "$PYTHON" "$RELEASE_HELPER"; then
    echo "PASS - release version bump; the release itself rewrites the changelog"
    exit 0
fi

if [ -z "$fragments" ]; then
    echo "FAIL - src/ changed but no changelog fragment was added"
    echo "Changed under src/:"
    printf '%s\n' "$src_changed" | sed 's/^/  /'
    echo ""
    echo "Add changelog.d/<number>.<category>.md with one entry — category is one of"
    echo "added, fixed, removed, internal. See changelog.d/README.md. Maintainers can"
    echo "exempt a pull request with the skip-changelog label."
    exit 1
fi

echo "PASS - changelog fragment(s) present:"
printf '%s' "$fragments" | sed 's/^/  /'
