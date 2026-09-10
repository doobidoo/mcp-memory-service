#!/usr/bin/env bash
# Fails a pull request whose source change is not proven by a test.
#
# Two checks, in this order:
#   1. A Python change under src/ must come with an added or modified test under
#      tests/, unless the src/ diff is cleanup-only (scripts/pr/lib/is_cleanup_only.py:
#      deletions, trimmed imports, dropped bindings).
#   2. Those tests, run against the base branch's src/, must fail. A test that is
#      green with and without the change proves nothing about the change. That is
#      the shape of a mock that encodes the bug it was meant to catch, and it is
#      what a green CI board could not tell a reviewer before this check existed.
#
# Usage: check_tests_prove_fix.sh <base-ref>
#
# The working tree is left with src/ at the base commit. Run it in CI or in a
# throwaway checkout, never in a tree you are working in.
#
# Exemption: a maintainer adds the `skip-prove-fix` label to the PR. Label-gated
# jobs do not re-trigger on their own, so re-run the failed job afterwards.
set -uo pipefail

# A .pyc compiled from the PR's source stays valid after the swap when the base
# file has the same size and lands in the same second (git checkout stamps files
# with the current time). Write no bytecode, and drop what is already there.
export PYTHONDONTWRITEBYTECODE=1

BASE_REF="${1:?usage: $0 <base-ref>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_HELPER="$SCRIPT_DIR/../pr/lib/is_cleanup_only.py"
PYTHON="${PYTHON:-python3}"

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 2
cd "$REPO_ROOT" || exit 2

base="$(git merge-base "$BASE_REF" HEAD)" || {
    echo "FAIL - cannot resolve merge-base of $BASE_REF and HEAD (shallow clone?)"
    exit 2
}

src_changed="$(git diff --name-only "$base" HEAD -- 'src/' | grep -E '\.py$')"
if [ -z "$src_changed" ]; then
    echo "PASS - no Python change under src/"
    exit 0
fi

if git diff "$base" HEAD -- 'src/' | "$PYTHON" "$CLEANUP_HELPER"; then
    echo "PASS - cleanup-only change under src/ (nothing added), no test required"
    exit 0
fi

# Tests the PR added or modified and that still exist on HEAD.
tests_changed="$(git diff --name-only --diff-filter=AM "$base" HEAD -- 'tests/' \
    | grep -E '(^|/)test_[^/]*\.py$')"
if [ -z "$tests_changed" ]; then
    echo "FAIL - src/ changed but no test was added or modified"
    echo "Changed under src/:"
    printf '  %s\n' $src_changed
    echo "A behavior change needs a test that fails without it. See CONTRIBUTING.md,"
    echo "'Testing Requirements'. Maintainers can exempt a PR with the skip-prove-fix label."
    exit 1
fi

# The swap below only means something if the interpreter imports this tree.
pkg_file="$("$PYTHON" -c 'import mcp_memory_service, os; print(os.path.realpath(mcp_memory_service.__file__))')" || exit 2
case "$pkg_file" in
    "$(realpath "$REPO_ROOT")"/src/*) ;;
    *)
        echo "FAIL - mcp_memory_service imports from $pkg_file, not from $REPO_ROOT/src;"
        echo "swapping src/ would not be observed. Install the package editable from this tree."
        exit 2
        ;;
esac

# Put src/ at the base commit: drop files the PR added, then check out the rest.
git diff --name-only --diff-filter=A "$base" HEAD -- 'src/' | while IFS= read -r f; do
    rm -f -- "$f"
done
git checkout -q "$base" -- 'src/'
find src -type d -name __pycache__ -prune -exec rm -rf {} +

echo "Running the changed tests against src/ at ${base:0:8}:"
printf '  %s\n' $tests_changed
"$PYTHON" -m pytest $tests_changed -q -p no:cacheprovider -m "not benchmark" --timeout=120
rc=$?

case "$rc" in
    1|2)
        echo "PASS - the changed tests fail without the change (pytest exit $rc)"
        exit 0
        ;;
    0)
        echo "FAIL - the changed tests pass without the change; they do not exercise what this PR fixes"
        echo "A test that is green on both sides is not evidence. Drive the real code path"
        echo "(real storage via the conftest fixtures, not a mock of the function under test)."
        exit 1
        ;;
    5)
        echo "FAIL - no tests collected from the changed files"
        exit 1
        ;;
    *)
        echo "FAIL - pytest exited $rc; that is a script or environment error, not a verdict"
        exit 2
        ;;
esac
