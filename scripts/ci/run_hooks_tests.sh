#!/bin/bash
#
# Claude Code Hooks Test Runner
#
# Runs every *.test.js file under the repo's JS test directories -
# claude-hooks/tests and opencode/tests - and fails if any of them do.
#
# Until this existed, CI had no node steps at all: the three PR jobs were
# `pytest tests/`, a site/index.html version-string diff, and a release-only
# plugin manifest gate. None of them can observe a change to the hooks, so a
# JS-only PR got a full green board that meant nothing — #198 had to be checked
# by hand. The same blind spot is behind #170 and #177, where hook fixes shipped
# broken. See issue #202.
#
# Three details that decide how this is written:
#
# 1. The suite has two harness styles — five files use node:test and print
#    "ℹ pass N", four are hand-rolled and print "N/N tests passed". Parsing
#    either format would silently miss failures in the other, so this keys off
#    EXIT CODES only.
# 2. claude-hooks/tests/integration-test.js, phase2-integration-test.js,
#    test-code-execution.js and test-permission-request.js are deliberately
#    outside the *.test.js glob: they expect a live memory service. Do not widen
#    the glob to pull them in.
# 3. opencode/ is ESM and claude-hooks/ is CommonJS. node runs each file on its
#    own, so the two styles coexist without configuration - which is why one
#    runner can cover both.
#
# Env overrides (for tests):
#   MCS_HOOKS_TEST_DIRS space-separated directories to scan
#                       (default: "claude-hooks/tests opencode/tests")
#   MCS_HOOKS_TEST_DIR  a single directory, overriding the list above
#   MCS_NODE_BIN        node binary to use (default: node)
#
# Exit codes:
#   0 - every test file exited 0
#   1 - at least one file failed, or the runner could not run reliably

set -uo pipefail

# Every directory of JS tests in the repo. MCS_HOOKS_TEST_DIR overrides with a
# single directory (used by the test harness); MCS_HOOKS_TEST_DIRS with a
# space-separated list.
DEFAULT_TEST_DIRS="claude-hooks/tests opencode/tests"
if [ -n "${MCS_HOOKS_TEST_DIR:-}" ]; then
  TEST_DIRS="$MCS_HOOKS_TEST_DIR"
else
  TEST_DIRS="${MCS_HOOKS_TEST_DIRS:-$DEFAULT_TEST_DIRS}"
fi
NODE_BIN="${MCS_NODE_BIN:-node}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "FAIL: '$NODE_BIN' not found - the hooks suite needs node"
  echo "The ubuntu-latest runner image ships node (deploy-site.yml calls it"
  echo "directly), so a failure here means the image changed."
  exit 1
fi

# Collect first, so an empty glob is an error rather than a silent pass. A rename
# that takes files out of the *.test.js pattern must break the build, not quietly
# reduce coverage to nothing. Each configured directory is checked on its own, so
# emptying one cannot hide behind another still having tests.
FILES=()
for dir in $TEST_DIRS; do
  if [ ! -d "$dir" ]; then
    echo "FAIL: test directory not found: $dir"
    echo "If a suite moved or was removed, update this runner in the same change."
    exit 1
  fi
  found_in_dir=0
  for f in "$dir"/*.test.js; do
    [ -e "$f" ] || continue
    FILES+=("$f")
    found_in_dir=$((found_in_dir + 1))
  done
  if [ "$found_in_dir" -eq 0 ]; then
    echo "FAIL: no *.test.js files found in $dir"
    echo "If the suite moved, update this runner in the same change."
    exit 1
  fi
done

echo "Running ${#FILES[@]} JS test files from: $TEST_DIRS"
echo "node $("$NODE_BIN" --version)"
echo

PASSED=0
FAILED=0
FAILED_FILES=()

for f in "${FILES[@]}"; do
  name="$f"
  if OUTPUT=$("$NODE_BIN" "$f" 2>&1); then
    printf 'ok   - %s\n' "$name"
    PASSED=$((PASSED + 1))
  else
    printf 'FAIL - %s\n' "$name"
    printf '%s\n' "$OUTPUT" | sed 's/^/       /'
    FAILED=$((FAILED + 1))
    FAILED_FILES+=("$name")
  fi
done

echo
echo "Results: $PASSED file(s) passed, $FAILED failed"

if [ $FAILED -ne 0 ]; then
  echo
  echo "Failing files:"
  printf '  %s\n' "${FAILED_FILES[@]}"
  echo
  echo "Run one directly to iterate: node <dir>/<file>"
  exit 1
fi

exit 0
