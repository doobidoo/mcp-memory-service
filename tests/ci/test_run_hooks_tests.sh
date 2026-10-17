#!/usr/bin/env bash
# Test harness for scripts/ci/run_hooks_tests.sh
# Plain bash, same shape as test_hooks_config_secrets_check.sh (bats not available).
#
# A test runner that cannot fail is worse than no runner, because it reports a
# green board. Each case builds a throwaway test directory in $TMPDIR_LOCAL and
# points the runner at it via MCS_HOOKS_TEST_DIR, so nothing depends on the real
# claude-hooks suite and the cases do not rot as that suite changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci/run_hooks_tests.sh"
TMPDIR_LOCAL="$(mktemp -d)"

PASS=0
FAIL=0

run_test() {
  local name="$1"
  shift
  if "$@" 2>&1; then
    echo "ok - $name"
    PASS=$((PASS + 1))
  else
    echo "not ok - $name"
    FAIL=$((FAIL + 1))
  fi
}

run_on() {
  ( cd "$REPO_ROOT" && MCS_HOOKS_TEST_DIR="$1" bash "$SCRIPT" )
}

make_dir() {
  local d="$TMPDIR_LOCAL/$1"
  mkdir -p "$d"
  echo "$d"
}

test_passing_files_pass() {
  local d
  d="$(make_dir all-good)"
  printf "process.exit(0);\n" > "$d/a.test.js"
  printf "process.exit(0);\n" > "$d/b.test.js"
  local out
  out="$(run_on "$d")" || return 1
  printf '%s' "$out" | grep -q "2 file(s) passed, 0 failed"
}

# The case the whole runner exists for.
test_failing_node_test_file_fails() {
  local d out
  d="$(make_dir node-test-fail)"
  cat > "$d/broken.test.js" <<'EOF'
const { test } = require('node:test');
const assert = require('node:assert');
test('deliberately broken', () => { assert.strictEqual(1, 2); });
EOF
  if out="$(run_on "$d" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -qE "FAIL - .*broken\.test\.js"
}

# Four of the nine real files are hand-rolled and print their own "PASS:" lines.
# Keying off output instead of exit codes would call this a pass.
test_handrolled_failure_is_caught() {
  local d out
  d="$(make_dir handrolled-fail)"
  printf "console.log('PASS: looks fine');\nconsole.log('1/2 tests passed');\nprocess.exit(1);\n" \
    > "$d/liar.test.js"
  if out="$(run_on "$d" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -qE "FAIL - .*liar\.test\.js"
}

test_failure_output_is_shown() {
  local d out
  d="$(make_dir shows-output)"
  printf "console.error('a distinctive failure marker');\nprocess.exit(1);\n" > "$d/noisy.test.js"
  if out="$(run_on "$d" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "a distinctive failure marker"
}

# A rename that empties the glob must break the build, not silently reduce
# coverage to zero and report success.
test_empty_dir_fails() {
  local d out
  d="$(make_dir empty)"
  if out="$(run_on "$d" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "no \*.test.js files found"
}

test_missing_dir_fails() {
  local out
  if out="$(run_on "$TMPDIR_LOCAL/does-not-exist" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "test directory not found"
}

test_missing_node_fails() {
  local out
  if out="$( cd "$REPO_ROOT" && MCS_NODE_BIN=definitely-not-node bash "$SCRIPT" 2>&1 )"; then return 1; fi
  printf '%s' "$out" | grep -q "needs node"
}

# Non-.test.js files in the same directory must be left alone - the real suite
# keeps live-server scripts next to the unit tests.
test_non_test_files_ignored() {
  local d out
  d="$(make_dir ignores-others)"
  printf "process.exit(0);\n" > "$d/a.test.js"
  printf "process.exit(1);\n" > "$d/integration-test.js"
  out="$(run_on "$d")" || return 1
  printf '%s' "$out" | grep -q "1 file(s) passed, 0 failed"
}

# The real suite must pass, so a broken runner cannot hide behind fixtures.
test_repository_suite_passes() {
  ( cd "$REPO_ROOT" && bash "$SCRIPT" ) >/dev/null
}

run_multi() {
  ( cd "$REPO_ROOT" && MCS_HOOKS_TEST_DIRS="$1" bash "$SCRIPT" )
}

# The reason this runner grew a list: opencode/ had no coverage because nothing
# outside claude-hooks/tests ran (#210).
test_multiple_dirs_are_all_run() {
  local a b out
  a="$(make_dir multi-a)"
  b="$(make_dir multi-b)"
  printf "process.exit(0);\n" > "$a/one.test.js"
  printf "process.exit(0);\n" > "$b/two.test.js"
  out="$(run_multi "$a $b")" || return 1
  printf '%s' "$out" | grep -q "2 file(s) passed, 0 failed"
}

# A failure in the second directory must fail the run, not be masked by the first.
test_failure_in_second_dir_fails() {
  local a b out
  a="$(make_dir second-ok)"
  b="$(make_dir second-bad)"
  printf "process.exit(0);\n" > "$a/fine.test.js"
  printf "process.exit(1);\n" > "$b/broken.test.js"
  if out="$(run_multi "$a $b" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "broken.test.js"
}

# Emptying one directory must not hide behind another still having tests -
# otherwise a rename silently drops a whole suite and the board stays green.
test_one_empty_dir_fails_the_run() {
  local a b out
  a="$(make_dir nonempty)"
  b="$(make_dir emptied)"
  printf "process.exit(0);\n" > "$a/fine.test.js"
  if out="$(run_multi "$a $b" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "no \*.test.js files found"
}

test_missing_dir_in_list_fails() {
  local a out
  a="$(make_dir present)"
  printf "process.exit(0);\n" > "$a/fine.test.js"
  if out="$(run_multi "$a $TMPDIR_LOCAL/absent" 2>&1)"; then return 1; fi
  printf '%s' "$out" | grep -q "test directory not found"
}

# Same-named files in two directories must be distinguishable in the output.
test_output_disambiguates_same_named_files() {
  local a b out
  a="$(make_dir dup-a)"
  b="$(make_dir dup-b)"
  printf "process.exit(0);\n" > "$a/same.test.js"
  printf "process.exit(0);\n" > "$b/same.test.js"
  out="$(run_multi "$a $b")" || return 1
  [ "$(printf '%s' "$out" | grep -c "same.test.js")" -eq 2 ] || return 1
  printf '%s' "$out" | grep -q "dup-a/same.test.js"
}

# The real repository directories must all be covered by the default.
test_default_covers_all_clients() {
  local out
  out="$( cd "$REPO_ROOT" && bash "$SCRIPT" )" || return 1
  printf '%s' "$out" | grep -q "claude-hooks/tests" &&
  printf '%s' "$out" | grep -q "opencode/tests" &&
  printf '%s' "$out" | grep -q "codex/tests"
}

run_test "passing files pass"                          test_passing_files_pass
run_test "a failing node:test file fails the run"      test_failing_node_test_file_fails
run_test "hand-rolled failure caught despite PASS out" test_handrolled_failure_is_caught
run_test "failing file's output is printed"            test_failure_output_is_shown
run_test "empty test dir fails, not passes silently"   test_empty_dir_fails
run_test "missing test dir is reported"                test_missing_dir_fails
run_test "missing node binary is reported"             test_missing_node_fails
run_test "non-.test.js files are ignored"              test_non_test_files_ignored
run_test "the repository's own hooks suite passes"     test_repository_suite_passes
run_test "multiple dirs are all run"                   test_multiple_dirs_are_all_run
run_test "failure in the second dir fails the run"     test_failure_in_second_dir_fails
run_test "one empty dir fails, not masked by another"  test_one_empty_dir_fails_the_run
run_test "missing dir in the list is reported"         test_missing_dir_in_list_fails
run_test "same-named files are disambiguated"          test_output_disambiguates_same_named_files
run_test "default covers all three clients"            test_default_covers_all_clients

rm -rf "$TMPDIR_LOCAL"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
