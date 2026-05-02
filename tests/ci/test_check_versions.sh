#!/usr/bin/env bash
# Test harness for scripts/ci/check_versions.sh
# Uses plain bash (bats not available; install via: brew install bats-core)
# Each test is a function; run_test tracks pass/fail counts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci/check_versions.sh"
FIXTURES="$(dirname "$0")/fixtures"
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

# --- Test: exits 0 when no drift ---
test_no_drift() {
  local output status
  output=$(env MCS_VERSION_SCAN_TARGETS="$FIXTURES/version_drift_neg.md" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 0 ]
}

# --- Test: exits 1 when drift found ---
test_drift_found() {
  local output status
  output=$(env MCS_VERSION_SCAN_TARGETS="$FIXTURES/version_drift_pos.md" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 1 ] && echo "$output" | grep -q "v9.0.0"
}

# --- Test: skips CHANGELOG even with old refs ---
test_skip_changelog() {
  local tmpfile="$TMPDIR_LOCAL/CHANGELOG.md"
  echo "This file references v1.0.0" > "$tmpfile"
  local output status
  output=$(env MCS_VERSION_SCAN_TARGETS="$tmpfile" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 0 ]
}

# Run tests
run_test "exits 0 when no drift"            test_no_drift
run_test "exits 1 when drift found"         test_drift_found
run_test "skips CHANGELOG even with old refs" test_skip_changelog

# Cleanup
rm -rf "$TMPDIR_LOCAL"

# Summary
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
