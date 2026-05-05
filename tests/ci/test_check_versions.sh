#!/usr/bin/env bash
# Test harness for scripts/ci/check_versions.sh
# Uses plain bash (bats not available; install via: brew install bats-core)
# Each test is a function; run_test tracks pass/fail counts.
#
# Tests are self-contained: each one writes a fake _version.py and a fake
# scan target in $TMPDIR_LOCAL and points the script at them via the
# MCS_VERSION_FILE / MCS_VERSION_SCAN_TARGETS env overrides. This means the
# tests do NOT rot every time the canonical version changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci/check_versions.sh"
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

# Helper: write a fake _version.py with the given version string
write_version_file() {
  local version="$1"
  local path="$2"
  cat > "$path" <<EOF
__version__ = "$version"
EOF
}

# --- Test: exits 0 when scan target matches canonical exactly ---
test_no_drift_exact_match() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local target="$TMPDIR_LOCAL/exact.md"
  write_version_file "10.49.2" "$vfile"
  echo "Current release: v10.49.2" > "$target"
  local status
  env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$target" \
    bash "$SCRIPT" >/dev/null 2>&1 && status=0 || status=$?
  [ "$status" -eq 0 ]
}

# --- Test: exits 1 on MAJOR drift ---
test_drift_major() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local target="$TMPDIR_LOCAL/major.md"
  write_version_file "10.49.2" "$vfile"
  echo "Stale ref: v9.0.0" > "$target"
  local output status
  output=$(env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$target" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 1 ] && echo "$output" | grep -q "v9.0.0"
}

# --- Test: exits 1 on MINOR drift ---
test_drift_minor() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local target="$TMPDIR_LOCAL/minor.md"
  write_version_file "10.49.2" "$vfile"
  echo "Stale ref: v10.47.2" > "$target"
  local output status
  output=$(env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$target" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 1 ] && echo "$output" | grep -q "v10.47.2"
}

# --- Test: PATCH-only drift is ignored (landing page MINOR/MAJOR only) ---
test_patch_drift_ignored() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local target="$TMPDIR_LOCAL/patch.md"
  write_version_file "10.49.2" "$vfile"
  echo "Older patch ref: v10.49.1" > "$target"
  local status
  env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$target" \
    bash "$SCRIPT" >/dev/null 2>&1 && status=0 || status=$?
  [ "$status" -eq 0 ]
}

# --- Test: PATCH drift mixed with MINOR drift still flags MINOR ---
test_patch_ignored_but_minor_caught() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local target="$TMPDIR_LOCAL/mixed.md"
  write_version_file "10.49.2" "$vfile"
  cat > "$target" <<EOF
Older patch (ok): v10.49.1
Older minor (bad): v10.48.0
EOF
  local output status
  output=$(env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$target" \
    bash "$SCRIPT" 2>&1) && status=0 || status=$?
  [ "$status" -eq 1 ] && echo "$output" | grep -q "v10.48.0" \
    && ! echo "$output" | grep -q "v10.49.1"
}

# --- Test: skips CHANGELOG even with old refs ---
test_skip_changelog() {
  local vfile="$TMPDIR_LOCAL/_version.py"
  local tmpfile="$TMPDIR_LOCAL/CHANGELOG.md"
  write_version_file "10.49.2" "$vfile"
  echo "This file references v1.0.0" > "$tmpfile"
  local status
  env MCS_VERSION_FILE="$vfile" MCS_VERSION_SCAN_TARGETS="$tmpfile" \
    bash "$SCRIPT" >/dev/null 2>&1 && status=0 || status=$?
  [ "$status" -eq 0 ]
}

# Run tests
run_test "exits 0 when scan target matches canonical exactly"  test_no_drift_exact_match
run_test "exits 1 on MAJOR drift"                              test_drift_major
run_test "exits 1 on MINOR drift"                              test_drift_minor
run_test "PATCH-only drift is ignored"                         test_patch_drift_ignored
run_test "PATCH ignored but MINOR drift still caught"          test_patch_ignored_but_minor_caught
run_test "skips CHANGELOG even with old refs"                  test_skip_changelog

# Cleanup
rm -rf "$TMPDIR_LOCAL"

# Summary
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
