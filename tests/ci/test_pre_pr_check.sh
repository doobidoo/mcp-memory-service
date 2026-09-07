#!/usr/bin/env bash
# Test harness for the reporting behaviour of scripts/pr/pre_pr_check.sh
# Plain bash, same shape as test_check_versions.sh (bats not available).
#
# The gate itself runs the full suite and takes minutes, so these tests do not
# execute it. They cover the two reporting defects that made the gate unusable:
#
#   1. a failing test run aborted the script under `set -e` before its own
#      TEST_EXIT_CODE handling could report anything
#   2. a skipped quality gate (no analysis backend available) was reported as a pass
#
# Both are verified against the extracted logic rather than a full run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GATE="$REPO_ROOT/scripts/pr/pre_pr_check.sh"
QUALITY_GATE="$REPO_ROOT/scripts/pr/quality_gate.sh"

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

# --- Test: a failing command in $(...) must not abort the surrounding script ---
# Mirrors the step 3 construct. Without `set +e` the assignment kills the shell.
test_failing_command_substitution_does_not_abort() {
  local out
  out=$(
    set -e
    set +e
    CAPTURED=$(sh -c 'echo "FAILED tests/x.py::test_y"; exit 1' 2>&1)
    CODE=$?
    set -e
    echo "code=$CODE reported=${CAPTURED}"
  ) || return 1
  [[ "$out" == *"code=1"* ]] || { echo "   exit code was lost: $out"; return 1; }
  [[ "$out" == *"FAILED tests/x.py::test_y"* ]] || { echo "   captured output lost: $out"; return 1; }
}

# --- Test: the gate uses that construct, not a bare assignment ---
test_gate_guards_the_coverage_run() {
  grep -q 'set +e' "$GATE" || { echo "   no 'set +e' guard found"; return 1; }
  # The assignment must be preceded by the guard, not run bare under set -e
  awk '/^set \+e$/{guard=NR} /^COVERAGE_OUTPUT=\$\(/{if (guard && NR-guard < 8) found=1} END{exit !found}' "$GATE" \
    || { echo "   COVERAGE_OUTPUT assignment is not guarded by set +e"; return 1; }
}

# --- Test: quality_gate.sh signals "skipped" distinctly when Gemini is absent ---
test_quality_gate_skip_uses_exit_3() {
  local tmpbin out code
  tmpbin="$(mktemp -d)"
  # PATH without python3 or gemini, so no analysis backend resolves. `gh` is kept
  # available because PR mode requires it; --staged does not.
  for tool in gh git bash grep sed awk; do
    command -v "$tool" >/dev/null 2>&1 && ln -sf "$(command -v "$tool")" "$tmpbin/$tool"
  done
  out=$(PATH="$tmpbin" bash "$QUALITY_GATE" --staged 2>&1)
  code=$?
  rm -rf "$tmpbin"
  if [[ "$out" == *"GitHub CLI (gh) is not installed"* ]]; then
    echo "   skipped: gh not available in this environment"
    return 0
  fi
  [ "$code" -eq 3 ] || { echo "   expected exit 3 for a skipped gate, got $code"; return 1; }
  [[ "$out" == *"Skipped, NOT passed"* ]] || { echo "   skip message missing"; return 1; }
}

# --- Test: the gate maps exit 3 to SKIP rather than PASS ---
test_gate_maps_exit_3_to_skip() {
  grep -q 'status -eq 3' "$GATE" || { echo "   check_status has no skip branch"; return 1; }
  grep -q 'QUALITY_GATE_EXIT -eq 3' "$GATE" || { echo "   quality gate skip not mapped"; return 1; }
  # A skipped check must not be counted as failed
  awk '/status -eq 3/{found=NR} /SKIPPED_CHECKS=\$\(\(SKIPPED_CHECKS \+ 1\)\)/{if (found && NR-found < 4) ok=1} END{exit !ok}' "$GATE" \
    || { echo "   skip branch does not increment SKIPPED_CHECKS"; return 1; }
}

# --- Test: local test selection matches CI so a green gate means what CI says ---
test_selection_matches_ci() {
  local ci="$REPO_ROOT/.github/workflows/ci.yml"
  [ -f "$ci" ] || { echo "   ci.yml not found"; return 1; }
  local ignore
  for ignore in tests/benchmarks tests/integration/test_cli_interfaces.py; do
    grep -q -- "--ignore=$ignore" "$ci" || { echo "   ci.yml no longer ignores $ignore"; return 1; }
    grep -q -- "--ignore=$ignore" "$GATE" || { echo "   gate does not ignore $ignore"; return 1; }
  done
  # #1145: consolidation and integration run everywhere now; neither side may drop them again.
  for ignore in tests/consolidation tests/integration; do
    grep -q -- "--ignore=$ignore \\\\" "$GATE" && { echo "   gate ignores $ignore, CI runs it"; return 1; }
  done
  grep -q -- '-m "not benchmark"' "$GATE" || { echo "   gate does not deselect benchmark marker"; return 1; }
}

run_test "failing command substitution does not abort the script" test_failing_command_substitution_does_not_abort
run_test "gate guards the coverage run with set +e" test_gate_guards_the_coverage_run
run_test "quality_gate.sh exits 3 when it skips" test_quality_gate_skip_uses_exit_3
run_test "gate maps exit 3 to SKIP, not PASS" test_gate_maps_exit_3_to_skip
run_test "local test selection matches CI" test_selection_matches_ci

echo ""
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
