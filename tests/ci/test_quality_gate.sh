#!/usr/bin/env bash
# Test harness for scripts/pr/quality_gate.sh --staged.
# Plain bash, same shape as test_pre_pr_check.sh (bats not available).
#
# The gate calls a model, so these tests do not run a full analysis pass. They
# cover the parts that decide whether the gate evaluates anything at all, plus
# the three defects that made the checks unable to report:
#
#   1. checks 1 and 2 looped over a pipe, so their subshell threw away exit_code
#      and the warnings array - neither check could fail the gate
#   2. `grep -c ... || echo 0` produced "0\n0" and killed the arithmetic test
#      in check 3 under set -e
#   3. the complexity pattern matched score 8, which the documented budget allows
#   4. complexity was scored per file, so a change to a file holding an unrelated
#      complex function failed the gate on proximity (#1118)
#   5. check 4 grepped for the word "breaking", which matches the model's own
#      "No breaking changes found."
#   6. the findings that set exit 1 were printed under "WARNINGS (non-blocking)"

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GATE="$REPO_ROOT/scripts/pr/quality_gate.sh"
HELPER="$REPO_ROOT/scripts/pr/lib/llm_prompt.py"
SCOPE="$REPO_ROOT/scripts/pr/lib/scope_findings.py"
TOUCHED="$REPO_ROOT/scripts/pr/lib/touched_functions.py"

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

# --- Test: no backend reachable is reported as skipped, not passed ---
test_unreachable_backend_exits_3() {
  local out code
  out=$(cd "$REPO_ROOT" && MCP_QUALITY_LLM_URL="http://127.0.0.1:1/v1" bash "$GATE" --staged 2>&1)
  code=$?
  [ "$code" -eq 3 ] || { echo "   expected exit 3, got $code"; return 1; }
  [[ "$out" == *"Skipped, NOT passed"* ]] || { echo "   skip message missing: $out"; return 1; }
}

# --- Test: staged mode does not require forge access ---
test_staged_mode_needs_no_gh() {
  grep -q 'MODE" = "pr" \] && ! command -v gh' "$GATE" \
    || { echo "   the gh requirement is not scoped to PR mode"; return 1; }
  grep -q 'git diff --cached --name-only' "$GATE" \
    || { echo "   staged mode does not read the index"; return 1; }
}

# --- Test: the analysis loops are not piped, so their results escape ---
test_loops_use_process_substitution() {
  local piped
  piped=$(grep -c 'echo "\$changed_files" | while' "$GATE" || true)
  [ "$piped" -eq 0 ] || { echo "   $piped analysis loop(s) still read from a pipe"; return 1; }
  local substituted
  substituted=$(grep -c 'done < <(echo "\$changed_files")' "$GATE" || true)
  [ "$substituted" -eq 2 ] || { echo "   expected 2 process-substitution loops, found $substituted"; return 1; }
}

# --- Test: check 3 counts files without the "0\n0" fallback ---
test_file_counts_use_grep_exit_only() {
  grep -qF "grep -Ec '^tests/.*\.(py|sh)\$' || true" "$GATE" \
    || { echo "   test-file count is not the grep -Ec form over py and sh"; return 1; }
  ! grep -q 'grep -c .* || echo "0"' "$GATE" \
    || { echo "   an '|| echo 0' fallback remains and yields two lines"; return 1; }
}

# --- Test: complexity 8 is within budget and must not be flagged ---
test_complexity_threshold_allows_8() {
  local out
  out=$(printf 'handler: Score 8 - borderline\n' | python3 "$SCOPE" handler) && {
    echo "   score 8 was reported as a finding: $out"; return 1
  }
  out=$(printf 'handler: Score 9 - too much\n' | python3 "$SCOPE" handler) \
    || { echo "   score 9 was not reported"; return 1; }
  [[ "$out" == *"Score 9"* ]] || { echo "   finding text lost: $out"; return 1; }
}

# --- Test: a finding on a function the diff did not touch is dropped (#1118) ---
test_findings_scoped_to_touched_functions() {
  printf 'untouched_helper: Score 10 - pre-existing\n' | python3 "$SCOPE" the_one_i_edited && {
    echo "   a finding on an untouched function still blocks"; return 1
  }
  local out
  out=$(printf 'MyClass.the_one_i_edited: Score 9 - mine\n' | python3 "$SCOPE" the_one_i_edited) \
    || { echo "   a qualified name did not match its own leaf"; return 1; }
  [[ "$out" == *"Score 9"* ]] || { echo "   qualified-name finding lost: $out"; return 1; }
}

# --- Test: touched_functions attributes a change to the function that holds it ---
test_touched_functions_maps_lines_to_functions() {
  local tmp
  tmp=$(mktemp -d)
  (
    cd "$tmp" || exit 1
    git init -q . && git config user.email t@example.com && git config user.name t
    printf 'def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n' > m.py
    git add m.py && git commit -qm base
    printf 'def alpha():\n    return 1\n\n\ndef beta():\n    return 22\n' > m.py
    git add m.py
    python3 "$TOUCHED" --staged m.py
  ) > "$tmp/out" 2>/dev/null
  local out
  out=$(cat "$tmp/out")
  rm -rf "$tmp"
  [[ "$out" == *"beta"* ]] || { echo "   the edited function was not reported: $out"; return 1; }
  [[ "$out" != *"alpha"* ]] || { echo "   an untouched function was reported: $out"; return 1; }
}

# --- Test: the breaking-change check needs a marker, not the word "breaking" ---
test_breaking_change_needs_marker() {
  grep -q 'grep -q "\^BREAKING_CHANGE_DETECTED:"' "$GATE" \
    || { echo "   check 4 does not key off a machine marker"; return 1; }
  ! grep -q 'grep -qi "breaking\\|CRITICAL\\|HIGH"' "$GATE" \
    || { echo "   the old pattern remains and matches \"No breaking changes found\""; return 1; }
}

# --- Test: an acknowledged breaking change is reported without blocking (#1311) ---
test_breaking_change_can_be_acknowledged() {
  grep -q 'lib/breaking_change_ack.py' "$GATE" \
    || { echo "   check 4 does not look for a Breaking-Change-Acknowledged line"; return 1; }
  grep -qF 'Breaking change acknowledged ($ack_reason), not blocking' "$GATE" \
    || { echo "   an acknowledged breaking change is not printed with its reason"; return 1; }
}

# --- Test: findings that set exit 1 are not labelled non-blocking ---
test_blocking_findings_are_labelled_blocking() {
  grep -q 'echo "FINDINGS (blocking)"' "$GATE" \
    || { echo "   the failing summary does not say it blocks"; return 1; }
  ! grep -q 'WARNINGS (non-blocking)' "$GATE" \
    || { echo "   the gate still calls its blocking findings non-blocking"; return 1; }
}

# --- Test: the helper reports an unusable endpoint as no-backend, not as a reply ---
test_helper_signals_no_backend() {
  local code
  echo "ping" | MCP_QUALITY_LLM_URL="http://127.0.0.1:1/v1" python3 "$HELPER" >/dev/null 2>&1
  code=$?
  [ "$code" -eq 3 ] || { echo "   expected exit 3 from the helper, got $code"; return 1; }
}

# --- Test: a pyscn that never ran is not summarised as OK ---
test_pyscn_skip_not_reported_as_ok() {
  grep -q 'pyscn_ran=true' "$GATE" \
    || { echo "   pyscn run is not tracked"; return 1; }
  grep -q 'pyscn analysis: SKIPPED (pyscn not installed)' "$GATE" \
    || { echo "   a skipped pyscn analysis has no distinct summary line"; return 1; }
  awk '/if \[ "\$pyscn_ran" = true \]; then/{found=1} END{exit !found}' "$GATE" \
    || { echo "   the summary still keys off RUN_PYSCN rather than pyscn_ran"; return 1; }
}

run_test "unreachable backend exits 3, not 0" test_unreachable_backend_exits_3
run_test "staged mode needs no gh" test_staged_mode_needs_no_gh
run_test "analysis loops use process substitution" test_loops_use_process_substitution
run_test "file counts rely on grep's own output" test_file_counts_use_grep_exit_only
run_test "complexity threshold allows a score of 8" test_complexity_threshold_allows_8
run_test "helper exits 3 when the endpoint is unusable" test_helper_signals_no_backend
run_test "skipped pyscn is not summarised as OK" test_pyscn_skip_not_reported_as_ok
run_test "findings are scoped to the functions the diff touched" test_findings_scoped_to_touched_functions
run_test "touched_functions maps changed lines to their function" test_touched_functions_maps_lines_to_functions
run_test "breaking-change check needs a marker" test_breaking_change_needs_marker
run_test "breaking change can be acknowledged" test_breaking_change_can_be_acknowledged
run_test "blocking findings are labelled blocking" test_blocking_findings_are_labelled_blocking

echo ""
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
