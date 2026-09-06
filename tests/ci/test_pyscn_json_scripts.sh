#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMPDIR_LOCAL="$(mktemp -d)"
FAKEBIN="$TMPDIR_LOCAL/bin"
mkdir -p "$FAKEBIN"

cat > "$FAKEBIN/pyscn" <<'EOF'
#!/usr/bin/env bash
set -e
if [ "${PYSCN_FAKE_FAIL:-}" = "1" ]; then
  echo "simulated analyzer failure" >&2
  exit 9
fi
mkdir -p .pyscn/reports
timestamp="$(date +%Y%m%d_%H%M%S)"
cat > ".pyscn/reports/analyze_${timestamp}.json" <<'JSON'
{
  "summary": {
    "health_score": 83,
    "complexity_score": 71,
    "dead_code_score": 82,
    "duplication_score": 93,
    "coupling_score": 84,
    "dependency_score": 75,
    "architecture_score": 96,
    "average_complexity": 3.5,
    "total_functions": 42,
    "dead_code_count": 7,
    "code_duplication_percentage": 2.5
  },
  "complexity": {"summary": {"max_complexity": 11}}
}
JSON
printf '<html><body>Current report layout without legacy score markup</body></html>\n' \
  > ".pyscn/reports/analyze_${timestamp}.html"
EOF
chmod +x "$FAKEBIN/pyscn"

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

reset_reports() {
  rm -rf "$TMPDIR_LOCAL/work/.pyscn"
  mkdir -p "$TMPDIR_LOCAL/work"
}

test_pr_gate_reads_json_summary() {
  reset_reports
  local out
  out="$(cd "$TMPDIR_LOCAL/work" && PATH="$FAKEBIN:$PATH" \
    bash "$REPO_ROOT/scripts/pr/run_pyscn_analysis.sh" --threshold 80 2>&1)" || {
      echo "$out"
      return 1
    }
  [[ "$out" == *"Overall Health Score: 83/100"* ]] || return 1
  [[ "$out" == *"Avg: 3.5, Max: 11"* ]] || return 1
  [[ "$out" == *"analyze_"*".json"* ]]
}

test_tracker_reads_same_json_contract() {
  reset_reports
  local out
  out="$(cd "$TMPDIR_LOCAL/work" && PATH="$FAKEBIN:$PATH" \
    bash "$REPO_ROOT/scripts/quality/track_pyscn_metrics.sh" 2>&1)" || {
      echo "$out"
      return 1
    }
  [[ "$out" == *"Health Score: 83/100"* ]] || return 1
  [[ "$out" == *"Coupling: 84/100"* ]] || return 1
  grep -q ',83,71,82,93,84,75,96,3.5,11,2.5%,7$' \
    "$TMPDIR_LOCAL/work/.pyscn/history/metrics.csv"
}

test_reader_rejects_incomplete_reports() {
  printf '{"summary": {"health_score": 83}, "complexity": {"summary": {}}}\n' \
    > "$TMPDIR_LOCAL/incomplete.json"
  local out
  if out="$(python3 "$REPO_ROOT/scripts/quality/read_pyscn_summary.py" \
    "$TMPDIR_LOCAL/incomplete.json" 2>&1)"; then
    return 1
  fi
  [[ "$out" == *"missing or invalid numeric metric"* ]]
}

test_pr_gate_propagates_analyzer_failure() {
  reset_reports
  local out
  if out="$(cd "$TMPDIR_LOCAL/work" && PATH="$FAKEBIN:$PATH" PYSCN_FAKE_FAIL=1 \
    bash "$REPO_ROOT/scripts/pr/run_pyscn_analysis.sh" 2>&1)"; then
    return 1
  fi
  [[ "$out" == *"Analysis failed"* ]] && [[ "$out" == *"simulated analyzer failure"* ]]
}

run_test "PR gate reads the current pyscn JSON summary" test_pr_gate_reads_json_summary
run_test "trend tracker reads the same JSON contract" test_tracker_reads_same_json_contract
run_test "reader rejects incomplete reports instead of recording zero" test_reader_rejects_incomplete_reports
run_test "PR gate propagates analyzer failures through tee" test_pr_gate_propagates_analyzer_failure

rm -rf "$TMPDIR_LOCAL"

echo ""
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
