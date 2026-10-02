#!/usr/bin/env bash
# Test harness for the reporting behaviour of scripts/pr/pre_pr_check.sh
# Plain bash, same shape as test_check_versions.sh (bats not available).
#
# The gate itself runs pytest and can take minutes, so these tests do not
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

# --- Test: docs-only detection agrees with ci.yml paths-ignore ---
# The gate skips the suite exactly when CI does. On pull_request, CI skips through
# its `changes` job, which evaluates this same is_docs_only(); on push, through
# paths-ignore. If push's list changes, this fails until is_docs_only() is updated
# to match. pull_request must carry no paths-ignore: the test jobs are required
# checks, and a workflow that never starts never reports them.
test_docs_only_matches_ci_paths_ignore() {
  local ci="$REPO_ROOT/.github/workflows/ci.yml"
  local expected="paths-ignore: ['docs/**', '*.md', 'changelog.d/**', '.github/**/*.md', 'LICENSE', 'NOTICE', '.gitignore']"
  local n
  n=$(grep -cF -- "$expected" "$ci")
  [ "$n" -eq 1 ] || { echo "   ci.yml paths-ignore changed (push only, none on pull_request); update is_docs_only()"; return 1; }
  # Read from the base commit, so a PR cannot redefine what skips its own tests.
  grep -qF -- "git show HEAD^1:scripts/pr/pre_pr_check.sh | awk '/^is_docs_only\\(\\) \\{/,/^\\}/'" "$ci" \
    || { echo "   ci.yml changes job no longer reads is_docs_only() from the base commit"; return 1; }
  # No path filter of any kind between the pull_request trigger and jobs:.
  awk '/^  pull_request:/{pr=1} /^jobs:/{pr=0} pr && /paths/{bad=1} END{exit bad}' "$ci" \
    || { echo "   pull_request has a path filter; required test checks would never report"; return 1; }
  # Every test job waits for the classifier and skips only on an explicit 'false'.
  local job
  for job in tests-prove-fix test ml-extras-tests milvus-tests; do
    awk -v job="  $job:" '$0==job{in_job=1; next} in_job && /^  [a-z0-9_-]+:$/{in_job=0}
         in_job && /needs: changes/{needs=1}
         in_job && /needs\.changes\.outputs\.code != .false./{cond=1}
         END{exit !(needs && cond)}' "$ci" \
      || { echo "   job '$job' is not gated on the changes classifier"; return 1; }
  done

  eval "$(awk '/^is_docs_only\(\) \{/,/^\}/' "$GATE")"
  declare -F is_docs_only >/dev/null || { echo "   is_docs_only() not found in gate"; return 1; }

  local f
  for f in SPONSORS.md docs/a/b.md docs/x.png changelog.d/1.md .github/ISSUE_TEMPLATE/bug.md LICENSE NOTICE .gitignore; do
    is_docs_only "$f" || { echo "   '$f' should count as docs-only"; return 1; }
  done
  for f in src/x.py claude-hooks/README.md tests/README.md .github/workflows/ci.yml pyproject.toml; do
    is_docs_only "$f" && { echo "   '$f' must not count as docs-only"; return 1; }
  done
  is_docs_only $'README.md\nsrc/x.py' && { echo "   mixed change must not count as docs-only"; return 1; }
  is_docs_only $'README.md\ndocs/a.md' || { echo "   multi-file docs change should count as docs-only"; return 1; }
  is_docs_only "" && { echo "   an empty file list must not count as docs-only"; return 1; }
  return 0
}

# --- Test: committed code plus a staged doc is not a docs-only PR ---
# The gate reads the index, but CI judges the whole PR diff. A branch that already
# committed code and only stages a README must still run the suite.
test_pr_changed_files_sees_committed_code() {
  eval "$(awk '/^is_docs_only\(\) \{/,/^\}/' "$GATE")"
  eval "$(awk '/^pr_changed_files\(\) \{/,/^\}/' "$GATE")"
  declare -F pr_changed_files >/dev/null || { echo "   pr_changed_files() not found in gate"; return 1; }

  local repo files
  repo="$(mktemp -d)"
  (
    cd "$repo" || exit 1
    git init -q
    git -c user.name=t -c user.email=t@t commit -q --allow-empty -m base
    git update-ref refs/remotes/origin/main HEAD
    mkdir src && echo "x = 1" > src/x.py && git add src/x.py
    git -c user.name=t -c user.email=t@t commit -q -m code
    echo "doc" > README.md && git add README.md
  ) || { rm -rf "$repo"; echo "   could not build the fixture repo"; return 1; }

  files=$(cd "$repo" && pr_changed_files)
  [[ "$files" == *"src/x.py"* ]] || { rm -rf "$repo"; echo "   committed src/x.py missing: '$files'"; return 1; }
  [[ "$files" == *"README.md"* ]] || { rm -rf "$repo"; echo "   staged README.md missing: '$files'"; return 1; }
  is_docs_only "$files" && { rm -rf "$repo"; echo "   committed code + staged doc counted as docs-only"; return 1; }

  # Without origin/main there is no base: nothing is printed, so the suite runs.
  files=$(cd "$repo" && git update-ref -d refs/remotes/origin/main && pr_changed_files)
  rm -rf "$repo"
  [ -z "$files" ] || { echo "   expected no files without a base, got '$files'"; return 1; }
  is_docs_only "$files" && { echo "   no base must not count as docs-only"; return 1; }
  return 0
}

# --- Test: the gate wires the classifier to the PR diff and reports SKIP ---
test_gate_wires_docs_only_skip() {
  grep -q '^if is_docs_only "\$(pr_changed_files)"; then$' "$GATE" \
    || { echo "   step 3 does not classify pr_changed_files"; return 1; }
  awk '/^if is_docs_only /{start=NR}
       start && /check_status "Test suite" 3/ && NR-start < 5 {suite=1}
       start && /check_status "Test coverage" 3/ && NR-start < 5 {cov=1}
       END{exit !(suite && cov)}' "$GATE" \
    || { echo "   docs-only branch does not report both checks as SKIP"; return 1; }
}

# --- Test: step 3 runs the selected tests, with the full suite still reachable ---
# The selection itself is covered by tests/ci/test_select_tests.sh.
test_gate_wires_test_selection() {
  grep -q 'scripts/pr/lib/select_tests.py" --repo "\$REPO_ROOT"' "$GATE" \
    || { echo "   step 3 does not ask select_tests.py what to run"; return 1; }
  grep -q 'PRE_PR_FULL_SUITE' "$GATE" \
    || { echo "   PRE_PR_FULL_SUITE no longer forces the full suite"; return 1; }
  grep -q '\[ -z "\$CHANGED_FILES" \]' "$GATE" \
    || { echo "   a change without a base must run the full suite"; return 1; }
  grep -q 'COVERAGE_OUTPUT=$($PYTEST_BIN "${TEST_ARGS\[@\]}"' "$GATE" \
    || { echo "   pytest does not run the selected targets"; return 1; }
}

# --- Test: the link check runs in the gate and on every pull request ---
# ci.yml skips docs-only PRs, which are the ones that break links, so links.yml
# must not inherit a paths filter.
test_link_check_is_wired() {
  local wf="$REPO_ROOT/.github/workflows/links.yml"
  grep -q '"\$PYTHON_BIN" scripts/ci/check_md_links.py' "$GATE" \
    || { echo "   gate does not run check_md_links.py"; return 1; }
  grep -q 'run: python3 scripts/ci/check_md_links.py$' "$wf" \
    || { echo "   links.yml does not run the internal check"; return 1; }
  grep -q 'run: python3 scripts/ci/check_md_links.py --external$' "$wf" \
    || { echo "   links.yml does not run the external check"; return 1; }
  grep -qE '^\s*(paths|paths-ignore):' "$wf" \
    && { echo "   links.yml must not filter by path"; return 1; }
  return 0
}

run_test "failing command substitution does not abort the script" test_failing_command_substitution_does_not_abort
run_test "gate guards the coverage run with set +e" test_gate_guards_the_coverage_run
run_test "quality_gate.sh exits 3 when it skips" test_quality_gate_skip_uses_exit_3
run_test "gate maps exit 3 to SKIP, not PASS" test_gate_maps_exit_3_to_skip
run_test "local test selection matches CI" test_selection_matches_ci
run_test "docs-only detection matches ci.yml paths-ignore" test_docs_only_matches_ci_paths_ignore
run_test "committed code plus a staged doc is not docs-only" test_pr_changed_files_sees_committed_code
run_test "gate wires docs-only detection to SKIP" test_gate_wires_docs_only_skip
run_test "gate runs the selected tests, full suite on request" test_gate_wires_test_selection
run_test "link check runs in the gate and on every PR" test_link_check_is_wired

echo ""
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
