#!/usr/bin/env bash
# Tests for scripts/pr/lib/select_tests.py, the pre-PR gate's test selection.
# Plain bash, same shape as the other tests/ci harnesses (bats not available).
# Each case feeds a list of changed files into the selector over a small fixture
# repository and compares what it prints.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SELECT="$REPO_ROOT/scripts/pr/lib/select_tests.py"
PY="$(command -v python3)"

PASS=0
FAIL=0

FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT
mkdir -p "$FIX/src/mcp_memory_service/storage" "$FIX/src/mcp_memory_service/web" \
         "$FIX/tests/storage" "$FIX/tests/unit" "$FIX/tests/benchmarks" "$FIX/scripts/ci"
touch "$FIX/src/mcp_memory_service/storage/sqlite_vec.py" \
      "$FIX/src/mcp_memory_service/storage/lonely.py" \
      "$FIX/src/mcp_memory_service/web/app.py" "$FIX/src/mcp_memory_service/orphan.py" \
      "$FIX/scripts/ci/check_links.py" "$FIX/scripts/ci/untested.py"
echo "from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage" > "$FIX/tests/unit/test_dotted.py"
echo "from mcp_memory_service.web import app" > "$FIX/tests/unit/test_from_import.py"
echo "SCRIPT = 'scripts/ci/check_links.py'" > "$FIX/tests/unit/test_script.py"
echo "x = 1" > "$FIX/tests/storage/test_other.py"
mkdir -p "$FIX/tests/ci" "$FIX/scripts/pr" && touch "$FIX/scripts/pr/gate.sh"
echo 'GATE="$REPO_ROOT/scripts/pr/gate.sh"' > "$FIX/tests/ci/test_gate.sh"
# Benchmarks never count as an importer: the gate ignores tests/benchmarks.
echo "from mcp_memory_service.orphan import thing" > "$FIX/tests/benchmarks/test_bench.py"

# check <name> <expected output> <changed file>...
check() {
  local name="$1" expected="$2"
  shift 2
  local out
  out=$(printf '%s\n' "$@" | "$PY" "$SELECT" --repo "$FIX")
  if [ "$out" = "$expected" ]; then
    echo "ok - $name"
    PASS=$((PASS + 1))
  else
    echo "not ok - $name"
    echo "   expected: $(echo "$expected" | tr '\n' ' ')"
    echo "   got:      $(echo "$out" | tr '\n' ' ')"
    FAIL=$((FAIL + 1))
  fi
}

check "docs-only change selects nothing" "" docs/a.md README.md
check "a dependency change selects the full suite" "ALL" uv.lock
check "pyproject.toml selects the full suite" "ALL" pyproject.toml docs/a.md
check "a conftest.py selects the full suite" "ALL" tests/unit/conftest.py
check "a non-test helper under tests/ selects the full suite" "ALL" tests/helpers.py
check "packaged data under src/ selects the full suite" "ALL" src/mcp_memory_service/web/static/app.js
check "a src module selects its importers and its tests/ subdirectory" \
  "$(printf 'tests/storage\ntests/unit/test_dotted.py')" src/mcp_memory_service/storage/sqlite_vec.py
check "'from pkg import mod' counts as an import" \
  "tests/unit/test_from_import.py" src/mcp_memory_service/web/app.py
check "a src module no test names still selects its tests/ subdirectory" \
  "tests/storage" src/mcp_memory_service/storage/lonely.py
check "a src module with no importer and no tests/ subdirectory selects the full suite" \
  "ALL" src/mcp_memory_service/orphan.py
check "a changed test file selects itself" "tests/storage/test_other.py" tests/storage/test_other.py
check "a deleted test file selects nothing" "" tests/unit/test_gone.py
check "a script selects the tests that mention it" "tests/unit/test_script.py" scripts/ci/check_links.py
check "a script no test mentions selects nothing" "" scripts/ci/untested.py
check "a gate script selects the tests/ci harness that exercises it" \
  "tests/ci/test_gate.sh" scripts/pr/gate.sh
check "a changed tests/ci harness selects itself" "tests/ci/test_gate.sh" tests/ci/test_gate.sh
check "a .sh outside tests/ci is not a test file" "" tests/fixtures/test_helper.sh

echo ""
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
