#!/usr/bin/env bash
# Covers scripts/ci/check_tests_prove_fix.sh: a src/ change needs a test, and the
# test has to fail against the base branch's src/.
#
# Each case builds a throwaway git repo with a stub mcp_memory_service package,
# commits a base on `main`, commits the PR on `pr`, and runs the real script
# against it. Needs an interpreter with pytest and pytest-timeout; falls back to
# the repo .venv when python3 lacks them.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci/check_tests_prove_fix.sh"
failures=0

pick_python() {
    local candidate
    for candidate in "${PYTHON:-python3}" "$REPO_ROOT/.venv/bin/python"; do
        if "$candidate" -c 'import pytest, pytest_timeout' >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}
PY="$(pick_python)" || {
    echo "FAIL - no interpreter with pytest and pytest-timeout (tried python3 and .venv)"
    exit 1
}

git_c() { git -c user.name=t -c user.email=t@example.com -c commit.gpgsign=false "$@"; }

# new_repo <dir>: base commit on main, then branch pr checked out.
new_repo() {
    local d="$1"
    mkdir -p "$d/src/mcp_memory_service" "$d/tests"
    cat > "$d/src/mcp_memory_service/__init__.py" <<'EOF'
import os


def answer():
    return 41
EOF
    cat > "$d/tests/test_answer.py" <<'EOF'
from mcp_memory_service import answer


def test_answer():
    assert answer() == 41
EOF
    echo "docs" > "$d/README.md"
    git_c -C "$d" init -q
    git_c -C "$d" symbolic-ref HEAD refs/heads/main
    git_c -C "$d" add -A
    git_c -C "$d" commit -q -m base
    git_c -C "$d" checkout -q -b pr
}

# run_case <name> <expected-exit> <dir> [env...]: commits whatever is in <dir> as
# the PR and runs the script from inside it.
run_case() {
    local name="$1" expected="$2" d="$3"
    shift 3
    git_c -C "$d" add -A
    git_c -C "$d" commit -q -m pr
    local out actual
    out="$(cd "$d" && env PYTHONPATH="$d/src" PYTHON="$PY" "$@" bash "$SCRIPT" main 2>&1)"
    actual=$?
    if [ "$actual" -eq "$expected" ]; then
        echo "PASS - $name"
    else
        echo "FAIL - $name (expected exit $expected, got $actual)"
        echo "$out" | sed 's/^/    /'
        failures=$((failures + 1))
    fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Docs only: nothing under src/ changed.
d="$TMP/docs"; new_repo "$d"
echo "more docs" >> "$d/README.md"
run_case "docs-only change passes early" 0 "$d"

# Dropping an unused import is cleanup; no test demanded.
d="$TMP/cleanup"; new_repo "$d"
sed -i.bak '/^import os$/d' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
run_case "cleanup-only src change passes" 0 "$d"

# Behavior change without any test.
d="$TMP/notest"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
run_case "src change without a test fails" 1 "$d"

# The good PR: the test encodes the new behavior, so it is red on base.
d="$TMP/proven"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
sed -i.bak 's/== 41/== 42/' "$d/tests/test_answer.py" && rm "$d/tests/test_answer.py.bak"
run_case "test that fails on base passes" 0 "$d"

# The Codex shape: the test accepts both the old and the new answer.
d="$TMP/vacuous"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
sed -i.bak 's/== 41/in (41, 42)/' "$d/tests/test_answer.py" && rm "$d/tests/test_answer.py.bak"
run_case "test that is green on base too fails" 1 "$d"

# A new module plus a test for it: on base the import errors, which counts as red.
d="$TMP/newmodule"; new_repo "$d"
printf 'def extra():\n    return 1\n' > "$d/src/mcp_memory_service/extra.py"
printf 'from mcp_memory_service.extra import extra\n\n\ndef test_extra():\n    assert extra() == 1\n' > "$d/tests/test_extra.py"
run_case "new module with new test passes via collection error" 0 "$d"

# A space in a test path must not split it into two pytest arguments.
d="$TMP/spaced"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
printf 'from mcp_memory_service import answer\n\n\ndef test_answer():\n    assert answer() == 42\n' > "$d/tests/test_spaced answer.py"
run_case "test path with a space is passed whole" 0 "$d"

# Interpreter that does not import this tree: refuse rather than report a verdict.
d="$TMP/wrongenv"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
sed -i.bak 's/== 41/== 42/' "$d/tests/test_answer.py" && rm "$d/tests/test_answer.py.bak"
git_c -C "$d" add -A && git_c -C "$d" commit -q -m pr
out="$(cd "$d" && env -u PYTHONPATH PYTHON="$PY" bash "$SCRIPT" main 2>&1)"
actual=$?
if [ "$actual" -eq 2 ]; then
    echo "PASS - interpreter outside the tree is refused"
else
    echo "FAIL - interpreter outside the tree is refused (expected exit 2, got $actual)"
    echo "$out" | sed 's/^/    /'
    failures=$((failures + 1))
fi

# A test module that skips itself at collection (importorskip of an optional extra
# the job does not install): still a failure, but the message names the module
# instead of claiming nothing was collected. The exit code is 1 either way, so the
# case checks the output (#1709).
d="$TMP/modskip"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
printf 'import pytest\n\npytest.importorskip("no_such_module_1709")\n\n\ndef test_answer():\n    assert True\n' > "$d/tests/test_answer.py"
git_c -C "$d" add -A && git_c -C "$d" commit -q -m pr
out="$(cd "$d" && env PYTHONPATH="$d/src" PYTHON="$PY" bash "$SCRIPT" main 2>&1)"
actual=$?
if [ "$actual" -eq 1 ] && grep -q "not installed in this job: no_such_module_1709" <<<"$out"; then
    echo "PASS - module skipped at collection fails and names the module"
else
    echo "FAIL - module skipped at collection fails and names the module (got exit $actual)"
    echo "$out" | sed 's/^/    /'
    failures=$((failures + 1))
fi

# Same, with importorskip(reason=...): the module name is gone from the skip line,
# so the failure quotes the reason instead of claiming nothing was collected.
d="$TMP/modskipreason"; new_repo "$d"
sed -i.bak 's/return 41/return 42/' "$d/src/mcp_memory_service/__init__.py" && rm "$d/src/mcp_memory_service/__init__.py.bak"
printf 'import pytest\n\npytest.importorskip("no_such_module_1709", reason="needs the 1709 extra")\n\n\ndef test_answer():\n    assert True\n' > "$d/tests/test_answer.py"
git_c -C "$d" add -A && git_c -C "$d" commit -q -m pr
out="$(cd "$d" && env PYTHONPATH="$d/src" PYTHON="$PY" bash "$SCRIPT" main 2>&1)"
actual=$?
if [ "$actual" -eq 1 ] && grep -q "needs the 1709 extra" <<<"$out" && ! grep -q "no tests collected" <<<"$out"; then
    echo "PASS - module skipped with a custom reason fails and quotes the reason"
else
    echo "FAIL - module skipped with a custom reason fails and quotes the reason (got exit $actual)"
    echo "$out" | sed 's/^/    /'
    failures=$((failures + 1))
fi

if [ "$failures" -gt 0 ]; then
    echo "$failures test(s) failed"
    exit 1
fi
echo "All tests passed"
