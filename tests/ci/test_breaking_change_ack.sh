#!/usr/bin/env bash
# Covers scripts/pr/lib/breaking_change_ack.py, the acknowledgement that lets check 4
# of the quality gate report a deliberate breaking change without blocking (#1311).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$REPO_ROOT/scripts/pr/lib/breaking_change_ack.py"
failures=0

check() {
    local name="$1" expected_code="$2" expected_reason="$3" text="$4"
    local reason code
    reason=$(printf '%s' "$text" | python3 "$HELPER")
    code=$?
    if [ "$code" -eq "$expected_code" ] && [ "$reason" = "$expected_reason" ]; then
        echo "PASS - $name"
    else
        echo "FAIL - $name (expected exit $expected_code '$expected_reason', got $code '$reason')"
        failures=$((failures + 1))
    fi
}

check "trailer in a commit message" 0 "GHSA-7w86-2vmv-fqwm, /mcp/health leaked statistics" 'fix(security): strip statistics from /mcp/health

Breaking-Change-Acknowledged: GHSA-7w86-2vmv-fqwm, /mcp/health leaked statistics
'

check "line in a PR body" 0 "the field was the leak" '## Why

Breaking-Change-Acknowledged: the field was the leak

More text.'

check "no marker" 1 "" 'fix: something

This is not a breaking change.'

check "marker without a reason" 1 "" 'Breaking-Change-Acknowledged:
'

check "marker with only spaces" 1 "" 'Breaking-Change-Acknowledged:    
'

check "marker mentioned inside a sentence" 1 "" 'Use a Breaking-Change-Acknowledged: line to skip this.'

check "empty input" 1 "" ''

if [ "$failures" -gt 0 ]; then
    echo "$failures test(s) failed"
    exit 1
fi
echo "All tests passed"
