#!/usr/bin/env bash
# Covers scripts/pr/lib/is_cleanup_only.py, the rule that lets the quality gate's
# test-coverage check pass a dead-code removal while still blocking added behavior.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$REPO_ROOT/scripts/pr/lib/is_cleanup_only.py"
failures=0

check() {
    local name="$1" expected="$2" diff="$3"
    printf '%s' "$diff" | python3 "$HELPER"
    local actual=$?
    if [ "$actual" -eq "$expected" ]; then
        echo "PASS - $name"
    else
        echo "FAIL - $name (expected exit $expected, got $actual)"
        failures=$((failures + 1))
    fi
}

# Shortening an import list: the added line is an import.
check "trimmed import list" 0 '--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-from typing import Dict, List, Optional
+from typing import List
'

# Dropping an unused binding: the added line is contained in the removed one.
check "dropped unused binding" 0 '--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-            loop = asyncio.get_running_loop()
+            asyncio.get_running_loop()
'

# Pure deletion.
check "pure deletion" 0 '--- a/x.py
+++ b/x.py
@@ -1,2 +1,0 @@
-import traceback
-import os
'

# A genuinely new statement must still demand a test.
check "new statement" 1 '--- a/x.py
+++ b/x.py
@@ -1,1 +1,2 @@
 def f():
+    return compute_something()
'

# A changed condition is a behavior change, not cleanup.
check "changed condition" 1 '--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-    if a > b:
+    if a >= b:
'

# The same edit in a non-Python file must not make a Python change look clean.
check "non-python ignored" 0 '--- a/x.md
+++ b/x.md
@@ -1,1 +1,1 @@
-old prose
+entirely new prose
'

# A cleanup hunk plus a behavior hunk in one diff is still a behavior change.
check "mixed diff blocks" 1 '--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-from typing import Dict, List
+from typing import List
@@ -20,1 +20,2 @@
 def f():
+    side_effect()
'

if [ "$failures" -gt 0 ]; then
    echo "$failures test(s) failed"
    exit 1
fi
echo "All tests passed"
