#!/usr/bin/env bash
# Covers scripts/pr/lib/is_release_bump.py, the rule that exempts release version bumps
# from test-coverage requirements in quality gates.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$REPO_ROOT/scripts/pr/lib/is_release_bump.py"
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

# ACCEPT cases - exit 0

# Only _version.py changed
check "version file only" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
'

# _version.py + pyproject.toml + CHANGELOG.md (typical release)
check "typical release files" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -4,1 +4,1 @@
-version = "11.12.0"
+version = "11.13.0"
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,1 +1,3 @@
+# 11.13.0
+- New feature
+
 # 11.12.0
'

# Full release workflow files (all allowed paths)
check "full release workflow" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -4,1 +4,1 @@
-version = "11.12.0"
+version = "11.13.0"
--- a/uv.lock
+++ b/uv.lock
@@ -1,1 +1,1 @@
-# Lock file content
+# Updated lock file
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,1 +1,3 @@
+# 11.13.0
+- Release notes
+
 # 11.12.0
--- a/README.md
+++ b/README.md
@@ -10,1 +10,1 @@
-Version: 11.12.0
+Version: 11.13.0
--- a/site/index.html
+++ b/site/index.html
@@ -1,1 +1,1 @@
-<title>MCP Memory v11.12.0</title>
+<title>MCP Memory v11.13.0</title>
--- a/claude-hooks/.claude-plugin/plugin.json
+++ b/claude-hooks/.claude-plugin/plugin.json
@@ -2,1 +2,1 @@
-  "version": "11.12.0"
+  "version": "11.13.0"
'

# REJECT cases - exit 1

# _version.py + another Python file in src/
check "version plus src python" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/src/mcp_memory_service/config.py
+++ b/src/mcp_memory_service/config.py
@@ -10,1 +10,2 @@
 def load_config():
+    print("debug message")
     return {}
'

# _version.py + a file outside the release set
check "version plus non-release file" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/docs/api.md
+++ b/docs/api.md
@@ -1,1 +1,2 @@
 # API Documentation
+New section added
'

# Only a non-release file (no _version.py at all)
check "non-release file only" 1 '--- a/tests/test_something.py
+++ b/tests/test_something.py
@@ -1,1 +1,2 @@
 def test_feature():
+    assert True
'

# Mixed: _version.py + release files + non-release file
check "mixed with non-release" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -4,1 +4,1 @@
-version = "11.12.0"
+version = "11.13.0"
--- a/src/mcp_memory_service/handlers.py
+++ b/src/mcp_memory_service/handlers.py
@@ -50,1 +50,2 @@
     async def handle_request():
+        logger.debug("Processing request")
         pass
'

# Python file outside src/ (should be rejected)
check "python outside src" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/scripts/build.py
+++ b/scripts/build.py
@@ -1,1 +1,2 @@
 #!/usr/bin/env python3
+import sys
'

if [ "$failures" -gt 0 ]; then
    echo "$failures test(s) failed"
    exit 1
fi
echo "All tests passed"