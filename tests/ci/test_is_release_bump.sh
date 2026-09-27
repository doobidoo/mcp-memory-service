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

# Greptile P1: release bump that ALSO DELETES an unrelated file must be rejected.
# Deleted files appear as "+++ /dev/null"; the deleted path is only on "--- a/".
check "version plus deleted non-release file" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/src/mcp_memory_service/secret_module.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def important():
-    return 42
-
'

# Deleting a release-set file alongside the bump is still fine (path is allowed).
check "version plus deleted release file allowed" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,1 +1,1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/README.md
+++ /dev/null
@@ -1,2 +0,0 @@
-# Old readme
-
'

# --- _version.py content must be a bump, not smuggled behavior (#1250 review) ---

# A function added inside _version.py rides the release-file allowlist but is real
# behavior — must be REJECTED even though the only path is _version.py.
check "version file with added function rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,2 +1,6 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
+
+def parse(v):
+    return tuple(int(x) for x in v.split("."))
'

# A moved/reflowed comment next to the bump is formatting noise, not behavior —
# must stay ACCEPTED so the content check is not brittle.
check "version bump with moved comment accepted" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1,3 +1,3 @@
-# version of the package
-__version__ = "11.12.0"
+__version__ = "11.13.0"
+# version of the package
'

# A statement chained after the assignment with ";" must be REJECTED — the
# content check anchors the whole line, so a prefix match can't let it ride.
check "version assignment with chained statement rejected" 1 'diff --git a/src/mcp_memory_service/_version.py b/src/mcp_memory_service/_version.py
--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"; run_new_behavior()
'

# Deleting an allowlisted file whose removed lines are REAL content (not just
# comments/blanks) must still be ACCEPTED — the deletion must not be judged as
# _version.py content. Guards the "+++ /dev/null" flag-reset regression.
check "version bump plus deleted allowlisted file with real content accepted" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
--- a/site/index.html
+++ /dev/null
@@ -1,2 +0,0 @@
-<html>
-</html>
'

# Deleting _version.py itself is NOT a bump — it removes the version module
# (breaks imports and release tooling) and must still require tests. Its path
# is recorded as changed, but it must be present (added/modified), not deleted.
check "deleted version file rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ /dev/null
@@ -1 +0,0 @@
-__version__ = "11.12.0"
'

# A PEP 526 type annotation on the version is a legitimate release form.
check "version bump with type annotation accepted" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__: str = "11.12.0"
+__version__: str = "11.13.0"
'

# A binary change alongside the bump cannot be inspected and must be rejected.
check "version bump plus binary change rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.12.0"
+__version__ = "11.13.0"
diff --git a/site/index.html b/site/index.html
Binary files a/site/index.html and b/site/index.html differ
'

# collect_changelog.py merges every changelog.d fragment into CHANGELOG.md and
# deletes it, so a release diff always deletes one file per fragment accumulated
# since the last release (#1276). Before this was allowed, the deletions read as
# paths outside the release set and every release failed the coverage check.
check "version bump plus collected changelog fragments" 0 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.13.0"
+__version__ = "11.14.0"
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,1 +1,2 @@
+## [11.14.0]
 ## [11.13.0]
--- a/changelog.d/1305.fixed.md
+++ /dev/null
@@ -1,1 +0,0 @@
-- **GET /mcp/health no longer discloses storage statistics.**
--- a/changelog.d/1297.added.md
+++ /dev/null
@@ -1,1 +0,0 @@
-- **agent_id filter on memory_search.**
'

# The fragment allowance is scoped to changelog.d/*.md — a deleted file that
# merely starts with the directory name is still outside the release set.
check "deleted file outside changelog.d rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.13.0"
+__version__ = "11.14.0"
--- a/changelog.d/nested/1305.fixed.md
+++ /dev/null
@@ -1,1 +0,0 @@
-- entry
'

# Greptile P2: the allowance must match the collector'"'"'s fragment format, not any
# markdown in the directory. changelog.d/README.md documents the format and is
# never collected — deleting it alongside a bump is a real change needing gates.
check "deleted changelog.d README rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.13.0"
+__version__ = "11.14.0"
--- a/changelog.d/README.md
+++ /dev/null
@@ -1,1 +0,0 @@
-# Changelog fragments
'

# A name the collector would not read is not a fragment either: it would be
# dropped silently at release time, which is what the gates exist to catch.
check "deleted changelog.d file with unknown category rejected" 1 '--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.13.0"
+__version__ = "11.14.0"
--- a/changelog.d/1305.notacategory.md
+++ /dev/null
@@ -1,1 +0,0 @@
-- entry
'

# Collecting deletes fragments; it never adds one. An ADDED fragment beside a
# bump is a normal change and must still face the gates.
check "added changelog fragment rejected" 1 '--- /dev/null
+++ b/changelog.d/1311.fixed.md
@@ -0,0 +1 @@
+- **Something new.**
--- a/src/mcp_memory_service/_version.py
+++ b/src/mcp_memory_service/_version.py
@@ -1 +1 @@
-__version__ = "11.13.0"
+__version__ = "11.14.0"
'

if [ "$failures" -gt 0 ]; then
    echo "$failures test(s) failed"
    exit 1
fi
echo "All tests passed"