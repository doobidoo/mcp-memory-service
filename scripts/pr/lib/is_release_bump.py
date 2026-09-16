#!/usr/bin/env python3
"""Decide whether a unified diff is a release version bump only.

The test-coverage checks in quality gates demand test files whenever Python files 
change. Release version bumps touch _version.py but add no new behavior to test.
The gates were unpassable for releases, requiring manual skip-prove-fix labels.

A change counts as a release bump when:
- The only Python file changed is src/mcp_memory_service/_version.py
- All other changed files are in the release workflow set: pyproject.toml, 
  uv.lock, CHANGELOG.md, README.md, site/index.html, 
  claude-hooks/.claude-plugin/plugin.json

Any other Python file or path outside the release set makes this a normal change
that requires tests.

Reads a unified diff on stdin. Exit 0 = release-bump-only, 1 = needs tests.
"""
import re
import sys

# Added/modified files: the post-image path on the "+++ b/<path>" header.
ADD_RE = re.compile(r"^\+\+\+ b/(.*)$")
# Deleted files: the pre-image path on the "--- a/<path>" header. A deletion is
# marked by the post-image header being "+++ /dev/null" on the following line,
# so we must also account for the deleted path — it does not appear as "+++ b/".
DEL_FROM_RE = re.compile(r"^--- a/(.*)$")
DEV_NULL_RE = re.compile(r"^\+\+\+ /dev/null$")

# Files that the release workflow is allowed to modify (besides _version.py)
RELEASE_FILES = {
    "pyproject.toml",
    "uv.lock", 
    "CHANGELOG.md",
    "README.md",
    "site/index.html",
    "claude-hooks/.claude-plugin/plugin.json"
}

def is_release_bump(diff: str) -> bool:
    """Check if diff is a release version bump only.

    Considers added, modified AND deleted files. A release PR that also deletes
    an unrelated file must NOT be exempted: deleted files show up as
    "+++ /dev/null", so their path is only recoverable from the "--- a/<path>"
    header. Any path outside the release set (including a deletion) disqualifies.
    """
    changed_files = []

    lines = diff.splitlines()
    for i, line in enumerate(lines):
        m = ADD_RE.match(line)
        if m:
            path = m.group(1)
            if path != "/dev/null":
                changed_files.append(path)
            continue
        # Detect a deletion: "--- a/<path>" immediately followed by "+++ /dev/null".
        m = DEL_FROM_RE.match(line)
        if m and i + 1 < len(lines) and DEV_NULL_RE.match(lines[i + 1]):
            changed_files.append(m.group(1))
    
    if not changed_files:
        return False
    
    # Must include _version.py for it to be a release bump
    version_file = "src/mcp_memory_service/_version.py"
    if version_file not in changed_files:
        return False
    
    # Check all files against allowed sets
    for path in changed_files:
        if path == version_file:
            continue  # _version.py is always allowed
        elif path in RELEASE_FILES:
            continue  # Release workflow files are allowed
        else:
            # Any other file (including other Python files or a deleted file)
            # disqualifies this as release-only.
            return False
    
    return True


if __name__ == "__main__":
    sys.exit(0 if is_release_bump(sys.stdin.read()) else 1)