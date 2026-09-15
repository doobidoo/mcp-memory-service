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

FILE_RE = re.compile(r"^\+\+\+ b/(.*)$")

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
    """Check if diff is a release version bump only."""
    changed_files = []
    
    for line in diff.splitlines():
        m = FILE_RE.match(line)
        if m:
            path = m.group(1)
            changed_files.append(path)
    
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
            # Any other file (including other Python files) disqualifies this as release-only
            return False
    
    return True


if __name__ == "__main__":
    sys.exit(0 if is_release_bump(sys.stdin.read()) else 1)