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

VERSION_FILE = "src/mcp_memory_service/_version.py"

# A changed line inside _version.py is only allowed to be a __version__
# assignment, a blank line, or a comment. Anything else (a def, an import, any
# other statement) means the version file grew real behavior and must be tested.
_VERSION_ASSIGN_RE = re.compile(r"^__version__\s*=")


def _version_change_is_bump_only(diff: str) -> bool:
    """Whether every +/- line inside _version.py's hunks is version-bump noise.

    Scans the diff sections and, within the ``_version.py`` file only, inspects
    each added (``+``) and removed (``-``) content line. Allowed: a
    ``__version__ = ...`` assignment, a blank line, or a comment. A moved comment
    or reflowed blank line stays accepted (so the check is not brittle on
    formatting); an added function/import/statement is rejected.

    File headers (``+++``/``---``) and hunk headers (``@@``) are not content and
    are skipped.
    """
    in_version_file = False
    for line in diff.splitlines():
        # Track which file's hunks we are inside. "diff --git" or the "+++ b/"
        # header switches the current file.
        if line.startswith("+++ b/"):
            in_version_file = line[len("+++ b/"):] == VERSION_FILE
            continue
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@") \
                or line.startswith("diff --git") or line.startswith("index "):
            continue
        if not in_version_file:
            continue
        if not line or line[0] not in "+-":
            continue  # context line
        content = line[1:].strip()
        if content == "":
            continue  # blank line added/removed — formatting noise
        if content.startswith("#"):
            continue  # comment added/removed/moved
        if _VERSION_ASSIGN_RE.match(content):
            continue  # the version assignment itself
        # Anything else — a def, import, or any statement — is real behavior.
        return False
    return True


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
    if VERSION_FILE not in changed_files:
        return False
    
    # Check all files against allowed sets
    for path in changed_files:
        if path == VERSION_FILE:
            continue  # _version.py is always allowed
        elif path in RELEASE_FILES:
            continue  # Release workflow files are allowed
        else:
            # Any other file (including other Python files or a deleted file)
            # disqualifies this as release-only.
            return False

    # File set is release-only, but _version.py itself must carry only a version
    # bump — not new behavior smuggled into the exempted file.
    if not _version_change_is_bump_only(diff):
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if is_release_bump(sys.stdin.read()) else 1)