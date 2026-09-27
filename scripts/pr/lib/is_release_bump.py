#!/usr/bin/env python3
"""Decide whether a unified diff is a release version bump only.

The test-coverage checks in quality gates demand test files whenever Python files 
change. Release version bumps touch _version.py but add no new behavior to test.
The gates were unpassable for releases, requiring manual skip-prove-fix labels.

A change counts as a release bump when:
- The only Python file changed is src/mcp_memory_service/_version.py
- All other changed files are in the release workflow set: pyproject.toml,
  uv.lock, CHANGELOG.md, README.md, site/index.html,
  claude-hooks/.claude-plugin/plugin.json, or a changelog.d/*.md fragment
  (collect_changelog.py merges them into CHANGELOG.md and deletes them, so a
  release necessarily deletes every fragment accumulated since the last one)

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
# A binary change ("Binary files a/... and b/... differ") has no reviewable text
# lines, so the content check can't inspect it. A release bump never changes a
# binary, so any binary section disqualifies the diff.
BINARY_RE = re.compile(r"^Binary files .* differ$")

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

# scripts/release/collect_changelog.py merges every changelog.d/<n>.<cat>.md
# fragment into CHANGELOG.md and deletes it, so a release diff always carries a
# deletion per fragment accumulated since the last release (#1276). Without this
# the deletions read as "paths outside the release set" and every release since
# the fragment workflow landed fails the test-coverage check.
#
# The shape must stay in step with FRAGMENT_RE in collect_changelog.py and in
# check_changelog_entry.sh: only a name those accept is a fragment the release
# actually consumed. A looser pattern would let an unrelated markdown file in
# this directory — changelog.d/README.md, or a doc someone parked there — ride
# the release allowlist and bypass both gates.
CHANGELOG_FRAGMENT_RE = re.compile(
    r"^changelog\.d/[^./]+\.(added|fixed|removed|internal)\.md$"
)

# A changed line inside _version.py is only allowed to be a __version__
# assignment, a blank line, or a comment. Anything else (a def, an import, any
# other statement — or a statement chained after the assignment with ";") means
# the version file grew real behavior and must be tested. The pattern anchors the
# WHOLE line: __version__ = "<literal>" with nothing trailing, so a prefix match
# can't let `; run_new_behavior()` ride along. An optional PEP 526 type
# annotation (`__version__: str = "..."`) is a legitimate release form and is
# allowed; the annotation is restricted to a bare identifier so it can't smuggle
# a call.
_VERSION_ASSIGN_RE = re.compile(
    r"""^__version__\s*(?::\s*[A-Za-z_][A-Za-z0-9_]*\s*)?=\s*["'][^"']*["']\s*$"""
)


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
        # A new file section starts with "diff --git" (full git diff) or, when
        # that header is absent, with the "--- a/<path>" pre-image header. Reset
        # on either so a deletion (post-image "+++ /dev/null", not "+++ b/")
        # can never inherit the previous file's flag and get its removed lines
        # judged as _version.py content. The "+++ b/<path>" header then confirms
        # which file the following hunk edits.
        if line.startswith("diff --git") or line.startswith("--- "):
            in_version_file = False
            continue
        if line.startswith("+++ b/"):
            in_version_file = line[len("+++ b/"):] == VERSION_FILE
            continue
        if line.startswith("+++ ") or line.startswith("@@") or line.startswith("index "):
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
    deleted_files = set()

    lines = diff.splitlines()
    for i, line in enumerate(lines):
        # A binary change can't be inspected line-by-line; a release bump never
        # touches binaries, so disqualify the whole diff.
        if BINARY_RE.match(line):
            return False
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
            deleted_files.add(m.group(1))
    
    if not changed_files:
        return False
    
    # Must include _version.py as a *present* file (added/modified) for it to be
    # a release bump. A diff that DELETES _version.py records its path here too,
    # but removing the version module is not a bump — it breaks imports and
    # release tooling and must still require tests.
    if VERSION_FILE not in changed_files or VERSION_FILE in deleted_files:
        return False
    
    # Check all files against allowed sets
    for path in changed_files:
        if path == VERSION_FILE:
            continue  # _version.py is always allowed
        elif path in RELEASE_FILES:
            continue  # Release workflow files are allowed
        elif CHANGELOG_FRAGMENT_RE.match(path) and path in deleted_files:
            # A fragment the release just collected. Only a DELETION qualifies:
            # collecting removes fragments, it never adds or edits one, so an
            # added or modified fragment is a normal change that needs its gates.
            continue
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