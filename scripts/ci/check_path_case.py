#!/usr/bin/env python3
"""Flag string literals that name a tracked file in the wrong letter case.

macOS and Windows resolve paths case-insensitively, Linux does not. A test that
read the PR template under its lower-case name passed on a Mac and died with
FileNotFoundError on the Linux runner, because the tracked file is
.github/PULL_REQUEST_TEMPLATE.md (PR #1571). Nothing local caught that.

For every quoted literal in the given files this compares the literal (a
leading ./ stripped), and failing that its last path segment, against
`git ls-files`. It reports a literal only when a
tracked path or file name matches it ignoring case but not exactly, so names of
files that exist only at run time (temp databases, reports) never count.
Literals without a file extension are skipped: words like "Default" are not
paths. A line that means a literal as a label, not a path, opts out with a
`path-case: ok` comment.

Usage: check_path_case.py FILE...   (exit 1 on any finding)
Standard library only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LITERAL = re.compile(r"""["']([\w./-]+\.\w+)["']""")
OPT_OUT = "path-case: ok"


def case_mismatches(source: str, tracked: list[str]) -> list[tuple[str, str]]:
    """(literal, tracked spelling) for each literal that differs from a tracked path only in case."""
    exact = set(tracked)
    exact_names = {p.rsplit("/", 1)[-1] for p in tracked}
    by_lower = {p.lower(): p for p in tracked}
    names_by_lower = {name.lower(): name for name in exact_names}
    found = []
    literals = (
        lit for line in source.splitlines() if OPT_OUT not in line for lit in LITERAL.findall(line)
    )
    for literal in literals:
        # A leading ./ names the same tracked path; other prefixes (../repo/,
        # absolute paths) fall back to the file name below.
        path = literal.removeprefix("./")
        if path in exact:
            continue
        if path.lower() in by_lower:
            found.append((literal, by_lower[path.lower()]))
            continue
        name = path.rsplit("/", 1)[-1]
        if name not in exact_names and name.lower() in names_by_lower:
            found.append((literal, names_by_lower[name.lower()]))
    return found


def main(paths: list[str]) -> int:
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    status = 0
    for path in paths:
        for literal, spelling in case_mismatches(Path(path).read_text(errors="replace"), tracked):
            print(f"CASE {path}: '{literal}' is tracked as '{spelling}'")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
