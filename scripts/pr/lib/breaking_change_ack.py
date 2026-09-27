#!/usr/bin/env python3
"""Find a deliberate breaking change acknowledgement in PR text or commit messages.

Check 4 of quality_gate.sh blocks on any breaking change the model reports. Some
breaking changes are the fix itself: an information-disclosure advisory is fixed by
removing fields from a response, which is a breaking change by any definition (#1311).
Without a way to say so, a correct security fix could never pass the gate.

A line of the form

    Breaking-Change-Acknowledged: <reason>

in the PR body or a commit message acknowledges it. The reason is required, so the
acknowledgement says why; a bare marker does not count.

Reads the text on stdin. Prints the reason and exits 0 when an acknowledgement is
found, exits 1 otherwise.
"""
import re
import sys

MARKER_RE = re.compile(r"^\s*Breaking-Change-Acknowledged:[ \t]*(\S.*?)\s*$", re.MULTILINE)


def find_reason(text: str) -> str | None:
    match = MARKER_RE.search(text)
    return match.group(1) if match else None


def main() -> int:
    reason = find_reason(sys.stdin.read())
    if reason is None:
        return 1
    print(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
