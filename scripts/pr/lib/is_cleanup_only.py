#!/usr/bin/env python3
"""Decide whether a unified diff only removes code rather than adding behavior.

The test-coverage check in quality_gate.sh demands a test file whenever Python
files change. Dead-code removal cannot satisfy that: there is no new behavior to
cover, so the gate was unpassable for exactly the cleanup it keeps asking for.

A change counts as cleanup-only when every added line is either an import, or a
trimmed form of a line removed in the same hunk -- which is what shortening an
import list or dropping an unused binding looks like:

    -from typing import Dict, List, Optional      -    loop = asyncio.get_running_loop()
    +from typing import List                      +    asyncio.get_running_loop()

Anything else -- a genuinely new statement, a changed condition -- is a behavior
change and still needs a test.

Reads a unified diff on stdin. Exit 0 = cleanup-only, 1 = adds behavior.
"""
import re
import sys

FILE_RE = re.compile(r"^\+\+\+ b/(.*)$")
IMPORT_RE = re.compile(r"^(import |from \S+ import )")


def is_cleanup_only(diff: str) -> bool:
    in_python = False
    removed: list[str] = []
    added: list[str] = []
    verdict = True

    def flush() -> bool:
        # Every added line must be an import, or contained in something removed.
        for line in added:
            if IMPORT_RE.match(line):
                continue
            if any(line in r for r in removed):
                continue
            return False
        return True

    for raw in diff.splitlines():
        m = FILE_RE.match(raw)
        if m:
            verdict = verdict and flush()
            removed, added = [], []
            in_python = m.group(1).endswith(".py")
            continue
        if raw.startswith("@@"):
            verdict = verdict and flush()
            removed, added = [], []
            continue
        if not in_python:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            stripped = raw[1:].strip()
            if stripped and not stripped.startswith("#"):
                added.append(stripped)
        elif raw.startswith("-") and not raw.startswith("---"):
            stripped = raw[1:].strip()
            if stripped:
                removed.append(stripped)

    return verdict and flush()


if __name__ == "__main__":
    sys.exit(0 if is_cleanup_only(sys.stdin.read()) else 1)
