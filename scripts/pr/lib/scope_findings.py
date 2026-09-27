#!/usr/bin/env python3
"""Keep only the complexity findings that name a function the diff touched.

Reads the model's report on stdin and the touched-function names as arguments
(as produced by touched_functions.py). Prints the findings that survive, one
per line, and exits 1 when none do -- so the caller can branch on the exit
status instead of testing for an empty string.

The model is asked for `FunctionName: Score X - Reason` but does not always
comply exactly: it prefixes bullets, wraps names in backticks, and sometimes
qualifies them as `Class.method`. Matching is therefore on the last dotted
segment, case-insensitively, which is the part both sides agree on.
"""

import re
import sys

SCORE = re.compile(r"score\s*(9|10)\b", re.IGNORECASE)
NAME = re.compile(r"^[\s\-*>`]*([A-Za-z_][A-Za-z0-9_.]*)\s*[:(]")


def leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def main() -> int:
    touched = {leaf(n) for n in sys.argv[1:] if n}
    if not touched:
        return 1

    kept = []
    for line in sys.stdin.read().splitlines():
        if not SCORE.search(line):
            continue
        m = NAME.match(line)
        if not m:
            # A finding we cannot attribute to a name is reported rather than
            # dropped: silently swallowing it would turn a parsing gap into a
            # clean gate.
            kept.append(line.strip())
            continue
        if leaf(m.group(1)) in touched:
            kept.append(line.strip())

    for line in kept:
        print(line)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
