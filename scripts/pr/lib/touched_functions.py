#!/usr/bin/env python3
"""Print the names of the functions a diff actually touches.

The quality gate scores whole files, so any change to a file that already
contains a function above the complexity budget failed the gate regardless of
what the change did -- it punished proximity rather than the change (#1118).
This narrows it: a finding only counts when the diff touched the function it
names.

Usage:
    touched_functions.py --staged <file.py>
    touched_functions.py --range <base>...<head> <file.py>

Prints one name per line. A change outside any function (imports, module-level
constants, a new decorator) prints the sentinel ``__module__`` so the caller can
tell "nothing touched" from "module scope touched" -- the latter still deserves
a look, it just cannot be attributed to a function.

Exits 0 with no output when the file has no touched function, and also when the
file cannot be parsed: a syntax error is the compiler's job to report, not this
helper's, and failing the gate on it would only hide the real message.
"""

import argparse
import ast
import re
import subprocess
import sys

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_lines(diff: str) -> set[int]:
    """Line numbers touched on the new side of a unified diff."""
    lines: set[int] = set()
    new_lineno = 0
    in_hunk = False
    for line in diff.splitlines():
        m = HUNK.match(line)
        if m:
            new_lineno = int(m.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            lines.add(new_lineno)
            new_lineno += 1
        elif line.startswith("-"):
            # Deleted lines have no position on the new side. The line that
            # follows them does, and it is already covered by the context or
            # addition that comes next.
            continue
        elif line.startswith("\\"):
            continue
        else:
            new_lineno += 1
    return lines


def function_ranges(source: str) -> list[tuple[str, int, int]]:
    """(qualified name, first line, last line) for every function in the file.

    The first line is the decorator when there is one: changing a decorator
    changes the function's behaviour, so it belongs to the function's range.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    ranges: list[tuple[str, int, int]] = []

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min(
                    [child.lineno] + [d.lineno for d in child.decorator_list]
                )
                ranges.append((prefix + child.name, start, child.end_lineno))
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")

    walk(tree, "")
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true")
    group.add_argument("--range", dest="rev_range")
    parser.add_argument("path")
    args = parser.parse_args()

    cmd = ["git", "diff", "--unified=0"]
    if args.staged:
        cmd.append("--cached")
    else:
        cmd.append(args.rev_range)
    cmd += ["--", args.path]

    diff = subprocess.run(cmd, capture_output=True, text=True).stdout
    touched = changed_lines(diff)
    if not touched:
        return 0

    try:
        with open(args.path, encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return 0

    names = []
    covered: set[int] = set()
    for name, start, end in function_ranges(source):
        span = set(range(start, end + 1))
        covered |= span
        if span & touched:
            names.append(name)

    if touched - covered:
        names.append("__module__")

    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
