#!/usr/bin/env python3
"""Merge changelog.d/ fragments into the [Unreleased] section of CHANGELOG.md.

Run it while preparing a release, before the version bump:

    python3 scripts/release/collect_changelog.py            # merge and delete fragments
    python3 scripts/release/collect_changelog.py --dry-run  # print the result, change nothing

Fragments are named <number>.<category>.md and hold one markdown list item each
(changelog.d/README.md has the format). Each category maps to a section heading;
entries are appended to that section if it already exists under [Unreleased], and
the section is created in canonical order if it does not.

Deleting the fragments is the point: what is merged must not be merged twice.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_DIR = REPO_ROOT / "changelog.d"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# Order is the order these sections appear in a released version block.
SECTIONS = ["added", "fixed", "removed", "internal"]
HEADING = {
    "added": "### Added",
    "fixed": "### Fixed",
    "removed": "### Removed",
    "internal": "### Internal",
}
FRAGMENT_RE = re.compile(r"^(?P<number>[^.]+)\.(?P<category>added|fixed|removed|internal)\.md$")


def read_fragments() -> tuple[dict[str, list[str]], list[Path]]:
    """Group fragment bodies by category, sorted by number so output is stable."""
    by_category: dict[str, list[str]] = {c: [] for c in SECTIONS}
    used: list[Path] = []
    for path in sorted(FRAGMENT_DIR.glob("*.md"), key=lambda p: (p.name.split(".")[0].zfill(8), p.name)):
        match = FRAGMENT_RE.match(path.name)
        if not match:
            if path.name != "README.md":
                print(f"skipping {path.name}: not <number>.<category>.md", file=sys.stderr)
            continue
        body = path.read_text().strip()
        if not body:
            print(f"skipping {path.name}: empty", file=sys.stderr)
            continue
        by_category[match.group("category")].append(body)
        used.append(path)
    return by_category, used


def split_unreleased(text: str) -> tuple[str, str, str]:
    """Return (before, unreleased_body, after) around the [Unreleased] section."""
    start = re.search(r"^## \[Unreleased\][^\n]*\n", text, re.MULTILINE)
    if not start:
        raise SystemExit("CHANGELOG.md has no '## [Unreleased]' heading")
    rest = text[start.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    end = start.end() + (nxt.start() if nxt else len(rest))
    return text[: start.end()], text[start.end() : end], text[end:]


def merge_section(body: str, heading: str, entries: list[str]) -> str:
    """Append entries under `heading` inside the [Unreleased] body, creating it if needed."""
    pattern = re.compile(rf"^{re.escape(heading)}\s*\n", re.MULTILINE)
    found = pattern.search(body)
    block = "\n".join(entries)
    if not found:
        return body.rstrip("\n") + f"\n\n{heading}\n\n{block}\n"
    nxt = re.search(r"^###? ", body[found.end() :], re.MULTILINE)
    cut = found.end() + (nxt.start() if nxt else len(body) - found.end())
    head, tail = body[:cut], body[cut:]
    return head.rstrip("\n") + f"\n{block}\n\n" + tail.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the merged section, write nothing")
    args = parser.parse_args()

    by_category, used = read_fragments()
    if not used:
        print("no fragments to collect")
        return 0

    text = CHANGELOG.read_text()
    before, body, after = split_unreleased(text)
    for category in SECTIONS:
        if by_category[category]:
            body = merge_section(body, HEADING[category], by_category[category])

    merged = before + body.rstrip("\n") + "\n\n" + after.lstrip("\n")

    if args.dry_run:
        print(before.rstrip("\n"))
        print(body.rstrip("\n"))
        print(f"\n--- would delete {len(used)} fragment(s) ---")
        for path in used:
            print(f"  {path.relative_to(REPO_ROOT)}")
        return 0

    CHANGELOG.write_text(merged)
    for path in used:
        path.unlink()
    counts = ", ".join(f"{len(v)} {k}" for k, v in by_category.items() if v)
    print(f"merged {len(used)} fragment(s) into [Unreleased] ({counts}); fragments deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
