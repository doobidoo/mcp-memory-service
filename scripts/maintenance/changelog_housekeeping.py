#!/usr/bin/env python3
"""Changelog housekeeping: archive old CHANGELOG entries.

Usage:
    python scripts/maintenance/changelog_housekeeping.py [--dry-run]

Trigger criterion:
    - CHANGELOG.md > 150 lines

What it does:
    1. Keeps last 6-8 versions in CHANGELOG.md (current minor -2)
    2. Archives older entries to docs/archive/CHANGELOG-HISTORIC.md
    3. Validates no content is lost (version count before == after)

The README used to carry a "Previous Releases" list that this script trimmed in
the same run. That section was removed when the README was cut down; the README
now links CHANGELOG.md, so there is nothing there to maintain.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
ARCHIVE = REPO_ROOT / "docs" / "archive" / "CHANGELOG-HISTORIC.md"

# Config
KEEP_VERSIONS = 8  # max versions to keep in CHANGELOG.md
MIN_LINES_TRIGGER = 150  # skip housekeeping if CHANGELOG.md below this

# Regex for version headers: ## [10.26.1] - 2026-03-08
VERSION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]")
UNRELEASED_RE = re.compile(r"^## \[Unreleased\]", re.IGNORECASE)


def parse_changelog_blocks(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Parse CHANGELOG.md into header, unreleased block, and version blocks.

    Returns:
        (header, unreleased_block, [(version_string, block_text), ...])
    """
    lines = text.split("\n")
    header_lines: list[str] = []
    unreleased_lines: list[str] = []
    version_blocks: list[tuple[str, list[str]]] = []
    current_version: str | None = None
    current_lines: list[str] = []
    in_header = True
    in_unreleased = False

    for line in lines:
        if UNRELEASED_RE.match(line):
            in_header = False
            in_unreleased = True
            unreleased_lines.append(line)
            continue

        version_match = VERSION_RE.match(line)
        if version_match:
            in_header = False
            if in_unreleased:
                in_unreleased = False
            if current_version is not None:
                version_blocks.append((current_version, current_lines))
            current_version = version_match.group(1)
            current_lines = [line]
            continue

        if in_header:
            header_lines.append(line)
        elif in_unreleased:
            unreleased_lines.append(line)
        elif current_version is not None:
            current_lines.append(line)
        else:
            header_lines.append(line)

    # Don't forget the last block
    if current_version is not None:
        version_blocks.append((current_version, current_lines))

    header = "\n".join(header_lines)
    unreleased = "\n".join(unreleased_lines)
    blocks = [(v, "\n".join(ls)) for v, ls in version_blocks]
    return header, unreleased, blocks


def extract_archive_versions(text: str) -> set[str]:
    """Extract all version numbers already present in the archive."""
    return set(re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE))


def update_header_range(header: str, oldest_kept: str, newest_archived: str) -> str:
    """Update the header comment to reflect new version range.

    Args:
        oldest_kept: The oldest version still in CHANGELOG.md
        newest_archived: The newest version moved to the archive

    Replaces lines like:
        **Recent releases for MCP Memory Service (v10.25.0 and later)**
    with the correct oldest kept version, and updates the archive boundary
    to reference the newest archived version.
    """
    pattern = re.compile(
        r"\*\*Recent releases for MCP Memory Service \(v[\d.]+ and later\)\*\*"
    )
    replacement = f"**Recent releases for MCP Memory Service (v{oldest_kept} and later)**"
    new_header = pattern.sub(replacement, header)

    # Update the archive reference to point to the newest archived version
    # Pattern: **Versions vX.Y.Z and earlier** – See [...]
    ver_pattern = re.compile(
        r"\*\*Versions v[\d.]+ and earlier\*\*"
    )
    new_header = ver_pattern.sub(
        f"**Versions v{newest_archived} and earlier**", new_header
    )
    return new_header


def update_archive_header_boundary(archive_header: str, newest_archived: str) -> str:
    """Patch the plain-prose boundary in docs/archive/CHANGELOG-HISTORIC.md.

    Matches lines like:
        Older changelog entries for MCP Memory Service (v10.24.0 and earlier).

    The `update_header_range()` family matches bolded CHANGELOG headers;
    this helper covers the prose variant in the archive file (#717).
    """
    # Boundary suffix is optional so the helper can also insert it when missing
    # (e.g. a recreated fallback header without the version range). The trailing
    # period acts as a required anchor — without it, matching the prefix without
    # the optional version group would sit *before* an existing boundary and
    # produce a duplicate "(vX and earlier) (vY and earlier)." chain.
    # [\w.-]+ in the version group accepts pre-release/metadata identifiers
    # like 1.0.0-beta, 1.0.0-rc.1, or 1.0.0+build.5.
    pattern = re.compile(
        r"(Older changelog entries for MCP Memory Service)"
        r"(?:\s*\(v[\w.+-]+ and earlier\))?(\.)"
    )
    return pattern.sub(
        rf"\1 (v{newest_archived} and earlier)\2",
        archive_header,
    )


def run(dry_run: bool = False) -> int:
    """Execute changelog housekeeping. Returns 0 on success, 1 on error."""
    # --- Read files ---
    if not CHANGELOG.exists():
        print(f"ERROR: {CHANGELOG} not found")
        return 1
    if not ARCHIVE.exists():
        print(f"ERROR: {ARCHIVE} not found")
        return 1

    changelog_text = CHANGELOG.read_text(encoding="utf-8")
    archive_text = ARCHIVE.read_text(encoding="utf-8")

    changelog_lines = len(changelog_text.splitlines())
    print(f"CHANGELOG.md: {changelog_lines} lines")

    # --- Check trigger ---
    if changelog_lines < MIN_LINES_TRIGGER:
        print(f"\nNo housekeeping needed (CHANGELOG < {MIN_LINES_TRIGGER} lines)")
        return 0

    # --- Parse CHANGELOG ---
    header, unreleased, version_blocks = parse_changelog_blocks(changelog_text)
    total_versions_before = len(version_blocks)
    print(f"\nVersions in CHANGELOG.md: {total_versions_before}")

    if total_versions_before <= KEEP_VERSIONS:
        print(f"Only {total_versions_before} versions — nothing to archive")
        return 0

    # --- Split: keep vs archive ---
    keep_blocks = version_blocks[:KEEP_VERSIONS]
    archive_blocks = version_blocks[KEEP_VERSIONS:]

    oldest_kept = keep_blocks[-1][0]
    print(f"Keeping versions down to v{oldest_kept}")
    print(f"Archiving {len(archive_blocks)} versions: ", end="")
    print(", ".join(f"v{v}" for v, _ in archive_blocks))

    # --- Deduplicate against archive ---
    existing_archive_versions = extract_archive_versions(archive_text)
    new_archive_blocks = [
        (v, text)
        for v, text in archive_blocks
        if v not in existing_archive_versions
    ]
    skipped = len(archive_blocks) - len(new_archive_blocks)
    if skipped:
        print(f"Skipping {skipped} versions already in archive")

    # --- Build new CHANGELOG.md ---
    newest_archived = archive_blocks[0][0]
    new_header = update_header_range(header, oldest_kept, newest_archived)
    parts = [new_header.rstrip(), "", unreleased.rstrip(), ""]
    for _v, block_text in keep_blocks:
        parts.append(block_text.rstrip())
        parts.append("")
    new_changelog = "\n".join(parts) + "\n"

    # --- Build new ARCHIVE ---
    if new_archive_blocks:
        # Insert after archive header (everything before first ## [version])
        archive_lines = archive_text.split("\n")
        archive_header: list[str] = []
        archive_body: list[str] = []
        found_first_version = False
        for line in archive_lines:
            if not found_first_version and VERSION_RE.match(line):
                found_first_version = True
            if found_first_version:
                archive_body.append(line)
            else:
                archive_header.append(line)

        # Update archive header reference
        archive_header_text = "\n".join(archive_header)
        # Update "For current versions (vX.Y.Z+)" line
        archive_header_text = re.sub(
            r"For current versions \(v[\d.]+\+\)",
            f"For current versions (v{oldest_kept}+)",
            archive_header_text,
        )
        # Update "Older changelog entries ... (vX.Y.Z and earlier)" prose boundary (#717)
        archive_header_text = update_archive_header_boundary(
            archive_header_text, newest_archived
        )

        new_blocks_text = "\n\n".join(
            block_text.rstrip() for _, block_text in new_archive_blocks
        )

        if archive_body:
            new_archive = (
                archive_header_text.rstrip()
                + "\n\n"
                + new_blocks_text
                + "\n\n"
                + "\n".join(archive_body).lstrip("\n")
            )
        else:
            new_archive = archive_header_text.rstrip() + "\n\n" + new_blocks_text + "\n"
    else:
        new_archive = archive_text

    # --- Validate ---
    _, _, new_version_blocks = parse_changelog_blocks(new_changelog)
    new_archive_versions = extract_archive_versions(new_archive)

    all_before = set(v for v, _ in version_blocks) | existing_archive_versions
    all_after = set(v for v, _ in new_version_blocks) | new_archive_versions

    missing = all_before - all_after
    if missing:
        print(f"\nERROR: Would lose versions: {missing}")
        return 1

    new_changelog_lines = len(new_changelog.splitlines())
    print(f"\nValidation:")
    print(f"  Versions before: {len(all_before)}, after: {len(all_after)}")
    print(f"  CHANGELOG.md: {changelog_lines} → {new_changelog_lines} lines")

    if dry_run:
        print("\n[DRY RUN] No files modified")
        return 0

    # --- Write files ---
    CHANGELOG.write_text(new_changelog, encoding="utf-8")
    ARCHIVE.write_text(new_archive, encoding="utf-8")

    print(f"\nDone. Archived {len(new_archive_blocks)} versions.")
    print(f"Oldest version in CHANGELOG.md: v{oldest_kept}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Changelog housekeeping: archive old CHANGELOG entries"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
