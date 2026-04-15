#!/usr/bin/env python3
"""Changelog housekeeping: archive old entries, trim README previous releases.

Usage:
    python scripts/maintenance/changelog_housekeeping.py [--dry-run]

Trigger criteria (any one suffices):
    - CHANGELOG.md > 150 lines
    - README.md "Previous Releases" section > 8 entries

What it does:
    1. Keeps last 6-8 versions in CHANGELOG.md (current minor -2)
    2. Archives older entries to docs/archive/CHANGELOG-HISTORIC.md
    3. Trims README.md "Previous Releases" to max 7 entries
    4. Validates no content is lost (version count before == after)
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
README = REPO_ROOT / "README.md"

# Config
KEEP_VERSIONS = 8  # max versions to keep in CHANGELOG.md
MIN_LINES_TRIGGER = 150  # skip housekeeping if CHANGELOG.md below this
MAX_README_ENTRIES = 7  # max entries in "Previous Releases"
MAX_README_ENTRIES_TRIGGER = 8  # trigger threshold

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


def update_readme_footer_boundary(text: str, newest_archived: str) -> str:
    """Patch the \"Older versions\" link text in README.md.

    Matches the markdown-link text in:
        **Full version history**: [CHANGELOG.md](...) | [Older versions (v10.22.0 and earlier)](docs/archive/CHANGELOG-HISTORIC.md) | [All Releases](...)

    The boundary is the oldest version NOT kept in CHANGELOG.md (i.e. the
    newest version that just got archived), so readers following the link
    land at a file whose first entry matches what the footer promises (#717).
    """
    # Boundary suffix is optional so the helper can also insert it into the
    # fallback footer that trim_readme_previous_releases recreates when the
    # "Previous Releases" section ends up bare. The trailing `\]\([^)]+\)`
    # anchor (the markdown link URL part) prevents the prefix from matching
    # *before* an existing boundary and producing a duplicate.
    # [\w.-]+ in the version group accepts pre-release/metadata identifiers
    # like 1.0.0-beta, 1.0.0-rc.1, or 1.0.0+build.5.
    pattern = re.compile(
        r"(\[Older versions)(?:\s*\(v[\w.+-]+ and earlier\))?(\]\([^)]+\))"
    )
    return pattern.sub(
        rf"\1 (v{newest_archived} and earlier)\2",
        text,
    )


def _is_previous_releases_start(line: str) -> bool:
    """Detect 'Previous Releases' section start in various formats.

    Matches:
        ## Previous Releases       (markdown header)
        **Previous Releases**:     (bold text, as used in current README)
    """
    if re.match(r"^#{1,4}\s.*[Pp]revious\s+[Rr]eleases", line):
        return True
    if re.match(r"^\*\*[Pp]revious\s+[Rr]eleases\*\*", line):
        return True
    return False


def _is_section_end(line: str) -> bool:
    """Detect the end of the 'Previous Releases' section.

    Ends at:
        - Any markdown header (## ..., ### ...)
        - Horizontal rule (---) separating sections
        - 'Full version history' footer link (marks end of release list)
    """
    if re.match(r"^#{1,3}\s", line):
        return True
    if re.match(r"^---\s*$", line):
        return True
    if re.match(r"^\*\*Full version history\*\*", line):
        return True
    return False


def trim_readme_previous_releases(text: str, max_entries: int) -> tuple[str, int]:
    """Trim the 'Previous Releases' section in README.md.

    Only trims release entry lines (- **vX.Y.Z** ...) beyond max_entries.
    All other content (headers, footers, subsequent sections) is preserved.

    Returns (new_text, entries_removed).
    """
    lines = text.split("\n")
    result: list[str] = []
    in_section = False
    entry_count = 0
    entries_removed = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect section start
        if not in_section and _is_previous_releases_start(line):
            in_section = True
            result.append(line)
            i += 1
            continue

        if in_section:
            # Detect section end — stop trimming, preserve everything after
            if _is_section_end(line) and not _is_previous_releases_start(line):
                in_section = False
                result.append(line)
                i += 1
                continue

            # Count release entries (lines starting with "- **v")
            if re.match(r"^\s*-\s+\*\*v\d+", line):
                entry_count += 1
                if entry_count > max_entries:
                    entries_removed += 1
                    i += 1
                    continue

            result.append(line)
        else:
            result.append(line)

        i += 1

    # If "Previous Releases" was the last section, ensure footer link is present
    footer_link = (
        "**Full version history**: [CHANGELOG.md](CHANGELOG.md) "
        "| [Older versions](docs/archive/CHANGELOG-HISTORIC.md) "
        "| [All Releases](https://github.com/doobidoo/mcp-memory-service/releases)"
    )
    if in_section and entries_removed > 0:
        if not any(footer_link[:30] in r for r in result[-5:]):
            result.append("")
            result.append(footer_link)
            result.append("")

    return "\n".join(result), entries_removed


def count_readme_entries(text: str) -> int:
    """Count entries in the 'Previous Releases' section."""
    in_section = False
    count = 0
    for line in text.split("\n"):
        if not in_section and _is_previous_releases_start(line):
            in_section = True
            continue
        if in_section:
            if _is_section_end(line):
                break
            if re.match(r"^\s*-\s+\*\*v\d+", line):
                count += 1
    return count


def run(dry_run: bool = False) -> int:
    """Execute changelog housekeeping. Returns 0 on success, 1 on error."""
    # --- Read files ---
    if not CHANGELOG.exists():
        print(f"ERROR: {CHANGELOG} not found")
        return 1
    if not ARCHIVE.exists():
        print(f"ERROR: {ARCHIVE} not found")
        return 1
    if not README.exists():
        print(f"ERROR: {README} not found")
        return 1

    changelog_text = CHANGELOG.read_text(encoding="utf-8")
    archive_text = ARCHIVE.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")

    changelog_lines = len(changelog_text.splitlines())
    readme_entries = count_readme_entries(readme_text)

    print(f"CHANGELOG.md: {changelog_lines} lines")
    print(f"README Previous Releases: {readme_entries} entries")

    # --- Check triggers ---
    if changelog_lines < MIN_LINES_TRIGGER and readme_entries <= MAX_README_ENTRIES_TRIGGER:
        print(
            f"\nNo housekeeping needed "
            f"(CHANGELOG < {MIN_LINES_TRIGGER} lines, "
            f"README entries <= {MAX_README_ENTRIES_TRIGGER})"
        )
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

    # --- Trim README ---
    new_readme, readme_removed = trim_readme_previous_releases(
        readme_text, MAX_README_ENTRIES
    )
    # Also patch the "Older versions (vX.Y.Z and earlier)" link text so the
    # boundary tracks the newest-archived version (#717).
    new_readme = update_readme_footer_boundary(new_readme, newest_archived)

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
    if readme_removed:
        print(f"  README: removed {readme_removed} old entries")

    if dry_run:
        print("\n[DRY RUN] No files modified")
        return 0

    # --- Write files ---
    CHANGELOG.write_text(new_changelog, encoding="utf-8")
    ARCHIVE.write_text(new_archive, encoding="utf-8")
    if new_readme != readme_text:
        README.write_text(new_readme, encoding="utf-8")

    print(f"\nDone. Archived {len(new_archive_blocks)} versions.")
    print(f"Oldest version in CHANGELOG.md: v{oldest_kept}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Changelog housekeeping: archive old entries, trim README"
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
