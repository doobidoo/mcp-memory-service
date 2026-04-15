"""Unit tests for scripts/maintenance/changelog_housekeeping.py boundary helpers (#717)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "changelog_housekeeping.py"


@pytest.fixture(scope="module")
def housekeeping():
    spec = importlib.util.spec_from_file_location("changelog_housekeeping", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_update_archive_header_boundary_rewrites_prose_version(housekeeping):
    original = "Older changelog entries for MCP Memory Service (v10.24.0 and earlier).\n"
    updated = housekeeping.update_archive_header_boundary(original, "10.36.3")
    assert "v10.36.3 and earlier" in updated
    assert "v10.24.0" not in updated


def test_update_archive_header_boundary_leaves_unrelated_text(housekeeping):
    original = (
        "# Historic Changelog Archive\n\n"
        "Older changelog entries for MCP Memory Service (v10.24.0 and earlier).\n\n"
        "For current versions (v10.25.0+), see [CHANGELOG.md](../../CHANGELOG.md).\n"
    )
    updated = housekeeping.update_archive_header_boundary(original, "10.36.3")
    assert "# Historic Changelog Archive" in updated
    assert "For current versions (v10.25.0+)" in updated


def test_update_readme_footer_boundary_rewrites_link_text(housekeeping):
    original = (
        "**Full version history**: [CHANGELOG.md](CHANGELOG.md) | "
        "[Older versions (v10.22.0 and earlier)](docs/archive/CHANGELOG-HISTORIC.md) | "
        "[All Releases](https://github.com/doobidoo/mcp-memory-service/releases)\n"
    )
    updated = housekeeping.update_readme_footer_boundary(original, "10.36.3")
    assert "[Older versions (v10.36.3 and earlier)](docs/archive/CHANGELOG-HISTORIC.md)" in updated
    assert "v10.22.0" not in updated


def test_update_readme_footer_boundary_idempotent(housekeeping):
    current = "[Older versions (v10.36.3 and earlier)](docs/archive/CHANGELOG-HISTORIC.md)"
    assert housekeeping.update_readme_footer_boundary(current, "10.36.3") == current


def test_update_readme_footer_boundary_no_link_match_is_noop(housekeeping):
    """Prose that mentions `Older versions` without a markdown link must not match."""
    prose = "Older versions (v9.0.0 and earlier) are deprecated.\n"
    assert housekeeping.update_readme_footer_boundary(prose, "10.36.3") == prose


def test_update_archive_header_boundary_no_match_is_noop(housekeeping):
    prose = "Random text mentioning (v10.0.0 and earlier) without the phrase.\n"
    assert housekeeping.update_archive_header_boundary(prose, "10.36.3") == prose


def test_update_archive_header_boundary_inserts_when_missing(housekeeping):
    """When the boundary suffix is absent (e.g. recreated fallback), insert it."""
    bare = "Older changelog entries for MCP Memory Service.\n"
    updated = housekeeping.update_archive_header_boundary(bare, "10.36.3")
    assert "(v10.36.3 and earlier)" in updated


def test_update_readme_footer_boundary_inserts_when_missing(housekeeping):
    """Fallback footer from trim_readme_previous_releases omits the boundary;
    helper must still inject the correct one (#717 Gemini feedback)."""
    fallback = (
        "**Full version history**: [CHANGELOG.md](CHANGELOG.md) "
        "| [Older versions](docs/archive/CHANGELOG-HISTORIC.md) "
        "| [All Releases](https://github.com/doobidoo/mcp-memory-service/releases)\n"
    )
    updated = housekeeping.update_readme_footer_boundary(fallback, "10.36.3")
    assert "[Older versions (v10.36.3 and earlier)](docs/archive/CHANGELOG-HISTORIC.md)" in updated


def test_update_archive_header_boundary_handles_prerelease(housekeeping):
    """Pre-release versions like 1.0.0-beta must be matched and replaced cleanly."""
    original = "Older changelog entries for MCP Memory Service (v1.0.0-beta and earlier).\n"
    updated = housekeeping.update_archive_header_boundary(original, "10.36.3")
    assert "(v10.36.3 and earlier)" in updated
    assert "v1.0.0-beta" not in updated
    # No double-version
    assert updated.count("and earlier") == 1


def test_update_readme_footer_boundary_handles_prerelease(housekeeping):
    original = (
        "[Older versions (v1.0.0-rc.1 and earlier)](docs/archive/CHANGELOG-HISTORIC.md)"
    )
    updated = housekeeping.update_readme_footer_boundary(original, "10.36.3")
    assert "[Older versions (v10.36.3 and earlier)](docs/archive/CHANGELOG-HISTORIC.md)" in updated
    assert "v1.0.0-rc.1" not in updated


def test_update_archive_header_boundary_no_double_version_when_unknown_format(housekeeping):
    """If the existing boundary has an unparseable version, the period-anchored
    regex still requires a period — so no match means no-op (not duplicate)."""
    # Hypothetical corrupted boundary without period at end → no match, no-op
    weird = "Older changelog entries for MCP Memory Service (v???) somewhere in the middle\n"
    assert housekeeping.update_archive_header_boundary(weird, "10.36.3") == weird


def test_update_header_range_still_works_after_refactor(housekeeping):
    """Regression: the two new helpers must not shadow the bolded-header pattern."""
    header = (
        "**Recent releases for MCP Memory Service (v10.25.0 and later)**\n\n"
        "**Versions v10.24.0 and earlier** – See [HISTORIC](./docs/archive/CHANGELOG-HISTORIC.md).\n"
    )
    updated = housekeeping.update_header_range(header, "10.36.4", "10.36.3")
    assert "**Recent releases for MCP Memory Service (v10.36.4 and later)**" in updated
    assert "**Versions v10.36.3 and earlier**" in updated
