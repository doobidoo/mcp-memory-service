"""Verify scripts use package imports and commands independently of the cwd."""

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
IMPORT_PATTERN = re.compile(r"^\s*(?:from\s+src\.|import\s+src\.)", re.MULTILINE)


def test_no_src_prefix_imports_in_scripts():
    """No script should import through a shadow src package."""
    assert SCRIPTS.is_dir(), f"scripts/ not found next to tests/: {SCRIPTS}"
    hits = [
        f"{path.relative_to(SCRIPTS.parent)}: {match.group().strip()}"
        for path in sorted(SCRIPTS.rglob("*.py"))
        for match in IMPORT_PATTERN.finditer(path.read_text(errors="replace"))
    ]
    assert not hits, "scripts/ still import through src.:\n" + "\n".join(hits)


def test_no_src_package_references_in_script_commands():
    """Generated configuration and shell commands use the installed package."""
    assert SCRIPTS.is_dir(), f"scripts/ not found next to tests/: {SCRIPTS}"
    hits = [
        str(path.relative_to(SCRIPTS.parent))
        for path in sorted(SCRIPTS.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
        and "src.mcp_memory_service" in path.read_text(errors="replace")
    ]
    assert not hits, "scripts/ still reference the shadow package:\n" + "\n".join(hits)
