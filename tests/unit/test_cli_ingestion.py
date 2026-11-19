"""Exercise mcp_memory_service.cli.ingestion through the public Click command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp_memory_service.cli.main import cli


@pytest.mark.parametrize(
    ("options", "expected_files"),
    [
        (["--extensions", "pdf,md"], ["manual.pdf", "notes.md"]),
        (["-e", ".pdf", "-e", "md"], ["manual.pdf", "notes.md"]),
        (["-e", "pdf, .md,", "-e", ".pdf"], ["manual.pdf", "notes.md"]),
        ([], ["manual.pdf", "notes.md", "other.txt"]),
        (["-e", ""], []),
        (["-e", ", ,"], []),
    ],
)
def test_ingest_directory_extensions(
    tmp_path: Path, options: list[str], expected_files: list[str]
) -> None:
    files = [tmp_path / name for name in ("manual.pdf", "notes.md", "other.txt")]
    for path in files:
        path.touch()

    result = CliRunner().invoke(
        cli, ["ingest-directory", str(tmp_path), "--dry-run", *options]
    )

    assert result.exit_code == (0 if expected_files else 1), result.output
    for path in files:
        assert result.output.count(str(path)) == expected_files.count(path.name)
