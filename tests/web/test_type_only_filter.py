"""Run the real dashboard's type-only search filter regression tests (#1971)."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_type_only_filter_pages_through_server_side_type_filter():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["node", "--test", "tests/web/test_type_only_filter.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
