"""Run the real dashboard's client-side tag filter regression tests (#1972)."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tag_management_minimum_count_filter():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["node", "--test", "tests/web/test_tag_management_filter.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
