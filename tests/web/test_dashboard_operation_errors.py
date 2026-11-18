"""Run long-operation persistent error regressions (#2008)."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_operation_errors():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["node", "--test", "tests/web/test_dashboard_operation_errors.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
