"""The tag usage chart must not report a percentage for overlapping tags.

Regression test for #1736: the minor-tag bucket summed the per-tag percentages, and
a memory carrying three tags is counted in three of them, so the bucket passed 100%
and drew a bar past the 200px scale it is measured against.

The chart is plain JavaScript inside ``app.js``, so the assertions live in a node
test (``tests/web/test_tag_usage_chart.cjs``, written in the same ``node --test``
style as ``test_dashboard_live_refresh.cjs``) and this file is the pytest entry
point that CI actually runs.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_TEST = REPO_ROOT / "tests" / "web" / "test_tag_usage_chart.cjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tag_usage_chart_never_reports_more_than_100_percent():
    result = subprocess.run(
        ["node", "--test", str(NODE_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "the tag usage chart regressed\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
