"""The dashboard follows the system color scheme until the user picks a theme (#1737).

Before this, the default theme was a hardcoded ``'light'`` and nothing read
``prefers-color-scheme``, so the dashboard always opened in light mode.

The theme logic is plain JavaScript inside ``app.js``, so the behaviour is
asserted in a node test (``tests/web/test_dashboard_system_theme.cjs``, same
``node --test`` style as ``test_tag_usage_chart.cjs``) and this file is the
pytest entry point CI runs. The settings select is checked on the HTML the app
actually serves.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_TEST = REPO_ROOT / "tests" / "web" / "test_dashboard_system_theme.cjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_theme_follows_system_color_scheme():
    result = subprocess.run(
        ["node", "--test", str(NODE_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "the dashboard does not follow the system color scheme\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_theme_select_offers_system():
    client = TestClient(app_module.create_app())
    response = client.get("/")

    select = re.search(
        r'<select id="themeSelect">(.*?)</select>', response.text, re.DOTALL
    )
    assert select, "settings modal should have a theme select"
    values = re.findall(r'<option value="([^"]+)"', select.group(1))
    assert values == ["light", "dark", "system"]
