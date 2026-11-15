"""The header's status island: its rules, and how the dashboard is wired to it.

The rules live in plain JavaScript (``web/static/status-island.js``), so they are
asserted in a node test (``tests/web/test_status_island.cjs``, same ``node --test``
style as ``test_dashboard_live_refresh.cjs``) and this file is the pytest entry
point that CI runs. The static checks below need no node.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "src" / "mcp_memory_service" / "web" / "static"
NODE_TEST = REPO_ROOT / "tests" / "web" / "test_status_island.cjs"
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
ISLAND_JS = (STATIC / "status-island.js").read_text(encoding="utf-8")
STYLE_CSS = (STATIC / "style.css").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_status_island_rules_and_connection_wiring():
    result = subprocess.run(
        ["node", "--test", str(NODE_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "the status island regressed\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_island_script_loads_before_app_js():
    scripts = re.findall(r'<script\b[^>]*\bsrc="(static/[^"?]+)', INDEX_HTML)
    assert "static/status-island.js" in scripts
    assert scripts.index("static/status-island.js") < scripts.index("static/app.js")


def test_container_is_empty_and_hidden_until_a_source_reports():
    # No text in the markup: the island builds itself, so nothing is left untranslated
    # and nothing claims "Connected" before the stream is open.
    assert re.search(r'<div id="statusIsland" class="status-island" hidden></div>', INDEX_HTML)
    assert 'id="connectionStatus"' not in INDEX_HTML


def test_old_fixed_indicator_is_gone_everywhere():
    assert "connection-status" not in STYLE_CSS
    assert "connectionStatus" not in APP_JS


def test_view_writes_text_only_through_textcontent():
    # Source texts can carry server values; the view must never build markup from them.
    assert "innerHTML" not in ISLAND_JS
    assert "insertAdjacentHTML" not in ISLAND_JS


def test_island_adds_no_catalog_keys_of_its_own():
    # The island's own catalog keys; the parity test wants each in all eleven catalogs.
    keys = set(re.findall(r"\bt\('([a-zA-Z.]+)'", ISLAND_JS))
    assert keys == {"island.dismiss", "island.details"}


def test_reduced_motion_stops_the_island_animations():
    block = STYLE_CSS[STYLE_CSS.index("@media (prefers-reduced-motion: reduce)"):]
    assert ".status-island-dot" in block[:400]
    assert "animation: none" in block[:400]
