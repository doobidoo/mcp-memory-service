"""Regression test for GHSA-j664-v823-6p9f: dashboard escapeHtml() must encode quotes.

Callers interpolate escapeHtml() output into quoted attributes such as
data-tag="...". A stored tag containing a double quote used to close the
attribute and add an onmouseover handler to the rendered button.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "app.js"


def _escape_html_body() -> str:
    match = re.search(r"^    escapeHtml\(text\) \{\n(.*?)^    \}\n", APP_JS.read_text(), re.S | re.M)
    assert match, "escapeHtml(text) not found in app.js"
    return match.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_escape_html_encodes_attribute_delimiters():
    inputs = ['z" onmouseover="window.x=1" q="', "a'b", "<b>&</b>", None, 5]
    script = (
        f"const escapeHtml = function(text) {{\n{_escape_html_body()}}};\n"
        f"console.log(JSON.stringify({json.dumps(inputs)}.map(escapeHtml)));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [
        "z&quot; onmouseover=&quot;window.x=1&quot; q=&quot;",
        "a&#39;b",
        "&lt;b&gt;&amp;&lt;/b&gt;",
        "",
        "5",
    ]
