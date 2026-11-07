"""Time-of-day cues accompany, rather than replace, the local clock time."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).parents[2] / "src/mcp_memory_service/web/static/app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("timezone", ["Europe/Zurich", "America/New_York"])
def test_time_icons_cover_local_day_and_preserve_clock(timezone):
    source = APP_JS.read_text(encoding="utf-8")
    script = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script += """
const app = Object.create(MemoryDashboard.prototype);
const results = [];
for (let hour = 0; hour < 24; hour++) {
    for (const minute of [0, 59]) {
        const date = new Date(2026, 10, 7, hour, minute);
        results.push({
            hour,
            clock: date.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'}),
            html: app.renderTimeIcon(date.getTime() / 1000),
        });
    }
}
console.log(JSON.stringify({results, missing: app.renderTimeIcon(null)}));
"""
    result = subprocess.run(
        ["node", "-"], input=script, capture_output=True, text=True,
        encoding="utf-8", check=True, env={**os.environ, "TZ": timezone},
    )
    output = json.loads(result.stdout)
    icons = ["🌙", "🌅", "☕", "💻", "🍽️", "⛅", "🍷", "🛏️"]
    for row in output["results"]:
        assert icons[row["hour"] // 3] in row["html"]
        assert row["clock"] in row["html"]
    assert output["missing"] == ""
