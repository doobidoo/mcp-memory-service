"""Run the real ``theme-init.js`` in node against a stubbed ``localStorage`` and ``body`` (#1705)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

THEME_INIT = (
    Path(__file__).resolve().parents[2]
    / "src" / "mcp_memory_service" / "web" / "static" / "theme-init.js"
)

HARNESS = r"""
const vm = require('vm');
const [source, stored, blocked] = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const classes = new Set();
const localStorage = {
    getItem(key) {
        if (blocked) throw new Error('SecurityError');
        return key === 'memoryDashboardSettings' ? stored : null;
    },
};
const document = { body: { classList: { add: (c) => classes.add(c) } } };
vm.runInNewContext(source, { localStorage, document });
process.stdout.write(JSON.stringify([...classes]));
"""


def _body_classes(stored, blocked=False):
    payload = json.dumps([THEME_INIT.read_text(encoding="utf-8"), stored, blocked])
    run = subprocess.run(
        ["node", "-e", HARNESS], input=payload, capture_output=True,
        text=True, encoding="utf-8", check=True,
    )
    return json.loads(run.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    ("stored", "blocked", "expected"),
    [
        (json.dumps({"theme": "dark"}), False, ["dark-mode"]),
        (json.dumps({"theme": "light"}), False, []),
        (None, False, []),
        ("{not json", False, []),
        (None, True, []),
    ],
    ids=["dark", "light", "nothing-saved", "malformed-json", "storage-blocked"],
)
def test_theme_init_applies_only_a_saved_dark_theme(stored, blocked, expected):
    assert _body_classes(stored, blocked) == expected
