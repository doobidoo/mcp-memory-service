"""Run the real ``theme-init.js`` in node against a stubbed ``localStorage``, ``matchMedia`` and ``body`` (#1705, #1737).

``app.js`` is not loaded here, so a ``dark-mode`` class on ``<body>`` can only come
from the initializer, which is what keeps the first paint from flashing light.
"""
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
const [source, stored, blocked, osDark] = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const classes = new Set();
const localStorage = {
    getItem(key) {
        if (blocked) throw new Error('SecurityError');
        return key === 'memoryDashboardSettings' ? stored : null;
    },
};
const document = { body: { classList: { add: (c) => classes.add(c) } } };
const window = osDark === null ? {} : {
    matchMedia: (query) => ({ matches: query === '(prefers-color-scheme: dark)' && osDark }),
};
vm.runInNewContext(source, { localStorage, document, window });
process.stdout.write(JSON.stringify([...classes]));
"""


def _body_classes(stored, blocked=False, os_dark=False):
    payload = json.dumps([THEME_INIT.read_text(encoding="utf-8"), stored, blocked, os_dark])
    run = subprocess.run(
        ["node", "-e", HARNESS], input=payload, capture_output=True,
        text=True, encoding="utf-8", check=True,
    )
    return json.loads(run.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    ("stored", "blocked", "os_dark", "expected"),
    [
        (json.dumps({"theme": "dark"}), False, False, ["dark-mode"]),
        (json.dumps({"theme": "dark"}), False, None, ["dark-mode"]),
        (json.dumps({"theme": "light"}), False, False, []),
        (json.dumps({"theme": "light"}), False, True, []),
    ],
    ids=["dark-on-light-os", "dark-without-matchmedia", "light-on-light-os", "light-on-dark-os"],
)
def test_theme_init_applies_an_explicit_choice_whatever_the_os(stored, blocked, os_dark, expected):
    assert _body_classes(stored, blocked, os_dark) == expected


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    ("stored", "blocked"),
    [
        (None, False),
        (json.dumps({"theme": "system"}), False),
        (json.dumps({"viewDensity": "compact"}), False),
        ("{not json", False),
        (None, True),
    ],
    ids=["nothing-saved", "system", "settings-without-theme", "malformed-json", "storage-blocked"],
)
def test_theme_init_follows_the_os_without_an_explicit_choice(stored, blocked):
    assert _body_classes(stored, blocked, os_dark=True) == ["dark-mode"]
    assert _body_classes(stored, blocked, os_dark=False) == []
    assert _body_classes(stored, blocked, os_dark=None) == []
