"""The Knowledge Graph toolbar and its fullscreen view are translated (#1811).

The static tests read the real catalogs, index.html and app.js. The behavior test
extracts the real rotation-toggle methods from app.js and runs them in node against
the real German catalog, with a stub only for document.getElementById.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static"
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
I18N_DIR = STATIC / "i18n"

# Every key #1811 wires, as en.json has it.
GRAPH_STRINGS = {
    "analytics.graphNodesAll": "All",
    "analytics.graphDimensionTip": "Toggle 3D / 2D rendering",
    "analytics.graphLayoutLabel": "Graph layout",
    "analytics.rotateTip": "Pause / resume auto-rotation (3D only)",
    "analytics.fullscreenRotateTip": "Pause / resume rotation",
    "analytics.rotatePause": "Pause",
    "analytics.rotatePlay": "Play",
    "analytics.fullscreenTip": "Fullscreen Mode",
    "analytics.exitFullscreenTip": "Exit Fullscreen (Esc)",
    "analytics.legendTip": "Show / hide legend",
    "analytics.legend": "Legend",
    "analytics.typeFilterTip": "Click a type to show/hide its nodes",
    "analytics.graphNodeHint": "Click node to view full memory",
    "analytics.graphRingHint": "Click to center; double-click for details",
    "analytics.noConnectedMemories": "No connected memories found. Try lowering the minimum connections filter.",
    "analytics.graphLoadFailed": "Failed to load graph visualization",
}

# index.html: (attribute, key, how many sites), main toolbar and fullscreen header together.
HTML_SITES = [
    ("data-i18n", "analytics.graphNodesAll", 1),
    ("data-i18n-title", "analytics.graphDimensionTip", 1),
    ("data-i18n-aria", "analytics.graphLayoutLabel", 2),
    ("data-i18n-title", "analytics.rotateTip", 1),
    ("data-i18n", "analytics.rotatePause", 2),
    ("data-i18n-title", "analytics.fullscreenTip", 1),
    ("data-i18n-title", "analytics.typeFilterTip", 2),
    ("data-i18n-title", "analytics.fullscreenRotateTip", 1),
    ("data-i18n-title", "analytics.legendTip", 1),
    ("data-i18n-title", "analytics.exitFullscreenTip", 1),
]
# app.js: key -> how many this.t() calls use it.
JS_SITES = {
    "analytics.noConnectedMemories": 2,
    "analytics.graphLoadFailed": 1,
    "analytics.graphNodeHint": 1,
    "analytics.graphRingHint": 1,
}


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))


def test_english_catalog_has_every_graph_string():
    english = _catalog("en.json")
    assert {k: english.get(k) for k in GRAPH_STRINGS} == GRAPH_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_graph_key(name):
    catalog = _catalog(name)
    missing = [k for k in GRAPH_STRINGS if not str(catalog.get(k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"


def test_toolbar_and_fullscreen_markup_reference_their_keys():
    html = INDEX_HTML.read_text(encoding="utf-8")
    counts = {(attr, key): len(re.findall(rf'{attr}="{re.escape(key)}"', html)) for attr, key, _ in HTML_SITES}
    assert counts == {(attr, key): n for attr, key, n in HTML_SITES}
    # The fullscreen Legend button carries its text key as well as its title key
    assert re.search(r'id="graphFullscreenLegendBtn"[^>]*data-i18n="analytics\.legend"', html)


def test_toolbar_and_fullscreen_markup_have_no_english_left():
    html = INDEX_HTML.read_text(encoding="utf-8")
    untranslated = [
        m.group(0)
        for m in re.finditer(r'<[^>]*(?:title|aria-label)="(?:Toggle 3D|Graph layout|Pause /|Fullscreen Mode|'
                             r'Click a type|Show / hide legend|Exit Fullscreen)[^>]*>', html)
        if "data-i18n-title" not in m.group(0) and "data-i18n-aria" not in m.group(0)
    ]
    assert untranslated == []
    assert re.findall(r'<option value="10000">All</option>', html) == []


def test_app_js_uses_the_keys_with_english_fallbacks():
    js = APP_JS.read_text(encoding="utf-8")
    calls = re.findall(r"this\.t\('(analytics\.\w+)', '((?:[^'\\]|\\.)*)'\)", js)
    used = {}
    for key, fallback in calls:
        if key in JS_SITES:
            assert fallback == GRAPH_STRINGS[key], key
            used[key] = used.get(key, 0) + 1
    assert used == JS_SITES
    for literal in ("'<p>No connected memories found", "'<p class=\"error\">Failed to load graph visualization",
                    "interactionHint = 'Click node", "formatGraphTooltip(item, 'Click to center",
                    "? 'Play' : 'Pause'"):
        assert literal not in js, literal


def _method(js, name):
    match = re.search(rf"^    {name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, name
    return match.group(0)


HARNESS = r"""
const de = __DE__, en = __EN__;
const els = {};
for (const id of ['graphRotateLabel', 'graphFullscreenRotateLabel', 'graphRotateBtn', 'graphFullscreenRotateBtn']) {
  els[id] = { id, textContent: 'Pause', dataset: { i18n: 'analytics.rotatePause' }, disabled: false };
}
globalThis.document = { getElementById: id => els[id] || null };
class App {
__METHODS__
}
const app = new App();
app.translations = de; app.fallbackTranslations = en;
app.use3DGraph = true; app.graphLayout = 'force';
const out = [];
for (const paused of [true, false]) {
  app.graphRotationPaused = paused;
  app._updateRotateButton();
  out.push(['graphRotateLabel', 'graphFullscreenRotateLabel'].map(id => [els[id].textContent, els[id].dataset.i18n]));
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_rotate_toggle_shows_the_selected_language_and_keeps_its_key():
    js = APP_JS.read_text(encoding="utf-8")
    methods = "".join(_method(js, name) for name in ("t", "replacePlaceholders", "setTranslatedText", "_updateRotateButton"))
    de, en = _catalog("de.json"), _catalog("en.json")
    script = (HARNESS.replace("__DE__", json.dumps(de)).replace("__EN__", json.dumps(en))
              .replace("__METHODS__", methods))
    run = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, encoding="utf-8", check=True)
    paused, playing = json.loads(run.stdout)
    play = [de["analytics.rotatePlay"], "analytics.rotatePlay"]
    pause = [de["analytics.rotatePause"], "analytics.rotatePause"]
    # The key travels with the text, so applyTranslations() after a language switch keeps Play/Pause right
    assert paused == [play, play]
    assert playing == [pause, pause]
