"""Manage-tab labels and search counts use catalog keys (#1808).

The static checks read the real catalogs, index.html and app.js. They do not
mock the dictionaries. A key the markup names but no catalog defines fails
here, which is how analytics.filterByType and manage.bulk.deleteUntagged.title
used to slip through.
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

# Keys this change adds. The first two were already referenced and undefined.
NEW_STRINGS = {
    "analytics.filterByType": "Filter by type:",
    "manage.bulk.deleteUntagged.title": "Delete Untagged",
    "manage.bulk.deleteUntagged.desc": "Remove memories without any tags ({count} found)",
    "manage.bulk.deleteUntagged.confirm": "Delete {count} memories without tags?",
    "manage.tags.loadFailed": "Failed to load tag statistics",
    "toast.noUntaggedMemories": "No untagged memories to delete",
    "toast.deleteUntaggedFailed": "Delete untagged operation failed",
    "toast.operationFailed": "Operation failed",
    "search.results.count.one": "{count} result",
    "modal.memoryViewer.chunksFound.one": "{count} chunk found",
}

# Locales whose existing plural string is a different word from the singular.
PLURAL_MARKING = ("de.json", "en.json", "es.json", "fr.json", "pt.json")

# Existing keys the listed sites start using.
REUSED_KEYS = (
    "manage.tags.column.tag",
    "manage.tags.column.count",
    "manage.tags.column.actions",
    "manage.tags.rename",
    "manage.tags.delete",
    "search.results.count",
    "modal.memoryViewer.chunksFound",
    "search.filters.type.all",
)

# English that used to be written straight into the page.
OLD_LITERALS = [
    "<th>Tag</th>",
    "<th>Count</th>",
    "<th>Actions</th>",
    ">Rename</button>",
    ">Delete</button>",
    "Failed to load tag statistics</p>",
    "showToast('No untagged memories to delete'",
    "showToast('Delete untagged operation failed'",
    "Delete ${count} memories without tags?",
    "${count} result${count !== 1 ? 's' : ''}",
    "chunk${response.total_found !== 1 ? 's' : ''} found",
    "Remove memories without any tags (<span",
    "this.t('search.filters.type.all') || 'All types'",
]


def _catalog(name):
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def _tokens(text):
    return sorted(re.findall(r"\{\w+\}", text))


def _referenced_keys():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    html_keys = re.findall(
        r'data-i18n(?:-html|-placeholder|-aria|-title)?="([^"]+)"',
        html,
    )
    js_keys = re.findall(r"this\.t\(\s*'([A-Za-z0-9_.]+)'", js)
    return html_keys, js_keys


CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))
NON_ENGLISH = [name for name in CATALOGS if name != "en.json"]


def test_every_referenced_catalog_key_exists_in_english():
    """data-i18n in index.html and literal this.t() keys in app.js are in en.json."""
    english = _catalog("en.json")
    html_keys, js_keys = _referenced_keys()
    missing = sorted({key for key in html_keys + js_keys if key not in english})
    assert missing == []


def test_english_catalog_has_the_new_strings():
    english = _catalog("en.json")
    assert {key: english.get(key) for key in NEW_STRINGS} == NEW_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_has_the_new_strings(name):
    catalog = _catalog(name)
    missing = [key for key in NEW_STRINGS if not str(catalog.get(key, "")).strip()]
    assert missing == []
    wrong_tokens = [
        key for key, english in NEW_STRINGS.items() if _tokens(catalog[key]) != _tokens(english)
    ]
    assert wrong_tokens == []
    assert not [key for key in NEW_STRINGS if "<" in catalog[key] or ">" in catalog[key]]


@pytest.mark.parametrize("name", NON_ENGLISH)
def test_non_english_catalogs_translate_the_new_strings(name):
    catalog = _catalog(name)
    untranslated = [key for key, english in NEW_STRINGS.items() if catalog[key] == english]
    assert untranslated == []


def test_new_keys_sit_next_to_their_prefix():
    for name in CATALOGS:
        text = (I18N_DIR / name).read_text(encoding="utf-8")
        keys = [match.group(1) for match in re.finditer(r'^\s*"([^"]+)"\s*:', text, re.M)]
        for key in NEW_STRINGS:
            prefix = key.split(".", 1)[0]
            index = keys.index(key)
            neighbours = keys[max(index - 1, 0):index] + keys[index + 1:index + 2]
            assert any(item.startswith(prefix + ".") for item in neighbours), (name, key)


def test_listed_markup_references_its_keys():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert re.search(
        r'id="resultsCount"[^>]*data-i18n="search\.results\.count"[^>]*data-count="0"',
        html,
    )
    assert re.search(
        r'id="docSearchCount"[^>]*data-i18n="search\.results\.count"[^>]*data-count="0"',
        html,
    )
    assert re.search(
        r'id="memoryViewerStats"[^>]*data-i18n="modal\.memoryViewer\.chunksFound"[^>]*data-count="0"',
        html,
    )
    assert re.search(
        r'id="untaggedCount"[^>]*data-i18n="manage\.bulk\.deleteUntagged\.desc"[^>]*data-count="0"',
        html,
    )
    assert 'data-i18n="analytics.filterByType"' in html
    assert 'data-i18n="manage.bulk.deleteUntagged.title"' in html
    # The count is the element's data-count, not a child span applyTranslations would erase.
    assert 'id="untaggedCount"' in html
    assert "<span id=\"untaggedCount\">" not in html


def test_app_js_references_the_listed_keys_with_fallbacks():
    js = APP_JS.read_text(encoding="utf-8")
    expected = {
        "manage.tags.column.tag": "Tag",
        "manage.tags.column.count": "Count",
        "manage.tags.column.actions": "Actions",
        "manage.tags.rename": "Rename",
        "manage.tags.delete": "Delete",
        "manage.tags.loadFailed": "Failed to load tag statistics",
        "search.filters.type.all": "All types",
        "manage.bulk.deleteUntagged.desc": "Remove memories without any tags ({count} found)",
        "manage.bulk.deleteUntagged.confirm": "Delete {count} memories without tags?",
        "toast.noUntaggedMemories": "No untagged memories to delete",
        "toast.deleteUntaggedFailed": "Delete untagged operation failed",
        "toast.operationFailed": "Operation failed",
    }
    for key, fallback in expected.items():
        # this.t() or setCountedText(); the fallback stays next to the key.
        assert re.search(
            rf"'{re.escape(key)}',\s*'{re.escape(fallback)}'",
            js,
        ), key
    for key in (
        "manage.tags.column.tag",
        "manage.tags.column.count",
        "manage.tags.column.actions",
        "manage.tags.rename",
        "manage.tags.delete",
        "manage.tags.loadFailed",
    ):
        assert f'data-i18n="{key}"' in js
    assert "allOpt.dataset.i18n = 'search.filters.type.all';" in js


def test_reused_keys_already_exist_in_english():
    english = _catalog("en.json")
    missing = [key for key in REUSED_KEYS if not str(english.get(key, "")).strip()]
    assert missing == []


def test_app_js_has_no_english_left_at_the_listed_sites():
    js = APP_JS.read_text(encoding="utf-8")
    left = [literal for literal in OLD_LITERALS if literal in js]
    assert left == []


def test_plural_marking_locales_use_a_different_singular():
    pairs = (
        ("search.results.count.one", "search.results.count"),
        ("modal.memoryViewer.chunksFound.one", "modal.memoryViewer.chunksFound"),
    )
    for name in PLURAL_MARKING:
        catalog = _catalog(name)
        for singular, plural in pairs:
            assert catalog[singular] != catalog[plural], (name, singular)


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, f"{name}() not found in app.js"
    return match.group(0).strip()


def test_one_result_and_one_chunk_stay_singular_after_a_language_switch():
    """updateResultsCount and viewDocumentMemory pick .one only for count 1.

    applyTranslations() then repeats that stored key, so a switch does not
    turn "1 result" back into the plural catalog string.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = APP_JS.read_text(encoding="utf-8")
    english = json.dumps(_catalog("en.json"), ensure_ascii=False)
    german = json.dumps(_catalog("de.json"), ensure_ascii=False)
    methods = ",\n".join(_method(js, name) for name in (
        "replacePlaceholders", "t", "setCountedText", "applyTranslations",
        "updateResultsCount", "viewDocumentMemory",
    ))
    script = f"""
const en = {english};
const de = {german};
const attrs = {{}};
function makeEl() {{
  const own = {{}};
  return {{
    style: {{}},
    innerHTML: '',
    textContent: '',
    dataset: new Proxy(own, {{
      get: (t, k) => t['data-' + String(k).replace(/[A-Z]/g, m => '-' + m.toLowerCase())],
      set: (t, k, v) => {{ t['data-' + String(k).replace(/[A-Z]/g, m => '-' + m.toLowerCase())] = String(v); return true; }},
      deleteProperty: (t, k) => {{ delete t['data-' + String(k).replace(/[A-Z]/g, m => '-' + m.toLowerCase())]; return true; }}
    }}),
    hasAttribute: name => Object.prototype.hasOwnProperty.call(own, name),
    getAttribute: name => own[name] ?? null
  }};
}}
const nodes = {{
  resultsCount: makeEl(),
  memoryViewerModal: makeEl(),
  memoryViewerFilename: makeEl(),
  memoryViewerStats: makeEl(),
  memoryChunksList: makeEl()
}};
const document = {{
  title: '',
  querySelectorAll: () => Object.values(nodes),
  querySelector: () => null,
  getElementById: id => nodes[id] || null
}};
const app = {{
  translations: en,
  fallbackTranslations: en,
  escapeHtml: s => s,
  setLoading() {{}},
  showToast() {{}},
  updateLanguageSwitcher() {{}},
  async apiCall() {{ return {{ status: 'success', filename: 'notes.md', total_found: app._found, memories: [] }}; }},
  {methods}
}};
const snap = el => ({{ key: el.dataset.i18n, count: el.dataset.count, text: el.textContent }});
(async () => {{
  app.updateResultsCount(1);
  const one = snap(nodes.resultsCount);
  app.translations = de;
  await app.applyTranslations();
  const oneDe = snap(nodes.resultsCount);
  app.updateResultsCount(2);
  const manyDe = snap(nodes.resultsCount);
  app.updateResultsCount(0);
  const zeroDe = snap(nodes.resultsCount);
  app.translations = en;
  await app.applyTranslations();
  const zeroEn = snap(nodes.resultsCount);
  app._found = 1;
  await app.viewDocumentMemory('demo');
  const chunkOne = snap(nodes.memoryViewerStats);
  app.translations = de;
  await app.applyTranslations();
  const chunkOneDe = snap(nodes.memoryViewerStats);
  app._found = 2;
  await app.viewDocumentMemory('demo');
  const chunkManyDe = snap(nodes.memoryViewerStats);
  app._found = 0;
  await app.viewDocumentMemory('demo');
  const chunkZeroDe = snap(nodes.memoryViewerStats);
  console.log(JSON.stringify({{ one, oneDe, manyDe, zeroDe, zeroEn, chunkOne, chunkOneDe, chunkManyDe, chunkZeroDe }}));
}})();
"""
    # node - reads the program from stdin: the inlined catalogues make `-e` exceed the
    # 32k Windows CreateProcessW command-line limit (WinError 206).
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True,
                            encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["one"] == {"key": "search.results.count.one", "count": "1", "text": "1 result"}
    assert out["oneDe"] == {"key": "search.results.count.one", "count": "1", "text": "1 Ergebnis"}
    assert out["manyDe"] == {"key": "search.results.count", "count": "2", "text": "2 Ergebnisse"}
    assert out["zeroDe"] == {"key": "search.results.count", "count": "0", "text": "0 Ergebnisse"}
    assert out["zeroEn"] == {"key": "search.results.count", "count": "0", "text": "0 results"}
    assert out["chunkOne"] == {"key": "modal.memoryViewer.chunksFound.one", "count": "1", "text": "1 chunk found"}
    assert out["chunkOneDe"] == {"key": "modal.memoryViewer.chunksFound.one", "count": "1", "text": "1 Chunk gefunden"}
    assert out["chunkManyDe"] == {"key": "modal.memoryViewer.chunksFound", "count": "2", "text": "2 Chunks gefunden"}
    assert out["chunkZeroDe"] == {"key": "modal.memoryViewer.chunksFound", "count": "0", "text": "0 Chunks gefunden"}
