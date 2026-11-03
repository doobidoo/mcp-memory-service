"""Regression coverage for the filter tooltips and the Documents tab (#1817).

The search filter tooltips showed their English `title` text in every language
(the keys existed, only `data-i18n-title` was missing), and most of the Documents
tab -- upload history, the document action buttons, the drop zone rebuilt by
`app.js`, the chunk-overlap illustration -- was hardcoded English. These tests read
the real catalogs, the real `index.html` and the real `app.js`; nothing is mocked,
so they fail on the base branch and pass with the change.
"""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static"
I18N_DIR = STATIC_DIR / "i18n"
EN_CATALOG = I18N_DIR / "en.json"
INDEX_HTML = STATIC_DIR / "index.html"
APP_JS = STATIC_DIR / "app.js"

# Keys that already existed but were only wired to aria-label: the visible
# tooltip needs data-i18n-title next to the data-i18n-aria.
TITLE_KEYS = [
    "search.filters.tooltip",
    "search.filters.modeTooltip",
    "search.filters.tagsTip",
    "search.filters.dateTip",
    "search.filters.typeTip",
    "documents.processingTip",
    "documents.chunkSizeTip",
    "documents.chunkOverlapTip",
]

# New keys wired into static markup.
MARKUP_KEYS = {
    "documents.overlapHelp.chunk1": "data-i18n",
    "documents.overlapHelp.shared": "data-i18n",
    "documents.overlapHelp.chunk2": "data-i18n",
    "documents.processingHelp.title": "data-i18n",
    "modal.close": "data-i18n-aria",
}

# New keys looked up from app.js, with the English fallback each call must carry.
SCRIPT_KEYS = {
    "documents.modeDescriptionIndividual": "Each file will be processed individually with the same tags.",
    "documents.uploadHistoryError": "Failed to load upload history. Please check the console for details.",
    "documents.noUploads": "No uploads yet",
    "documents.view": "View",
    "documents.viewChunksTip": "View memory chunks",
    "documents.remove": "Remove",
    "documents.removeWithMemoriesTip": "Remove document and memories",
    "documents.removeTip": "Remove document",
    "documents.viewChunks": "View Chunks",
    "documents.viewAllChunksTip": "View all chunks",
    "documents.untitled": "Document",
    "documents.unknownFile": "Unknown file",
    "documents.noPreview": "No content preview available",
    "documents.noChunks": "No memory chunks found for this document.",
    "documents.batchUploadFailed": "Batch upload failed",
}

# Keys whose fallback spans quotes or newlines and is therefore checked separately.
INTERPOLATED_KEYS = {
    "documents.removeConfirm": "{filename}",
    "documents.uploadStatusError": "{status}",
}

NEW_KEYS = list(MARKUP_KEYS) + list(SCRIPT_KEYS) + ["documents.removeConfirm", "documents.uploadStatusError"]

# English literals that must no longer be built by hand in app.js.
RETIRED_LITERALS = [
    ">No uploads yet<",
    '<span>View</span>',
    '<span>Remove</span>',
    '<span>View Chunks</span>',
    'title="View memory chunks"',
    'title="View all chunks"',
    'title="Remove document"',
    'title="Remove document and memories"',
    "|| 'Unknown file'",
    ": 'No content preview available'",
    "|| 'Batch upload failed'",
    "<h3>Drag & drop files here</h3>",
]


def _catalogs():
    return sorted(I18N_DIR.glob("*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_english_catalog_defines_every_new_key():
    catalog = _load(EN_CATALOG)
    missing = [key for key in NEW_KEYS if key not in catalog]
    assert not missing, f"en.json does not define: {missing}"


def test_every_catalog_translates_every_new_key():
    problems = {}
    for path in _catalogs():
        catalog = _load(path)
        bad = [key for key in NEW_KEYS if not str(catalog.get(key, "")).strip()]
        if bad:
            problems[path.name] = bad
    assert not problems, f"missing or empty translations: {problems}"


def test_interpolated_keys_keep_their_token_in_every_catalog():
    """A translation that drops {filename} or {status} would hide the value."""
    problems = {}
    for path in _catalogs():
        catalog = _load(path)
        for key, token in INTERPOLATED_KEYS.items():
            if token not in str(catalog.get(key, "")):
                problems.setdefault(path.name, []).append(f"{key} lost {token}")
    assert not problems, f"interpolation tokens lost: {problems}"


def test_filter_tooltips_are_wired_to_their_existing_keys():
    """Each title attribute sits on the element that already had the aria key."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    catalog = _load(EN_CATALOG)

    missing, unpaired = [], []
    for key in TITLE_KEYS:
        if f'data-i18n-title="{key}"' not in html:
            missing.append(key)
            continue
        line = next(line for line in html.splitlines() if f'data-i18n-title="{key}"' in line)
        if f'title="{catalog[key]}"' not in line:
            unpaired.append(key)
        if f'data-i18n-aria="{key}"' not in line:
            unpaired.append(f"{key} (aria attribute replaced instead of kept)")
    assert not missing, f"index.html has no data-i18n-title for: {missing}"
    assert not unpaired, f"data-i18n-title does not sit on the English title it translates: {unpaired}"


def test_static_markup_references_its_new_keys():
    html = INDEX_HTML.read_text(encoding="utf-8")
    missing = [
        f'{attribute}="{key}"'
        for key, attribute in MARKUP_KEYS.items()
        if f'{attribute}="{key}"' not in html
    ]
    assert not missing, f"index.html does not reference: {missing}"
    assert html.count('data-i18n="documents.overlapHelp.shared"') == 2, (
        "both halves of the overlap illustration share one key"
    )
    assert html.count('data-i18n-aria="modal.close"') >= 3, (
        "every modal close button needs its aria-label translated"
    )


def test_document_strings_go_through_the_translation_helper():
    script = APP_JS.read_text(encoding="utf-8")
    catalog = _load(EN_CATALOG)

    for key, fallback in SCRIPT_KEYS.items():
        pattern = r"this\.t\(\s*'" + re.escape(key) + r"'\s*,\s*'([^']*)'"
        match = re.search(pattern, script)
        assert match, f"app.js does not look up {key} through this.t()"
        assert match.group(1) == fallback == catalog[key], (
            f"the English fallback for {key} drifted from en.json"
        )

    for key in INTERPOLATED_KEYS:
        assert f"this.t('{key}'" in script, f"app.js does not look up {key} through this.t()"


def test_retired_document_literals_are_gone_from_app_js():
    script = APP_JS.read_text(encoding="utf-8")
    leftovers = [literal for literal in RETIRED_LITERALS if literal in script]
    assert not leftovers, f"app.js still builds English by hand: {leftovers}"


def test_supported_formats_list_matches_the_accepted_extensions():
    """The drop zone accepts .docx/.pptx, so every catalog must name them.

    `semtools_loader.py` registers pdf/docx/doc/pptx/xlsx and the file input
    accepts .pdf,.docx,.pptx,.txt,.md,.json, but the catalogs still advertised
    `PDF, TXT, MD, JSON` only.
    """
    problems = {}
    for path in _catalogs():
        value = _load(path)["documents.supported"]
        for fmt in ("PDF", "DOCX", "PPTX", "TXT", "MD", "JSON"):
            if fmt not in value:
                problems.setdefault(path.name, []).append(fmt)
    assert not problems, f"documents.supported does not list: {problems}"


# Every element that carries a tooltip key must sit next to the translated text,
# never inside it: applyTranslations() writes textContent/innerHTML, which drops
# the element's children.
TOOLTIP_HOSTS = [
    "search.filters.tooltip",
    "search.filters.modeTooltip",
    "search.filters.tagsTip",
    "search.filters.dateTip",
    "search.filters.typeTip",
    "documents.processingTip",
    "documents.chunkSizeTip",
    "documents.chunkOverlapTip",
]

VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}


class _Tree(HTMLParser):
    """The real index.html as a tag tree, enough for the DOM stub below."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "#root", "attrs": {}, "children": [], "line": 0}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": dict(attrs), "children": [], "line": self.getpos()[0]}
        self.stack[-1]["children"].append(node)
        if tag not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1]["children"].append(
            {"tag": tag, "attrs": dict(attrs), "children": [], "line": self.getpos()[0]}
        )

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data.strip():
            self.stack[-1]["children"].append(data)


def _tree(html):
    parser = _Tree()
    parser.feed(html)
    return parser.root


def _i18n_attrs(node):
    return {k: v for k, v in node["attrs"].items() if k.startswith("data-i18n")}


def test_no_tooltip_lives_inside_a_translated_element():
    """A tooltip nested in a data-i18n parent is wiped on the first render."""
    root = _tree(INDEX_HTML.read_text(encoding="utf-8"))

    def walk(node, translated_ancestor):
        problems = []
        for child in node["children"]:
            if not isinstance(child, dict):
                continue
            if translated_ancestor and _i18n_attrs(child):
                problems.append(
                    f"line {child['line']} <{child['tag']} {sorted(_i18n_attrs(child))}> "
                    f"inside line {translated_ancestor['line']} <{translated_ancestor['tag']}>"
                )
            owner = child if {"data-i18n", "data-i18n-html"} & set(child["attrs"]) else translated_ancestor
            problems += walk(child, owner)
        return problems

    nested = walk(root, None)
    assert not nested, "these nodes are removed when their parent is translated: " + "; ".join(nested)


def test_the_file_input_accepts_what_app_js_rebuilds():
    """app.js re-renders the upload card; the static input must match it."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")
    static_accepts = re.findall(r'<input type="file" id="fileInput"[^>]*accept="([^"]+)"', html)
    rebuilt = set(re.findall(r'<input type="file" id="fileInput"[^>]*accept=\\?"([^"\\]+)', script))
    assert len(static_accepts) == 1, static_accepts
    assert rebuilt, "app.js no longer rebuilds the file input"
    assert set(static_accepts) == rebuilt, (
        f"index.html accepts {static_accepts[0]}, app.js rebuilds {sorted(rebuilt)}"
    )
    assert ".docx" in static_accepts[0] and ".pptx" in static_accepts[0]


# Each help icon is reached with querySelector, so every hook has to be unique:
# a bare `.info-icon` matched whichever icon came first in the document.
ICON_SELECTORS = ["info-icon-chunk", "info-icon-overlap", "info-icon-processing"]


def test_each_help_icon_has_its_own_unique_hook():
    """`document.querySelector('.info-icon')` picked up the processing-mode icon."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")

    counts = {sel: len(re.findall(rf'class="[^"]*\b{sel}\b[^"]*"', html)) for sel in ICON_SELECTORS}
    assert counts == {sel: 1 for sel in ICON_SELECTORS}, counts

    queried = set(re.findall(r"querySelector\('\.(info-icon[\w-]*)'\)", script))
    assert queried == set(ICON_SELECTORS), (
        f"app.js queries {sorted(queried)}; expected each icon by its own class"
    )


# The real applyTranslations() under a minimal DOM: textContent and innerHTML
# drop children, exactly as a browser does, and querySelectorAll is static.
HARNESS = r"""
const tree = __TREE__, de = __DE__, en = __EN__;

const camel = name => name.replace(/^data-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class El {
  constructor(node) {
    this.tagName = node.tag.toUpperCase();
    this.attrs = { ...node.attrs };
    this.dataset = {};
    for (const [k, v] of Object.entries(this.attrs)) {
      if (k.startsWith('data-')) this.dataset[camel(k)] = v;
    }
    this.children = node.children.map(child =>
      typeof child === 'string' ? child : new El(child));
    for (const child of this.children) if (child instanceof El) child.parent = this;
  }
  get textContent() {
    return this.children.map(c => (typeof c === 'string' ? c : c.textContent)).join('');
  }
  set textContent(value) { this.children = [String(value)]; }
  get innerHTML() { return this.textContent; }
  set innerHTML(value) { this.children = [String(value)]; }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  hasAttribute(name) { return name in this.attrs; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  *walk() { for (const c of this.children) if (c instanceof El) { yield c; yield* c.walk(); } }
  querySelectorAll(selector) {
    const wanted = selector.split(',').map(s => s.trim().replace(/^\[|\]$/g, ''));
    return [...this.walk()].filter(el => wanted.some(a => a in el.attrs));
  }
}

const root = new El(tree);
const before = [...root.walk()].filter(el => el.dataset.i18nTitle || el.dataset.i18nAria);

globalThis.document = {
  querySelectorAll: s => root.querySelectorAll(s),
  getElementById: id => (id === 'searchModeText'
    ? { textContent: 'Live Search', dataset: {} } : null),
  title: 'MCP Memory Service',
};

class App {
__METHODS__
  updateLanguageSwitcher() {}
}

const app = new App();
app.translations = de;
app.fallbackTranslations = en;
app.liveSearchEnabled = true;
await app.applyTranslations();

const live = new Set([...root.walk()]);
console.log(JSON.stringify(before.map(el => ({
  key: el.dataset.i18nTitle || el.dataset.i18nAria,
  tag: el.tagName,
  connected: live.has(el),
  title: el.getAttribute('title'),
  parentText: el.parent ? el.parent.textContent.trim() : null,
}))));
"""


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, name
    return match.group(0)


def _run_apply_translations():
    js = APP_JS.read_text(encoding="utf-8")
    methods = "".join(_method(js, name) for name in ("t", "replacePlaceholders", "applyTranslations"))
    script = (HARNESS.replace("__TREE__", json.dumps(_tree(INDEX_HTML.read_text(encoding="utf-8"))))
              .replace("__DE__", json.dumps(_load(I18N_DIR / "de.json")))
              .replace("__EN__", json.dumps(_load(EN_CATALOG)))
              .replace("__METHODS__", methods))
    run = subprocess.run(["node", "--input-type=module", "-"], input=script,
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tooltips_survive_the_first_render_and_speak_the_chosen_language():
    german = _load(I18N_DIR / "de.json")
    results = _run_apply_translations()
    present = {entry["key"] for entry in results}

    missing = [key for key in TOOLTIP_HOSTS if key not in present]
    assert not missing, f"index.html no longer carries: {missing}"

    dropped = [f"{e['key']} (<{e['tag']}>)" for e in results if not e["connected"]]
    assert not dropped, f"applyTranslations() removed these from the page: {dropped}"

    wrong = {
        e["key"]: e["title"]
        for e in results
        if e["key"] in TOOLTIP_HOSTS and e["title"] != german[e["key"]]
    }
    assert not wrong, f"tooltips not translated into German: {wrong}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_translated_labels_keep_their_text_and_their_icon():
    """The label reads German and still owns the tooltip icon next to it."""
    german = _load(I18N_DIR / "de.json")
    results = {e["key"]: e for e in _run_apply_translations()}
    for tip_key, text_key in [
        ("search.filters.tooltip", "search.filters.title"),
        ("search.filters.tagsTip", "search.filters.tags"),
        ("search.filters.dateTip", "search.filters.date"),
        ("search.filters.typeTip", "search.filters.type"),
        ("documents.processingTip", "documents.processingMode"),
    ]:
        parent_text = results[tip_key]["parentText"]
        assert german[text_key] in parent_text, (
            f"{text_key}: expected {german[text_key]!r} next to the icon, got {parent_text!r}"
        )
