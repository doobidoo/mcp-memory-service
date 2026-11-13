"""The Manage tab shows one section at a time behind accessible sub-tabs (#1973).

The static tests read the real index.html and catalogs. The behavior tests
extract the real methods from app.js and run them in node against a small DOM
stub built from the real markup of the Manage view (the same approach as
test_i18n_settings_tabs.py).
"""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static"
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
I18N_DIR = STATIC / "i18n"
CATALOGS = sorted(p.name for p in I18N_DIR.glob("*.json"))

NEW_STRINGS = {
    "manage.tabs.label": "Manage sections",
    "manage.tabs.bulk": "Bulk Operations",
    "manage.tabs.tags": "Tags",
    "manage.tabs.ontology": "Ontology",
}


class _ManageView(HTMLParser):
    """Collects the tablist, its tabs and the sections inside #manageView."""

    def __init__(self):
        super().__init__()
        self.depth = 0  # >0 while inside #manageView
        self.tablists, self.tabs, self.sections = [], [], []
        self._tab = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") == "manageView":
            self.depth = 1
            return
        if not self.depth:
            return
        if tag not in ("input", "br", "img", "hr"):
            self.depth += 1
        if attrs.get("role") == "tablist":
            self.tablists.append(attrs)
        if attrs.get("role") == "tab":
            self._tab = dict(attrs, text="")
            self.tabs.append(self._tab)
        if tag == "section":
            self.sections.append(attrs)

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1
            if tag == "button":
                self._tab = None

    def handle_data(self, data):
        if self._tab is not None:
            self._tab["text"] += data.strip()


def _view():
    parser = _ManageView()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parser


def _catalog(name):
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    return match.group(0).strip() if match else None


# --- markup -------------------------------------------------------------------


def test_manage_view_has_one_labelled_tablist():
    view = _view()
    assert len(view.tablists) == 1
    assert view.tablists[0].get("data-i18n-aria") == "manage.tabs.label"
    assert view.tablists[0].get("aria-label") == NEW_STRINGS["manage.tabs.label"]


def test_every_section_is_a_tabpanel_owned_by_one_tab():
    view = _view()
    assert len(view.sections) == len(view.tabs) == 3
    for tab, section in zip(view.tabs, view.sections):
        assert tab["aria-controls"] == section["id"]
        assert section.get("role") == "tabpanel"
        assert section.get("aria-labelledby") == tab["id"]


def test_only_the_first_section_is_shown_at_load():
    view = _view()
    assert [t.get("aria-selected") for t in view.tabs] == ["true", "false", "false"]
    assert [t.get("tabindex") for t in view.tabs] == [None, "-1", "-1"]
    assert ["hidden" in s for s in view.sections] == [False, True, True]


def test_tabs_are_buttons_wired_to_the_catalog():
    view = _view()
    keys = [t.get("data-i18n") for t in view.tabs]
    assert keys == ["manage.tabs.bulk", "manage.tabs.tags", "manage.tabs.ontology"]
    assert [t["text"] for t in view.tabs] == [NEW_STRINGS[k] for k in keys]
    assert all(t.get("type") == "button" for t in view.tabs)


def test_settings_tab_handlers_do_not_pick_up_the_manage_tabs():
    # initializeSettingsTabs() binds every .settings-tab; the sub-tabs must not be among them.
    assert all("settings-tab" not in t.get("class", "").split() for t in _view().tabs)


# --- catalogs -----------------------------------------------------------------


def test_english_catalog_has_every_new_string():
    english = _catalog("en.json")
    assert {k: english.get(k) for k in NEW_STRINGS} == NEW_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_labels_the_sub_tabs(name):
    catalog = _catalog(name)
    assert [k for k in NEW_STRINGS if not str(catalog.get(k, "")).strip()] == []


# --- behavior in node ---------------------------------------------------------

HARNESS = r"""
class El {
  constructor(attrs) {
    this.attrs = Object.assign({}, attrs);
    this.id = this.attrs.id || '';
    this.hidden = 'hidden' in this.attrs;
    this.tabIndex = this.attrs.tabindex === undefined ? 0 : Number(this.attrs.tabindex);
    this.listeners = {};
    this.textContent = this.attrs.text || '';
    this.dataset = {};
    for (const [k, v] of Object.entries(this.attrs)) {
      if (k.startsWith('data-')) this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v;
    }
    const cls = new Set((this.attrs.class || '').split(/\s+/).filter(Boolean));
    this.classList = {
      add: c => cls.add(c), remove: c => cls.delete(c), contains: c => cls.has(c),
      toggle: (c, on) => (on ? cls.add(c) : cls.delete(c)),
    };
  }
  getAttribute(n) { return n in this.attrs ? String(this.attrs[n]) : null; }
  hasAttribute(n) { return n in this.attrs; }
  setAttribute(n, v) { this.attrs[n] = String(v); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, extra) {
    const e = Object.assign({ target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } }, extra);
    (this.listeners[type] || []).forEach(fn => fn(e));
    return e;
  }
  focus() { focused = this.id; }
}
let focused = null;
const markup = __MARKUP__;
const tablist = new El(markup.tablist);
const tabs = markup.tabs.map(a => new El(a));
const panels = markup.sections.map(a => new El(a));
const views = [new El({ id: 'dashboardView', class: 'view-container' }), new El({ id: 'manageView', class: 'view-container' })];
const byId = Object.fromEntries([...tabs, ...panels, ...views].map(e => [e.id, e]));
global.document = {
  title: 'MCP Memory Service - Dashboard',
  getElementById: id => byId[id] || null,
  querySelector: () => null,
  querySelectorAll: sel => {
    if (sel === '#manageView [role="tab"]') return tabs;
    if (sel === '.view-container') return views;
    if (sel.startsWith('[data-i18n]')) return [tablist, ...tabs];
    return [];
  },
};
// Only the network and the browser storage are stubbed: the catalog fetch
// serves the real de.json, and the Manage loaders' API calls are recorded.
const catalogs = { de: __DE__ };
global.fetch = async url => {
  const lang = url.match(/i18n\/(\w+)\.json$/)[1];
  return { ok: lang in catalogs, json: async () => catalogs[lang] };
};
global.localStorage = { setItem() {} };
const app = {
  loaded: [],
  apiLoads: 0,
  currentView: 'dashboard',
  currentLang: 'en',
  supportedLanguages: ['en', 'de'],
  translations: __EN__,
  fallbackTranslations: __EN__,
  async loadTagSelectOptions() { this.apiLoads++; },
  async loadTagManagementStats() { this.apiLoads++; },
  async loadUntaggedCount() { this.apiLoads++; },
  async loadOntology() { this.apiLoads++; },
  async loadDashboardData() {},
  updateActiveFilters() {},
  updateLanguageSwitcher() {},
  showToast() {},
  __METHODS__
};
// Record the views opened, then hand over to the real loadViewData().
const realLoadViewData = app.loadViewData;
app.loadViewData = async function (name) { this.loaded.push(name); return realLoadViewData.call(this, name); };
const state = () => ({
  selected: tabs.filter(t => t.getAttribute('aria-selected') === 'true').map(t => t.id),
  active: tabs.filter(t => t.classList.contains('active')).map(t => t.id),
  tabIndex: tabs.map(t => t.tabIndex),
  shown: panels.filter(p => !p.hidden).map(p => p.id),
  focused,
});
const out = {};
app.initializeManageTabs();
byId.manageTabTags.dispatch('click');
out.clickTags = state();
const right = byId.manageTabTags.dispatch('keydown', { key: 'ArrowRight' });
out.arrowRight = Object.assign(state(), { prevented: right.defaultPrevented });
byId.manageTabOntology.dispatch('keydown', { key: 'ArrowRight' });
out.wrapRight = state();
byId.manageTabBulk.dispatch('keydown', { key: 'ArrowLeft' });
out.wrapLeft = state();
byId.manageTabOntology.dispatch('keydown', { key: 'Home' });
out.home = state();
byId.manageTabBulk.dispatch('keydown', { key: 'End' });
out.end = state();
const other = byId.manageTabOntology.dispatch('keydown', { key: 'a' });
out.otherKey = Object.assign(state(), { prevented: other.defaultPrevented });
app.switchView('dashboard');
app.switchView('manage');
out.reopen = Object.assign(state(), { loaded: [...app.loaded] });
(async () => {
  app.switchManageTab('manageTabTags');
  // Let the load that switchView('manage') started settle first, so only the
  // language switch's own reload is counted below.
  await new Promise(r => setTimeout(r, 0));
  const loadsBefore = app.apiLoads;
  await app.setLanguage('de');  // the real switch: catalog, applyTranslations, rerenderActiveView
  out.languageSwitch = Object.assign(state(), {
    lang: app.currentLang,
    reloaded: app.apiLoads > loadsBefore,
    labels: tabs.map(t => t.textContent),
    listLabel: tablist.getAttribute('aria-label'),
  });
  process.stdout.write(JSON.stringify(out));
})().catch(e => { process.stdout.write(JSON.stringify({ error: String(e.stack || e) })); });
"""


@pytest.fixture(scope="module")
def results():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = APP_JS.read_text(encoding="utf-8")
    names = [
        "initializeManageTabs", "switchManageTab", "switchView", "loadViewData", "loadManageData",
        "setLanguage", "loadTranslations", "applyTranslations", "rerenderActiveView", "t", "replacePlaceholders",
    ]
    methods = [_method(js, n) for n in names]
    assert all(methods), dict(zip(names, methods))
    view = _view()
    markup = {"tablist": view.tablists[0], "tabs": view.tabs, "sections": view.sections}
    script = (
        HARNESS.replace("__MARKUP__", json.dumps(markup))
        .replace("__DE__", json.dumps(_catalog("de.json")))
        .replace("__EN__", json.dumps(_catalog("en.json")))
        .replace("__METHODS__", ",\n".join(methods))
    )
    # Through stdin: with both catalogs inlined the script is too long for argv.
    result = subprocess.run([node], input=script, capture_output=True, text=True, timeout=60, check=True)
    out = json.loads(result.stdout)
    assert "error" not in out, out.get("error")
    return out


def _only(tab, panel, focused=None):
    return {
        "selected": [tab],
        "active": [tab],
        "tabIndex": [0 if t == tab else -1 for t in ("manageTabBulk", "manageTabTags", "manageTabOntology")],
        "shown": [panel],
        "focused": focused,
    }


def test_clicking_a_sub_tab_shows_only_its_section(results):
    assert results["clickTags"] == _only("manageTabTags", "managePanelTags")


def test_arrow_keys_move_and_wrap(results):
    assert results["arrowRight"] == dict(_only("manageTabOntology", "managePanelOntology", "manageTabOntology"), prevented=True)
    assert results["wrapRight"] == _only("manageTabBulk", "managePanelBulk", "manageTabBulk")
    assert results["wrapLeft"] == _only("manageTabOntology", "managePanelOntology", "manageTabOntology")


def test_home_and_end_jump_to_the_ends(results):
    assert results["home"] == _only("manageTabBulk", "managePanelBulk", "manageTabBulk")
    assert results["end"] == _only("manageTabOntology", "managePanelOntology", "manageTabOntology")


def test_other_keys_are_left_alone(results):
    assert results["otherKey"] == dict(_only("manageTabOntology", "managePanelOntology", "manageTabOntology"), prevented=False)


def test_opening_the_manage_tab_starts_on_the_first_sub_tab(results):
    reopen = results["reopen"]
    assert reopen.pop("loaded") == ["dashboard", "manage"]
    assert reopen == _only("manageTabBulk", "managePanelBulk", "manageTabOntology")


def test_a_language_switch_keeps_the_chosen_sub_tab_and_relabels_it(results):
    switch = results["languageSwitch"]
    de = _catalog("de.json")
    # setLanguage() really ran: German loaded and the Manage view reloaded its data.
    assert (switch.pop("lang"), switch.pop("reloaded")) == ("de", True)
    assert switch.pop("labels") == [de["manage.tabs.bulk"], de["manage.tabs.tags"], de["manage.tabs.ontology"]]
    assert switch.pop("listLabel") == de["manage.tabs.label"]
    assert switch == _only("manageTabTags", "managePanelTags", "manageTabOntology")


def test_german_labels_differ_from_english():
    # Otherwise the relabel check above could pass on untranslated English.
    de = _catalog("de.json")
    assert [k for k in ("manage.tabs.label", "manage.tabs.bulk", "manage.tabs.ontology") if de[k] == NEW_STRINGS[k]] == []
