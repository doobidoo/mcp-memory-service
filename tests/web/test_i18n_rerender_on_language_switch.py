"""A language switch re-renders the JavaScript-built text of the active view (#1807).

applyTranslations() only rewrites elements with a data-i18n attribute. The
dashboard has no JS test runner, so these tests extract the real
setLanguage() and rerenderActiveView() from app.js and run them in node with
call-recording stubs for every renderer and loader.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "app.js"

STUBS = [
    "loadTranslations", "applyTranslations", "showToast",
    "renderRecentMemories", "renderSearchResults", "renderTagsCloud", "loadViewData",
    "loadAnalyticsOverview", "loadMemoryGrowthChart", "loadTagUsageChart", "loadMemoryTypesChart",
    "loadActivityHeatmapChart", "loadTopTagsReport", "loadRecentActivityReport", "loadStorageReport",
    "loadRelationshipTypesChart", "loadGraphVisualization", "loadMemoryTypes",
    "updateActiveFilters",
]


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, f"{name}() not found in app.js"
    return match.group(0).strip()


def _script(view, state):
    """A node script that switches de -> en and prints the recorded calls."""
    js = APP_JS.read_text(encoding="utf-8")
    state = {"memories": [], "searchResults": [], "tags": [], **(state or {})}
    stubs = ",\n".join(f"  async {name}(...a) {{ calls.push('{name}'); }}" for name in STUBS)
    return f"""
const calls = [];
const app = {{
  supportedLanguages: ['de', 'en'],
  currentLang: 'de',
  currentView: {json.dumps(view)},
  ...{json.dumps(state)},
  t(key, fallback) {{ return fallback; }},
{stubs},
  {_method(js, "setLanguage")},
  {_method(js, "rerenderActiveView")}
}};
app.setLanguage('en').then(() => console.log(JSON.stringify(calls)));
"""


def _run_node(script):
    """Run a node script and return the JSON it prints."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def _switch_language(view, state=None):
    """Switch de -> en with the given view active; return the recorded calls."""
    return _run_node(_script(view, state))


def test_dashboard_rerenders_cached_memories_without_a_request():
    calls = _switch_language("dashboard", {"memories": [{"content_hash": "a"}]})
    assert "renderRecentMemories" in calls
    assert not [c for c in calls if c.startswith("load") and c != "loadTranslations"]


@pytest.mark.parametrize("view", ["dashboard", "search", "browse", "analytics", "manage"])
def test_every_switch_rebuilds_the_active_filter_chips(view):
    # The chips keep their text while another view is open (#1815).
    assert "updateActiveFilters" in _switch_language(view)


def test_search_rerenders_from_cache():
    assert "renderSearchResults" in _switch_language("search", {"searchResults": [{"memory": {}}]})


def test_browse_is_left_alone_so_a_selected_tag_stays_open():
    # renderTagsCloud() hides the memories of the selected tag.
    calls = _switch_language("browse", {"tags": [{"tag": "x", "count": 1}]})
    assert "renderTagsCloud" not in calls and "loadViewData" not in calls


def test_empty_search_keeps_its_empty_state():
    assert "renderSearchResults" not in _switch_language("search")


def test_analytics_reloads_its_reports_but_not_the_graph():
    calls = _switch_language("analytics")
    assert "loadRecentActivityReport" in calls and "loadActivityHeatmapChart" in calls
    assert "loadGraphVisualization" not in calls


@pytest.mark.parametrize("view", ["documents", "manage", "qualityAnalytics"])
def test_other_views_rerun_their_own_loader(view):
    assert "loadViewData" in _switch_language(view)


def test_switch_never_reloads_memory_types():
    # loadMemoryTypes() would reset the selected type filter.
    for view in ["dashboard", "search", "analytics", "manage"]:
        calls = _switch_language(view)
        assert "loadMemoryTypes" not in calls
        assert calls.index("applyTranslations") < calls.index("showToast")


# Just enough DOM for the manage-view methods. removeChild() clears the value
# the way a real <select> does when its selected option goes away.
FAKE_DOM = """
const out = {};
function el(tag) {
  return { tag, children: [], value: '', dataset: {}, listeners: 0, innerHTML: '',
    querySelector(key) { this.parts ??= {}; return this.parts[key] ??= el(key); },
    appendChild(c) { this.children.push(c); },
    removeChild() { this.children.pop(); this.value = ''; },
    get lastChild() { return this.children[this.children.length - 1]; },
    addEventListener() { this.listeners++; } };
}
const nodes = { deleteTagSelect: el('select'), tagManagementContainer: el('div') };
nodes.deleteTagSelect.children.push(el('option'));
const document = { getElementById: id => nodes[id], createElement: el };
const data = { tags: [{ tag: 'alpha', count: 2 }, { tag: 'beta', count: 1 }] };
"""


def _run_manage_method(name, body):
    """Run a manage-view method from app.js against FAKE_DOM."""
    method = _method(APP_JS.read_text(encoding="utf-8"), name)
    app = f"const app = {{ async apiCall() {{ return data; }}, escapeHtml: s => s, t: (key, fallback) => fallback || key, {method} }};"
    return _run_node(f"{FAKE_DOM}\n{app}\n(async () => {{ {body} console.log(JSON.stringify(out)); }})();")


def test_reloading_tag_options_keeps_the_selected_tag():
    out = _run_manage_method("loadTagSelectOptions", """
      await app.loadTagSelectOptions();
      nodes.deleteTagSelect.value = 'beta';
      await app.loadTagSelectOptions();
      out.value = nodes.deleteTagSelect.value;
      out.options = nodes.deleteTagSelect.children.length;
    """)
    assert out == {"value": "beta", "options": 3}


def test_rerendering_the_tag_table_keeps_one_click_handler():
    out = _run_manage_method("renderTagManagementTable", """
      app.renderTagManagementTable(data);
      app.renderTagManagementTable(data);
      const c = nodes.tagManagementContainer;
      out.handlers = c.listeners + (typeof c.onclick === 'function' ? 1 : 0);
    """)
    assert out == {"handlers": 1}


def test_all_types_option_is_reached_by_apply_translations():
    js = APP_JS.read_text(encoding="utf-8")
    assert "allOpt.dataset.i18n = 'search.filters.type.all';" in js
