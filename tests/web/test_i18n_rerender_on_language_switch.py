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
]


def _method(js, name):
    match = re.search(rf"^    async {name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, f"{name}() not found in app.js"
    return match.group(0).strip()


def _script(view, state):
    """A node script that switches de -> en and prints the recorded calls."""
    js = APP_JS.read_text(encoding="utf-8")
    state = {"memories": [], "searchResults": [], "tags": [], **(state or {})}
    stubs = ",\n".join(f"  async {name}(...a) {{ calls.push('{name}'); }}" for name in STUBS)
    return f"""
const calls = [];
const window = {{ scrollY: 420, scrollTo(x, y) {{ calls.push('scrollTo:' + y); }} }};
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


def _switch_language(view, state=None):
    """Switch de -> en with the given view active; return the recorded calls."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run([node, "-e", _script(view, state)], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_dashboard_rerenders_cached_memories_without_a_request():
    calls = _switch_language("dashboard", {"memories": [{"content_hash": "a"}]})
    assert "renderRecentMemories" in calls
    assert not [c for c in calls if c.startswith("load") and c != "loadTranslations"]


def test_search_and_browse_rerender_from_cache():
    assert "renderSearchResults" in _switch_language("search", {"searchResults": [{"memory": {}}]})
    assert "renderTagsCloud" in _switch_language("browse", {"tags": [{"tag": "x", "count": 1}]})


def test_empty_search_keeps_its_empty_state():
    assert "renderSearchResults" not in _switch_language("search")


def test_analytics_reloads_its_reports_but_not_the_graph():
    calls = _switch_language("analytics")
    assert "loadRecentActivityReport" in calls and "loadActivityHeatmapChart" in calls
    assert "loadGraphVisualization" not in calls


@pytest.mark.parametrize("view", ["documents", "manage", "qualityAnalytics"])
def test_other_views_rerun_their_own_loader(view):
    assert "loadViewData" in _switch_language(view)


def test_switch_never_reloads_memory_types_and_restores_scroll():
    # loadMemoryTypes() would reset the selected type filter.
    for view in ["dashboard", "search", "analytics", "manage"]:
        calls = _switch_language(view)
        assert "loadMemoryTypes" not in calls
        assert calls.index("applyTranslations") < calls.index("scrollTo:420") < calls.index("showToast")


def test_all_types_option_is_reached_by_apply_translations():
    js = APP_JS.read_text(encoding="utf-8")
    assert "allOpt.dataset.i18n = 'search.filters.type.all';" in js
