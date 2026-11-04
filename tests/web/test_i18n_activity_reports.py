"""The activity report, the activity heatmap and the top-tags report are translated (#1809).

The static tests read the real catalogs and app.js. The behavior tests extract the real
render methods from app.js and run them in node against the real German and English
catalogs; only the container element is a stub.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static"
APP_JS = STATIC / "app.js"
I18N_DIR = STATIC / "i18n"

# Every key #1809 wires, as en.json has it.
REPORT_STRINGS = {
    "analytics.topTagsLoadFailed": "Failed to load top tags",
    "analytics.topTagsEmpty": "No tags found",
    "analytics.reportPeriod": "Period: {period}",
    "analytics.memoriesPercent": "{count} memories ({percent}%)",
    "analytics.oftenWith": "Often with:",
    "analytics.recentActivityLoadFailed": "Failed to load recent activity",
    "analytics.activeDays": "Active Days:",
    "analytics.currentStreak": "Current Streak:",
    "analytics.longestStreak": "Longest Streak:",
    "analytics.streakDays": "{count} days",
    "analytics.peakTimes": "Peak Times:",
    "analytics.heatmapLoadFailed": "Failed to load activity heatmap",
    "analytics.heatmapEmpty": "No activity data available",
    "analytics.heatmapActiveDays": "{count} active days",
    "analytics.heatmapMaxPerDay": "Max: {count} memories/day",
    "analytics.heatmapDayTip": "{date}: {count} memories",
    "analytics.heatmapLess": "Less",
    "analytics.heatmapMore": "More",
}
CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def _tokens(text):
    return sorted(re.findall(r"\{\w+\}", text))


def test_english_catalog_has_every_report_string():
    english = _catalog("en.json")
    assert {k: english.get(k) for k in REPORT_STRINGS} == REPORT_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_report_key_with_the_same_tokens(name):
    catalog = _catalog(name)
    missing = [k for k in REPORT_STRINGS if not str(catalog.get(k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"
    wrong_tokens = [k for k, v in REPORT_STRINGS.items() if _tokens(catalog[k]) != _tokens(v)]
    assert not wrong_tokens, f"{name} changes the placeholders of {wrong_tokens}"
    if name != "en.json":
        untranslated = [k for k, v in REPORT_STRINGS.items() if catalog[k] == v]
        assert not untranslated, f"{name} keeps English for {untranslated}"


def test_new_keys_sit_with_the_analytics_keys():
    # Placed next to their prefix group, not appended, so parallel #1806 slices merge cleanly.
    for name in CATALOGS:
        lines = (I18N_DIR / name).read_text(encoding="utf-8").splitlines()
        anchor = next(i for i, line in enumerate(lines) if '"analytics.loadingHeatmap"' in line)
        block = [re.match(r'\s*"([^"]+)"', line).group(1) for line in lines[anchor + 1:anchor + 1 + len(REPORT_STRINGS)]]
        assert block == list(REPORT_STRINGS), name


def test_app_js_uses_the_keys_with_english_fallbacks():
    js = APP_JS.read_text(encoding="utf-8")
    used = {}
    for key, fallback in re.findall(r"(?:this\.t\(|staticTextHtml\('\w+', '[\w ]*', )'(analytics\.\w+)', '((?:[^'\\]|\\.)*)'", js):
        if key in REPORT_STRINGS:
            assert fallback == REPORT_STRINGS[key], key
            used[key] = used.get(key, 0) + 1
    assert set(used) == set(REPORT_STRINGS)
    for literal in ("Failed to load top tags</p>", "'<p>No tags found</p>';\n    return;", "Period: ${",
                    "} memories (${", "'<div class=\"tag-cooccurrence\">Often with: '",
                    "Failed to load recent activity</p>", "<strong>Active Days:</strong>",
                    "<strong>Current Streak:</strong>", "<strong>Longest Streak:</strong>", "<strong>Peak Times:</strong>",
                    "Failed to load activity heatmap</p>", "No activity data available</p>", "} active days</span>",
                    "<span>Max: ${", "} memories`;", "'<span>Less</span>'", "'<span>More</span>'"):
        assert literal not in js, literal


def _method(js, name):
    """The source of one class method, found by matching braces from its header."""
    match = re.search(rf"^    {name}\([^)]*\) \{{\n", js, re.MULTILINE)
    assert match, name
    depth, i = 0, match.end() - 2
    while True:
        depth += {"{": 1, "}": -1}.get(js[i], 0)
        i += 1
        if depth == 0:
            return js[match.start():i] + "\n"


HARNESS = r"""
const catalogs = { de: __DE__, en: __EN__ };
class App {
__METHODS__
}
const app = new App();
app.fallbackTranslations = catalogs.en;
const out = {};
for (const lang of ['de', 'en']) {
  app.translations = catalogs[lang]; app.currentLang = lang;
  const box = () => ({ innerHTML: '' });
  const tags = box(), noTags = box(), activity = box(), heat = box(), noHeat = box();
  app.renderTopTagsReport(tags, { period: '30d', tags: [
    { tag: 'python', count: 12, percentage: 57.1, trending: false, growth_rate: null,
      co_occurring_tags: [{ tag: 'testing', strength: 0.8 }] }] });
  app.renderTopTagsReport(noTags, { period: '30d', tags: [] });
  app.renderRecentActivityReport(activity, { active_days: 16, total_days: 25, current_streak: 3,
    longest_streak: 6, peak_times: ['Monday'], peak_periods: [{ label: 'Monday', weekday: 0 }],
    breakdown: [{ label: 'Monday', weekday: 0, count: 6 }, { label: 'Tuesday', weekday: 1, count: 2 }] });
  app.renderActivityHeatmapChart(heat, { total_days: 16, max_count: 3,
    data: [{ date: '2026-10-05', count: 2, level: 3 }] });
  app.renderActivityHeatmapChart(noHeat, { data: [] });
  out[lang] = { tags: tags.innerHTML, noTags: noTags.innerHTML, activity: activity.innerHTML,
                heat: heat.innerHTML, noHeat: noHeat.innerHTML };
}
console.log(JSON.stringify(out));
"""

METHODS = ("t", "replacePlaceholders", "escapeHtml", "staticTextHtml", "formatActivityPeriod",
           "renderTopTagsReport", "renderRecentActivityReport", "renderActivityHeatmapChart")


@pytest.fixture(scope="module")
def rendered():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = APP_JS.read_text(encoding="utf-8")
    script = (HARNESS.replace("__DE__", json.dumps(_catalog("de.json")))
              .replace("__EN__", json.dumps(_catalog("en.json")))
              .replace("__METHODS__", "".join(_method(js, name) for name in METHODS)))
    run = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(run.stdout)


def test_reports_render_in_german(rendered):
    de = rendered["de"]
    assert "Zeitraum: Letzte 30 Tage" in de["tags"]
    assert "12 Erinnerungen (57.1%)" in de["tags"]
    assert "Oft zusammen mit: testing (0.80)" in de["tags"]
    assert de["noTags"] == '<p data-i18n="analytics.topTagsEmpty">Keine Tags gefunden</p>'
    for text in ('<strong data-i18n="analytics.activeDays">Aktive Tage:</strong> 16/25',
                 '<strong data-i18n="analytics.currentStreak">Aktuelle Serie:</strong> 3 Tage',
                 '<strong data-i18n="analytics.longestStreak">Längste Serie:</strong> 6 Tage',
                 '<strong data-i18n="analytics.peakTimes">Spitzenzeiten:</strong> Montag',
                 'title="6 Erinnerungen (75.0%)"'):
        assert text in de["activity"], text
    for text in ("<span>16 aktive Tage</span>", "<span>Max.: 3 Erinnerungen/Tag</span>",
                 'title="2026-10-05: 2 Erinnerungen"',
                 '<span data-i18n="analytics.heatmapLess">Weniger</span>',
                 '<span data-i18n="analytics.heatmapMore">Mehr</span>'):
        assert text in de["heat"], text
    assert de["noHeat"] == '<p data-i18n="analytics.heatmapEmpty">Keine Aktivitätsdaten verfügbar</p>'
    for english in ("Period:", "memories", "Often with", "Active Days", "Streak", "Peak Times",
                    "active days", "Max:", "Less", "More"):
        # The data-i18n key names are English by design; only the visible text counts.
        assert all(english not in re.sub(r' data-i18n="[^"]*"', "", html) for html in de.values()), english


def test_reports_still_read_the_same_in_english(rendered):
    en = rendered["en"]
    assert "Period: Last 30 Days" in en["tags"]
    assert "12 memories (57.1%)" in en["tags"]
    assert "Often with: testing (0.80)" in en["tags"]
    assert "Current Streak:</strong> 3 days" in en["activity"]
    assert 'title="6 memories (75.0%)"' in en["activity"]
    assert "<span>16 active days</span><span>Max: 3 memories/day</span>" in en["heat"]
    assert 'title="2026-10-05: 2 memories"' in en["heat"]
    assert en["noHeat"] == '<p data-i18n="analytics.heatmapEmpty">No activity data available</p>'
