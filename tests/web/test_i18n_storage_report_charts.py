"""The storage report and the Analytics chart labels are translated (#1810).

The static tests read the real catalogs and app.js. The behavior test loads the real
app.js in node, gives the dashboard the real German catalog and renders the storage
report, the growth, tag usage, memory types and relationship types charts and the
quality charts. Only the DOM lookups, the API call and Chart.js are stubbed.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "mcp_memory_service" / "web" / "static"
APP_JS = STATIC / "app.js"
I18N_DIR = STATIC / "i18n"

# Every key #1810 wires, as en.json has it.
STRINGS = {
    "analytics.growthLoadFailed": "Failed to load growth chart",
    "analytics.noGrowthData": "No growth data available",
    "analytics.tagUsageLoadFailed": "Failed to load tag usage chart",
    "analytics.noTagUsage": "No tags found",
    "analytics.minorTagsCount": "{count} tags",
    "analytics.minorTagsLabel": "more tags (≤10 memories each)",
    "analytics.minorTagsTip": "{assignments} tag assignments across {count} tags with 10 or fewer memories each",
    "analytics.memoryTypesLoadFailed": "Failed to load memory types chart",
    "analytics.noMemoryTypes": "No memory types found",
    "analytics.minorTypesLabel": "diverse ({count} types)",
    "analytics.minorTypesTip": "{count} types with ≤10 memories each",
    "analytics.relationshipTypesLoadFailed": "Failed to load relationship types chart",
    "analytics.noRelationshipTypes": "No relationships found",
    "analytics.storageLoadFailed": "Failed to load storage report",
    "analytics.storageTotalSize": "Total Size:",
    "analytics.storageSizeMb": "{size} MB",
    "analytics.storageAverageMemory": "Average Memory:",
    "analytics.storageAverageChars": "{count} chars",
    "analytics.storageEfficiency": "Efficiency:",
    "analytics.largestMemories": "Largest Memories",
    "analytics.storageUnknownDate": "Unknown",
    "analytics.storageSizeKb": "{size} KB",
    "analytics.storageSizeBytes": "{size} bytes",
    "analytics.storageTags": "Tags: {tags}",
    "analytics.storageNoTags": "none",
    "quality.analytics.loadFailed": "Failed to load quality analytics",
    "quality.chart.distribution.low": "Low (<0.5)",
    "quality.chart.distribution.medium": "Medium (0.5-0.7)",
    "quality.chart.distribution.high": "High (≥0.7)",
    "quality.chart.distribution.memoriesLabel": "Number of Memories",
    "quality.chart.providers.local": "Local SLM",
    "quality.chart.providers.implicit": "Implicit Only",
}
# Keys that already existed and are now looked up: key -> English text.
REUSED = {"quality.chart.distribution.countLabel": "Count"}
# app.js: key -> how many this.t() calls use it.
JS_SITES = {key: 1 for key in STRINGS}
JS_SITES.update({
    "quality.chart.providers.local": 2,
    "quality.chart.providers.implicit": 2,
    "quality.chart.distribution.countLabel": 1,
})
# The English literals the issue lists; none may be left in app.js.
LITERALS = [
    "'<p class=\"error\">Failed to load growth chart",
    "'<p>No growth data available",
    "'<p class=\"error\">Failed to load tag usage chart",
    "'<p class=\"error\">Failed to load memory types chart",
    "'<p>No memory types found",
    "'<p class=\"error\">Failed to load storage report",
    "<strong>Total Size:</strong>",
    "<strong>Average Memory:</strong>",
    "<strong>Efficiency:</strong>",
    "'<h4>Largest Memories</h4>'",
    "toLocaleDateString() : 'Unknown'",
    "} chars</div>",
    "• Tags: ${",
    "'<p class=\"error\">Failed to load relationship types chart",
    "'<p>No relationships found",
    "this.showToast('Failed to load quality analytics'",
    "labels: ['Low (<0.5)'",
    "label: 'Number of Memories'",
    "text: 'Count'",
    "return 'Local SLM'",
    "return 'Implicit Only'",
    "tags</span>",
    ">more tags (≤10 memories each)</span>",
    ">diverse (${",
]


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))


def _placeholders(text):
    return sorted(re.findall(r"\{(\w+)\}", text))


def test_english_catalog_has_every_string():
    english = _catalog("en.json")
    expected = {**STRINGS, **REUSED}
    assert {k: english.get(k) for k in expected} == expected


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_key_with_its_placeholders(name):
    catalog = _catalog(name)
    missing = [k for k in STRINGS if not str(catalog.get(k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"
    wrong = [k for k in STRINGS if _placeholders(catalog[k]) != _placeholders(STRINGS[k])]
    assert not wrong, f"{name} changes the placeholders of {wrong}"


def test_app_js_uses_the_keys_with_english_fallbacks():
    js = APP_JS.read_text(encoding="utf-8")
    calls = re.findall(r"this\.t\('((?:analytics|quality)\.[\w.]+)',\s*'((?:[^'\\]|\\.)*)'", js)
    english = {**STRINGS, **REUSED}
    used = {}
    for key, fallback in calls:
        if key in JS_SITES:
            assert fallback == english[key], key
            used[key] = used.get(key, 0) + 1
    assert used == JS_SITES
    for literal in LITERALS:
        assert literal not in js, literal


HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const [appPath, dePath, enPath] = process.argv.slice(2);
const de = JSON.parse(fs.readFileSync(dePath, 'utf8'));
const en = JSON.parse(fs.readFileSync(enPath, 'utf8'));
const charts = [];
const elements = {};
const el = id => (elements[id] = elements[id] || { id, innerHTML: '', clientWidth: 600, getContext: () => ({}) });
const context = vm.createContext({
    console: { log() {}, error() {}, warn() {} },
    document: {
        hidden: false,
        body: { classList: { contains: () => false } },
        addEventListener() {},
        getElementById: el,
        querySelector: () => null,
        querySelectorAll: () => [],
    },
    window: { addEventListener() {} },
    localStorage: { getItem: () => null, setItem() {} },
    getComputedStyle: () => ({ paddingLeft: '0', paddingRight: '0' }),
    Chart: class { constructor(ctx, config) { charts.push(config); } destroy() {} },
    setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
});
vm.runInContext(fs.readFileSync(appPath, 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
const app = Object.create(context.Dashboard.prototype);
app.translations = de;
app.fallbackTranslations = en;
const toasts = [];
app.showToast = (message) => toasts.push(message);
const render = (method, data) => { const c = { innerHTML: '' }; app[method](c, data); return c.innerHTML; };
(async () => {
    const out = {};
    out.storage = render('renderStorageReport', {
        total_size_mb: 1.5, average_memory_size: 120, storage_efficiency: 87,
        largest_memories: [
            { created_at: null, size_kb: 4.2, preview: 'a', tags: [] },
            { created_at: null, size_bytes: 512, preview: 'b', tags: ['<b>x</b>', 'y'] },
        ],
    });
    out.growthEmpty = render('renderMemoryGrowthChart', { data_points: [] });
    out.tagsEmpty = render('renderTagUsageChart', { tags: [] });
    out.tagsMinor = render('renderTagUsageChart', { tags: [
        { tag: 'a', count: 3, percentage: 25 }, { tag: 'b', count: 2, percentage: 10 }] });
    out.typesEmpty = render('renderMemoryTypesChart', { types: [] });
    out.typesMinor = render('renderMemoryTypesChart', { types: [
        { memory_type: 'note', count: 3, percentage: 30 }, { memory_type: 'idea', count: 1, percentage: 10 }] });
    out.relationsEmpty = render('renderRelationshipTypesChart', {});
    // The load paths: the API call fails, so each chart shows its error state.
    app.apiCall = async () => { throw new Error('offline'); };
    for (const [method, id] of [['loadMemoryGrowthChart', 'memoryGrowthChart'], ['loadTagUsageChart', 'tagUsageChart'],
                                ['loadMemoryTypesChart', 'memoryTypesChart'], ['loadStorageReport', 'storageReport'],
                                ['loadRelationshipTypesChart', 'relationshipTypesChart']]) {
        el('growthPeriodSelect').value = 'month';
        await app[method]();
        out[method] = el(id).innerHTML;
    }
    await app.loadQualityAnalytics();
    out.toasts = toasts;
    app.renderQualityDistributionChart({ low_quality_count: 1, medium_quality_count: 2, high_quality_count: 3 });
    app.renderQualityProviderChart({ ONNXRankerModel: 1, GroqEvaluator: 2, GeminiEvaluator: 3, ImplicitSignalsEvaluator: 4, local: 5, implicit: 6 });
    out.distribution = charts[0].data.labels;
    out.distributionLabel = charts[0].data.datasets[0].label;
    out.distributionAxis = charts[0].options.scales.y.title.text;
    out.providers = charts[1].data.labels;
    console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_storage_report_and_charts_render_in_the_selected_language(tmp_path):
    harness = tmp_path / "harness.cjs"
    harness.write_text(HARNESS, encoding="utf-8")
    run = subprocess.run(
        ["node", str(harness), str(APP_JS), str(I18N_DIR / "de.json"), str(I18N_DIR / "en.json")],
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=120,
    )
    out = json.loads(run.stdout)
    de = _catalog("de.json")

    def t(key, **values):
        return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), de[key])

    storage = out["storage"]
    for piece in (
        f'<strong data-i18n="analytics.storageTotalSize">{de["analytics.storageTotalSize"]}</strong> {t("analytics.storageSizeMb", size=1.5)}',
        f'<strong data-i18n="analytics.storageAverageMemory">{de["analytics.storageAverageMemory"]}</strong> {t("analytics.storageAverageChars", count=120)}',
        f'<strong data-i18n="analytics.storageEfficiency">{de["analytics.storageEfficiency"]}</strong> 87%',
        f'<h4 data-i18n="analytics.largestMemories">{de["analytics.largestMemories"]}</h4>',
        t("analytics.storageSizeKb", size=4.2),
        t("analytics.storageSizeBytes", size=512),
        f'{de["analytics.storageUnknownDate"]} • {t("analytics.storageTags", tags=de["analytics.storageNoTags"])}',
        # A tag is user data: it is escaped, not rendered as markup.
        t("analytics.storageTags", tags="&lt;b&gt;x&lt;/b&gt;, y"),
    ):
        assert piece in storage, piece
    # German keeps "Tags", so only the other words are checked; the data-i18n keys are not text.
    text = re.sub(r' data-i18n="[^"]*"', "", storage)
    for english in ("Total Size", "Average Memory", "Efficiency", "Largest Memories", "Unknown", " chars", "bytes", "none", "<b>x</b>"):
        assert english not in text, english

    assert out["growthEmpty"] == f'<p data-i18n="analytics.noGrowthData">{de["analytics.noGrowthData"]}</p>'
    assert out["tagsEmpty"] == f'<p data-i18n="analytics.noTagUsage">{de["analytics.noTagUsage"]}</p>'
    assert out["typesEmpty"] == f'<p data-i18n="analytics.noMemoryTypes">{de["analytics.noMemoryTypes"]}</p>'
    assert out["relationsEmpty"] == f'<p data-i18n="analytics.noRelationshipTypes">{de["analytics.noRelationshipTypes"]}</p>'

    assert f'<span class="chart-value">{t("analytics.minorTagsCount", count=2)}</span>' in out["tagsMinor"]
    assert f'title="{t("analytics.minorTagsTip", assignments=5, count=2)}"' in out["tagsMinor"]
    assert f'>{de["analytics.minorTagsLabel"]}</span>' in out["tagsMinor"]
    assert f'title="{t("analytics.minorTypesTip", count=2)}">{t("analytics.minorTypesLabel", count=2)}</span>' in out["typesMinor"]

    for method, key in (
        ("loadMemoryGrowthChart", "analytics.growthLoadFailed"),
        ("loadTagUsageChart", "analytics.tagUsageLoadFailed"),
        ("loadMemoryTypesChart", "analytics.memoryTypesLoadFailed"),
        ("loadStorageReport", "analytics.storageLoadFailed"),
        ("loadRelationshipTypesChart", "analytics.relationshipTypesLoadFailed"),
    ):
        assert out[method] == f'<p class="error" data-i18n="{key}">{de[key]}</p>', method

    assert out["toasts"] == [de["quality.analytics.loadFailed"]]
    assert out["distribution"] == [de["quality.chart.distribution.low"], de["quality.chart.distribution.medium"],
                                   de["quality.chart.distribution.high"]]
    assert out["distributionLabel"] == de["quality.chart.distribution.memoriesLabel"]
    assert out["distributionAxis"] == de["quality.chart.distribution.countLabel"]
    local, implicit = de["quality.chart.providers.local"], de["quality.chart.providers.implicit"]
    # Product names stay as they are
    assert out["providers"] == [local, "Groq API", "Gemini API", implicit, local, implicit]
