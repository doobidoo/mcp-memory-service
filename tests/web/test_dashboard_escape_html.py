"""Regression test for GHSA-j664-v823-6p9f: dashboard escapeHtml() must encode quotes.

Callers interpolate escapeHtml() output into quoted attributes such as
data-tag="...". A stored tag containing a double quote used to close the
attribute and add an onmouseover handler to the rendered button.
"""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "app.js"


def _escape_html_body() -> str:
    match = re.search(r"^    escapeHtml\(text\) \{\n(.*?)^    \}\n", APP_JS.read_text(), re.S | re.M)
    assert match, "escapeHtml(text) not found in app.js"
    return match.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_escape_html_encodes_attribute_delimiters():
    inputs = ['z" onmouseover="window.x=1" q="', "a'b", "<b>&</b>", None, 5]
    script = (
        f"const escapeHtml = function(text) {{\n{_escape_html_body()}}};\n"
        f"console.log(JSON.stringify({json.dumps(inputs)}.map(escapeHtml)));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [
        "z&quot; onmouseover=&quot;window.x=1&quot; q=&quot;",
        "a&#39;b",
        "&lt;b&gt;&amp;&lt;/b&gt;",
        "",
        "5",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("view", ["management", "usage", "top", "cooccurrence", "types"])
def test_dashboard_renders_stored_labels_as_text(view):
    """Stored markup must not become an element or an event attribute."""
    payload = '<img src=x onerror="window.compromised=1">'
    source = APP_JS.read_text()
    # Load the real class without bootstrapping the application or its network calls.
    script = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script += "\nconst app = Object.create(MemoryDashboard.prototype);\n"
    script += f"const payload = {json.dumps(payload)};\n"
    script += """
const container = {innerHTML: '', addEventListener() {}};
global.document = {getElementById: () => container};
const tag = {tag: payload, count: 11, percentage: 100, growth_rate: null};
const view = process.argv[2];
if (view === 'management') app.renderTagManagementTable({tags: [tag]});
if (view === 'usage') app.renderTagUsageChart(container, {tags: [tag]});
if (view === 'top') app.renderTopTagsReport(container, {tags: [tag], period: '30d'});
if (view === 'cooccurrence') app.renderTopTagsReport(container, {
    tags: [{...tag, tag: 'safe', co_occurring_tags: [{tag: payload, strength: 1}]}], period: '30d'
});
if (view === 'types') app.renderMemoryTypesChart(container, {
    types: [{memory_type: payload, count: 11, percentage: 100}]
});
console.log(container.innerHTML);
"""
    result = subprocess.run(["node", "-", view], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = result.stdout

    class ParsedHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            self.elements.append((tag, dict(attrs)))

        def handle_data(self, data):
            self.text.append(data)

    parsed = ParsedHTML()
    parsed.feed(output)
    assert all(tag != "img" for tag, _ in parsed.elements)
    assert all(not key.startswith("on") for _, attrs in parsed.elements for key in attrs)
    assert payload in "".join(parsed.text)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_tag_usage_diverse_bucket_has_no_summed_percentage():
    """Regression for #1736: per-tag percentages are shares of total memories, and a
    memory can carry several tags, so summing them can exceed 100% and overflow the
    200px bar scale (the 'diverse' row used to read '30 (249.4%)')."""
    source = APP_JS.read_text()
    script = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script += "\nconst app = Object.create(MemoryDashboard.prototype);\n"
    script += """
const container = {innerHTML: '', addEventListener() {}};
global.document = {getElementById: () => container};
function render(tags) {
  app.renderTagUsageChart(container, {tags});
  return container.innerHTML;
}
const results = [];
// Issue repro: 12 memories, 20 tags -> all minor (count <= 10), summed percentage 249.4%.
const overflow = [
  {tag: 'work', count: 6, percentage: 50.0, growth_rate: null},
  {tag: 'home', count: 6, percentage: 50.0, growth_rate: null},
];
for (let i = 0; i < 18; i++) overflow.push({tag: 't' + i, count: 1, percentage: 8.3, growth_rate: null});
results.push(render(overflow));
// Mixed case: 2 significant tags + 18 minor tags (bar must stay proportional and <= 200px).
const mixed = [
  {tag: 'work', count: 12, percentage: 30.0, growth_rate: null},
  {tag: 'home', count: 12, percentage: 30.0, growth_rate: null},
];
for (let i = 0; i < 18; i++) mixed.push({tag: 't' + i, count: 1, percentage: 2.5, growth_rate: null});
results.push(render(mixed));
console.log(JSON.stringify(results));
"""
    result = subprocess.run(["node", "-"], check=False, input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout.splitlines()[-1])
    assert len(outputs) == 2
    for output in outputs:
        # The diverse bucket is labelled by tag count only, never 'N (P%)'.
        assert re.search(r"chart-value\">\d+ \(\d+\.\d+%\)", output) is None
        assert "more tags" in output
        # Bars never exceed the 200px scale.
        for match in re.finditer(r'chart-bar" style="width: ([\d.]+)px"', output):
            assert float(match.group(1)) <= 200.0, output
    assert "20 more tags" in outputs[0]
    assert "18 more tags" in outputs[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_analytics_db_size_shows_0_mb_when_falsy():
    """Regression for #1736: a 4 KB database rounds to database_size_mb: 0.0, which is
    falsy, so the tile previously read N/A although the API returned a real size."""
    source = APP_JS.read_text()
    script = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script += "\nconst app = Object.create(MemoryDashboard.prototype);\n"
    script += """
(async () => {
  const els = {};
  global.document = {getElementById: (id) => (els[id] = els[id] || {textContent: ''})};
  app.apiCall = async () => ({total_memories: 12, memories_this_week: 3, unique_tags: 20, database_size_mb: 0.0});
  await app.loadAnalyticsOverview();
  const zero = els.analyticsDbSize.textContent;
  app.apiCall = async () => ({total_memories: 12, memories_this_week: 3, unique_tags: 20, database_size_mb: null});
  await app.loadAnalyticsOverview();
  const missing = els.analyticsDbSize.textContent;
  console.log(JSON.stringify({zero, missing}));
})();
"""
    result = subprocess.run(["node", "-"], check=False, input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout.splitlines()[-1])
    assert values["zero"] == "0.0 MB"
    assert values["missing"] == "N/A"
