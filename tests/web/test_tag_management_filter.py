"""Tests for the Tag Management minimum-count filter (#1972)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "src" / "mcp_memory_service" / "web" / "static" / "app.js"
INDEX_HTML = REPO_ROOT / "src" / "mcp_memory_service" / "web" / "static" / "index.html"
I18N_DIR = REPO_ROOT / "src" / "mcp_memory_service" / "web" / "static" / "i18n"


def test_index_html_has_filter_controls():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="tagMinCountInput"' in html
    assert 'id="tagFilterShowing"' in html
    assert 'min="1"' in html
    assert 'value="2"' in html
    assert 'data-i18n="manage.tags.filter.minCount"' in html
    assert 'data-i18n="manage.tags.filter.showing"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tag_filtering_by_min_count():
    source = APP_JS.read_text(encoding="utf-8")
    class_src = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script = """
function el(tag, initial = {}) {
  const listeners = {};
  return {
    tag,
    value: initial.value ?? '',
    innerHTML: '',
    textContent: '',
    dataset: {},
    closest: () => null,
    addEventListener(evt, fn) { (listeners[evt] = listeners[evt] || []).push(fn); },
    trigger(evt) { (listeners[evt] || []).forEach(fn => fn()); }
  };
}

const minInput = el('input', { value: '2' });
const showingEl = el('p');
const tableContainer = el('div');

global.document = {
  getElementById(id) {
    if (id === 'tagMinCountInput') return minInput;
    if (id === 'tagFilterShowing') return showingEl;
    if (id === 'tagManagementContainer') return tableContainer;
    return null;
  }
};

""" + class_src + """
const app = Object.create(MemoryDashboard.prototype);
app.translations = {
  'manage.tags.filter.showing': 'Showing {count} of {extra} tags',
  'manage.tags.column.tag': 'Tag',
  'manage.tags.column.count': 'Count',
  'manage.tags.column.actions': 'Actions',
  'manage.tags.rename': 'Rename',
  'manage.tags.delete': 'Delete'
};
app.fallbackTranslations = app.translations;

const data = {
  tags: [
    { tag: 'single-use', count: 1 },
    { tag: 'double-use', count: 2 },
    { tag: 'frequent', count: 10 }
  ]
};

// 1. Default filter (minCount = 2)
app.renderTagManagementTable(data);
const defaultShowing = showingEl.textContent;
const defaultHtml = tableContainer.innerHTML;

// 2. Change filter to minCount = 1
minInput.value = '1';
app.renderTagManagementTable(data);
const allShowing = showingEl.textContent;
const allHtml = tableContainer.innerHTML;

// 3. Change filter to minCount = 10
minInput.value = '10';
app.renderTagManagementTable(data);
const tenShowing = showingEl.textContent;
const tenHtml = tableContainer.innerHTML;

console.log(JSON.stringify({
  defaultShowing,
  hasSingleInDefault: defaultHtml.includes('single-use'),
  hasDoubleInDefault: defaultHtml.includes('double-use'),
  hasFrequentInDefault: defaultHtml.includes('frequent'),
  allShowing,
  hasSingleInAll: allHtml.includes('single-use'),
  tenShowing,
  hasDoubleInTen: defaultHtml.includes('double-use'),
  hasFrequentInTen: tenHtml.includes('frequent'),
  notHasDoubleInTen: !tenHtml.includes('double-use')
}));
"""
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["defaultShowing"] == "Showing 2 of 3 tags"
    assert not out["hasSingleInDefault"]
    assert out["hasDoubleInDefault"]
    assert out["hasFrequentInDefault"]

    assert out["allShowing"] == "Showing 3 of 3 tags"
    assert out["hasSingleInAll"]

    assert out["tenShowing"] == "Showing 1 of 3 tags"
    assert out["notHasDoubleInTen"]
    assert out["hasFrequentInTen"]
