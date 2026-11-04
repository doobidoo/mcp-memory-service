"""The memory detail, add and edit modals and their empty states are translated (#1815).

The static tests read the real catalogs, index.html and app.js. The behavior tests
extract the real methods from app.js and run them in node against the real German
catalog, with stubs only for the DOM.
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

# Every key #1815 adds, as en.json has it.
NEW_STRINGS = {
    "toast.memorySavedSuccess": "Memory saved successfully",
    "toast.memoryUpdatedSuccess": "Memory updated successfully",
    "toast.memoryDetailsLoading": "Loading memory details...",
    "toast.memoryDetailsLoadFailed": "Failed to load memory details",
    "toast.ratingSaved": "Rating saved, quality score is now {score}",
    "toast.ratingSaveFailed": "Failed to save rating",
    "search.filters.activeTag": "Tag: {tag}",
    "search.filters.activeDate": "Date: {date}",
    "search.filters.activeType": "Type: {type}",
    "search.results.none": "No results found. Try a different search term.",
    "browse.noTags": "No tags found. Start adding tags to your memories to see them here.",
    "browse.noTagsShort": "No tags found.",
    "browse.noTaggedMemories": "No memories found with this tag.",
    "sections.recentMemoriesEmpty": "No memories found.",
    "sections.addFirstMemory": "Add your first memory",
    "modal.memoryDetails.created": "Created:",
    "modal.memoryDetails.updated": "Updated:",
    "modal.memoryDetails.type": "Type:",
    "modal.memoryDetails.id": "ID:",
    "modal.memoryDetails.content": "Content",
    "modal.memoryDetails.tags": "Tags",
    "modal.memoryDetails.metadata": "Metadata",
    "modal.memoryDetails.noMetadata": "No metadata available",
    "modal.memoryDetails.showFullContent": "Show full content",
    "modal.memoryDetails.collapse": "Collapse",
    "modal.memoryDetails.confirmDelete": "Are you sure you want to delete this memory? This action cannot be undone.",
    "modal.memoryDetails.shareTitle": "Memory from MCP Memory Service",
    "modal.addMemory.editTitle": "Edit Memory",
    "modal.addMemory.update": "Update Memory",
    "quality.noMemories": "No memories available",
    "quality.badgeTitle": "Quality score {score} ({provider})",
}

# Existing keys this change starts using from app.js.
REUSED_STRINGS = {
    "modal.addMemory.title": "Add New Memory",
    "modal.addMemory.save": "Save Memory",
    "modal.memoryDetails.title": "Memory Details",
    "search.filters.date.today": "Today",
    "search.filters.date.week": "This week",
    "search.filters.date.month": "This month",
    "search.filters.date.year": "This year",
    "search.filters.type.note": "Notes",
    "search.filters.type.code": "Code",
    "search.filters.type.reference": "References",
    "search.filters.type.idea": "Ideas",
}

# app.js: key -> how many this.t() / setTranslatedText() calls use it.
JS_SITES = {key: 1 for key in {**NEW_STRINGS, **REUSED_STRINGS}}
JS_SITES.update({
    "toast.memoryUpdatedSuccess": 2,  # metadata-only PUT and content change
    "toast.memoryDetailsLoading": 2,  # 3D and 2D graph node click
    "toast.memoryDetailsLoadFailed": 2,
    "quality.noMemories": 2,  # top and bottom quality lists
    "modal.memoryDetails.showFullContent": 2,  # rendered button and its toggle
})

# Identical to English is a real translation here (an abbreviation or a loanword).
SAME_AS_ENGLISH_OK = {
    "modal.memoryDetails.id",
    "modal.memoryDetails.tags",
    "modal.memoryDetails.metadata",
    "search.filters.activeTag",
}

# English literals that must no longer reach the DOM, a dialog or a toast.
OLD_LITERALS = [
    "textContent = 'Add New Memory'",
    "textContent = 'Save Memory'",
    "textContent = 'Edit Memory'",
    "textContent = 'Update Memory'",
    "textContent = 'Memory Details'",
    "successMessage = 'Memory updated successfully'",
    "successMessage = 'Memory saved successfully'",
    "<strong>Created:</strong>",
    "<strong>Updated:</strong>",
    "<strong>Type:</strong>",
    "<strong>ID:</strong>",
    "<h4>Content</h4>",
    "<h4>Tags</h4>",
    "</span> Metadata",
    "'<p class=\"metadata-empty\">No metadata available",
    "confirm('Are you sure you want to delete this memory?",
    "title: 'Memory from MCP Memory Service'",
    "No memories found. <a",
    "'<div class=\"no-tags\">No tags found.",
    "'<p class=\"empty-state\">No results found.",
    "'<p class=\"text-neutral-600\">No tags found.",
    "'<p class=\"empty-state\">No memories found with this tag.",
    "`Tag: ${tag}`",
    "`Date: ${",
    "`Type: ${",
    "'today': 'Today'",
    "'note': 'Notes'",
    "showToast('Loading memory details...'",
    "showToast('Failed to load memory details'",
    "'<p class=\"text-muted\">No memories available",
    "title=\"Quality score ${",
    "`Rating saved, quality score is now ${",
    "showToast('Failed to save rating'",
    "'▼ Show full content'",
    "'▲ Collapse'",
    "toggle-content\">▼ Show full content<",
]


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def _tokens(text):
    return sorted(re.findall(r"\{\w+\}", text))


CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))
NON_ENGLISH = [name for name in CATALOGS if name != "en.json"]


def test_english_catalog_has_every_new_and_reused_string():
    english = _catalog("en.json")
    expected = {**NEW_STRINGS, **REUSED_STRINGS}
    assert {k: english.get(k) for k in expected} == expected


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_key_with_the_same_tokens(name):
    catalog = _catalog(name)
    missing = [k for k in NEW_STRINGS if not str(catalog.get(k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"
    wrong_tokens = [k for k, en in NEW_STRINGS.items() if _tokens(catalog[k]) != _tokens(en)]
    assert not wrong_tokens, f"{name} changes the placeholders of {wrong_tokens}"
    assert not [k for k in NEW_STRINGS if "<" in catalog[k] or ">" in catalog[k]]


@pytest.mark.parametrize("name", NON_ENGLISH)
def test_non_english_catalogs_really_translate(name):
    catalog = _catalog(name)
    untranslated = [k for k, en in NEW_STRINGS.items() if k not in SAME_AS_ENGLISH_OK and catalog[k] == en]
    assert not untranslated, f"{name} keeps English for {untranslated}"
    # The product name stays as it is
    assert "MCP Memory Service" in catalog["modal.memoryDetails.shareTitle"]


def test_new_keys_sit_next_to_their_prefix_group():
    # Appending at the end of a catalog conflicts with every parallel #1806 slice.
    for name in CATALOGS:
        keys = [m.group(1) for m in re.finditer(r'^\s*"([^"]+)"\s*:', (I18N_DIR / name).read_text(encoding="utf-8"), re.M)]
        for key in NEW_STRINGS:
            prefix = key.rsplit(".", 1)[0]
            i = keys.index(key)
            neighbours = keys[max(i - 1, 0):i] + keys[i + 1:i + 2]
            assert any(n.startswith(prefix.split(".")[0] + ".") for n in neighbours), (name, key)


def test_app_js_uses_the_keys_with_english_fallbacks():
    js = APP_JS.read_text(encoding="utf-8")
    expected = {**NEW_STRINGS, **REUSED_STRINGS}
    calls = re.findall(r"(?:this\.t|this\.setTranslatedText\(\w+,)\(?\s*'([\w.]+)', '((?:[^'\\]|\\.)*)'", js)
    used = {}
    for key, fallback in calls:
        if key in JS_SITES:
            assert fallback == expected[key], key
            used[key] = used.get(key, 0) + 1
    assert used == JS_SITES


def test_app_js_has_no_english_left_at_the_listed_sites():
    js = APP_JS.read_text(encoding="utf-8")
    left = [literal for literal in OLD_LITERALS if literal in js]
    assert left == []


def test_static_modal_markup_keeps_its_keys():
    # The title and save button start from these keys, and the add/edit switch now moves them.
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert re.search(r'id="modalTitle"[^>]*data-i18n="modal\.memoryDetails\.title"', html)
    assert re.search(r'id="saveMemoryBtn"[^>]*data-i18n="modal\.addMemory\.save"', html)
    assert 'data-i18n="modal.addMemory.title"' in html


def _method(js, name):
    match = re.search(rf"^    {name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    assert match, name
    return match.group(0)


HARNESS = r"""
const de = __DE__, en = __EN__;
const mk = (id, text, i18n) => ({ id, textContent: text, innerHTML: '', value: '', style: {},
  dataset: i18n ? { i18n } : {}, focus() {}, reset() {}, set onclick(f) {} });
const title = mk('addTitle', 'Add New Memory', 'modal.addMemory.title');
const els = {
  saveMemoryBtn: mk('saveMemoryBtn', 'Save Memory', 'modal.addMemory.save'),
  modalTitle: mk('modalTitle', 'Memory Details', 'modal.memoryDetails.title'),
  modalContent: mk('modalContent', ''),
  memoryContent: mk('memoryContent', ''), memoryTags: mk('memoryTags', ''), memoryType: mk('memoryType', ''),
  addMemoryForm: mk('addMemoryForm', ''), editMemoryBtn: mk('e', ''), deleteMemoryBtn: mk('d', ''), shareMemoryBtn: mk('s', ''),
  searchResultsList: mk('searchResultsList', ''), tagsCloudContainer: mk('tagsCloudContainer', ''),
  taggedMemoriesContainer: mk('taggedMemoriesContainer', ''),
  'quality-top-memories-list': mk('q1', ''), 'quality-bottom-memories-list': mk('q2', ''),
  activeFilters: mk('activeFilters', ''), activeFiltersList: mk('activeFiltersList', ''),
  tagFilter: mk('tagFilter', ''), dateFilter: mk('dateFilter', ''), typeFilter: mk('typeFilter', ''),
};
els.addMemoryModal = { querySelector: () => title };
els.memoryModal = {};
els.activeFiltersList.addEventListener = () => {};
globalThis.document = { getElementById: id => els[id] || null };
globalThis.setTimeout = () => {};
class App {
__METHODS__
  openModal() {} closeModal() {}
}
const app = new App();
app.translations = de; app.fallbackTranslations = en;
const out = {};
const memory = { content: 'x'.repeat(250), content_hash: 'abc123', created_at: 1700000000, updated_at: 1700000100,
                 memory_type: 'note', tags: ['work'], metadata: null };
out.details = app.renderMemoryDetails(memory);
out.noMetadata = app.renderMetadata(null);
app.showMemoryDetails(memory);
out.detailTitle = [els.modalTitle.textContent, els.modalTitle.dataset.i18n];
app.editMemory(memory);
out.edit = [title.textContent, title.dataset.i18n, els.saveMemoryBtn.textContent, els.saveMemoryBtn.dataset.i18n];
app.resetAddMemoryModal();
out.reset = [title.textContent, title.dataset.i18n, els.saveMemoryBtn.textContent, els.saveMemoryBtn.dataset.i18n];
app.renderSearchResults([]);
out.noResults = els.searchResultsList.innerHTML;
app.tags = [];
app.renderTagsCloud();
out.noTags = els.tagsCloudContainer.innerHTML;
app.renderTagsSidebar([]);
out.noTagsShort = els.tagsCloudContainer.innerHTML;
app.renderMemoriesInContainer([], els.taggedMemoriesContainer);
out.noTagged = els.taggedMemoriesContainer.innerHTML;
app.renderTopQualityMemories([]);
out.noQuality = els['quality-top-memories-list'].innerHTML;
out.badge = app.renderQualityBadge({ quality_score: 0.8, quality_provider: 'groq"<x>' });
els.tagFilter.value = 'work'; els.dateFilter.value = 'week'; els.typeFilter.value = 'idea';
app.updateActiveFilters();
out.filters = els.activeFiltersList.innerHTML;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def german_run():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = APP_JS.read_text(encoding="utf-8")
    names = ("t", "replacePlaceholders", "setTranslatedText", "escapeHtml", "renderMemoryDetails",
             "renderMetadata", "showMemoryDetails", "editMemory", "resetAddMemoryModal",
             "renderSearchResults", "renderTagsCloud", "renderTagsSidebar", "renderMemoriesInContainer",
             "renderTopQualityMemories", "renderQualityBadge", "updateActiveFilters")
    methods = "".join(_method(js, name) for name in names)
    de, en = _catalog("de.json"), _catalog("en.json")
    script = (HARNESS.replace("__DE__", json.dumps(de)).replace("__EN__", json.dumps(en))
              .replace("__METHODS__", methods))
    run = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout), de


def test_detail_modal_renders_in_the_selected_language(german_run):
    out, de = german_run
    for key in ("created", "updated", "type", "id", "content", "tags", "showFullContent"):
        assert de[f"modal.memoryDetails.{key}"] in out["details"], key
    for key in ("created", "updated", "type", "id", "content", "tags"):
        assert f'data-i18n="modal.memoryDetails.{key}"' in out["details"], key
    # German keeps "Tags", so that heading is checked through its key above
    for english in ("Created:", "Updated:", ">Content<", "Show full content"):
        assert english not in out["details"], english
    assert de["modal.memoryDetails.noMetadata"] in out["noMetadata"]
    assert out["detailTitle"] == [de["modal.memoryDetails.title"], "modal.memoryDetails.title"]


def test_edit_and_add_titles_carry_their_key(german_run):
    # Before, editing left data-i18n="modal.addMemory.save" on "Update Memory", so a
    # language switch turned the edit button back into "Save Memory".
    out, de = german_run
    assert out["edit"] == [de["modal.addMemory.editTitle"], "modal.addMemory.editTitle",
                           de["modal.addMemory.update"], "modal.addMemory.update"]
    assert out["reset"] == [de["modal.addMemory.title"], "modal.addMemory.title",
                            de["modal.addMemory.save"], "modal.addMemory.save"]


def test_empty_states_render_in_the_selected_language(german_run):
    out, de = german_run
    for slot, key in (("noResults", "search.results.none"), ("noTags", "browse.noTags"),
                      ("noTagsShort", "browse.noTagsShort"), ("noTagged", "browse.noTaggedMemories"),
                      ("noQuality", "quality.noMemories")):
        assert de[key] in out[slot], slot
        assert f'data-i18n="{key}"' in out[slot], slot


def test_quality_badge_title_is_translated_and_escaped(german_run):
    out, de = german_run
    expected = de["quality.badgeTitle"].replace("{score}", "0.80").replace("{provider}", 'groq&quot;&lt;x&gt;')
    assert f'title="{expected}"' in out["badge"]


def test_active_filter_chips_use_the_selected_language(german_run):
    out, de = german_run
    filters = out["filters"]
    assert de["search.filters.activeTag"].replace("{tag}", "work") in filters
    assert de["search.filters.activeDate"].replace("{date}", de["search.filters.date.week"]) in filters
    assert de["search.filters.activeType"].replace("{type}", de["search.filters.type.idea"]) in filters
    for english in ("Date: ", "Type: ", "This week", "Ideas"):
        assert english not in filters, english
