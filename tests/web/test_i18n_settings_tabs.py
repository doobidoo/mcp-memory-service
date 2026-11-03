"""The Settings tab names and the Environment, System Info and Backup tabs are translated (#1812).

The static tests read the real catalogs, index.html and app.js. The behavior
tests extract the real methods from app.js and run them in node against the
real German catalog and the real index.html markup of the fields they write,
with stubs only for the DOM, the API and the clipboard (the same approach as
test_i18n_server_tab.py).
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

# Every new key, as en.json has it.
NEW_STRINGS = {
    "modal.settings.tabs.preferences": "Preferences",
    "modal.settings.tabs.credentials": "Credentials",
    "modal.settings.tabs.environment": "Environment",
    "modal.settings.tabs.system": "System Info",
    "modal.settings.tabs.quality": "Quality",
    "modal.settings.tabs.backup": "Backup",
    "modal.settings.tabs.server": "Server",
    "modal.settings.env.description": (
        "View your current environment configuration. "
        "To change settings, edit your .env file and restart the server."
    ),
    "modal.settings.env.copy": "Copy Configuration",
    "modal.settings.env.sensitive": "Sensitive",
    "modal.settings.env.notSet": "(not set)",
    "modal.settings.env.enabled": "✓ Enabled",
    "modal.settings.env.disabled": "✗ Disabled",
    "modal.settings.env.options": "(options: {options})",
    "modal.settings.env.loadFailed": "Failed to load configuration",
    "modal.settings.env.noConfig": "No configuration loaded",
    "modal.settings.env.copied": "Configuration copied to clipboard",
    "modal.settings.system.notAvailable": "N/A",
    "modal.settings.system.enabled": "✓ Enabled",
    "modal.settings.system.disabled": "Disabled",
    "modal.settings.system.error": "Error",
    "modal.settings.backup.disabled": "Backups disabled",
    "modal.settings.backup.scheduled": "Scheduled",
    "modal.settings.backup.notScheduled": "Not scheduled",
    "modal.backups.title": "Backup History",
    "modal.backups.loading": "Loading backups...",
    "modal.backups.close": "Close",
    "modal.backups.summaryOne": "{count} backup · {size} total",
    "modal.backups.summaryMany": "{count} backups · {size} total",
    "modal.backups.empty": 'No backups found. Create one using "{button}".',
    "modal.backups.unknownDate": "Unknown",
    "modal.backups.ageOne": "1 day ago",
    "modal.backups.ageDays": "{days} days ago",
    "modal.backups.loadFailed": "Failed to load backups.",
    "time.seconds": "{count}s",
    "time.minutes": "{count}m",
    "time.hours": "{count}h",
    "time.days": "{count}d",
    "time.underAMinute": "< 1m",
    "time.ago": "{time} ago",
}
# Existing keys the change reuses; app.js passes them with the same fallbacks.
REUSED_KEYS = {"modal.settings.never", "modal.settings.backupNow", "status.loading", "toast.copyFailed"}
# The static markup that carries the new keys, by data-i18n key.
HTML_KEYS = {
    *(f"modal.settings.tabs.{tab}" for tab in
      ("preferences", "credentials", "environment", "system", "quality", "backup", "server")),
    "modal.settings.env.description", "modal.settings.env.copy",
    "modal.backups.title", "modal.backups.loading", "modal.backups.close",
}
METHODS = [
    "formatDuration", "formatTimeDelta", "formatUptime", "checkBackupStatus", "openViewBackupsModal",
    "loadBackupsList", "loadEnvironmentConfig", "renderParamValue", "copyEnvironmentConfig",
    "resetSystemInfoLoadingState", "loadSystemInfo",
]
TOKEN_RE = re.compile(r"\{[A-Za-z0-9_]+\}")
CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    return match.group(0).strip() if match else None


# --- catalogs ---------------------------------------------------------------


def test_english_catalog_has_every_new_string():
    english = _catalog("en.json")
    assert {k: english.get(k) for k in NEW_STRINGS} == NEW_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_new_key(name):
    catalog = _catalog(name)
    missing = [k for k in NEW_STRINGS if not str(catalog.get(k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"
    tokens = {
        k: (sorted(TOKEN_RE.findall(catalog[k])), sorted(TOKEN_RE.findall(english)))
        for k, english in NEW_STRINGS.items()
    }
    assert {k: got for k, (got, want) in tokens.items() if got != want} == {}


@pytest.mark.parametrize("name", CATALOGS)
def test_env_file_name_stays_untranslated(name):
    assert ".env" in _catalog(name)["modal.settings.env.description"]


def test_every_new_key_is_used():
    sources = APP_JS.read_text(encoding="utf-8") + INDEX_HTML.read_text(encoding="utf-8")
    assert [k for k in NEW_STRINGS if f"'{k}'" not in sources and f'"{k}"' not in sources] == []


# --- index.html -------------------------------------------------------------


class _RegionText(HTMLParser):
    """Collect (text, innermost element attrs) inside elements matching an id or a class."""

    VOID = {"br", "img", "input", "meta", "link", "hr", "path"}

    def __init__(self, ids=(), classes=()):
        super().__init__(convert_charrefs=True)
        self.ids, self.classes = set(ids), set(classes)
        self.stack, self.region_depth, self.found = [], None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.VOID:
            return
        self.stack.append(attrs)
        in_region = attrs.get("id") in self.ids or self.classes & set((attrs.get("class") or "").split())
        if self.region_depth is None and in_region:
            self.region_depth = len(self.stack)

    def handle_endtag(self, tag):
        if tag in self.VOID or not self.stack:
            return
        if self.region_depth == len(self.stack):
            self.region_depth = None
        self.stack.pop()

    def handle_data(self, data):
        if self.region_depth is not None and data.strip():
            self.found.append((" ".join(data.split()), self.stack[-1]))


def _markup(ids=(), classes=()):
    parser = _RegionText(ids, classes)
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parser.found


def test_settings_tabs_and_backup_history_have_no_unwired_text():
    # Text with letters only: the &times; close glyph and the "-" / "0" value slots are not words.
    found = _markup(
        ids={"environmentTab", "systemTab", "backupTab", "viewBackupsModal"}, classes={"settings-tabs"}
    )
    unwired = [text for text, attrs in found if "data-i18n" not in attrs and re.search(r"[A-Za-z]", text)]
    assert unwired == []


def test_markup_keys_match_their_english_text():
    english = _catalog("en.json")
    found = _markup(ids={"settingsModal", "viewBackupsModal"})
    wired = {attrs["data-i18n"]: text for text, attrs in found if attrs.get("data-i18n") in NEW_STRINGS}
    assert set(wired) == HTML_KEYS
    assert {key: english[key] for key in wired} == wired


# --- app.js -----------------------------------------------------------------


def test_app_js_fallbacks_match_the_english_catalog():
    english = _catalog("en.json")
    js = APP_JS.read_text(encoding="utf-8")
    keys = "|".join(re.escape(k) for k in [*NEW_STRINGS, *REUSED_KEYS])
    calls = re.findall(rf"'({keys})', '((?:[^'\\]|\\.)*)'", js)
    fallbacks = {}
    for key, text in calls:
        fallbacks.setdefault(key, set()).add(text.replace("\\'", "'"))
    # The Backup History loading text is in index.html and is written again by openViewBackupsModal().
    assert set(fallbacks) == (NEW_STRINGS.keys() - HTML_KEYS) | REUSED_KEYS | {"modal.backups.loading"}
    assert {k: v for k, v in fallbacks.items() if v != {english[k]}} == {}


def test_the_methods_show_no_untranslated_literals():
    js = APP_JS.read_text(encoding="utf-8")
    # A quoted literal handed to the DOM or a toast, or a text node inside an HTML template.
    literal = re.compile(r"(?:textContent\s*=|showToast\(|setValueText\([^,]+,)\s*['\"`][^'\"`$]*[A-Za-z]{2}")
    text_node = re.compile(r">([^<>${}'\"`]*[A-Za-z]{2}[^<>${}'\"`]*)<")
    found = {name: literal.findall(_method(js, name) or "") + text_node.findall(_method(js, name) or "")
             for name in METHODS}
    assert [name for name in METHODS if not _method(js, name)] == []
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_no_caller_strips_the_english_ago():
    # formatTimeDelta() is translated now, so " ago" cannot be cut off its result; a
    # caller that wants the bare duration (the sync cadence) calls formatDuration().
    js = APP_JS.read_text(encoding="utf-8")
    assert re.findall(r"formatTimeDelta\([^;]*?\.replace\(", js) == []
    assert "+ ' ago'" not in js
    assert "const cadence = this.formatDuration(delta.interval_seconds);" in js


def test_system_info_formatters_have_no_literals():
    js = APP_JS.read_text(encoding="utf-8")
    config = re.search(r"^    static SYSTEM_INFO_CONFIG = \{.*?^    \};\n", js, re.S | re.M).group(0)
    formatters = re.findall(r"formatter: .*?(?=\n        \})", config, re.S)
    assert len(formatters) == 9
    # Once the app.t('key', 'fallback') calls are taken out, no quoted text may remain.
    without_lookups = [re.sub(r"app\.t\('[^']+', '[^']*'\)", "", f) for f in formatters]
    assert [f for f in without_lookups if re.search(r"'[^']*[A-Za-z]{2}[^']*'", f)] == []


# --- behavior in node -------------------------------------------------------

HARNESS = r"""
const de = __DE__, en = __EN__, markup = __MARKUP__;
class El {
  constructor(id, text = '', i18n = null) {
    this.id = id; this.textContent = text; this.innerHTML = ''; this.dataset = {}; this.style = {};
    this.children = []; this.className = '';
    if (i18n) this.dataset.i18n = i18n;
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value ?? ''); }  // the DOM stores text as a string
  getAttribute() { return null; }
  setAttribute() {}
  appendChild(child) { this.children.push(child); return child; }
}
let els, toasts, clipboard, apiLog;
globalThis.document = {
  getElementById: id => els[id] || null,
  createElement: () => new El(''),
  querySelectorAll: () => Object.values(els).filter(e => e.dataset.i18n),
  title: '',
};
Object.defineProperty(globalThis, 'navigator', {
  configurable: true, writable: true,
  value: { clipboard: { writeText: async text => { if (clipboard.fail) throw new Error('denied'); clipboard.text = text; } } },
});
globalThis.MemoryDashboard = { SYSTEM_INFO_CONFIG: __CONFIG__ };

function freshDom() {
  els = {};
  for (const [id, [text, key]] of Object.entries(markup)) els[id] = new El(id, text, key);
  for (const id of ['backupsList', 'backupsSummary', 'viewBackupsModal', 'envConfigContainer',
                    'oauthBackendRow', 'oauthClientsRow', 'oauthTokensRow']) els[id] = new El(id);
  toasts = []; clipboard = { fail: false, text: null }; apiLog = [];
}

function makeApp(respond) {
  const app = {
    translations: de, fallbackTranslations: en, liveSearchEnabled: false,
    updateLanguageSwitcher() {},
    openModal() {},
    showToast(message, type) { toasts.push([message, type]); },
    async apiCall(path) { apiLog.push(path); return respond(path); },
    __METHODS__
  };
  return app;
}
const field = id => ({ text: els[id].textContent, key: els[id].dataset.i18n || null, loading: els[id].dataset.loading || null });
const switchTo = async (app, catalog) => { app.translations = catalog; await app.applyTranslations(); };
const SYSTEM_IDS = Object.keys(MemoryDashboard.SYSTEM_INFO_CONFIG);
const systemFields = () => Object.fromEntries(SYSTEM_IDS.map(id => [id, field(id)]));

const scenarios = {
  async durations() {
    const app = makeApp(() => ({}));
    const run = () => ({
      seconds: app.formatDuration(45), minutes: app.formatDuration(300), hours: app.formatDuration(7200),
      days: app.formatDuration(172800), ago: app.formatTimeDelta(300),
      uptime: app.formatUptime(93784), uptimeShort: app.formatUptime(30),
    });
    const german = run();
    app.translations = en;
    return { german, english: run() };
  },
  async backupStatusRecent() {
    const app = makeApp(() => ({ enabled: true, time_since_last_seconds: 300.7, backup_count: 4,
                                 total_size_bytes: 3 * 1024 * 1024, next_backup_at: null, scheduler_running: true }));
    await app.checkBackupStatus();
    const before = { last: field('settingsLastBackup'), next: field('settingsNextBackup'), count: field('settingsBackupCount') };
    await switchTo(app, en);
    return { before, afterSwitch: { last: field('settingsLastBackup'), next: field('settingsNextBackup') } };
  },
  async backupStatusNever() {
    const app = makeApp(() => ({ enabled: true, time_since_last_seconds: null, backup_count: 0,
                                 total_size_bytes: 0, next_backup_at: null, scheduler_running: false }));
    await app.checkBackupStatus();
    const before = { last: field('settingsLastBackup'), next: field('settingsNextBackup') };
    await switchTo(app, en);
    return { before, afterSwitch: { last: field('settingsLastBackup'), next: field('settingsNextBackup') } };
  },
  async backupStatusDisabled() {
    const app = makeApp(() => ({ enabled: false }));
    await app.checkBackupStatus();
    const before = { last: field('settingsLastBackup'), next: field('settingsNextBackup') };
    await switchTo(app, en);
    return { before, afterSwitch: { last: field('settingsLastBackup') } };
  },
  async backupStatusDateAfterScheduled() {
    // A field that showed "Scheduled" (with its key) and later shows a date must drop the key.
    let status = { enabled: true, time_since_last_seconds: 60, backup_count: 1, total_size_bytes: 0,
                   next_backup_at: null, scheduler_running: true };
    const app = makeApp(() => status);
    await app.checkBackupStatus();
    status = { ...status, next_backup_at: '2026-11-04T03:00:00Z' };
    await app.checkBackupStatus();
    const before = field('settingsNextBackup');
    await switchTo(app, en);
    return { before, afterSwitch: field('settingsNextBackup'), date: new Date('2026-11-04T03:00:00Z').toLocaleString() };
  },
  async backupsModal() {
    const app = makeApp(() => new Promise(() => {}));
    app.loadBackupsList = async () => {};
    await app.openViewBackupsModal();
    return { list: els.backupsList.innerHTML };
  },
  async backupsListMany() {
    const app = makeApp(path => path === '/backup/list'
      ? { backups: [
            { filename: 'a.db', size_bytes: 1048576, created_at: '2026-11-02T10:00:00Z', age_days: 1 },
            { filename: 'b.db', size_bytes: 524288, created_at: null, age_days: 3 },
            { filename: 'c.db', size_bytes: 0, created_at: null, age_days: null } ],
          total_count: 3, total_size_bytes: 1572864 }
      : { backup_directory: '/data/<backups>' });
    await app.loadBackupsList();
    return { summary: els.backupsSummary.innerHTML, list: els.backupsList.innerHTML };
  },
  async backupsListOne() {
    const app = makeApp(path => path === '/backup/list'
      ? { backups: [{ filename: 'a.db', size_bytes: 1048576, created_at: null, age_days: 0 }], total_count: 1,
          total_size_bytes: 1048576 }
      : null);
    await app.loadBackupsList();
    return { summary: els.backupsSummary.innerHTML };
  },
  async backupsListEmpty() {
    const app = makeApp(path => path === '/backup/list' ? { backups: [], total_count: 0, total_size_bytes: 0 } : null);
    await app.loadBackupsList();
    return { list: els.backupsList.innerHTML };
  },
  async backupsListFails() {
    const app = makeApp(() => Promise.reject(new Error('HTTP 500')));
    await app.loadBackupsList();
    return { list: els.backupsList.innerHTML };
  },
  async environment() {
    const app = makeApp(() => ({ categories: [{ name: 'Storage', description: 'Where memories live', parameters: [
      { key: 'API_KEY', value: 'sk-***', sensitive: true, description: 'Secret' },
      { key: 'EMPTY_ONE', value: '' },
      { key: 'FLAG_ON', value: 'true', type: 'boolean' },
      { key: 'FLAG_OFF', value: 'false', type: 'boolean' },
      { key: 'BACKEND', value: 'hybrid', choices: ['sqlite_vec', 'hybrid', 'a<b'] },
    ] }] }));
    await app.loadEnvironmentConfig();
    return { html: els.envConfigContainer.children.map(c => c.innerHTML).join('\n') };
  },
  async environmentFails() {
    const app = makeApp(() => Promise.reject(new Error('HTTP 403')));
    await app.loadEnvironmentConfig();
    return { html: els.envConfigContainer.innerHTML };
  },
  async copyConfig() {
    const app = makeApp(() => ({}));
    await app.copyEnvironmentConfig();
    app.currentEnvConfig = { categories: [{ name: 'Storage', description: '', parameters: [{ key: 'A', value: '1' }] }] };
    await app.copyEnvironmentConfig();
    const copied = clipboard.text;
    clipboard.fail = true;
    await app.copyEnvironmentConfig();
    return { toasts, copied };
  },
  async systemInfo() {
    const app = makeApp(path => {
      if (path === '/health/detailed') return { version: '12.5.0', uptime_seconds: 93784,
        storage: { storage_backend: 'sqlite_vec', database_size_mb: 1.5, total_memories: 1234 } };
      if (path === '/oauth/status') return { oauth_enabled: false };
      return {};
    });
    app.resetSystemInfoLoadingState();
    const loading = field('settingsVersion');
    await app.loadSystemInfo();
    const loaded = systemFields();
    await switchTo(app, en);
    return { loading, loaded, afterSwitch: systemFields() };
  },
  async systemInfoOAuth() {
    const app = makeApp(path => path === '/oauth/status'
      ? { oauth_enabled: true, storage_backend: null, registered_clients_count: 2, active_tokens_count: null }
      : Promise.reject(new Error('HTTP 503')));
    app.resetSystemInfoLoadingState();
    await app.loadSystemInfo();
    return { fields: systemFields(), backend: els.settingsOAuthBackend.textContent,
             clients: els.settingsOAuthClients.textContent, tokens: els.settingsOAuthTokens.textContent };
  },
  async systemInfoError() {
    // A formatter that throws (a size that is not a number) stops the update half way.
    const app = makeApp(path => path === '/health/detailed'
      ? { version: '12.5.0', storage: { storage_backend: 'sqlite_vec', database_size_mb: 'n/a' } }
      : {});
    app.resetSystemInfoLoadingState();
    await app.loadSystemInfo();
    return systemFields();
  },
};
(async () => {
  const out = {};
  for (const [name, run] of Object.entries(scenarios)) {
    freshDom();
    try { out[name] = await run(); } catch (e) { out[name] = { error: String((e && e.stack) || e) }; }
  }
  console.log(JSON.stringify(out));
})();
"""

MARKUP_IDS = [
    "settingsLastBackup", "settingsBackupCount", "settingsNextBackup", "settingsOAuthBackend",
    "settingsOAuthClients", "settingsOAuthTokens", "settingsVersion", "settingsBackend", "settingsPrimaryBackend",
    "settingsEmbeddingModel", "settingsEmbeddingDim", "settingsDbSize", "settingsTotalMemories", "settingsUptime",
    "settingsOAuthEnabled",
]


def _markup_of(element_id):
    text, key = None, None

    class Finder(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.inside = False

        def handle_starttag(self, tag, attrs):
            nonlocal key
            attrs = dict(attrs)
            if attrs.get("id") == element_id:
                self.inside, key = True, attrs.get("data-i18n")

        def handle_endtag(self, tag):
            self.inside = False

        def handle_data(self, data):
            nonlocal text
            if self.inside and text is None:
                text = data.strip()

    Finder().feed(INDEX_HTML.read_text(encoding="utf-8"))
    return [text, key]


@pytest.fixture(scope="module")
def run():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = APP_JS.read_text(encoding="utf-8")
    helpers = ["t", "replacePlaceholders", "setTranslatedText", "setValueText", "staticTextHtml",
               "applyTranslations", "escapeHtml", "getNestedValue"]
    methods = ",\n".join(m for m in (_method(js, n) for n in [*helpers, *METHODS]) if m)
    config = re.search(r"^    static SYSTEM_INFO_CONFIG = (\{.*?^    \});\n", js, re.S | re.M).group(1)
    script = (
        HARNESS.replace("__DE__", json.dumps(_catalog("de.json")))
        .replace("__EN__", json.dumps(_catalog("en.json")))
        .replace("__MARKUP__", json.dumps({i: _markup_of(i) for i in MARKUP_IDS}))
        .replace("__CONFIG__", config)
        .replace("__METHODS__", methods)
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60, check=True)
    results = json.loads(result.stdout)

    def scenario(name):
        assert "error" not in results[name], results[name]["error"]
        return results[name]

    return scenario


DE = _catalog("de.json")
EN = _catalog("en.json")


def de(key, **values):
    text = DE[key]
    for name, value in values.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def test_german_differs_from_english_for_every_key_the_behavior_tests_read():
    # Otherwise the behavior tests below could pass on untranslated English.
    same_in_german = {"modal.settings.tabs.backup", "modal.settings.tabs.server"}
    assert [k for k in NEW_STRINGS.keys() - same_in_german if DE[k] == NEW_STRINGS[k]] == []
    assert [k for k in REUSED_KEYS if DE[k] == EN[k]] == []


def test_durations_and_uptime_are_translated_and_english_is_unchanged(run):
    r = run("durations")
    assert r["german"] == {
        "seconds": de("time.seconds", count=45), "minutes": de("time.minutes", count=5),
        "hours": de("time.hours", count=2), "days": de("time.days", count=2),
        "ago": de("time.ago", time=de("time.minutes", count=5)),
        "uptime": " ".join([de("time.days", count=1), de("time.hours", count=2), de("time.minutes", count=3)]),
        "uptimeShort": de("time.underAMinute"),
    }
    # The English output is what the dashboard showed before; the sync cadence used to strip " ago".
    assert r["english"] == {"seconds": "45s", "minutes": "5m", "hours": "2h", "days": "2d", "ago": "5m ago",
                            "uptime": "1d 2h 3m", "uptimeShort": "< 1m"}


def test_last_backup_says_ago_once_and_keeps_its_value_after_a_switch(run):
    r = run("backupStatusRecent")
    ago = de("time.ago", time=de("time.minutes", count=5))
    # index.html gives the field data-i18n="modal.settings.never"; a switch must not put "Never" back.
    assert r["before"]["last"] == {"text": ago, "key": None, "loading": None}
    assert r["afterSwitch"]["last"]["text"] == ago
    assert r["before"]["next"] == {"text": de("modal.settings.backup.scheduled"),
                                   "key": "modal.settings.backup.scheduled", "loading": None}
    assert r["afterSwitch"]["next"]["text"] == "Scheduled"
    assert r["before"]["count"]["text"] == "4 (3.0 MB)"


def test_never_and_not_scheduled_follow_a_language_switch(run):
    r = run("backupStatusNever")
    assert r["before"]["last"]["text"] == de("modal.settings.never")
    assert r["before"]["next"]["text"] == de("modal.settings.backup.notScheduled")
    assert r["afterSwitch"]["last"]["text"] == "Never"
    assert r["afterSwitch"]["next"]["text"] == "Not scheduled"


def test_disabled_backups(run):
    r = run("backupStatusDisabled")
    assert r["before"]["last"]["text"] == de("modal.settings.backup.disabled")
    assert r["before"]["next"]["text"] == "-"
    assert r["afterSwitch"]["last"]["text"] == "Backups disabled"


def test_a_scheduled_date_replaces_the_scheduled_key(run):
    r = run("backupStatusDateAfterScheduled")
    assert r["before"] == {"text": r["date"], "key": None, "loading": None}
    assert r["afterSwitch"]["text"] == r["date"]


def test_backup_history_loading_text(run):
    assert run("backupsModal")["list"] == (
        f'<p class="loading-text" data-i18n="modal.backups.loading">{de("modal.backups.loading")}</p>'
    )


def test_backup_history_summary_ages_and_unknown_date(run):
    r = run("backupsListMany")
    sentence = de("modal.backups.summaryMany").replace("{count}", "<strong>3</strong>")
    assert r["summary"] == (
        sentence.replace("{size}", "<strong>1.50 MB</strong>")
        + '<br><span class="backup-directory-path">/data/&lt;backups&gt;</span>'
    )
    assert de("modal.backups.ageOne") in r["list"]
    assert de("modal.backups.ageDays", days=3) in r["list"]
    assert de("modal.backups.ageDays", days="?") in r["list"]
    assert r["list"].count(de("modal.backups.unknownDate")) == 2
    assert " ago" not in r["list"] and "Unknown" not in r["list"]


def test_backup_history_summary_for_one_backup(run):
    sentence = de("modal.backups.summaryOne").replace("{count}", "<strong>1</strong>")
    assert run("backupsListOne")["summary"] == sentence.replace("{size}", "<strong>1.00 MB</strong>")


def test_backup_history_empty_state_names_the_translated_button(run):
    empty = de("modal.backups.empty", button=DE["modal.settings.backupNow"])
    escaped = empty.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    assert run("backupsListEmpty")["list"] == f'<p class="loading-text">{escaped}</p>'


def test_backup_history_load_failure(run):
    assert run("backupsListFails")["list"] == (
        '<p class="loading-text" style="color: var(--error);" data-i18n="modal.backups.loadFailed">'
        f'{de("modal.backups.loadFailed")}</p>'
    )


def test_environment_labels_are_translated_and_keep_their_keys(run):
    html = run("environment")["html"]
    for key, css in [("modal.settings.env.sensitive", "sensitive-badge"), ("modal.settings.env.notSet", "env-value-empty"),
                     ("modal.settings.env.enabled", "env-value-boolean true"),
                     ("modal.settings.env.disabled", "env-value-boolean false")]:
        assert f'<span class="{css}" data-i18n="{key}">{de(key)}</span>' in html
    options = de("modal.settings.env.options", options="sqlite_vec, hybrid, a&lt;b")
    assert f'<span class="env-choices">{options}</span>' in html
    assert not re.search(r"Sensitive|not set|Enabled|Disabled|options:", html)


def test_environment_load_failure(run):
    assert run("environmentFails")["html"] == (
        f'<p class="error" data-i18n="modal.settings.env.loadFailed">{de("modal.settings.env.loadFailed")}</p>'
    )


def test_copy_configuration_toasts_and_untouched_clipboard_text(run):
    r = run("copyConfig")
    assert r["toasts"] == [
        [de("modal.settings.env.noConfig"), "error"],
        [de("modal.settings.env.copied"), "success"],
        [DE["toast.copyFailed"], "error"],
    ]
    # The clipboard text is a .env file and stays English.
    assert r["copied"].startswith("# MCP Memory Service Configuration\n# Exported: ")
    assert "\nA=1\n" in r["copied"]


def test_system_info_values_survive_a_language_switch(run):
    r = run("systemInfo")
    assert r["loading"] == {"text": DE["status.loading"], "key": "status.loading", "loading": "true"}
    texts = {k: v["text"] for k, v in r["loaded"].items()}
    na = de("modal.settings.system.notAvailable")
    uptime = " ".join([de("time.days", count=1), de("time.hours", count=2), de("time.minutes", count=3)])
    # toLocaleString() follows node's locale, so the count is checked by its digits.
    assert re.sub(r"\D", "", texts.pop("settingsTotalMemories")) == "1234"
    assert texts == {
        "settingsVersion": "12.5.0", "settingsBackend": "sqlite_vec", "settingsPrimaryBackend": na,
        "settingsEmbeddingModel": na, "settingsEmbeddingDim": na, "settingsDbSize": "1.50 MB",
        "settingsUptime": uptime, "settingsOAuthEnabled": de("modal.settings.system.disabled"),
    }
    assert {k: (v["key"], v["loading"]) for k, v in r["loaded"].items() if v["key"] or v["loading"]} == {}
    # index.html gives every field data-i18n="status.loading"; a switch must not put "Loading..." back.
    assert {k: v["text"] for k, v in r["afterSwitch"].items()} == {k: v["text"] for k, v in r["loaded"].items()}


def test_system_info_oauth_details(run):
    r = run("systemInfoOAuth")
    na = de("modal.settings.system.notAvailable")
    assert r["fields"]["settingsOAuthEnabled"]["text"] == de("modal.settings.system.enabled")
    assert (r["backend"], r["clients"], r["tokens"]) == (na, "2", na)
    assert r["fields"]["settingsVersion"]["text"] == na


def test_system_info_error_marks_only_the_fields_still_loading(run):
    r = run("systemInfoError")
    texts = {k: v["text"] for k, v in r.items()}
    error = de("modal.settings.system.error")
    assert texts["settingsVersion"] == "12.5.0"
    assert texts["settingsBackend"] == "sqlite_vec"
    assert [k for k in ("settingsDbSize", "settingsTotalMemories", "settingsUptime", "settingsOAuthEnabled")
            if texts[k] != error] == []
    assert r["settingsDbSize"]["key"] == "modal.settings.system.error"
    assert [k for k, v in r.items() if v["loading"]] == []
