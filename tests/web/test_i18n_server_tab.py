"""The Settings Server tab and the restart overlay are translated (#1814).

The static tests read the real catalogs, index.html and app.js. The behavior
tests extract the real server-management methods from app.js and run them in
node against the real German catalog, with stubs only for the DOM, the API and
the browser dialogs (the same approach as test_i18n_rerender_on_language_switch).
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

PREFIX = "modal.settings.server."
# Every string the Server tab and the restart overlay show, as en.json has it.
SERVER_STRINGS = {
    "title": "Server Management",
    "pid": "Server PID:",
    "platform": "Platform:",
    "updateAvailable": "Update Available:",
    "notChecked": "Not checked yet",
    "checkUpdates": "Check for Updates",
    "updateRestart": "Update & Restart",
    "restart": "Restart Server",
    "processing": "Processing...",
    "checking": "Checking for updates...",
    "commitsAvailable": "Yes - {count} commit(s) available",
    "updatesAvailable": "{count} update(s) available",
    "upToDate": "Up to date",
    "alreadyUpToDate": "Already up to date.",
    "checkFailed": "Check failed",
    "checkFailedWithError": "Failed to check for updates: {error}",
    "confirmUpdate": "This will pull the latest code, install dependencies, and restart the server. Continue?",
    "pulling": "Pulling updates and installing...",
    "updateComplete": "Update complete. Server restarting...",
    "updateBlocked": "Update blocked: {error}",
    "confirmForcePull": (
        "The server has uncommitted local changes that will block the pull.\n\n{error}\n\n"
        "Force-pull anyway? (Local changes may be overwritten.)"
    ),
    "updateFailed": "Update failed: {error}",
    "confirmRestart": "Are you sure you want to restart the server? This will temporarily disconnect all clients.",
    "restartFailed": "Restart failed: {error}",
    "restarting": "Server Restarting...",
    "waiting": "Waiting for server to come back online...",
    "reconnectingIn": "Reconnecting in {seconds}s",
    "attempt": "Attempt {attempt}/{max} - Waiting for server...",
    "notBackOnline": "Server did not come back online. Please check manually.",
    "backOnline": "Server is back online (v{version}, pid {pid})! Reloading...",
    "didNotRestart": (
        "Process did not restart (still pid {pid}, v{version}). "
        "The update may have failed silently — check server logs."
    ),
    "backOnlineReloading": "Server is back online, reloading",
}
# The labels index.html carries as data-i18n; app.js renders all the others.
HTML_KEYS = {"title", "pid", "platform", "updateAvailable", "notChecked", "checkUpdates",
             "updateRestart", "restart", "processing", "restarting", "waiting"}
SERVER_METHODS = ["checkForUpdates", "updateAndRestart", "restartServer", "showRestartOverlay", "pollServerHealth"]
TOKEN_RE = re.compile(r"\{[A-Za-z0-9_]+\}")


def _catalog(name):
    # Plain json.loads: some older catalogs carry a duplicate key that is not ours to fix here.
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


CATALOGS = sorted(path.name for path in I18N_DIR.glob("*.json"))


def _method(js, name):
    match = re.search(rf"^    (?:async )?{name}\([^)]*\) \{{\n.*?^    \}}\n", js, re.S | re.M)
    return match.group(0).strip() if match else None


# --- catalogs ---------------------------------------------------------------


def test_english_catalog_has_every_server_string():
    english = _catalog("en.json")
    assert {k: english.get(PREFIX + k) for k in SERVER_STRINGS} == SERVER_STRINGS


@pytest.mark.parametrize("name", CATALOGS)
def test_every_catalog_translates_every_server_key(name):
    catalog = _catalog(name)
    missing = [k for k in SERVER_STRINGS if not str(catalog.get(PREFIX + k, "")).strip()]
    assert not missing, f"{name} lacks {missing}"
    tokens = {
        k: (sorted(TOKEN_RE.findall(catalog[PREFIX + k])), sorted(TOKEN_RE.findall(english)))
        for k, english in SERVER_STRINGS.items()
    }
    assert {k: got for k, (got, want) in tokens.items() if got != want} == {}


# --- index.html -------------------------------------------------------------


class _RegionText(HTMLParser):
    """Collect (text, innermost element attrs) inside the elements with the given ids."""

    def __init__(self, ids):
        super().__init__(convert_charrefs=True)
        self.ids, self.stack, self.region_depth, self.found = ids, [], None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"br", "img", "input", "meta", "link", "hr"}:
            return
        self.stack.append(attrs)
        if self.region_depth is None and attrs.get("id") in self.ids:
            self.region_depth = len(self.stack)

    def handle_endtag(self, tag):
        if tag in {"br", "img", "input", "meta", "link", "hr"} or not self.stack:
            return
        if self.region_depth == len(self.stack):
            self.region_depth = None
        self.stack.pop()

    def handle_data(self, data):
        if self.region_depth is not None and data.strip():
            self.found.append((data.strip(), self.stack[-1]))


def _server_markup():
    parser = _RegionText({"serverTab", "restartOverlay"})
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parser.found


# Value slots that JavaScript fills, and the countdown line that
# showRestartOverlay() rebuilds from modal.settings.server.reconnectingIn.
VALUE_SLOTS = {"settingsServerPid", "settingsServerPlatform", "restartCountdown"}


def test_server_tab_and_restart_overlay_have_no_unwired_text():
    unwired = [
        text for text, attrs in _server_markup()
        if "data-i18n" not in attrs
        and attrs.get("id") not in VALUE_SLOTS
        and "restart-countdown" not in attrs.get("class", "").split()
    ]
    assert unwired == []


def test_server_markup_keys_match_their_english_text():
    english = _catalog("en.json")
    wired = {attrs["data-i18n"]: text for text, attrs in _server_markup() if "data-i18n" in attrs}
    assert set(wired) == {PREFIX + k for k in HTML_KEYS}
    assert {key: english.get(key) for key in wired} == wired


# --- app.js -----------------------------------------------------------------


def test_server_methods_show_no_untranslated_literals():
    js = APP_JS.read_text(encoding="utf-8")
    literal = re.compile(r"(?:textContent\s*=|showToast\(|confirm\()\s*['\"`]")
    found = {name: literal.findall(_method(js, name) or "") for name in SERVER_METHODS}
    assert all(_method(js, name) for name in SERVER_METHODS)
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_app_js_fallbacks_match_the_english_catalog():
    js = APP_JS.read_text(encoding="utf-8")
    calls = re.findall(r"'(modal\.settings\.server\.\w+)', '((?:[^'\\]|\\.)*)'", js)
    fallbacks = {key: text.replace("\\n", "\n").replace("\\'", "'") for key, text in calls}
    assert set(fallbacks) == {PREFIX + k for k in SERVER_STRINGS.keys() - HTML_KEYS}
    assert {k: v for k, v in fallbacks.items() if SERVER_STRINGS[k[len(PREFIX):]] != v} == {}


# --- behavior in node -------------------------------------------------------

HARNESS = r"""
const de = __DE__, en = __EN__, markup = __MARKUP__;
class El {
  constructor(id, text = '', i18n = null) {
    this.id = id; this.textContent = text; this.dataset = {}; this.style = {};
    if (i18n) this.dataset.i18n = i18n;
    this.parentElement = null; this.childNodes = null;
  }
  getAttribute() { return null; }
  setAttribute() {}
  replaceChildren(...nodes) {
    this.childNodes = nodes;
    nodes.forEach(n => { if (typeof n !== 'string') n.parentElement = this; });
  }
  get shown() {
    return this.childNodes ? this.childNodes.map(n => typeof n === 'string' ? n : n.textContent).join('') : this.textContent;
  }
}
let els, prompts, answers, toasts, ticks, healthOk;
globalThis.document = {
  getElementById: id => els[id] || null,
  querySelectorAll: () => Object.values(els).filter(e => e.dataset.i18n),
  title: '',
};
globalThis.confirm = text => { prompts.push(text); return answers.shift(); };
globalThis.setInterval = fn => { ticks.push(fn); return ticks.length; };
globalThis.clearInterval = () => {};
globalThis.setTimeout = () => 0;
globalThis.fetch = async () => ({ ok: healthOk });
globalThis.window = { location: { reload() {} } };

function freshDom() {
  els = {};
  for (const id of ['settingsUpdateAvailable', 'serverActionText', 'restartStatus']) {
    const [text, key] = markup[id];
    els[id] = new El(id, text, key);
  }
  for (const id of ['updateServerBtn', 'serverActionStatus', 'restartOverlay']) els[id] = new El(id);
  const line = new El('', 'Reconnecting in 5s');
  els.restartCountdown = new El('restartCountdown', '5');
  els.restartCountdown.parentElement = line;
  els.countdownLine = line;
  prompts = []; answers = []; toasts = []; ticks = []; healthOk = false;
}

function makeApp(respond) {
  const seen = [];
  const app = {
    translations: de, fallbackTranslations: en, liveSearchEnabled: false,
    updateLanguageSwitcher() {},
    showToast(message, type) { toasts.push([message, type]); },
    async apiCall(path) {
      seen.push({ path, action: els.serverActionText.textContent, actionKey: els.serverActionText.dataset.i18n || null,
                  status: els.restartStatus.textContent });
      return respond(path);
    },
    __METHODS__
  };
  return { app, seen };
}
const state = () => ({
  update: els.settingsUpdateAvailable.textContent, updateKey: els.settingsUpdateAvailable.dataset.i18n || null,
  action: els.serverActionText.textContent, status: els.restartStatus.textContent,
  statusKey: els.restartStatus.dataset.i18n || null, toasts, prompts,
});
const ok = v => () => Promise.resolve(v);
const fail = e => () => Promise.reject(e);

const scenarios = {
  async updateAvailable() {
    const { app, seen } = makeApp(ok({ update_available: true, commits_behind: 3 }));
    await app.checkForUpdates();
    const before = state();
    app.translations = en;
    await app.applyTranslations();
    return { seen, before, afterSwitch: state() };
  },
  async upToDate() {
    const { app } = makeApp(ok({ update_available: false }));
    await app.checkForUpdates();
    const before = state();
    app.translations = en;
    await app.applyTranslations();
    return { before, afterSwitch: state() };
  },
  async checkFails() {
    const { app } = makeApp(fail(new Error('timed out')));
    await app.checkForUpdates();
    return state();
  },
  async updateDeclined() {
    const { app, seen } = makeApp(ok({}));
    answers = [false];
    await app.updateAndRestart();
    return { ...state(), calls: seen.length };
  },
  async updateRuns() {
    const { app, seen } = makeApp(ok({ pre_restart_pid: 1, pre_restart_version: '1.0' }));
    answers = [true];
    await app.updateAndRestart();
    const line = els.countdownLine;
    const first = line.shown;
    ticks[0]();
    return { ...state(), seen, first, second: line.shown,
             spanKept: Array.isArray(line.childNodes) && line.childNodes.includes(els.restartCountdown) };
  },
  async updateBlocked() {
    const err = Object.assign(new Error('working tree has local changes'), { status: 409 });
    const { app } = makeApp(fail(err));
    answers = [true, false];
    await app.updateAndRestart();
    return state();
  },
  async updateFails() {
    const { app } = makeApp(fail(new Error('pip install failed')));
    answers = [true];
    await app.updateAndRestart();
    return state();
  },
  async restartDeclined() {
    const { app, seen } = makeApp(ok({}));
    answers = [false];
    await app.restartServer();
    return { ...state(), calls: seen.length };
  },
  async restartFails() {
    const { app } = makeApp(fail(new Error('HTTP 500')));
    answers = [true];
    await app.restartServer();
    return state();
  },
  async pollGivesUp() {
    const { app } = makeApp(ok({}));
    await app.pollServerHealth(els.restartStatus, els.restartOverlay, 30, 30, 1, '1.0');
    return state();
  },
  async pollBackOnline() {
    const { app, seen } = makeApp(ok({ pid: 2, version: '1.1' }));
    await app.pollServerHealth(els.restartStatus, els.restartOverlay, 0, 30, 1, '1.0');
    return { ...state(), seen };
  },
  async pollSameProcess() {
    const { app } = makeApp(ok({ pid: 1, version: '1.0' }));
    await app.pollServerHealth(els.restartStatus, els.restartOverlay, 29, 30, 1, '1.0');
    return state();
  },
  async pollHealthFallback() {
    const { app } = makeApp(fail(new Error('Failed to fetch')));
    healthOk = true;
    await app.pollServerHealth(els.restartStatus, els.restartOverlay, 0, 30, null, null);
    return state();
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
    names = ["t", "replacePlaceholders", "setTranslatedText", "applyTranslations", *SERVER_METHODS]
    methods = ",\n".join(m for m in (_method(js, n) for n in names) if m)
    markup = {i: _markup_of(i) for i in ("settingsUpdateAvailable", "serverActionText", "restartStatus")}
    script = (
        HARNESS.replace("__DE__", json.dumps(_catalog("de.json")))
        .replace("__EN__", json.dumps(_catalog("en.json")))
        .replace("__MARKUP__", json.dumps(markup))
        .replace("__METHODS__", methods)
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60, check=True)
    results = json.loads(result.stdout)

    def scenario(name):
        assert "error" not in results[name], results[name]["error"]
        return results[name]

    return scenario


DE = _catalog("de.json")


def de(key, **values):
    text = DE[PREFIX + key]
    for name, value in values.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def test_german_differs_from_english_for_every_server_key():
    # Otherwise the behavior tests below could pass on untranslated English.
    assert [k for k, v in SERVER_STRINGS.items() if DE[PREFIX + k] == v] == []


def test_update_check_reports_available_commits_in_the_selected_language(run):
    r = run("updateAvailable")
    assert r["seen"][0]["action"] == de("checking")
    assert r["before"]["update"] == de("commitsAvailable", count=3)
    assert r["before"]["toasts"] == [[de("updatesAvailable", count=3), "info"]]


def test_language_switch_does_not_reset_a_reported_update_count(run):
    # index.html gives the field data-i18n="...notChecked"; a switch must not put that text back.
    r = run("updateAvailable")
    assert r["before"]["updateKey"] is None
    assert r["afterSwitch"]["update"] == de("commitsAvailable", count=3)


def test_up_to_date_follows_a_language_switch(run):
    r = run("upToDate")
    assert r["before"]["update"] == de("upToDate")
    assert r["before"]["toasts"] == [[de("alreadyUpToDate"), "success"]]
    assert r["afterSwitch"]["update"] == SERVER_STRINGS["upToDate"]


def test_failed_update_check(run):
    r = run("checkFails")
    assert r["update"] == de("checkFailed")
    assert r["toasts"] == [[de("checkFailedWithError", error="timed out"), "error"]]


def test_update_confirmation_is_translated_and_cancel_sends_nothing(run):
    r = run("updateDeclined")
    assert r["prompts"] == [de("confirmUpdate")]
    assert r["calls"] == 0


def test_update_progress_and_restart_countdown(run):
    r = run("updateRuns")
    assert r["seen"][0]["action"] == de("pulling")
    assert r["action"] == de("updateComplete")
    assert r["first"] == de("reconnectingIn", seconds=5)
    assert r["second"] == de("reconnectingIn", seconds=4)
    assert r["spanKept"] is True


def test_blocked_update_keeps_the_server_message_untranslated(run):
    r = run("updateBlocked")
    message = "working tree has local changes"
    assert r["toasts"] == [[de("updateBlocked", error=message), "error"]]
    assert r["prompts"] == [de("confirmUpdate"), de("confirmForcePull", error=message)]


def test_failed_update(run):
    assert run("updateFails")["toasts"] == [[de("updateFailed", error="pip install failed"), "error"]]


def test_restart_confirmation_is_translated_and_cancel_sends_nothing(run):
    r = run("restartDeclined")
    assert r["prompts"] == [de("confirmRestart")]
    assert r["calls"] == 0


def test_failed_restart(run):
    assert run("restartFails")["toasts"] == [[de("restartFailed", error="HTTP 500"), "error"]]


def test_restart_overlay_messages(run):
    assert run("pollGivesUp")["status"] == de("notBackOnline")
    assert run("pollBackOnline")["seen"][0]["status"] == de("attempt", attempt=1, max=30)
    assert run("pollBackOnline")["status"] == de("backOnline", version="1.1", pid=2)
    assert run("pollSameProcess")["status"] == de("didNotRestart", pid=1, version="1.0")
    assert run("pollHealthFallback")["status"] == de("backOnlineReloading")
