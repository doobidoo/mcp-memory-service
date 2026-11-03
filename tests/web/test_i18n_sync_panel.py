"""The header controls, delta-sync panel and connection status are translated
in every dashboard catalog (#1816).

Reads the real catalogs, ``index.html`` and ``app.js``; nothing is mocked.
"""

import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src/mcp_memory_service/web/static"
I18N_DIR = STATIC / "i18n"
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")

TOKEN_RE = re.compile(r"\{[A-Za-z0-9_]+\}")
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Keys wired in index.html through a data-i18n* attribute.
MARKUP_KEYS = [
    "header.searchLabel",
    "header.addMemoryTitle",
    "header.toggleDarkMode",
    "header.settingsTitle",
    "header.languageLabel",
    "sync.deltaSync",
    "sync.checking",
    "sync.pauseTitle",
    "sync.resumeTitle",
    "sync.nowTitle",
]

# Keys looked up from app.js through this.t(...).
SCRIPT_KEYS = [
    "sync.status.paused",
    "sync.status.syncing",
    "sync.status.pending",
    "sync.status.error",
    "sync.status.synced",
    "sync.statusUnavailable",
    "sync.pendingOps",
    "sync.opsCount",
    "sync.failedCount",
    "sync.scheduleEveryNext",
    "sync.notScheduled",
    "sync.automaticDisabled",
    "sync.every",
    "sync.nextSyncInHm",
    "sync.nextSyncInMs",
    "sync.nextSyncInS",
    "sync.schedulePaused",
    "sync.scheduledDue",
    "sync.noPeers",
    "sync.urlNotConfigured",
    "sync.pull",
    "sync.push",
    "sync.pullReceived",
    "sync.pushSent",
    "sync.noCycleYet",
    "sync.lastCycle",
    "sync.waitingFirstCycle",
    "sync.details",
    "sync.pullCursor",
    "sync.pushCursor",
    "sync.backlog",
    "sync.notAvailable",
    "sync.unknown",
    "status.connecting",
    "status.unknown",
]

# Existing keys the panel now shares with the connection indicator.
SHARED_KEYS = [
    "status.connected",
    "status.disconnected",
]

ALL_KEYS = sorted(set(MARKUP_KEYS + SCRIPT_KEYS))

# Keys covered by the catalog checks above but referenced from init(),
# outside the blocks listed below.
EXTRA_SCRIPT_KEYS = ["toast.connectionInsecure"]

CATALOG_KEYS = ALL_KEYS + EXTRA_SCRIPT_KEYS

# updateConnectionStatus() looks these up; status.connected/disconnected
# predate this change, status.connecting/status.unknown are new.
CONNECTION_KEYS = [
    "status.connected",
    "status.connecting",
    "status.disconnected",
    "status.unknown",
]

# The status word shown in the delta-sync badge, looked up through a
# `sync.status.${state}` template literal.
STATE_KEYS = [
    "sync.status.disabled",
    "sync.status.syncing",
    "sync.status.synced",
]

# app.js regions this slice touches, as (start anchor, end anchor) pairs.
BLOCKS = [
    ("handleSyncCompleted(data)", "// Show completion notification"),
    ("async checkSyncStatus()", "async checkBackupStatus()"),
    ("async forceSync()", "const result = await this.apiCall('/sync/force', 'POST');"),
    ("updateConnectionStatus(status)", "async apiCall(endpoint"),
]


def _blocks():
    return [APP_JS[APP_JS.index(start):APP_JS.index(end)] for start, end in BLOCKS]


def _catalog(path):
    # Plain json.loads: some older catalogs repeat analytics.memoryTypes, which a
    # strict duplicate check would reject for reasons unrelated to this panel.
    return json.loads(path.read_text(encoding="utf-8"))


def _catalogs():
    return {path.name: _catalog(path) for path in sorted(I18N_DIR.glob("*.json"))}


def test_sync_keys_exist_with_text_in_every_catalog():
    missing = {}
    for name, catalog in _catalogs().items():
        bad = [
            key
            for key in CATALOG_KEYS + SHARED_KEYS + STATE_KEYS
            if not isinstance(catalog.get(key), str) or not catalog[key].strip()
        ]
        if bad:
            missing[name] = bad

    assert not missing, f"Missing or empty sync keys: {missing}"


def test_sync_translations_keep_tokens_and_markup():
    english = _catalog(I18N_DIR / "en.json")
    mismatched = {}
    for name, catalog in _catalogs().items():
        for key in CATALOG_KEYS:
            if sorted(TOKEN_RE.findall(english[key])) != sorted(
                TOKEN_RE.findall(catalog[key])
            ) or TAG_RE.findall(english[key]) != TAG_RE.findall(catalog[key]):
                mismatched.setdefault(name, []).append(key)

    assert not mismatched, f"Tokens or HTML differ from en.json: {mismatched}"


def test_non_english_catalogs_translate_the_panel():
    english = _catalog(I18N_DIR / "en.json")
    sampled = ("header.toggleDarkMode", "sync.checking", "sync.noPeers",
               "sync.pullReceived", "sync.details", "toast.connectionInsecure")
    untranslated = {
        name: [key for key in sampled if catalog[key] == english[key]]
        for name, catalog in _catalogs().items()
        if name != "en.json"
    }
    untranslated = {name: keys for name, keys in untranslated.items() if keys}

    assert not untranslated, f"Still English: {untranslated}"


def test_markup_sites_reference_their_keys():
    # The header buttons and the two sync controls sit in separate regions of
    # index.html; every key is unique, so match the whole file.
    unwired = [
        key
        for key in MARKUP_KEYS
        if not re.search(
            rf'data-i18n(?:-html|-title|-placeholder|-aria)?="{re.escape(key)}"',
            INDEX_HTML,
        )
    ]

    assert not unwired, f"Not wired in index.html: {unwired}"
    # The visible "Delta sync" heading and the section's aria-label share a key.
    assert INDEX_HTML.count('data-i18n-aria="sync.deltaSync"') == 1
    assert INDEX_HTML.count('data-i18n="sync.deltaSync"') == 1


def test_script_sites_reference_their_keys():
    blocks = _blocks()
    unwired = [
        key
        for key in SCRIPT_KEYS
        if not any(f"'{key}'" in block for block in blocks)
    ]
    assert not unwired, f"Not looked up in app.js: {unwired}"

    for key in SHARED_KEYS:
        assert any(f"'{key}'" in block for block in blocks), (
            f"{key} not used by the connection indicator"
        )

    # The insecure-connection toast fires from init(), outside those blocks.
    assert "'toast.connectionInsecure'" in APP_JS

    # A language switch re-renders the live status word instead of resetting
    # it to the widget's "Checking..." default.
    apply_block = APP_JS[
        APP_JS.index('async applyTranslations()'):APP_JS.index('updateLanguageSwitcher()')
    ]
    assert "`sync.status.${syncState}`" in apply_block

    # The badge's status word is resolved through a template literal.
    sync_block = blocks[1]
    assert "`sync.status.${state}`" in sync_block

    # No status word or panel label is assigned as a bare English literal.
    stripped = re.sub(r"this\.t\([^)]*\)", "", sync_block)
    for literal in (
        "'Paused'", "'Syncing'", "'Pending'", "'Error'", "'Synced'",
        "'Status unavailable'", "'No peers configured'", "Sync details",
        "'Pull cursor'", "'Push cursor'", "'Backlog'",
    ):
        assert literal not in stripped, f"Bare literal remains: {literal}"
