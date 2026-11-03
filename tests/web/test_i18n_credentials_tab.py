"""The Settings Credentials tab is translated in every dashboard catalog (#1813).

Reads the real catalogs, ``index.html`` and ``app.js``; nothing is mocked.
"""

import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src/mcp_memory_service/web/static"
I18N_DIR = STATIC / "i18n"
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")

PREFIX = "settings.credentials."
TOKEN_RE = re.compile(r"\{[A-Za-z0-9_]+\}")
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Keys wired in index.html through a data-i18n* attribute.
MARKUP_KEYS = [
    "title",
    "description",
    "apiToken",
    "apiTokenPlaceholder",
    "showHide",
    "apiTokenHelp",
    "accountId",
    "accountIdPlaceholder",
    "d1Id",
    "vectorizeName",
    "vectorizePlaceholder",
    "syncTitle",
    "syncOwner",
    "syncOwnerHttp",
    "syncOwnerBoth",
    "syncOwnerMcp",
    "syncHelpHttp",
    "syncHelpBoth",
    "syncHelpMcp",
    "testButton",
    "hint",
]

# Keys looked up from app.js through this.t(...).
SCRIPT_KEYS = [
    "required",
    "testing",
    "testButton",
    "testFirst",
    "tokenActive",
    "tokenActiveExpires",
    "invalidToken",
    "requestFailed",
    "saved",
]

# Existing keys the tab now shares with the Quality tab.
SHARED_KEYS = [
    "settings.quality.save",
    "settings.quality.saving",
    "settings.quality.saveFailed",
]

ALL_KEYS = sorted(
    {PREFIX + key for key in MARKUP_KEYS + SCRIPT_KEYS}
)


def _catalog(path):
    # Plain json.loads: some older catalogs repeat analytics.memoryTypes, which a
    # strict duplicate check would reject for reasons unrelated to this tab.
    return json.loads(path.read_text(encoding="utf-8"))


def _catalogs():
    return {path.name: _catalog(path) for path in sorted(I18N_DIR.glob("*.json"))}


def test_credentials_keys_exist_with_text_in_every_catalog():
    missing = {}
    for name, catalog in _catalogs().items():
        bad = [
            key
            for key in ALL_KEYS + SHARED_KEYS
            if not isinstance(catalog.get(key), str) or not catalog[key].strip()
        ]
        if bad:
            missing[name] = bad

    assert not missing, f"Missing or empty credentials keys: {missing}"


def test_credentials_translations_keep_tokens_and_markup():
    english = _catalog(I18N_DIR / "en.json")
    mismatched = {}
    for name, catalog in _catalogs().items():
        for key in ALL_KEYS:
            if sorted(TOKEN_RE.findall(english[key])) != sorted(
                TOKEN_RE.findall(catalog[key])
            ) or TAG_RE.findall(english[key]) != TAG_RE.findall(catalog[key]):
                mismatched.setdefault(name, []).append(key)

    assert not mismatched, f"Tokens or HTML differ from en.json: {mismatched}"


def test_non_english_catalogs_translate_the_tab():
    english = _catalog(I18N_DIR / "en.json")
    untranslated = {
        name: [
            key
            for key in ("title", "testButton", "hint", "saved")
            if catalog[PREFIX + key] == english[PREFIX + key]
        ]
        for name, catalog in _catalogs().items()
        if name != "en.json"
    }
    untranslated = {name: keys for name, keys in untranslated.items() if keys}

    assert not untranslated, f"Still English: {untranslated}"


def test_markup_sites_reference_their_keys():
    tab = INDEX_HTML[
        INDEX_HTML.index('id="credentialsTab"') : INDEX_HTML.index('id="qualityTab"')
    ]
    unwired = [
        key
        for key in MARKUP_KEYS
        if not re.search(
            rf'data-i18n(?:-html|-title|-placeholder)?="{re.escape(PREFIX + key)}"',
            tab,
        )
    ]

    assert not unwired, f"Not wired in index.html: {unwired}"
    assert re.search(r'id="credSaveBtn"[^>]*data-i18n="settings\.quality\.save"', tab)


def test_script_sites_reference_their_keys():
    start = APP_JS.index("async testCloudflareCredentials()")
    end = APP_JS.index("async loadQualitySettings()")
    block = APP_JS[start:end]

    unwired = [key for key in SCRIPT_KEYS if f"'{PREFIX + key}'" not in block]
    assert not unwired, f"Not looked up in app.js: {unwired}"

    for key in SHARED_KEYS:
        assert f"'{key}'" in block, f"{key} not used by the credentials code"

    # No button label or status message is assigned as a bare English literal.
    assert not re.search(r"textContent = ['\"]", block)
    assert not re.search(r"_showCredResult\([^,]+,\s*['\"]", block)
    assert "Request failed" not in re.sub(r"this\.t\([^)]*\)", "", block)
