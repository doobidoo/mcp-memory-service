"""Regression coverage for the authentication dialog translations (#1818).

The dialog shown when the dashboard needs an API key or OAuth login was
hardcoded in English, and so were the toasts after a login attempt. These tests
read the real catalogs, the real `index.html` and the real `app.js` -- nothing is
mocked -- so they fail on the base branch and pass with the change.
"""

import json
import re
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static"
I18N_DIR = STATIC_DIR / "i18n"
EN_CATALOG = I18N_DIR / "en.json"
INDEX_HTML = STATIC_DIR / "index.html"
APP_JS = STATIC_DIR / "app.js"

# The nine strings that live in index.html, with the attribute each one needs.
MARKUP_KEYS = {
    "modal.auth.title": "data-i18n",
    "modal.auth.description": "data-i18n",
    "modal.auth.apiKeyTitle": "data-i18n",
    "modal.auth.apiKeyPlaceholder": "data-i18n-placeholder",
    "modal.auth.authenticate": "data-i18n",
    "modal.auth.oauthTitle": "data-i18n",
    "modal.auth.oauthDescription": "data-i18n",
    "modal.auth.oauthLogin": "data-i18n",
    "modal.auth.localDevNote": "data-i18n-html",
}

# The four strings that live in app.js, each reached through this.t().
SCRIPT_KEYS = [
    "error.authRequired",
    "toast.authSessionExpired",
    "toast.authSuccess",
    "toast.authFailed",
]

REQUIRED_KEYS = list(MARKUP_KEYS) + SCRIPT_KEYS

# The English literals that must no longer be hardcoded in app.js.
RETIRED_LITERALS = [
    "'Session expired or credentials changed. Please refresh the page.'",
    "'Authentication successful'",
    "'Authentication failed. Please check your API key.'",
]


def _catalogs():
    return sorted(I18N_DIR.glob("*.json"))


def test_english_catalog_defines_every_auth_key():
    """en.json is the source of truth for the authentication dialog keys."""
    catalog = json.loads(EN_CATALOG.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in catalog]
    assert not missing, f"en.json does not define: {missing}"


def test_every_catalog_translates_every_auth_key():
    """Every catalog must carry a nonempty translation for every new key."""
    problems = {}
    for path in _catalogs():
        catalog = json.loads(path.read_text(encoding="utf-8"))
        bad = [key for key in REQUIRED_KEYS if not str(catalog.get(key, "")).strip()]
        if bad:
            problems[path.name] = bad
    assert not problems, f"missing or empty authentication translations: {problems}"


def test_local_dev_note_keeps_its_markup_and_variable_name():
    """The local-development sentence is wired with data-i18n-html.

    Every catalog has to keep the <strong>/<code> markup and the environment
    variable name intact, otherwise a language switch would drop them.
    """
    problems = {}
    for path in _catalogs():
        catalog = json.loads(path.read_text(encoding="utf-8"))
        note = catalog["modal.auth.localDevNote"]
        for fragment in ("<strong>", "</strong>", "<code>", "</code>", "MCP_ALLOW_ANONYMOUS_ACCESS=true"):
            if fragment not in note:
                problems.setdefault(path.name, []).append(fragment)
    assert not problems, f"modal.auth.localDevNote lost markup or the variable name: {problems}"


def test_auth_modal_markup_references_its_keys():
    """Each string in the dialog carries the i18n attribute that translates it."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    modal = html.split('<div id="authModal"', 1)
    assert len(modal) == 2, "authModal is no longer in index.html"
    modal_html = modal[1].split("</script>", 1)[0]

    missing = [
        f'{attribute}="{key}"'
        for key, attribute in MARKUP_KEYS.items()
        if f'{attribute}="{key}"' not in modal_html
    ]
    assert not missing, f"authModal markup does not reference: {missing}"


def test_auth_messages_go_through_the_translation_helper():
    """The four app.js messages are looked up with this.t(), fallback included."""
    script = APP_JS.read_text(encoding="utf-8")
    catalog = json.loads(EN_CATALOG.read_text(encoding="utf-8"))

    for key in SCRIPT_KEYS:
        pattern = r"this\.t\(\s*'" + re.escape(key) + r"'\s*,\s*'([^']*)'"
        match = re.search(pattern, script)
        assert match, f"app.js does not look up {key} through this.t()"
        assert match.group(1) == catalog[key], (
            f"the English fallback for {key} drifted from en.json"
        )


def test_retired_auth_literals_are_gone_from_app_js():
    """The old hardcoded toasts must not come back."""
    script = APP_JS.read_text(encoding="utf-8")
    leftovers = [
        literal
        for literal in RETIRED_LITERALS
        if re.search(r"showToast\(\s*" + re.escape(literal), script)
    ]
    assert not leftovers, f"app.js still toasts hardcoded English: {leftovers}"
