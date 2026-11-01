"""The Indonesian catalog must cover every key the dashboard can ask for (#1762).

Before this, `id.json` did not exist, so Indonesian users fell through to the
English `t()` fallback for the whole dashboard. These tests walk the real
`backend` catalogs instead of synthetic dictionaries, so a key added to
`en.json` without an Indonesian translation fails here instead of accumulating.
"""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module

I18N_DIR = (
    Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "i18n"
)
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z0-9_]+\}")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _catalog(name: str) -> dict:
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def test_indonesian_catalog_exists():
    assert (I18N_DIR / "id.json").is_file()


def test_indonesian_catalog_covers_every_english_key():
    english = _catalog("en.json")
    indonesian = _catalog("id.json")

    missing = sorted(set(english) - set(indonesian))

    assert missing == [], f"keys missing from id.json: {missing}"


def test_indonesian_catalog_has_no_unknown_keys():
    english = _catalog("en.json")
    indonesian = _catalog("id.json")

    extra = sorted(set(indonesian) - set(english))

    assert extra == [], f"keys not defined in en.json: {extra}"


def test_indonesian_values_are_not_empty():
    indonesian = _catalog("id.json")

    blank = sorted(key for key, value in indonesian.items() if not value.strip())

    assert blank == [], f"empty Indonesian translations: {blank}"


def test_indonesian_metadata_identifies_the_language():
    indonesian = _catalog("id.json")

    assert indonesian["meta.language.code"] == "id"
    assert indonesian["meta.language.nativeName"] == "Bahasa Indonesia"
    assert indonesian["meta.language.englishName"] == "Indonesian"


def test_indonesian_keeps_every_interpolation_token():
    english = _catalog("en.json")
    indonesian = _catalog("id.json")

    mismatched = {}
    for key, source in english.items():
        if key not in indonesian:
            continue
        expected = sorted(PLACEHOLDER_RE.findall(source))
        actual = sorted(PLACEHOLDER_RE.findall(indonesian[key]))
        if expected != actual:
            mismatched[key] = {"en": expected, "id": actual}

    assert mismatched == {}


def test_indonesian_keeps_the_english_html_structure():
    english = _catalog("en.json")
    indonesian = _catalog("id.json")

    mismatched = {}
    for key, source in english.items():
        if key not in indonesian:
            continue
        expected = HTML_TAG_RE.findall(source)
        actual = HTML_TAG_RE.findall(indonesian[key])
        if expected != actual:
            mismatched[key] = {"en": expected, "id": actual}

    assert mismatched == {}


def test_languages_api_discovers_indonesian():
    """The picker is built from `/api/languages`, so the new catalog must show up there."""
    client = TestClient(app_module.create_app())

    response = client.get("/api/languages")

    assert response.status_code == 200
    assert "id" in response.json()["languages"]
