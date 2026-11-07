"""Regression coverage for the Russian dashboard locale (#1763)."""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module

I18N_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/mcp_memory_service/web/static/i18n"
)

PLACEHOLDER_RE = re.compile(r"\{[A-Za-z0-9_]+\}")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
URL_RE = re.compile(r'https?://[^\s<>"\']+')


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate catalog key: {key}"
        result[key] = value
    return result


def _catalog(name):
    return json.loads(
        (I18N_DIR / name).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_keys,
    )


def test_russian_catalog_exists():
    assert (I18N_DIR / "ru.json").is_file()


def test_russian_catalog_covers_every_english_key():
    english = _catalog("en.json")
    russian = _catalog("ru.json")

    missing = sorted(set(english) - set(russian))
    extra = sorted(set(russian) - set(english))

    assert not missing, f"Missing Russian keys: {missing}"
    assert not extra, f"Unknown Russian keys: {extra}"


def test_russian_values_are_nonempty_strings():
    russian = _catalog("ru.json")

    invalid = {
        key: value
        for key, value in russian.items()
        if not isinstance(value, str) or not value.strip()
    }

    assert not invalid, f"Empty or invalid Russian values: {invalid}"


def test_russian_metadata_identifies_the_language():
    russian = _catalog("ru.json")

    assert russian["meta.language.code"] == "ru"
    assert russian["meta.language.nativeName"] == "Русский"
    assert russian["meta.language.englishName"] == "Russian"
    assert russian["meta.language.flag"] == "🇷🇺"


def test_russian_preserves_interpolation_tokens():
    english = _catalog("en.json")
    russian = _catalog("ru.json")

    mismatched = {}

    for key, source in english.items():
        expected = sorted(PLACEHOLDER_RE.findall(source))
        actual = sorted(PLACEHOLDER_RE.findall(russian[key]))

        if expected != actual:
            mismatched[key] = {"en": expected, "ru": actual}

    assert not mismatched, f"Interpolation mismatches: {mismatched}"


def test_russian_preserves_html_structure():
    english = _catalog("en.json")
    russian = _catalog("ru.json")

    mismatched = {}

    for key, source in english.items():
        expected = HTML_TAG_RE.findall(source)
        actual = HTML_TAG_RE.findall(russian[key])

        if expected != actual:
            mismatched[key] = {"en": expected, "ru": actual}

    assert not mismatched, f"HTML structure mismatches: {mismatched}"


def test_russian_preserves_url_targets():
    english = _catalog("en.json")
    russian = _catalog("ru.json")

    mismatched = {}

    for key, source in english.items():
        expected = URL_RE.findall(source)
        actual = URL_RE.findall(russian[key])

        if expected != actual:
            mismatched[key] = {"en": expected, "ru": actual}

    assert not mismatched, f"URL mismatches: {mismatched}"


def test_languages_api_discovers_russian():
    """The real application must discover ru from the actual catalog directory."""
    client = TestClient(app_module.create_app())

    response = client.get("/api/languages")

    assert response.status_code == 200
    assert "ru" in response.json()["languages"]


def test_russian_catalog_is_served_by_the_real_application():
    client = TestClient(app_module.create_app())

    response = client.get("/static/i18n/ru.json")

    assert response.status_code == 200

    catalog = response.json()

    assert catalog["meta.language.code"] == "ru"
    assert catalog["meta.language.nativeName"] == "Русский"
    assert catalog["meta.language.englishName"] == "Russian"
    assert catalog["meta.language.flag"] == "🇷🇺"
