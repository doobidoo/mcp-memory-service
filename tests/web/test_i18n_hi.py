"""Regression coverage for the Hindi dashboard locale (#1761)."""

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


def test_hindi_catalog_exists():
    assert (I18N_DIR / "hi.json").is_file()


def test_hindi_catalog_covers_every_english_key():
    english = _catalog("en.json")
    hindi = _catalog("hi.json")

    missing = sorted(set(english) - set(hindi))
    extra = sorted(set(hindi) - set(english))

    assert not missing, f"Missing Hindi keys: {missing}"
    assert not extra, f"Unknown Hindi keys: {extra}"


def test_hindi_values_are_nonempty_strings():
    hindi = _catalog("hi.json")

    invalid = {
        key: value
        for key, value in hindi.items()
        if not isinstance(value, str) or not value.strip()
    }

    assert not invalid, f"Empty or invalid Hindi values: {invalid}"


def test_hindi_metadata_identifies_the_language():
    hindi = _catalog("hi.json")

    assert hindi["meta.language.code"] == "hi"
    assert hindi["meta.language.nativeName"] == "हिन्दी"
    assert hindi["meta.language.englishName"] == "Hindi"


def test_hindi_preserves_interpolation_tokens():
    english = _catalog("en.json")
    hindi = _catalog("hi.json")

    mismatched = {}

    for key, source in english.items():
        expected = sorted(PLACEHOLDER_RE.findall(source))
        actual = sorted(PLACEHOLDER_RE.findall(hindi[key]))

        if expected != actual:
            mismatched[key] = {"en": expected, "hi": actual}

    assert not mismatched, f"Interpolation mismatches: {mismatched}"


def test_hindi_preserves_html_structure():
    english = _catalog("en.json")
    hindi = _catalog("hi.json")

    mismatched = {
        key: {
            "en": HTML_TAG_RE.findall(source),
            "hi": HTML_TAG_RE.findall(hindi[key]),
        }
        for key, source in english.items()
        if HTML_TAG_RE.findall(source) != HTML_TAG_RE.findall(hindi[key])
    }

    assert not mismatched, f"HTML structure mismatches: {mismatched}"


def test_hindi_preserves_url_targets():
    english = _catalog("en.json")
    hindi = _catalog("hi.json")

    mismatched = {
        key: {
            "en": URL_RE.findall(source),
            "hi": URL_RE.findall(hindi[key]),
        }
        for key, source in english.items()
        if URL_RE.findall(source) != URL_RE.findall(hindi[key])
    }

    assert not mismatched, f"URL mismatches: {mismatched}"


def test_languages_api_discovers_hindi():
    """The real application must discover hi from the actual catalog directory."""
    client = TestClient(app_module.create_app())

    response = client.get("/api/languages")

    assert response.status_code == 200
    assert "hi" in response.json()["languages"]


def test_hindi_catalog_is_served_by_the_real_application():
    client = TestClient(app_module.create_app())

    response = client.get("/static/i18n/hi.json")

    assert response.status_code == 200

    catalog = response.json()

    assert catalog["meta.language.code"] == "hi"
    assert catalog["meta.language.nativeName"] == "हिन्दी"
    assert catalog["meta.language.englishName"] == "Hindi"
