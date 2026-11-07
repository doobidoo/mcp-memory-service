import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module


def _unique_keys(pairs):
    catalog = dict(pairs)
    assert len(catalog) == len(pairs), "Translation catalog contains duplicate keys"
    return catalog


def test_pt_catalog_parity():
    """
    Reads the real en.json and pt.json catalogs from disk to verify coverage,
    non-empty values, and interpolation token parity.
    """
    project_root = Path(__file__).resolve().parents[2]
    i18n_dir = project_root / "src/mcp_memory_service/web/static/i18n"
    en_path = i18n_dir / "en.json"
    pt_path = i18n_dir / "pt.json"

    assert en_path.exists(), "Base en.json catalog not found"
    assert pt_path.exists(), "New pt.json catalog not found"

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    with open(pt_path, "r", encoding="utf-8") as f:
        pt_data = json.load(f, object_pairs_hook=_unique_keys)

    en_keys = set(en_data.keys())
    pt_keys = set(pt_data.keys())

    # 1. Verify Portuguese covers every key in the English catalog
    missing_keys = en_keys - pt_keys
    extra_keys = pt_keys - en_keys
    assert not missing_keys, f"pt.json is missing keys: {missing_keys}"
    assert not extra_keys, f"pt.json has extra keys not in en.json: {extra_keys}"

    # 2. Token extraction regex (matches {any_text})
    token_pattern = re.compile(r"\{[^}]+\}")
    html_pattern = re.compile(r"</?[a-zA-Z][^>]*>")
    url_pattern = re.compile(r"https?://[^\s<>\"]+")

    # 3. Verify non-empty values and identical tokens
    for key in en_keys:
        en_val = str(en_data[key])
        pt_val = str(pt_data[key])

        assert len(pt_val.strip()) > 0, f"pt.json has an empty string for key: '{key}'"

        en_tokens = sorted(token_pattern.findall(en_val))
        pt_tokens = sorted(token_pattern.findall(pt_val))

        assert en_tokens == pt_tokens, (
            f"Interpolation token mismatch for key '{key}'. "
            f"Expected {en_tokens}, got {pt_tokens}"
        )
        assert html_pattern.findall(en_val) == html_pattern.findall(pt_val), key
        assert url_pattern.findall(en_val) == url_pattern.findall(pt_val), key


def test_api_languages_discovers_pt():
    """
    Hits the real application endpoint to verify dynamic discovery
    of the Portuguese locale.
    """
    client = TestClient(app_module.create_app())
    response = client.get("/api/languages")

    assert response.status_code == 200
    assert "pt" in response.json()["languages"]

    catalog_response = client.get("/static/i18n/pt.json")
    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    assert catalog["meta.language.code"] == "pt"
    assert catalog["meta.language.nativeName"] == "Português (Brasil)"
    assert catalog["meta.language.englishName"] == "Portuguese (Brazil)"
