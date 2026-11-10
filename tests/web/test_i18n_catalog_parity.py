"""Every dashboard locale catalog carries exactly the key set of en.json (#1928).

The non-English catalogs once drifted ahead of en.json: after the analytics
graph slices (#1806) landed, five catalogs still carried seven ``analytics.*``
keys that en.json no longer had and nothing referenced. This test keeps that
drift from coming back in both directions — a catalog that misses a key the
English catalog has, or adds a key it does not, fails with the diff.
"""

import json
from pathlib import Path

import pytest

I18N_DIR = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "i18n"


def _key_set(catalog: Path) -> set:
    def flatten(node: dict, prefix: str):
        for key, value in node.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                yield from flatten(value, name + ".")
            else:
                yield name

    return set(flatten(json.loads(catalog.read_text(encoding="utf-8")), ""))


@pytest.mark.parametrize("locale", sorted(p.stem for p in I18N_DIR.glob("*.json") if p.stem != "en"))
def test_catalog_matches_english_key_set(locale: str):
    english = _key_set(I18N_DIR / "en.json")
    catalog = _key_set(I18N_DIR / f"{locale}.json")

    missing = sorted(english - catalog)
    extra = sorted(catalog - english)

    assert not missing and not extra, (
        f"{locale}.json must carry exactly the en.json key set. "
        f"Missing from {locale}.json: {missing}. "
        f"Extra in {locale}.json (absent from en.json): {extra}. "
        "Copy new keys from en.json, and remove keys that en.json dropped."
    )
