"""Regression tests for the French Knowledge Graph translations (#1752)."""

import json
from pathlib import Path

I18N_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mcp_memory_service"
    / "web"
    / "static"
    / "i18n"
)

REQUIRED_KEYS = [
    "analytics.edgeTypes",
    "analytics.forceLayout",
    "analytics.graphNodes",
    "analytics.knowledgeGraphDescription",
    "analytics.legend",
    "analytics.minConnections",
    "analytics.minConnectionsTip",
    "analytics.minDegree",
    "analytics.minDegreeTip",
    "analytics.nodeColors",
    "analytics.other",
    "analytics.refresh",
    "analytics.ringLayout",
]


def _catalog(name: str) -> dict:
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


def test_english_catalog_defines_every_knowledge_graph_key():
    english = _catalog("en.json")
    missing = [key for key in REQUIRED_KEYS if key not in english]
    assert not missing, f"en.json is missing keys: {missing}"


def test_french_catalog_has_every_knowledge_graph_key():
    french = _catalog("fr.json")
    missing = [key for key in REQUIRED_KEYS if key not in french]
    assert not missing, f"fr.json is missing keys: {missing}"


def test_french_knowledge_graph_values_are_not_empty():
    french = _catalog("fr.json")
    empty = [
        key for key in REQUIRED_KEYS
        if not isinstance(french.get(key), str) or not french[key].strip()
    ]
    assert not empty, f"Empty French translations: {empty}"
