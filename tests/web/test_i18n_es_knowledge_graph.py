"""Regression tests for the Spanish Knowledge Graph translations (#1751)."""

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


def test_spanish_catalog_has_every_knowledge_graph_key():
    spanish = _catalog("es.json")
    missing = [key for key in REQUIRED_KEYS if key not in spanish]
    assert not missing, f"es.json is missing keys: {missing}"


def test_spanish_knowledge_graph_values_are_not_empty():
    spanish = _catalog("es.json")
    empty = [
        key for key in REQUIRED_KEYS
        if not isinstance(spanish.get(key), str) or not spanish[key].strip()
    ]
    assert not empty, f"Empty Spanish translations: {empty}"
