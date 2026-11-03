"""Regression coverage for the Japanese Knowledge Graph translations (#1753)."""

import json
import re
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "i18n"
EN_CATALOG = I18N_DIR / "en.json"
JA_CATALOG = I18N_DIR / "ja.json"

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


def test_english_catalog_defines_every_knowledge_graph_key():
    """The en.json catalog is the source of truth for the required keys."""
    catalog = json.loads(EN_CATALOG.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in catalog]
    assert not missing, f"en.json no longer defines: {missing}"


def test_japanese_catalog_has_every_knowledge_graph_key():
    """The real ja.json must define every Knowledge Graph key (#1753)."""
    catalog = json.loads(JA_CATALOG.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in catalog]
    assert not missing, f"ja.json is missing Knowledge Graph keys: {missing}"


def test_japanese_knowledge_graph_values_are_not_empty():
    """Every Knowledge Graph translation must be a nonempty string."""
    catalog = json.loads(JA_CATALOG.read_text(encoding="utf-8"))
    empty = [key for key in REQUIRED_KEYS if not catalog.get(key, "").strip()]
    assert not empty, f"ja.json has empty Knowledge Graph translations: {empty}"


def test_japanese_catalog_duplicate_key_is_preserved():
    """ja.json carries a pre-existing duplicate analytics.memoryTypes key.

    A json round trip would silently collapse it, so guard the raw text: the key
    must still appear exactly twice after this change.
    """
    raw = JA_CATALOG.read_text(encoding="utf-8")
    keys = re.findall(r'^\s*"([^"]+)":', raw, re.M)
    assert keys.count("analytics.memoryTypes") == 2
