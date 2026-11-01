"""The German catalog must carry the Knowledge Graph strings the dashboard asks for.

#1750: thirteen `analytics.*` keys were defined in `en.json` but missing from
`de.json`, so selecting German left the graph labels, layout toggles and the two
filter tooltips in English through the dashboard's `t()` fallback.
"""

import json
from pathlib import Path

I18N_DIR = Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "i18n"

KNOWLEDGE_GRAPH_KEYS = [
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


def test_english_catalog_defines_the_keys_we_translated():
    # If a key is renamed upstream this fails here rather than leaving a German
    # translation that nothing reads.
    english = _catalog("en.json")
    assert [key for key in KNOWLEDGE_GRAPH_KEYS if key not in english] == []


def test_german_catalog_has_every_knowledge_graph_key():
    german = _catalog("de.json")
    assert [key for key in KNOWLEDGE_GRAPH_KEYS if key not in german] == []


def test_german_knowledge_graph_values_are_not_empty():
    german = _catalog("de.json")
    blank = [key for key in KNOWLEDGE_GRAPH_KEYS if not german[key].strip()]
    assert blank == [], f"empty German translations: {blank}"


def test_german_catalog_still_keeps_the_duplicate_memory_types_key():
    """Guard against a reserialize of this file collapsing a pre-existing duplicate.

    `analytics.memoryTypes` is defined twice in `de.json`. That is a separate bug and
    out of scope here, but rewriting the file through a JSON serializer would silently
    drop one of them, which is why this is asserted on the raw text.
    """
    raw = (I18N_DIR / "de.json").read_text(encoding="utf-8")
    assert raw.count('"analytics.memoryTypes"') == 2
