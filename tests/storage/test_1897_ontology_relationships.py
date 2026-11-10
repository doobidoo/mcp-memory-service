"""Relationship types and inference type patterns in the ontology overlay (#1897, phase 3).

Relationship types and the (source, target) -> relationship pairs that
inference uses were Python constants. The overlay adds to both, the same way
it adds memory types: it can never replace or re-scope a built-in.
"""

import json
import sqlite3

import pytest

from mcp_memory_service.consolidation.relationship_inference import RelationshipInferenceEngine
from mcp_memory_service.models import ontology
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.ontology_overlay import upsert_relationship
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(f"{temp_db_path}/test_1897_relationships.db")
    await s.initialize()
    yield s
    await s.close()
    ontology.clear_ontology_caches()


def _write_directly(storage, kind, key, payload):
    """A second connection stands in for another process, or a peer's row."""
    other = sqlite3.connect(storage.db_path)
    try:
        with other:
            other.execute(
                "INSERT INTO ontology_entries (kind, key, payload_json, updated_at) VALUES (?, ?, ?, 0)",
                (kind, key, payload if isinstance(payload, str) else json.dumps(payload)),
            )
            other.execute("UPDATE metadata SET value = value + 1 WHERE key = 'ontology_version'")
    finally:
        other.close()


def test_has_entity_is_a_builtin_relationship():
    assert ontology.validate_relationship("has_entity")
    assert not ontology.is_symmetric_relationship("has_entity")


async def test_added_relationship_validates_and_scopes_its_pairs(storage):
    name = await storage.add_relationship_type(
        "Blocks", ["decision → error", "pattern -> decision"], description="A blocks B"
    )
    assert name == "blocks"

    assert ontology.validate_relationship("blocks")
    assert not ontology.is_symmetric_relationship("blocks")
    assert ontology.is_allowed_pair("blocks", "decision", "error")
    assert ontology.is_allowed_pair("blocks", "pattern", "decision")
    assert not ontology.is_allowed_pair("blocks", "error", "decision")
    assert ontology.get_relationship_types()["blocks"]["valid_patterns"] == [
        "decision → error",
        "pattern → decision",
    ]


async def test_graph_storage_accepts_an_added_symmetric_relationship(storage):
    graph = GraphStorage(storage.db_path)
    assert not await graph.store_association("a" * 64, "b" * 64, 0.9, ["semantic"], relationship_type="pairs_with")

    await storage.add_relationship_type("pairs_with", ["any → any"], symmetric=True)

    assert await graph.store_association("a" * 64, "b" * 64, 0.9, ["semantic"], relationship_type="pairs_with")
    # Symmetric, so the reverse edge was written too.
    assert (await graph.get_relationship_types("b" * 64)).get("pairs_with")


@pytest.mark.parametrize(
    "name, patterns, kwargs",
    [
        ("fixes", ["any → any"], {}),            # built-in
        ("has_entity", ["any → any"], {}),       # built-in since this phase
        ("1bad", ["any → any"], {}),             # invalid name
        ("blocks", [], {}),                      # no pattern
        ("blocks", "any → any", {}),             # a string, not a list
        ("blocks", ["decision"], {}),            # no arrow
        ("blocks", ["nope → error"], {}),        # unknown end
        ("blocks", ["insight → error"], {}),     # subtype, not a base type
        ("blocks", ["any → any"], {"symmetric": "yes"}),
        ("blocks", ["any → any"], {"description": 3}),
    ],
)
async def test_rejects_relationships_that_would_shadow_or_never_match(storage, name, patterns, kwargs):
    with pytest.raises(ValueError):
        await storage.add_relationship_type(name, patterns, **kwargs)


async def test_type_pattern_reaches_inference(storage):
    engine = RelationshipInferenceEngine(min_confidence=0.0, min_typed_confidence=0.0)
    assert engine._analyze_type_combination("decision", "error") == []

    await storage.add_relationship_type("blocks", ["decision → error"])
    assert await storage.set_type_pattern("Decision", "error", "blocks", 0.8) == "decision → error"

    assert engine._analyze_type_combination("decision", "error") == [("blocks", 0.8)]
    # Subtypes resolve to their base type first, as for the built-in pairs.
    assert engine._analyze_type_combination("architecture", "error") == [("blocks", 0.8)]

    await storage.set_type_pattern("decision", "error", "blocks", 0.9)  # a value: replaced
    assert engine._analyze_type_combination("decision", "error") == [("blocks", 0.9)]

    assert await storage.clear_type_pattern("decision", "error")
    assert engine._analyze_type_combination("decision", "error") == []


@pytest.mark.parametrize(
    "source, target, relationship, confidence",
    [
        ("learning", "error", "fixes", 0.9),       # built-in pair
        ("decision", "error", "nope", 0.8),        # unknown relationship
        ("observation", "decision", "fixes", 0.8),  # fixes does not allow this pair
        ("insight", "error", "related", 0.8),      # subtype end
        ("decision", "error", "related", 0),
        ("decision", "error", "related", 1.5),
        ("decision", "error", "related", True),
    ],
)
async def test_rejects_type_patterns_inference_would_drop(storage, source, target, relationship, confidence):
    with pytest.raises(ValueError):
        await storage.set_type_pattern(source, target, relationship, confidence)


async def test_removing_a_relationship_drops_its_type_patterns(storage):
    await storage.add_relationship_type("blocks", ["decision → error"])
    await storage.set_type_pattern("decision", "error", "blocks", 0.8)

    assert await storage.remove_relationship_type("blocks")
    assert not await storage.remove_relationship_type("blocks")
    assert not await storage.remove_relationship_type("fixes")  # built-ins are not in the table

    assert not ontology.validate_relationship("blocks")
    assert ("decision", "error") not in ontology.get_type_patterns()

    await storage.add_relationship_type("blocks", ["decision → error"])  # revive
    assert ontology.get_type_patterns()[("decision", "error")] == ("blocks", 0.8)


async def test_write_by_another_process_is_picked_up(storage):
    other = sqlite3.connect(storage.db_path)
    try:
        upsert_relationship(other, "blocks", {"description": "", "valid_patterns": ["any → any"], "symmetric": False})
    finally:
        other.close()

    assert ontology.validate_relationship("blocks")


async def test_overlay_rows_cannot_rescope_builtins_or_smuggle_bad_specs(storage):
    _write_directly(storage, "relationship", "fixes", {"valid_patterns": ["any → any"]})
    _write_directly(storage, "relationship", "loose", {"valid_patterns": "any → any"})
    _write_directly(storage, "relationship", "broken", "not json")
    _write_directly(storage, "type_pattern", "learning → error", {"relationship": "related", "confidence": 0.9})
    _write_directly(storage, "type_pattern", "decision → error", {"relationship": "nope", "confidence": 0.9})

    assert not ontology.is_allowed_pair("fixes", "observation", "decision")
    assert not ontology.validate_relationship("loose")
    assert not ontology.validate_relationship("broken")
    patterns = ontology.get_type_patterns()
    assert patterns[("learning", "error")] == ("fixes", 0.8)
    assert ("decision", "error") not in patterns


async def test_in_memory_database_relationships_load(monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(":memory:")
    await s.initialize()
    try:
        await s.add_relationship_type("blocks", ["decision → error"])
        await s.set_type_pattern("decision", "error", "blocks", 0.8)
        assert ontology.validate_relationship("blocks")
        assert ontology.get_type_patterns()[("decision", "error")] == ("blocks", 0.8)
    finally:
        await s.close()
        ontology.clear_ontology_caches()
