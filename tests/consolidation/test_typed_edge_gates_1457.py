"""Issue #1457: the typed-edge gates consolidation really uses.

The gates are confidence 0.75 and similarity 0.65, the second on the score the
consolidator passes in, ``association.similarity_score`` = ``(cos + 1) / 2``.
These tests build the consolidator on real SqliteVecMemoryStorage and
GraphStorage, keep the real inference engine, and read the stored edge back.

``MCP_TYPED_EDGES_ENABLED`` is patched on: since #1883 it defaults to off, and
with it off every edge is ``related``, whatever the gates say.
"""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from mcp_memory_service.consolidation.base import ConsolidationConfig, MemoryAssociation
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.consolidation.relationship_inference import RelationshipInferenceEngine
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

_MOD = "mcp_memory_service.consolidation.consolidator"
_T0 = 1735689600.0


async def _store(storage, content, memory_type, created_at):
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["__test__"],
        memory_type=memory_type,
        created_at=created_at,
    )
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory


async def _stored_edge(temp_db_path, source_type, target_type, source_text, target_text, similarity):
    """Run one pair through the real graph-table path and return its edge."""
    db_path = os.path.join(temp_db_path, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    consolidator = DreamInspiredConsolidator(storage, ConsolidationConfig())
    consolidator.graph_storage = GraphStorage(db_path)
    try:
        # A week apart, so the temporal rules stay out of it.
        source = await _store(storage, source_text, source_type, _T0 + 604800)
        target = await _store(storage, target_text, target_type, _T0)
        association = MemoryAssociation(
            source_memory_hashes=[source.content_hash, target.content_hash],
            similarity_score=similarity,
            connection_type="semantic",
            discovery_method="test",
            discovery_date=datetime.now(timezone.utc),
        )
        await consolidator._store_associations_in_graph_table([association])
        return await consolidator.graph_storage.get_association(source.content_hash, target.content_hash)
    finally:
        await consolidator.graph_storage.close()
        await storage.close()


def test_constructor_defaults_are_the_chosen_gates():
    engine = RelationshipInferenceEngine()
    assert engine.min_confidence == 0.6
    assert engine.min_typed_confidence == 0.75
    assert engine.min_typed_similarity == 0.65


@pytest.mark.asyncio
async def test_consolidator_engine_carries_the_chosen_gates(temp_db_path):
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await storage.initialize()
    try:
        engine = DreamInspiredConsolidator(storage, ConsolidationConfig()).relationship_inference
        assert engine.min_confidence == 0.6
        assert engine.min_typed_confidence == 0.75
        assert engine.min_typed_similarity == 0.65
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_consolidator_passes_the_gates_explicitly(temp_db_path):
    """The consolidator states the values; it does not lean on constructor defaults."""
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await storage.initialize()
    try:
        with patch(f"{_MOD}.RelationshipInferenceEngine") as engine_class:
            DreamInspiredConsolidator(storage, ConsolidationConfig())
    finally:
        await storage.close()
    kwargs = engine_class.call_args.kwargs
    assert kwargs["min_confidence"] == 0.6
    assert kwargs["min_typed_confidence"] == 0.75
    assert kwargs["min_typed_similarity"] == 0.65


@pytest.mark.asyncio
@patch(f"{_MOD}.TYPED_EDGES_ENABLED", True)
async def test_error_error_causes_at_060_comes_back_related(temp_db_path):
    """The pair named in the issue: ``causes`` 0.6 with a shared keyword.

    Asked of the engine, because this one is not visible on the stored edge:
    ``apply_review_band`` already stores 0.60 and below as ``related``.
    """
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await storage.initialize()
    try:
        engine = DreamInspiredConsolidator(storage, ConsolidationConfig()).relationship_inference
        label, confidence = await engine.infer_relationship_type(
            source_type="error",
            target_type="error",
            source_content="redis connection pool exhausted during deploy",
            target_content="redis connection refused after restart",
            source_timestamp=_T0 + 604800,
            target_timestamp=_T0,
            similarity=0.69,
        )
    finally:
        await storage.close()
    assert label == "related"
    assert confidence == 0.6


@pytest.mark.asyncio
@patch(f"{_MOD}.TYPED_EDGES_ENABLED", True)
async def test_confidence_between_060_and_075_is_stored_related(temp_db_path, unique_content, monkeypatch):
    """Two resolution keywords give ``fixes`` 0.7: above the review band's
    floor, so before this gate it was stored typed. Similarity clears its gate."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    edge = await _stored_edge(
        temp_db_path,
        "decision",
        "error",
        unique_content("redis pool timeout fixed and resolved by raising the limit"),
        unique_content("redis pool timeout during deploy"),
        similarity=0.69,
    )
    assert edge["relationship_type"] == "related"


@pytest.mark.asyncio
@patch(f"{_MOD}.TYPED_EDGES_ENABLED", True)
@pytest.mark.parametrize(
    "similarity, label",
    [(0.60, "related"), (0.6499, "related"), (0.65, "fixes"), (0.69, "fixes")],
)
async def test_similarity_gate_reads_the_normalized_score(temp_db_path, unique_content, monkeypatch, similarity, label):
    """``learning -> error`` gives ``fixes`` 0.8, which clears the confidence
    gate, so only similarity decides. 0.65 sits under the 0.7 association
    ceiling: read as a cosine it would leave nothing to type."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    edge = await _stored_edge(
        temp_db_path,
        "learning",
        "error",
        unique_content("redis pool timeout: raise the limit before deploy"),
        unique_content("redis pool timeout during deploy"),
        similarity=similarity,
    )
    assert edge["relationship_type"] == label


@pytest.mark.asyncio
@patch(f"{_MOD}.TYPED_EDGES_ENABLED", False)
async def test_default_off_writes_related_even_past_both_gates(temp_db_path, unique_content, monkeypatch):
    """With the #1883 default, a pair that clears both gates is still ``related``."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    edge = await _stored_edge(
        temp_db_path,
        "learning",
        "error",
        unique_content("redis pool timeout: raise the limit before deploy"),
        unique_content("redis pool timeout during deploy"),
        similarity=0.69,
    )
    assert edge["relationship_type"] == "related"
