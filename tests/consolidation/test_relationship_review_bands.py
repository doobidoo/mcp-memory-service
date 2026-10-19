"""Issue #1460: uncertain inferred edges must not hide memories."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.consolidation.base import ConsolidationConfig, MemoryAssociation
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def consolidator(temp_db_path):
    db_path = f"{temp_db_path}/review.db"
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    instance = DreamInspiredConsolidator(storage, ConsolidationConfig())
    instance.graph_storage = GraphStorage(db_path)
    try:
        yield instance
    finally:
        await instance.graph_storage.close()
        await storage.close()


async def _pair(consolidator, source, target, types, similarity=0.7, metadata=None):
    memories = []
    for content, memory_type, timestamp in zip(
        (source, target), types, (1735689600.0, 1746057600.0)
    ):
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            memory_type=memory_type,
            tags=["__test__", "project"],
            created_at=timestamp,
        )
        success, message = await consolidator.storage.store(memory)
        assert success, message
        memories.append(memory)
    association = MemoryAssociation(
        source_memory_hashes=[m.content_hash for m in memories],
        similarity_score=similarity,
        connection_type="semantic",
        discovery_method="test",
        discovery_date=datetime.now(timezone.utc),
        metadata=metadata or {},
    )
    return memories, association


@pytest.mark.parametrize(
    "confidence, expected_type, needs_review",
    [
        (0.0, "related", False),
        (0.5999, "related", False),
        (0.6, "related", False),
        (0.6001, "contradicts", True),
        (0.75, "contradicts", True),
        (0.8499, "contradicts", True),
        (0.85, "contradicts", False),
        (1.0, "contradicts", False),
    ],
)
def test_band_boundaries(confidence, expected_type, needs_review):
    from mcp_memory_service.consolidation.relationship_inference import (
        apply_review_band,
    )

    assert apply_review_band("contradicts", confidence) == (expected_type, needs_review)


@pytest.mark.parametrize("confidence", [0.0, 0.6, 0.75, 0.85, 1.0])
def test_related_edges_do_not_acquire_a_typed_label_or_review_flag(confidence):
    from mcp_memory_service.consolidation.relationship_inference import (
        apply_review_band,
    )

    assert apply_review_band("related", confidence) == ("related", False)


@pytest.mark.parametrize("auto_supersede", [False, True])
async def test_keyword_contradiction_keeps_both_memories_visible(
    consolidator, monkeypatch, auto_supersede
):
    monkeypatch.setattr(
        "mcp_memory_service.consolidation.consolidator.CONSOLIDATION_AUTO_SUPERSEDE",
        auto_supersede,
    )
    memories, association = await _pair(
        consolidator,
        "Never commit to main",
        "The cache was never warmed",
        ("observation", "observation"),
        # Discovery metadata must not override the actual inference result.
        metadata={
            "relationship_confidence": 1.0,
            "needs_review": False,
            "context": "kept",
        },
    )

    await consolidator._store_associations_in_graph_table([association])

    for source, target in (memories, list(reversed(memories))):
        edge = await consolidator.graph_storage.get_association(
            source.content_hash, target.content_hash
        )
        assert edge["relationship_type"] == "contradicts"
        assert edge["metadata"]["relationship_confidence"] == 0.75
        assert edge["metadata"]["needs_review"] is True
        assert edge["metadata"]["context"] == "kept"
        found = await consolidator.storage.search_memories(
            query=source.content, limit=10
        )
        assert source.content_hash in {m["content_hash"] for m in found["memories"]}


@pytest.mark.parametrize(
    "source, target, types, expected_type, confidence, needs_review",
    [
        (
            "Database learning",
            "Database decision",
            ("learning", "decision"),
            "related",
            0.6,
            False,
        ),
        (
            "Database learning",
            "Database error",
            ("learning", "error"),
            "fixes",
            0.8,
            True,
        ),
        (
            "Fixed, resolved, corrected and patched the database",
            "Database error",
            ("learning", "error"),
            "fixes",
            0.9,
            False,
        ),
        (
            "Database choice",
            "Database plan",
            ("decision", "decision"),
            "related",
            0.0,
            False,
        ),
    ],
)
async def test_real_inference_bands_are_persisted(
    consolidator, source, target, types, expected_type, confidence, needs_review
):
    memories, association = await _pair(consolidator, source, target, types)
    await consolidator._store_associations_in_graph_table([association])
    edge = await consolidator.graph_storage.get_association(
        *(m.content_hash for m in memories)
    )
    assert edge["relationship_type"] == expected_type
    assert edge["metadata"]["relationship_confidence"] == confidence
    assert edge["metadata"]["needs_review"] is needs_review


async def test_similarity_guard_still_prevents_typed_edges(consolidator):
    memories, association = await _pair(
        consolidator,
        "Never commit to main",
        "The cache was never warmed",
        ("observation", "observation"),
        similarity=0.1,
    )
    await consolidator._store_associations_in_graph_table([association])
    edge = await consolidator.graph_storage.get_association(
        *(m.content_hash for m in memories)
    )
    assert edge["relationship_type"] == "related"
    assert edge["metadata"]["relationship_confidence"] == 0.75
    assert edge["metadata"]["needs_review"] is False


async def test_inference_failure_keeps_related_edge_with_zero_confidence(
    consolidator, monkeypatch
):
    memories, association = await _pair(
        consolidator, "Database learning", "Database error", ("learning", "error")
    )
    # Fault injection: the graph and memory storage still execute normally.
    monkeypatch.setattr(
        consolidator.relationship_inference,
        "infer_relationship_type",
        AsyncMock(side_effect=RuntimeError("inference unavailable")),
    )
    await consolidator._store_associations_in_graph_table([association])
    edge = await consolidator.graph_storage.get_association(
        *(m.content_hash for m in memories)
    )
    assert edge["relationship_type"] == "related"
    assert edge["metadata"]["relationship_confidence"] == 0.0
    assert edge["metadata"]["needs_review"] is False


async def test_missing_memory_still_stores_zero_confidence(consolidator):
    memories, association = await _pair(
        consolidator, "Database learning", "Database error", ("learning", "error")
    )
    await consolidator.storage.delete(memories[1].content_hash)
    await consolidator._store_associations_in_graph_table([association])
    edge = await consolidator.graph_storage.get_association(
        *(m.content_hash for m in memories)
    )
    assert edge["relationship_type"] == "related"
    assert edge["metadata"]["relationship_confidence"] == 0.0
    assert edge["metadata"]["needs_review"] is False


async def test_typed_edges_disabled_stays_related(consolidator):
    consolidator.relationship_inference.typed_edges_enabled = False
    memories, association = await _pair(
        consolidator,
        "Never commit to main",
        "The cache was never warmed",
        ("observation", "observation"),
    )
    await consolidator._store_associations_in_graph_table([association])
    edge = await consolidator.graph_storage.get_association(
        *(m.content_hash for m in memories)
    )
    assert edge["relationship_type"] == "related"
    assert edge["metadata"]["relationship_confidence"] == 0.0
    assert edge["metadata"]["needs_review"] is False
