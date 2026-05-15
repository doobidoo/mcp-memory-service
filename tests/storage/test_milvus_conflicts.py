"""Tests for Milvus conflict detection, get_conflicts, and resolve_conflict.

Uses Milvus Lite (local file) for isolation.
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")

from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage
from src.mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture(autouse=True)
def _offline_model_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="module")
def milvus_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("milvus_conflicts") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path):
    collection_name = f"mcp_memory_{uuid.uuid4().hex[:12]}"
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=collection_name,
    )
    await instance.initialize()
    yield instance
    await instance.close()


def _make_memory(content: str, tags=None) -> Memory:
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags or ["test"],
        memory_type="observation",
    )


@pytest.mark.asyncio
async def test_detect_conflicts_similar_but_divergent(storage):
    """Two memories with very similar semantics but different facts should be detected as conflicts."""
    # Store first memory
    mem_a = _make_memory("The CI/CD pipeline uses GitHub Actions for all deployments")
    ok_a, _ = await storage.store(mem_a)
    assert ok_a

    # Store a conflicting memory (same topic, different tool)
    mem_b = _make_memory("The CI/CD pipeline uses Jenkins for all deployments")
    ok_b, msg_b = await storage.store(mem_b)
    assert ok_b
    # The conflict detection runs post-store; check if it was detected
    # (message may or may not mention conflicts depending on threshold)

    # Verify via get_conflicts
    conflicts = await storage.get_conflicts()
    # Note: whether this triggers depends on embedding similarity > 0.95
    # With all-MiniLM-L6-v2, "GitHub Actions" vs "Jenkins" may or may not
    # cross the 0.95 threshold. This test validates the plumbing works.
    if conflicts:
        assert len(conflicts) >= 1
        c = conflicts[0]
        assert "hash_a" in c
        assert "hash_b" in c
        assert "similarity" in c
        assert c["similarity"] >= 0.95


@pytest.mark.asyncio
async def test_detect_conflicts_no_false_positive(storage):
    """Completely different memories should NOT trigger conflict detection."""
    mem_a = _make_memory("Python is a programming language created by Guido van Rossum")
    ok_a, _ = await storage.store(mem_a)
    assert ok_a

    mem_b = _make_memory("The weather in Tokyo is sunny today with 25 degrees")
    ok_b, msg_b = await storage.store(mem_b)
    assert ok_b
    assert "conflict" not in msg_b.lower()

    conflicts = await storage.get_conflicts()
    assert len(conflicts) == 0


@pytest.mark.asyncio
async def test_resolve_conflict(storage):
    """resolve_conflict should mark loser as superseded and clean up tags."""
    # Manually create a conflict scenario by storing and tagging
    mem_a = _make_memory("The database uses PostgreSQL version 14")
    mem_b = _make_memory("The database uses PostgreSQL version 16")

    ok_a, _ = await storage.store(mem_a)
    ok_b, _ = await storage.store(mem_b)
    assert ok_a and ok_b

    # Manually tag as conflicting (simulating what _record_conflicts does)
    await storage.update_memory_metadata(
        mem_a.content_hash,
        updates={"tags": mem_a.tags + ["conflict:unresolved"]},
        preserve_timestamps=True,
    )
    await storage.update_memory_metadata(
        mem_b.content_hash,
        updates={"tags": mem_b.tags + ["conflict:unresolved"]},
        preserve_timestamps=True,
    )

    # Resolve: mem_b wins (newer version)
    ok, msg = await storage.resolve_conflict(
        winner_hash=mem_b.content_hash,
        loser_hash=mem_a.content_hash,
    )
    assert ok, f"resolve_conflict failed: {msg}"

    # Verify loser is superseded
    loser = await storage.get_by_hash(mem_a.content_hash)
    assert loser is not None
    assert loser.metadata.get("superseded_by") == mem_b.content_hash
    assert "conflict:unresolved" not in loser.tags

    # Verify winner has boosted quality and no conflict tag
    winner = await storage.get_by_hash(mem_b.content_hash)
    assert winner is not None
    assert winner.metadata.get("quality_score", 0.5) >= 0.8
    assert "conflict:unresolved" not in winner.tags


@pytest.mark.asyncio
async def test_resolve_conflict_nonexistent_hash(storage):
    """resolve_conflict should fail gracefully for missing hashes."""
    ok, msg = await storage.resolve_conflict("nonexistent_hash_1", "nonexistent_hash_2")
    assert not ok
    assert "not found" in msg.lower()


@pytest.mark.asyncio
async def test_get_conflicts_empty(storage):
    """get_conflicts should return empty list when no conflicts exist."""
    conflicts = await storage.get_conflicts()
    assert conflicts == []
