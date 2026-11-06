"""Entity linking must not overwrite contradicts edges (#1874).

memory_graph keys a pair by (source_hash, target_hash) and store_association()
writes with INSERT OR REPLACE, so a pair holds one edge and the last writer
wins. The linker runs after conflict detection on every store, and conflicting
memories usually share an entity, so it used to turn the contradicts edges into
shares_entity and memory_conflicts lost the pair.

Real sqlite-vec storage and the real GraphStorage, linker and extractor; only
the service's graph lookup is pointed at the test database, as in
tests/quality/test_rebuild_entities.py.
"""

import os
import time
from math import cos, radians, sin

import pytest

import mcp_memory_service.config as cfg
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.reasoning.entity_linker import EntityLinker
from mcp_memory_service.server.handlers import graph as graph_handlers
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

A = "The primary database is PostgreSQL 16 on the storage node"
B = "Primary data lives in PostgreSQL 11 running inside a container"
C = "Nightly backups of the cluster are copied to object storage"


@pytest.fixture
async def linked(temp_db_path, monkeypatch):
    """Storage, service and graph on one database, with entity linking on."""
    path = os.path.join(temp_db_path, "linker_conflicts.db")
    storage = SqliteVecMemoryStorage(path)
    await storage.initialize()
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "")
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
    graph = GraphStorage(path)

    async def service_graph():
        return graph

    monkeypatch.setattr(graph_handlers, "get_graph_storage", service_graph)
    try:
        yield storage, graph
    finally:
        await graph.close()
        await storage.close()


def _pair_types(storage, x, y):
    """Relationship types on both directions of a pair."""
    return sorted(row[0] for row in storage.conn.execute(
        "SELECT relationship_type FROM memory_graph "
        "WHERE (source_hash = ? AND target_hash = ?) "
        "OR (source_hash = ? AND target_hash = ?)",
        (x, y, y, x),
    ))


@pytest.mark.asyncio
async def test_store_keeps_conflict_when_linker_runs(linked, monkeypatch):
    """A contradicting store keeps both contradicts edges with linking on."""
    storage, _ = linked
    dimension = storage.embedding_dimension

    def embedding_at(degrees):
        angle = radians(degrees)
        return [cos(angle), sin(angle), *([0.0] * (dimension - 2))]

    # 8 degrees apart: cosine ~0.99, above the 0.95 conflict threshold.
    embeddings = {A: embedding_at(0), B: embedding_at(8)}
    monkeypatch.setattr(
        storage, "_generate_embedding",
        lambda content: embeddings.get(content, embedding_at(90)),
    )
    service = MemoryService(storage)

    first = await service.store_memory(A, tags=["repro"])
    assert first.get("success"), first
    a = first["memory"]["content_hash"]
    # Older than the 24 h semantic dedup window, so B is stored, not rejected.
    old = time.time() - 48 * 3600
    storage.conn.execute(
        "UPDATE memories SET created_at = ?, updated_at = ? WHERE content_hash = ?",
        (old, old, a),
    )
    storage.conn.commit()

    second = await service.store_memory(B, tags=["repro"])
    assert second.get("success"), second
    b = second["memory"]["content_hash"]

    # Precondition: a and b share an entity, so the linker had this pair to link.
    shared = storage.conn.execute(
        "SELECT target_hash FROM memory_graph "
        "WHERE relationship_type = 'has_entity' AND source_hash = ? "
        "INTERSECT "
        "SELECT target_hash FROM memory_graph "
        "WHERE relationship_type = 'has_entity' AND source_hash = ?",
        (a, b),
    ).fetchall()
    assert shared

    assert _pair_types(storage, a, b) == ["contradicts", "contradicts"]
    conflicts = await storage.get_conflicts()
    assert {tuple(sorted((c["hash_a"], c["hash_b"]))) for c in conflicts} == {
        tuple(sorted((a, b)))
    }


@pytest.mark.asyncio
async def test_linker_skips_protected_pair_and_links_the_rest(linked):
    """A pair with a non-entity edge is skipped; an unlinked pair still links."""
    storage, graph = linked
    hashes = []
    for content in (A, B, C):
        memory = Memory(content=content, content_hash=generate_content_hash(content))
        ok, message = await storage.store(memory, skip_semantic_dedup=True)
        assert ok, message
        hashes.append(memory.content_hash)
    a, b, c = hashes

    for h in (a, c):
        await graph.store_entity_link(h, "postgres", "technology")
    # Setup only: the private writer store_memory() uses for contradicts edges.
    await storage._record_conflicts(
        b, [{"existing_hash": a, "similarity": 0.97, "divergence": 0.5}]
    )
    assert _pair_types(storage, a, b) == ["contradicts", "contradicts"]
    assert _pair_types(storage, b, c) == []

    created = await EntityLinker().link_by_entities(b, ["postgres"], graph)

    assert _pair_types(storage, a, b) == ["contradicts", "contradicts"]
    assert _pair_types(storage, b, c) == ["shares_entity", "shares_entity"]
    assert created == 1


@pytest.mark.asyncio
async def test_linker_keeps_one_way_edge_seen_from_its_target(linked):
    """An asymmetric edge stored only as A -> B is protected when linking B."""
    storage, graph = linked
    hashes = []
    for content in (A, B):
        memory = Memory(content=content, content_hash=generate_content_hash(content))
        ok, message = await storage.store(memory, skip_semantic_dedup=True)
        assert ok, message
        hashes.append(memory.content_hash)
    a, b = hashes

    await graph.store_entity_link(a, "postgres", "technology")
    assert await graph.store_association(
        source_hash=a, target_hash=b, similarity=0.9,
        connection_types=["test"], relationship_type="derived_from",
    )
    assert _pair_types(storage, a, b) == ["derived_from"]

    created = await EntityLinker().link_by_entities(b, ["postgres"], graph)

    assert _pair_types(storage, a, b) == ["derived_from"]
    assert created == 0


@pytest.mark.asyncio
async def test_linker_relinks_a_pair_that_already_shares_an_entity(linked):
    """The guard skips only non-entity edges: shares_entity is rewritten."""
    storage, graph = linked
    hashes = []
    for content in (B, C):
        memory = Memory(content=content, content_hash=generate_content_hash(content))
        ok, message = await storage.store(memory, skip_semantic_dedup=True)
        assert ok, message
        hashes.append(memory.content_hash)
    b, c = hashes

    await graph.store_entity_link(c, "postgres", "technology")
    assert await EntityLinker().link_by_entities(b, ["postgres"], graph) == 1
    assert _pair_types(storage, b, c) == ["shares_entity", "shares_entity"]

    created = await EntityLinker().link_by_entities(b, ["postgres"], graph)

    assert _pair_types(storage, b, c) == ["shares_entity", "shares_entity"]
    assert created == 1
