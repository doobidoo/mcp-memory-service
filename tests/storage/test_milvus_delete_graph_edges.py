"""Milvus delete must remove the memory's graph edges (#1703).

SQLite cascades graph edges on delete since #633. On Milvus the edges in
``<collection>_graph`` survived, so with entity linking enabled a new memory
sharing an entity was linked to the deleted one.

Runs against a real Milvus Lite file; only the embedding is stubbed.

Run: pytest tests/storage/test_milvus_delete_graph_edges.py -v
"""

import hashlib
import time
import uuid

import numpy as np
import pytest
import pytest_asyncio

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from mcp_memory_service.services.memory_service import MemoryService  # noqa: E402
from mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402
import mcp_memory_service.server.handlers.graph as graph_handlers  # noqa: E402

_DIM = 64


class _StubEncoder:
    """Deterministic, near-orthogonal vectors, so semantic dedup never merges the test memories."""

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        return np.array([
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
            .standard_normal(_DIM).astype(np.float32)
            for t in texts
        ])


@pytest_asyncio.fixture
async def service(tmp_path, monkeypatch):
    async def _stub_init_embedding(self):
        self.embedding_model = _StubEncoder()
        self.embedding_dimension = _DIM

    monkeypatch.setattr(MilvusMemoryStorage, "_initialize_embedding_model", _stub_init_embedding)
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
    storage = MilvusMemoryStorage(
        uri=str(tmp_path / "milvus.db"),
        collection_name=f"mcp_delgraph_{uuid.uuid4().hex[:12]}",
        embedding_model="stub",
    )
    await storage.initialize()

    async def _graph():
        return await storage._get_graph_storage()

    monkeypatch.setattr(graph_handlers, "get_graph_storage", _graph)
    try:
        yield MemoryService(storage), storage
    finally:
        await storage.close()


async def _edges_touching(graph, content_hash):
    return await graph._drain_edges(
        f'source_hash == "{content_hash}" or target_hash == "{content_hash}"',
        ["source_hash", "target_hash", "relationship_type"],
    )


async def test_delete_removes_graph_edges(service):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-x", tags=["shared-x"])
    b = await svc.store_memory("beta note about module-x", tags=["shared-x"])
    assert a["success"] and b["success"], (a, b)
    hash_a = a["memory"]["content_hash"]
    assert any(e["relationship_type"] == "shares_entity" for e in await _edges_touching(graph, hash_a))

    result = await svc.delete_memory(hash_a)
    assert result["success"], result
    assert await storage.get_by_hash(hash_a) is None

    assert await _edges_touching(graph, hash_a) == []


async def test_new_memory_is_not_linked_to_deleted(service):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-y", tags=["shared-y"])
    b = await svc.store_memory("beta note about module-y", tags=["shared-y"])
    assert a["success"] and b["success"], (a, b)
    hash_a = a["memory"]["content_hash"]
    hash_b = b["memory"]["content_hash"]
    assert (await svc.delete_memory(hash_a))["success"]

    c = await svc.store_memory("gamma note about module-y", tags=["shared-y"])
    assert c["success"], c
    hash_c = c["memory"]["content_hash"]
    linked = {
        e["target_hash"] if e["source_hash"] == hash_c else e["source_hash"]
        for e in await _edges_touching(graph, hash_c)
        if e["relationship_type"] == "shares_entity"
    }
    assert linked == {hash_b}


async def test_delete_by_tags_removes_graph_edges(service):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-z", tags=["shared-z", "drop-me"])
    b = await svc.store_memory("beta note about module-z", tags=["shared-z", "drop-me"])
    k = await svc.store_memory("kappa note about module-z", tags=["shared-z"])
    assert a["success"] and b["success"] and k["success"], (a, b, k)
    hash_a, hash_b, hash_k = (r["memory"]["content_hash"] for r in (a, b, k))
    assert await _edges_touching(graph, hash_a)

    count, _, deleted = await storage.delete_by_tags(["drop-me"])
    assert count == 2 and set(deleted) == {hash_a, hash_b}

    assert await _edges_touching(graph, hash_a) == []
    assert await _edges_touching(graph, hash_b) == []
    assert any(e["relationship_type"] == "has_entity" for e in await _edges_touching(graph, hash_k))


@pytest.mark.parametrize(
    "bulk_delete",
    [lambda s: s.delete_by_tags(["drop-p"]), lambda s: s.delete_by_tag("drop-p")],
    ids=["delete_by_tags", "delete_by_tag"],
)
async def test_partial_bulk_delete_cleans_only_deleted_edges(service, monkeypatch, bulk_delete):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-p", tags=["shared-p", "drop-p"])
    b = await svc.store_memory("beta note about module-p", tags=["shared-p", "drop-p"])
    assert a["success"] and b["success"], (a, b)
    hashes = {a["memory"]["content_hash"], b["memory"]["content_hash"]}

    # One hash per chunk; the second memory-collection delete fails.
    monkeypatch.setattr(storage, "_GET_BY_ID_CHUNK", 1)
    original_call = storage._call_client
    deletes = 0

    async def _fail_second_delete(method_name, *args, **kwargs):
        nonlocal deletes
        if method_name == "delete" and kwargs.get("collection_name") == storage.collection_name:
            deletes += 1
            if deletes == 2:
                raise RuntimeError("injected failure")
        return await original_call(method_name, *args, **kwargs)

    monkeypatch.setattr(storage, "_call_client", _fail_second_delete)
    with pytest.raises(RuntimeError, match="injected failure"):
        await bulk_delete(storage)

    deleted = {h for h in hashes if await storage.get_by_hash(h) is None}
    assert len(deleted) == 1, deleted
    (gone,), (kept,) = deleted, hashes - deleted
    assert await _edges_touching(graph, gone) == []
    assert any(e["relationship_type"] == "has_entity" for e in await _edges_touching(graph, kept))


async def test_delete_succeeds_when_graph_cleanup_fails(service, monkeypatch):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-w", tags=["shared-w"])
    assert a["success"], a
    hash_a = a["memory"]["content_hash"]

    async def _fail(_hashes):
        raise RuntimeError("graph collection unavailable")

    monkeypatch.setattr(graph, "delete_edges_for_memories", _fail)
    assert await storage.delete(hash_a) == (True, f"Successfully deleted memory {hash_a}")
    assert await storage.get_by_hash(hash_a) is None


async def test_purge_deleted_removes_graph_edges(service):
    svc, storage = service
    graph = await storage._get_graph_storage()

    a = await svc.store_memory("alpha note about module-q", tags=["shared-q"])
    b = await svc.store_memory("beta note about module-q", tags=["shared-q"])
    assert a["success"] and b["success"], (a, b)
    hash_a, hash_b = a["memory"]["content_hash"], b["memory"]["content_hash"]
    assert await _edges_touching(graph, hash_a)

    # A tombstone exists on Milvus only when deleted_at is written explicitly (#1708).
    ok, msg = await storage.update_memory_metadata(hash_a, {"metadata": {"deleted_at": time.time()}})
    assert ok, msg
    assert await storage.purge_deleted(older_than_days=0) == 1

    assert await _edges_touching(graph, hash_a) == []
    assert any(e["relationship_type"] == "has_entity" for e in await _edges_touching(graph, hash_b))
