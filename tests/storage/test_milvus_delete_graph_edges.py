"""Milvus delete must remove the memory's graph edges (#1703).

SQLite cascades graph edges on delete since #633. On Milvus the edges in
``<collection>_graph`` survived, so with entity linking enabled a new memory
sharing an entity was linked to the deleted one.

Runs against a real Milvus Lite file; only the embedding is stubbed.

Run: pytest tests/storage/test_milvus_delete_graph_edges.py -v
"""

import hashlib
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
    assert a["success"], a
    hash_a = a["memory"]["content_hash"]
    assert (await svc.delete_memory(hash_a))["success"]

    c = await svc.store_memory("gamma note about module-y", tags=["shared-y"])
    assert c["success"], c
    to_deleted = [
        e for e in await _edges_touching(graph, c["memory"]["content_hash"])
        if hash_a in (e["source_hash"], e["target_hash"])
    ]
    assert to_deleted == []
