"""Entity rebuild over a multi-store Milvus backend, on Milvus Lite (#2123).

Since #2119 the Milvus graph refuses a ``shares_entity`` pair whose two
memories live in different stores. The rebuild used to plan such pairs, delete
every entity edge and then stop on the first refused write. These cases drive
the real ``rebuild_entities`` tool against a real Milvus Lite memory collection
and association collection; only the embedding is stubbed.

Run: pytest tests/storage/test_milvus_rebuild_store_scope.py -v
"""

import hashlib
import json
import uuid

import numpy as np
import pytest
import pytest_asyncio

pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")

import mcp_memory_service.config as cfg
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server import MemoryServer
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.storage.milvus_graph import MilvusGraphStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class _Encoder:
    """Offline embeddings, independent of the graph behaviour under test."""

    def encode(self, texts, **kwargs):
        return np.array([
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
            .standard_normal(64).astype(np.float32)
            for t in texts
        ])


@pytest_asyncio.fixture
async def backend(tmp_path, monkeypatch):
    async def offline_embeddings(self):
        self.embedding_model = _Encoder()
        self.embedding_dimension = 64

    monkeypatch.setattr(
        MilvusMemoryStorage, "_initialize_embedding_model", offline_embeddings
    )
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "")
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
    storage = MilvusMemoryStorage(
        uri=str(tmp_path / "milvus.db"),
        collection_name=f"rebuild_{uuid.uuid4().hex[:12]}",
        embedding_model="offline",
    )
    await storage.initialize()
    graph = await storage._get_graph_storage()
    try:
        yield storage, graph
    finally:
        await graph.close()
        await storage.close()


async def _store(storage, index, tags, store="default"):
    content = f"plain note number {index}"
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags,
        created_at=1_700_000_000.0 + index,
    )
    ok, message = await storage.store(memory, store=store)
    assert ok, message
    return memory.content_hash


async def _run(storage, **arguments):
    server = MemoryServer(storage=storage)
    result = await server.handle_memory_quality(
        {"action": "rebuild_entities", **arguments}
    )
    return json.loads(result[0].text)


async def _shares_entity_pairs(graph):
    rows = await graph._drain_edges(
        'relationship_type == "shares_entity"', ["source_hash", "target_hash"]
    )
    return {tuple(sorted((r["source_hash"], r["target_hash"]))) for r in rows}


async def test_multi_store_rebuild_skips_cross_store_pairs(backend):
    storage, graph = backend
    a = await _store(storage, 1, ["shared"], store="docs")
    b = await _store(storage, 2, ["shared"], store="docs")
    c = await _store(storage, 3, ["shared"], store="notes")
    assert graph._has_store is True

    preview = await _run(storage)
    assert preview["success"], preview
    assert preview["shares_entity_pairs_after"] == 1
    assert preview["shares_entity_pairs_skipped_cross_store"] == 2
    assert preview["shares_entity_pairs_skipped_unknown_store"] == 0

    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert result["edges_created"] == {"has_entity": 3, "shares_entity": 2}
    assert result["edges_created"] == preview["edges_after"]
    assert result["shares_entity_pairs_skipped_cross_store"] == 2
    assert await _shares_entity_pairs(graph) == {tuple(sorted((a, b)))}
    assert c not in {h for pair in await _shares_entity_pairs(graph) for h in pair}


async def test_preview_of_a_missing_collection_counts_the_skips(backend):
    storage, graph = backend
    await _store(storage, 1, ["shared"], store="docs")
    await _store(storage, 2, ["shared"], store="notes")
    await storage._call_client(
        "drop_collection", collection_name=graph.collection_name
    )

    preview = await _run(storage)
    assert preview["success"], preview
    assert preview["shares_entity_pairs_skipped_cross_store"] == 1
    assert not await storage._call_client(
        "has_collection", collection_name=graph.collection_name
    )
    applied = await _run(storage, dry_run=False)
    assert applied["success"], applied
    assert applied["edges_created"] == preview["edges_after"]


async def test_unreadable_schema_stops_before_deleting(backend, monkeypatch):
    storage, graph = backend
    a = await _store(storage, 1, ["shared"])
    await _store(storage, 2, ["shared"])
    assert await graph.store_entity_link(a, "old", "tag")
    fields = ["source_hash", "target_hash", "relationship_type"]
    before = sorted(map(str, await graph._drain_edges("", fields)))

    def unreadable(self):
        self._has_store = None

    monkeypatch.setattr(MilvusGraphStorage, "_detect_store_field", unreadable)
    for dry_run in (True, False):
        result = await _run(storage, dry_run=dry_run)
        assert result["success"] is False
        assert result["edges_deleted"] == 0
        assert "schema could not be read" in result["errors"][0]
    assert sorted(map(str, await graph._drain_edges("", fields))) == before
