"""Entity rebuilds use the real tool, extractor, linker, and graph (#1722)."""

import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

import mcp_memory_service.config as cfg
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers import graph as graph_handlers
from mcp_memory_service.server_impl import MemoryServer
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.tools.registry import TOOL_REGISTRY
from mcp_memory_service.utils.hashing import generate_content_hash


class _Encoder:
    """Offline embeddings, independent of the graph behavior under test."""

    def encode(self, texts, **kwargs):
        return np.array([
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
            .standard_normal(64).astype(np.float32)
            for t in texts
        ])


@pytest.fixture(params=["sqlite_vec", "hybrid", "milvus"])
async def backend(request, temp_db_path, monkeypatch):
    """Use real isolated storage and graph implementations for every backend."""
    path = f"{temp_db_path}/rebuild.db"
    if request.param == "hybrid":
        storage = HybridMemoryStorage(path)
        primary = storage.primary
    elif request.param == "milvus":
        pytest.importorskip("pymilvus")
        pytest.importorskip("milvus_lite")

        async def offline_embeddings(self):
            self.embedding_model = _Encoder()
            self.embedding_dimension = 64

        monkeypatch.setattr(
            MilvusMemoryStorage, "_initialize_embedding_model", offline_embeddings
        )
        storage = primary = MilvusMemoryStorage(
            uri=path, collection_name="rebuild_test", embedding_model="offline"
        )
    else:
        storage = primary = SqliteVecMemoryStorage(path)
    await primary.initialize()
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "")
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
    graph = (
        await primary._get_graph_storage()
        if request.param == "milvus" else GraphStorage(path)
    )

    async def service_graph():
        return graph

    monkeypatch.setattr(graph_handlers, "get_graph_storage", service_graph)
    try:
        yield storage, primary, graph
    finally:
        await graph.close()
        await primary.close()


async def _run(storage, **arguments):
    server = MemoryServer(storage=storage)
    result = await server.handle_memory_quality(
        {"action": "rebuild_entities", **arguments}
    )
    return json.loads(result[0].text)


async def _rows(graph):
    """Read all persisted edge fields to detect unintended mutations."""
    fields = [
        "source_hash", "target_hash", "relationship_type", "similarity",
        "connection_types", "metadata", "created_at",
    ]
    if not isinstance(graph, GraphStorage):
        rows = await graph._drain_edges("", fields)
        return sorted(tuple(row[f] for f in fields) for row in rows)
    conn = await graph._get_connection()
    return [tuple(row) for row in conn.execute(
        "SELECT " + ", ".join(fields) +
        " FROM memory_graph ORDER BY source_hash, target_hash"
    )]


async def _store(primary, index, tags, store="default", content=None, metadata=None):
    content = content or f"plain note number {index}"
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags,
        metadata=metadata or {},
        created_at=1_700_000_000.0 + index,
    )
    ok, message = await primary.store(memory, store=store)
    assert ok, message
    return memory


async def test_rebuild_filters_old_hubs_and_preview_changes_nothing(
    backend, monkeypatch
):
    storage, primary, graph = backend
    for i in range(10):
        task = 0 if i < 5 else 1 if i < 8 else i - 6
        tags = [f"task:{task}"]
        if i < 9:
            tags.append("bugfix")
        if i < 5 or i == 9:
            tags.append("project:hub")
        memory = await _store(primary, i, tags)
        await MemoryService(primary)._maybe_link_entities(memory)
    before = await _rows(graph)
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "task:*")
    report = await _run(storage)
    assert report["success"], report
    assert report["dry_run"] is True
    assert report["shares_entity_pairs_before"] == 41
    assert report["shares_entity_pairs_after"] == 13
    assert await _rows(graph) == before

    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert result["edges_created"] == report["edges_after"]
    assert result["edges_before"] == report["edges_before"]
    assert {e["entity_name"] for e in await graph.list_entities()} == {
        "task:0", "task:1", "task:2", "task:3"
    }
    assert await graph.find_memories_by_entity("bugfix") == []
    assert len(await graph.find_memories_by_entity("task:2")) == 1


async def test_rebuild_covers_all_stores_and_deleted_rows(backend, monkeypatch):
    storage, primary, graph = backend
    monkeypatch.setattr(cfg, "MAINTAIN_SCAN_LIMIT", 1)
    a = await _store(primary, 1, ["task:shared"])
    b = await _store(primary, 2, ["task:shared"], store="docs")
    assert await primary.mark_superseded_batch([
        (a.content_hash, b.content_hash)
    ]) == 1
    gone = await _store(primary, 3, ["task:deleted"])
    assert (await primary.delete(gone.content_hash))[0]
    assert await graph.store_entity_link(gone.content_hash, "stale", "tag")
    if isinstance(primary, MilvusMemoryStorage):
        tombstone = await _store(primary, 4, ["task:tombstone"])
        assert (await primary.update_memory_metadata(
            tombstone.content_hash, {"metadata": {"deleted_at": time.time()}}
        ))[0]
        assert await graph.store_entity_link(
            tombstone.content_hash, "tombstone", "tag"
        )
    snapshots = [await primary.get_by_hash(m.content_hash) for m in (a, b)]

    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert result["memories_scanned"] == 2
    assert result["edges_after"] == {"has_entity": 2, "shares_entity": 2}
    assert await graph.get_entities_for_memory(gone.content_hash) == []
    assert await graph.find_memories_by_entity("tombstone") == []
    assert await graph.has_edge(a.content_hash, b.content_hash, "shares_entity")
    assert [await primary.get_by_hash(m.content_hash) for m in (a, b)] == snapshots


async def test_preserves_other_edges_even_on_a_shared_entity_pair(backend):
    storage, primary, graph = backend
    a = await _store(primary, 1, ["shared"])
    b = await _store(primary, 2, ["shared"])
    c = await _store(primary, 3, ["shared"])
    assert await graph.store_association(
        a.content_hash, b.content_hash, 0.8, ["manual"],
        {"reason": "keep"}, created_at=123.0, relationship_type="causes"
    )
    assert await graph.store_association(
        a.content_hash, c.content_hash, 0.9, ["semantic"],
        {"reason": "also keep"}, created_at=124.0, relationship_type="contradicts"
    )
    preserved = await _rows(graph)
    preview = await _run(storage)
    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert result["edges_created"] == preview["edges_after"]
    assert result["shares_entity_pairs_after"] == 1
    assert set(preserved).issubset(await _rows(graph))
    assert await graph.has_edge(b.content_hash, c.content_hash, "shares_entity")


async def test_creation_order_and_link_cap_match_store_time(backend):
    storage, primary, graph = backend
    memories = []
    for i in range(24):
        memory = await _store(primary, i, ["shared"])
        memories.append(memory)
        await MemoryService(primary)._maybe_link_entities(memory)
    expected = {row[:3] for row in await _rows(graph)}
    result = await _run(storage, dry_run=False)
    assert result["success"], result
    actual = {row[:3] for row in await _rows(graph)}
    assert actual == expected
    assert not await graph.has_edge(
        memories[-1].content_hash, memories[0].content_hash, "shares_entity"
    )


async def test_empty_store_and_invalid_dry_run(backend):
    storage, _, graph = backend
    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert result["memories_scanned"] == 0
    assert result["edges_created"] == {"has_entity": 0, "shares_entity": 0}
    invalid = await _run(storage, dry_run="false")
    assert invalid["success"] is False
    assert await _rows(graph) == []


async def test_current_custom_terms_and_legacy_metadata_tags(backend, monkeypatch):
    storage, primary, graph = backend
    memory = await _store(
        primary, 1, ["task:kept", "task:skip", "bugfix"],
        content="a mention of Zephyr and #contenttag",
        metadata={"tags": "task:legacy,bugfix"},
    )
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "false")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "task:*")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "task:skip")
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "Zephyr")
    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert set(await graph.get_entities_for_memory(memory.content_hash)) == {
        "task:kept", "task:legacy", "Zephyr", "contenttag",
    }


async def test_extraction_failure_does_not_delete_old_edges(backend, monkeypatch):
    storage, primary, graph = backend
    memory = await _store(primary, 1, ["old"])
    assert await graph.store_entity_link(memory.content_hash, "old", "tag")
    before = await _rows(graph)
    # Invalid configuration fails during the real extractor's config parsing.
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", 42)
    result = await _run(storage, dry_run=False)
    assert result["success"] is False
    assert result["errors"]
    assert result["edges_deleted"] == 0
    assert await _rows(graph) == before


async def test_failed_write_reports_partial_rebuild_and_can_be_retried(backend):
    storage, primary, graph = backend
    if not isinstance(graph, GraphStorage):
        pytest.skip("SQLite trigger exercises a real storage write failure")
    a = await _store(primary, 1, ["shared"])
    await _store(primary, 2, ["shared"])
    assert await graph.store_entity_link(a.content_hash, "old", "tag")
    conn = await graph._get_connection()
    conn.execute("""
        CREATE TRIGGER fail_entity_write BEFORE INSERT ON memory_graph
        WHEN NEW.relationship_type = 'shares_entity'
        BEGIN SELECT RAISE(FAIL, 'write failure'); END
    """)
    conn.commit()
    result = await _run(storage, dry_run=False)
    assert result["success"] is False
    assert "incomplete" in result["errors"][0]
    assert result["edges_deleted"] == 1
    assert result["edges_created"] == {"has_entity": 2, "shares_entity": 0}
    conn.execute("DROP TRIGGER fail_entity_write")
    conn.commit()
    retry = await _run(storage, dry_run=False)
    assert retry["success"], retry
    assert retry["edges_created"] == {"has_entity": 2, "shares_entity": 2}


async def test_no_graph_backend_is_reported():
    # Construction is local; no Cloudflare initialization or request is made.
    storage = CloudflareStorage("test", "test", "test", "test")
    result = await _run(storage, dry_run=False)
    assert result["success"] is False
    assert "graph" in result["errors"][0].lower()


@pytest.mark.parametrize("backend", ["sqlite_vec", "hybrid"], indirect=True)
async def test_uses_the_servers_database_not_the_global_path(
    backend, tmp_path, monkeypatch
):
    storage, primary, graph = backend
    memory = await _store(primary, 1, ["actual"])
    other = SqliteVecMemoryStorage(str(tmp_path / "other.db"))
    await other.initialize()
    other_graph = GraphStorage(other.db_path)
    try:
        foreign = await _store(other, 2, ["foreign"])
        assert await other_graph.store_entity_link(
            foreign.content_hash, "foreign", "tag"
        )
        before = await _rows(other_graph)
        monkeypatch.setattr(graph_handlers, "SQLITE_VEC_PATH", other.db_path)
        result = await _run(storage, dry_run=False)
        assert result["success"], result
        assert await graph.get_entities_for_memory(memory.content_hash) == ["actual"]
        assert await _rows(other_graph) == before
    finally:
        await other_graph.close()
        await other.close()


@pytest.mark.parametrize("backend", ["milvus"], indirect=True)
async def test_preview_does_not_create_a_missing_graph_collection(backend):
    storage, primary, graph = backend
    await _store(primary, 1, ["shared"])
    await primary._call_client("drop_collection", collection_name=graph.collection_name)
    preview = await _run(storage)
    assert preview["success"], preview
    assert preview["edges_before"] == {"has_entity": 0, "shares_entity": 0}
    assert preview["edges_after"] == {"has_entity": 1, "shares_entity": 0}
    assert not await primary._call_client(
        "has_collection", collection_name=graph.collection_name
    )
    applied = await _run(storage, dry_run=False)
    assert applied["success"], applied
    assert await graph.get_entities_for_memory(
        (await primary.get_all_memories())[0].content_hash
    ) == ["shared"]


@pytest.mark.parametrize("backend", ["sqlite_vec", "hybrid"], indirect=True)
async def test_scan_errors_cannot_be_mistaken_for_an_empty_store(backend):
    storage, primary, graph = backend
    memory = await _store(primary, 1, ["old"])
    assert await graph.store_entity_link(memory.content_hash, "old", "tag")
    before = await _rows(graph)

    def deny_memory_reads(action, table, column, database, source):
        if action == sqlite3.SQLITE_READ and table == "memories":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    primary.conn.set_authorizer(deny_memory_reads)
    try:
        # Ordinary listing keeps its old fallback behavior.
        assert await primary.get_all_memories() == []
        assert await primary.count_all_memories() == 0
        result = await _run(storage, dry_run=False)
        assert await primary.get_all_memories() == []
        assert await primary.count_all_memories() == 0
    finally:
        primary.conn.set_authorizer(None)
    assert result["success"] is False
    assert result["edges_deleted"] == 0
    assert await _rows(graph) == before


async def test_entity_surface_form_matches_store_time(backend):
    storage, primary, graph = backend
    memory = await _store(
        primary, 1, [], content="a mention of #Foo",
        metadata={"tags": [" Foo "]},
    )
    await MemoryService(primary)._maybe_link_entities(memory)
    before = set(await graph.get_entities_for_memory(memory.content_hash))
    result = await _run(storage, dry_run=False)
    assert result["success"], result
    assert set(await graph.get_entities_for_memory(memory.content_hash)) == before


@pytest.mark.parametrize("backend", ["milvus"], indirect=True)
async def test_milvus_count_rpc_failure_stops_before_deletion(backend, monkeypatch):
    storage, primary, graph = backend
    memory = await _store(primary, 1, ["old"])
    assert await graph.store_entity_link(memory.content_hash, "old", "tag")
    before = await _rows(graph)
    query = primary.client.query

    def count_rpc_failure(*args, **kwargs):
        if kwargs.get("output_fields") == ["count(*)"]:
            raise RuntimeError("count RPC unavailable")
        return query(*args, **kwargs)

    monkeypatch.setattr(primary.client, "query", count_rpc_failure)
    assert await primary.count_all_memories() == 0
    result = await _run(storage, dry_run=False)
    assert result["success"] is False
    assert result["edges_deleted"] == 0
    assert "count RPC unavailable" in result["errors"][0]
    assert await primary.count_all_memories() == 0
    assert await _rows(graph) == before


async def test_two_rebuilds_cannot_interleave(backend):
    storage, primary, graph = backend
    await _store(primary, 1, ["shared"])
    await _store(primary, 2, ["shared"])
    results = await asyncio.gather(
        _run(storage, dry_run=False), _run(storage, dry_run=False)
    )
    assert sum(r["success"] for r in results) == 1
    failure = next(r for r in results if not r["success"])
    assert "already running" in failure["errors"][0]
    assert len(await _rows(graph)) == 4


@pytest.mark.parametrize("backend", ["sqlite_vec"], indirect=True)
@pytest.mark.parametrize("scope", ["read", "write:default", "read write"])
@pytest.mark.parametrize("dry_run", [True, False])
async def test_http_requires_global_write_scope(
    backend, monkeypatch, scope, dry_run
):
    from mcp_memory_service.web.api import mcp as transport  # inline import
    from mcp_memory_service.web.oauth.middleware import (  # inline import
        AuthenticationResult, get_current_user,
    )

    storage, primary, graph = backend
    await _store(primary, 1, ["shared"])
    await _store(primary, 2, ["shared"], store="docs")
    before = await _rows(graph)
    monkeypatch.setattr(transport, "_memory_server", MemoryServer(storage=storage))
    app = FastAPI()
    app.include_router(transport.router)

    async def caller():
        return AuthenticationResult(True, scope=scope, auth_method="test")

    app.dependency_overrides[get_current_user] = caller
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "memory_quality",
                "arguments": {
                    "action": "rebuild_entities", "dry_run": dry_run,
                    "store": "default",
                },
            },
        })
    if scope != "read write":
        assert response.status_code == 403, response.text
        assert await _rows(graph) == before
    else:
        assert response.status_code == 200, response.text
        report = json.loads(response.json()["result"]["content"][0]["text"])
        assert report["success"], report
        assert report["memories_scanned"] == 2


def test_rebuild_is_advertised_as_a_destructive_write_action():
    tool = next(t for t in TOOL_REGISTRY if t.name == "memory_quality")
    assert "rebuild_entities" in tool.input_schema["properties"]["action"]["enum"]
    assert not tool.annotations.get("readOnlyHint", False)
    assert tool.annotations["destructiveHint"] is True


def test_ordinary_quality_import_does_not_load_milvus():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import mcp_memory_service.server.handlers.quality; "
            "assert 'mcp_memory_service.storage.milvus' not in sys.modules"
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
