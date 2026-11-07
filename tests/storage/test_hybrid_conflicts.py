"""Conflict API on the hybrid backend (#1910).

Hybrid writes through its sqlite-vec primary, and the primary's store path
detects conflicts: it tags both memories conflict:unresolved and links them
with 'contradicts' edges. Listing, resolving and reconciling those conflicts
has to reach the same primary, through the storage methods, the MCP handlers,
the REST routes and the maintain run.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.consolidation import contradictions
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers import quality
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.api import conflicts as conflicts_api
from mcp_memory_service.web.dependencies import get_storage

TAG = "conflict:unresolved"
BACKUP_NAS = "Nightly backups are written to the office NAS at 02:00"
BACKUP_S3 = "Backups go to an S3 bucket every evening around 23:30 UTC"
UNRELATED = "The team standup moved to Tuesdays and Thursdays"


def _unit(dimension, axis, tilt_axis=None, degrees=0.0):
    vector = np.zeros(dimension)
    vector[axis] = np.cos(np.radians(degrees))
    if tilt_axis is not None:
        vector[tilt_axis] = np.sin(np.radians(degrees))
    return vector.tolist()


@pytest_asyncio.fixture
async def storage(temp_db_path, monkeypatch):
    """Hybrid with no secondary configured: everything runs on the local primary."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    hybrid = HybridMemoryStorage(os.path.join(temp_db_path, "hybrid-1910.db"))
    await hybrid.initialize()
    dimension = hybrid.primary.embedding_dimension
    # The two backup texts sit 5 degrees apart (cosine ~0.996), well inside the
    # detection threshold; the unrelated text is orthogonal to both.
    vectors = {
        BACKUP_NAS: _unit(dimension, 3),
        BACKUP_S3: _unit(dimension, 3, tilt_axis=7, degrees=5),
        UNRELATED: _unit(dimension, 11),
    }
    monkeypatch.setattr(hybrid.primary, "_generate_embedding", vectors.__getitem__)
    yield hybrid
    await hybrid.close()


async def _remember(hybrid, content, *extra_tags):
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["hybrid-1910", *extra_tags],
    )
    ok, message = await hybrid.store(memory, skip_semantic_dedup=True)
    assert ok, message
    return memory.content_hash


async def _has_tag(hybrid, content_hash):
    memory = await hybrid.primary.get_by_hash(content_hash)
    return TAG in memory.tags


@pytest_asyncio.fixture
async def backup_conflict(storage):
    nas = await _remember(storage, BACKUP_NAS)
    s3 = await _remember(storage, BACKUP_S3, "offsite")
    # Detection really ran in the primary during hybrid.store().
    assert await _has_tag(storage, nas) and await _has_tag(storage, s3)
    return nas, s3


@pytest.mark.asyncio
async def test_lists_the_same_conflicts_as_the_primary(storage, backup_conflict):
    await _remember(storage, UNRELATED)

    listed = await storage.get_conflicts()

    assert listed == await storage.primary.get_conflicts()
    assert [{c["hash_a"], c["hash_b"]} for c in listed] == [set(backup_conflict)]


@pytest.mark.asyncio
async def test_lists_nothing_when_memories_do_not_conflict(storage):
    await _remember(storage, BACKUP_NAS)
    await _remember(storage, UNRELATED)

    assert await storage.get_conflicts() == []


@pytest.mark.asyncio
async def test_resolve_supersedes_loser_without_queueing_a_sync(
    storage, backup_conflict
):
    nas, s3 = backup_conflict
    sync = MagicMock(enqueue_operation=AsyncMock())
    storage.sync_service = sync
    try:
        ok, message = await storage.resolve_conflict(s3, nas)
    finally:
        storage.sync_service = None

    assert ok, message
    assert await storage.get_conflicts() == []
    assert not await _has_tag(storage, s3)
    loser = storage.primary.conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (nas,)
    ).fetchone()
    assert loser[0] == s3
    sync.enqueue_operation.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_failure_comes_from_the_primary(storage, backup_conflict):
    nas, _ = backup_conflict

    ok, message = await storage.resolve_conflict(nas, nas)

    assert not ok
    assert "not supported" not in message
    assert len(await storage.get_conflicts()) == 1


@pytest.mark.asyncio
async def test_reconcile_clears_tag_left_by_bulk_tag_delete(storage, backup_conflict):
    nas, _ = backup_conflict
    # delete_by_tag drops the edges but leaves the surviving partner tagged.
    deleted, message = await storage.delete_by_tag("offsite")
    assert deleted == 1, message
    assert await _has_tag(storage, nas)

    preview = await storage.reconcile_conflict_tags()
    assert preview["supported"] is True
    assert (preview["found"], preview["cleared"]) == (1, 0)
    assert await _has_tag(storage, nas)

    applied = await storage.reconcile_conflict_tags(dry_run=False)
    assert (applied["found"], applied["cleared"]) == (1, 1)
    assert not await _has_tag(storage, nas)


@pytest.mark.asyncio
async def test_mcp_handlers_list_and_resolve(storage, backup_conflict):
    nas, s3 = backup_conflict
    server = MagicMock(storage=storage, _ensure_storage_initialized=AsyncMock())

    listed = (await quality.handle_memory_conflicts(server, {}))[0].text
    assert listed.startswith("Found 1 conflict(s)"), listed

    resolved = (
        await quality.handle_memory_resolve(
            server, {"hashes": [{"winner_hash": nas, "loser_hash": s3}]}
        )
    )[0].text
    assert resolved.startswith("Resolved 1 conflict(s)"), resolved
    assert "not supported" not in resolved

    listed = (await quality.handle_memory_conflicts(server, {}))[0].text
    assert listed == "No unresolved conflicts found."


@pytest.mark.asyncio
async def test_rest_routes_list_and_resolve(storage, backup_conflict):
    nas, s3 = backup_conflict
    app = FastAPI()
    app.include_router(conflicts_api.router, prefix="/api")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[conflicts_api.require_read_access] = lambda: None
    app.dependency_overrides[conflicts_api.require_write_access] = lambda: None
    client = TestClient(app)

    listing = client.get("/api/conflicts")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1

    resolved = client.post(
        "/api/conflicts/resolve", json={"winner_hash": nas, "loser_hash": s3}
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["success"] is True

    assert client.get("/api/conflicts").json()["count"] == 0


@pytest.mark.asyncio
async def test_maintain_counts_and_reconciles_hybrid_conflicts(
    storage, backup_conflict, monkeypatch
):
    monkeypatch.setattr(quality, "MAINTAIN_AUTO_RESOLVE", False)
    monkeypatch.setattr(contradictions, "CONTRADICTION_ENABLED", False)
    server = MagicMock(_ensure_storage_initialized=AsyncMock(return_value=storage))
    graph = MagicMock(store_entity_link=AsyncMock(return_value=True))

    with patch(
        "mcp_memory_service.server.handlers.graph.get_graph_storage",
        AsyncMock(return_value=graph),
    ):
        report = json.loads((await quality.handle_maintain(server, {}))[0].text)

    step = report["steps"]["conflicts"]
    assert step["total"] == 1
    assert step["stale_tags"]["supported"] is True
    assert step["stale_tags"]["found"] == 0
