# Copyright 2026 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for #1870: delete-untagged went around the storage layer.

``POST /api/manage/delete-untagged`` used to tombstone untagged rows with a
raw ``UPDATE`` on the shared connection, so graph edges survived the delete,
beliefs and summaries kept citing the deleted hashes (#1865), and delta-sync
peers never saw the deletes. The work now lives in
``DeleteMixin.delete_untagged_memories()``, which is what these tests drive.
"""

import hashlib
import json
import time

import pytest
import pytest_asyncio

from fastapi import HTTPException

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.base import UntaggedCountChanged
from mcp_memory_service.storage.mixins import delete as delete_mixin
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.web.api import manage


@pytest_asyncio.fixture
async def storage(tmp_path):
    s = SqliteVecMemoryStorage(str(tmp_path / "untagged.db"))
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


def _hash(content: str) -> str:
    return hashlib.sha256(content.strip().lower().encode()).hexdigest()


async def _store(storage, content: str, tags):
    """Store a memory, then strip its tags in SQL when it must be untagged.

    ``Memory.__post_init__`` rewrites an empty tag list to ``["untagged"]``,
    so a row with a genuinely empty tag column cannot be created through the
    public API — which is what the endpoint exists to clean up.
    """
    memory = Memory(content=content, content_hash=_hash(content), tags=tags or ["keep"])
    success, message = await storage.store(memory)
    assert success, message
    if not tags:
        storage.conn.execute(
            "UPDATE memories SET tags = '' WHERE content_hash = ?", (memory.content_hash,)
        )
        storage.conn.commit()
    return memory.content_hash


def _deleted_at(storage, content_hash):
    row = storage.conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_count_and_delete_see_the_same_rows(storage):
    untagged = await _store(storage, "no tags here", None)
    tagged = await _store(storage, "this one is tagged", ["keep"])

    assert await storage.count_untagged_memories() == 1

    count, message = await storage.delete_untagged_memories()

    assert count == 1
    assert "1" in message
    assert _deleted_at(storage, untagged) is not None
    assert _deleted_at(storage, tagged) is None
    assert await storage.count_untagged_memories() == 0


@pytest.mark.asyncio
async def test_empty_store_reports_nothing_deleted(storage):
    await _store(storage, "this one is tagged", ["keep"])

    count, message = await storage.delete_untagged_memories()

    assert count == 0
    assert message == "No untagged memories found"


@pytest.mark.asyncio
async def test_graph_edges_go_with_the_memory(storage):
    untagged = await _store(storage, "no tags here", None)
    tagged = await _store(storage, "this one is tagged", ["keep"])
    other = await _store(storage, "another tagged one", ["keep"])
    for source, target in ((untagged, tagged), (tagged, untagged), (tagged, other)):
        storage.conn.execute(
            "INSERT INTO memory_graph (source_hash, target_hash, similarity, "
            "connection_types, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (source, target, 0.9, json.dumps(["semantic"]), None, time.time()),
        )
    storage.conn.commit()

    await storage.delete_untagged_memories()

    rows = storage.conn.execute(
        "SELECT source_hash, target_hash FROM memory_graph"
    ).fetchall()
    assert rows == [(tagged, other)]


@pytest.mark.asyncio
async def test_citations_are_retracted_before_the_commit(storage, monkeypatch):
    untagged = await _store(storage, "no tags here", None)
    seen = {}

    def _spy(conn, hashes, sync=None):
        seen["hashes"] = list(hashes)
        seen["in_transaction"] = conn.in_transaction

    monkeypatch.setattr(delete_mixin, "retract_sources_for_deleted_hashes", _spy)

    await storage.delete_untagged_memories()

    assert seen["hashes"] == [untagged]
    # Retraction and tombstone have to land in the same transaction, or a
    # failed retraction leaves a memory deleted with its beliefs still active.
    assert seen["in_transaction"] is True


@pytest.mark.asyncio
async def test_one_delete_event_per_deleted_hash(storage, monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    first = await _store(storage, "no tags here", None)
    second = await _store(storage, "none here either", None)

    count, _message = await storage.delete_untagged_memories()
    assert count == 2

    rows = storage.conn.execute(
        "SELECT content_hash FROM sync_events WHERE op = 'delete'"
    ).fetchall()
    assert sorted(row[0] for row in rows) == sorted([first, second])


@pytest.mark.asyncio
async def test_a_failed_retraction_undoes_the_tombstone(storage, monkeypatch):
    untagged = await _store(storage, "no tags here", None)

    def _boom(conn, hashes, sync=None):
        raise RuntimeError("retraction failed")

    monkeypatch.setattr(delete_mixin, "retract_sources_for_deleted_hashes", _boom)

    count, message = await storage.delete_untagged_memories()

    assert count == 0
    assert "retraction failed" in message
    assert _deleted_at(storage, untagged) is None


@pytest.mark.asyncio
async def test_a_concurrent_untag_does_not_widen_the_delete(storage):
    """A confirmed count is re-checked against the rows that are about to go.

    The caller counts in one request and deletes in the next. Between the two
    another request can strip the last tag off a memory, and the delete would
    then take a row the user never confirmed.
    """
    first = await _store(storage, "no tags here", None)
    confirmed = await storage.count_untagged_memories()
    assert confirmed == 1

    # The concurrent write: a second memory loses its last tag.
    second = await _store(storage, "this one is tagged", ["keep"])
    storage.conn.execute(
        "UPDATE memories SET tags = '' WHERE content_hash = ?", (second,)
    )
    storage.conn.commit()

    with pytest.raises(UntaggedCountChanged) as caught:
        await storage.delete_untagged_memories(expected_count=confirmed)

    assert (caught.value.expected, caught.value.actual) == (1, 2)
    # Nothing was written: neither row is tombstoned.
    assert _deleted_at(storage, first) is None
    assert _deleted_at(storage, second) is None


@pytest.mark.asyncio
async def test_a_matching_count_still_deletes(storage):
    untagged = await _store(storage, "no tags here", None)

    count, _message = await storage.delete_untagged_memories(expected_count=1)

    assert count == 1
    assert _deleted_at(storage, untagged) is not None


@pytest.mark.asyncio
async def test_the_endpoint_answers_400_when_the_set_moves(storage, monkeypatch):
    """End to end: the widened delete is refused with the stale-count 400."""
    first = await _store(storage, "no tags here", None)
    second = await _store(storage, "this one is tagged", ["keep"])

    real_count = storage.count_untagged_memories

    async def _count_then_untag_another():
        count = await real_count()
        # Runs where another request would: after the endpoint has read the
        # count it holds the caller to, before the delete.
        storage.conn.execute(
            "UPDATE memories SET tags = '' WHERE content_hash = ?", (second,)
        )
        storage.conn.commit()
        return count

    monkeypatch.setattr(storage, "count_untagged_memories", _count_then_untag_another)

    with pytest.raises(HTTPException) as caught:
        await manage.delete_untagged_memories(
            confirm_count=1, storage=storage, user=None
        )

    assert caught.value.status_code == 400
    assert "Confirmation count mismatch" in caught.value.detail
    assert _deleted_at(storage, first) is None
    assert _deleted_at(storage, second) is None


class _Primary:
    async def count_untagged_memories(self):
        return 0

    async def delete_untagged_memories(self, expected_count=None):
        return 0, "primary"


class _Hybrid:
    def __init__(self):
        self.primary = _Primary()


def test_backend_resolver_prefers_the_storage_itself():
    primary = _Primary()
    assert manage._untagged_backend(primary) is primary


def test_backend_resolver_reaches_the_hybrid_primary():
    hybrid = _Hybrid()
    # Hybrid storage does not re-export the bulk deletes, and the endpoint has
    # to land on the storage-layer method rather than the raw connection.
    assert manage._untagged_backend(hybrid) is hybrid.primary


def test_backend_resolver_reports_unsupported_backends():
    assert manage._untagged_backend(object()) is None
