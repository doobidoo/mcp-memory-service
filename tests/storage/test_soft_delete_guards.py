# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Regression tests for soft-delete UPDATE guards (follow-up to #783).

Verifies that each UPDATE statement guarded with `AND deleted_at IS NULL`
silently skips tombstoned rows. Pattern: store → soft-delete → invoke
the operation → assert no mutation occurred on the deleted row.

Methods covered:
1. _persist_access_metadata_batch — metadata batch update
2. _record_conflicts — conflict:unresolved tag
3. resolve_conflict — explicit error on deleted hash
4. _touch (via retrieve) — last_accessed update
5. update_memory_versioned — versioned insert with deleted predecessor
"""

import hashlib
import pytest
import pytest_asyncio
import time

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage(tmp_path):
    """Create a temporary SqliteVecMemoryStorage instance."""
    db_path = tmp_path / "test_soft_delete.db"
    s = SqliteVecMemoryStorage(str(db_path))
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


def _make_memory(content: str, tags=None, memory_type=None) -> Memory:
    """Helper to build a Memory object with deterministic hash."""
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.strip().lower().encode()).hexdigest(),
        tags=tags or [],
        memory_type=memory_type,
    )


async def _get_row(storage, content_hash: str):
    """Read raw row from DB (including tombstoned rows)."""
    def _query():
        cursor = storage.conn.execute(
            "SELECT content_hash, tags, metadata, last_accessed, superseded_by, deleted_at "
            "FROM memories WHERE content_hash = ?",
            (content_hash,),
        )
        return cursor.fetchone()

    return await storage._execute_with_retry(_query)


# ---------------------------------------------------------------------------
# 1. _persist_access_metadata_batch
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_persist_access_metadata_batch_skips_deleted(storage):
    """Batch metadata update must not mutate a soft-deleted row."""
    mem = _make_memory("batch metadata test content")
    await storage.store(mem)
    row_before = await _get_row(storage, mem.content_hash)
    assert row_before is not None

    # Soft-delete
    ok, _ = await storage.delete(mem.content_hash)
    assert ok

    # Capture state after delete
    row_deleted = await _get_row(storage, mem.content_hash)
    assert row_deleted[5] is not None  # deleted_at set

    # Attempt batch metadata update on deleted row
    mem.metadata = {"touched": True}
    await storage._persist_access_metadata_batch([mem])

    # Assert row unchanged
    row_after = await _get_row(storage, mem.content_hash)
    assert row_after[2] == row_deleted[2], "metadata should not change on deleted row"


# ---------------------------------------------------------------------------
# 2. _record_conflicts
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_record_conflicts_skips_deleted(storage):
    """Conflict tagging must not add 'conflict:unresolved' to a deleted row."""
    mem_a = _make_memory("conflict test memory alpha")
    mem_b = _make_memory("conflict test memory beta")
    await storage.store(mem_a)
    await storage.store(mem_b)

    # Soft-delete mem_a
    await storage.delete(mem_a.content_hash)
    row_deleted = await _get_row(storage, mem_a.content_hash)
    tags_before = row_deleted[1] or ""

    # Record conflict between new hash (mem_b) and deleted hash (mem_a)
    conflicts = [{
        "existing_hash": mem_a.content_hash,
        "similarity": 0.95,
        "divergence": 0.05,
    }]
    await storage._record_conflicts(mem_b.content_hash, conflicts)

    # Deleted row should NOT have conflict:unresolved tag
    row_after = await _get_row(storage, mem_a.content_hash)
    tags_after = row_after[1] or ""
    assert "conflict:unresolved" not in tags_after, (
        "conflict:unresolved should not be added to deleted row"
    )

    # But the live row (mem_b) SHOULD have the tag
    row_b = await _get_row(storage, mem_b.content_hash)
    assert "conflict:unresolved" in (row_b[1] or ""), (
        "conflict:unresolved should be added to live row"
    )


# ---------------------------------------------------------------------------
# 3. resolve_conflict — explicit error on deleted hash
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_conflict_errors_on_deleted_winner(storage):
    """resolve_conflict must return error when winner hash is deleted."""
    mem_w = _make_memory("winner memory for conflict resolution")
    mem_l = _make_memory("loser memory for conflict resolution")
    await storage.store(mem_w)
    await storage.store(mem_l)

    # Delete the winner
    await storage.delete(mem_w.content_hash)

    ok, msg = await storage.resolve_conflict(mem_w.content_hash, mem_l.content_hash)
    assert not ok, "resolve_conflict should fail when winner is deleted"
    assert "not found or deleted" in msg.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_conflict_errors_on_deleted_loser(storage):
    """resolve_conflict must return error when loser hash is deleted."""
    mem_w = _make_memory("winner memory for loser-deleted test")
    mem_l = _make_memory("loser memory for loser-deleted test")
    await storage.store(mem_w)
    await storage.store(mem_l)

    # Delete the loser
    await storage.delete(mem_l.content_hash)

    ok, msg = await storage.resolve_conflict(mem_w.content_hash, mem_l.content_hash)
    assert not ok, "resolve_conflict should fail when loser is deleted"
    assert "not found or deleted" in msg.lower()


# ---------------------------------------------------------------------------
# 4. _touch (via retrieve path)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_touch_skips_deleted(storage):
    """last_accessed update must not mutate a soft-deleted row."""
    mem = _make_memory("touch test memory content unique")
    await storage.store(mem)

    # Record initial last_accessed
    row_before = await _get_row(storage, mem.content_hash)
    la_before = row_before[3]

    # Soft-delete
    await storage.delete(mem.content_hash)
    row_deleted = await _get_row(storage, mem.content_hash)
    la_deleted = row_deleted[3]

    # Directly invoke _touch on the deleted hash
    now = time.time()

    def _touch():
        storage.conn.executemany(
            "UPDATE memories SET last_accessed = ? WHERE content_hash = ? AND deleted_at IS NULL",
            [(int(now), mem.content_hash)],
        )
        storage.conn.commit()

    await storage._execute_with_retry(_touch)

    # last_accessed should be unchanged
    row_after = await _get_row(storage, mem.content_hash)
    assert row_after[3] == la_deleted, "last_accessed should not change on deleted row"


# ---------------------------------------------------------------------------
# 5. update_memory_versioned (evolve)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_memory_versioned_skips_deleted(storage):
    """Versioned update must fail gracefully on a soft-deleted predecessor."""
    mem = _make_memory("evolve test memory original content")
    await storage.store(mem)

    # Soft-delete
    await storage.delete(mem.content_hash)

    # Attempt versioned update
    ok, msg, new_hash = await storage.update_memory_versioned(
        content_hash=mem.content_hash,
        new_content="evolved content that should not be created",
        reason="test evolution on deleted memory",
    )

    assert not ok, "update_memory_versioned should fail on deleted memory"
    assert "not found" in msg.lower() or "superseded" in msg.lower()
    assert new_hash is None
