# Copyright 2026 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Regression tests for sqlite-vec rowid desync / orphaned embeddings.

Root cause (production incident 2026-06-20):
  - memories has no AUTOINCREMENT, so SQLite reuses rowids after rows are
    hard-deleted (purge_deleted / _purge_tombstone) without deleting the
    matching memory_embeddings row.
  - The reused rowid then collides with a leftover (orphaned) embedding on
    the next store, raising "UNIQUE constraint failed: memory_embeddings".

These tests reproduce the collision and verify:
  1. The store path defends against a pre-existing embedding at the target
     rowid (write succeeds, no UNIQUE error).
  2. purge_deleted removes embeddings for the rows it hard-deletes.
  3. The health check surfaces orphaned/missing embeddings instead of
     reporting a blind "healthy".
"""

import hashlib
import pytest
import pytest_asyncio

from sqlite_vec import serialize_float32

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage(tmp_path):
    db_path = tmp_path / "test_orphan.db"
    s = SqliteVecMemoryStorage(str(db_path))
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


def _make_memory(content: str) -> Memory:
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.strip().lower().encode()).hexdigest(),
    )


def _inject_orphan_embedding(storage, rowid: int) -> None:
    """Insert an embedding at `rowid` with no matching memories row."""
    vec = storage._generate_embedding("orphan placeholder content")
    storage.conn.execute(
        "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (?, ?, ?)",
        (rowid, serialize_float32(vec), "default"),
    )
    storage.conn.commit()


def _count_orphans(storage) -> int:
    cur = storage.conn.execute(
        """SELECT COUNT(*) FROM memory_embeddings e
           LEFT JOIN memories m ON e.rowid = m.rowid
           WHERE m.rowid IS NULL"""
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Write-path collision defense (the production bug)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_survives_orphan_embedding_at_reused_rowid(storage):
    """store() must succeed even if an orphan embedding occupies the next rowid."""
    first = _make_memory("first memory establishing a rowid")
    ok, _ = await storage.store(first)
    assert ok

    max_rowid = storage.conn.execute("SELECT MAX(rowid) FROM memories").fetchone()[0]
    # Plant an orphan embedding exactly where the next memories row will land.
    _inject_orphan_embedding(storage, max_rowid + 1)
    assert _count_orphans(storage) == 1

    # Without the fix this raises "UNIQUE constraint failed: memory_embeddings".
    second = _make_memory("second memory that reuses the colliding rowid")
    ok, msg = await storage.store(second)
    assert ok, f"store should succeed despite orphan collision, got: {msg}"

    # The new memory must be retrievable and own a real embedding.
    assert _count_orphans(storage) == 0


# ---------------------------------------------------------------------------
# 2. purge_deleted removes embeddings (no orphan accumulation)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_purge_deleted_cleans_orphan_embeddings(storage):
    """purge_deleted must delete embeddings for rows it hard-deletes."""
    mem = _make_memory("memory to be purged with surviving embedding")
    await storage.store(mem)
    rowid = storage.conn.execute(
        "SELECT id FROM memories WHERE content_hash = ?", (mem.content_hash,)
    ).fetchone()[0]

    # Soft-delete, then simulate a surviving embedding (corrupt-blob fallback path).
    await storage.delete(mem.content_hash)
    if storage.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE rowid = ?", (rowid,)
    ).fetchone()[0] == 0:
        _inject_orphan_embedding(storage, rowid)

    # Purge tombstones immediately (older_than_days=0).
    await storage.purge_deleted(older_than_days=0)

    remaining = storage.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE rowid = ?", (rowid,)
    ).fetchone()[0]
    assert remaining == 0, "purge_deleted must not leave an orphan embedding"
    assert _count_orphans(storage) == 0


# ---------------------------------------------------------------------------
# 3. Health check surfaces orphaned embeddings
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check_detects_orphaned_embeddings(storage):
    """SqliteHealthChecker must report orphaned embeddings, not blind 'healthy'."""
    from mcp_memory_service.utils.health_check import SqliteHealthChecker

    mem = _make_memory("healthy memory baseline")
    await storage.store(mem)

    max_rowid = storage.conn.execute("SELECT MAX(rowid) FROM memories").fetchone()[0]
    _inject_orphan_embedding(storage, max_rowid + 5)

    is_valid, message, stats = await SqliteHealthChecker().check_health(storage)
    assert stats.get("orphaned_embeddings", 0) >= 1, stats
    assert stats.get("status") == "degraded", stats


# ---------------------------------------------------------------------------
# 4. Health check surfaces missing embeddings (search blind spot)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check_detects_missing_embeddings(storage):
    """A live memory with no embedding row must degrade status, not report 'healthy'.

    Regression for the incident in #1225: the memory rows still satisfy tag/time
    queries and count_all_memories, so counts look right while the rows are invisible
    to semantic search. missing_embeddings was counted but only orphaned/collision
    downgraded the status, so this blind spot silently stayed 'healthy'.
    """
    from mcp_memory_service.utils.health_check import SqliteHealthChecker

    mem = _make_memory("healthy memory baseline")
    await storage.store(mem)

    # Drop the embedding row, leaving the (live) memories row behind.
    rowid = storage.conn.execute(
        "SELECT id FROM memories WHERE content_hash = ?", (mem.content_hash,)
    ).fetchone()[0]
    storage.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (rowid,))
    storage.conn.commit()

    is_valid, message, stats = await SqliteHealthChecker().check_health(storage)
    assert stats.get("missing_embeddings", 0) >= 1, stats
    assert stats.get("orphaned_embeddings", 0) == 0, stats
    assert stats.get("status") == "degraded", stats
