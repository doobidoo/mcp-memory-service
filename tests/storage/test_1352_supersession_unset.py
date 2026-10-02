# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Regression test for bug #1352: superseded_by cannot be unset via update_memory_metadata.

The bug: MetadataMixin.update_memory_metadata's _do_update() only updates 
tags, memory_type, metadata, updated_at, created_at columns. Since superseded_by 
is not in protected_fields, it gets stored in JSON metadata instead of the column.
Result: update_memory_metadata(hash, {'superseded_by': None}) is a silent no-op.
"""

import hashlib
import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage(tmp_path):
    """Create a temporary SqliteVecMemoryStorage instance."""
    db_path = tmp_path / "test_1352.db"
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_memory_metadata_cannot_unset_superseded_by(storage):
    """
    Bug #1352: update_memory_metadata cannot unset superseded_by column.
    
    Expected: update_memory_metadata(hash, {'superseded_by': None}) should clear the column.
    Actual: Column remains unchanged (superseded_by still points to winner).
    
    This test SHOULD FAIL until the bug is fixed.
    """
    # 1. Store winner and loser memories
    winner = _make_memory("Winner memory content")
    loser = _make_memory("Loser memory content") 
    await storage.store(winner)
    await storage.store(loser)
    
    # 2. Mark loser as superseded by winner (setup supersession state)
    # Use direct SQL since update_memory_metadata might also fail to SET the column
    def _set_superseded():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner.content_hash, loser.content_hash)
        )
        storage.conn.commit()
    
    await storage._execute_with_retry(_set_superseded)
    
    # 3. Confirm loser is superseded by winner
    row = await _get_row(storage, loser.content_hash)
    assert row is not None
    assert row[4] == winner.content_hash, "Setup failed: loser should be superseded by winner"
    
    # 4. THE BUG: Try to unset superseded_by via update_memory_metadata
    ok, msg = await storage.update_memory_metadata(loser.content_hash, {'superseded_by': None})
    assert ok, f"update_memory_metadata should succeed but returned: {msg}"
    
    # 5. THE ASSERTION THAT SHOULD PASS BUT FAILS (RED test)
    row_after = await _get_row(storage, loser.content_hash)
    assert row_after[4] in (None, ''), (
        f"BUG #1352: superseded_by should be NULL after unsetting, "
        f"but remains '{row_after[4]}'. The column is not updated by update_memory_metadata."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_superseded_by_via_api(storage):
    """update_memory_metadata can SET superseded_by (not only clear it)."""
    winner = _make_memory("winner set-path content")
    loser = _make_memory("loser set-path content")
    await storage.store(winner)
    await storage.store(loser)

    ok, msg = await storage.update_memory_metadata(
        loser.content_hash, {"superseded_by": winner.content_hash}
    )
    assert ok, msg
    row = await _get_row(storage, loser.content_hash)
    assert row[4] == winner.content_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_string_clears_superseded_by(storage):
    """An empty string normalizes to NULL (clears the column)."""
    winner = _make_memory("winner empty content")
    loser = _make_memory("loser empty content")
    await storage.store(winner)
    await storage.store(loser)

    def _set():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner.content_hash, loser.content_hash),
        )
        storage.conn.commit()
    await storage._execute_with_retry(_set)

    ok, msg = await storage.update_memory_metadata(loser.content_hash, {"superseded_by": ""})
    assert ok, msg
    row = await _get_row(storage, loser.content_hash)
    assert row[4] in (None, "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unrelated_update_preserves_superseded_by(storage):
    """An update that does NOT ask for superseded_by must leave it untouched.

    Guards the race Greptile flagged: rewriting the read-back value on every
    update would silently undo a supersession recorded between read and write.
    """
    winner = _make_memory("winner preserve content")
    loser = _make_memory("loser preserve content")
    await storage.store(winner)
    await storage.store(loser)

    def _set():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner.content_hash, loser.content_hash),
        )
        storage.conn.commit()
    await storage._execute_with_retry(_set)

    # Update only tags — superseded_by not in updates
    ok, msg = await storage.update_memory_metadata(loser.content_hash, {"tags": ["x"]})
    assert ok, msg
    row = await _get_row(storage, loser.content_hash)
    assert row[4] == winner.content_hash, "unrelated update must not erase superseded_by"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_superseded_orphans_finds_missing_winner(storage):
    """A loser whose winner hash does not exist at all is an orphan."""
    loser = _make_memory("orphan loser - absent winner")
    await storage.store(loser)

    def _point_at_ghost():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            ("deadbeef" * 8, loser.content_hash),  # 64-char hash that has no row
        )
        storage.conn.commit()
    await storage._execute_with_retry(_point_at_ghost)

    orphans = await storage.list_superseded_orphans()
    hashes = [o["content_hash"] for o in orphans]
    assert loser.content_hash in hashes
    found = next(o for o in orphans if o["content_hash"] == loser.content_hash)
    assert found["superseded_by"] == "deadbeef" * 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_superseded_orphans_finds_deleted_winner(storage):
    """A loser whose winner was soft-deleted is an orphan."""
    winner = _make_memory("orphan winner - will be deleted")
    loser = _make_memory("orphan loser - deleted winner")
    await storage.store(winner)
    await storage.store(loser)

    def _set():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner.content_hash, loser.content_hash),
        )
        storage.conn.commit()
    await storage._execute_with_retry(_set)
    await storage.delete(winner.content_hash)  # winner gone

    orphans = await storage.list_superseded_orphans()
    assert loser.content_hash in [o["content_hash"] for o in orphans]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_superseded_orphans_excludes_live_winner(storage):
    """A loser whose winner is alive is NOT an orphan (healthy supersession)."""
    winner = _make_memory("live winner healthy")
    loser = _make_memory("live loser healthy")
    await storage.store(winner)
    await storage.store(loser)

    def _set():
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner.content_hash, loser.content_hash),
        )
        storage.conn.commit()
    await storage._execute_with_retry(_set)

    orphans = await storage.list_superseded_orphans()
    assert loser.content_hash not in [o["content_hash"] for o in orphans]
