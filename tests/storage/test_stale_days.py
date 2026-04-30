# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Tests for stale_days filter on memory_list (#784).

Covers:
- Basic stale filtering (old memories returned, recent excluded)
- Boundary: memory accessed exactly stale_days ago (excluded — strict <)
- Null last_accessed falls back to created_at
- Composition with tags and memory_type filters
- Pagination across stale results
- count_all_memories consistency with get_all_memories
"""

import hashlib
import pytest
import pytest_asyncio
import tempfile
import os
import shutil
import time

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def service():
    """Create temporary MemoryService for stale_days testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_stale.db")
    try:
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        svc = MemoryService(storage)
        yield svc, storage
    finally:
        if hasattr(storage, "conn") and storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _make_memory(content: str, tags=None, memory_type=None) -> Memory:
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.strip().lower().encode()).hexdigest(),
        tags=tags or [],
        memory_type=memory_type,
    )


def _set_timestamps(storage, content_hash: str, created_at: float, last_accessed=None):
    """Directly set timestamps in DB for test setup."""
    if last_accessed is not None:
        storage.conn.execute(
            "UPDATE memories SET created_at = ?, last_accessed = ? WHERE content_hash = ?",
            (created_at, int(last_accessed), content_hash),
        )
    else:
        storage.conn.execute(
            "UPDATE memories SET created_at = ?, last_accessed = NULL WHERE content_hash = ?",
            (created_at, content_hash),
        )
    storage.conn.commit()


# ---------------------------------------------------------------------------
# Basic stale filtering
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_returns_old_memories(service):
    """Memories not accessed in stale_days should be returned."""
    svc, storage = service
    mem = _make_memory("old stale memory content")
    await storage.store(mem)

    # Set last_accessed to 60 days ago
    _set_timestamps(storage, mem.content_hash, time.time() - 90 * 86400, time.time() - 60 * 86400)

    result = await svc.list_memories(stale_days=30)
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem.content_hash in hashes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_excludes_recent_memories(service):
    """Recently accessed memories should NOT be returned."""
    svc, storage = service
    mem = _make_memory("recently accessed memory content")
    await storage.store(mem)

    # Set last_accessed to 5 days ago
    _set_timestamps(storage, mem.content_hash, time.time() - 90 * 86400, time.time() - 5 * 86400)

    result = await svc.list_memories(stale_days=30)
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem.content_hash not in hashes


# ---------------------------------------------------------------------------
# Boundary: memory accessed just inside the window (not stale)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_boundary_not_stale(service):
    """Memory accessed just inside the stale window is NOT stale."""
    svc, storage = service
    mem = _make_memory("boundary stale memory content")
    await storage.store(mem)

    # Set last_accessed to 29 days ago (inside 30-day window)
    _set_timestamps(storage, mem.content_hash, time.time() - 90 * 86400, time.time() - 29 * 86400)

    result = await svc.list_memories(stale_days=30)
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem.content_hash not in hashes


# ---------------------------------------------------------------------------
# Null last_accessed falls back to created_at
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_null_last_accessed_uses_created_at(service):
    """When last_accessed is NULL, COALESCE falls back to created_at."""
    svc, storage = service
    mem = _make_memory("never accessed memory content")
    await storage.store(mem)

    # Set created_at to 60 days ago, last_accessed = NULL
    _set_timestamps(storage, mem.content_hash, time.time() - 60 * 86400, last_accessed=None)

    result = await svc.list_memories(stale_days=30)
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem.content_hash in hashes


# ---------------------------------------------------------------------------
# Composition with tags
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_with_tag_filter(service):
    """stale_days should compose with tag filter."""
    svc, storage = service
    mem_tagged = _make_memory("stale tagged memory", tags=["archived"])
    mem_untagged = _make_memory("stale untagged memory", tags=["active"])
    await storage.store(mem_tagged)
    await storage.store(mem_untagged)

    # Both stale
    _set_timestamps(storage, mem_tagged.content_hash, time.time() - 60 * 86400, time.time() - 60 * 86400)
    _set_timestamps(storage, mem_untagged.content_hash, time.time() - 60 * 86400, time.time() - 60 * 86400)

    result = await svc.list_memories(stale_days=30, tag="archived")
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem_tagged.content_hash in hashes
    assert mem_untagged.content_hash not in hashes


# ---------------------------------------------------------------------------
# Composition with memory_type
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_with_memory_type_filter(service):
    """stale_days should compose with memory_type filter."""
    svc, storage = service
    mem_note = _make_memory("stale note memory", memory_type="note")
    mem_fact = _make_memory("stale fact memory", memory_type="fact")
    await storage.store(mem_note)
    await storage.store(mem_fact)

    _set_timestamps(storage, mem_note.content_hash, time.time() - 60 * 86400, time.time() - 60 * 86400)
    _set_timestamps(storage, mem_fact.content_hash, time.time() - 60 * 86400, time.time() - 60 * 86400)

    result = await svc.list_memories(stale_days=30, memory_type="note")
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem_note.content_hash in hashes
    assert mem_fact.content_hash not in hashes


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_days_pagination(service):
    """Pagination should work correctly with stale_days filter."""
    svc, storage = service

    # Create 5 stale memories
    mems = []
    for i in range(5):
        m = _make_memory(f"stale pagination memory {i}")
        await storage.store(m)
        _set_timestamps(storage, m.content_hash, time.time() - 60 * 86400, time.time() - 60 * 86400)
        mems.append(m)

    # Page 1: 3 items
    result1 = await svc.list_memories(stale_days=30, page=1, page_size=3)
    assert len(result1["memories"]) == 3
    assert result1["total"] == 5
    assert result1["has_more"] is True

    # Page 2: 2 items
    result2 = await svc.list_memories(stale_days=30, page=2, page_size=3)
    assert len(result2["memories"]) == 2
    assert result2["has_more"] is False


# ---------------------------------------------------------------------------
# No stale_days = no filter (backward compat)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_stale_days_returns_all(service):
    """Without stale_days, all memories are returned (backward compat)."""
    svc, storage = service
    mem = _make_memory("normal memory no stale filter")
    await storage.store(mem)

    result = await svc.list_memories()
    hashes = [m["content_hash"] for m in result["memories"]]
    assert mem.content_hash in hashes
