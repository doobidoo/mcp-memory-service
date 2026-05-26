"""Test that time filters work correctly with vector search (Issue #1012).

The bug: when using memory_search with both a query and time filters,
memories from the target time period were invisible if they weren't
among the top-K most semantically similar results across the entire database.

Fix: pass start_time/end_time into the SQL WHERE clause of the vector
search JOIN, and increase k_value to scan more candidates.
"""

import pytest
import time
import os
import tempfile

import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage():
    """Create a real SqliteVecStorage for integration testing."""
    tmp_dir = tempfile.mkdtemp(prefix="mcp-test-time-filter-")
    db_path = os.path.join(tmp_dir, "test.db")

    os.environ["SQLITE_VEC_PATH"] = db_path
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    # Cleanup
    if s.conn:
        s.conn.close()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_memory(content, tags=None, memory_type="note", created_at=None):
    import hashlib
    from datetime import datetime, timezone
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    m = Memory(content=content, content_hash=content_hash, tags=tags or [], memory_type=memory_type)
    if created_at is not None:
        # Set both fields consistently to bypass timestamp validation
        m.created_at = created_at
        m.created_at_iso = datetime.fromtimestamp(created_at, tz=timezone.utc).isoformat()
    return m


@pytest.mark.asyncio
async def test_time_filter_finds_recent_among_many_old(storage):
    """Time-filtered search should find recent memories even with many older similar ones."""
    # Store 15 old memories about "session checkpoint" (3 days ago)
    old_time = time.time() - 259200  # 3 days ago
    for i in range(15):
        m = _make_memory(
            f"Session checkpoint DNBSCDC289 — completed task {i}, reviewed code, pushed changes #{i}",
            tags=["sessao", "checkpoint"],
            created_at=old_time - (i * 60),
        )
        await storage.store(m)

    # Store 1 recent memory about "session checkpoint"
    recent = _make_memory(
        "Session checkpoint DNBSCDC289 — fixed time filter bug in memory search",
        tags=["sessao", "checkpoint"],
    )
    await storage.store(recent)

    # Search with time filter for recent memories only
    from datetime import datetime, timezone, timedelta
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")

    result = await storage.search_memories(
        query="session checkpoint",
        mode="semantic",
        after=two_hours_ago,
        limit=5,
    )

    assert result["total"] >= 1, (
        f"Time-filtered search should find the recent memory. "
        f"Got {result['total']} results. Bug #1012: time filter applied post-vector-search."
    )

    contents = [m["content"] for m in result["memories"]]
    assert any("fixed time filter bug" in c for c in contents), (
        f"Recent memory not found in results. Got: {contents}"
    )


@pytest.mark.asyncio
async def test_time_filter_excludes_old_with_after(storage):
    """Explicit after date should exclude old memories from semantic search."""
    # Store old memory
    m_old = _make_memory(
        "Architecture decision: use PostgreSQL for persistence layer",
        tags=["decision"],
        memory_type="decision",
        created_at=time.time() - 604800,  # 7 days ago
    )
    await storage.store(m_old)

    # Store recent memory
    m_new = _make_memory(
        "Architecture decision: use Redis for caching layer",
        tags=["decision"],
        memory_type="decision",
    )
    await storage.store(m_new)

    # Search with after=yesterday
    from datetime import datetime, timezone, timedelta
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    result = await storage.search_memories(
        query="architecture decision",
        mode="semantic",
        after=yesterday,
        limit=5,
    )

    assert result["total"] >= 1
    contents = [m["content"] for m in result["memories"]]
    assert any("Redis" in c for c in contents), "Recent memory should be found"
    assert not any("PostgreSQL" in c for c in contents), (
        "Old memory should be excluded by after filter"
    )
