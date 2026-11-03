"""Regression tests for issue #1803: superseded versions on the no-query paths.

retrieve() hides memories replaced by a versioned update unless
include_superseded is set. The search paths that bypass it (tag-only,
time-only, exact) and memory_list used to return the old versions as current.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest_asyncio.fixture
async def storage(temp_db_path):
    """SqliteVec storage with a stub 8-dim embedding model (hermetic, see #1724 tests)."""
    s = SqliteVecMemoryStorage(f"{temp_db_path}/test_1803.db")
    s.enable_cache = False
    s.embedding_dimension = 8

    async def _noop_model_init():
        return None

    s._initialize_embedding_model = _noop_model_init
    await s.initialize()
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 8), dtype=np.float32)
    s.embedding_model = model
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def versioned(storage):
    """Store "A v1" tagged demo, then a versioned update to "A v2". Returns (v1, v2)."""
    content = "A v1 of the demo rule"
    v1 = Memory(content=content, content_hash=generate_content_hash(content), tags=["demo"])
    ok, msg = await storage.store(v1, skip_semantic_dedup=True)
    assert ok, msg
    ok, msg, v2_hash = await storage.update_memory_versioned(
        v1.content_hash, "A v2 of the demo rule"
    )
    assert ok, msg
    return v1.content_hash, v2_hash


def _hashes(result):
    assert "error" not in result, result.get("error")
    return [m["content_hash"] for m in result["memories"]]


@pytest.mark.asyncio
async def test_tag_only_search_hides_superseded(storage, versioned):
    _v1, v2 = versioned
    assert _hashes(await storage.search_memories(tags=["demo"], limit=10)) == [v2]


@pytest.mark.asyncio
async def test_tag_only_search_matches_query_search(storage, versioned):
    """The same tag returns the same set with and without a query."""
    without_query = await storage.search_memories(tags=["demo"], limit=10)
    with_query = await storage.search_memories(query="demo rule", tags=["demo"], limit=10)
    assert _hashes(without_query) == _hashes(with_query)


@pytest.mark.asyncio
async def test_time_only_search_hides_superseded(storage, versioned):
    _v1, v2 = versioned
    result = await storage.search_memories(after="2000-01-01", limit=10)
    assert _hashes(result) == [v2]
    result = await storage.search_memories(time_expr="today", limit=10)
    assert _hashes(result) == [v2]


@pytest.mark.asyncio
async def test_exact_search_hides_superseded(storage, versioned):
    _v1, v2 = versioned
    result = await storage.search_memories(query="of the demo rule", mode="exact", limit=10)
    assert _hashes(result) == [v2]


@pytest.mark.asyncio
async def test_include_superseded_still_returns_old_versions(storage, versioned):
    v1, v2 = versioned
    result = await storage.search_memories(tags=["demo"], include_superseded=True, limit=10)
    by_hash = {m["content_hash"]: m for m in result["memories"]}
    assert set(by_hash) == {v1, v2}
    assert by_hash[v1]["status"] == "superseded"
    assert by_hash[v1]["superseded_by"] == v2
    assert by_hash[v2]["status"] == "current"

    exact = await storage.search_memories(
        query="of the demo rule", mode="exact", include_superseded=True, limit=10
    )
    assert set(_hashes(exact)) == {v1, v2}


@pytest.mark.asyncio
async def test_as_of_before_the_update_still_sees_v1(storage, versioned):
    """as_of keeps its own rule: a version replaced after as_of was current then."""
    v1, v2 = versioned

    t1 = datetime(2026, 9, 1, 12, tzinfo=UTC).timestamp()
    t2 = datetime(2026, 10, 1, 12, tzinfo=UTC).timestamp()

    def _backdate():
        for h, t in ((v1, t1), (v2, t2)):
            storage.conn.execute(
                "UPDATE memories SET created_at = ?, created_at_iso = ? WHERE content_hash = ?",
                (t, datetime.fromtimestamp(t, tz=UTC).isoformat(), h),
            )
        storage.conn.execute("UPDATE memories SET superseded_at = ? WHERE content_hash = ?", (t2, v1))
        storage.conn.commit()

    await storage._execute_with_retry(_backdate)
    result = await storage.search_memories(tags=["demo"], as_of="2026-09-15", limit=10)
    assert _hashes(result) == [v1]
    result = await storage.search_memories(tags=["demo"], as_of="2026-10-15", limit=10)
    assert _hashes(result) == [v2]


@pytest.mark.asyncio
async def test_list_memories_hides_superseded(storage, versioned):
    _v1, v2 = versioned
    service = MemoryService(storage)
    result = await service.list_memories(tags=["demo"])
    assert [m["content_hash"] for m in result["memories"]] == [v2]
    assert result["total"] == 1
    assert result["has_more"] is False

    everything = await service.list_memories()
    assert everything["total"] == 1


@pytest.mark.asyncio
async def test_list_memories_include_superseded(storage, versioned):
    v1, v2 = versioned
    result = await MemoryService(storage).list_memories(tags=["demo"], include_superseded=True)
    assert {m["content_hash"] for m in result["memories"]} == {v1, v2}
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_list_memories_pages_count_current_versions(storage):
    """Page totals come from the database, so they must not count old versions."""
    service = MemoryService(storage)
    hashes = []
    for i in range(3):
        content = f"rule {i} v1"
        m = Memory(content=content, content_hash=generate_content_hash(content), tags=["paged"])
        assert (await storage.store(m, skip_semantic_dedup=True))[0]
        ok, _, new_hash = await storage.update_memory_versioned(m.content_hash, f"rule {i} v2")
        assert ok
        hashes.append(new_hash)
    page1 = await service.list_memories(page=1, page_size=2, tags=["paged"])
    page2 = await service.list_memories(page=2, page_size=2, tags=["paged"])
    assert page1["total"] == page2["total"] == 3
    assert page1["has_more"] is True and page2["has_more"] is False
    listed = [m["content_hash"] for m in page1["memories"] + page2["memories"]]
    assert sorted(listed) == sorted(hashes)


@pytest.mark.asyncio
async def test_get_all_memories_default_keeps_superseded(storage, versioned):
    """Sync, export and maintenance callers still see every live row by default."""
    v1, v2 = versioned
    assert {m.content_hash for m in await storage.get_all_memories()} == {v1, v2}
    assert await storage.count_all_memories() == 2
    current = await storage.get_all_memories(include_superseded=False)
    assert [m.content_hash for m in current] == [v2]
    assert await storage.count_all_memories(include_superseded=False) == 1
