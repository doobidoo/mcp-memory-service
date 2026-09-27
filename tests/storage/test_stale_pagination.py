"""Deterministic ordering and stale filtering in the storage backends."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_cloudflare_stale_query_filters_before_deterministic_pagination():
    """D1 applies stale cutoff before deterministic limit and offset."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://d1.example"
    captured = {}

    async def retry(method, url, json):
        captured.update(method=method, url=url, payload=json)
        return type(
            "Response",
            (),
            {"json": lambda self: {"success": True, "result": [{"results": []}]}},
        )()

    storage._retry_request = AsyncMock(side_effect=retry)
    await storage.get_all_memories(limit=5, offset=10, stale_days=90)

    sql = captured["payload"]["sql"]
    params = captured["payload"]["params"]
    assert "json_extract(m.metadata_json, '$.last_accessed_at')" in sql
    assert " < ?" in sql
    assert "ORDER BY m.created_at DESC, m.id DESC" in sql
    assert "LIMIT ?" in sql and "OFFSET ?" in sql
    assert params[-2:] == [5, 10]
    assert params[-3] == pytest.approx(time.time() - 90 * 86400, abs=2)


@pytest.mark.asyncio
async def test_milvus_stale_page_sorts_created_at_and_id_before_offset():
    """Milvus pages stale rows deterministically instead of iterator order."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage._write_lock = asyncio.Lock()
    storage.client = object()
    storage._has_access_collection = False
    storage._drain_main_ids_and_created_at = lambda _filter: [
        {"id": "a", "created_at": 1},
        {"id": "c", "created_at": 2},
        {"id": "b", "created_at": 2},
    ]
    storage._query_memories = AsyncMock(return_value=[])

    await storage._get_stale_memories(
        "", stale_days=1, limit=1, offset=1, include_embeddings=True
    )

    query = storage._query_memories.await_args.kwargs
    assert 'id == "b"' in query["filter_expr"]
    assert query["limit"] == 1
    assert query["include_embeddings"] is True


@pytest.mark.asyncio
async def test_sqlite_vec_stale_page_breaks_created_at_ties_by_content_hash(
    temp_db_path,
):
    """sqlite-vec pages tied timestamps in one fixed order, and covers the tail.

    Six memories share a single ``created_at``. ``ORDER BY m.created_at DESC``
    alone leaves their relative order up to the scan, so a page boundary can
    fall in a different place on the next fetch and step over a row. The
    tie-breaker pins it: the expected order below is content-hash descending,
    which is deliberately not the insertion order.
    """
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/tie_break.db")
    await storage.initialize()
    try:
        shared_created_at = time.time() - 400 * 86400
        contents = [f"tie-break candidate {index}" for index in range(6)]
        hashes = []
        for content in contents:
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["stale-tail"],
            )
            success, message = await storage.store(memory)
            assert success, message
            hashes.append(memory.content_hash)
            storage.conn.execute(
                "UPDATE memories SET created_at = ?, last_accessed = NULL "
                "WHERE content_hash = ?",
                (shared_created_at, memory.content_hash),
            )
        storage.conn.commit()

        expected = sorted(hashes, reverse=True)
        # The fix is only observable if the contract order is not the order the
        # rows were written in; otherwise a scan in rowid order would satisfy
        # the assertions by accident.
        assert expected != hashes

        whole = await storage.get_all_memories(stale_days=90)
        assert [memory.content_hash for memory in whole] == expected

        pages = []
        for offset in (0, 2, 4):
            page = await storage.get_all_memories(limit=2, offset=offset, stale_days=90)
            pages.append([memory.content_hash for memory in page])
        assert pages == [expected[0:2], expected[2:4], expected[4:6]]

        # Every row seen exactly once across the walk: no skips, no repeats.
        walked = [content_hash for page in pages for content_hash in page]
        assert sorted(walked) == sorted(hashes)
    finally:
        await storage.close()
