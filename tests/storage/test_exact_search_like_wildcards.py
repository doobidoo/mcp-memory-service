"""Exact-mode search matches ``%`` and ``_`` in the query literally."""

import sqlite3
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

CONTENTS = [
    "renamed the column to user_id",
    "the userXid column is gone",
    "coverage is at 100% now",
    "coverage passed 1000 lines",
    r"backup written to C:\new_dir",
    r"backup written to C:\newXdir",
]


@pytest.mark.asyncio
async def test_sqlite_vec_exact_search_treats_wildcards_literally(temp_db_path):
    """``_`` and ``%`` in an exact query must not act as LIKE wildcards."""
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/exact.db")
    await storage.initialize()
    try:
        for content in CONTENTS:
            ok, message = await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content))
            )
            assert ok, message

        async def exact(query):
            result = await storage.search_memories(query=query, mode="exact")
            return sorted(m["content"] for m in result["memories"])

        assert await exact("user_id") == ["renamed the column to user_id"]
        assert await exact("100%") == ["coverage is at 100% now"]
        assert await exact("_") == [
            r"backup written to C:\new_dir",
            "renamed the column to user_id",
        ]
        assert await exact("%") == ["coverage is at 100% now"]
        assert await exact(r"C:\new_") == [r"backup written to C:\new_dir"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cloudflare_exact_search_treats_wildcards_literally():
    """D1 gets an escaped pattern; run its SQL against SQLite, which D1 is."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://d1.example"
    captured = []

    async def retry(method, url, json):
        captured.append(json)
        return type(
            "Response",
            (),
            {"json": lambda self: {"success": True, "result": [{"results": []}]}},
        )()

    storage._retry_request = AsyncMock(side_effect=retry)

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE memories (content TEXT, deleted_at REAL, created_at REAL)")
    db.executemany(
        "INSERT INTO memories VALUES (?, NULL, ?)",
        [(c, float(i)) for i, c in enumerate(CONTENTS)],
    )

    async def exact(query):
        captured.clear()
        await storage.get_by_exact_content(query)
        payload = captured[0]
        rows = db.execute(
            payload["sql"].replace("SELECT *", "SELECT content"), payload["params"]
        ).fetchall()
        return sorted(r[0] for r in rows)

    assert await exact("user_id") == ["renamed the column to user_id"]
    assert await exact("100%") == ["coverage is at 100% now"]
    assert await exact("_") == [
        r"backup written to C:\new_dir",
        "renamed the column to user_id",
    ]
    assert await exact(r"C:\new_") == [r"backup written to C:\new_dir"]
