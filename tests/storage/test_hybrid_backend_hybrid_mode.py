"""search_memories(mode="hybrid") must run BM25 + vector search on the hybrid backend (#1576).

search_memories() only takes the hybrid path when the backend has retrieve_hybrid().
HybridMemoryStorage did not, so mode="hybrid" fell through to plain vector search on
the primary, with no warning.
"""

import os

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_hybrid_mode_runs_bm25_on_hybrid_backend(temp_db_path, unique_content):
    primary = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await primary.initialize()
    try:
        # Only the primary is exercised; no Cloudflare secondary or sync service.
        hybrid = HybridMemoryStorage.__new__(HybridMemoryStorage)
        hybrid.primary = primary

        for base in (
            "Deploy runbook: restart the harbor-import.timer after a schema change",
            "The dashboard toggles use the brand cyan",
            "Release tags are pushed two at a time",
        ):
            content = unique_content(base)
            ok, msg = await primary.store(Memory(content=content, content_hash=generate_content_hash(content), tags=["__test__"]))
            assert ok, msg

        found = await hybrid.search_memories(query="harbor-import.timer", mode="hybrid", limit=3)

        assert found["memories"], found
        assert all(m["debug_info"]["backend"] == "hybrid-bm25-vector" for m in found["memories"])
    finally:
        await primary.close()
