"""search_memories(mode="hybrid") on the hybrid backend (#1576).

search_memories() only takes the hybrid path when the backend has retrieve_hybrid().
HybridMemoryStorage did not, so mode="hybrid" fell through to plain vector search on
the primary, with no warning. Delegating exposed the RRF path to store-scoped
searches, which returned before the store filter, so BM25 matches from other stores
leaked in; both fusion methods are covered here.

Settings that pick the path are patched per test, so the result does not depend on
MCP_HYBRID_FUSION_METHOD or MCP_HYBRID_SEARCH_ENABLED in the environment.
"""

import os
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

FUSION = [("weighted_average", "hybrid-bm25-vector"), ("rrf", "hybrid-rrf")]


@asynccontextmanager
async def hybrid_backend(temp_db_path, fusion):
    primary = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await primary.initialize()
    try:
        # Only the primary is exercised; no Cloudflare secondary or sync service.
        hybrid = HybridMemoryStorage.__new__(HybridMemoryStorage)
        hybrid.primary = primary
        with patch("mcp_memory_service.config.MCP_HYBRID_SEARCH_ENABLED", True), patch(
            "mcp_memory_service.config.MCP_HYBRID_FUSION_METHOD", fusion
        ):
            yield hybrid, primary
    finally:
        await primary.close()


async def put(primary, text, store="default"):
    memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["__test__"])
    ok, msg = await primary.store(memory, store=store)
    assert ok, msg
    return memory.content_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("fusion,backend", FUSION)
async def test_hybrid_mode_runs_bm25_on_hybrid_backend(temp_db_path, unique_content, fusion, backend):
    async with hybrid_backend(temp_db_path, fusion) as (hybrid, primary):
        for base in (
            "Deploy runbook: restart the harbor-import.timer after a schema change",
            "The dashboard toggles use the brand cyan",
            "Release tags are pushed two at a time",
        ):
            await put(primary, unique_content(base))

        found = await hybrid.search_memories(query="restart harbor runbook", mode="hybrid", limit=3)

        assert found["memories"], found
        assert all(m["debug_info"]["backend"] == backend for m in found["memories"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fusion,backend", FUSION)
async def test_hybrid_mode_stays_in_the_requested_store(temp_db_path, unique_content, fusion, backend):
    async with hybrid_backend(temp_db_path, fusion) as (hybrid, primary):
        # The keyword match lives only in "docs"; BM25 searches every store.
        other = await put(primary, unique_content("Deploy runbook: restart the harbor-import.timer"), store="docs")
        await put(primary, unique_content("The dashboard toggles use the brand cyan"))
        await put(primary, unique_content("Release tags are pushed two at a time"))

        found = await hybrid.search_memories(query="restart harbor runbook", mode="hybrid", limit=5, store="default")

        hashes = {m["content_hash"] for m in found["memories"]}
        assert other not in hashes, f"{fusion}: a memory from another store leaked into a store-scoped search"
        assert found["memories"], found
