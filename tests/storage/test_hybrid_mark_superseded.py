"""HybridMemoryStorage.mark_superseded_batch must reach the primary store.

Without an override the MemoryStorage base-class no-op applies: it returns 0
and marks nothing, so every caller (dedup consolidation, contradiction
detection) silently did nothing on the hybrid backend.
"""

import os

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_mark_superseded_batch_hides_loser_in_primary(temp_db_path, unique_content, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    primary = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await primary.initialize()
    try:
        # Only the primary is exercised; no Cloudflare secondary or sync service.
        hybrid = HybridMemoryStorage.__new__(HybridMemoryStorage)
        hybrid.primary = primary

        hashes = []
        for base in ("backup job runs nightly at two", "backup job now runs nightly at three"):
            content = unique_content(base)
            memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["__test__"])
            ok, msg = await primary.store(memory)
            assert ok, msg
            hashes.append(memory.content_hash)
        loser, winner = hashes

        marked = await hybrid.mark_superseded_batch([(winner, loser)])

        assert marked == 1
        found = await primary.search_memories(query="backup job nightly", limit=10)
        assert loser not in {m["content_hash"] for m in found["memories"]}
    finally:
        await primary.close()
