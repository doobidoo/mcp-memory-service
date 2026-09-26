"""Regression tests for consolidation code calling storage methods that don't exist (#1319).

Each call used to fail inside a broad ``except`` (or a ``hasattr`` guard that is
always False), so the feature silently did nothing instead of raising.
"""

import os
import sqlite3
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.consolidation.belief_service import BeliefService
from mcp_memory_service.consolidation.insights import InsightCard, store_insights
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


async def _storage(temp_db_path):
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "dead_calls.db"))
    await storage.initialize()
    return storage


async def _store(storage, text):
    memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["test"])
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory.content_hash


@pytest.mark.asyncio
async def test_belief_observation_lookup_finds_stored_memories(temp_db_path):
    """_get_observations_by_hashes called storage.get_memory_by_hash(), which no
    backend defines, so every lookup raised, was swallowed, and returned []."""
    storage = await _storage(temp_db_path)
    h = await _store(storage, "The nightly backup runs at 02:00 against the NAS share.")

    observations = await BeliefService(storage)._get_observations_by_hashes([h])

    assert [o["content_hash"] for o in observations] == [h]


@pytest.mark.asyncio
async def test_insight_cards_write_derived_from_edges(temp_db_path):
    """store_insights only wrote edges if the memory storage had
    store_association, which lives on GraphStorage, so no derived_from edge
    was ever written. The graph handle is now passed in explicitly."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Deploy note {i}: restart caddy after cert renewal.") for i in range(2)]
    graph = GraphStorage(storage.db_path)
    card = InsightCard(
        title="Caddy restarts follow renewals",
        content="Several notes restart caddy after certificate renewal.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.8,
    )

    stored = await store_insights([card], storage, graph=graph)

    assert len(stored) == 1
    with sqlite3.connect(storage.db_path) as conn:
        edges = conn.execute(
            "SELECT source_hash FROM memory_graph WHERE target_hash = ? AND relationship_type = 'derived_from'",
            (stored[0],),
        ).fetchall()
    assert sorted(e[0] for e in edges) == sorted(sources)


@pytest.mark.asyncio
async def test_cloudflare_storage_supports_consolidation_delete():
    """The consolidator applies forgetting through storage.delete_memory(), which
    CloudflareStorage did not have, so forgetting raised AttributeError there.
    The base class now provides it on top of delete()."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.delete = AsyncMock(return_value=(True, "deleted"))

    assert await storage.delete_memory("abc123") is True
    storage.delete.assert_awaited_once_with("abc123")
