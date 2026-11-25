"""Milvus versioned updates must give restored text a fresh hash (#2129).

After A→B the superseded row for A is still stored, so restoring A with the
plain content hash collides with that row and store() rejects it as an exact
duplicate: the edit can never be undone. These tests use a mocked client, in
the style of test_milvus_update_stored_vector.py, so they run without pymilvus.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

_A_TEXT = "Alice works at Acme"
_B_HASH = "b" * 64


def _make_storage(rows: dict) -> MilvusMemoryStorage:
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.uri = "./milvus.db"
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])

    async def call_client(method: str, **kwargs):
        if method == "get":
            return [{"id": h} for h in kwargs.get("ids") or [] if h in rows]
        return None

    storage._call_client = call_client

    async def store(memory, skip_semantic_dedup=False, store="default", **kwargs):
        # The real backend rejects an insert whose hash is already stored.
        if memory.content_hash in rows:
            return False, "Duplicate content detected (exact match)"
        rows[memory.content_hash] = memory
        return True, "Memory stored successfully"

    storage.store = store
    old = Memory(content="Alice works at Globex", content_hash=_B_HASH, metadata={})
    storage._get_for_write = AsyncMock(return_value=(old, "default"))
    storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))
    storage._cleanup_unlinked_version = AsyncMock(return_value=True)
    return storage


@pytest.mark.asyncio
async def test_restoring_earlier_text_stores_a_new_version():
    rows = {generate_content_hash(_A_TEXT): object()}  # superseded A left behind
    storage = _make_storage(rows)

    ok, message, new_hash = await storage.update_memory_versioned(_B_HASH, _A_TEXT)

    assert ok, message
    assert new_hash is not None
    assert new_hash != generate_content_hash(_A_TEXT)
    assert new_hash in rows
    assert rows[new_hash].content == _A_TEXT


@pytest.mark.asyncio
async def test_repeated_restores_get_distinct_salted_hashes():
    rows = {generate_content_hash(_A_TEXT): object()}
    storage = _make_storage(rows)

    ok1, _, first = await storage.update_memory_versioned(_B_HASH, _A_TEXT)
    ok2, _, second = await storage.update_memory_versioned(_B_HASH, _A_TEXT)

    assert ok1 and ok2
    assert first != second
    assert first in rows and second in rows
