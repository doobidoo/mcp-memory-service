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
_A_HASH = generate_content_hash(_A_TEXT)
_B_HASH = "b" * 64


def _memory(content: str, content_hash: str, superseded_by=None) -> Memory:
    metadata = {} if superseded_by is None else {"superseded_by": superseded_by}
    return Memory(content=content, content_hash=content_hash, metadata=metadata)


def _make_storage(rows: dict) -> MilvusMemoryStorage:
    rows.setdefault(_B_HASH, _memory("Alice works at Globex", _B_HASH))
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

    async def get_by_hash(content_hash, store=None):
        return rows.get(content_hash)

    async def get_for_write(content_hash, store, **kwargs):
        return rows.get(content_hash), "default"

    async def update_memory_metadata(content_hash, updates, **kwargs):
        rows[content_hash].metadata.update(updates.get("metadata", {}))
        return True, "ok"

    async def store(memory, skip_semantic_dedup=False, store="default", **kwargs):
        if memory.content_hash in rows:
            return False, "Duplicate content detected (exact match)"
        rows[memory.content_hash] = memory
        return True, "Memory stored successfully"

    storage._call_client = call_client
    storage.get_by_hash = get_by_hash
    storage._get_for_write = get_for_write
    storage.update_memory_metadata = update_memory_metadata
    storage.store = store
    storage._cleanup_unlinked_version = AsyncMock(return_value=True)
    return storage


@pytest.mark.asyncio
async def test_restoring_earlier_text_stores_a_new_version():
    rows = {_A_HASH: _memory(_A_TEXT, _A_HASH, superseded_by=_B_HASH)}
    storage = _make_storage(rows)

    ok, message, new_hash = await storage.update_memory_versioned(_B_HASH, _A_TEXT)

    assert ok, message
    assert new_hash is not None
    assert new_hash != _A_HASH
    assert new_hash in rows
    assert rows[new_hash].content == _A_TEXT


@pytest.mark.asyncio
async def test_repeated_restores_get_distinct_salted_hashes():
    rows = {_A_HASH: _memory(_A_TEXT, _A_HASH, superseded_by=_B_HASH)}
    storage = _make_storage(rows)

    ok1, _, first = await storage.update_memory_versioned(_B_HASH, _A_TEXT)
    assert ok1
    ok2, _, second = await storage.update_memory_versioned(first, _A_TEXT)

    assert ok2
    assert first != second
    assert first in rows and second in rows


@pytest.mark.asyncio
async def test_independent_duplicate_is_not_recreated_as_a_version():
    rows = {_A_HASH: _memory(_A_TEXT, _A_HASH)}
    storage = _make_storage(rows)

    ok, message, new_hash = await storage.update_memory_versioned(_B_HASH, _A_TEXT)

    assert not ok
    assert "Duplicate content detected" in message
    assert new_hash is None
    assert len(rows) == 2
    assert rows[_A_HASH].metadata == {}


@pytest.mark.asyncio
async def test_unrelated_superseded_duplicate_is_not_in_the_version_chain():
    rows = {_A_HASH: _memory(_A_TEXT, _A_HASH, superseded_by="c" * 64)}
    storage = _make_storage(rows)

    ok, message, new_hash = await storage.update_memory_versioned(_B_HASH, _A_TEXT)

    assert not ok
    assert "Duplicate content detected" in message
    assert new_hash is None
    assert len(rows) == 2
    assert rows[_A_HASH].metadata == {"superseded_by": "c" * 64}
