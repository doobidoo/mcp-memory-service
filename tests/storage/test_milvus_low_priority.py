"""Unit tests for MilvusMemoryStorage low-priority method overrides:
- search_by_tag_chronological
- count_memories_by_tag
- is_deleted
- purge_deleted

Mock-based — no live Milvus server required.

Reference: https://github.com/doobidoo/mcp-memory-service/issues/888
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("sentence_transformers")

from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402


# -- Fixtures ----------------------------------------------------------------


def _make_storage() -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage skipping __init__."""
    import asyncio
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = True
    storage._lock = None
    storage._write_lock = asyncio.Lock()
    storage._call_client = AsyncMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return storage


def _make_row(
    content_hash: str = "h1",
    content: str = "test",
    tags: str = ",python,",
    metadata: Optional[Dict[str, Any]] = None,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a Milvus row dict."""
    now = time.time()
    return {
        "id": content_hash,
        "content": content,
        "tags": tags,
        "memory_type": "note",
        "metadata": json.dumps(metadata or {}),
        "created_at": created_at or (now - 100),
        "updated_at": now - 50,
        "created_at_iso": None,
        "updated_at_iso": None,
    }


# -- search_by_tag_chronological --------------------------------------------


class TestSearchByTagChronological:
    """Tests for native search_by_tag_chronological."""

    @pytest.mark.asyncio
    async def test_returns_memories_sorted_by_created_at(self):
        """Results are sorted newest first via Milvus sort."""
        storage = _make_storage()
        now = time.time()
        mem1 = Memory(content="c1", content_hash="h1", tags=["python"],
                      created_at=now - 10)
        mem2 = Memory(content="c2", content_hash="h2", tags=["python"],
                      created_at=now - 5)
        storage._query_memories = AsyncMock(return_value=[mem2, mem1])

        results = await storage.search_by_tag_chronological(["python"])

        assert len(results) == 2
        storage._query_memories.assert_called_once()
        # Verify sort_desc_key="created_at" was passed
        call_kwargs = storage._query_memories.call_args[1]
        assert call_kwargs["sort_desc_key"] == "created_at"

    @pytest.mark.asyncio
    async def test_empty_tags_returns_empty(self):
        """Empty tags list returns empty."""
        storage = _make_storage()
        results = await storage.search_by_tag_chronological([])
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_and_offset_applied(self):
        """Limit and offset are passed to _query_memories."""
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])

        await storage.search_by_tag_chronological(["python"], limit=5, offset=2)

        call_kwargs = storage._query_memories.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["offset"] == 2

    @pytest.mark.asyncio
    async def test_not_initialized_returns_empty(self):
        """Returns empty when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        results = await storage.search_by_tag_chronological(["python"])
        assert results == []


# -- count_memories_by_tag ---------------------------------------------------


class TestCountMemoriesByTag:
    """Tests for native count_memories_by_tag."""

    @pytest.mark.asyncio
    async def test_returns_count(self):
        """Returns the count from Milvus query."""
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=[{"count(*)": 42}])

        count = await storage.count_memories_by_tag(["python"])

        assert count == 42

    @pytest.mark.asyncio
    async def test_empty_tags_returns_zero(self):
        """Empty tags returns 0."""
        storage = _make_storage()
        count = await storage.count_memories_by_tag([])
        assert count == 0
        storage._call_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_failure_returns_zero(self):
        """Returns 0 on query failure."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=Exception("network error"))

        count = await storage.count_memories_by_tag(["python"])

        assert count == 0

    @pytest.mark.asyncio
    async def test_not_initialized_returns_zero(self):
        """Returns 0 when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        count = await storage.count_memories_by_tag(["python"])
        assert count == 0


# -- is_deleted --------------------------------------------------------------


class TestIsDeleted:
    """Tests for native is_deleted."""

    @pytest.mark.asyncio
    async def test_deleted_memory_returns_true(self):
        """Memory with deleted_at in metadata returns True."""
        storage = _make_storage()
        deleted_mem = Memory(
            content="old content",
            content_hash="h1",
            tags=["t"],
            metadata={"deleted_at": 1700000000.0},
        )
        storage.get_by_hash = AsyncMock(return_value=deleted_mem)

        assert await storage.is_deleted("h1") is True

    @pytest.mark.asyncio
    async def test_non_deleted_memory_returns_false(self):
        """Memory without deleted_at returns False."""
        storage = _make_storage()
        mem = Memory(content="content", content_hash="h1", tags=["t"], metadata={})
        storage.get_by_hash = AsyncMock(return_value=mem)

        assert await storage.is_deleted("h1") is False

    @pytest.mark.asyncio
    async def test_nonexistent_memory_returns_false(self):
        """Non-existent hash returns False."""
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(return_value=None)

        assert await storage.is_deleted("nonexistent") is False

    @pytest.mark.asyncio
    async def test_not_initialized_returns_false(self):
        """Returns False when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        assert await storage.is_deleted("h1") is False


# -- purge_deleted -----------------------------------------------------------


class TestPurgeDeleted:
    """Tests for native purge_deleted."""

    @pytest.mark.asyncio
    async def test_purges_old_tombstones(self):
        """Deletes memories with deleted_at older than threshold."""
        storage = _make_storage()
        now = time.time()
        old_time = now - 90 * 86400  # 90 days ago

        storage._call_client = AsyncMock(side_effect=[
            # First call: query returns old memories
            [
                {"id": "h1", "metadata": json.dumps({"deleted_at": old_time - 100})},
                {"id": "h2", "metadata": json.dumps({"deleted_at": old_time - 200})},
                {"id": "h3", "metadata": json.dumps({})},  # Not deleted
            ],
            # Second call: delete
            None,
        ])

        count = await storage.purge_deleted(older_than_days=30)

        assert count == 2
        # Verify delete was called with only the tombstoned IDs
        delete_call = storage._call_client.call_args_list[1]
        assert delete_call[0][0] == "delete"
        assert set(delete_call[1]["ids"]) == {"h1", "h2"}

    @pytest.mark.asyncio
    async def test_no_tombstones_returns_zero(self):
        """Returns 0 when no tombstones found."""
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=[
            {"id": "h1", "metadata": json.dumps({})},
        ])

        count = await storage.purge_deleted(older_than_days=30)

        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_result_returns_zero(self):
        """Returns 0 when query returns no rows."""
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=[])

        count = await storage.purge_deleted(older_than_days=30)

        assert count == 0

    @pytest.mark.asyncio
    async def test_query_failure_returns_zero(self):
        """Returns 0 on query failure."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=Exception("error"))

        count = await storage.purge_deleted(older_than_days=30)

        assert count == 0

    @pytest.mark.asyncio
    async def test_not_initialized_returns_zero(self):
        """Returns 0 when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        count = await storage.purge_deleted(older_than_days=30)
        assert count == 0
