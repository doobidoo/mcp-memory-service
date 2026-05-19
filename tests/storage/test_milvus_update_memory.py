"""Unit tests for MilvusMemoryStorage.update_memory and
update_memories_batch native overrides.

These tests are mock-based and do NOT require a live Milvus server
or the sentence-transformers model cache. They verify that the native
Milvus upsert path is used instead of the base-class fallback.

Reference: https://github.com/doobidoo/mcp-memory-service/issues/888
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("sentence_transformers")

from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402


# -- Fixtures ----------------------------------------------------------------


def _make_storage() -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage skipping __init__ so no network
    or model loading happens."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = False
    storage._lock = None
    # Mock _call_client as async
    storage._call_client = AsyncMock()
    # Mock _generate_embedding to return a fixed vector
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return storage


def _make_memory(
    content_hash: str = "hash_abc",
    content: str = "test content",
    tags: Optional[List[str]] = None,
    memory_type: str = "note",
    metadata: Optional[Dict[str, Any]] = None,
    created_at: Optional[float] = None,
    updated_at: Optional[float] = None,
) -> Memory:
    """Build a Memory object for testing."""
    now = time.time()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or ["test"],
        memory_type=memory_type,
        metadata=metadata or {},
        created_at=created_at or (now - 100),
        updated_at=updated_at or (now - 50),
        created_at_iso=None,
        updated_at_iso=None,
    )


# -- update_memory -----------------------------------------------------------


class TestUpdateMemory:
    """Tests for MilvusMemoryStorage.update_memory native override."""

    @pytest.mark.asyncio
    async def test_successful_update(self):
        """Normal update: existing memory found, upsert called once."""
        storage = _make_storage()
        existing = _make_memory(content_hash="hash_abc", tags=["old_tag"])
        updated = _make_memory(content_hash="hash_abc", tags=["new_tag"], memory_type="decision")

        storage.get_by_hash = AsyncMock(return_value=existing)

        result = await storage.update_memory(updated)

        assert result is True
        storage.get_by_hash.assert_called_once_with("hash_abc")
        storage._generate_embedding.assert_called_once_with(existing.content)
        storage._call_client.assert_called_once()
        call_args = storage._call_client.call_args
        assert call_args[0][0] == "upsert"
        assert call_args[1]["collection_name"] == "unit_test_collection"
        # Verify entity data
        entity = call_args[1]["data"][0]
        assert entity["id"] == "hash_abc"
        assert "new_tag" in entity["tags"]
        assert entity["memory_type"] == "decision"

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self):
        """Update on non-existent hash returns False."""
        storage = _make_storage()
        memory = _make_memory(content_hash="nonexistent")

        storage.get_by_hash = AsyncMock(return_value=None)

        result = await storage.update_memory(memory)

        assert result is False
        storage._call_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_initialized_returns_false(self):
        """Returns False when storage is not initialized."""
        storage = _make_storage()
        storage._initialized = False

        memory = _make_memory()
        result = await storage.update_memory(memory)

        assert result is False

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_false(self):
        """Returns False when embedding generation fails."""
        storage = _make_storage()
        existing = _make_memory()
        storage.get_by_hash = AsyncMock(return_value=existing)
        storage._generate_embedding = MagicMock(side_effect=RuntimeError("model error"))

        result = await storage.update_memory(_make_memory())

        assert result is False
        storage._call_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_failure_returns_false(self):
        """Returns False when Milvus upsert call raises."""
        storage = _make_storage()
        existing = _make_memory()
        storage.get_by_hash = AsyncMock(return_value=existing)
        storage._call_client = AsyncMock(side_effect=Exception("connection lost"))

        result = await storage.update_memory(_make_memory())

        assert result is False

    @pytest.mark.asyncio
    async def test_preserves_created_at(self):
        """created_at from existing memory is preserved, updated_at is refreshed."""
        storage = _make_storage()
        original_created = 1700000000.0
        existing = _make_memory(created_at=original_created, updated_at=1700001000.0)
        storage.get_by_hash = AsyncMock(return_value=existing)

        before = time.time()
        await storage.update_memory(_make_memory())
        after = time.time()

        entity = storage._call_client.call_args[1]["data"][0]
        assert entity["created_at"] == original_created
        assert before <= entity["updated_at"] <= after

    @pytest.mark.asyncio
    async def test_content_lower_field_when_enabled(self):
        """When _has_content_lower is True, entity includes content_lower."""
        storage = _make_storage()
        storage._has_content_lower = True
        existing = _make_memory(content="Hello World")
        storage.get_by_hash = AsyncMock(return_value=existing)

        await storage.update_memory(_make_memory())

        entity = storage._call_client.call_args[1]["data"][0]
        assert entity["content_lower"] == "hello world"


# -- update_memories_batch ---------------------------------------------------


class TestUpdateMemoriesBatch:
    """Tests for MilvusMemoryStorage.update_memories_batch native override."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        """Empty input returns empty results."""
        storage = _make_storage()
        result = await storage.update_memories_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_not_initialized_returns_all_false(self):
        """Returns all False when storage is not initialized."""
        storage = _make_storage()
        storage._initialized = False

        memories = [_make_memory(content_hash="h1"), _make_memory(content_hash="h2")]
        result = await storage.update_memories_batch(memories)

        assert result == [False, False]

    @pytest.mark.asyncio
    async def test_batch_update_single_upsert_call(self):
        """Multiple memories are sent in a single upsert call."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1", tags=["tag1"])
        m2 = _make_memory(content_hash="h2", tags=["tag2"])
        m3 = _make_memory(content_hash="h3", tags=["tag3"])

        existing_map = {
            "h1": _make_memory(content_hash="h1", content="content1"),
            "h2": _make_memory(content_hash="h2", content="content2"),
            "h3": _make_memory(content_hash="h3", content="content3"),
        }

        async def mock_get_by_hash(h):
            return existing_map.get(h)

        storage.get_by_hash = mock_get_by_hash

        result = await storage.update_memories_batch([m1, m2, m3])

        assert result == [True, True, True]
        # Single upsert call with 3 entities
        storage._call_client.assert_called_once()
        call_args = storage._call_client.call_args
        assert call_args[0][0] == "upsert"
        assert len(call_args[1]["data"]) == 3

    @pytest.mark.asyncio
    async def test_partial_failure_skips_not_found(self):
        """Memories not found are skipped (False), others succeed."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1")
        m2 = _make_memory(content_hash="h2_missing")
        m3 = _make_memory(content_hash="h3")

        existing_map = {
            "h1": _make_memory(content_hash="h1", content="c1"),
            "h3": _make_memory(content_hash="h3", content="c3"),
        }

        async def mock_get_by_hash(h):
            return existing_map.get(h)

        storage.get_by_hash = mock_get_by_hash

        result = await storage.update_memories_batch([m1, m2, m3])

        assert result == [True, False, True]
        # Only 2 entities in the upsert call
        entity_data = storage._call_client.call_args[1]["data"]
        assert len(entity_data) == 2

    @pytest.mark.asyncio
    async def test_preserve_timestamps_true(self):
        """When preserve_timestamps=True, updated_at is NOT refreshed."""
        storage = _make_storage()
        original_updated = 1700001000.0
        existing = _make_memory(content_hash="h1", updated_at=original_updated)

        async def mock_get_by_hash(h):
            return existing

        storage.get_by_hash = mock_get_by_hash

        await storage.update_memories_batch(
            [_make_memory(content_hash="h1")], preserve_timestamps=True
        )

        entity = storage._call_client.call_args[1]["data"][0]
        assert entity["updated_at"] == original_updated

    @pytest.mark.asyncio
    async def test_preserve_timestamps_false(self):
        """When preserve_timestamps=False (default), updated_at is refreshed."""
        storage = _make_storage()
        original_updated = 1700001000.0
        existing = _make_memory(content_hash="h1", updated_at=original_updated)

        async def mock_get_by_hash(h):
            return existing

        storage.get_by_hash = mock_get_by_hash

        before = time.time()
        await storage.update_memories_batch(
            [_make_memory(content_hash="h1")], preserve_timestamps=False
        )
        after = time.time()

        entity = storage._call_client.call_args[1]["data"][0]
        assert before <= entity["updated_at"] <= after

    @pytest.mark.asyncio
    async def test_batch_upsert_failure_returns_all_false(self):
        """When the batch upsert call fails, all results are False."""
        storage = _make_storage()
        existing = _make_memory(content_hash="h1")

        async def mock_get_by_hash(h):
            return existing

        storage.get_by_hash = mock_get_by_hash
        storage._call_client = AsyncMock(side_effect=Exception("network error"))

        result = await storage.update_memories_batch([_make_memory(content_hash="h1")])

        assert result == [False]

    @pytest.mark.asyncio
    async def test_embedding_failure_skips_item(self):
        """When embedding fails for one item, it's skipped but others proceed."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1")
        m2 = _make_memory(content_hash="h2")

        existing_map = {
            "h1": _make_memory(content_hash="h1", content="c1"),
            "h2": _make_memory(content_hash="h2", content="c2"),
        }

        async def mock_get_by_hash(h):
            return existing_map.get(h)

        storage.get_by_hash = mock_get_by_hash

        call_count = [0]
        def mock_embed(text):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("model error")
            return [0.1, 0.2, 0.3, 0.4]

        storage._generate_embedding = mock_embed

        result = await storage.update_memories_batch([m1, m2])

        # h1 failed embedding, h2 succeeded
        assert result == [False, True]
        entity_data = storage._call_client.call_args[1]["data"]
        assert len(entity_data) == 1
        assert entity_data[0]["id"] == "h2"
