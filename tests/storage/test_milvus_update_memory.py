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
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
    # Default: batch encode returns array of vectors
    storage.embedding_model.encode = MagicMock(
        return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
    )
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = False
    storage._lock = None
    # Mock _call_client as async
    storage._call_client = AsyncMock()
    # Mock _generate_embedding for update_memory (single item path)
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
    """Tests for MilvusMemoryStorage.update_memory native override.

    update_memory now delegates to update_memory_metadata with
    preserve_timestamps=False.
    """

    @pytest.mark.asyncio
    async def test_successful_update(self):
        """Normal update: delegates to update_memory_metadata, returns True."""
        storage = _make_storage()
        existing = _make_memory(content_hash="hash_abc", tags=["old_tag"])
        updated = _make_memory(content_hash="hash_abc", tags=["new_tag"], memory_type="decision")

        storage.get_by_hash = AsyncMock(return_value=existing)

        result = await storage.update_memory(updated)

        assert result is True
        # Verify upsert was called (through update_memory_metadata)
        storage._call_client.assert_called_once()
        call_args = storage._call_client.call_args
        assert call_args[0][0] == "upsert"

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self):
        """Update on non-existent hash returns False."""
        storage = _make_storage()
        memory = _make_memory(content_hash="nonexistent")

        storage.get_by_hash = AsyncMock(return_value=None)

        result = await storage.update_memory(memory)

        assert result is False

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
    async def test_refreshes_updated_at(self):
        """updated_at is refreshed (preserve_timestamps=False)."""
        storage = _make_storage()
        original_updated = 1700001000.0
        existing = _make_memory(created_at=1700000000.0, updated_at=original_updated)
        storage.get_by_hash = AsyncMock(return_value=existing)

        before = time.time()
        await storage.update_memory(_make_memory())
        after = time.time()

        entity = storage._call_client.call_args[1]["data"][0]
        # updated_at should be refreshed (newer than original)
        assert entity["updated_at"] >= before

    @pytest.mark.asyncio
    async def test_metadata_is_merged(self):
        """Metadata from the update is merged with existing, not replaced."""
        storage = _make_storage()
        existing = _make_memory(
            content_hash="hash_abc",
            metadata={"existing_key": "old_value", "keep_me": "yes"},
        )
        updated = _make_memory(
            content_hash="hash_abc",
            metadata={"existing_key": "new_value", "new_key": "added"},
        )
        storage.get_by_hash = AsyncMock(return_value=existing)

        result = await storage.update_memory(updated)

        assert result is True
        entity = storage._call_client.call_args[1]["data"][0]
        metadata = json.loads(entity["metadata"])
        # Merged: existing_key updated, keep_me preserved, new_key added
        assert metadata["existing_key"] == "new_value"
        assert metadata["keep_me"] == "yes"
        assert metadata["new_key"] == "added"


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
    async def test_batch_uses_single_get_call(self):
        """All existing records fetched in one client.get() call."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1", tags=["tag1"])
        m2 = _make_memory(content_hash="h2", tags=["tag2"])

        now = time.time()
        # Simulate batch get returning rows
        storage._call_client = AsyncMock(side_effect=[
            # First call: "get" returns existing rows
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
                {"id": "h2", "content": "c2", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            # Second call: "upsert" succeeds
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )

        result = await storage.update_memories_batch([m1, m2])

        assert result == [True, True]
        # First call is "get", second is "upsert"
        calls = storage._call_client.call_args_list
        assert calls[0][0][0] == "get"
        assert calls[0][1]["ids"] == ["h1", "h2"]
        assert calls[1][0][0] == "upsert"
        assert len(calls[1][1]["data"]) == 2

    @pytest.mark.asyncio
    async def test_batch_embedding_single_encode_call(self):
        """All embeddings generated in a single encode() call."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1")
        m2 = _make_memory(content_hash="h2")
        m3 = _make_memory(content_hash="h3")

        now = time.time()
        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
                {"id": "h2", "content": "c2", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
                {"id": "h3", "content": "c3", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]] * 3)
        )

        await storage.update_memories_batch([m1, m2, m3])

        # encode called once with all 3 contents
        storage.embedding_model.encode.assert_called_once()
        texts = storage.embedding_model.encode.call_args[0][0]
        assert texts == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_partial_failure_skips_not_found(self):
        """Memories not found in batch get are skipped (False), others succeed."""
        storage = _make_storage()
        m1 = _make_memory(content_hash="h1")
        m2 = _make_memory(content_hash="h2_missing")
        m3 = _make_memory(content_hash="h3")

        now = time.time()
        # Only h1 and h3 exist
        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
                {"id": "h3", "content": "c3", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )

        result = await storage.update_memories_batch([m1, m2, m3])

        assert result == [True, False, True]

    @pytest.mark.asyncio
    async def test_preserve_timestamps_true(self):
        """When preserve_timestamps=True, updated_at is NOT refreshed
        for non-structural changes."""
        storage = _make_storage()
        now = time.time()
        original_updated = now - 500

        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 1000, "updated_at": original_updated,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
        )

        # Same tags and memory_type → non-structural change
        m = _make_memory(content_hash="h1", tags=["test"], memory_type="note")
        await storage.update_memories_batch([m], preserve_timestamps=True)

        entity = storage._call_client.call_args_list[1][1]["data"][0]
        assert entity["updated_at"] == original_updated

    @pytest.mark.asyncio
    async def test_preserve_timestamps_false(self):
        """When preserve_timestamps=False (default), updated_at is refreshed."""
        storage = _make_storage()
        now = time.time()
        original_updated = now - 500

        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 1000, "updated_at": original_updated,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
        )

        before = time.time()
        m = _make_memory(content_hash="h1", tags=["new_tag"])
        await storage.update_memories_batch([m], preserve_timestamps=False)
        after = time.time()

        entity = storage._call_client.call_args_list[1][1]["data"][0]
        assert before <= entity["updated_at"] <= after

    @pytest.mark.asyncio
    async def test_batch_upsert_failure_returns_all_false(self):
        """When the batch upsert call fails, all results are False."""
        storage = _make_storage()
        now = time.time()

        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            Exception("network error"),
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
        )

        result = await storage.update_memories_batch([_make_memory(content_hash="h1")])

        assert result == [False]

    @pytest.mark.asyncio
    async def test_batch_fetch_failure_returns_all_false(self):
        """When the batch get call fails, all results are False."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=Exception("connection lost"))

        result = await storage.update_memories_batch([_make_memory(content_hash="h1")])

        assert result == [False]

    @pytest.mark.asyncio
    async def test_batch_embedding_failure_returns_all_false(self):
        """When batch encode fails, all results are False."""
        storage = _make_storage()
        now = time.time()

        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": "{}", "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
        ])
        storage.embedding_model.encode = MagicMock(side_effect=RuntimeError("OOM"))

        result = await storage.update_memories_batch([_make_memory(content_hash="h1")])

        assert result == [False]

    @pytest.mark.asyncio
    async def test_metadata_is_merged_not_replaced(self):
        """Metadata from updates is merged with existing metadata."""
        storage = _make_storage()
        now = time.time()

        storage._call_client = AsyncMock(side_effect=[
            [
                {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
                 "metadata": json.dumps({"keep": "yes", "old": "value"}),
                 "created_at": now - 100, "updated_at": now - 50,
                 "created_at_iso": None, "updated_at_iso": None},
            ],
            None,
        ])
        storage.embedding_model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
        )

        m = _make_memory(content_hash="h1", metadata={"old": "updated", "new": "added"})
        result = await storage.update_memories_batch([m])

        assert result == [True]
        entity = storage._call_client.call_args_list[1][1]["data"][0]
        metadata = json.loads(entity["metadata"])
        assert metadata["keep"] == "yes"
        assert metadata["old"] == "updated"
        assert metadata["new"] == "added"
