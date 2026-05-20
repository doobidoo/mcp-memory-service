"""Unit tests for MilvusMemoryStorage.mark_superseded_batch native override.

Mock-based tests — no live Milvus server required.

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
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage.embedding_model.encode = MagicMock(
        return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
    )
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = False
    storage._lock = None
    storage._call_client = AsyncMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return storage


def _make_row(
    content_hash: str = "hash_abc",
    content: str = "test content",
    metadata: Optional[Dict[str, Any]] = None,
    vector: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build a Milvus row dict."""
    now = time.time()
    return {
        "id": content_hash,
        "content": content,
        "tags": ",test,",
        "memory_type": "note",
        "metadata": json.dumps(metadata or {}),
        "created_at": now - 100,
        "updated_at": now - 50,
        "created_at_iso": None,
        "updated_at_iso": None,
        "vector": vector or [0.1, 0.2, 0.3, 0.4],
    }


# -- Tests -------------------------------------------------------------------


class TestMarkSupersededBatch:
    """Tests for MilvusMemoryStorage.mark_superseded_batch."""

    @pytest.mark.asyncio
    async def test_empty_pairs_returns_zero(self):
        """Empty input returns 0."""
        storage = _make_storage()
        result = await storage.mark_superseded_batch([])
        assert result == 0
        storage._call_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_initialized_returns_zero(self):
        """Returns 0 when storage is not initialized."""
        storage = _make_storage()
        storage._initialized = False
        result = await storage.mark_superseded_batch([("w1", "l1")])
        assert result == 0

    @pytest.mark.asyncio
    async def test_single_pair_marks_superseded(self):
        """Single (winner, loser) pair sets superseded_by in loser metadata."""
        storage = _make_storage()

        storage._call_client = AsyncMock(side_effect=[
            # First call: "get" returns the loser with vector
            [_make_row(content_hash="loser1", content="old content", vector=[0.5, 0.6, 0.7, 0.8])],
            # Second call: "upsert" succeeds
            None,
        ])

        result = await storage.mark_superseded_batch([("winner1", "loser1")])

        assert result == 1
        # Verify upsert was called with superseded_by in metadata
        upsert_call = storage._call_client.call_args_list[1]
        assert upsert_call[0][0] == "upsert"
        entity = upsert_call[1]["data"][0]
        metadata = json.loads(entity["metadata"])
        assert metadata["superseded_by"] == "winner1"
        # Vector should be reused from the existing record
        assert entity["vector"] == [0.5, 0.6, 0.7, 0.8]

    @pytest.mark.asyncio
    async def test_multiple_pairs_single_upsert(self):
        """Multiple pairs are processed in one batch upsert call."""
        storage = _make_storage()

        storage._call_client = AsyncMock(side_effect=[
            # "get" returns 3 losers with vectors
            [
                _make_row(content_hash="l1", content="c1", vector=[0.1, 0.2, 0.3, 0.4]),
                _make_row(content_hash="l2", content="c2", vector=[0.2, 0.3, 0.4, 0.5]),
                _make_row(content_hash="l3", content="c3", vector=[0.3, 0.4, 0.5, 0.6]),
            ],
            # "upsert" succeeds
            None,
        ])

        pairs = [("w1", "l1"), ("w2", "l2"), ("w3", "l3")]
        result = await storage.mark_superseded_batch(pairs)

        assert result == 3
        # Single get + single upsert = 2 calls total
        assert storage._call_client.call_count == 2
        # No embedding encode should be called (vectors reused)
        storage.embedding_model.encode.assert_not_called()
        # Verify all entities have correct superseded_by
        entities = storage._call_client.call_args_list[1][1]["data"]
        assert len(entities) == 3
        for entity in entities:
            meta = json.loads(entity["metadata"])
            assert "superseded_by" in meta

    @pytest.mark.asyncio
    async def test_loser_not_found_skipped(self):
        """Losers not found in batch fetch are skipped."""
        storage = _make_storage()

        # Only l1 exists, l2 doesn't
        storage._call_client = AsyncMock(side_effect=[
            [_make_row(content_hash="l1", content="c1")],
            None,
        ])

        pairs = [("w1", "l1"), ("w2", "l2")]
        result = await storage.mark_superseded_batch(pairs)

        assert result == 1

    @pytest.mark.asyncio
    async def test_batch_fetch_failure_returns_zero(self):
        """When batch get fails, returns 0."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=Exception("network error"))

        result = await storage.mark_superseded_batch([("w1", "l1")])

        assert result == 0

    @pytest.mark.asyncio
    async def test_vector_missing_falls_back_to_embedding(self):
        """When vector is not in fetched row, falls back to _generate_embedding."""
        storage = _make_storage()

        # Row without vector field
        row = _make_row(content_hash="l1", content="c1")
        del row["vector"]
        storage._call_client = AsyncMock(side_effect=[
            [row],
            None,
        ])

        result = await storage.mark_superseded_batch([("w1", "l1")])

        assert result == 1
        # Fallback embedding should have been called
        storage._generate_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_missing_and_embedding_fails_skips(self):
        """When vector is missing and fallback embedding fails, item is skipped."""
        storage = _make_storage()

        row = _make_row(content_hash="l1", content="c1")
        del row["vector"]
        storage._call_client = AsyncMock(side_effect=[
            [row],
        ])
        storage._generate_embedding = MagicMock(side_effect=RuntimeError("OOM"))

        result = await storage.mark_superseded_batch([("w1", "l1")])

        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_upsert_failure_returns_zero(self):
        """When upsert fails, returns 0."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=[
            [_make_row(content_hash="l1", content="c1")],
            Exception("upsert failed"),
        ])

        result = await storage.mark_superseded_batch([("w1", "l1")])

        assert result == 0

    @pytest.mark.asyncio
    async def test_preserves_existing_metadata(self):
        """Existing metadata fields are preserved, only superseded_by is added."""
        storage = _make_storage()

        existing_meta = {"quality_score": 0.7, "source": "user_input"}
        storage._call_client = AsyncMock(side_effect=[
            [_make_row(content_hash="l1", content="c1", metadata=existing_meta)],
            None,
        ])

        result = await storage.mark_superseded_batch([("w1", "l1")])

        assert result == 1
        entity = storage._call_client.call_args_list[1][1]["data"][0]
        metadata = json.loads(entity["metadata"])
        assert metadata["superseded_by"] == "w1"
        assert metadata["quality_score"] == 0.7
        assert metadata["source"] == "user_input"

    @pytest.mark.asyncio
    async def test_duplicate_loser_uses_last_winner(self):
        """If same loser appears multiple times, last winner wins."""
        storage = _make_storage()

        storage._call_client = AsyncMock(side_effect=[
            [_make_row(content_hash="l1", content="c1")],
            None,
        ])

        # Same loser, different winners — last one ("w2") should win
        pairs = [("w1", "l1"), ("w2", "l1")]
        result = await storage.mark_superseded_batch(pairs)

        assert result == 1
        entity = storage._call_client.call_args_list[1][1]["data"][0]
        metadata = json.loads(entity["metadata"])
        assert metadata["superseded_by"] == "w2"

    @pytest.mark.asyncio
    async def test_preserves_timestamps(self):
        """Timestamps are preserved (preserve_timestamps=True)."""
        storage = _make_storage()

        now = time.time()
        original_updated = now - 500
        row = _make_row(content_hash="l1", content="c1")
        row["updated_at"] = original_updated

        storage._call_client = AsyncMock(side_effect=[
            [row],
            None,
        ])

        await storage.mark_superseded_batch([("w1", "l1")])

        entity = storage._call_client.call_args_list[1][1]["data"][0]
        # updated_at should be preserved (not refreshed)
        assert entity["updated_at"] == original_updated
