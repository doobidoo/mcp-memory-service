"""Tests for temporal contradiction detection."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.consolidation.contradictions import (
    detect_contradictions,
    check_contradiction_on_store,
)


@pytest.fixture
def mock_storage():
    storage = AsyncMock()

    def _list_memories(page=1, page_size=500):
        if page == 1:
            return {
                "memories": [
                    {"content_hash": "hash_old", "content": "The sky is blue", "memory_type": "observation", "created_at": "2026-01-01T00:00:00Z", "metadata": {}},
                    {"content_hash": "hash_new", "content": "The sky is red", "memory_type": "observation", "created_at": "2026-05-01T00:00:00Z", "metadata": {}},
                    {"content_hash": "hash_unrelated", "content": "Python is great", "memory_type": "note", "created_at": "2026-03-01T00:00:00Z", "metadata": {}},
                ]
            }
        return {"memories": []}

    storage.list_memories = AsyncMock(side_effect=_list_memories)
    storage.search_memories = AsyncMock(return_value=[
        {"content_hash": "hash_new", "similarity": 0.6, "memory_type": "observation", "created_at": "2026-05-01T00:00:00Z"},
    ])
    storage.add_graph_edge = AsyncMock()
    storage.update_memory_metadata = AsyncMock()
    return storage


@pytest.fixture
def mock_storage_no_contradiction():
    storage = AsyncMock()

    def _list_memories(page=1, page_size=500):
        if page == 1:
            return {
                "memories": [
                    {"content_hash": "hash1", "content": "Hello", "memory_type": "note", "created_at": "2026-01-01T00:00:00Z", "metadata": {}},
                ]
            }
        return {"memories": []}

    storage.list_memories = AsyncMock(side_effect=_list_memories)
    storage.search_memories = AsyncMock(return_value=[
        {"content_hash": "hash1", "similarity": 1.0, "memory_type": "note", "created_at": "2026-01-01T00:00:00Z"},
    ])
    return storage


class TestDetectContradictions:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ENABLED", True)
    async def test_detects_contradiction_dry_run(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert result["pairs_detected"] >= 1
        assert result["dry_run"] is True
        assert result["edges_created"] == 0  # dry run

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ENABLED", True)
    async def test_detects_contradiction_live(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=False)
        assert result["pairs_detected"] >= 1
        assert result["edges_created"] >= 1
        assert result["superseded_marked"] >= 1
        mock_storage.add_graph_edge.assert_called()
        mock_storage.update_memory_metadata.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ENABLED", False)
    async def test_skipped_when_disabled(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert result["skipped"] is True

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ENABLED", True)
    async def test_no_contradiction_found(self, mock_storage_no_contradiction):
        result = await detect_contradictions(mock_storage_no_contradiction, dry_run=True)
        assert result["pairs_detected"] == 0

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ENABLED", True)
    async def test_empty_memories(self):
        storage = AsyncMock()
        storage.list_memories = AsyncMock(return_value={"memories": []})
        result = await detect_contradictions(storage, dry_run=True)
        assert "No memories" in result.get("message", "")


class TestCheckContradictionOnStore:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ON_STORE", True)
    async def test_finds_contradiction(self):
        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[
            {"content_hash": "existing_hash", "similarity": 0.55, "memory_type": "observation", "created_at": "2026-01-01T00:00:00Z"},
        ])
        storage.add_graph_edge = AsyncMock()
        storage.update_memory_metadata = AsyncMock()

        result = await check_contradiction_on_store(storage, "New contradicting content", "new_hash")
        assert result is not None
        assert "contradicts" in result
        storage.add_graph_edge.assert_called_once()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ON_STORE", False)
    async def test_skipped_when_disabled(self):
        storage = AsyncMock()
        result = await check_contradiction_on_store(storage, "content", "hash")
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ON_STORE", True)
    async def test_no_contradiction(self):
        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[
            {"content_hash": "other", "similarity": 0.9, "memory_type": "note", "created_at": "2026-01-01T00:00:00Z"},  # too similar (dedup zone)
        ])
        result = await check_contradiction_on_store(storage, "content", "hash")
        assert result is None
