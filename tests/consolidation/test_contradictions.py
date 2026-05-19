"""Tests for temporal contradiction detection."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.consolidation.contradictions import (
    detect_contradictions,
    check_contradiction_on_store,
)


def _make_memory(content_hash, content, memory_type="observation", created_at=None, metadata=None):
    """Create a mock Memory dataclass instance."""
    m = MagicMock()
    m.content_hash = content_hash
    m.content = content
    m.memory_type = memory_type
    m.created_at = created_at or 0.0
    m.metadata = metadata or {}
    return m


@pytest.fixture
def mock_storage():
    storage = AsyncMock()

    # get_all_memories returns a list of Memory dataclass instances
    storage.get_all_memories = AsyncMock(return_value=[
        _make_memory("hash_old", "The sky is blue", "observation", created_at=1735689600.0),
        _make_memory("hash_new", "The sky is red", "observation", created_at=1746057600.0),
        _make_memory("hash_unrelated", "Python is great", "note", created_at=1740873600.0),
    ])

    # search_memories returns dict {"memories": [dict, ...]} where dicts have similarity_score
    storage.search_memories = AsyncMock(return_value={
        "memories": [
            {
                "content_hash": "hash_new",
                "similarity_score": 0.6,
                "type": "observation",
                "created_at": 1746057600.0,
            },
        ]
    })
    storage.add_graph_edge = AsyncMock()
    storage.update_memory_metadata = AsyncMock()
    return storage


@pytest.fixture
def mock_storage_no_contradiction():
    storage = AsyncMock()

    storage.get_all_memories = AsyncMock(return_value=[
        _make_memory("hash1", "Hello", "note", created_at=1735689600.0),
    ])
    # Returns only self — should be filtered out
    storage.search_memories = AsyncMock(return_value={
        "memories": [
            {"content_hash": "hash1", "similarity_score": 1.0, "type": "note", "created_at": 1735689600.0},
        ]
    })
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
        storage.get_all_memories = AsyncMock(return_value=[])
        result = await detect_contradictions(storage, dry_run=True)
        assert "No memories" in result.get("message", "")


class TestCheckContradictionOnStore:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.consolidation.contradictions.CONTRADICTION_ON_STORE", True)
    async def test_finds_contradiction(self):
        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value={
            "memories": [
                {
                    "content_hash": "existing_hash",
                    "similarity_score": 0.55,
                    "type": "observation",
                    "created_at": 1735689600.0,
                },
            ]
        })
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
        storage.search_memories = AsyncMock(return_value={
            "memories": [
                {
                    "content_hash": "other",
                    "similarity_score": 0.9,
                    "type": "note",
                    "created_at": 1735689600.0,
                },  # too similar (dedup zone, above SIMILARITY_MAX=0.75)
            ]
        })
        result = await check_contradiction_on_store(storage, "content", "hash")
        assert result is None
