"""Tests for cascading search fallback (issue #873)."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mcp import types

from mcp_memory_service.server.handlers.memory import handle_memory_search


@pytest.fixture
def mock_server():
    """Create a mock server with storage."""
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock()
    return server


@pytest.fixture
def mock_storage(mock_server):
    """Create mock storage with configurable search results."""
    storage = AsyncMock()
    mock_server._ensure_storage_initialized.return_value = storage
    return storage


@pytest.mark.asyncio
async def test_fallback_disabled_by_default(mock_server, mock_storage):
    """When fallback=False (default), no fallback is attempted."""
    mock_storage.search_memories.return_value = {
        "memories": [],
        "total": 0,
        "query": "socrates session",
        "mode": "semantic",
    }

    result = await handle_memory_search(mock_server, {
        "query": "socrates session",
    })

    # Only one call to search_memories (no fallback)
    assert mock_storage.search_memories.call_count == 1
    assert "No memories found" in result[0].text


@pytest.mark.asyncio
async def test_fallback_triggers_on_sparse_results(mock_server, mock_storage):
    """When fallback=True and results are sparse, BM25 and tag fallback are tried."""
    # First call (semantic): returns 1 low-score result
    semantic_result = {
        "memories": [
            {"content": "partial match", "content_hash": "aaa", "similarity_score": 0.2,
             "tags": [], "created_at_iso": "2026-01-01T00:00:00Z"}
        ],
        "total": 1,
        "query": "socrates session",
        "mode": "semantic",
    }
    # Second call (exact fallback): returns 1 new result
    exact_result = {
        "memories": [
            {"content": "session on socrates machine", "content_hash": "bbb",
             "tags": ["session", "socrates"], "created_at_iso": "2026-01-02T00:00:00Z"}
        ],
        "total": 1,
        "query": "socrates session",
        "mode": "exact",
    }
    # Third call (tag fallback): returns 1 more
    tag_result = {
        "memories": [
            {"content": "socrates hardware info", "content_hash": "ccc",
             "tags": ["socrates", "hardware"], "created_at_iso": "2026-01-03T00:00:00Z"}
        ],
        "total": 1,
        "mode": "semantic",
    }

    mock_storage.search_memories.side_effect = [semantic_result, exact_result, tag_result]

    result = await handle_memory_search(mock_server, {
        "query": "socrates session",
        "fallback": True,
    })

    # Three calls: semantic + exact + tag
    assert mock_storage.search_memories.call_count == 3
    assert "fallback: exact+tag" in result[0].text
    assert "3 memories" in result[0].text


@pytest.mark.asyncio
async def test_fallback_not_triggered_when_results_sufficient(mock_server, mock_storage):
    """When semantic returns enough high-score results, no fallback."""
    mock_storage.search_memories.return_value = {
        "memories": [
            {"content": f"result {i}", "content_hash": f"hash{i}", "similarity_score": 0.8,
             "tags": [], "created_at_iso": "2026-01-01T00:00:00Z"}
            for i in range(5)
        ],
        "total": 5,
        "query": "python patterns",
        "mode": "semantic",
    }

    result = await handle_memory_search(mock_server, {
        "query": "python patterns",
        "fallback": True,
    })

    # Only one call — no fallback needed
    assert mock_storage.search_memories.call_count == 1
    assert "fallback" not in result[0].text


@pytest.mark.asyncio
async def test_fallback_deduplicates_results(mock_server, mock_storage):
    """Fallback doesn't return duplicate memories."""
    shared_memory = {
        "content": "found in both", "content_hash": "same_hash",
        "similarity_score": 0.3, "tags": ["test"],
        "created_at_iso": "2026-01-01T00:00:00Z"
    }

    mock_storage.search_memories.side_effect = [
        {"memories": [shared_memory], "total": 1, "query": "test", "mode": "semantic"},
        {"memories": [shared_memory], "total": 1, "query": "test", "mode": "exact"},
        {"memories": [], "total": 0, "mode": "semantic"},
    ]

    result = await handle_memory_search(mock_server, {
        "query": "test query",
        "fallback": True,
    })

    # Should only have 1 result (deduplicated)
    assert "1 memories" in result[0].text


@pytest.mark.asyncio
async def test_fallback_marks_match_method(mock_server, mock_storage):
    """Each result is tagged with its match_method."""
    mock_storage.search_memories.side_effect = [
        {"memories": [
            {"content": "semantic hit", "content_hash": "s1", "similarity_score": 0.3,
             "tags": [], "created_at_iso": "2026-01-01T00:00:00Z"}
        ], "total": 1, "query": "test", "mode": "semantic"},
        {"memories": [
            {"content": "exact hit", "content_hash": "e1",
             "tags": [], "created_at_iso": "2026-01-02T00:00:00Z"}
        ], "total": 1, "query": "test", "mode": "exact"},
        {"memories": [], "total": 0, "mode": "semantic"},
    ]

    result = await handle_memory_search(mock_server, {
        "query": "test query",
        "fallback": True,
    })

    text = result[0].text
    assert "via semantic" in text
    assert "via exact_fallback" in text


@pytest.mark.asyncio
async def test_fallback_only_for_semantic_hybrid_modes(mock_server, mock_storage):
    """Fallback is not triggered for exact mode searches."""
    mock_storage.search_memories.return_value = {
        "memories": [],
        "total": 0,
        "query": "test",
        "mode": "exact",
    }

    result = await handle_memory_search(mock_server, {
        "query": "test",
        "mode": "exact",
        "fallback": True,
    })

    # Only one call — fallback doesn't apply to exact mode
    assert mock_storage.search_memories.call_count == 1
