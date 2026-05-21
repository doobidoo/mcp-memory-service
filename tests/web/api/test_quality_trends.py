"""Tests for GET /api/quality/trends.

Regression coverage for issue #981 — the endpoint was calling two storage
methods that don't exist on any backend (storage.recall_by_timeframe, which
is a server-tool handler not a storage method, and storage.search_all_memories,
which was never implemented), so every call returned HTTP 500 with
'... object has no attribute search_all_memories'.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.web.api.quality import router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import require_read_access


def _memory(content_hash: str, created_at: float, quality_score: float | None = 0.7) -> Memory:
    metadata = {}
    if quality_score is not None:
        metadata["quality_score"] = quality_score
    return Memory(
        content=f"memory {content_hash}",
        content_hash=content_hash,
        tags=["test"],
        created_at=created_at,
        metadata=metadata,
    )


@pytest.fixture
def mock_storage():
    now = time.time()
    one_day = 86400
    in_range = [
        _memory("in_range_1", now - 2 * one_day, quality_score=0.8),
        _memory("in_range_2", now - 5 * one_day, quality_score=0.6),
        _memory("in_range_3", now - 5 * one_day, quality_score=0.4),
        _memory("missing_score", now - 1 * one_day, quality_score=None),
    ]

    async def _range(start_time, end_time, include_embeddings=False):
        # Mirror real backends: DB-side BETWEEN, newest-first ordering.
        filtered = [m for m in in_range if start_time <= m.created_at <= end_time]
        return sorted(filtered, key=lambda m: m.created_at, reverse=True)

    storage = MagicMock()
    storage.get_memories_by_time_range = AsyncMock(side_effect=_range)
    storage.get_all_memories = MagicMock(
        side_effect=AssertionError("/trends must use get_memories_by_time_range, not get_all_memories")
    )
    storage.recall_by_timeframe = MagicMock(
        side_effect=AssertionError("/trends must not call recall_by_timeframe (issue #981)")
    )
    storage.search_all_memories = MagicMock(
        side_effect=AssertionError("/trends must not call search_all_memories (issue #981)")
    )
    return storage


@pytest.fixture
def client(mock_storage):
    app = FastAPI()
    app.include_router(router, prefix="/api/quality")
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[require_read_access] = lambda: None
    return TestClient(app)


def test_trends_returns_200_without_attribute_error(client, mock_storage):
    """Issue #981: endpoint previously 500'd with AttributeError on every backend."""
    response = client.get("/api/quality/trends?days=30")
    assert response.status_code == 200, response.text
    mock_storage.get_memories_by_time_range.assert_awaited_once()


def test_trends_passes_window_bounds_to_storage(client, mock_storage):
    """The DB-side filter must receive the timeframe bounds, not be filtered in Python."""
    client.get("/api/quality/trends?days=30")
    call = mock_storage.get_memories_by_time_range.await_args
    start = call.kwargs["start_time"]
    end = call.kwargs["end_time"]
    assert end - start == pytest.approx(30 * 86400, rel=1e-3)


def test_trends_reflects_storage_window(client):
    response = client.get("/api/quality/trends?days=30")
    payload = response.json()
    # Storage mock returns 4 in-range memories
    assert payload["total_memories"] == 4
    assert payload["days_analyzed"] == 30


def test_trends_groups_same_day_memories(client):
    response = client.get("/api/quality/trends?days=30")
    payload = response.json()
    same_day = [d for d in payload["trend_data"] if d["memory_count"] == 2]
    assert same_day, "two memories created 5 days ago should be grouped into one bucket"
    assert same_day[0]["average_quality_score"] == pytest.approx((0.6 + 0.4) / 2, rel=1e-3)


def test_trends_defaults_missing_quality_score(client):
    """Memories without an explicit quality_score should default to 0.5, not crash."""
    response = client.get("/api/quality/trends?days=7")
    assert response.status_code == 200, response.text
    payload = response.json()
    # Find the day with the score-less memory (1 day ago)
    matching = [d for d in payload["trend_data"] if d["memory_count"] == 1 and d["max_score"] == 0.5]
    assert matching, "memory without quality_score should be counted with default 0.5"
