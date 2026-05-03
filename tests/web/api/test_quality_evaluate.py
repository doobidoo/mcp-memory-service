"""Tests for POST /api/quality/memories/{hash}/evaluate.

Regression coverage for issue #797 — the endpoint must not pass the memory's
own content as the relevance query when no body is supplied. Doing so collapsed
the relevance prompt to "rate how relevant X is to X", causing every memory
to score 1.0.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.web.api.quality import router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import require_write_access
from mcp_memory_service.models.memory import Memory


@pytest.fixture
def mock_memory():
    return Memory(
        content="A long memory body that would have been used as a self-relevance query.",
        content_hash="abc123",
        tags=["test"],
    )


@pytest.fixture
def mock_storage(mock_memory):
    storage = MagicMock()
    storage.get_by_hash = AsyncMock(return_value=mock_memory)
    storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))
    return storage


@pytest.fixture
def app(mock_storage):
    app = FastAPI()
    app.include_router(router, prefix="/api/quality")
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[require_write_access] = lambda: None
    return app


def _patch_scorer(captured_queries: list, score: float = 0.6):
    """Patch QualityScorer.calculate_quality_score to capture the query argument."""

    async def fake_calculate(memory, query):
        captured_queries.append(query)
        memory.metadata['quality_score'] = score
        memory.metadata['quality_provider'] = 'openai_compatible'
        memory.metadata['quality_components'] = {'implicit_score': 0.5}
        memory.metadata['ai_scores'] = [{'score': score}]
        return score

    return patch(
        'mcp_memory_service.web.api.quality.QualityScorer',
        return_value=MagicMock(calculate_quality_score=AsyncMock(side_effect=fake_calculate)),
    )


def test_evaluate_with_no_body_uses_empty_query(app):
    """No body → scorer must be called with empty query, NOT memory.content[:200] (#797)."""
    captured = []
    with _patch_scorer(captured):
        client = TestClient(app)
        response = client.post("/api/quality/memories/abc123/evaluate")

    assert response.status_code == 200, response.text
    assert captured == [""], f"Expected empty query, got {captured!r}"


def test_evaluate_with_explicit_query_passes_through(app):
    """Body with non-empty query → that query reaches the scorer unchanged."""
    captured = []
    with _patch_scorer(captured):
        client = TestClient(app)
        response = client.post(
            "/api/quality/memories/abc123/evaluate",
            json={"query": "python decorators"},
        )

    assert response.status_code == 200, response.text
    assert captured == ["python decorators"]


def test_evaluate_with_whitespace_query_treated_as_empty(app):
    """Body with whitespace-only query → empty query (force absolute-quality prompt)."""
    captured = []
    with _patch_scorer(captured):
        client = TestClient(app)
        response = client.post(
            "/api/quality/memories/abc123/evaluate",
            json={"query": "   "},
        )

    assert response.status_code == 200, response.text
    assert captured == [""]
