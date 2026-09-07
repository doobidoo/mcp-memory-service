"""Tests for composite scoring opt-in wiring in memory_explore/memory_detail."""

import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.server.handlers.graph import (
    _build_knowledge_map,
    _hydrate_chunks,
)


@pytest.fixture
def sample_entities():
    return [{"entity_name": "python", "count": 5}]


@pytest.fixture
def sample_chunks():
    return [
        {"hash": "h1", "content": "Python is great.", "relevance": 0.9},
        {"hash": "h2", "content": "Python async.", "relevance": 0.7},
    ]


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    graph.get_entity_profile = AsyncMock(return_value={"entity_types": ["language"]})
    graph.find_memories_by_entity = AsyncMock(return_value=["h1", "h2"])
    graph.find_connected = AsyncMock(return_value=[("h2", 1), ("h3", 2)])
    return graph


class TestBuildKnowledgeMapScoring:
    @pytest.mark.asyncio
    async def test_default_no_scoring_fields(self, mock_graph, sample_entities, sample_chunks):
        """When scoring=None, chunks have NO composite_score field."""
        result = await _build_knowledge_map(mock_graph, sample_entities, sample_chunks, 3)
        for chunk in result[0]["top_chunks"]:
            assert "composite_score" not in chunk
            assert "score_components" not in chunk

    @pytest.mark.asyncio
    async def test_composite_adds_fields(self, mock_graph, sample_entities, sample_chunks):
        """When scoring='composite', chunks gain composite_score + score_components."""
        result = await _build_knowledge_map(
            mock_graph, sample_entities, sample_chunks, 3, scoring="composite"
        )
        for chunk in result[0]["top_chunks"]:
            assert "composite_score" in chunk
            assert "score_components" in chunk
            assert "relevance" in chunk  # preserved
            assert set(chunk["score_components"].keys()) == {"relevance", "proximity", "centrality"}

    @pytest.mark.asyncio
    async def test_composite_preserves_relevance(self, mock_graph, sample_entities, sample_chunks):
        """Original relevance field is never removed."""
        result = await _build_knowledge_map(
            mock_graph, sample_entities, sample_chunks, 3, scoring="composite"
        )
        chunks = result[0]["top_chunks"]
        assert all(c["relevance"] is not None for c in chunks)

    @pytest.mark.asyncio
    async def test_composite_no_graph_fallback(self, sample_entities, sample_chunks):
        """When graph=None, composite scoring falls back gracefully (no crash)."""
        result = await _build_knowledge_map(
            None, sample_entities, sample_chunks, 3, scoring="composite"
        )
        # Should still produce composite fields (with 0 proximity/centrality)
        for chunk in result[0]["top_chunks"]:
            assert "composite_score" in chunk
            assert chunk["score_components"]["proximity"] == 0.0


class TestProximityFailureIsVisible:
    """A failed proximity lookup drops a scoring term; it must not be silent."""

    @pytest.mark.asyncio
    async def test_build_knowledge_map_logs_the_failure_at_debug(
        self, mock_graph, sample_entities, sample_chunks, caplog
    ):
        mock_graph.find_connected = AsyncMock(side_effect=RuntimeError("graph offline"))

        with caplog.at_level(
            logging.DEBUG, logger="mcp_memory_service.server.handlers.graph"
        ):
            result = await _build_knowledge_map(
                mock_graph, sample_entities, sample_chunks, 3, scoring="composite"
            )

        # Still scores, just without the proximity term.
        assert all("composite_score" in c for c in result[0]["top_chunks"])
        assert any(
            "Proximity lookup failed" in r.message and "python" in r.message
            for r in caplog.records
        ), caplog.text

    @pytest.mark.asyncio
    async def test_log_value_is_sanitized(
        self, mock_graph, sample_chunks, caplog
    ):
        """An entity name is user-controlled, so it cannot forge a log line."""
        mock_graph.find_connected = AsyncMock(side_effect=RuntimeError("graph offline"))
        entities = [{"entity_name": "py\nINFO: forged", "count": 5}]

        with caplog.at_level(
            logging.DEBUG, logger="mcp_memory_service.server.handlers.graph"
        ):
            await _build_knowledge_map(
                mock_graph, entities, sample_chunks, 3, scoring="composite"
            )

        records = [r for r in caplog.records if "Proximity lookup failed" in r.message]
        assert records
        assert "\n" not in records[0].message


class TestHydrateChunksScoring:
    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        mem1 = MagicMock()
        mem1.content = "Memory one content"
        mem1.quality_score = 0.8
        mem2 = MagicMock()
        mem2.content = "Memory two content"
        mem2.quality_score = 0.6

        async def get_by_hash(h):
            return {"h1": mem1, "h2": mem2}.get(h)

        storage.get_by_hash = get_by_hash
        return storage

    @pytest.mark.asyncio
    async def test_default_no_scoring(self, mock_storage):
        """Default: no composite fields."""
        chunks = await _hydrate_chunks(mock_storage, ["h1", "h2"])
        for c in chunks:
            assert "composite_score" not in c
            assert set(c.keys()) == {"hash", "content", "relevance"}

    @pytest.mark.asyncio
    async def test_composite_adds_fields(self, mock_storage, mock_graph):
        """With scoring='composite', composite fields are added."""
        chunks = await _hydrate_chunks(
            mock_storage, ["h1", "h2"], scoring="composite", graph=mock_graph, entity_degree=5
        )
        for c in chunks:
            assert "composite_score" in c
            assert "score_components" in c
            assert "relevance" in c  # preserved

    @pytest.mark.asyncio
    async def test_composite_without_graph(self, mock_storage):
        """Composite scoring without graph: proximity=0, still works."""
        chunks = await _hydrate_chunks(
            mock_storage, ["h1", "h2"], scoring="composite", graph=None, entity_degree=3
        )
        for c in chunks:
            assert c["score_components"]["proximity"] == 0.0
            assert c["composite_score"] > 0  # relevance still contributes
