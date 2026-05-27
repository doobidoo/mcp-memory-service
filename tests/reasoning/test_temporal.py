"""Tests for temporal edges (RFC #1008 §4)."""

import time
import pytest
from unittest.mock import AsyncMock


class TestTemporalEdgeStorage:
    def test_import(self):
        from mcp_memory_service.reasoning.temporal import TemporalEdge
        assert TemporalEdge is not None

    @pytest.mark.asyncio
    async def test_store_with_valid_from(self):
        from mcp_memory_service.reasoning.temporal import store_temporal_association
        graph = AsyncMock()
        graph.store_association = AsyncMock(return_value=True)
        now = time.time()
        result = await store_temporal_association(
            graph, source_hash="a", target_hash="b",
            similarity=0.8, connection_types=["semantic"],
            relationship_type="related", valid_from=now,
        )
        assert result is True
        call_kwargs = graph.store_association.call_args[1]
        assert call_kwargs["metadata"]["valid_from"] == now

    @pytest.mark.asyncio
    async def test_store_with_valid_until(self):
        from mcp_memory_service.reasoning.temporal import store_temporal_association
        graph = AsyncMock()
        graph.store_association = AsyncMock(return_value=True)
        now = time.time()
        result = await store_temporal_association(
            graph, source_hash="a", target_hash="b",
            similarity=0.8, connection_types=["version"],
            relationship_type="supersedes",
            valid_from=now - 86400, valid_until=now,
        )
        assert result is True


class TestPointInTimeQuery:
    @pytest.mark.asyncio
    async def test_filter_by_as_of(self):
        from mcp_memory_service.reasoning.temporal import filter_temporal_edges, TemporalEdge
        now = time.time()
        edges = [
            TemporalEdge(source="a", target="b", valid_from=now - 86400*30, valid_until=now - 86400*10),
            TemporalEdge(source="a", target="c", valid_from=now - 86400*5, valid_until=None),
            TemporalEdge(source="a", target="d", valid_from=None, valid_until=None),
        ]
        active = filter_temporal_edges(edges, as_of=now)
        assert len(active) == 2
        targets = [e.target for e in active]
        assert "b" not in targets
        assert "c" in targets and "d" in targets

    @pytest.mark.asyncio
    async def test_filter_before_valid_from(self):
        from mcp_memory_service.reasoning.temporal import filter_temporal_edges, TemporalEdge
        now = time.time()
        edges = [
            TemporalEdge(source="a", target="b", valid_from=now + 86400, valid_until=None),
            TemporalEdge(source="a", target="c", valid_from=now - 3600, valid_until=None),
        ]
        active = filter_temporal_edges(edges, as_of=now)
        assert len(active) == 1
        assert active[0].target == "c"

    @pytest.mark.asyncio
    async def test_no_filter_without_as_of(self):
        from mcp_memory_service.reasoning.temporal import filter_temporal_edges, TemporalEdge
        edges = [
            TemporalEdge(source="a", target="b", valid_from=0, valid_until=1),
            TemporalEdge(source="a", target="c", valid_from=None, valid_until=None),
        ]
        active = filter_temporal_edges(edges, as_of=None)
        assert len(active) == 2


class TestTemporalContradictionAwareness:
    @pytest.mark.asyncio
    async def test_superseded_is_evolution(self):
        from mcp_memory_service.reasoning.temporal import classify_temporal_relationship
        now = time.time()
        result = classify_temporal_relationship(
            edge_a_valid_until=now - 86400, edge_b_valid_from=now - 86400,
        )
        assert result == "evolution"

    @pytest.mark.asyncio
    async def test_overlapping_is_contradiction(self):
        from mcp_memory_service.reasoning.temporal import classify_temporal_relationship
        now = time.time()
        result = classify_temporal_relationship(
            edge_a_valid_until=None, edge_b_valid_from=now - 86400,
        )
        assert result == "contradiction"
