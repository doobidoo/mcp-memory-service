"""Tests for temporal edges (RFC #1008 §4)."""

import time
import pytest
from unittest.mock import AsyncMock


class TestTemporalEdgeStorage:
    def test_import(self):
        from mcp_memory_service.reasoning.temporal import TemporalEdge
        assert TemporalEdge is not None

    def test_invalid_temporal_range_raises(self):
        from mcp_memory_service.reasoning.temporal import TemporalEdge
        now = time.time()
        with pytest.raises(ValueError, match="valid_from cannot be greater"):
            TemporalEdge(source="a", target="b", valid_from=now, valid_until=now - 86400)

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

    @pytest.mark.asyncio
    async def test_store_does_not_mutate_caller_metadata(self):
        from mcp_memory_service.reasoning.temporal import store_temporal_association
        graph = AsyncMock()
        graph.store_association = AsyncMock(return_value=True)
        original_meta = {"key": "value"}
        now = time.time()
        await store_temporal_association(
            graph, source_hash="a", target_hash="b",
            similarity=0.8, connection_types=["semantic"],
            valid_from=now, metadata=original_meta,
        )
        assert "valid_from" not in original_meta

    @pytest.mark.asyncio
    async def test_store_invalid_range_raises(self):
        from mcp_memory_service.reasoning.temporal import store_temporal_association
        graph = AsyncMock()
        now = time.time()
        with pytest.raises(ValueError):
            await store_temporal_association(
                graph, source_hash="a", target_hash="b",
                similarity=0.8, connection_types=["semantic"],
                valid_from=now, valid_until=now - 86400,
            )


class TestPointInTimeQuery:
    def test_filter_by_as_of(self):
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

    def test_filter_before_valid_from(self):
        from mcp_memory_service.reasoning.temporal import filter_temporal_edges, TemporalEdge
        now = time.time()
        edges = [
            TemporalEdge(source="a", target="b", valid_from=now + 86400, valid_until=None),
            TemporalEdge(source="a", target="c", valid_from=now - 3600, valid_until=None),
        ]
        active = filter_temporal_edges(edges, as_of=now)
        assert len(active) == 1
        assert active[0].target == "c"

    def test_no_filter_without_as_of(self):
        from mcp_memory_service.reasoning.temporal import filter_temporal_edges, TemporalEdge
        edges = [
            TemporalEdge(source="a", target="b", valid_from=0, valid_until=1),
            TemporalEdge(source="a", target="c", valid_from=None, valid_until=None),
        ]
        active = filter_temporal_edges(edges, as_of=None)
        assert len(active) == 2


class TestTemporalContradictionAwareness:
    def test_superseded_is_evolution(self):
        from mcp_memory_service.reasoning.temporal import classify_temporal_relationship, TemporalEdge
        now = time.time()
        edge_a = TemporalEdge(source="a", target="b", valid_until=now - 86400)
        edge_b = TemporalEdge(source="a", target="c", valid_from=now - 86400)
        result = classify_temporal_relationship(edge_a, edge_b)
        assert result == "evolution"

    def test_overlapping_is_contradiction(self):
        from mcp_memory_service.reasoning.temporal import classify_temporal_relationship, TemporalEdge
        now = time.time()
        edge_a = TemporalEdge(source="a", target="b", valid_until=None)
        edge_b = TemporalEdge(source="a", target="c", valid_from=now - 86400)
        result = classify_temporal_relationship(edge_a, edge_b)
        assert result == "contradiction"
