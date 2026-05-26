"""Tests for transitive closure with decay and edge-type whitelist."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.reasoning.inference import (
    SemanticReasoner,
    TRAVERSABLE_EDGE_TYPES,
    NON_TRAVERSABLE,
)


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.find_connected = AsyncMock()
    graph.shortest_path = AsyncMock()
    graph.transitive_closure = AsyncMock(return_value=[])
    return graph


@pytest.fixture
def reasoner(mock_graph):
    return SemanticReasoner(mock_graph)


class TestEdgeTypeWhitelist:
    """Test traversable vs non-traversable edge types."""

    @pytest.mark.asyncio
    async def test_non_traversable_contradicts_raises(self, reasoner):
        with pytest.raises(ValueError, match="non-traversable"):
            await reasoner.infer_transitive("contradicts")

    @pytest.mark.asyncio
    async def test_non_traversable_contradicted_by_raises(self, reasoner):
        with pytest.raises(ValueError, match="non-traversable"):
            await reasoner.infer_transitive("contradicted_by")

    @pytest.mark.asyncio
    async def test_traversable_types_accepted(self, reasoner, mock_graph):
        for edge_type in TRAVERSABLE_EDGE_TYPES:
            mock_graph.transitive_closure.return_value = []
            result = await reasoner.infer_transitive(edge_type)
            assert result == []

    def test_whitelist_constants(self):
        assert 'contradicts' in NON_TRAVERSABLE
        assert 'contradicted_by' in NON_TRAVERSABLE
        assert 'relates_to' in TRAVERSABLE_EDGE_TYPES
        assert 'causes' in TRAVERSABLE_EDGE_TYPES
        assert 'fixes' in TRAVERSABLE_EDGE_TYPES


class TestDecayWeight:
    """Test decay weight calculation."""

    @pytest.mark.asyncio
    async def test_default_decay(self, reasoner, mock_graph):
        mock_graph.transitive_closure.return_value = [
            ("a", "b", 2),
            ("a", "c", 3),
        ]
        results = await reasoner.infer_transitive("related")
        assert results == [
            ("a", "b", 2, 0.5),   # 1.0 / 2
            ("a", "c", 3, 1.0 / 3),  # 1.0 / 3
        ]

    @pytest.mark.asyncio
    async def test_custom_decay_factor(self, reasoner, mock_graph):
        mock_graph.transitive_closure.return_value = [
            ("x", "y", 2),
            ("x", "z", 4),
        ]
        results = await reasoner.infer_transitive("related", decay_factor=2.0)
        assert results == [
            ("x", "y", 2, 1.0),   # 2.0 / 2
            ("x", "z", 4, 0.5),   # 2.0 / 4
        ]

    @pytest.mark.asyncio
    async def test_result_tuple_has_four_elements(self, reasoner, mock_graph):
        mock_graph.transitive_closure.return_value = [("a", "b", 2)]
        results = await reasoner.infer_transitive("causes")
        assert len(results[0]) == 4
        src, tgt, dist, weight = results[0]
        assert src == "a"
        assert tgt == "b"
        assert dist == 2
        assert weight == 0.5


class TestEmptyGraph:
    """Test with empty graph (no edges)."""

    @pytest.mark.asyncio
    async def test_empty_results(self, reasoner, mock_graph):
        mock_graph.transitive_closure.return_value = []
        results = await reasoner.infer_transitive("related", max_hops=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_no_transitive_closure_method(self):
        graph = MagicMock(spec=[])
        graph.find_connected = AsyncMock()
        graph.shortest_path = AsyncMock()
        # No transitive_closure attribute
        reasoner = SemanticReasoner(graph)
        results = await reasoner.infer_transitive("related")
        assert results == []


class TestCycleHandling:
    """Test that cycles don't cause infinite loops (handled by SQL visited set)."""

    @pytest.mark.asyncio
    async def test_cycle_returns_finite_results(self, reasoner, mock_graph):
        # Simulate a cycle: A→B→C→A — SQL CTE handles this via visited set
        # The mock just returns what the SQL would return (finite results)
        mock_graph.transitive_closure.return_value = [
            ("a", "c", 2),
            ("b", "a", 2),
            ("c", "b", 2),
        ]
        results = await reasoner.infer_transitive("related", max_hops=3)
        assert len(results) == 3
        # Verify weights are computed correctly
        for src, tgt, dist, weight in results:
            assert weight == 1.0 / dist
