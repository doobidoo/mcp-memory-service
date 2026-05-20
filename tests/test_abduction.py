"""Tests for abductive reasoning — find probable causes from effects."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.reasoning.inference import SemanticReasoner


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.find_connected = AsyncMock(return_value=[])
    graph.shortest_path = AsyncMock()
    graph.transitive_closure = AsyncMock(return_value=[])
    graph.common_neighbors = AsyncMock(return_value=[])
    return graph


@pytest.fixture
def reasoner(mock_graph):
    return SemanticReasoner(mock_graph)


class TestAbductionBasic:
    """Test basic abductive reasoning."""

    @pytest.mark.asyncio
    async def test_known_cause_effect_edges(self, reasoner, mock_graph):
        """A cause connected via 'causes' incoming edge is found."""
        # effect_hash has cause_A incoming via 'causes'
        mock_graph.find_connected.side_effect = [
            [("cause_A", 1)],  # causes incoming
            [],                # fixes incoming
            [],                # cause_A outgoing causes (siblings)
            [],                # cause_A outgoing fixes (siblings)
        ]

        results = await reasoner.abduct("effect_hash")
        assert len(results) == 1
        assert results[0]["cause_hash"] == "cause_A"
        assert results[0]["confidence"] == 0.5
        assert results[0]["evidence_count"] == 1

    @pytest.mark.asyncio
    async def test_no_causes_returns_empty(self, reasoner, mock_graph):
        """No incoming edges means empty result."""
        mock_graph.find_connected.side_effect = [
            [],  # causes incoming
            [],  # fixes incoming
        ]

        results = await reasoner.abduct("orphan_hash")
        assert results == []

    @pytest.mark.asyncio
    async def test_fixes_edge_also_found(self, reasoner, mock_graph):
        """A cause connected via 'fixes' incoming edge is found."""
        mock_graph.find_connected.side_effect = [
            [],                # causes incoming
            [("fix_A", 1)],    # fixes incoming
            [],                # fix_A outgoing causes
            [],                # fix_A outgoing fixes
        ]

        results = await reasoner.abduct("effect_hash")
        assert len(results) == 1
        assert results[0]["cause_hash"] == "fix_A"


class TestAbductionConfidenceBoosting:
    """Test confidence boosting when multiple effects share same cause."""

    @pytest.mark.asyncio
    async def test_shared_effects_boost_confidence(self, reasoner, mock_graph):
        """Confidence increases when a cause has multiple sibling effects."""
        mock_graph.find_connected.side_effect = [
            [("cause_A", 1)],              # causes incoming
            [],                             # fixes incoming
            [("sibling1", 1), ("sibling2", 1), ("sibling3", 1)],  # cause_A outgoing causes
            [],                             # cause_A outgoing fixes
        ]

        results = await reasoner.abduct("effect_hash")
        assert len(results) == 1
        # 3 siblings → confidence = 0.5 + 0.1*3 = 0.8
        assert results[0]["confidence"] == 0.8
        assert results[0]["evidence_count"] == 4  # 1 + 3 siblings
        assert len(results[0]["shared_effects"]) == 3

    @pytest.mark.asyncio
    async def test_confidence_capped_at_one(self, reasoner, mock_graph):
        """Confidence never exceeds 1.0."""
        mock_graph.find_connected.side_effect = [
            [("cause_A", 1)],  # causes incoming
            [],                # fixes incoming
            [("s1", 1), ("s2", 1), ("s3", 1), ("s4", 1), ("s5", 1), ("s6", 1)],  # 6 siblings
            [],                # cause_A outgoing fixes
        ]

        results = await reasoner.abduct("effect_hash")
        # 6 siblings → 0.5 + 0.6 = 1.1 → capped at 1.0
        assert results[0]["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_causes_ranked_by_confidence(self, reasoner, mock_graph):
        """Multiple causes are ranked by confidence (most evidence first)."""
        # Use a single cause from 'causes' and one from 'fixes' to guarantee order
        mock_graph.find_connected.side_effect = [
            [("cause_A", 1)],  # causes incoming
            [("cause_B", 1)],  # fixes incoming
            # cause_A siblings (outgoing causes) — less evidence
            [("s1", 1), ("s2", 1)],
            # cause_A outgoing fixes
            [],
            # cause_B siblings (outgoing causes) — more evidence
            [("s3", 1), ("s4", 1), ("s5", 1), ("s6", 1)],
            # cause_B outgoing fixes
            [],
        ]

        results = await reasoner.abduct("effect_hash")
        assert len(results) == 2
        # cause_B has more evidence → higher confidence → ranked first
        assert results[0]["cause_hash"] == "cause_B"
        assert results[0]["confidence"] == 0.9  # 0.5 + 0.1*4
        assert results[1]["cause_hash"] == "cause_A"
        assert results[1]["confidence"] == 0.7  # 0.5 + 0.1*2


class TestAbductionMaxDepth:
    """Test max_depth parameter."""

    @pytest.mark.asyncio
    async def test_max_depth_passed_to_method(self, reasoner, mock_graph):
        """max_depth parameter is accepted without error."""
        mock_graph.find_connected.side_effect = [
            [],  # causes incoming
            [],  # fixes incoming
        ]

        # Should not raise
        results = await reasoner.abduct("effect_hash", max_depth=4)
        assert results == []

    @pytest.mark.asyncio
    async def test_effect_excluded_from_shared_effects(self, reasoner, mock_graph):
        """The original effect is not listed in shared_effects."""
        mock_graph.find_connected.side_effect = [
            [("cause_A", 1)],                          # causes incoming
            [],                                         # fixes incoming
            [("effect_hash", 1), ("sibling1", 1)],     # cause_A outgoing causes (includes effect itself)
            [],                                         # cause_A outgoing fixes
        ]

        results = await reasoner.abduct("effect_hash")
        assert "effect_hash" not in results[0]["shared_effects"]
        assert results[0]["shared_effects"] == ["sibling1"]
