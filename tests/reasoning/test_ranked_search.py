"""Tests for multi-signal ranked search (RFC #1008 §2).

Defines expected behavior for mode="ranked" in search_memories.
"""

import math
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestRankedSearchWeights:
    """Test weight normalization and defaults."""

    def test_import(self):
        from mcp_memory_service.reasoning.ranked_search import RankedSearchWeights
        assert RankedSearchWeights is not None

    def test_defaults_sum_to_one(self):
        from mcp_memory_service.reasoning.ranked_search import RankedSearchWeights
        w = RankedSearchWeights()
        assert abs((w.semantic + w.time_decay + w.access_frequency + w.quality) - 1.0) < 0.001

    def test_custom_weights_normalized(self):
        from mcp_memory_service.reasoning.ranked_search import RankedSearchWeights
        w = RankedSearchWeights(semantic=6, time_decay=2, access_frequency=1, quality=1)
        n = w.normalized()
        assert abs((n.semantic + n.time_decay + n.access_frequency + n.quality) - 1.0) < 0.001
        assert abs(n.semantic - 0.6) < 0.001

    def test_from_dict(self):
        from mcp_memory_service.reasoning.ranked_search import RankedSearchWeights
        w = RankedSearchWeights.from_dict({"semantic": 0.5, "time_decay": 0.3, "access_frequency": 0.1, "quality": 0.1})
        assert abs(w.semantic - 0.5) < 0.001


class TestComputeRankedScore:
    """Test the scoring function."""

    def test_pure_semantic(self):
        """With w1=1.0 and others=0, score should equal semantic."""
        from mcp_memory_service.reasoning.ranked_search import compute_ranked_score, RankedSearchWeights
        w = RankedSearchWeights(semantic=1.0, time_decay=0.0, access_frequency=0.0, quality=0.0)
        mem = MagicMock(
            metadata={"access_count": 5, "last_accessed_at": None, "quality_score": 0.8},
            created_at=1000.0,
        )
        mem.quality_score = 0.8
        mem.access_count = 5
        mem.last_accessed_at = None
        score, breakdown = compute_ranked_score(0.85, mem, weights=w)
        assert abs(score - 0.85) < 0.01

    def test_access_count_boosts(self):
        """Higher access_count should produce higher score (all else equal)."""
        from mcp_memory_service.reasoning.ranked_search import compute_ranked_score, RankedSearchWeights
        import time
        w = RankedSearchWeights(semantic=0.0, time_decay=0.0, access_frequency=1.0, quality=0.0)
        now = time.time()

        mem_low = MagicMock(quality_score=0.5, access_count=1, last_accessed_at=now, created_at=now)
        mem_high = MagicMock(quality_score=0.5, access_count=50, last_accessed_at=now, created_at=now)

        score_low, _ = compute_ranked_score(0.5, mem_low, weights=w, now=now)
        score_high, _ = compute_ranked_score(0.5, mem_high, weights=w, now=now)
        assert score_high > score_low

    def test_recent_memory_scores_higher(self):
        """Recently accessed memory should score higher on time_decay."""
        from mcp_memory_service.reasoning.ranked_search import compute_ranked_score, RankedSearchWeights
        import time
        w = RankedSearchWeights(semantic=0.0, time_decay=1.0, access_frequency=0.0, quality=0.0)
        now = time.time()

        mem_old = MagicMock(quality_score=0.5, access_count=1, last_accessed_at=now - 86400*60, created_at=now - 86400*60)
        mem_new = MagicMock(quality_score=0.5, access_count=1, last_accessed_at=now - 3600, created_at=now - 3600)

        score_old, _ = compute_ranked_score(0.5, mem_old, weights=w, now=now)
        score_new, _ = compute_ranked_score(0.5, mem_new, weights=w, now=now)
        assert score_new > score_old

    def test_breakdown_contains_all_signals(self):
        """Breakdown dict should contain all 4 signal scores."""
        from mcp_memory_service.reasoning.ranked_search import compute_ranked_score
        import time
        now = time.time()
        mem = MagicMock(quality_score=0.7, access_count=3, last_accessed_at=now, created_at=now)
        _, breakdown = compute_ranked_score(0.8, mem, now=now)
        assert "semantic_score" in breakdown
        assert "time_decay_score" in breakdown
        assert "access_score" in breakdown
        assert "quality_score" in breakdown
        assert "ranked_score" in breakdown


class TestApplyRankedRerank:
    """Test the reranking function on a list of candidates."""

    def test_reranks_by_composite_score(self):
        """Candidates should be reordered by ranked score."""
        from mcp_memory_service.reasoning.ranked_search import apply_ranked_rerank
        import time
        now = time.time()

        # Candidate A: high semantic but old
        a = MagicMock()
        a.relevance_score = 0.9
        a.memory = MagicMock(quality_score=0.3, access_count=0, last_accessed_at=now - 86400*90, created_at=now - 86400*90)
        a.debug_info = None

        # Candidate B: lower semantic but recent + high quality
        b = MagicMock()
        b.relevance_score = 0.7
        b.memory = MagicMock(quality_score=0.9, access_count=20, last_accessed_at=now - 60, created_at=now - 3600)
        b.debug_info = None

        results = apply_ranked_rerank([a, b], now=now)
        # B should rank higher due to recency + quality + access
        assert results[0] is b

    def test_preserves_debug_info(self):
        """Reranking should add debug info to each result."""
        from mcp_memory_service.reasoning.ranked_search import apply_ranked_rerank
        import time
        now = time.time()

        r = MagicMock()
        r.relevance_score = 0.8
        r.memory = MagicMock(quality_score=0.5, access_count=2, last_accessed_at=now, created_at=now)
        r.debug_info = None

        results = apply_ranked_rerank([r], now=now)
        assert results[0].debug_info is not None
        assert results[0].debug_info.get("ranked") is True
