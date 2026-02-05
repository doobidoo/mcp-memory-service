"""
Unit tests for hybrid search utilities.

Tests the pure functions: keyword extraction, RRF scoring,
adaptive alpha calculation, and recency decay.
"""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from mcp_memory_service.utils.hybrid_search import (
    STOP_WORDS,
    apply_recency_decay,
    combine_results_rrf,
    extract_query_keywords,
    get_adaptive_alpha,
    rrf_score,
)

# =============================================================================
# Keyword Extraction Tests
# =============================================================================


class TestExtractQueryKeywords:
    """Tests for extract_query_keywords function."""

    def test_basic_tokenization(self):
        """Should tokenize and lowercase query."""
        result = extract_query_keywords("Rathole Project Architecture")
        assert "rathole" in result
        assert "project" in result
        assert "architecture" in result

    def test_stop_word_removal(self):
        """Should remove common stop words."""
        result = extract_query_keywords("the quick brown fox")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_punctuation_handling(self):
        """Should split on punctuation."""
        result = extract_query_keywords("python,api,bug-fix")
        assert "python" in result
        assert "api" in result
        assert "bug" in result
        assert "fix" in result

    def test_short_tokens_removed(self):
        """Should remove single-character tokens."""
        result = extract_query_keywords("a b c python d")
        assert len([r for r in result if len(r) == 1]) == 0
        assert "python" in result

    def test_empty_query(self):
        """Should handle empty query."""
        result = extract_query_keywords("")
        assert result == []

    def test_only_stop_words(self):
        """Should return empty list if query is all stop words."""
        result = extract_query_keywords("the is a an")
        assert result == []

    def test_tag_validation_filters(self):
        """Should filter to only matching tags when existing_tags provided."""
        existing_tags = {"python", "api", "memory"}
        result = extract_query_keywords("python api bug fix memory", existing_tags=existing_tags)
        assert "python" in result
        assert "api" in result
        assert "memory" in result
        assert "bug" not in result  # Not in existing_tags
        assert "fix" not in result  # Not in existing_tags

    def test_tag_validation_case_insensitive(self):
        """Should match tags case-insensitively."""
        existing_tags = {"Python", "API"}
        result = extract_query_keywords("python api test", existing_tags=existing_tags)
        assert "python" in result
        assert "api" in result

    def test_deduplication(self):
        """Should deduplicate keywords."""
        result = extract_query_keywords("python python python api")
        assert result.count("python") == 1


# =============================================================================
# RRF Score Tests
# =============================================================================


class TestRRFScore:
    """Tests for rrf_score function."""

    def test_rank_1_default_k(self):
        """Rank 1 with k=60 should give 1/61."""
        score = rrf_score(1)
        assert math.isclose(score, 1 / 61, rel_tol=1e-9)

    def test_rank_1_custom_k(self):
        """Rank 1 with k=30 should give 1/31."""
        score = rrf_score(1, k=30)
        assert math.isclose(score, 1 / 31, rel_tol=1e-9)

    def test_rank_10(self):
        """Rank 10 should give 1/70."""
        score = rrf_score(10)
        assert math.isclose(score, 1 / 70, rel_tol=1e-9)

    def test_invalid_rank_zero(self):
        """Rank 0 should return 0."""
        score = rrf_score(0)
        assert score == 0.0

    def test_invalid_rank_negative(self):
        """Negative rank should return 0."""
        score = rrf_score(-1)
        assert score == 0.0

    def test_large_rank(self):
        """Large rank should give small but non-zero score."""
        score = rrf_score(1000)
        assert math.isclose(score, 1 / 1060, rel_tol=1e-9)
        assert score > 0


# =============================================================================
# Combine Results RRF Tests
# =============================================================================


class TestCombineResultsRRF:
    """Tests for combine_results_rrf function."""

    def _make_memory(self, content_hash: str, updated_at: str = None):
        """Create a mock Memory object."""
        memory = MagicMock()
        memory.content_hash = content_hash
        memory.updated_at_iso = updated_at or datetime.now().isoformat()
        return memory

    def _make_query_result(self, content_hash: str, similarity: float):
        """Create a mock MemoryQueryResult."""
        result = MagicMock()
        result.memory = self._make_memory(content_hash)
        result.similarity_score = similarity
        return result

    def test_alpha_1_pure_vector(self):
        """Alpha=1.0 should only use vector scores."""
        vector_results = [
            self._make_query_result("hash1", 0.9),
            self._make_query_result("hash2", 0.8),
        ]
        tag_matches = [self._make_memory("hash3")]

        results = combine_results_rrf(vector_results, tag_matches, alpha=1.0)

        # hash3 should have zero score (tags ignored)
        hash3_result = next((r for r in results if r[0].content_hash == "hash3"), None)
        assert hash3_result[1] == 0.0

        # hash1 should be ranked highest
        assert results[0][0].content_hash == "hash1"

    def test_alpha_0_pure_tags(self):
        """Alpha=0.0 should only use tag scores."""
        vector_results = [
            self._make_query_result("hash1", 0.9),
        ]
        tag_matches = [self._make_memory("hash2")]

        results = combine_results_rrf(vector_results, tag_matches, alpha=0.0)

        # hash1 should have zero vector contribution
        hash1_result = next(r for r in results if r[0].content_hash == "hash1")
        assert hash1_result[2]["vector_rrf"] > 0  # Still calculated
        assert hash1_result[2]["tag_boost"] == 0.0

        # hash2 should have tag contribution only
        hash2_result = next(r for r in results if r[0].content_hash == "hash2")
        assert hash2_result[1] > 0  # Has tag score

    def test_overlap_handling(self):
        """Same memory in both lists should have combined score."""
        vector_results = [
            self._make_query_result("hash1", 0.9),
        ]
        # Create tag match with same hash
        tag_memory = self._make_memory("hash1")
        tag_matches = [tag_memory]

        results = combine_results_rrf(vector_results, tag_matches, alpha=0.5)

        # Should only have one result
        assert len(results) == 1
        result = results[0]

        # Should have both vector and tag contributions
        assert result[2]["vector_rrf"] > 0
        assert result[2]["tag_boost"] > 0

    def test_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        # Empty vector results
        results = combine_results_rrf([], [self._make_memory("hash1")], alpha=0.5)
        assert len(results) == 1

        # Empty tag matches
        results = combine_results_rrf([self._make_query_result("hash1", 0.9)], [], alpha=0.5)
        assert len(results) == 1

        # Both empty
        results = combine_results_rrf([], [], alpha=0.5)
        assert len(results) == 0

    def test_debug_info_populated(self):
        """Debug info should be populated for all results."""
        vector_results = [self._make_query_result("hash1", 0.9)]
        tag_matches = [self._make_memory("hash2")]

        results = combine_results_rrf(vector_results, tag_matches, alpha=0.7)

        for _memory, _score, debug in results:
            assert "vector_score" in debug
            assert "vector_rank" in debug
            assert "vector_rrf" in debug
            assert "tag_boost" in debug
            assert "final_score" in debug
            assert "alpha_used" in debug
            assert debug["alpha_used"] == 0.7


# =============================================================================
# Adaptive Alpha Tests
# =============================================================================


class TestGetAdaptiveAlpha:
    """Tests for get_adaptive_alpha function."""

    def _make_config(self, hybrid_alpha=None, threshold_small=500, threshold_large=5000):
        """Create a mock HybridSearchSettings."""
        config = MagicMock()
        config.hybrid_alpha = hybrid_alpha
        config.adaptive_threshold_small = threshold_small
        config.adaptive_threshold_large = threshold_large
        return config

    def test_small_corpus_balanced(self):
        """Corpus < 500 should get alpha=0.5 (balanced)."""
        config = self._make_config()
        alpha = get_adaptive_alpha(100, 0, config)
        assert alpha == 0.5

    def test_medium_corpus_semantic_biased(self):
        """Corpus 500-5000 should get alpha=0.7 (semantic-biased)."""
        config = self._make_config()
        alpha = get_adaptive_alpha(1000, 0, config)
        assert alpha == 0.7

    def test_large_corpus_strong_semantic(self):
        """Corpus >= 5000 should get alpha=0.8 (strong semantic)."""
        config = self._make_config()
        alpha = get_adaptive_alpha(10000, 0, config)
        assert alpha == 0.8

    def test_tag_match_boost(self):
        """3+ matching tags should boost tag weight by 1.5x."""
        config = self._make_config()

        # Medium corpus without tag boost
        alpha_no_boost = get_adaptive_alpha(1000, 2, config)
        assert alpha_no_boost == 0.7

        # Medium corpus with tag boost (3 matches)
        alpha_boosted = get_adaptive_alpha(1000, 3, config)
        # Tag weight = 1 - 0.7 = 0.3, boosted = 0.45, new alpha = 0.55
        assert math.isclose(alpha_boosted, 0.55, rel_tol=1e-9)

    def test_explicit_alpha_override(self):
        """Explicit alpha in config should override adaptive logic."""
        config = self._make_config(hybrid_alpha=0.9)
        alpha = get_adaptive_alpha(100, 5, config)  # Would be 0.5 with boost otherwise
        assert alpha == 0.9

    def test_boundary_values(self):
        """Test exact boundary values."""
        config = self._make_config()

        # Exactly at small threshold
        alpha = get_adaptive_alpha(500, 0, config)
        assert alpha == 0.7  # >= 500 means medium

        # Exactly at large threshold
        alpha = get_adaptive_alpha(5000, 0, config)
        assert alpha == 0.8  # >= 5000 means large


# =============================================================================
# Recency Decay Tests
# =============================================================================


class TestApplyRecencyDecay:
    """Tests for apply_recency_decay function."""

    def _make_result(self, content_hash: str, score: float, days_ago: int = 0):
        """Create a (memory, score, debug_info) tuple."""
        memory = MagicMock()
        memory.content_hash = content_hash
        # Use UTC-aware datetime with 'Z' suffix to match production data format
        updated_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
        memory.updated_at_iso = updated_at.isoformat().replace("+00:00", "Z")
        return (memory, score, {"vector_score": score})

    def test_decay_formula_70_days(self):
        """70 days old with decay=0.01 should give ~0.5x multiplier."""
        results = [self._make_result("hash1", 1.0, days_ago=70)]
        decayed = apply_recency_decay(results, decay_rate=0.01)

        # exp(-0.01 * 70) = exp(-0.7) ≈ 0.4966
        expected_factor = math.exp(-0.01 * 70)
        assert math.isclose(decayed[0][1], expected_factor, rel_tol=0.01)
        assert math.isclose(decayed[0][2]["recency_factor"], expected_factor, rel_tol=0.01)

    def test_decay_disabled_zero_rate(self):
        """Decay rate 0 should not modify scores."""
        results = [self._make_result("hash1", 1.0, days_ago=365)]
        decayed = apply_recency_decay(results, decay_rate=0)

        assert decayed[0][1] == 1.0
        assert decayed[0][2]["recency_factor"] == 1.0

    def test_fresh_memory_boost(self):
        """Fresh memories should rank higher than old ones."""
        results = [
            self._make_result("old", 1.0, days_ago=100),
            self._make_result("new", 1.0, days_ago=1),
        ]
        decayed = apply_recency_decay(results, decay_rate=0.01)

        # New should be ranked first after decay
        assert decayed[0][0].content_hash == "new"
        assert decayed[1][0].content_hash == "old"

    def test_very_old_memory(self):
        """Very old memories should still have positive score."""
        results = [self._make_result("ancient", 1.0, days_ago=365)]
        decayed = apply_recency_decay(results, decay_rate=0.01)

        # exp(-0.01 * 365) ≈ 0.026
        assert decayed[0][1] > 0
        assert decayed[0][1] < 0.1

    def test_debug_info_updated(self):
        """Debug info should include recency fields."""
        results = [self._make_result("hash1", 1.0, days_ago=30)]
        decayed = apply_recency_decay(results, decay_rate=0.01)

        debug = decayed[0][2]
        assert "recency_factor" in debug
        assert "days_old" in debug
        assert "final_score" in debug
        assert math.isclose(debug["days_old"], 30, abs_tol=0.1)


# =============================================================================
# Stop Words Tests
# =============================================================================


class TestStopWords:
    """Tests for STOP_WORDS constant."""

    def test_common_words_included(self):
        """Common English words should be in stop words."""
        common = ["the", "is", "a", "an", "and", "or", "but", "in", "on", "at"]
        for word in common:
            assert word in STOP_WORDS, f"'{word}' should be a stop word"

    def test_meaningful_words_excluded(self):
        """Technical/meaningful words should not be stop words."""
        technical = ["python", "api", "memory", "database", "error", "config"]
        for word in technical:
            assert word not in STOP_WORDS, f"'{word}' should not be a stop word"

    def test_is_frozenset(self):
        """STOP_WORDS should be immutable frozenset."""
        assert isinstance(STOP_WORDS, frozenset)
