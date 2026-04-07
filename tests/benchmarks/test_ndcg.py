"""Tests for ndcg_at_k metric."""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))

from locomo_evaluator import ndcg_at_k


class TestNdcgAtK:
    def test_perfect_ranking_single_relevant_returns_1(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_perfect_ranking_two_relevant_returns_1(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_no_relevant_items_returns_1(self):
        # Nothing to find = perfect score by convention
        assert ndcg_at_k([], set(), k=5) == 1.0
        assert ndcg_at_k(["a", "b"], set(), k=5) == 1.0

    def test_relevant_not_in_top_k_returns_0(self):
        retrieved = ["x1", "x2", "x3", "x4", "x5", "a"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=5) == 0.0

    def test_relevant_at_bottom_of_top_k_scores_lower_than_top(self):
        # "a" at position 1 (first)
        top_score = ndcg_at_k(["a", "x", "y", "z", "w"], {"a"}, k=5)
        # "a" at position 5 (last in top-5)
        bottom_score = ndcg_at_k(["x", "y", "z", "w", "a"], {"a"}, k=5)
        assert top_score > bottom_score

    def test_empty_retrieved_with_relevant_returns_0(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_k_0_returns_1_for_empty_relevant(self):
        assert ndcg_at_k([], set(), k=0) == 1.0

    def test_uses_log2_discounting(self):
        # Manual calculation: single relevant item at position 1
        # DCG = 1/log2(2) = 1.0; IDCG = 1/log2(2) = 1.0; NDCG = 1.0
        assert ndcg_at_k(["a"], {"a"}, k=1) == pytest.approx(1.0)
        # Single relevant item at position 2 (0-indexed)
        # DCG = 1/log2(3); IDCG = 1/log2(2); NDCG = log2(2)/log2(3)
        expected = math.log2(2) / math.log2(3)
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)
