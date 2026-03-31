"""Tests for LoCoMo benchmark evaluator metrics."""

import pytest
from locomo_evaluator import (
    BenchmarkResult,
    aggregate_results,
    mrr,
    precision_at_k,
    recall_at_k,
    token_f1,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_k_limits_results(self):
        # k=2 misses "c" (relevant); k=4 finds both relevant items
        assert recall_at_k(["x", "y", "a", "b"], {"a", "b"}, k=2) == 0.0
        assert recall_at_k(["x", "y", "a", "b"], {"a", "b"}, k=4) == 1.0

    def test_empty_relevant_returns_one(self):
        assert recall_at_k(["a", "b", "c"], set(), k=3) == 1.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "b"], {"a"}) == 0.5

    def test_not_found(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_uses_first(self):
        # "b" is at position 2, "c" is at position 3 — first hit is "b" -> 0.5
        assert mrr(["x", "b", "c"], {"b", "c"}) == 0.5


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        # "she works at Google" (4 tokens) vs "Google" (1 token)
        # overlap=1, precision=1/4=0.25, recall=1/1=1.0, F1=2*0.25*1.0/1.25=0.4
        result = token_f1("she works at Google", "Google")
        assert abs(result - 0.4) < 1e-9

    def test_no_overlap(self):
        assert token_f1("hello world", "foo bar") == 0.0

    def test_case_insensitive(self):
        assert token_f1("Hello World", "hello world") == 1.0

    def test_empty_prediction(self):
        assert token_f1("", "hello world") == 0.0

    def test_empty_gold(self):
        assert token_f1("hello world", "") == 0.0


class TestAggregateResults:
    def test_aggregate_by_category(self):
        per_question = [
            {"category": "single", "f1": 1.0, "recall": 1.0},
            {"category": "single", "f1": 0.5, "recall": 0.5},
            {"category": "multi", "f1": 0.8, "recall": 0.6},
        ]
        result = aggregate_results(per_question, "conv1", "baseline", {"k": 5})

        assert isinstance(result, BenchmarkResult)
        assert result.conversation_id == "conv1"
        assert result.mode == "baseline"
        assert result.config == {"k": 5}

        # Overall averages: f1=(1.0+0.5+0.8)/3, recall=(1.0+0.5+0.6)/3
        assert abs(result.overall["f1"] - (1.0 + 0.5 + 0.8) / 3) < 1e-9
        assert abs(result.overall["recall"] - (1.0 + 0.5 + 0.6) / 3) < 1e-9

        # By category
        assert abs(result.by_category["single"]["f1"] - 0.75) < 1e-9
        assert abs(result.by_category["multi"]["f1"] - 0.8) < 1e-9

        # by_conversation keyed by conversation_id
        assert "conv1" in result.by_conversation

    def test_aggregate_empty(self):
        result = aggregate_results([], "conv_empty", "test", {})
        assert result.overall == {}
        assert result.by_category == {}
        assert result.by_conversation == {}
