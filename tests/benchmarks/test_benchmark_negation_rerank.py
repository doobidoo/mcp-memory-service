"""Tests for the negation-vs-rerank benchmark (#1674): shape, pairing, guards.

Model-free by design: the ranking functions are replaced, so these tests check
the accounting that a wrong number would hide behind — which member is credited,
where pool coverage is measured, and that a fail-open stage aborts instead of
being scored as a treatment.
"""
import asyncio
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import benchmark_negation_rerank as bench
import pytest


def test_negation_pairs_are_well_formed():
    data = bench.load_pairs()
    assert len(data["pairs"]) == 50
    assert len(data["distractors"]) == 15
    seen = set()
    for pair in data["pairs"]:
        assert set(pair) >= {"id", "pos", "neg", "q_pos", "q_neg", "style"}
        assert pair["id"] not in seen
        seen.add(pair["id"])
        assert pair["pos"] != pair["neg"]
        assert pair["style"] in {"explicit", "implicit"}
        assert pair["q_pos"] and pair["q_neg"]
    implicit = [pair for pair in data["pairs"] if pair["style"] == "implicit"]
    assert len(implicit) == 25


def test_cosine_is_bounded_and_symmetric():
    assert bench._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert bench._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert bench._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert bench._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert bench._cosine([1.0, 2.0], [3.0, 4.0]) == pytest.approx(bench._cosine([3.0, 4.0], [1.0, 2.0]))


def test_cosine_order_ranks_the_closest_vector_first():
    vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0]}
    order = bench.cosine_order([1.0, 0.0], vectors)
    assert order[0] == "a"
    assert order[-1] == "c"
    assert set(order) == {"a", "b", "c"}


def test_pairwise_credit_follows_the_member_each_query_asks_about():
    data = {
        "pairs": [
            {"id": "p1", "pos": "alpha is on", "neg": "alpha is not on", "q_pos": "is alpha on?",
             "q_neg": "alpha is not on, right?", "style": "explicit"},
        ]
    }
    members = {"p1": ("pos-1", "neg-1")}

    async def perfect(query):
        # Both queries get their own member first: full credit.
        order = ["pos-1", "neg-1"] if query == "is alpha on?" else ["neg-1", "pos-1"]
        return order, {"pos-1", "neg-1"}, 0.0

    scored = asyncio.run(bench.score_condition(perfect, data, members, corpus_size=2))
    assert scored["pairwise_all"] == 1.0
    assert scored["pairwise_pos"] == 1.0
    assert scored["pairwise_neg"] == 1.0
    assert scored["top1_pos"] == 1.0
    assert scored["decisions_in_pool"] == 2  # both decisions of the single pair

    async def inverted(query):
        # Cosine-like failure: the affirmative sentence wins both queries.
        order = ["pos-1", "neg-1"]
        return order, {"pos-1", "neg-1"}, 0.0

    scored = asyncio.run(bench.score_condition(inverted, data, members, corpus_size=2))
    assert scored["pairwise_pos"] == 1.0
    assert scored["pairwise_neg"] == 0.0
    assert scored["pairwise_all"] == 0.5
    assert scored["neg_explicit"] == 0.0


def test_pool_coverage_uses_the_pre_rerank_pool():
    """A member outside the pool cannot be reordered; coverage must say so."""
    data = {
        "pairs": [
            {"id": "in", "q_pos": "q-in", "q_neg": "q-in-neg", "style": "explicit"},
            {"id": "out", "q_pos": "q-out", "q_neg": "q-out-neg", "style": "implicit"},
        ]
    }
    members = {"in": ("p-in", "n-in"), "out": ("p-out", "n-out")}

    async def ranker(query):
        order = ["p-in", "n-in", "p-out", "n-out"]
        pool = {"p-in", "n-in"} if query.startswith("q-in") else {"p-out"}
        return order, pool, 0.0

    scored = asyncio.run(bench.score_condition(ranker, data, members, corpus_size=4))
    # Both members inside the pool for the first pair, only one for the second.
    assert scored["decisions_in_pool"] == 2  # both decisions of the single pair


def test_paired_bootstrap_counts_and_interval():
    baseline = {("p", 1): 0.0, ("p", 2): 1.0, ("p", 3): 0.0, ("p", 4): 1.0}
    treatment = {("p", 1): 1.0, ("p", 2): 1.0, ("p", 3): 0.0, ("p", 4): 0.0}
    stats = bench.paired_bootstrap(baseline, treatment, resamples=2000)
    assert stats["improved"] == 1
    assert stats["regressed"] == 1
    assert stats["queries"] == 4
    assert stats["delta"] == pytest.approx(0.0)
    low, high = stats["ci95"]
    assert low <= stats["delta"] <= high
    assert math.isclose(bench.paired_bootstrap(baseline, treatment, resamples=2000)["delta"], 0.0)


def test_fail_open_rerank_aborts_instead_of_scoring_the_baseline(monkeypatch):
    async def degraded(query, memories, config):
        return None

    monkeypatch.setattr(bench, "rerank_candidates", degraded)
    vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1]}
    with pytest.raises(RuntimeError, match="degraded"):
        asyncio.run(bench.rerank_order("q", [1.0, 0.0], vectors, {"a": "A", "b": "B"},
                                       config=None, pool_size=2))


def test_rerank_order_keeps_the_tail_in_baseline_order(monkeypatch):
    async def reversed_pool(query, memories, config):
        return list(reversed(memories))

    monkeypatch.setattr(bench, "rerank_candidates", reversed_pool)
    vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0]}
    order, pool, _latency = asyncio.run(
        bench.rerank_order("q", [1.0, 0.0], vectors, {k: k.upper() for k in vectors}, None, 2)
    )
    assert pool == {"a", "b"}
    assert order[:2] == ["b", "a"]
    assert order[2:] == ["c"]


def _scored(**overrides):
    base = {column: 0.0 for column in (
        "pairwise_all", "pairwise_pos", "pairwise_neg", "neg_explicit", "neg_implicit",
        "top1_pos", "top1_neg", "decisions_in_pool", "p50_ms", "p95_ms",
    )}
    base["decisions"] = {}
    base.update(overrides)
    return base


def test_storage_is_built_with_the_requested_model(monkeypatch):
    """P1 guard: the model must reach the constructor, not just the environment.

    ``SqliteVecMemoryStorage`` defaults to the English encoder, so a benchmark
    that only exports ``MCP_EMBEDDING_MODEL`` measures that default twice and
    calls it a cross-model result.
    """
    captured = {}

    class FakeStorage:
        def __init__(self, db_path, embedding_model="all-MiniLM-L6-v2"):
            captured["model"] = embedding_model
            captured["path"] = db_path
            self.embedding_backend_degraded = False

        async def initialize(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(bench, "SqliteVecMemoryStorage", FakeStorage)
    _storage, directory = asyncio.run(
        bench.isolated_storage("probe-", "paraphrase-multilingual-MiniLM-L12-v2")
    )
    try:
        assert captured["model"] == "paraphrase-multilingual-MiniLM-L12-v2"
        assert captured["path"].endswith("memories.db")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_json_out_goes_through_the_cli_writer(tmp_path, monkeypatch):
    """P2 guard: exercise the real ``--json-out`` path, not a hand-built dict."""
    async def fake_run_negation(candidates, embedding_model):
        info = {"embedding_model": embedding_model, "rerank_model": "probe",
                "corpus_size": 3, "queries": 2}
        results = {
            "cosine": _scored(pairwise_all=0.5, decisions={("p", "pos"): 0.0}),
            "rerank@40": _scored(pairwise_all=1.0, decisions={("p", "pos"): 1.0}),
        }
        bootstrap = {"rerank@40": {"delta": 0.5, "ci95": [0.1, 0.9],
                                   "improved": 1, "regressed": 0, "queries": 1}}
        return info, results, bootstrap

    monkeypatch.setattr(bench, "run_negation", fake_run_negation)
    target = tmp_path / "out.json"
    code = asyncio.run(bench.main(["--embedding-model", "probe-model", "--json-out", str(target)]))
    assert code == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["info"]["embedding_model"] == "probe-model"
    assert payload["results"]["rerank@40"]["pairwise_all"] == 1.0
    assert payload["bootstrap"]["rerank@40"]["delta"] == 0.5
    # The decision matrix is internal bookkeeping and must not be serialised.
    assert "decisions" not in payload["results"]["cosine"]
    assert "decisions" not in payload["results"]["rerank@40"]
