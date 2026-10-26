"""Tests for the resonance-vs-cosine benchmark (#1674): dataset shape and pair scoring."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_resonance import _NEGATION_PATH, _pair_hits, _percentile


class _FakeRanker:
    """Returns a fixed ranking, so _pair_hits is tested without a model."""

    def __init__(self, ranking):
        self.vectors = dict.fromkeys(ranking)
        self._ranking = ranking

    async def search(self, query, n_results, mode):
        return [SimpleNamespace(memory=SimpleNamespace(content_hash=h)) for h in self._ranking[:n_results]]


def test_negation_pairs_are_well_formed():
    with open(_NEGATION_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = data["pairs"]
    assert len({p["id"] for p in pairs}) == len(pairs) == 50
    assert sum(p.get("style") == "implicit" for p in pairs) == 25
    for p in pairs:
        assert p["pos"] != p["neg"] and p["q_pos"] and p["q_neg"]
    assert data["distractors"]


def test_pair_hits_compares_ranks_and_top1():
    ranker = _FakeRanker(["a", "x", "b"])
    assert asyncio.run(_pair_hits(ranker, "cosine", "q", "a", "b")) == (1.0, 1.0)
    assert asyncio.run(_pair_hits(ranker, "cosine", "q", "b", "a")) == (0.0, 0.0)
    # A member missing from the ranking counts as ranked last.
    assert asyncio.run(_pair_hits(ranker, "cosine", "q", "b", "missing")) == (1.0, 0.0)


def test_percentile_picks_nearest_rank():
    values = list(range(1, 101))
    assert _percentile(values, 50) == 51  # index round(49.5) == 50
    assert _percentile(values, 95) == 95
    assert _percentile([7.0], 95) == 7.0
