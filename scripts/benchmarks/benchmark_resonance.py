#!/usr/bin/env python3
"""Compare resonance search (#1673, arXiv:2509.09691) with cosine retrieval.

Configurations, all against the same isolated sqlite-vec database:

- ``retrieve``: ``storage.retrieve()``, the sqlite-vec KNN users get today.
- ``cosine``: cosine similarity over the same candidate rows the resonance
  modes score. Isolates the scoring function from the candidate set; it should
  rank like ``retrieve``.
- ``resonance``, ``phase``, ``hybrid``: ``ResonanceDB.search`` in each mode.

Suites:

- ``devbench``: the 30-memory / 20-query DevBench set (Recall@K, MRR, FPR).
- ``negation``: minimal pairs from ``negation_pairs.json`` that differ only by
  an explicit negation. Headline metric: pairwise accuracy, i.e. whether the
  intended member outranks the other, split by query direction.
- ``locomo``: LoCoMo retrieval (Recall@K, MRR), per conversation.
- ``latency``: p50/p95 per request at growing store sizes, ``retrieve`` against
  the resonance endpoint's own per-request path (#1674).

Decision rule, fixed before the first run: resonance or hybrid is worth
promoting only if it is not worse than ``cosine`` on DevBench Recall@5 and MRR,
not worse on LoCoMo, and better on the negation pairs.

``_embed_to_waveform`` derives phase from the real-valued embedding, so this
measures that heuristic, not a model trained to emit phase as the paper assumes.

Usage:
    python scripts/benchmarks/benchmark_resonance.py --suite devbench negation
    python scripts/benchmarks/benchmark_resonance.py --suite locomo --conversations 2
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.mixins.base import deserialize_embedding
from mcp_memory_service.storage.resonance_db import ResonanceDB, _cosine_similarity
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

from locomo_evaluator import mrr, recall_at_k

MODES = ("retrieve", "cosine", "resonance", "phase", "hybrid")
_NEGATION_PATH = os.path.join(os.path.dirname(__file__), "negation_pairs.json")

# Same rows the /api/memories/resonance-search endpoint scores.
_CANDIDATES_SQL = """
    SELECT m.content_hash, e.content_embedding
    FROM memories m
    JOIN memory_embeddings e ON m.id = e.rowid
    WHERE m.deleted_at IS NULL
      AND (m.embedding_pending IS NULL OR m.embedding_pending = 0)
      AND (m.superseded_by IS NULL OR m.superseded_by = '')
"""


class Ranker:
    """Ranks one storage's memories in every mode; built once after ingest."""

    def __init__(self, storage):
        self.storage = storage
        dim = storage.embedding_dimension
        self.vectors = {}
        for content_hash, blob in storage.conn.execute(_CANDIDATES_SQL).fetchall():
            values = deserialize_embedding(blob)
            if values is not None and len(values) == dim:
                self.vectors[content_hash] = values
        self.db = ResonanceDB(dimension=dim)
        self.db.batch_add(list(self.vectors.items()))
        self.latency_ms = defaultdict(list)

    async def search(self, query, n_results, mode):
        """Return MemoryQueryResult objects, so the existing matchers apply."""
        start = time.perf_counter()
        if mode == "retrieve":
            results = await self.storage.retrieve(query, n_results=n_results)
        else:
            query_embedding = self.storage._generate_embedding(query)
            if mode == "cosine":
                scored = sorted(
                    ((h, _cosine_similarity(query_embedding, v)) for h, v in self.vectors.items()),
                    key=lambda item: item[1], reverse=True,
                )[:n_results]
            else:
                scored = [(r["content_hash"], r["score"])
                          for r in self.db.search(query_embedding, top_k=n_results, mode=mode)]
            results = []
            for content_hash, score in scored:
                memory = await self.storage.get_by_hash(content_hash)
                if memory is not None:
                    results.append(MemoryQueryResult(memory=memory, relevance_score=score))
        self.latency_ms[mode].append((time.perf_counter() - start) * 1000)
        return results


_EMBEDDING_INFO = {}


async def _isolated_storage(prefix):
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "memories.db"))
    await storage.initialize()
    # An empty database lets the hash fallback through; its vectors carry no
    # semantics, so every score would look normal and measure nothing.
    if getattr(storage, "embedding_backend_degraded", False):
        await storage.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("embedding model unavailable (hash fallback); install the ML extras")
    _EMBEDDING_INFO.update(model=storage.embedding_model_name, backend=type(storage.embedding_model).__name__,
                           dimension=storage.embedding_dimension)
    return storage, tmp_dir


async def _store(storage, content, tags):
    memory = Memory(content=content, content_hash=hashlib.sha256(content.encode()).hexdigest(), tags=tags)
    ok, msg = await storage.store(memory, skip_semantic_dedup=True)
    if not ok:
        raise RuntimeError(f"store failed: {msg}")
    return memory.content_hash


def _mean(values):
    return statistics.fmean(values) if values else float("nan")


# --------------------------------------------------------------------------
# DevBench
# --------------------------------------------------------------------------

async def run_devbench(top_k):
    from benchmark_devbench import false_positive_rate, ingest_memories, load_devbench_dataset, match_by_tags

    data = load_devbench_dataset()
    storage, tmp_dir = await _isolated_storage("resonance-devbench-")
    try:
        await ingest_memories(storage, data["memories"])
        ranker = Ranker(storage)
        rows = defaultdict(lambda: defaultdict(list))
        for mode in MODES:
            for query in data["queries"]:
                results = await ranker.search(query["question"], max(5, max(top_k)), mode)
                if query["category"] == "negative":
                    rows[mode]["fpr@5"].append(false_positive_rate(results, 5))
                    continue
                labels, relevant = match_by_tags(results, query.get("expected_tags", []))
                for k in top_k:
                    rows[mode][f"recall@{k}"].append(recall_at_k(labels, relevant, k))
                rows[mode]["mrr"].append(mrr(labels, relevant))
        return {mode: {key: _mean(vals) for key, vals in metrics.items()} | {"ms/query": _mean(ranker.latency_ms[mode])}
                for mode, metrics in rows.items()}
    finally:
        await storage.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Negation minimal pairs
# --------------------------------------------------------------------------

async def _pair_hits(ranker, mode, query, want, other):
    """(want outranks other, want is top-1) for one query over the whole store."""
    n_all = len(ranker.vectors)
    ranked = [r.memory.content_hash for r in await ranker.search(query, n_all, mode)]
    rank_want = ranked.index(want) if want in ranked else n_all
    rank_other = ranked.index(other) if other in ranked else n_all
    return float(rank_want < rank_other), float(ranked[:1] == [want])


async def _score_negation_mode(ranker, mode, pairs, members):
    hits = defaultdict(list)
    for pair in pairs:
        pos_hash, neg_hash = members[pair["id"]]
        pos_hit, pos_top1 = await _pair_hits(ranker, mode, pair["q_pos"], pos_hash, neg_hash)
        neg_hit, neg_top1 = await _pair_hits(ranker, mode, pair["q_neg"], neg_hash, pos_hash)
        hits["pos"].append(pos_hit)
        hits["neg"].append(neg_hit)
        hits[f"neg_{pair.get('style', 'explicit')}"].append(neg_hit)
        hits["top1_pos"].append(pos_top1)
        hits["top1_neg"].append(neg_top1)
    return {
        "pairwise_pos": _mean(hits["pos"]),
        "pairwise_neg": _mean(hits["neg"]),
        "pairwise_neg_explicit": _mean(hits["neg_explicit"]),
        "pairwise_neg_implicit": _mean(hits["neg_implicit"]),
        "pairwise_all": _mean(hits["pos"] + hits["neg"]),
        "top1_pos": _mean(hits["top1_pos"]),
        "top1_neg": _mean(hits["top1_neg"]),
        "ms/query": _mean(ranker.latency_ms[mode]),
    }


async def run_negation():
    with open(_NEGATION_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    storage, tmp_dir = await _isolated_storage("resonance-negation-")
    try:
        members = {}
        for pair in data["pairs"]:
            members[pair["id"]] = (
                await _store(storage, pair["pos"], ["neg-bench", f"pair:{pair['id']}", "pos"]),
                await _store(storage, pair["neg"], ["neg-bench", f"pair:{pair['id']}", "neg"]),
            )
        for i, text in enumerate(data["distractors"]):
            await _store(storage, text, ["neg-bench", f"distractor:{i}"])
        ranker = Ranker(storage)
        return {mode: await _score_negation_mode(ranker, mode, data["pairs"], members) for mode in MODES}
    finally:
        await storage.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Latency of the real endpoint path
# --------------------------------------------------------------------------

_SUBJECTS = ["the sync job", "the dashboard", "the HTTP server", "consolidation", "the hooks",
             "the release workflow", "the quality scorer", "the graph store", "the exporter", "OAuth"]
_VERBS = ["failed after", "was fixed by", "depends on", "was moved behind", "now logs",
          "was measured against", "times out on", "replaced", "caches", "validates"]
_OBJECTS = ["a database lock", "the Cloudflare token", "the delta cursor", "a missing index",
            "the ONNX session", "a stale manifest", "the WAL pragma", "a retry budget",
            "the SSE ticket", "an unpartitioned table"]


def _synthetic(i):
    return (f"Note {i}: {_SUBJECTS[i % 10]} {_VERBS[(i // 10) % 10]} "
            f"{_OBJECTS[(i // 100) % 10]} during run {i // 1000}.")


def _percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


async def _time_requests(storage, queries):
    """ms per request: storage.retrieve() against the resonance endpoint's own path,
    which reads every row, rebuilds the waveforms and scores them per request."""
    from mcp_memory_service.web.api.memories import _fetch_resonance_candidates, _score_resonance_candidates

    timings = defaultdict(list)
    for query in queries:
        start = time.perf_counter()
        await storage.retrieve(query, n_results=10)
        timings["retrieve"].append((time.perf_counter() - start) * 1000)
        for mode in ("resonance", "phase", "hybrid"):
            start = time.perf_counter()
            rows = _fetch_resonance_candidates(storage.conn, None)
            query_embedding = storage._generate_embedding(query)
            _score_resonance_candidates(rows, query_embedding, storage.embedding_dimension, 10, mode)
            timings[mode].append((time.perf_counter() - start) * 1000)
    return timings


async def _grow(storage, start, size):
    """Store synthetic memories start..size-1, so each size extends the last one."""
    for i in range(start, size):
        await _store(storage, _synthetic(i), ["latency"])


def _summarize(timings, size):
    return {f"{mode} @{size}": {"p50_ms": _percentile(values, 50), "p95_ms": _percentile(values, 95)}
            for mode, values in timings.items()}


async def run_latency(sizes, requests):
    """p50/p95 per request at each store size; the store grows between sizes."""
    queries = [f"{_SUBJECTS[i % 10]} {_VERBS[i % 7]} {_OBJECTS[i % 9]}" for i in range(requests)]
    storage, tmp_dir = await _isolated_storage("resonance-latency-")
    out = {}
    try:
        previous = 0
        for size in sorted(sizes):
            await _grow(storage, previous, size)
            previous = size
            out |= _summarize(await _time_requests(storage, queries), size)
            print(f"  latency: {size} memories measured", file=sys.stderr)
        return out
    finally:
        await storage.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# LoCoMo
# --------------------------------------------------------------------------

async def run_locomo(top_k, conversations):
    from benchmark_locomo import _match_evidence, ingest_conversation
    from locomo_dataset import load_dataset

    convs = load_dataset()
    if conversations:
        convs = convs[:conversations]
    rows = defaultdict(lambda: defaultdict(list))
    latency = defaultdict(list)
    for conv in convs:
        storage, tmp_dir = await _isolated_storage("resonance-locomo-")
        try:
            await ingest_conversation(storage, conv)
            ranker = Ranker(storage)
            for mode in MODES:
                for qa in conv.qa_pairs:
                    results = await ranker.search(qa.question, max(top_k), mode)
                    labels, relevant = _match_evidence(results, qa.evidence)
                    for k in top_k:
                        rows[mode][f"recall@{k}"].append(recall_at_k(labels, relevant, k))
                    rows[mode]["mrr"].append(mrr(labels, relevant))
                latency[mode].extend(ranker.latency_ms[mode])
            print(f"  {conv.sample_id}: {len(conv.qa_pairs)} questions, {len(ranker.vectors)} memories", file=sys.stderr)
        finally:
            await storage.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return {mode: {key: _mean(vals) for key, vals in metrics.items()}
            | {"questions": len(metrics["mrr"]), "ms/query": _mean(latency[mode])}
            for mode, metrics in rows.items()}


def _print_table(title, table):
    keys = sorted({k for metrics in table.values() for k in metrics})
    print(f"\n### {title}\n")
    print("| mode | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for mode in MODES:
        if mode in table:
            cells = [f"{table[mode][k]:.4f}" if isinstance(table[mode].get(k), float) else str(table[mode].get(k, ""))
                     for k in keys]
            print(f"| {mode} | " + " | ".join(cells) + " |")


async def main(args):
    results = {}
    if "devbench" in args.suite:
        results["devbench"] = await run_devbench(args.top_k)
        _print_table("DevBench", results["devbench"])
    if "negation" in args.suite:
        results["negation"] = await run_negation()
        _print_table("Negation minimal pairs", results["negation"])
    if "locomo" in args.suite:
        results["locomo"] = await run_locomo(args.top_k, args.conversations)
        _print_table("LoCoMo", results["locomo"])
    if "latency" in args.suite:
        results["latency"] = await run_latency(args.sizes, args.requests)
        print("\n### Latency per request (ms)\n")
        print("| mode @ memories | p50 | p95 |\n|---|---|---|")
        for row, values in results["latency"].items():
            print(f"| {row} | {values['p50_ms']:.1f} | {values['p95_ms']:.1f} |")
    results["embedding"] = dict(_EMBEDDING_INFO)
    print(f"\nEmbedding: {results['embedding']}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs="+", choices=["devbench", "negation", "locomo", "latency"],
                        default=["devbench", "negation"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[5])
    parser.add_argument("--conversations", type=int, default=0, help="LoCoMo: first N conversations (0 = all)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000],
                        help="Latency: store sizes to measure at")
    parser.add_argument("--requests", type=int, default=30, help="Latency: requests per mode and size")
    parser.add_argument("--output", help="Write the result tables as JSON")
    asyncio.run(main(parser.parse_args()))
