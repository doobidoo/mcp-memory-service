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
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.mixins.base import deserialize_embedding
from mcp_memory_service.storage.resonance_db import ResonanceDB, _cosine_similarity

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


async def _isolated_storage(prefix):
    import tempfile
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "memories.db"))
    await storage.initialize()
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
                results = await ranker.search(query["question"], max(top_k), mode)
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
        n_all = len(ranker.vectors)

        out = {}
        for mode in MODES:
            pairwise = {"pos": [], "neg": []}
            top1 = {"pos": [], "neg": []}
            for pair in data["pairs"]:
                pos_hash, neg_hash = members[pair["id"]]
                for direction, query, want, other in (
                    ("pos", pair["q_pos"], pos_hash, neg_hash),
                    ("neg", pair["q_neg"], neg_hash, pos_hash),
                ):
                    ranked = [r.memory.content_hash for r in await ranker.search(query, n_all, mode)]
                    rank_want = ranked.index(want) if want in ranked else n_all
                    rank_other = ranked.index(other) if other in ranked else n_all
                    pairwise[direction].append(1.0 if rank_want < rank_other else 0.0)
                    top1[direction].append(1.0 if ranked[:1] == [want] else 0.0)
            out[mode] = {
                "pairwise_pos": _mean(pairwise["pos"]),
                "pairwise_neg": _mean(pairwise["neg"]),
                "pairwise_all": _mean(pairwise["pos"] + pairwise["neg"]),
                "top1_pos": _mean(top1["pos"]),
                "top1_neg": _mean(top1["neg"]),
                "ms/query": _mean(ranker.latency_ms[mode]),
            }
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
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs="+", choices=["devbench", "negation", "locomo"],
                        default=["devbench", "negation"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[5])
    parser.add_argument("--conversations", type=int, default=0, help="LoCoMo: first N conversations (0 = all)")
    parser.add_argument("--output", help="Write the result tables as JSON")
    asyncio.run(main(parser.parse_args()))
