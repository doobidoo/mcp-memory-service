"""Negation minimal pairs: second-stage rerank against plain cosine (#1674).

``benchmark_resonance.py`` fixes the negation suite that measures a *scoring
function* (resonance) against cosine. This script measures the other answer to
the same problem: keeping cosine and reordering its candidates with the
opt-in cross-encoder from ``memory_search``'s ``rerank`` flag (#1602 / #1661).

Corpus and metric are the maintainer's, unchanged
(``negation_pairs.json``): 50 minimal pairs plus 15 distractors, each pair
queried twice — affirmatively (expects ``pos``) and negated (expects ``neg``)
— 100 decisions in total. Headline metric: pairwise accuracy, i.e. whether the
member the query is about outranks its counterpart.

Conditions, per suite run:

- ``cosine``: cosine over every stored row, the same rows the rerank stage
  scores, sorted exactly as the baseline does.
- ``rerank@N``: the top N rows of that baseline reordered by
  ``quality/search_rerank.py`` — the production path, ``bge-reranker-base``
  int8, ``MCP_RERANK_CANDIDATES`` default 40.
- ``rerank@all``: upper bound, every row in the pool (the corpus is small).

Two conventions from the review of #1674 are enforced rather than described:

- **Pool coverage is measured on the pre-rerank ranking.** Reading it from the
  post-rerank order reports 100/100 by construction and measures nothing.
- **A fail-open stage aborts the run.** ``rerank_candidates`` returns ``None``
  when the model cannot load or inference raises; continuing would report the
  baseline dressed as the treatment.

Usage::

    python scripts/benchmarks/benchmark_negation_rerank.py
    python scripts/benchmarks/benchmark_negation_rerank.py --embedding-model all-MiniLM-L6-v2 \\
        --candidates 40 100 --json-out /tmp/negation-rerank.json

Pass ``--embedding-model`` to choose the index: it is handed to
``SqliteVecMemoryStorage`` as a constructor argument, so two runs with different
models really do build different indexes.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


#: The index model to build, because ``SqliteVecMemoryStorage`` takes it as a
#: constructor argument and defaults to the English model regardless of
#: ``MCP_EMBEDDING_MODEL``. Reading the environment here is a fallback only.
DEFAULT_EMBEDDING_MODEL = os.getenv("MCP_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"

from mcp_memory_service.models.memory import Memory  # noqa: E402
from mcp_memory_service.quality.search_rerank import (  # noqa: E402
    DEFAULT_RERANK_CANDIDATES,
    SearchRerankConfig,
    rerank_candidates,
)
from mcp_memory_service.storage.mixins.base import deserialize_embedding  # noqa: E402
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage  # noqa: E402

_NEGATION_PATH = os.path.join(os.path.dirname(__file__), "negation_pairs.json")

#: Same rows the rerank stage scores: live, embedded, not superseded.
_CANDIDATES_SQL = """
    SELECT m.content_hash, m.content, e.content_embedding
    FROM memories m
    JOIN memory_embeddings e ON m.id = e.rowid
    WHERE m.deleted_at IS NULL
      AND (m.embedding_pending IS NULL OR m.embedding_pending = 0)
      AND (m.superseded_by IS NULL OR m.superseded_by = '')
"""

_SUMMARY = "Negation minimal pairs: second-stage rerank against plain cosine (#1674)."

_BOOTSTRAP_RESAMPLES = 10000


def _cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def _mean(values):
    return statistics.fmean(values) if values else float("nan")


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[position]


def load_pairs(path=_NEGATION_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


async def isolated_storage(prefix, embedding_model=DEFAULT_EMBEDDING_MODEL):
    directory = tempfile.mkdtemp(prefix=prefix)
    # The model must be passed here: the default is the English encoder, so
    # setting an environment variable and omitting the argument silently builds
    # the wrong index and makes a cross-model comparison compare nothing.
    storage = SqliteVecMemoryStorage(os.path.join(directory, "memories.db"), embedding_model=embedding_model)
    await storage.initialize()
    # The hash fallback produces vectors with no semantics: every score would
    # look normal while measuring nothing.
    if getattr(storage, "embedding_backend_degraded", False):
        await storage.close()
        shutil.rmtree(directory, ignore_errors=True)
        raise RuntimeError("embedding model unavailable (hash fallback); install the ML extras")
    return storage, directory


async def _store(checkpoint, content, tags):
    memory = Memory(
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        tags=tags,
    )
    stored, message = await checkpoint.store(memory, skip_semantic_dedup=True)
    if not stored:
        raise RuntimeError(f"store failed: {message}")
    return memory.content_hash


async def seed_corpus(storage, data):
    """Store the pairs and distractors; return ``{pair_id: (pos, neg)}``."""
    members = {}
    for pair in data["pairs"]:
        members[pair["id"]] = (
            await _store(storage, pair["pos"], ["neg-bench", f"pair:{pair['id']}", "pos"]),
            await _store(storage, pair["neg"], ["neg-bench", f"pair:{pair['id']}", "neg"]),
        )
    for index, text in enumerate(data["distractors"]):
        await _store(storage, text, ["neg-bench", f"distractor:{index}"])
    return members


def read_corpus(storage):
    """``{content_hash: vector}`` and ``{content_hash: content}``."""
    vectors, contents = {}, {}
    for content_hash, content, blob in storage.conn.execute(_CANDIDATES_SQL).fetchall():
        values = deserialize_embedding(blob)
        if values is not None:
            vectors[content_hash] = values
            contents[content_hash] = content
    return vectors, contents


def cosine_order(query_embedding, vectors):
    """Whole-store cosine ranking: the baseline, and the pool the stage sees."""
    scored = sorted(
        ((content_hash, _cosine(query_embedding, vector)) for content_hash, vector in vectors.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [content_hash for content_hash, _score in scored]


async def rerank_order(query, query_embedding, vectors, contents, config, pool_size):
    """Baseline order with the top ``pool_size`` rows reordered by the stage.

    Returns ``(order, pool, latency_ms)``. ``pool`` is the *pre-rerank* top of
    the baseline, which is the only set the stage can reorder.
    """
    baseline = cosine_order(query_embedding, vectors)
    if pool_size is None:
        return baseline, set(baseline), 0.0
    pool = baseline[:pool_size]
    candidates = [{"content": contents[content_hash], "content_hash": content_hash} for content_hash in pool]
    started = time.perf_counter()
    reordered = await rerank_candidates(query, candidates, config)
    latency_ms = (time.perf_counter() - started) * 1000
    if reordered is None:
        # Fail-open would silently return the baseline as the treatment.
        raise RuntimeError(
            f"rerank stage degraded to the unreranked order for {query!r}; "
            "refusing to score it as a result"
        )
    head = [memory["content_hash"] for memory in reordered]
    tail = baseline[pool_size:]
    return head + tail, set(pool), latency_ms


async def score_condition(ranked_for, data, members, corpus_size):
    """Pairwise accuracy and its breakdown for one ranking function."""
    hits = defaultdict(list)
    latencies = []
    decisions_in_pool = 0
    decisions = {}
    for pair in data["pairs"]:
        pos_hash, neg_hash = members[pair["id"]]
        style = pair.get("style", "explicit")
        for query, want, other, key in (
            (pair["q_pos"], pos_hash, neg_hash, "pos"),
            (pair["q_neg"], neg_hash, pos_hash, "neg"),
        ):
            order, pool, latency_ms = await ranked_for(query)
            latencies.append(latency_ms)
            rank_want = order.index(want) if want in order else corpus_size
            rank_other = order.index(other) if other in order else corpus_size
            hit = float(rank_want < rank_other)
            hits[key].append(hit)
            hits[f"{key}_{style}"].append(hit)
            hits[f"top1_{key}"].append(float(order[:1] == [want]))
            decisions[(pair["id"], key)] = hit
            if want in pool and other in pool:
                # Counted per decision, on the PRE-rerank pool: the stage can
                # only reorder what the baseline already surfaced.
                decisions_in_pool += 1
    return {
        "pairwise_all": _mean(hits["pos"] + hits["neg"]),
        "pairwise_pos": _mean(hits["pos"]),
        "pairwise_neg": _mean(hits["neg"]),
        "neg_explicit": _mean(hits["neg_explicit"]),
        "neg_implicit": _mean(hits["neg_implicit"]),
        "top1_pos": _mean(hits["top1_pos"]),
        "top1_neg": _mean(hits["top1_neg"]),
        "decisions_in_pool": decisions_in_pool,
        "decisions": decisions,
        "p50_ms": round(statistics.median(latencies), 1) if any(latencies) else 0.0,
        "p95_ms": round(_percentile(latencies, 0.95), 1) if any(latencies) else 0.0,
    }


def paired_bootstrap(baseline, treatment, resamples=_BOOTSTRAP_RESAMPLES, seed=20261026):
    """Delta with a 95% CI and the improved/regressed query counts."""
    keys = sorted(baseline)
    deltas = [treatment[key] - baseline[key] for key in keys]
    if not deltas:
        raise ValueError("no decisions to bootstrap")
    generator = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = sum(deltas[generator.randrange(len(deltas))] for _ in range(len(deltas)))
        means.append(total / len(deltas))
    means.sort()
    return {
        "delta": _mean(deltas),
        "ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
        "improved": sum(1 for delta in deltas if delta > 0),
        "regressed": sum(1 for delta in deltas if delta < 0),
        "queries": len(deltas),
    }


async def run_negation(candidate_sizes, embedding_model=DEFAULT_EMBEDDING_MODEL):
    data = load_pairs()
    storage, directory = await isolated_storage("negation-rerank-", embedding_model)
    try:
        members = await seed_corpus(storage, data)
        vectors, contents = read_corpus(storage)
        corpus_size = len(vectors)
        config = SearchRerankConfig.from_env()
        info = {
            "embedding_model": storage.embedding_model_name,
            "rerank_model": config.model,
            "corpus_size": corpus_size,
            "queries": 2 * len(data["pairs"]),
        }

        async def baseline_ranker(query):
            return cosine_order(storage._generate_embedding(query), vectors), set(vectors), 0.0

        results = {"cosine": await score_condition(baseline_ranker, data, members, corpus_size)}
        for size in candidate_sizes:
            pool_size = corpus_size if size in ("all", 0) else int(size)
            label = f"rerank@{size}"

            async def ranker(query, pool_size=pool_size):
                return await rerank_order(
                    query, storage._generate_embedding(query), vectors, contents, config, pool_size
                )

            results[label] = await score_condition(ranker, data, members, corpus_size)

        bootstrap = {}
        for label, scored in results.items():
            if label == "cosine":
                continue
            bootstrap[label] = paired_bootstrap(results["cosine"]["decisions"], scored["decisions"])
        return info, results, bootstrap
    finally:
        await storage.close()
        shutil.rmtree(directory, ignore_errors=True)


def _print_table(title, info, results, bootstrap, corpus_size):
    print(f"\n{title}")
    print(f"embedding={info['embedding_model']}  rerank={info['rerank_model']}  "
          f"corpus={info['corpus_size']}  queries={info['queries']}")
    columns = ["pairwise_all", "pairwise_pos", "pairwise_neg", "neg_explicit", "neg_implicit",
               "top1_pos", "top1_neg", "decisions_in_pool", "p50_ms", "p95_ms"]
    print("| condition | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for label, scored in results.items():
        cells = [f"{scored[c]}/{info['queries']}" if c == "decisions_in_pool" else str(scored[c])
                 for c in columns]
        print(f"| {label} | " + " | ".join(cells) + " |")
    for label, stats in bootstrap.items():
        low, high = stats["ci95"]
        print(f"paired bootstrap cosine -> {label}: delta {stats['delta'] * 100:+.1f} p.p. "
              f"95% CI [{low * 100:+.1f}, {high * 100:+.1f}] over {stats['queries']} queries; "
              f"improved={stats['improved']} regressed={stats['regressed']}")


async def main(argv):
    parser = argparse.ArgumentParser(description=_SUMMARY)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"index model to build (default: {DEFAULT_EMBEDDING_MODEL}; "
                             "it is passed to the storage constructor, not exported as an env var)")
    parser.add_argument("--candidates", nargs="+", default=[str(DEFAULT_RERANK_CANDIDATES)],
                        help="candidate pool sizes to rerank (use 'all' for the whole corpus)")
    parser.add_argument("--json-out", default=None, help="write the full result to this path")
    args = parser.parse_args(argv)

    info, results, bootstrap = await run_negation(args.candidates, args.embedding_model)
    _print_table("Negation minimal pairs (cosine vs second-stage rerank)",
                 info, results, bootstrap, info["corpus_size"])

    if args.json_out:
        payload = {"info": info,
                   "results": {label: {k: v for k, v in scored.items() if k != "decisions"}
                               for label, scored in results.items()},
                   "bootstrap": bootstrap}
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
