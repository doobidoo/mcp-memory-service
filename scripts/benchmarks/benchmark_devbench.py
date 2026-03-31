#!/usr/bin/env python3
"""DevBench: Practical benchmark for MCP Memory Service using project-derived memories."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

from locomo_evaluator import (
    BenchmarkResult,
    aggregate_results,
    mrr,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)

_DATASET_PATH = os.path.join(os.path.dirname(__file__), "devbench_dataset.json")


def load_devbench_dataset(path=None):
    """Load DevBench dataset from JSON. Returns dict with 'memories' and 'queries'."""
    dataset_path = path or _DATASET_PATH
    with open(dataset_path, encoding="utf-8") as fh:
        return json.load(fh)


def create_isolated_storage():
    """Create isolated SQLite-Vec storage in temp dir. Returns (storage, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="devbench-")
    db_path = os.path.join(tmp_dir, "memories.db")
    storage = SqliteVecMemoryStorage(db_path)
    return storage, tmp_dir


async def ingest_memories(storage, memories):
    """Ingest all memories into storage. Returns count of successfully stored memories."""
    stored = 0
    for mem in memories:
        content_hash = hashlib.sha256(mem["content"].encode("utf-8")).hexdigest()
        created_at = None
        if mem.get("created_at_iso"):
            try:
                dt = datetime.fromisoformat(mem["created_at_iso"].replace("Z", "+00:00"))
                created_at = int(dt.timestamp())
            except ValueError:
                pass

        memory = Memory(
            content=mem["content"],
            content_hash=content_hash,
            tags=mem.get("tags", []),
            memory_type=mem.get("memory_type"),
            created_at=created_at,
        )
        success, _msg = await storage.store(memory, skip_semantic_dedup=True)
        if success:
            stored += 1
    return stored


def match_by_tags(retrieved_results, expected_tags):
    """Match retrieved memories against expected bench tags.

    Returns (retrieved_labels, relevant_labels).
    relevant_labels is the set of expected bench: tags.
    retrieved_labels maps each result to a matched tag or irrelevant_N.
    """
    relevant_labels = set(expected_tags)
    retrieved_labels = []
    for i, result in enumerate(retrieved_results):
        tags = result.memory.tags or []
        matched_label = None
        for expected_tag in expected_tags:
            if expected_tag in tags:
                matched_label = expected_tag
                break
        if matched_label:
            retrieved_labels.append(matched_label)
        else:
            retrieved_labels.append(f"irrelevant_{i}")
    return retrieved_labels, relevant_labels


def false_positive_rate(retrieved_results, k):
    """Fraction of top-K results that contain any bench: tag.

    Used for negative queries where expected_tags is empty.
    """
    top_k = retrieved_results[:k]
    if not top_k:
        return 0.0
    false_positives = sum(
        1
        for r in top_k
        if any(t.startswith("bench:") for t in (r.memory.tags or []))
    )
    return false_positives / k


async def evaluate_retrieval(storage, queries, top_k=None):
    """For each query: search memories, match by tags, compute metrics.

    Negative queries use false_positive_rate. Returns list of per-query metric dicts.
    """
    if top_k is None:
        top_k = [5]
    max_k = max(top_k)
    per_query = []

    for query in queries:
        category = query["category"]
        expected_tags = query.get("expected_tags", [])
        results = await storage.retrieve(query["question"], n_results=max_k)

        if category == "negative":
            metrics = {"category": category}
            for k in top_k:
                metrics[f"false_positive_rate_at_{k}"] = false_positive_rate(results, k)
        else:
            retrieved_labels, relevant_labels = match_by_tags(results, expected_tags)
            metrics = {"category": category}
            for k in top_k:
                metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
                metrics[f"precision_at_{k}"] = precision_at_k(retrieved_labels, relevant_labels, k)
            metrics["mrr"] = mrr(retrieved_labels, relevant_labels)

        per_query.append(metrics)

    return per_query


async def evaluate_ablation(storage, queries, top_k=None):
    """Compare retrieval configurations: baseline, +quality_boost, +quality_w0.5.

    Returns list of per-query metric dicts with 'config_name' key.
    """
    if top_k is None:
        top_k = [5]
    max_k = max(top_k)

    configs = [
        ("baseline", "retrieve", {}),
        ("+quality_boost", "retrieve_with_quality_boost", {}),
        ("+quality_w0.5", "retrieve_with_quality_boost", {"quality_weight": 0.5}),
    ]

    all_results = []

    for config_name, method_name, extra_kwargs in configs:
        retrieve_fn = getattr(storage, method_name)
        for query in queries:
            category = query["category"]
            expected_tags = query.get("expected_tags", [])
            results = await retrieve_fn(query["question"], n_results=max_k, **extra_kwargs)

            if category == "negative":
                metrics = {"config_name": config_name, "category": category}
                for k in top_k:
                    metrics[f"false_positive_rate_at_{k}"] = false_positive_rate(results, k)
            else:
                retrieved_labels, relevant_labels = match_by_tags(results, expected_tags)
                metrics = {"config_name": config_name, "category": category}
                for k in top_k:
                    metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
                    metrics[f"precision_at_{k}"] = precision_at_k(
                        retrieved_labels, relevant_labels, k
                    )
                metrics["mrr"] = mrr(retrieved_labels, relevant_labels)

            all_results.append(metrics)

    return all_results


def _separate_negative(per_query):
    """Split per-query results into normal and negative lists."""
    normal = [q for q in per_query if q.get("category") != "negative"]
    negative = [q for q in per_query if q.get("category") == "negative"]
    return normal, negative


def _aggregate_negative(negative_results, top_k):
    """Average false_positive_rate across negative queries."""
    if not negative_results:
        return {}
    agg = {}
    for k in top_k:
        key = f"false_positive_rate_at_{k}"
        vals = [q[key] for q in negative_results if key in q]
        agg[key] = sum(vals) / len(vals) if vals else 0.0
    return agg


def format_results_table(result):
    """Terminal table output."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"  DevBench Practical Benchmark -- {result.mode.upper()} mode")
    lines.append("=" * 60)

    lines.append("\nOverall Metrics:")
    lines.append(f"  {'Metric':<30} {'Value':>10}")
    lines.append("  " + "-" * 42)
    for key, val in sorted(result.overall.items()):
        lines.append(f"  {key:<30} {val:>10.4f}")

    if result.config.get("negative_metrics"):
        lines.append("\nNegative Query Metrics:")
        lines.append(f"  {'Metric':<30} {'Value':>10}")
        lines.append("  " + "-" * 42)
        for key, val in sorted(result.config["negative_metrics"].items()):
            lines.append(f"  {key:<30} {val:>10.4f}")

    if result.by_category:
        lines.append("\nBy Category:")
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        header = f"  {'Category':<20}" + "".join(f"  {k:<22}" for k in all_metric_keys)
        lines.append(header)
        lines.append("  " + "-" * (20 + 24 * len(all_metric_keys)))
        for cat, metrics in sorted(result.by_category.items()):
            row = f"  {cat:<20}" + "".join(
                f"  {metrics.get(k, 0.0):<22.4f}" for k in all_metric_keys
            )
            lines.append(row)

    lines.append("\n" + "=" * 60 + "\n")
    return "\n".join(lines)


def format_results_markdown(result):
    """Markdown table output."""
    lines = []
    lines.append(f"## DevBench Practical Benchmark -- {result.mode.upper()} mode")
    lines.append("")
    lines.append("### Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, val in sorted(result.overall.items()):
        lines.append(f"| {key} | {val:.4f} |")
    lines.append("")

    if result.config.get("negative_metrics"):
        lines.append("### Negative Query Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, val in sorted(result.config["negative_metrics"].items()):
            lines.append(f"| {key} | {val:.4f} |")
        lines.append("")

    if result.by_category:
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        lines.append("### By Category")
        lines.append("")
        header = "| Category | " + " | ".join(all_metric_keys) + " |"
        separator = "|----------|" + "|".join(["--------"] * len(all_metric_keys)) + "|"
        lines.append(header)
        lines.append(separator)
        for cat, metrics in sorted(result.by_category.items()):
            row = (
                f"| {cat} | "
                + " | ".join(f"{metrics.get(k, 0.0):.4f}" for k in all_metric_keys)
                + " |"
            )
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


async def run_benchmark(args):
    """Full pipeline: load dataset, ingest memories, evaluate queries, aggregate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    top_k = args.top_k if args.top_k else [5]
    data_path = getattr(args, "data_path", None)

    logger.info("Loading DevBench dataset...")
    dataset = load_devbench_dataset(data_path)
    memories = dataset["memories"]
    queries = dataset["queries"]
    logger.info("Loaded %d memories, %d queries", len(memories), len(queries))

    storage, tmp_dir = create_isolated_storage()
    try:
        await storage.initialize()
        count = await ingest_memories(storage, memories)
        logger.info("Ingested %d memories", count)

        ablation = getattr(args, "ablation", False)

        if ablation:
            per_query = await evaluate_ablation(storage, queries, top_k=top_k)
        else:
            per_query = await evaluate_retrieval(storage, queries, top_k=top_k)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    config = {
        "top_k": top_k,
        "num_memories": len(memories),
        "num_queries": len(queries),
        "mode": "ablation" if ablation else "retrieval",
    }

    if ablation:
        from collections import defaultdict

        by_config = defaultdict(list)
        for q in per_query:
            by_config[q["config_name"]].append(q)

        results = []
        for config_name, questions in by_config.items():
            normal, negative = _separate_negative(questions)
            neg_metrics = _aggregate_negative(negative, top_k)
            cfg = dict(config)
            cfg["negative_metrics"] = neg_metrics
            r = aggregate_results(
                normal,
                conversation_id="devbench",
                mode=f"ablation:{config_name}",
                config=cfg,
            )
            results.append(r)
        return results

    normal, negative = _separate_negative(per_query)
    neg_metrics = _aggregate_negative(negative, top_k)
    config["negative_metrics"] = neg_metrics

    result = aggregate_results(
        normal,
        conversation_id="devbench",
        mode="retrieval",
        config=config,
    )
    return result


def parse_args(argv=None):
    """CLI args: --data-path, --top-k, --ablation, --markdown, --output-dir"""
    parser = argparse.ArgumentParser(
        description="Run the DevBench practical benchmark against MCP Memory Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to devbench_dataset.json. Defaults to bundled dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5],
        help="Top-K values for retrieval metrics (default: 5)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run ablation study comparing retrieval configurations",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=False,
        help="Output results as Markdown table",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON results (optional)",
    )
    return parser.parse_args(argv)


def main():
    """Entry point."""
    args = parse_args()
    result = asyncio.run(run_benchmark(args))

    results = result if isinstance(result, list) else [result]

    for r in results:
        if args.markdown:
            print(format_results_markdown(r))
        else:
            print(format_results_table(r))

    if args.output_dir:
        import datetime as dt_module

        os.makedirs(args.output_dir, exist_ok=True)
        for r in results:
            timestamp = dt_module.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"devbench_{r.mode}_{timestamp}.json"
            filepath = os.path.join(args.output_dir, filename)
            data = {
                "mode": r.mode,
                "overall": r.overall,
                "by_category": r.by_category,
                "config": r.config,
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
