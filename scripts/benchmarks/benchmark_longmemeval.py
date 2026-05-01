#!/usr/bin/env python3
"""LongMemEval Benchmark for MCP Memory Service."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

from longmemeval_dataset import LongMemEvalItem, load_dataset
from locomo_evaluator import (
    BenchmarkResult,
    aggregate_results,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def create_isolated_storage() -> Tuple[SqliteVecMemoryStorage, str]:
    """Create isolated SQLite-Vec storage in temp dir. Returns (storage, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="longmemeval-bench-")
    db_path = os.path.join(tmp_dir, "memories.db")
    storage = SqliteVecMemoryStorage(db_path)
    return storage, tmp_dir


async def ingest_item(storage: SqliteVecMemoryStorage, item: LongMemEvalItem) -> int:
    """Store conversation turns as memories. Returns count of stored memories."""
    stored = 0
    for session in item.sessions:
        for turn_idx, turn in enumerate(session.turns):
            content = turn.content
            if not content or not content.strip():
                continue
            content_hash = hashlib.sha256(
                f"{item.question_id}:{session.session_id}:{turn_idx}:{content}".encode()
            ).hexdigest()
            tags = ["longmemeval", item.question_id, session.session_id, turn.role]
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type="conversation_turn",
            )
            success, _ = await storage.store(memory, skip_semantic_dedup=True)
            if success:
                stored += 1
    return stored


async def ingest_item_session(storage: SqliteVecMemoryStorage, item: LongMemEvalItem) -> int:
    """Store each conversation session as one memory unit. Returns count of stored sessions."""
    stored = 0
    for session in item.sessions:
        lines = []
        for turn in session.turns:
            content = (turn.content or "").strip()
            if content:
                lines.append(f"[{turn.role}] {content}")
        if not lines:
            continue
        content = "\n".join(lines)
        content_hash = hashlib.sha256(
            f"{item.question_id}:{session.session_id}:session".encode()
        ).hexdigest()
        tags = ["longmemeval", item.question_id, session.session_id, "session"]
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=tags,
            memory_type="session",
        )
        success, _ = await storage.store(memory, skip_semantic_dedup=True)
        if success:
            stored += 1
    return stored


async def ingest_item_hybrid(storage: SqliteVecMemoryStorage, item: LongMemEvalItem) -> int:
    """Store both session units and individual turns. Returns total count."""
    session_count = await ingest_item_session(storage, item)
    turn_count = await ingest_item(storage, item)
    return session_count + turn_count


def _match_evidence(
    retrieved_results,
    answer_session_ids: List[str],
) -> Tuple[List[str], Set[str]]:
    """Match retrieved memories against evidence session IDs.

    Each evidence session can only be credited once (first occurrence in top-K).
    Subsequent retrieval of the same session are counted as irrelevant to avoid
    recall > 1.0 when multiple turns from the same session are retrieved.

    Returns (retrieved_labels, relevant_labels).
    """
    relevant_labels = set(answer_session_ids)

    retrieved_labels = []
    already_matched: Set[str] = set()
    for i, result in enumerate(retrieved_results):
        tags = result.memory.tags or []
        matched_label = None
        for session_id in answer_session_ids:
            if session_id in tags and session_id not in already_matched:
                matched_label = session_id
                already_matched.add(session_id)
                break
        if matched_label:
            retrieved_labels.append(matched_label)
        else:
            retrieved_labels.append(f"irrelevant_{i}")

    return retrieved_labels, relevant_labels


async def evaluate_retrieval(
    storage: SqliteVecMemoryStorage,
    item: LongMemEvalItem,
    top_k: Optional[List[int]] = None,
) -> Dict:
    """Search memories for the question, compute retrieval metrics.

    Returns a dict with question_type and recall/ndcg/mrr metrics.
    """
    if top_k is None:
        top_k = [5, 10]
    max_k = max(top_k)

    results = await storage.retrieve(item.question, n_results=max_k)
    retrieved_labels, relevant_labels = _match_evidence(results, item.answer_session_ids)

    metrics: Dict = {"question_type": item.question_type}
    for k in top_k:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
    metrics["ndcg_at_10"] = ndcg_at_k(retrieved_labels, relevant_labels, 10)
    metrics["mrr"] = mrr(retrieved_labels, relevant_labels)

    return metrics


async def run_ablation(
    storage: SqliteVecMemoryStorage,
    item: LongMemEvalItem,
    top_k: Optional[List[int]] = None,
) -> List[Dict]:
    """Compare retrieval configurations: baseline, +quality_boost, +quality_w0.5.

    Returns list of dicts each with config_name, question_type, and metrics.
    """
    if top_k is None:
        top_k = [5, 10]
    max_k = max(top_k)

    configs = [
        ("baseline", "retrieve", {}),
        ("+quality_boost", "retrieve_with_quality_boost", {}),
        ("+quality_w0.5", "retrieve_with_quality_boost", {"quality_weight": 0.5}),
    ]

    all_results: List[Dict] = []

    for config_name, method_name, extra_kwargs in configs:
        if not hasattr(storage, method_name):
            logger.warning("Storage does not support method %s, skipping config %s", method_name, config_name)
            continue
        retrieve_fn = getattr(storage, method_name)
        results = await retrieve_fn(item.question, n_results=max_k, **extra_kwargs)
        retrieved_labels, relevant_labels = _match_evidence(results, item.answer_session_ids)

        metrics: Dict = {"config_name": config_name, "question_type": item.question_type}
        for k in top_k:
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
        metrics["ndcg_at_10"] = ndcg_at_k(retrieved_labels, relevant_labels, 10)
        metrics["mrr"] = mrr(retrieved_labels, relevant_labels)

        all_results.append(metrics)

    return all_results


def format_results_table(result: BenchmarkResult) -> str:
    """Terminal table output."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"  LongMemEval Benchmark — {result.mode.upper()} mode")
    lines.append(f"  Dataset: {result.conversation_id}")
    lines.append("=" * 60)

    lines.append("\nOverall Metrics:")
    lines.append(f"  {'Metric':<25} {'Value':>10}")
    lines.append("  " + "-" * 35)
    for key, val in sorted(result.overall.items()):
        lines.append(f"  {key:<25} {val:>10.4f}")

    if result.by_category:
        lines.append("\nBy Question Type:")
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        header = f"  {'Question Type':<25}" + "".join(f"  {k:<18}" for k in all_metric_keys)
        lines.append(header)
        lines.append("  " + "-" * (25 + 20 * len(all_metric_keys)))
        for cat, metrics in sorted(result.by_category.items()):
            row = f"  {cat:<25}" + "".join(
                f"  {metrics.get(k, 0.0):<18.4f}" for k in all_metric_keys
            )
            lines.append(row)

    lines.append("\n" + "=" * 60 + "\n")
    return "\n".join(lines)


def format_results_markdown(result: BenchmarkResult) -> str:
    """Markdown table output."""
    lines = []
    lines.append(f"## LongMemEval Benchmark — {result.mode.upper()} mode")
    lines.append(f"**Dataset:** {result.conversation_id}  ")
    lines.append("")
    lines.append("### Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, val in sorted(result.overall.items()):
        lines.append(f"| {key} | {val:.4f} |")
    lines.append("")

    if result.by_category:
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        lines.append("### By Question Type")
        lines.append("")
        header = "| Question Type | " + " | ".join(all_metric_keys) + " |"
        separator = "|---------------|" + "|".join(["--------"] * len(all_metric_keys)) + "|"
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


def save_results(result: BenchmarkResult, output_dir: str) -> str:
    """Save results as JSON. Returns filepath."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"longmemeval_{result.mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    data = {
        "timestamp": timestamp,
        "conversation_id": result.conversation_id,
        "mode": result.mode,
        "overall": result.overall,
        "by_category": result.by_category,
        "by_conversation": result.by_conversation,
        "config": result.config,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Results saved to %s", filepath)
    return filepath


async def run_benchmark(args: argparse.Namespace):
    """Full pipeline: load dataset, per-item ingest+evaluate, aggregate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    top_k = args.top_k if args.top_k else [5, 10]
    data_path = getattr(args, "data_path", None)
    limit = getattr(args, "limit", None)
    ingestion_mode = getattr(args, "ingestion_mode", "turn")

    # "both" mode: run all three ingestion modes and compare
    if ingestion_mode == "both":
        results = []
        for mode in ("turn", "session", "hybrid"):
            args_copy = argparse.Namespace(**vars(args))
            args_copy.ingestion_mode = mode
            logger.info("--- Running ingestion_mode=%s ---", mode)
            results.append(await run_benchmark(args_copy))
        return results

    logger.info("Loading LongMemEval dataset (ingestion_mode=%s)...", ingestion_mode)
    items = load_dataset(data_path=data_path, limit=limit)
    logger.info("Loaded %d items", len(items))

    _ingest_fn = {
        "turn": ingest_item,
        "session": ingest_item_session,
        "hybrid": ingest_item_hybrid,
    }[ingestion_mode]

    all_per_item: List[Dict] = []

    for i, item in enumerate(items):
        if i > 0 and i % 50 == 0:
            logger.info("Progress: %d/%d items processed", i, len(items))

        storage, tmp_dir = create_isolated_storage()
        try:
            await storage.initialize()
            count = await _ingest_fn(storage, item)
            logger.debug("Ingested %d units for item %s", count, item.question_id)

            if args.mode == "retrieval":
                metrics = await evaluate_retrieval(storage, item, top_k=top_k)
                # rename question_type -> category for aggregate_results compatibility
                metrics["category"] = metrics.pop("question_type")
                all_per_item.append(metrics)
            elif args.mode == "ablation":
                per_config = await run_ablation(storage, item, top_k=top_k)
                for m in per_config:
                    m["category"] = m.pop("question_type")
                all_per_item.extend(per_config)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    config = {
        "mode": args.mode,
        "ingestion_mode": ingestion_mode,
        "top_k": top_k,
        "num_items": len(items),
    }

    if args.mode == "ablation":
        by_config: Dict[str, List[Dict]] = defaultdict(list)
        for q in all_per_item:
            by_config[q["config_name"]].append(q)
        results = []
        for config_name, questions in by_config.items():
            r = aggregate_results(
                questions,
                conversation_id="all",
                mode=f"ablation:{config_name}",
                config=config,
            )
            results.append(r)
        return results

    result = aggregate_results(
        all_per_item,
        conversation_id="all",
        mode=args.mode,
        config=config,
    )
    return result


def parse_args(argv=None) -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(
        description="Run the LongMemEval benchmark against MCP Memory Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to longmemeval_s.json. Auto-downloads if not specified.",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "ablation"],
        default="retrieval",
        help="Benchmark mode (default: retrieval)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Top-K values for retrieval metrics (default: 5 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to evaluate (default: all)",
    )
    parser.add_argument(
        "--ingestion-mode",
        choices=["turn", "session", "hybrid", "both"],
        default="turn",
        help="Ingestion granularity: 'turn' stores each message separately (default), "
             "'session' stores a full conversation as one memory, "
             "'hybrid' stores both session units and individual turns, "
             "'both' runs all three modes and prints a comparison table.",
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

    # Ablation returns a list of results (one per config)
    results = result if isinstance(result, list) else [result]

    for r in results:
        if args.markdown:
            print(format_results_markdown(r))
        else:
            print(format_results_table(r))

    if args.output_dir:
        for r in results:
            filepath = save_results(r, args.output_dir)
            print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
