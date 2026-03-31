#!/usr/bin/env python3
"""LoCoMo Benchmark for MCP Memory Service."""
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
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

from locomo_dataset import LocomoConversation, load_dataset
from locomo_evaluator import (
    BenchmarkResult,
    aggregate_results,
    mrr,
    precision_at_k,
    recall_at_k,
    token_f1,
)

logger = logging.getLogger(__name__)


def create_isolated_storage() -> Tuple[SqliteVecMemoryStorage, str]:
    """Create isolated SQLite-Vec storage in temp dir. Returns (storage, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="locomo-bench-")
    db_path = os.path.join(tmp_dir, "memories.db")
    storage = SqliteVecMemoryStorage(db_path)
    return storage, tmp_dir


def build_evidence_map(conv: LocomoConversation) -> Dict[str, str]:
    """Map dia_id -> turn text."""
    return {turn.dia_id: turn.text for turn in conv.turns}


def _match_evidence(
    retrieved_results,
    evidence_ids: List[str],
) -> Tuple[List[str], set]:
    """Match retrieved memories against evidence via dia_ref tags.

    Each observation is tagged with 'dia:D1:3' etc. during ingestion.
    We check if a retrieved memory's tags contain any evidence dia_id.

    Returns (retrieved_labels, relevant_labels) where each evidence item
    gets a unique label so multi-evidence recall is computed correctly.
    """
    relevant_labels = {f"ev_{dia_id}" for dia_id in evidence_ids}

    retrieved_labels = []
    for result in retrieved_results:
        tags = result.memory.tags or []
        matched_label = None
        for dia_id in evidence_ids:
            if f"dia:{dia_id}" in tags:
                matched_label = f"ev_{dia_id}"
                break
        if matched_label:
            retrieved_labels.append(matched_label)
        else:
            retrieved_labels.append(f"irrelevant_{len(retrieved_labels)}")

    return retrieved_labels, relevant_labels


def _parse_session_date(date_str: str) -> float:
    """Parse LoCoMo date string to Unix timestamp."""
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).timestamp()
        except ValueError:
            continue
    logger.warning("Could not parse session date: %s", date_str)
    return 0.0


async def ingest_conversation(storage: SqliteVecMemoryStorage, conv: LocomoConversation) -> int:
    """Store observations as memories. Returns count of stored memories."""
    stored = 0
    for obs in conv.observations:
        session_date = ""
        for turn in conv.turns:
            if turn.session_id == obs.session_id:
                session_date = turn.session_date
                break
        created_at = _parse_session_date(session_date) if session_date else None
        content_hash = hashlib.sha256(obs.text.encode("utf-8")).hexdigest()
        tags = ["locomo", conv.sample_id, obs.speaker, obs.session_id]
        if obs.dia_ref:
            tags.append(f"dia:{obs.dia_ref}")
        memory = Memory(
            content=obs.text,
            content_hash=content_hash,
            tags=tags,
            memory_type="observation",
            created_at=created_at if created_at else None,
        )
        success, msg = await storage.store(memory, skip_semantic_dedup=True)
        if success:
            stored += 1
    return stored


async def evaluate_retrieval(
    storage: SqliteVecMemoryStorage,
    conv: LocomoConversation,
    evidence_map: Dict[str, str],
    top_k: Optional[List[int]] = None,
) -> List[Dict]:
    """For each QA pair: search memories, check if evidence found, compute metrics."""
    if top_k is None:
        top_k = [5]
    max_k = max(top_k)
    per_question = []
    for qa in conv.qa_pairs:
        results = await storage.retrieve(qa.question, n_results=max_k)
        retrieved_labels, relevant_labels = _match_evidence(results, qa.evidence)
        metrics: Dict = {"category": qa.category}
        for k in top_k:
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
            metrics[f"precision_at_{k}"] = precision_at_k(retrieved_labels, relevant_labels, k)
        metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
        per_question.append(metrics)
    return per_question


async def run_ablation(
    storage: SqliteVecMemoryStorage,
    conv: LocomoConversation,
    evidence_map: Dict[str, str],
    top_k: Optional[List[int]] = None,
) -> List[Dict]:
    """Compare retrieval configurations: baseline, +quality, +quality+decay.

    Configurations:
    1. baseline — retrieve() with cosine similarity only
    2. +quality_boost — retrieve_with_quality_boost()
    3. +quality+min_confidence — quality boost with min_confidence filtering
    """
    if top_k is None:
        top_k = [5]
    max_k = max(top_k)

    configs = [
        ("baseline", "retrieve", {}),
        ("+quality_boost", "retrieve_with_quality_boost", {}),
        ("+quality_w0.5", "retrieve_with_quality_boost", {"quality_weight": 0.5}),
    ]

    all_results: List[Dict] = []

    for config_name, method_name, extra_kwargs in configs:
        retrieve_fn = getattr(storage, method_name)
        for qa in conv.qa_pairs:
            results = await retrieve_fn(qa.question, n_results=max_k, **extra_kwargs)
            retrieved_labels, relevant_labels = _match_evidence(results, qa.evidence)
            metrics: Dict = {"config_name": config_name, "category": qa.category}
            for k in top_k:
                metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
                metrics[f"precision_at_{k}"] = precision_at_k(retrieved_labels, relevant_labels, k)
            metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
            all_results.append(metrics)

    return all_results


def format_results_table(result: BenchmarkResult) -> str:
    """Terminal table output."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"  LoCoMo Benchmark — {result.mode.upper()} mode")
    lines.append(f"  Conversation: {result.conversation_id}")
    lines.append("=" * 60)

    lines.append("\nOverall Metrics:")
    lines.append(f"  {'Metric':<25} {'Value':>10}")
    lines.append("  " + "-" * 35)
    for key, val in sorted(result.overall.items()):
        lines.append(f"  {key:<25} {val:>10.4f}")

    if result.by_category:
        lines.append("\nBy Category:")
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        header = f"  {'Category':<20}" + "".join(f"  {k:<18}" for k in all_metric_keys)
        lines.append(header)
        lines.append("  " + "-" * (20 + 20 * len(all_metric_keys)))
        for cat, metrics in sorted(result.by_category.items()):
            row = f"  {cat:<20}" + "".join(
                f"  {metrics.get(k, 0.0):<18.4f}" for k in all_metric_keys
            )
            lines.append(row)

    lines.append("\n" + "=" * 60 + "\n")
    return "\n".join(lines)


def format_results_markdown(result: BenchmarkResult) -> str:
    """Markdown table output."""
    lines = []
    lines.append(f"## LoCoMo Benchmark — {result.mode.upper()} mode")
    lines.append(f"**Conversation:** {result.conversation_id}  ")
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


def save_results(result: BenchmarkResult, output_dir: str) -> str:
    """Save results as JSON. Returns filepath."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"locomo_{result.mode}_{result.conversation_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    data = {
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


async def evaluate_qa(
    storage: SqliteVecMemoryStorage,
    conv: LocomoConversation,
    evidence_map: Dict[str, str],
    adapter,
    top_k: Optional[List[int]] = None,
) -> List[Dict]:
    """QA evaluation: retrieve + LLM answer + F1."""
    if top_k is None:
        top_k = [5]
    max_k = max(top_k)
    per_question = []
    for qa in conv.qa_pairs:
        results = await storage.retrieve(qa.question, n_results=max_k)
        context_texts = [r.memory.content for r in results]
        predicted = await adapter.generate_answer(qa.question, context_texts)
        f1 = token_f1(predicted, qa.answer)

        retrieved_labels, relevant_labels = _match_evidence(results, qa.evidence)

        metrics: Dict = {"category": qa.category, "token_f1": f1}
        for k in top_k:
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
            metrics[f"precision_at_{k}"] = precision_at_k(retrieved_labels, relevant_labels, k)
        metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
        per_question.append(metrics)
    return per_question


async def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    """Full pipeline: load dataset, per-conversation ingest+evaluate, aggregate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    top_k = args.top_k if args.top_k else [5]
    data_path = getattr(args, "data_path", None)

    logger.info("Loading LoCoMo dataset...")
    conversations = load_dataset(data_path=data_path)
    logger.info("Loaded %d conversations", len(conversations))

    all_per_question: List[Dict] = []

    for conv in conversations:
        logger.info("Processing conversation: %s", conv.sample_id)
        storage, tmp_dir = create_isolated_storage()
        try:
            await storage.initialize()
            count = await ingest_conversation(storage, conv)
            logger.info("Ingested %d observations for %s", count, conv.sample_id)
            evidence_map = build_evidence_map(conv)

            if args.mode == "retrieval":
                per_question = await evaluate_retrieval(storage, conv, evidence_map, top_k=top_k)
            elif args.mode == "qa":
                from locomo_llm import create_adapter
                llm_model = getattr(args, "llm_model", "") or ""
                llm_name = getattr(args, "llm", "mock") or "mock"
                adapter = create_adapter(llm_name, model=llm_model)
                per_question = await evaluate_qa(storage, conv, evidence_map, adapter, top_k=top_k)
            elif args.mode == "ablation":
                per_question = await run_ablation(storage, conv, evidence_map, top_k=top_k)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            all_per_question.extend(per_question)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    config = {
        "mode": args.mode,
        "top_k": top_k,
        "num_conversations": len(conversations),
    }

    if args.mode == "ablation":
        # Group by config_name and return list of results
        from collections import defaultdict
        by_config = defaultdict(list)
        for q in all_per_question:
            by_config[q["config_name"]].append(q)
        results = []
        for config_name, questions in by_config.items():
            r = aggregate_results(
                questions, conversation_id="all",
                mode=f"ablation:{config_name}", config=config,
            )
            results.append(r)
        return results

    result = aggregate_results(
        all_per_question,
        conversation_id="all",
        mode=args.mode,
        config=config,
    )
    return result


def parse_args(argv=None) -> argparse.Namespace:
    """CLI args: --data-path, --mode, --top-k, --llm, --llm-model, --markdown, --output-dir"""
    parser = argparse.ArgumentParser(
        description="Run the LoCoMo benchmark against MCP Memory Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to locomo10.json. Auto-downloads if not specified.",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "qa", "ablation"],
        default="retrieval",
        help="Benchmark mode (default: retrieval)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5],
        help="Top-K values for retrieval metrics (default: 5)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="mock",
        choices=["mock", "claude", "ollama"],
        help="LLM adapter for QA mode (default: mock)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="LLM model name override",
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
