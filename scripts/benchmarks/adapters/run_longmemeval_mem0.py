#!/usr/bin/env python3
"""Run LongMemEval benchmark against mem0 cloud API.

Usage:
    export MEM0_API_KEY=your_key
    python run_longmemeval_mem0.py --limit 5
"""
import asyncio
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from longmemeval_dataset import LongMemEvalItem, load_dataset
from locomo_evaluator import recall_at_k, ndcg_at_k, mrr

from mem0 import MemoryClient

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
USER_ID = "longmemeval-bench"


async def ingest_item(client: MemoryClient, item: LongMemEvalItem) -> int:
    """Store conversation sessions as memories in mem0."""
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
        messages = [{"role": "user", "content": content}]
        client.add(messages, user_id=USER_ID, metadata={
            "session_id": session.session_id,
            "question_id": item.question_id,
        })
        stored += 1
    return stored


async def evaluate_item(client: MemoryClient, item: LongMemEvalItem, top_k: list[int]) -> dict:
    """Search mem0 and compute retrieval metrics."""
    max_k = max(top_k)

    response = client.search(item.question, filters={"user_id": USER_ID}, limit=max_k)
    results = response.get("results", []) if isinstance(response, dict) else response

    # Match: check if retrieved memories contain the answer session IDs
    retrieved_labels = []
    for r in results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id", "")
        retrieved_labels.append(1 if sid in item.answer_session_ids else 0)

    relevant_labels = [1] * len(item.answer_session_ids)

    metrics = {"question_type": item.question_type}
    for k in top_k:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
    metrics["ndcg_at_10"] = ndcg_at_k(retrieved_labels, relevant_labels, 10)
    metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark against mem0")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to evaluate")
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()

    if not MEM0_API_KEY:
        print("ERROR: MEM0_API_KEY not set")
        sys.exit(1)

    client = MemoryClient(api_key=MEM0_API_KEY)
    dataset = load_dataset(args.data_path)
    items = dataset[:args.limit]

    print(f"=== LongMemEval vs mem0 ({len(items)} items) ===")

    all_metrics = []
    for i, item in enumerate(items):
        print(f"\n[{i+1}/{len(items)}] {item.question_id}: {item.question[:50]}...")

        # Ingest
        stored = await ingest_item(client, item)
        print(f"  Ingested: {stored} sessions")

        # Wait for indexing
        time.sleep(3)

        # Evaluate
        metrics = await evaluate_item(client, item, args.top_k)
        all_metrics.append(metrics)
        print(f"  R@5={metrics.get('recall_at_5', 0):.2f} R@10={metrics.get('recall_at_10', 0):.2f} MRR={metrics.get('mrr', 0):.2f}")

        # Cleanup this item
        client.delete_all(user_id=USER_ID)
        time.sleep(1)

    # Aggregate
    print("\n=== RESULTS ===")
    for k in args.top_k:
        key = f"recall_at_{k}"
        avg = sum(m.get(key, 0) for m in all_metrics) / len(all_metrics)
        print(f"  Recall@{k}: {avg:.3f}")
    avg_mrr = sum(m.get("mrr", 0) for m in all_metrics) / len(all_metrics)
    print(f"  MRR: {avg_mrr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
