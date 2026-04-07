"""LoCoMo benchmark evaluator: Retrieval and QA metrics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top-K retrieved."""
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/position of first relevant item."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @K.

    Relevance is binary (1 if item in relevant set, 0 otherwise).
    IDCG assumes all relevant items appear at top positions (ideal ranking).
    Returns 1.0 when relevant is empty (nothing to find = perfect by convention).
    """
    if not relevant:
        return 1.0
    if k == 0:
        return 1.0 if not relevant else 0.0

    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score. Case-insensitive, whitespace tokenized.

    Uses token count overlap (min of counts per shared token), not just set
    overlap, for correct F1 with repeated tokens.
    """
    pred_tokens = str(predicted).lower().split()
    gold_tokens = str(gold).lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    # Count overlapping tokens (by count, not just set membership)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts if t in gold_counts)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


@dataclass
class BenchmarkResult:
    conversation_id: str
    mode: str
    overall: Dict[str, float]
    by_category: Dict[str, Dict[str, float]]
    by_conversation: Dict[str, Dict[str, float]]
    config: Dict[str, Any] = field(default_factory=dict)


def aggregate_results(
    per_question: List[Dict[str, Any]],
    conversation_id: str,
    mode: str,
    config: Dict[str, Any],
) -> BenchmarkResult:
    """Aggregate per-question metrics into a BenchmarkResult.

    Each question dict must have a 'category' key plus numeric metric keys.
    Averages are computed per-category and overall.
    """
    if not per_question:
        return BenchmarkResult(
            conversation_id=conversation_id,
            mode=mode,
            overall={},
            by_category={},
            by_conversation={},
            config=config,
        )

    # Identify metric keys (all keys except 'category' and non-numeric values)
    sample = per_question[0]
    metric_keys = [
        k for k, v in sample.items()
        if k != "category" and isinstance(v, (int, float))
    ]

    # Aggregate by category
    category_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for q in per_question:
        cat = q.get("category", "unknown")
        category_buckets.setdefault(cat, []).append(q)

    by_category: Dict[str, Dict[str, float]] = {}
    for cat, questions in category_buckets.items():
        by_category[cat] = {
            key: sum(q[key] for q in questions if key in q) / len(questions)
            for key in metric_keys
        }

    # Overall averages
    overall: Dict[str, float] = {
        key: sum(q[key] for q in per_question if key in q) / len(per_question)
        for key in metric_keys
    }

    # by_conversation uses conversation_id as the single key
    by_conversation: Dict[str, Dict[str, float]] = {
        conversation_id: overall
    }

    return BenchmarkResult(
        conversation_id=conversation_id,
        mode=mode,
        overall=overall,
        by_category=by_category,
        by_conversation=by_conversation,
        config=config,
    )
