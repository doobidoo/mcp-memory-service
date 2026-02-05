#!/usr/bin/env python3
"""
Alpha Grid Search for Hybrid Search Tuning

Sweeps alpha values [0.3, 0.5, 0.6, 0.7, 0.8, 1.0] and reports:
- Hit Rate@10
- MRR (Mean Reciprocal Rank)
- NDCG@10

Usage:
    python tests/eval/sweep_alpha.py

Output:
    Markdown table with metrics for each alpha value.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_memory_service.config import settings
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def load_ground_truth():
    """Load ground truth test cases."""
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def calculate_metrics(results: list[dict], expected_hashes: list[str]):
    """Calculate Hit Rate@10, MRR, and NDCG@10."""
    import math

    k = 10
    result_hashes = [r["content_hash"] for r in results[:k]]

    # Hit Rate@10
    hit_rate = 1.0 if any(h in result_hashes for h in expected_hashes) else 0.0

    # MRR
    mrr = 0.0
    for i, h in enumerate(result_hashes):
        if h in expected_hashes:
            mrr = 1.0 / (i + 1)
            break

    # NDCG@10
    dcg = 0.0
    for i, h in enumerate(result_hashes):
        rel = 1.0 if h in expected_hashes else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_hashes), k)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return hit_rate, mrr, ndcg


async def run_sweep():
    """Run alpha sweep and print results."""
    # Setup
    gt = load_ground_truth()
    test_cases = gt.get("test_cases", [])
    memories_data = gt.get("memories", [])

    # Alpha values to test
    alpha_values = [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]

    print("Hybrid Search Alpha Sweep")
    print("=" * 60)
    print()

    # Create temp storage
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "sweep_test.db")

    try:
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        # Seed memories
        for memory_data in memories_data:
            memory = Memory(
                content=memory_data["content"],
                content_hash=memory_data["content_hash"],
                tags=memory_data.get("tags", []),
                memory_type=memory_data.get("memory_type", "note"),
            )
            await storage.store(memory)

        print(f"Seeded {len(memories_data)} memories, {len(test_cases)} test cases")
        print()

        results = []

        for alpha in alpha_values:
            # Create service with this alpha
            service = MemoryService(storage)

            # Override the hybrid_alpha config for this run
            original_alpha = settings.hybrid_search.hybrid_alpha
            settings.hybrid_search.hybrid_alpha = alpha

            # Clear tag cache
            service._tag_cache = None

            hit_rates = []
            mrr_scores = []
            ndcg_scores = []

            for tc in test_cases:
                result = await service.retrieve_memories(query=tc["query"], page=1, page_size=10)

                hr, mrr, ndcg = calculate_metrics(result["memories"], tc["expected_hashes"])
                hit_rates.append(hr)
                mrr_scores.append(mrr)
                ndcg_scores.append(ndcg)

            # Restore original alpha
            settings.hybrid_search.hybrid_alpha = original_alpha

            avg_hr = sum(hit_rates) / len(hit_rates)
            avg_mrr = sum(mrr_scores) / len(mrr_scores)
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

            results.append({"alpha": alpha, "hit_rate": avg_hr, "mrr": avg_mrr, "ndcg": avg_ndcg})

            mode = "Pure Vector" if alpha == 1.0 else "Hybrid" if alpha < 1.0 else "Pure Tags"
            print(f"Alpha {alpha:.1f} ({mode}): Hit@10={avg_hr:.2%}, MRR={avg_mrr:.3f}, NDCG={avg_ndcg:.3f}")

    finally:
        storage.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Print markdown table
    print()
    print("## Results (Markdown)")
    print()
    print("| Alpha | Hit@10 | MRR | NDCG@10 | Mode |")
    print("|-------|--------|-----|---------|------|")

    best_hr = max(r["hit_rate"] for r in results)
    for r in results:
        mode = "Pure Vector" if r["alpha"] == 1.0 else "Hybrid"
        marker = " **" if r["hit_rate"] == best_hr else ""
        print(f"| {r['alpha']:.1f} | {r['hit_rate']:.2%}{marker} | {r['mrr']:.3f} | {r['ndcg']:.3f} | {mode} |")

    # Find optimal
    optimal = max(results, key=lambda r: r["hit_rate"])
    print()
    print(f"**Optimal Alpha: {optimal['alpha']:.1f}** (Hit@10: {optimal['hit_rate']:.2%})")

    return results


if __name__ == "__main__":
    asyncio.run(run_sweep())
