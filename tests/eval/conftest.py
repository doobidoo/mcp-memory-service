"""
Evaluation harness fixtures and utilities.

Provides ground truth loading and storage setup for retrieval quality evaluation.
"""

import json
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService

# =============================================================================
# Ground Truth Utilities
# =============================================================================


def load_ground_truth() -> dict:
    """Load ground truth test cases from JSON."""
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def get_test_cases(category: str = None) -> list[dict]:
    """Get test cases, optionally filtered by category."""
    gt = load_ground_truth()
    cases = gt.get("test_cases", [])
    if category:
        cases = [c for c in cases if c.get("category") == category]
    return cases


# =============================================================================
# Storage Fixtures
# =============================================================================

try:
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def eval_storage() -> AsyncGenerator["SqliteVecMemoryStorage", None]:
    """Create storage seeded with evaluation test data."""
    if not SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec not available")

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "eval_test.db")

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()

    # Seed with ground truth memories
    gt = load_ground_truth()
    for memory_data in gt.get("memories", []):
        memory = Memory(
            content=memory_data["content"],
            content_hash=memory_data["content_hash"],
            tags=memory_data.get("tags", []),
            memory_type=memory_data.get("memory_type", "note"),
        )
        await storage.store(memory)

    yield storage

    await storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def eval_service(eval_storage) -> MemoryService:
    """Create MemoryService for evaluation."""
    return MemoryService(eval_storage)


# =============================================================================
# Metrics Utilities
# =============================================================================


def calculate_hit_rate(results: list[dict], expected_hashes: list[str], k: int = 10) -> float:
    """
    Calculate Hit Rate@K.

    Hit Rate = 1 if any expected hash appears in top K results, else 0.
    """
    if not expected_hashes:
        return 0.0

    result_hashes = {r["content_hash"] for r in results[:k]}
    return 1.0 if any(h in result_hashes for h in expected_hashes) else 0.0


def calculate_mrr(results: list[dict], expected_hashes: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1 / position of first relevant result.
    """
    if not expected_hashes:
        return 0.0

    for i, result in enumerate(results):
        if result["content_hash"] in expected_hashes:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(results: list[dict], expected_hashes: list[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    import math

    if not expected_hashes:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        rel = 1.0 if result["content_hash"] in expected_hashes else 0.0
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed

    # Calculate ideal DCG (all relevant items at top)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_hashes), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
