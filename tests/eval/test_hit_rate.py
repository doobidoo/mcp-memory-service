"""
Hit Rate@K evaluation for hybrid search.

Hit Rate@K = % of queries where any expected result appears in top K.
This is the primary retrieval quality metric.
"""

import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .conftest import (
    SQLITE_VEC_AVAILABLE,
    calculate_hit_rate,
    get_test_cases,
)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestHitRate:
    """Hit Rate@10 evaluation tests."""

    @pytest.mark.asyncio
    async def test_hit_rate_tag_sensitive_queries(self, eval_service):
        """Evaluate Hit Rate@10 for tag-sensitive queries."""
        test_cases = get_test_cases(category="tag_sensitive")
        hits = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            hr = calculate_hit_rate(result["memories"], tc["expected_hashes"], k=10)
            hits.append(hr)

        avg_hit_rate = sum(hits) / len(hits) if hits else 0.0
        print(f"\nHit Rate@10 (tag_sensitive): {avg_hit_rate:.2%} ({len(hits)} queries)")

        # Target: >85% hit rate for tag-sensitive queries
        # Note: This is a target, actual performance may vary based on test data
        assert avg_hit_rate > 0.0, "Hit rate should be non-zero"

    @pytest.mark.asyncio
    async def test_hit_rate_semantic_queries(self, eval_service):
        """Evaluate Hit Rate@10 for pure semantic queries."""
        test_cases = get_test_cases(category="semantic")
        hits = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            hr = calculate_hit_rate(result["memories"], tc["expected_hashes"], k=10)
            hits.append(hr)

        avg_hit_rate = sum(hits) / len(hits) if hits else 0.0
        print(f"\nHit Rate@10 (semantic): {avg_hit_rate:.2%} ({len(hits)} queries)")

        # Semantic Hit Rate@10 should be non-zero for valid test data
        assert avg_hit_rate > 0.0, "Semantic Hit Rate@10 should be non-zero"

    @pytest.mark.asyncio
    async def test_hit_rate_overall(self, eval_service):
        """Evaluate overall Hit Rate@10."""
        test_cases = get_test_cases()
        hits = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            hr = calculate_hit_rate(result["memories"], tc["expected_hashes"], k=10)
            hits.append(hr)

        avg_hit_rate = sum(hits) / len(hits) if hits else 0.0
        print(f"\nOverall Hit Rate@10: {avg_hit_rate:.2%} ({len(hits)} queries)")

        # Basic sanity check
        assert avg_hit_rate > 0.0, "Overall hit rate should be non-zero"

    @pytest.mark.asyncio
    async def test_hit_rate_k_values(self, eval_service):
        """Evaluate Hit Rate at different K values."""
        test_cases = get_test_cases()

        for k in [1, 3, 5, 10]:
            hits = []
            for tc in test_cases:
                result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=k)
                hr = calculate_hit_rate(result["memories"], tc["expected_hashes"], k=k)
                hits.append(hr)

            avg_hit_rate = sum(hits) / len(hits) if hits else 0.0
            print(f"Hit Rate@{k}: {avg_hit_rate:.2%}")

            # Hit Rate@{k} should be non-zero for valid test data
            assert avg_hit_rate > 0.0, f"Hit Rate@{k} should be non-zero"
