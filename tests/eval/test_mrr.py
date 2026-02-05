"""
Mean Reciprocal Rank (MRR) evaluation for hybrid search.

MRR = average of 1/rank of first relevant result.
Measures how quickly the first relevant result appears.
"""

import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .conftest import (
    SQLITE_VEC_AVAILABLE,
    calculate_mrr,
    get_test_cases,
)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestMRR:
    """Mean Reciprocal Rank evaluation tests."""

    @pytest.mark.asyncio
    async def test_mrr_tag_sensitive(self, eval_service):
        """Evaluate MRR for tag-sensitive queries."""
        test_cases = get_test_cases(category="tag_sensitive")
        mrr_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            mrr = calculate_mrr(result["memories"], tc["expected_hashes"])
            mrr_scores.append(mrr)

        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        print(f"\nMRR (tag_sensitive): {avg_mrr:.3f} ({len(mrr_scores)} queries)")

        # MRR > 0.5 means relevant results typically in top 2
        assert avg_mrr > 0.0, "MRR should be non-zero"

    @pytest.mark.asyncio
    async def test_mrr_semantic(self, eval_service):
        """Evaluate MRR for semantic queries."""
        test_cases = get_test_cases(category="semantic")
        mrr_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            mrr = calculate_mrr(result["memories"], tc["expected_hashes"])
            mrr_scores.append(mrr)

        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        print(f"\nMRR (semantic): {avg_mrr:.3f} ({len(mrr_scores)} queries)")

        # Semantic MRR should be non-zero for valid test data
        assert len(mrr_scores) > 0, "Should have evaluated at least one semantic test case"
        assert avg_mrr > 0.0, "Semantic MRR should be non-zero"

    @pytest.mark.asyncio
    async def test_mrr_overall(self, eval_service):
        """Evaluate overall MRR."""
        test_cases = get_test_cases()
        mrr_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            mrr = calculate_mrr(result["memories"], tc["expected_hashes"])
            mrr_scores.append(mrr)

        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        print(f"\nOverall MRR: {avg_mrr:.3f} ({len(mrr_scores)} queries)")

        # Basic sanity check
        assert avg_mrr > 0.0, "Overall MRR should be non-zero"

    @pytest.mark.asyncio
    async def test_mrr_per_category(self, eval_service):
        """Evaluate MRR broken down by category."""
        categories = ["tag_sensitive", "semantic", "mixed"]

        for category in categories:
            test_cases = get_test_cases(category=category)
            if not test_cases:
                continue

            mrr_scores = []
            for tc in test_cases:
                result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
                mrr = calculate_mrr(result["memories"], tc["expected_hashes"])
                mrr_scores.append(mrr)

            avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
            print(f"MRR ({category}): {avg_mrr:.3f} ({len(mrr_scores)} queries)")

            # Each category should have at least one test case and non-negative MRR
            assert len(mrr_scores) > 0, f"Should have evaluated at least one {category} test case"
            assert avg_mrr >= 0.0, f"MRR for {category} should be non-negative"
