"""Tests for DevBench practical benchmark."""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_devbench import (
    create_isolated_storage,
    ingest_memories,
    load_devbench_dataset,
    match_by_tags,
)


class TestLoadDataset:
    def test_loads_30_memories(self):
        dataset = load_devbench_dataset()
        assert len(dataset["memories"]) == 30

    def test_loads_20_queries(self):
        dataset = load_devbench_dataset()
        assert len(dataset["queries"]) == 20

    def test_all_bench_tags_present(self):
        dataset = load_devbench_dataset()
        ids = {m["id"] for m in dataset["memories"]}
        expected_prefixes = [
            "bench:decision-", "bench:learning-", "bench:bugfix-",
            "bench:arch-", "bench:config-",
        ]
        for prefix in expected_prefixes:
            matches = [i for i in ids if i.startswith(prefix)]
            assert len(matches) == 6, f"Expected 6 memories for {prefix}, got {len(matches)}"

    def test_four_query_categories(self):
        dataset = load_devbench_dataset()
        categories = {q["category"] for q in dataset["queries"]}
        assert categories == {"exact", "semantic", "cross_type", "negative"}

    def test_five_queries_per_category(self):
        dataset = load_devbench_dataset()
        from collections import Counter
        counts = Counter(q["category"] for q in dataset["queries"])
        for cat in ("exact", "semantic", "cross_type", "negative"):
            assert counts[cat] == 5, f"Expected 5 {cat} queries, got {counts[cat]}"

    def test_negative_queries_have_empty_expected_tags(self):
        dataset = load_devbench_dataset()
        for q in dataset["queries"]:
            if q["category"] == "negative":
                assert q["expected_tags"] == []


class TestMatchByTags:
    def _make_result(self, tags):
        """Create a minimal mock result with memory.tags."""
        class FakeMemory:
            def __init__(self, t):
                self.tags = t
        class FakeResult:
            def __init__(self, t):
                self.memory = FakeMemory(t)
        return FakeResult(tags)

    def test_match_found(self):
        results = [
            self._make_result(["devbench", "bench:decision-1", "storage"]),
            self._make_result(["devbench", "bench:learning-1"]),
        ]
        retrieved_labels, relevant_labels = match_by_tags(results, ["bench:decision-1"])
        assert "bench:decision-1" in retrieved_labels
        assert relevant_labels == {"bench:decision-1"}

    def test_no_match(self):
        results = [
            self._make_result(["devbench", "bench:arch-1"]),
        ]
        retrieved_labels, relevant_labels = match_by_tags(results, ["bench:decision-1"])
        assert retrieved_labels[0].startswith("irrelevant_")
        assert relevant_labels == {"bench:decision-1"}

    def test_negative_query_empty_expected(self):
        results = [
            self._make_result(["devbench", "bench:decision-1"]),
        ]
        retrieved_labels, relevant_labels = match_by_tags(results, [])
        # relevant_labels is empty set for negative queries
        assert relevant_labels == set()
        # retrieved label is irrelevant since no expected tags to match
        assert retrieved_labels[0].startswith("irrelevant_")

    def test_multiple_expected_tags(self):
        results = [
            self._make_result(["bench:decision-2", "storage"]),
            self._make_result(["bench:config-2", "environment"]),
            self._make_result(["unrelated"]),
        ]
        retrieved_labels, relevant_labels = match_by_tags(
            results, ["bench:decision-2", "bench:config-2"]
        )
        assert "bench:decision-2" in retrieved_labels
        assert "bench:config-2" in retrieved_labels
        assert retrieved_labels[2].startswith("irrelevant_")
        assert relevant_labels == {"bench:decision-2", "bench:config-2"}


class TestIngestAndRetrieve:
    def test_ingest_all_30_memories(self):
        async def _run():
            dataset = load_devbench_dataset()
            storage, tmp_dir = create_isolated_storage()
            try:
                await storage.initialize()
                count = await ingest_memories(storage, dataset["memories"])
                return count
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        count = asyncio.run(_run())
        assert count == 30

    def test_retrieve_exact_query_finds_something(self):
        async def _run():
            dataset = load_devbench_dataset()
            storage, tmp_dir = create_isolated_storage()
            try:
                await storage.initialize()
                await ingest_memories(storage, dataset["memories"])
                results = await storage.retrieve(
                    "Why did we choose SQLite-Vec for storage?", n_results=5
                )
                return results
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        results = asyncio.run(_run())
        assert len(results) > 0
        # Top result should include bench:decision-1
        all_tags = [t for r in results for t in (r.memory.tags or [])]
        assert "bench:decision-1" in all_tags

    def test_retrieve_negative_query_returns_results(self):
        """Negative query should return results but none should be about billing/payments."""
        async def _run():
            dataset = load_devbench_dataset()
            storage, tmp_dir = create_isolated_storage()
            try:
                await storage.initialize()
                await ingest_memories(storage, dataset["memories"])
                results = await storage.retrieve(
                    "How does the billing system handle payments?", n_results=5
                )
                return results
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        results = asyncio.run(_run())
        # Results exist (storage always returns something), but content shouldn't mention billing
        for r in results:
            assert "billing" not in r.memory.content.lower()
            assert "payment" not in r.memory.content.lower()
