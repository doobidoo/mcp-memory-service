"""Tests for the LongMemEval benchmark orchestrator."""
import asyncio
import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from longmemeval_dataset import LongMemEvalItem, LongMemEvalSession, LongMemEvalTurn
from benchmark_longmemeval import (
    _match_evidence,
    create_isolated_storage,
    ingest_item,
    evaluate_retrieval,
    run_ablation,
)
from locomo_evaluator import recall_at_k


def _make_test_item() -> LongMemEvalItem:
    return LongMemEvalItem(
        question_id="q001",
        question="What city did the user say they grew up in?",
        answer="Portland",
        question_type="single-session-user",
        sessions=[
            LongMemEvalSession(
                session_id="session_001",
                turns=[
                    LongMemEvalTurn("user", "I grew up in Portland."),
                    LongMemEvalTurn("assistant", "Portland is a great city!"),
                ],
            ),
            LongMemEvalSession(
                session_id="session_002",
                turns=[
                    LongMemEvalTurn("user", "I enjoy hiking on weekends."),
                    LongMemEvalTurn("assistant", "That sounds fun!"),
                ],
            ),
        ],
        answer_session_ids=["session_001"],
    )


class TestCreateIsolatedStorage:
    def test_creates_storage_and_tmpdir(self):
        storage, tmp_dir = create_isolated_storage()
        assert storage is not None
        assert os.path.isdir(tmp_dir)
        assert "longmemeval-bench-" in tmp_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestIngestItem:
    def test_ingest_stores_all_turns(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            count = asyncio.run(self._run_ingest(storage, item))
            # 2 sessions × 2 turns each = 4 turns total
            assert count == 4
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_ingest(self, storage, item):
        await storage.initialize()
        return await ingest_item(storage, item)


class TestEvaluateRetrieval:
    def test_retrieval_returns_expected_keys(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            result = asyncio.run(self._run_retrieval(storage, item))
            assert "question_type" in result
            assert result["question_type"] == "single-session-user"
            assert "recall_at_5" in result
            assert "recall_at_10" in result
            assert "ndcg_at_10" in result
            assert "mrr" in result
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_retrieval(self, storage, item):
        await storage.initialize()
        await ingest_item(storage, item)
        return await evaluate_retrieval(storage, item, top_k=[5, 10])


class TestRunAblation:
    def test_ablation_returns_list_with_config_names(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            results = asyncio.run(self._run_ablation(storage, item))
            assert isinstance(results, list)
            assert len(results) == 3
            assert all("config_name" in r for r in results)
            config_names = {r["config_name"] for r in results}
            assert "baseline" in config_names
            assert "+quality_boost" in config_names
            assert "+quality_w0.5" in config_names
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_ablation(self, storage, item):
        await storage.initialize()
        await ingest_item(storage, item)
        return await run_ablation(storage, item, top_k=[5, 10])


class TestMatchEvidence:
    """Tests for _match_evidence de-duplication behavior."""

    class _FakeMemory:
        def __init__(self, tags):
            self.tags = tags

    class _FakeResult:
        def __init__(self, tags):
            self.memory = None

        @classmethod
        def make(cls, tags):
            obj = cls.__new__(cls)
            obj.memory = TestMatchEvidence._FakeMemory(tags)
            return obj

    def test_multiple_hits_same_session_count_once(self):
        """Prevents recall > 1.0 when multiple turns from same session are retrieved."""
        results = [
            self._FakeResult.make(["longmemeval", "q1", "session_001", "user"]),
            self._FakeResult.make(["longmemeval", "q1", "session_001", "assistant"]),
            self._FakeResult.make(["longmemeval", "q1", "session_002", "user"]),
        ]
        retrieved, relevant = _match_evidence(results, ["session_001"])

        # Only 1 hit (first occurrence of session_001), despite 2 results from it
        hits = sum(1 for label in retrieved if label in relevant)
        assert hits == 1

    def test_recall_stays_between_0_and_1(self):
        """Recall must not exceed 1.0 regardless of how many turns are retrieved."""
        # 5 results all from the single evidence session
        results = [self._FakeResult.make(["session_001"]) for _ in range(5)]
        retrieved, relevant = _match_evidence(results, ["session_001"])
        r = recall_at_k(retrieved, relevant, k=5)
        assert 0.0 <= r <= 1.0

    def test_multiple_sessions_each_credited_once(self):
        """When multiple evidence sessions are retrieved, each is credited at most once."""
        results = [
            self._FakeResult.make(["session_001"]),
            self._FakeResult.make(["session_001"]),  # duplicate — should be ignored
            self._FakeResult.make(["session_002"]),
            self._FakeResult.make(["session_002"]),  # duplicate — should be ignored
        ]
        retrieved, relevant = _match_evidence(results, ["session_001", "session_002"])
        hits = sum(1 for label in retrieved if label in relevant)
        assert hits == 2  # one hit per evidence session, not four
