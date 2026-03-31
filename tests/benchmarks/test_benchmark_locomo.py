"""Tests for the main LoCoMo benchmark orchestrator."""
import asyncio
import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from locomo_dataset import LocomoConversation, LocomoObservation, LocomoTurn, LocomoQA
from benchmark_locomo import (
    ingest_conversation,
    build_evidence_map,
    create_isolated_storage,
    evaluate_retrieval,
    run_ablation,
)


def _make_test_conversation() -> LocomoConversation:
    return LocomoConversation(
        sample_id="test_1",
        turns=[
            LocomoTurn("Alice", "d1_1", "I work at Google.", "session_1", "January 15, 2024"),
            LocomoTurn("Bob", "d1_2", "Nice!", "session_1", "January 15, 2024"),
        ],
        observations=[
            LocomoObservation("session_1", "Alice works at Google.", "Alice"),
            LocomoObservation("session_1", "Bob congratulated Alice.", "Bob"),
        ],
        summaries={"session_1": "Alice told Bob about her job at Google."},
        qa_pairs=[LocomoQA("Where does Alice work?", "Google", "single-hop", ["d1_1"])],
    )


class TestBuildEvidenceMap:
    def test_maps_dia_ids_to_turns(self):
        conv = _make_test_conversation()
        emap = build_evidence_map(conv)
        assert "d1_1" in emap
        assert "I work at Google." in emap["d1_1"]

    def test_all_dia_ids_present(self):
        conv = _make_test_conversation()
        emap = build_evidence_map(conv)
        assert "d1_1" in emap
        assert "d1_2" in emap


class TestCreateIsolatedStorage:
    def test_creates_storage(self):
        storage, tmp_dir = create_isolated_storage()
        assert storage is not None
        assert os.path.isdir(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestIngestConversation:
    def test_ingest_stores_observations(self):
        conv = _make_test_conversation()
        storage, tmp_dir = create_isolated_storage()
        try:
            count = asyncio.run(self._run_ingest(storage, conv))
            assert count == 2
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_ingest(self, storage, conv):
        await storage.initialize()
        return await ingest_conversation(storage, conv)


class TestEvaluateRetrieval:
    def test_retrieval_returns_per_question_results(self):
        conv = _make_test_conversation()
        storage, tmp_dir = create_isolated_storage()
        try:
            results = asyncio.run(self._run_retrieval(storage, conv))
            assert len(results) == 1
            assert "category" in results[0]
            assert "recall_at_5" in results[0]
            assert "precision_at_5" in results[0]
            assert "mrr" in results[0]
            assert results[0]["category"] == "single-hop"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_retrieval(self, storage, conv):
        await storage.initialize()
        await ingest_conversation(storage, conv)
        evidence_map = build_evidence_map(conv)
        return await evaluate_retrieval(storage, conv, evidence_map, top_k=[5])


class TestAblation:
    def test_ablation_returns_multiple_configs(self):
        conv = _make_test_conversation()
        storage, tmp_dir = create_isolated_storage()
        try:
            results = asyncio.run(self._run_ablation(storage, conv))
            assert len(results) >= 2
            assert all("config_name" in r for r in results)
            config_names = {r["config_name"] for r in results}
            assert "baseline" in config_names
            assert "+quality_boost" in config_names
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_ablation(self, storage, conv):
        await storage.initialize()
        await ingest_conversation(storage, conv)
        evidence_map = build_evidence_map(conv)
        return await run_ablation(storage, conv, evidence_map, top_k=[5])
