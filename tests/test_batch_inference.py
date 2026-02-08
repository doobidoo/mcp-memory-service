"""
Tests for batched ONNX inference in the consolidation pipeline.

Covers:
- ONNXRankerModel.score_quality_batch() — classifier and cross-encoder
- QualityEvaluator.evaluate_quality_batch() — single model and fallback
- AsyncQualityScorer batched worker dequeue
- SqliteVecMemoryStorage.store_batch() — batched embedding + transaction
- SemanticCompressionEngine parallel cluster compression
"""

import asyncio
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model
from mcp_memory_service.quality.ai_evaluator import QualityEvaluator
from mcp_memory_service.quality.async_scorer import AsyncQualityScorer
from mcp_memory_service.quality.config import QualityConfig

# Check if ONNX models are available for integration tests
DEBERTA_AVAILABLE = Path.home().joinpath(
    ".cache/mcp_memory/onnx_models/nvidia-quality-classifier-deberta/model.onnx"
).exists()
MS_MARCO_AVAILABLE = Path.home().joinpath(
    ".cache/mcp_memory/onnx_models/ms-marco-MiniLM-L-6-v2/model.onnx"
).exists()


def _make_memory(content: str, tags=None) -> Memory:
    """Helper to create a Memory with auto-generated hash."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or [],
        memory_type="observation",
        metadata={},
    )


# ---------------------------------------------------------------------------
# ONNXRankerModel.score_quality_batch()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa ONNX model not exported")
class TestScoreQualityBatchDeBERTa:
    """Test batched scoring with DeBERTa classifier model."""

    def setup_method(self):
        self.model = get_onnx_ranker_model(
            model_name="nvidia-quality-classifier-deberta", device="cpu"
        )

    def test_batch_single_item(self):
        """Batch of 1 matches sequential score_quality() output."""
        content = "Well-structured code with comprehensive error handling and tests."
        seq_score = self.model.score_quality("", content)
        batch_scores = self.model.score_quality_batch([("", content)])
        assert len(batch_scores) == 1
        assert abs(batch_scores[0] - seq_score) < 1e-5, (
            f"Batch score {batch_scores[0]:.6f} != sequential {seq_score:.6f}"
        )

    def test_batch_multiple(self):
        """Batch of 10 produces 10 valid scores."""
        pairs = [("", f"Memory content number {i} with details.") for i in range(10)]
        scores = self.model.score_quality_batch(pairs)
        assert len(scores) == 10
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_batch_empty(self):
        """Empty list returns empty list."""
        assert self.model.score_quality_batch([]) == []

    def test_batch_exceeds_max(self):
        """64 items with max_batch_size=32 processes in 2 chunks correctly."""
        pairs = [("", f"Content item {i}") for i in range(64)]
        scores = self.model.score_quality_batch(pairs, max_batch_size=32)
        assert len(scores) == 64
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_batch_mixed_lengths(self):
        """Short and long texts in the same batch both produce valid scores."""
        short = "Hi"
        long_text = "Detailed technical implementation " * 50
        pairs = [("", short), ("", long_text)]
        scores = self.model.score_quality_batch(pairs)
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0


@pytest.mark.skipif(not MS_MARCO_AVAILABLE, reason="MS-MARCO ONNX model not exported")
class TestScoreQualityBatchMSMarco:
    """Test batched scoring with MS-MARCO cross-encoder model."""

    def setup_method(self):
        self.model = get_onnx_ranker_model(
            model_name="ms-marco-MiniLM-L-6-v2", device="cpu"
        )

    def test_batch_single_item(self):
        """Batch of 1 matches sequential output for cross-encoder."""
        query, content = "python async patterns", "Use asyncio.gather for concurrency."
        seq_score = self.model.score_quality(query, content)
        batch_scores = self.model.score_quality_batch([(query, content)])
        assert len(batch_scores) == 1
        assert abs(batch_scores[0] - seq_score) < 1e-5

    def test_batch_multiple(self):
        """Batch of 10 produces 10 valid cross-encoder scores."""
        pairs = [("search query", f"Document content {i}") for i in range(10)]
        scores = self.model.score_quality_batch(pairs)
        assert len(scores) == 10
        for s in scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# QualityEvaluator.evaluate_quality_batch()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa ONNX model not exported")
class TestEvaluateQualityBatch:
    """Test QualityEvaluator batch evaluation."""

    async def test_single_model_batch(self):
        """Single-model batch returns scores for all memories."""
        config = QualityConfig(
            ai_provider="local",
            local_model="nvidia-quality-classifier-deberta",
            local_device="cpu",
            fallback_enabled=False,
        )
        evaluator = QualityEvaluator(config=config)
        memories = [_make_memory(f"Memory content {i} with details.") for i in range(5)]

        scores = await evaluator.evaluate_quality_batch("test query", memories)
        assert len(scores) == 5
        for s in scores:
            assert 0.0 <= s <= 1.0
        # Check provider metadata was set
        for m in memories:
            assert m.metadata.get("quality_provider") == "onnx_local"

    async def test_disabled_returns_neutral(self):
        """Disabled quality system returns 0.5 for all."""
        config = QualityConfig(enabled=False)
        evaluator = QualityEvaluator(config=config)
        memories = [_make_memory("content")] * 3
        scores = await evaluator.evaluate_quality_batch("q", memories)
        assert scores == [0.5, 0.5, 0.5]

    async def test_empty_batch(self):
        """Empty memories list returns empty scores."""
        config = QualityConfig(enabled=False)
        evaluator = QualityEvaluator(config=config)
        assert await evaluator.evaluate_quality_batch("q", []) == []


@pytest.mark.skipif(
    not (DEBERTA_AVAILABLE and MS_MARCO_AVAILABLE),
    reason="Both DeBERTa and MS-MARCO required for fallback test",
)
class TestEvaluateQualityBatchFallback:
    """Test batch evaluation with DeBERTa + MS-MARCO fallback."""

    async def test_fallback_batch(self):
        """Fallback mode scores batch with two-pass strategy."""
        config = QualityConfig(
            ai_provider="local",
            local_model="nvidia-quality-classifier-deberta,ms-marco-MiniLM-L-6-v2",
            local_device="cpu",
            fallback_enabled=True,
        )
        evaluator = QualityEvaluator(config=config)
        memories = [
            _make_memory("Well-documented API with comprehensive examples."),
            _make_memory("x"),  # Low-quality, likely triggers MS-MARCO rescue
            _make_memory("Technical implementation of DBSCAN clustering algorithm."),
        ]
        scores = await evaluator.evaluate_quality_batch("", memories)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# AsyncQualityScorer batched worker
# ---------------------------------------------------------------------------

class TestAsyncScorerBatching:
    """Test AsyncQualityScorer batch dequeue."""

    async def test_worker_dequeues_multiple_items(self):
        """Worker accumulates and processes a batch of items."""
        scorer = AsyncQualityScorer()
        # Mock evaluator to track batch calls
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate_quality_batch = AsyncMock(
            return_value=[0.8, 0.7, 0.6]
        )
        scorer.evaluator = mock_evaluator

        # Mock composite scorer
        mock_composite = AsyncMock()
        mock_composite.calculate_quality_score = AsyncMock(return_value=0.75)
        scorer.scorer = mock_composite

        scorer.batch_size = 32
        await scorer.start()

        memories = [_make_memory(f"Memory {i}") for i in range(3)]
        for m in memories:
            await scorer.score_memory(m, "test query", None)

        # Give worker time to process
        await asyncio.sleep(2.0)
        await scorer.stop()

        assert scorer.stats["total_scored"] == 3
        assert scorer.stats["total_errors"] == 0

    async def test_batch_size_from_env(self, monkeypatch):
        """MCP_QUALITY_BATCH_SIZE env var configures batch size."""
        monkeypatch.setenv("MCP_QUALITY_BATCH_SIZE", "16")
        scorer = AsyncQualityScorer()
        assert scorer.batch_size == 16


# ---------------------------------------------------------------------------
# SqliteVecMemoryStorage.store_batch() — uses mocks to avoid full DB setup
# ---------------------------------------------------------------------------

class TestStoreBatchMocked:
    """Test store_batch() with mocked database and embedding model."""

    async def test_batch_returns_correct_count(self):
        """store_batch returns one result per input memory."""
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        # Mock minimal attributes
        storage.conn = MagicMock()
        storage.conn.execute = MagicMock()
        storage.embedding_model = MagicMock()
        storage.enable_cache = False
        storage.embedding_dimension = 384
        storage.semantic_dedup_enabled = False

        # Mock: no hash duplicates found
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        storage.conn.execute.return_value = mock_cursor

        # Mock embedding model — return fake embeddings
        fake_embeddings = np.random.rand(3, 384).astype(np.float32)
        storage.embedding_model.encode.return_value = fake_embeddings

        # Mock _execute_with_retry to just call the function
        async def mock_retry(fn, *args, **kwargs):
            return fn()
        storage._execute_with_retry = mock_retry

        # Mock cursor.lastrowid
        call_count = [0]
        orig_execute = storage.conn.execute

        def side_effect_execute(*args, **kwargs):
            result = MagicMock()
            call_count[0] += 1
            result.lastrowid = call_count[0]
            result.fetchone.return_value = None  # No duplicates
            return result

        storage.conn.execute = MagicMock(side_effect=side_effect_execute)
        storage.conn.commit = MagicMock()

        memories = [_make_memory(f"Batch test content {i}") for i in range(3)]
        results = await storage.store_batch(memories)

        assert len(results) == 3
        # Embedding model should have been called once with batch
        storage.embedding_model.encode.assert_called_once()

    async def test_batch_embedding_failure_rolls_back_memory(self):
        """If embedding INSERT fails, SAVEPOINT rollback prevents orphaned memories row."""
        import sqlite3
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        storage.conn = MagicMock()
        storage.embedding_model = MagicMock()
        storage.enable_cache = False
        storage.embedding_dimension = 384
        storage.semantic_dedup_enabled = False

        fake_embeddings = np.random.rand(1, 384).astype(np.float32)
        storage.embedding_model.encode.return_value = fake_embeddings

        async def mock_retry(fn, *args, **kwargs):
            return fn()
        storage._execute_with_retry = mock_retry
        storage.conn.commit = MagicMock()

        # Track which SQL statements are executed
        executed_sql = []

        def side_effect_execute(sql, *args, **kwargs):
            executed_sql.append(sql.strip() if isinstance(sql, str) else str(sql))
            # Fail the embedding INSERT
            if 'memory_embeddings' in str(sql):
                raise sqlite3.Error("simulated embedding failure")
            result = MagicMock()
            result.lastrowid = 1
            result.fetchone.return_value = None
            return result

        storage.conn.execute = MagicMock(side_effect=side_effect_execute)

        memories = [_make_memory("Will fail embedding")]
        results = await storage.store_batch(memories)

        assert len(results) == 1
        assert results[0][0] is False
        assert "Insert failed" in results[0][1]

        # Verify SAVEPOINT was created and rolled back
        sql_strs = " ".join(executed_sql)
        assert "SAVEPOINT batch_item" in sql_strs
        assert "ROLLBACK TO SAVEPOINT batch_item" in sql_strs
        assert "RELEASE SAVEPOINT batch_item" in sql_strs

    async def test_batch_empty(self):
        """Empty batch returns empty results."""
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        results = await storage.store_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# SemanticCompressionEngine parallel processing
# ---------------------------------------------------------------------------

class TestCompressionParallel:
    """Test that asyncio.gather produces same results as sequential."""

    async def test_parallel_matches_sequential(self):
        """Parallel compression produces same number of results."""
        from mcp_memory_service.consolidation.compression import (
            SemanticCompressionEngine,
        )
        from mcp_memory_service.consolidation.base import (
            ConsolidationConfig,
            MemoryCluster,
        )

        config = ConsolidationConfig()
        engine = SemanticCompressionEngine(config)

        # Create test memories and clusters
        memories = [_make_memory(f"Memory about topic {i} with detailed content.") for i in range(6)]
        for m in memories:
            m.embedding = np.random.rand(384).tolist()

        cluster1 = MemoryCluster(
            cluster_id="c1",
            memory_hashes=[memories[0].content_hash, memories[1].content_hash, memories[2].content_hash],
            centroid_embedding=np.mean([m.embedding for m in memories[:3]], axis=0).tolist(),
            coherence_score=0.8,
            created_at=datetime.now(),
            theme_keywords=["topic", "test"],
        )
        cluster2 = MemoryCluster(
            cluster_id="c2",
            memory_hashes=[memories[3].content_hash, memories[4].content_hash, memories[5].content_hash],
            centroid_embedding=np.mean([m.embedding for m in memories[3:]], axis=0).tolist(),
            coherence_score=0.7,
            created_at=datetime.now(),
            theme_keywords=["another", "test"],
        )

        results = await engine.process([cluster1, cluster2], memories)
        assert len(results) == 2
        assert results[0].cluster_id == "c1"
        assert results[1].cluster_id == "c2"
        for r in results:
            assert r.compressed_memory is not None
            assert r.source_memory_count == 3

    async def test_parallel_handles_exceptions(self):
        """One failing cluster doesn't abort the others."""
        from mcp_memory_service.consolidation.compression import (
            SemanticCompressionEngine,
        )
        from mcp_memory_service.consolidation.base import (
            ConsolidationConfig,
            MemoryCluster,
        )

        config = ConsolidationConfig()
        engine = SemanticCompressionEngine(config)

        memories = [_make_memory(f"Memory {i} about testing.") for i in range(3)]
        for m in memories:
            m.embedding = np.random.rand(384).tolist()

        good_cluster = MemoryCluster(
            cluster_id="good",
            memory_hashes=[m.content_hash for m in memories],
            centroid_embedding=np.random.rand(384).tolist(),
            coherence_score=0.8,
            created_at=datetime.now(),
            theme_keywords=["test"],
        )
        # Bad cluster references nonexistent hashes — will be filtered out
        bad_cluster = MemoryCluster(
            cluster_id="bad",
            memory_hashes=["nonexistent_hash_1", "nonexistent_hash_2"],
            centroid_embedding=np.random.rand(384).tolist(),
            coherence_score=0.5,
            created_at=datetime.now(),
            theme_keywords=["missing"],
        )

        results = await engine.process([good_cluster, bad_cluster], memories)
        # Good cluster should succeed, bad cluster has no matching memories → filtered
        assert len(results) == 1
        assert results[0].cluster_id == "good"
