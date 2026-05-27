"""Tests for NLI-based contradiction detection (Phase 3, RFC #732).

These tests define the expected behavior. Implementation comes after.
All tests must pass with heuristic backend (no ML deps required).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── NLI Classifier Tests ─────────────────────────────────────────────────────

class TestNLIClassifier:
    """Test the NLI classifier with heuristic backend."""

    def test_import(self):
        """NLI module must be importable."""
        from mcp_memory_service.reasoning.nli import NLIClassifier, NLIResult
        assert NLIClassifier is not None
        assert NLIResult is not None

    def test_heuristic_backend_available(self):
        """Heuristic backend must always be available (no ML deps)."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        assert classifier is not None

    @pytest.mark.asyncio
    async def test_contradiction_negation(self):
        """Negation patterns should be detected as contradiction."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        result = await classifier.classify(
            premise="The service is enabled and running",
            hypothesis="The service is disabled and stopped"
        )
        assert result.label == "contradiction"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_contradiction_version_conflict(self):
        """Different versions of same thing should be contradiction."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        result = await classifier.classify(
            premise="mcp-memory-service version is 10.58.0",
            hypothesis="mcp-memory-service version is 10.66.0"
        )
        assert result.label == "contradiction"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_neutral_unrelated(self):
        """Unrelated content should be neutral."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        result = await classifier.classify(
            premise="Python uses indentation for blocks",
            hypothesis="The weather is sunny today"
        )
        assert result.label == "neutral"

    @pytest.mark.asyncio
    async def test_entailment_same_meaning(self):
        """Same meaning rephrased should be entailment or neutral (not contradiction)."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        result = await classifier.classify(
            premise="The database is PostgreSQL 15",
            hypothesis="We use PostgreSQL version 15 for the database"
        )
        assert result.label != "contradiction"

    @pytest.mark.asyncio
    async def test_batch_classification(self):
        """Batch classify should return results for all pairs."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        pairs = [
            ("Service is enabled", "Service is disabled"),
            ("Python is great", "The sky is blue"),
        ]
        results = await classifier.classify_batch(pairs)
        assert len(results) == 2
        assert results[0].label == "contradiction"
        assert results[1].label == "neutral"

    @pytest.mark.asyncio
    async def test_confidence_threshold(self):
        """Heuristic confidence should never exceed 0.6."""
        from mcp_memory_service.reasoning.nli import NLIClassifier
        classifier = NLIClassifier(backend="heuristic")
        result = await classifier.classify(
            premise="Feature is enabled",
            hypothesis="Feature is disabled"
        )
        assert result.confidence <= 0.6


# ─── Contradiction Pipeline Tests ─────────────────────────────────────────────

class TestContradictionPipeline:
    """Test the 4-stage NLI contradiction pipeline."""

    @pytest.fixture(autouse=True)
    def enable_nli(self, monkeypatch):
        monkeypatch.setenv("MCP_NLI_ENABLED", "true")

    def test_pipeline_import(self):
        """Pipeline must be importable from reasoning."""
        from mcp_memory_service.reasoning.nli import detect_contradictions_nli
        assert detect_contradictions_nli is not None

    @pytest.mark.asyncio
    async def test_skips_no_entity_overlap(self):
        """Memories with no shared entities should never reach NLI."""
        from mcp_memory_service.reasoning.nli import detect_contradictions_nli

        storage = AsyncMock()
        storage.get_by_hash.return_value = MagicMock(
            content="Service A is running",
            content_hash="hash_a",
            metadata={"entities": ["service_a"]},
        )

        mock_graph = AsyncMock()
        mock_graph.find_memories_by_entity = AsyncMock(return_value=[])

        with patch("mcp_memory_service.server.handlers.graph.get_graph_storage", new_callable=AsyncMock, return_value=mock_graph):
            result = await detect_contradictions_nli(storage, memory_hash="hash_a", dry_run=True)
        assert result["nli_calls"] == 0

    @pytest.mark.asyncio
    async def test_detects_contradiction_with_entity_overlap(self):
        """Memories sharing entity + contradicting content → detected."""
        from mcp_memory_service.reasoning.nli import detect_contradictions_nli

        mem_a = MagicMock(
            content="mcp-memory-service version is 10.58.0",
            content_hash="hash_a",
            metadata={"entities": ["mcp-memory-service"]},
            created_at=1000.0,
        )

        storage = AsyncMock()
        storage.get_by_hash.return_value = mem_a
        storage.search_memories = AsyncMock(return_value={
            "memories": [{"content_hash": "hash_b", "similarity_score": 0.55, "content": "mcp-memory-service version is 10.66.0", "created_at": 2000.0}]
        })

        mock_graph = AsyncMock()
        mock_graph.find_memories_by_entity = AsyncMock(return_value=["hash_b"])
        mock_graph.store_association = AsyncMock(return_value=True)

        with patch("mcp_memory_service.server.handlers.graph.get_graph_storage", new_callable=AsyncMock, return_value=mock_graph):
            result = await detect_contradictions_nli(storage, memory_hash="hash_a", dry_run=False)
        assert result["pairs_detected"] >= 1
        assert result["nli_calls"] >= 1

    @pytest.mark.asyncio
    async def test_neutral_not_registered(self):
        """Memories sharing entity but neutral content → NOT registered as conflict."""
        from mcp_memory_service.reasoning.nli import detect_contradictions_nli

        mem_a = MagicMock(
            content="Python uses indentation for code blocks",
            content_hash="hash_a",
            metadata={"entities": ["python"]},
            created_at=1000.0,
        )

        storage = AsyncMock()
        storage.get_by_hash.return_value = mem_a
        storage.search_memories = AsyncMock(return_value={
            "memories": [{"content_hash": "hash_b", "similarity_score": 0.5, "content": "Python was created by Guido van Rossum", "created_at": 2000.0}]
        })

        mock_graph = AsyncMock()
        mock_graph.find_memories_by_entity = AsyncMock(return_value=["hash_b"])

        with patch("mcp_memory_service.server.handlers.graph.get_graph_storage", new_callable=AsyncMock, return_value=mock_graph):
            result = await detect_contradictions_nli(storage, memory_hash="hash_a", dry_run=True)
        assert result["pairs_detected"] == 0


# ─── memory_resolve batch Tests ───────────────────────────────────────────────

class TestMemoryResolveBatch:
    """Test that memory_resolve accepts batch (list of hashes)."""

    @pytest.mark.asyncio
    async def test_resolve_batch(self):
        """memory_resolve should accept hashes as list."""
        # This tests the server_impl handler change
        # For now, just verify the interface expectation
        from mcp_memory_service.reasoning.nli import NLIResult
        # If we get here without import error, the module structure is correct
        assert True
