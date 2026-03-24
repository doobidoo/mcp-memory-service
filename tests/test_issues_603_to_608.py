"""Tests for bug fixes #603-#608.

Each test class corresponds to a specific GitHub issue.
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.consolidation.base import ConsolidationBase, ConsolidationConfig
from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator, RelevanceScore


# ---------------------------------------------------------------------------
# #603 — invalid memory_type "learning_note" should not exist
# ---------------------------------------------------------------------------

class TestIssue603LearningNoteType:
    """memory_type='learning_note' is not in the ontology; it should be 'learning'."""

    def test_learning_note_is_coerced_to_observation(self):
        """Verify the ontology rejects 'learning_note' (pre-fix behaviour)."""
        m = Memory(content="test", content_hash="abc", memory_type="learning_note")
        assert m.memory_type == "observation"

    def test_learning_is_valid_type(self):
        """'learning' must be accepted as-is by the ontology."""
        m = Memory(content="test", content_hash="abc", memory_type="learning")
        assert m.memory_type == "learning"

    def test_server_impl_uses_learning_type(self):
        """Verify server_impl.py no longer contains 'learning_note' as a memory_type."""
        import pathlib
        server_impl_path = pathlib.Path(__file__).parent.parent / "src" / "mcp_memory_service" / "server_impl.py"
        source = server_impl_path.read_text()
        assert 'memory_type="learning_note"' not in source
        assert 'memory_type="learning"' in source


# ---------------------------------------------------------------------------
# #604 — update_memory_relevance_metadata must not call memory.touch()
# ---------------------------------------------------------------------------

class TestIssue604NoTouchInRelevanceScoring:
    """Relevance scoring is a read path and must not mutate updated_at."""

    @pytest.mark.asyncio
    async def test_relevance_metadata_does_not_advance_updated_at(self):
        """updated_at should remain unchanged after update_memory_relevance_metadata."""
        original_updated = 1700000000.0
        memory = Memory(
            content="test content",
            content_hash="hash604",
            tags=["test"],
            memory_type="observation",
            created_at=1690000000.0,
        )
        # Override updated_at after __post_init__ (which auto-sets it to now)
        memory.updated_at = original_updated

        score = RelevanceScore(
            memory_hash="hash604",
            total_score=0.75,
            base_importance=0.5,
            decay_factor=0.9,
            connection_boost=0.05,
            access_boost=0.1,
            metadata={},
        )

        config = ConsolidationConfig()
        calculator = ExponentialDecayCalculator(config)
        result = await calculator.update_memory_relevance_metadata(memory, score)

        # updated_at must not have changed
        assert result.updated_at == original_updated
        # But relevance metadata should be written
        assert result.metadata.get("relevance_score") == 0.75
        assert result.metadata.get("decay_factor") == 0.9

    def test_touch_not_called_in_source(self):
        """Verify source code does not call touch() as an active statement."""
        import inspect
        source = inspect.getsource(ExponentialDecayCalculator.update_memory_relevance_metadata)
        # Filter out comments — only check actual code lines
        code_lines = [
            line.strip() for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any("memory.touch()" in line for line in code_lines)


# ---------------------------------------------------------------------------
# #605 — preserve_timestamps in batch updates
# ---------------------------------------------------------------------------

class TestIssue605PreserveTimestamps:
    """update_memories_batch must accept preserve_timestamps parameter."""

    def test_update_memories_batch_signature_has_preserve_timestamps(self):
        """The method signature must include preserve_timestamps."""
        import inspect
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        sig = inspect.signature(SqliteVecMemoryStorage.update_memories_batch)
        assert "preserve_timestamps" in sig.parameters

    def test_base_storage_signature_has_preserve_timestamps(self):
        """MemoryStorage.update_memories_batch must also have the parameter."""
        import inspect
        from mcp_memory_service.storage.base import MemoryStorage

        sig = inspect.signature(MemoryStorage.update_memories_batch)
        assert "preserve_timestamps" in sig.parameters

    def test_hybrid_storage_signature_has_preserve_timestamps(self):
        """HybridMemoryStorage.update_memories_batch must also have the parameter."""
        import inspect
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage

        sig = inspect.signature(HybridMemoryStorage.update_memories_batch)
        assert "preserve_timestamps" in sig.parameters

    def test_update_memory_metadata_preserves_on_pure_metadata_change(self):
        """update_memory_metadata with preserve_timestamps=True should check structural changes."""
        import inspect
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        source = inspect.getsource(SqliteVecMemoryStorage.update_memory_metadata)
        assert "structural_change" in source


# ---------------------------------------------------------------------------
# #606 — _get_memory_age_days should use max(created_at, updated_at)
# ---------------------------------------------------------------------------

class _ConcreteConsolidation(ConsolidationBase):
    """Minimal concrete subclass for testing _get_memory_age_days."""
    async def process(self, memories):
        return memories


class TestIssue606AgeUsesUpdatedAt:
    """Memory age should be calculated from the most recent timestamp."""

    def _make_base(self):
        config = ConsolidationConfig()
        base = _ConcreteConsolidation(config)
        return base

    def test_age_uses_created_at_when_no_updated_at(self):
        """When updated_at is None, fall back to created_at."""
        base = self._make_base()
        now = datetime.now(timezone.utc)
        created_ts = now.timestamp() - (30 * 86400)

        memory = Memory(content="test", content_hash="h606a", created_at=created_ts)
        # Override updated_at to None after __post_init__ auto-sets it
        memory.updated_at = None
        age = base._get_memory_age_days(memory, reference_time=now)
        assert age == 30

    def test_age_uses_updated_at_when_more_recent(self):
        """When updated_at is more recent than created_at, use updated_at."""
        base = self._make_base()
        now = datetime.now(timezone.utc)
        created_ts = now.timestamp() - (90 * 86400)
        updated_ts = now.timestamp() - (5 * 86400)

        memory = Memory(content="test", content_hash="h606b", created_at=created_ts)
        memory.updated_at = updated_ts
        age = base._get_memory_age_days(memory, reference_time=now)
        assert age == 5

    def test_age_uses_created_at_when_more_recent_than_updated(self):
        """Edge case: created_at > updated_at (shouldn't happen, but be safe)."""
        base = self._make_base()
        now = datetime.now(timezone.utc)
        created_ts = now.timestamp() - (5 * 86400)
        updated_ts = now.timestamp() - (30 * 86400)

        memory = Memory(content="test", content_hash="h606c", created_at=created_ts)
        memory.updated_at = updated_ts
        age = base._get_memory_age_days(memory, reference_time=now)
        assert age == 5


# ---------------------------------------------------------------------------
# #607 — _DIMENSION_CACHE miss on _MODEL_CACHE hit
# ---------------------------------------------------------------------------

class TestIssue607DimensionCacheFallback:
    """When _MODEL_CACHE hits but _DIMENSION_CACHE misses, fall back to model attribute."""

    def test_source_has_fallback_for_external(self):
        """External embedding cache-hit block should have hasattr fallback."""
        import inspect
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        source = inspect.getsource(SqliteVecMemoryStorage._initialize_embedding_model)
        assert "hasattr(self.embedding_model, 'embedding_dimension')" in source

    def test_source_has_fallback_for_sentence_transformer(self):
        """SentenceTransformer cache-hit block should have get_sentence_embedding_dimension fallback."""
        import inspect
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        source = inspect.getsource(SqliteVecMemoryStorage._initialize_embedding_model)
        assert "get_sentence_embedding_dimension" in source


# ---------------------------------------------------------------------------
# #608 — _HashEmbeddingModel should detect existing DB dimension
# ---------------------------------------------------------------------------

class TestIssue608HashModelDimensionFromDB:
    """Hash embedding fallback must check existing DB schema for dimension."""

    def test_source_checks_existing_dimension_before_hash_model(self):
        """Both hash fallback paths should call _get_existing_db_embedding_dimension."""
        import inspect
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        source = inspect.getsource(SqliteVecMemoryStorage._initialize_embedding_model)
        count = source.count("_get_existing_db_embedding_dimension")
        assert count >= 2, f"Expected >=2 calls to _get_existing_db_embedding_dimension, found {count}"

    def test_hash_model_respects_custom_dimension(self):
        """_HashEmbeddingModel should work with any dimension."""
        from mcp_memory_service.storage.sqlite_vec import _HashEmbeddingModel

        model = _HashEmbeddingModel(768)
        assert model.embedding_dimension == 768
        result = model.encode(["hello world"])
        assert len(result[0]) == 768

    def test_hash_model_default_dimension(self):
        """_HashEmbeddingModel with default 384 should produce 384-dim vectors."""
        from mcp_memory_service.storage.sqlite_vec import _HashEmbeddingModel

        model = _HashEmbeddingModel(384)
        assert model.embedding_dimension == 384
        result = model.encode(["test"])
        assert len(result[0]) == 384
