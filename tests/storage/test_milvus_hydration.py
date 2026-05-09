"""Unit tests for MilvusMemoryStorage embedding hydration on the
consolidation read path.

These tests exercise the opt-in ``include_embeddings`` kwarg landed in
#878 (Milvus internals), #881 (base ABC), and the current PR (consumer
switch). They are mock-based and do NOT require a live Milvus server
or the sentence-transformers model cache — see
``tests/integration/test_milvus_lite_consolidation.py`` for the
end-to-end integration test.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The module under test imports pymilvus and sentence_transformers at
# module load time. Stub them out before importing the storage class.
import sys
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pymilvus", MagicMock())

from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402


def _make_storage() -> MilvusMemoryStorage:
    """Return a ``MilvusMemoryStorage`` skipping ``__init__`` so no network
    or model loading happens. Subsequent tests populate the attributes they
    need."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    return storage


def _row(**overrides: Any) -> Dict[str, Any]:
    """Build a well-formed Milvus row dict with optional overrides."""
    # Pick timestamps close to "now" so Memory.__post_init__ accepts them
    # unchanged (Memory auto-rewrites timestamps that are "too old" to the
    # current wall clock, which is harmless for storage but would add
    # nondeterministic microsecond drift between paired test calls).
    import time
    now = time.time()
    base = {
        "id": "abc123",
        "content": "hello world",
        "tags": ",python,",
        "memory_type": "note",
        "metadata": "{}",
        "created_at": now - 1.0,
        "updated_at": now,
        "created_at_iso": None,
        "updated_at_iso": None,
    }
    base.update(overrides)
    return base


# -- _coerce_vector ---------------------------------------------------------


class TestCoerceVector:
    """``_coerce_vector`` converts ``row['vector']`` to a ``list[float]``
    or returns ``None``. Never raises (Requirement 1.5)."""

    def test_missing_key_returns_none(self):
        assert _make_storage()._coerce_vector({"id": "x"}) is None

    def test_none_value_returns_none(self):
        assert _make_storage()._coerce_vector({"vector": None}) is None

    def test_empty_list_returns_none(self):
        assert _make_storage()._coerce_vector({"vector": []}) is None

    def test_list_passthrough(self):
        v = [0.1, 0.2, 0.3, 0.4]
        assert _make_storage()._coerce_vector({"vector": v}) == v

    def test_tuple_converted_to_list(self):
        out = _make_storage()._coerce_vector({"vector": (0.1, 0.2, 0.3, 0.4)})
        assert out == [0.1, 0.2, 0.3, 0.4]
        assert isinstance(out, list)

    def test_numpy_array_converted_via_tolist(self):
        np = pytest.importorskip("numpy")
        out = _make_storage()._coerce_vector({"vector": np.array([0.5, 0.6, 0.7, 0.8])})
        assert out == [0.5, 0.6, 0.7, 0.8]
        assert isinstance(out, list)

    @pytest.mark.parametrize("bad_value", [42, "not a vector", {"a": 1}, 3.14])
    def test_unexpected_type_returns_none_without_raising(self, bad_value):
        """Unexpected scalar types degrade to None + log warning, never raise."""
        out = _make_storage()._coerce_vector({"vector": bad_value, "id": "xyz"})
        assert out is None


# -- _entity_to_memory ------------------------------------------------------


class TestEntityToMemoryHydration:
    """``_entity_to_memory`` only hydrates ``Memory.embedding`` when the
    ``include_embedding=True`` flag is passed."""

    def test_default_flag_false_yields_embedding_none(self):
        row = _row(vector=[0.1, 0.2, 0.3, 0.4])
        mem = _make_storage()._entity_to_memory(row)
        assert mem is not None
        assert mem.embedding is None
        assert mem.content == "hello world"
        assert mem.content_hash == "abc123"

    def test_include_embedding_true_hydrates(self):
        row = _row(vector=[0.1, 0.2, 0.3, 0.4])
        mem = _make_storage()._entity_to_memory(row, include_embedding=True)
        assert mem is not None
        assert mem.embedding == [0.1, 0.2, 0.3, 0.4]

    def test_include_embedding_true_with_missing_vector_yields_none(self):
        row = _row()  # no 'vector' key
        mem = _make_storage()._entity_to_memory(row, include_embedding=True)
        assert mem is not None
        assert mem.embedding is None

    def test_include_embedding_true_with_none_vector_yields_none(self):
        row = _row(vector=None)
        mem = _make_storage()._entity_to_memory(row, include_embedding=True)
        assert mem is not None
        assert mem.embedding is None

    def test_scalar_fields_preserved_regardless_of_flag(self):
        row = _row(vector=[0.1, 0.2, 0.3, 0.4])
        storage = _make_storage()
        m_no = storage._entity_to_memory(row, include_embedding=False)
        m_yes = storage._entity_to_memory(row, include_embedding=True)
        for field in (
            "content",
            "content_hash",
            "tags",
            "memory_type",
            "metadata",
            "created_at",
            "updated_at",
            "created_at_iso",
            "updated_at_iso",
        ):
            assert getattr(m_no, field) == getattr(m_yes, field), (
                f"{field} differs: {getattr(m_no, field)!r} vs "
                f"{getattr(m_yes, field)!r}"
            )


# -- _OUTPUT_FIELDS_* constants ---------------------------------------------


class TestOutputFieldsConstants:
    def test_base_excludes_vector(self):
        assert "vector" not in MilvusMemoryStorage._OUTPUT_FIELDS_BASE

    def test_with_vector_includes_vector(self):
        assert "vector" in MilvusMemoryStorage._OUTPUT_FIELDS_WITH_VECTOR

    def test_alias_points_to_base(self):
        # Back-compat: the old name is still resolvable and maps to BASE.
        assert MilvusMemoryStorage._OUTPUT_FIELDS is MilvusMemoryStorage._OUTPUT_FIELDS_BASE


# -- output_fields selection in _drain_query_iterator ----------------------


class TestDrainQueryIteratorOutputFields:
    """Verify CRUD vs. consolidation read paths request different
    ``output_fields`` from the underlying Milvus client."""

    def _make_storage_with_mock_client(self):
        storage = _make_storage()

        # Fake iterator that produces nothing so we exit the drain loop quickly.
        mock_iterator = MagicMock()
        mock_iterator.next.return_value = []
        mock_iterator.close = MagicMock()

        storage.client = MagicMock()
        storage.client.query_iterator = MagicMock(return_value=mock_iterator)
        return storage

    def test_crud_path_excludes_vector_from_output_fields(self):
        storage = self._make_storage_with_mock_client()
        storage._drain_query_iterator("id != ''", include_embeddings=False)
        kwargs = storage.client.query_iterator.call_args.kwargs
        assert "vector" not in kwargs["output_fields"], (
            f"CRUD path must not request 'vector' but got "
            f"output_fields={kwargs['output_fields']!r}"
        )

    def test_consolidation_path_includes_vector_in_output_fields(self):
        storage = self._make_storage_with_mock_client()
        storage._drain_query_iterator("id != ''", include_embeddings=True)
        kwargs = storage.client.query_iterator.call_args.kwargs
        assert "vector" in kwargs["output_fields"], (
            f"Consolidation path must request 'vector' but got "
            f"output_fields={kwargs['output_fields']!r}"
        )


# -- _log_hydration_stats observability ------------------------------------


class TestLogHydrationStats:
    """DEBUG counter per call, WARNING iff total > 0 and hydrated == 0, and
    never log raw vector values (Requirement 7.3)."""

    def _make_memory(self, embedding):
        return Memory(
            content="x",
            content_hash="h",
            tags=[],
            memory_type="note",
            metadata={},
            embedding=embedding,
        )

    def test_debug_counter_when_all_hydrated(self, caplog):
        storage = _make_storage()
        memories = [self._make_memory([0.1, 0.2, 0.3, 0.4]) for _ in range(3)]
        with caplog.at_level("DEBUG", logger="src.mcp_memory_service.storage.milvus"):
            storage._log_hydration_stats(memories)
        debug_lines = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert any(
            "3/3" in r.message or "3/3" in r.getMessage() for r in debug_lines
        ), f"expected 3/3 counter, got {[r.getMessage() for r in debug_lines]}"
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_warning_when_total_positive_but_zero_hydrated(self, caplog):
        storage = _make_storage()
        memories = [self._make_memory(None) for _ in range(3)]
        with caplog.at_level("WARNING", logger="src.mcp_memory_service.storage.milvus"):
            storage._log_hydration_stats(memories)
        warn_lines = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warn_lines) == 1, (
            f"expected exactly 1 WARNING, got {[r.getMessage() for r in warn_lines]}"
        )
        assert storage.collection_name in warn_lines[0].getMessage()

    def test_no_warning_on_empty_result_set(self, caplog):
        storage = _make_storage()
        with caplog.at_level("WARNING", logger="src.mcp_memory_service.storage.milvus"):
            storage._log_hydration_stats([])
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_never_logs_raw_vector_values(self, caplog):
        storage = _make_storage()
        secret_vector = [0.1337, 0.4242, 0.9999, 0.5555]
        memories = [self._make_memory(secret_vector)]
        with caplog.at_level("DEBUG", logger="src.mcp_memory_service.storage.milvus"):
            storage._log_hydration_stats(memories)
        joined = " ".join(r.getMessage() for r in caplog.records)
        for value in secret_vector:
            assert str(value) not in joined, (
                f"raw vector value {value!r} leaked into logs: {joined!r}"
            )
