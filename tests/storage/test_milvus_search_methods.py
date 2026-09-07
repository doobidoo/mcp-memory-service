"""Unit tests for MilvusMemoryStorage search method overrides:
- retrieve_with_quality_boost
- recall_memory
- search_memories

Mock-based — no live Milvus server required.

Reference: https://github.com/doobidoo/mcp-memory-service/issues/888
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("sentence_transformers")

from mcp_memory_service.models.memory import Memory, MemoryQueryResult  # noqa: E402
from mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402


# -- Fixtures ----------------------------------------------------------------


def _make_storage(has_bm25: bool = True) -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage skipping __init__."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage.embedding_model.encode = MagicMock(
        return_value=np.array([[0.1, 0.2, 0.3, 0.4]])
    )
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = has_bm25
    storage._lock = None
    storage._call_client = AsyncMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return storage


def _make_hit(
    content_hash: str = "h1",
    content: str = "test content",
    distance: float = 0.9,
    metadata: Optional[Dict[str, Any]] = None,
    created_at: Optional[float] = None,
    tags: str = ",test,",
) -> Dict[str, Any]:
    """Build a Milvus search hit."""
    now = time.time()
    return {
        "id": content_hash,
        "distance": distance,
        "content": content,
        "tags": tags,
        "memory_type": "note",
        "metadata": json.dumps(metadata or {}),
        "created_at": created_at or (now - 100),
        "updated_at": now - 50,
        "created_at_iso": None,
        "updated_at_iso": None,
    }


# -- retrieve_with_quality_boost ---------------------------------------------


class TestRetrieveWithQualityBoost:
    """Tests for native retrieve_with_quality_boost."""

    @pytest.mark.asyncio
    async def test_quality_boost_disabled_uses_plain_retrieve(self):
        """When quality_boost=False, delegates to retrieve directly."""
        storage = _make_storage()
        storage.retrieve = AsyncMock(return_value=[
            MemoryQueryResult(
                memory=Memory(content="c1", content_hash="h1", tags=["t"]),
                relevance_score=0.9,
                debug_info=None,
            )
        ])

        results = await storage.retrieve_with_quality_boost(
            "test query", n_results=5, quality_boost=False
        )

        assert len(results) == 1
        storage.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_boost_reranks_by_composite_score(self):
        """Results are reranked by composite (semantic + quality) score."""
        storage = _make_storage()

        # Item A: high semantic, low quality
        mem_a = Memory(content="a", content_hash="ha", tags=["t"],
                       metadata={"quality_score": 0.2})
        # Item B: low semantic, high quality
        mem_b = Memory(content="b", content_hash="hb", tags=["t"],
                       metadata={"quality_score": 0.95})

        storage.retrieve = AsyncMock(return_value=[
            MemoryQueryResult(memory=mem_a, relevance_score=0.9, debug_info=None),
            MemoryQueryResult(memory=mem_b, relevance_score=0.5, debug_info=None),
        ])

        results = await storage.retrieve_with_quality_boost(
            "test", n_results=2, quality_boost=True, quality_weight=0.7
        )

        # B should rank higher: 0.3*0.5 + 0.7*0.95 = 0.815
        # A: 0.3*0.9 + 0.7*0.2 = 0.41
        assert results[0].memory.content_hash == "hb"
        assert results[1].memory.content_hash == "ha"

    @pytest.mark.asyncio
    async def test_quality_boost_over_fetches_3x(self):
        """Retrieves 3x n_results for reranking pool."""
        storage = _make_storage()
        storage.retrieve = AsyncMock(return_value=[])

        await storage.retrieve_with_quality_boost(
            "test", n_results=5, quality_boost=True
        )

        call_args = storage.retrieve.call_args
        assert call_args[0][1] == 15  # 5 * 3

    @pytest.mark.asyncio
    async def test_invalid_quality_weight_raises(self):
        """quality_weight outside 0-1 raises ValueError."""
        storage = _make_storage()
        with pytest.raises(ValueError):
            await storage.retrieve_with_quality_boost(
                "test", quality_weight=1.5
            )


# -- recall_memory -----------------------------------------------------------


class TestRecallMemory:
    """Tests for native recall_memory with time expression parsing."""

    @pytest.mark.asyncio
    async def test_no_time_expression_uses_retrieve(self):
        """When no time expression in query, falls back to retrieve."""
        storage = _make_storage()
        storage.retrieve = AsyncMock(return_value=[
            MemoryQueryResult(
                memory=Memory(content="c1", content_hash="h1", tags=["t"]),
                relevance_score=0.8,
                debug_info=None,
            )
        ])

        results = await storage.recall_memory("python patterns", n_results=3)

        assert len(results) == 1
        assert results[0].content == "c1"

    @pytest.mark.asyncio
    async def test_with_time_expression_applies_filter(self):
        """Time expressions are parsed and used as Milvus filter."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", content="recent stuff", distance=0.85),
        ])

        with patch("src.mcp_memory_service.utils.time_parser.parse_time_expression") as mock_parse:
            mock_parse.return_value = (time.time() - 86400, time.time())
            results = await storage.recall_memory("what happened yesterday", n_results=5)

        assert len(results) == 1
        # Verify hybrid search was called with a time filter
        call_args = storage._run_hybrid_search.call_args
        filter_expr = call_args[0][2]  # third positional arg is tag_filter
        assert "created_at >=" in filter_expr

    @pytest.mark.asyncio
    async def test_not_initialized_returns_empty(self):
        """Returns empty list when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        results = await storage.recall_memory("test")
        assert results == []


# -- search_memories ---------------------------------------------------------


class TestSearchMemories:
    """Tests for native search_memories with Milvus-pushed filters."""

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        """Invalid mode returns error dict."""
        storage = _make_storage()
        result = await storage.search_memories(query="test", mode="invalid")
        assert result["error"] is not None
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_invalid_quality_boost_returns_error(self):
        """quality_boost outside 0-1 returns error."""
        storage = _make_storage()
        result = await storage.search_memories(query="test", quality_boost=2.0)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_semantic_mode_uses_hybrid_search(self):
        """Semantic mode with BM25 enabled uses hybrid search."""
        storage = _make_storage(has_bm25=True)
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9),
        ])

        result = await storage.search_memories(query="test query", mode="semantic")

        assert result["total"] == 1
        storage._run_hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_exact_mode_uses_get_by_exact_content(self):
        """Exact mode uses get_by_exact_content."""
        storage = _make_storage()
        storage.get_by_exact_content = AsyncMock(return_value=[
            Memory(content="exact match", content_hash="h1", tags=["t"]),
        ])

        result = await storage.search_memories(query="exact match", mode="exact")

        assert result["total"] == 1
        storage.get_by_exact_content.assert_called_once_with("exact match")

    @pytest.mark.asyncio
    async def test_time_filter_pushed_to_milvus(self):
        """Time filters are pushed as Milvus filter expressions."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.85),
        ])

        result = await storage.search_memories(
            query="test", after="2026-01-01", before="2026-06-01"
        )

        # Verify filter expression includes created_at constraints
        call_args = storage._run_hybrid_search.call_args
        filter_expr = call_args[0][2]
        assert "created_at >=" in filter_expr
        assert "created_at <=" in filter_expr

    @pytest.mark.asyncio
    async def test_tag_filter_pushed_to_milvus(self):
        """Tag filters are pushed as Milvus like expressions."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.85, tags=",python,"),
        ])

        result = await storage.search_memories(
            query="test", tags=["python"]
        )

        call_args = storage._run_hybrid_search.call_args
        filter_expr = call_args[0][2]
        assert "like" in filter_expr.lower() or "tags" in filter_expr.lower()

    @pytest.mark.asyncio
    async def test_time_only_search_uses_query(self):
        """Time-only search (no query) uses Milvus query with filter."""
        storage = _make_storage()
        now = time.time()
        storage._call_client = AsyncMock(return_value=[
            {"id": "h1", "content": "c1", "tags": ",test,", "memory_type": "note",
             "metadata": "{}", "created_at": now - 50, "updated_at": now - 10,
             "created_at_iso": None, "updated_at_iso": None},
        ])

        result = await storage.search_memories(
            query=None, after="2026-01-01"
        )

        assert result["total"] == 1
        # Should use "query" method, not "search"
        call_args = storage._call_client.call_args
        assert call_args[0][0] == "query"

    @pytest.mark.asyncio
    async def test_include_debug_adds_debug_info(self):
        """include_debug=True adds debug section to response."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9),
        ])

        result = await storage.search_memories(
            query="test", include_debug=True
        )

        assert "debug" in result
        assert "milvus_filter" in result["debug"]

    @pytest.mark.asyncio
    async def test_superseded_filtered_by_default(self):
        """Superseded memories are filtered out by default."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", metadata={"superseded_by": "winner"}),
            _make_hit(content_hash="h2", metadata={}),
        ])

        result = await storage.search_memories(query="test")

        assert result["total"] == 1
        assert result["memories"][0]["content_hash"] == "h2"

    @pytest.mark.asyncio
    async def test_include_superseded_true(self):
        """include_superseded=True returns superseded memories."""
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", metadata={"superseded_by": "winner"}),
            _make_hit(content_hash="h2", metadata={}),
        ])

        result = await storage.search_memories(query="test", include_superseded=True)

        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_not_initialized_returns_error(self):
        """Returns error when not initialized."""
        storage = _make_storage()
        storage._initialized = False
        result = await storage.search_memories(query="test")
        assert "error" in result

    # -- ranked mode (regression for ranking_weights kwarg, commit b4d2723a) --

    @pytest.mark.asyncio
    async def test_ranking_weights_kwarg_accepted(self):
        """Passing ranking_weights must not raise TypeError.

        Regression: base.search_memories grew a ranking_weights param in
        b4d2723a (#1028) but the Milvus override was not updated, so the
        handler's unconditional ranking_weights=... passthrough crashed every
        Milvus search with 'unexpected keyword argument'.
        """
        storage = _make_storage()
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9),
        ])

        # Even in semantic mode (handler always passes ranking_weights) this
        # must succeed rather than TypeError.
        result = await storage.search_memories(
            query="test", mode="semantic", ranking_weights={"semantic": 0.5}
        )

        assert result["total"] == 1
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_ranked_mode_accepted(self):
        """ranked mode is a valid mode and returns results."""
        storage = _make_storage(has_bm25=True)
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9),
            _make_hit(content_hash="h2", distance=0.7),
        ])

        result = await storage.search_memories(query="test query", mode="ranked")

        assert result["mode"] == "ranked"
        assert "error" not in result
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_ranked_mode_applies_multi_signal_rerank(self):
        """ranked mode invokes apply_ranked_rerank on the candidate pool."""
        storage = _make_storage(has_bm25=True)
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9),
        ])

        with patch(
            "src.mcp_memory_service.reasoning.ranked_search.apply_ranked_rerank"
        ) as mock_rerank:
            mock_rerank.side_effect = lambda results, weights=None: results
            result = await storage.search_memories(query="test", mode="ranked")

        mock_rerank.assert_called_once()
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_ranked_mode_requires_query(self):
        """ranked mode without query returns an error."""
        storage = _make_storage()
        result = await storage.search_memories(query=None, mode="ranked")
        assert "error" in result
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_ranked_mode_pushes_tag_filter_to_milvus(self):
        """ranked mode still pushes tag/time filters server-side.

        Milvus applies filters at search time (combined_filter), unlike the
        base class which over-fetches then post-filters. This guards that
        ranked mode does not bypass the filter expression.
        """
        storage = _make_storage(has_bm25=True)
        storage._run_hybrid_search = AsyncMock(return_value=[
            _make_hit(content_hash="h1", distance=0.9, tags=",python,"),
        ])

        result = await storage.search_memories(
            query="test", mode="ranked", tags=["python"]
        )

        assert result["mode"] == "ranked"
        assert "error" not in result
        # filter expression (3rd positional arg) must carry the tag clause
        call_args = storage._run_hybrid_search.call_args
        filter_expr = call_args[0][2]
        assert "like" in filter_expr.lower() or "tags" in filter_expr.lower()

    @pytest.mark.asyncio
    async def test_ranked_mode_over_fetches_3x(self):
        """ranked mode over-fetches a larger candidate pool before reranking.

        Mirrors quality_boost's 3x over-fetch so the multi-signal rerank has
        enough candidates to reorder (fetch_limit = limit * 3).
        """
        storage = _make_storage(has_bm25=True)
        storage._run_hybrid_search = AsyncMock(return_value=[])
        # capture the fetch_n passed to _retrieve_fetch_limit
        captured = {}
        orig = storage._retrieve_fetch_limit

        def _spy(fetch_limit, combined_filter):
            captured["fetch_limit"] = fetch_limit
            return orig(fetch_limit, combined_filter)

        storage._retrieve_fetch_limit = _spy

        await storage.search_memories(query="test", mode="ranked", limit=10)

        assert captured["fetch_limit"] == 30  # 10 * 3
