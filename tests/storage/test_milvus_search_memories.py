"""Unit tests for MilvusMemoryStorage.search_memories native override.

Mock-based tests — no live Milvus server required.

Reference: https://github.com/doobidoo/mcp-memory-service/issues/888
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402


def _make_storage() -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage skipping __init__."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_bm25 = False
    storage._has_content_lower = True
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    storage._run_search = AsyncMock(return_value=[])
    storage._run_hybrid_search = AsyncMock(return_value=[])
    storage._query_memories = AsyncMock(return_value=[])
    return storage


def _make_row(
    content_hash: str,
    content: str,
    *,
    tags: Optional[List[str]] = None,
    created_at: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = created_at if created_at is not None else time.time()
    return {
        "id": content_hash,
        "content": content,
        "tags": "," + ",".join(tags or ["test"]) + ",",
        "memory_type": "note",
        "metadata": json.dumps(metadata or {}),
        "created_at": now,
        "updated_at": now,
        "created_at_iso": None,
        "updated_at_iso": None,
    }


def _make_hit(
    content_hash: str,
    content: str,
    distance: float,
    *,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": content_hash,
        "distance": distance,
        "entity": _make_row(
            content_hash,
            content,
            tags=tags,
            metadata=metadata,
        ),
    }


class TestMilvusSearchMemories:
    """Tests for MilvusMemoryStorage.search_memories."""

    @pytest.mark.asyncio
    async def test_semantic_search_pushes_tag_and_time_filters(self):
        storage = _make_storage()
        storage._run_search.return_value = [
            _make_hit("h1", "python async notes", 0.82, tags=["python", "async"])
        ]

        result = await storage.search_memories(
            query="python",
            tags=["python", "async"],
            tag_match="all",
            after="2024-01-01",
            before="2024-12-31",
            limit=5,
            include_debug=True,
        )

        assert result["total"] == 1
        storage._run_search.assert_awaited_once()
        _, filter_expr, fetch_n = storage._run_search.await_args.args
        assert 'tags like "%,python,%"' in filter_expr
        assert 'tags like "%,async,%"' in filter_expr
        assert "created_at >=" in filter_expr
        assert "created_at <=" in filter_expr
        assert fetch_n == 15
        assert result["debug"]["backend"] == "milvus"

    @pytest.mark.asyncio
    async def test_tag_only_search_uses_native_query_path(self):
        storage = _make_storage()
        memory = Memory(
            content="release checklist",
            content_hash="h1",
            tags=["release"],
            memory_type="note",
        )
        storage._query_memories.return_value = [memory]

        result = await storage.search_memories(
            query=None,
            tags=["release"],
            limit=3,
        )

        assert result["total"] == 1
        storage._query_memories.assert_awaited_once()
        kwargs = storage._query_memories.await_args.kwargs
        assert kwargs["filter_expr"] == 'tags like "%,release,%"'
        assert kwargs["limit"] == 9
        assert result["memories"][0]["content_hash"] == "h1"

    @pytest.mark.asyncio
    async def test_quality_boost_reranks_semantic_results(self):
        storage = _make_storage()
        storage._run_search.return_value = [
            _make_hit("semantic", "close vector match", 0.9, metadata={"quality_score": 0.1}),
            _make_hit("quality", "better maintained note", 0.6, metadata={"quality_score": 1.0}),
        ]

        result = await storage.search_memories(
            query="maintenance",
            quality_boost=0.8,
            limit=2,
        )

        assert [m["content_hash"] for m in result["memories"]] == ["quality", "semantic"]
        assert result["memories"][0]["debug_info"]["reranked"] is True
        assert result["memories"][0]["similarity_score"] > result["memories"][1]["similarity_score"]

    @pytest.mark.asyncio
    async def test_exact_search_filters_superseded_memories(self):
        storage = _make_storage()
        kept = Memory(content="api endpoint", content_hash="kept", tags=["api"])
        superseded = Memory(
            content="api endpoint old",
            content_hash="old",
            tags=["api"],
            metadata={"superseded_by": "kept"},
        )
        storage.get_by_exact_content = AsyncMock(return_value=[superseded, kept])

        result = await storage.search_memories(
            query="api endpoint",
            mode="exact",
            limit=5,
        )

        assert result["total"] == 1
        assert result["memories"][0]["content_hash"] == "kept"

    @pytest.mark.asyncio
    async def test_hybrid_mode_uses_milvus_hybrid_search_when_available(self):
        storage = _make_storage()
        storage._has_bm25 = True
        storage._run_hybrid_search.return_value = [
            _make_hit("h1", "bm25 and vector match", 0.77)
        ]

        result = await storage.search_memories(
            query="bm25 vector",
            mode="hybrid",
            limit=1,
        )

        assert result["total"] == 1
        storage._run_hybrid_search.assert_awaited_once()
        storage._run_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_quality_boost_returns_error(self):
        storage = _make_storage()

        result = await storage.search_memories(query="x", quality_boost=1.5)

        assert result["total"] == 0
        assert "Invalid quality_boost" in result["error"]
        storage._run_search.assert_not_awaited()
