"""Tests for multi-strategy retrieval with RRF fusion (RFC #1008 §6)."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestRRFFusion:
    def test_import(self):
        from mcp_memory_service.reasoning.multi_strategy import rrf_fuse
        assert rrf_fuse is not None

    def test_single_strategy(self):
        from mcp_memory_service.reasoning.multi_strategy import rrf_fuse
        result = rrf_fuse([["a", "b", "c"]], k=60)
        assert result[0] == "a"

    def test_two_strategies_consensus(self):
        from mcp_memory_service.reasoning.multi_strategy import rrf_fuse
        result = rrf_fuse([["a", "b", "c", "d"], ["b", "a", "d", "c"]], k=60)
        assert result[0] in ("a", "b")
        assert result[-1] in ("c", "d")

    def test_unique_items_from_different_strategies(self):
        from mcp_memory_service.reasoning.multi_strategy import rrf_fuse
        result = rrf_fuse([["a", "b"], ["c", "d"]], k=60)
        assert set(result) == {"a", "b", "c", "d"}

    def test_limit_results(self):
        from mcp_memory_service.reasoning.multi_strategy import rrf_fuse
        result = rrf_fuse([["a", "b", "c", "d", "e"]], k=60, limit=3)
        assert len(result) == 3


class TestMultiStrategySearch:
    def test_import_search(self):
        from mcp_memory_service.reasoning.multi_strategy import multi_strategy_search
        assert multi_strategy_search is not None

    @pytest.mark.asyncio
    async def test_returns_fused_results(self):
        from mcp_memory_service.reasoning.multi_strategy import multi_strategy_search
        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value={
            "memories": [
                {"content_hash": "a", "content": "memory a", "similarity_score": 0.9},
                {"content_hash": "b", "content": "memory b", "similarity_score": 0.8},
                {"content_hash": "c", "content": "memory c", "similarity_score": 0.7},
            ]
        })
        storage.search_by_tag_chronological = AsyncMock(return_value=[
            MagicMock(content_hash="b", content="memory b", tags=["important"]),
            MagicMock(content_hash="d", content="memory d", tags=["important"]),
        ])
        result = await multi_strategy_search(
            storage, query="test query", limit=5,
            strategies=["semantic", "tag"], tags=["important"],
        )
        assert "memories" in result
        assert result["mode"] == "multi_strategy"
        hashes = [m["content_hash"] for m in result["memories"]]
        assert "b" in hashes

    @pytest.mark.asyncio
    async def test_graceful_strategy_failure(self):
        from mcp_memory_service.reasoning.multi_strategy import multi_strategy_search
        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value={
            "memories": [{"content_hash": "a", "content": "ok", "similarity_score": 0.9}]
        })
        storage.search_by_tag_chronological = AsyncMock(side_effect=Exception("broken"))
        result = await multi_strategy_search(
            storage, query="test", limit=5, strategies=["semantic", "tag"],
        )
        assert len(result["memories"]) >= 1
