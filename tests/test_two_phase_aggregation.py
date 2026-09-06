"""Tests for two-phase aggregation logic (PR #78)."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    g.common_neighbors = AsyncMock(return_value=[])
    g.get_entities_for_memory = AsyncMock(return_value=[])
    g.find_connected = AsyncMock(return_value=[])
    g.get_entity_profile = AsyncMock(return_value={"count": 5, "memory_count": 5})
    g.list_entities = AsyncMock(return_value=[
        {"entity_name": "python", "count": 10},
        {"entity_name": "testing", "count": 5},
    ])
    g.find_memories_by_entity = AsyncMock(return_value=[])
    return g


@pytest.fixture
def mock_storage():
    s = AsyncMock()
    s.graph = AsyncMock()
    mem = MagicMock()
    mem.content_hash = "hash1"
    mem.content = "Test memory content"
    mem.quality_score = 0.8
    s.get_memories_batch = AsyncMock(return_value=[mem])
    return s


class TestDiscoverRelatedEntities:
    @pytest.mark.asyncio
    async def test_empty_hashes(self):
        from mcp_memory_service.server.handlers.graph import _discover_related_entities
        graph = AsyncMock()
        graph.common_neighbors = AsyncMock(return_value=[])
        result = await _discover_related_entities(graph, [], max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_common_neighbors(self):
        from mcp_memory_service.server.handlers.graph import _discover_related_entities
        graph = AsyncMock()
        graph.common_neighbors = AsyncMock(return_value=[])
        result = await _discover_related_entities(graph, ["hash1", "hash2"], max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_resolves_memory_hash_to_entities(self):
        from mcp_memory_service.server.handlers.graph import _discover_related_entities
        graph = AsyncMock()
        graph.common_neighbors = AsyncMock(return_value=[("cand_hash_1", 3, 5)])
        graph.get_entities_for_memory = AsyncMock(return_value=["Python", "FastAPI"])
        result = await _discover_related_entities(graph, ["hash1"], max_hops=2)
        assert len(result) == 2
        assert result[0]["name"] == "Python"
        assert result[0]["entity_id"] == "python"
        assert result[0]["shared_count"] == 3

    @pytest.mark.asyncio
    async def test_no_entity_links_skips(self):
        from mcp_memory_service.server.handlers.graph import _discover_related_entities
        graph = AsyncMock()
        graph.common_neighbors = AsyncMock(return_value=[("cand_hash_1", 3, 5)])
        graph.get_entities_for_memory = AsyncMock(return_value=[])
        result = await _discover_related_entities(graph, ["hash1"], max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_dedup_by_entity_id(self):
        from mcp_memory_service.server.handlers.graph import _discover_related_entities
        graph = AsyncMock()
        graph.common_neighbors = AsyncMock(side_effect=[
            [("h1", 3, 5)],
            [("h2", 5, 4)],
        ])
        graph.get_entities_for_memory = AsyncMock(return_value=["Python"])
        result = await _discover_related_entities(graph, ["a", "b"], max_hops=2)
        assert len(result) == 1
        assert result[0]["shared_count"] == 5  # max of 3 and 5


class TestBuildKnowledgeMap:
    @pytest.mark.asyncio
    async def test_empty_entities(self):
        from mcp_memory_service.server.handlers.graph import _build_knowledge_map
        graph = AsyncMock()
        result = await _build_knowledge_map(graph, [], chunk_pool=[], chunks_per_entity=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_normal_entities(self):
        from mcp_memory_service.server.handlers.graph import _build_knowledge_map
        graph = AsyncMock()
        graph.find_memories_by_entity = AsyncMock(return_value=["hash1"])
        graph.get_entity_profile = AsyncMock(return_value={"count": 10, "entity_types": ["language"]})
        entities = [{"entity_name": "python", "count": 10}]
        chunks = [{"hash": "hash1", "content": "about python", "relevance": 0.9}]
        result = await _build_knowledge_map(graph, entities, chunk_pool=chunks, chunks_per_entity=3)
        assert len(result) == 1
        assert result[0]["entity_id"] == "python"
        assert result[0]["name"] == "python"

    @pytest.mark.asyncio
    async def test_unmatched_entity_does_not_inherit_query_chunks(self):
        from mcp_memory_service.server.handlers.graph import _build_knowledge_map

        graph = AsyncMock()
        graph.find_memories_by_entity = AsyncMock(return_value=["other-hash"])
        graph.get_entity_profile = AsyncMock(return_value={"count": 2})
        entities = [{"entity_name": "unrelated"}]
        chunks = [{"hash": "query-hash", "content": "query result", "relevance": 0.9}]

        result = await _build_knowledge_map(graph, entities, chunks, chunks_per_entity=3)

        assert result[0]["top_chunks"] == []
        assert result[0]["summary"] == ""
        assert result[0]["relation_count"] == 2


class TestSelectExploreEntities:
    @pytest.mark.asyncio
    async def test_uses_entities_linked_to_highest_relevance_chunks(self):
        from mcp_memory_service.server.handlers.graph import _select_explore_entities

        graph = AsyncMock()
        graph.get_entities_for_memory = AsyncMock(
            side_effect=lambda memory_hash: {
                "high": ["Python", "Testing"],
                "low": ["Testing", "python", "SQLite"],
            }[memory_hash]
        )
        graph.list_entities = AsyncMock(return_value=[{"entity_name": "Global"}])
        chunks = [
            {"hash": "low", "relevance": 0.2},
            {"hash": "high", "relevance": 0.9},
        ]

        result = await _select_explore_entities(graph, chunks, max_entities=4)

        assert result == [
            {"entity_name": "Python"},
            {"entity_name": "Testing"},
            {"entity_name": "SQLite"},
        ]
        graph.list_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_global_entities_when_chunks_have_no_links(self):
        from mcp_memory_service.server.handlers.graph import _select_explore_entities

        graph = AsyncMock()
        graph.get_entities_for_memory = AsyncMock(return_value=[])
        graph.list_entities = AsyncMock(return_value=[{"entity_name": "Global", "count": 3}])

        result = await _select_explore_entities(
            graph, [{"hash": "query-hash", "relevance": 0.9}], max_entities=1
        )

        assert result == [{"entity_name": "Global", "count": 3}]
        graph.list_entities.assert_awaited_once_with(limit=1)


class TestMemoryExploreEntitySelection:
    @pytest.mark.asyncio
    async def test_handler_uses_entities_linked_to_retrieved_chunks(self, monkeypatch):
        from mcp_memory_service.server.handlers import graph as graph_handlers

        candidate = MagicMock()
        candidate.memory.content_hash = "query-hash"
        candidate.memory.content = "A query-matching memory."
        candidate.relevance_score = 0.9
        server = MagicMock()
        server.storage.retrieve = AsyncMock(return_value=[candidate])

        graph = AsyncMock()
        graph.get_entities_for_memory = AsyncMock(return_value=["Relevant"])
        graph.list_entities = AsyncMock(return_value=[{"entity_name": "Global"}])
        graph.find_memories_by_entity = AsyncMock(return_value=["query-hash"])
        graph.get_entity_profile = AsyncMock(
            return_value={"count": 1, "entity_types": ["topic"]}
        )
        monkeypatch.setattr(
            graph_handlers, "get_graph_storage", AsyncMock(return_value=graph)
        )

        response = await graph_handlers.handle_memory_explore(
            server, {"query": "relevant query", "max_entities": 5}
        )
        payload = json.loads(response[0].text)

        assert [entity["name"] for entity in payload["entities"]] == ["Relevant"]
        assert payload["entities"][0]["top_chunks"][0]["hash"] == "query-hash"
        graph.list_entities.assert_not_awaited()


class TestHydrateChunks:
    @pytest.mark.asyncio
    async def test_empty_hashes(self):
        from mcp_memory_service.server.handlers.graph import _hydrate_chunks
        storage = AsyncMock()
        storage.get_by_hash = AsyncMock(return_value=None)
        result = await _hydrate_chunks(storage, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_null_content_handled(self):
        from mcp_memory_service.server.handlers.graph import _hydrate_chunks
        mem = MagicMock()
        mem.content = None
        mem.quality_score = None
        storage = AsyncMock()
        storage.get_by_hash = AsyncMock(return_value=mem)
        result = await _hydrate_chunks(storage, ["h1"])
        assert len(result) == 1
        assert result[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_zero_quality_preserved(self):
        from mcp_memory_service.server.handlers.graph import _hydrate_chunks
        mem = MagicMock()
        mem.content = "test"
        mem.quality_score = 0.0
        storage = AsyncMock()
        storage.get_by_hash = AsyncMock(return_value=mem)
        result = await _hydrate_chunks(storage, ["h1"])
        assert len(result) == 1
        assert result[0]["relevance"] == 0.0
