"""Tests for entity extraction and graph entity storage."""

import pytest
import tempfile
import os

import mcp_memory_service.config as cfg
from mcp_memory_service.reasoning.entities import EntityExtractor, Entity
from mcp_memory_service.server.handlers import graph as graph_handlers
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


class TestEntityExtractor:
    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extract_mentions(self):
        entities = self.extractor.extract_entities("Talked to @john.doe and @alice about the project")
        names = [e.name for e in entities if e.entity_type == 'person']
        assert 'john.doe' in names
        assert 'alice' in names

    def test_extract_hashtags(self):
        entities = self.extractor.extract_entities("Working on #backend and #api-design")
        names = [e.name for e in entities if e.entity_type == 'tag']
        assert 'backend' in names
        assert 'api-design' in names

    def test_extract_urls(self):
        entities = self.extractor.extract_entities("See https://github.com/org/repo for details")
        urls = [e.name for e in entities if e.entity_type == 'url']
        assert any('github.com' in u for u in urls)

    def test_camelcase_no_longer_extracted(self):
        """CamelCase patterns removed (too noisy for free-form text)."""
        entities = self.extractor.extract_entities("The UserService handles authentication via AuthManager")
        names = [e.name for e in entities if e.entity_type == 'service']
        assert len(names) == 0

    def test_allcaps_no_longer_extracted(self):
        """ALLCAPS patterns removed (too noisy for free-form text)."""
        entities = self.extractor.extract_entities("Set REDIS_HOST and DATABASE_URL in config")
        names = [e.name for e in entities if e.entity_type == 'project']
        assert len(names) == 0

    def test_extract_paths(self):
        entities = self.extractor.extract_entities("Edit /etc/nginx/nginx.conf and src/main.py")
        names = [e.name for e in entities if e.entity_type == 'file']
        assert any('nginx.conf' in n for n in names)

    def test_metadata_tags(self):
        entities = self.extractor.extract_entities("some content", {"tags": ["python", "async"]})
        names = [e.name for e in entities if e.entity_type == 'tag' and e.source == 'metadata']
        assert 'python' in names
        assert 'async' in names

    def test_metadata_tags_string(self):
        entities = self.extractor.extract_entities("content", {"tags": "redis,docker"})
        names = [e.name for e in entities if e.entity_type == 'tag']
        assert 'redis' in names
        assert 'docker' in names

    def test_deduplication(self):
        entities = self.extractor.extract_entities("@bob and @Bob talked", {"tags": ["bob"]})
        bob_entities = [e for e in entities if e.name.lower() == 'bob']
        # Should deduplicate by (name.lower(), type) — but person vs tag are different types
        person_bobs = [e for e in bob_entities if e.entity_type == 'person']
        assert len(person_bobs) == 1



class TestTagEntityFilter:
    """MCP_ENTITY_TAG_INCLUDE / MCP_ENTITY_TAG_EXCLUDE filter the tag branch only (#1704)."""

    TAGS = ["task:1704", "Task:42", "bugfix", "project:memory"]

    @staticmethod
    def _entities(monkeypatch, include="", exclude="", content="some content"):
        monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", include, raising=False)
        monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", exclude, raising=False)
        return EntityExtractor().extract_entities(content, {"tags": TestTagEntityFilter.TAGS})

    def _tag_names(self, monkeypatch, **kwargs):
        return {e.name for e in self._entities(monkeypatch, **kwargs) if e.source == 'metadata'}

    def test_unset_keeps_every_tag(self, monkeypatch):
        assert self._tag_names(monkeypatch) == set(self.TAGS)

    def test_include_keeps_only_matching_tags(self, monkeypatch):
        assert self._tag_names(monkeypatch, include="task:*") == {"task:1704", "Task:42"}

    def test_exclude_drops_matching_tags(self, monkeypatch):
        assert self._tag_names(monkeypatch, exclude="BUGFIX, project:*") == {"task:1704", "Task:42"}

    def test_include_then_exclude(self, monkeypatch):
        names = self._tag_names(monkeypatch, include="task:*,bugfix", exclude="task:42")
        assert names == {"task:1704", "bugfix"}

    def test_content_entities_unaffected(self, monkeypatch):
        entities = self._entities(monkeypatch, include="task:*", content="Fixed #bugfix with @alice")
        assert {(e.name, e.source) for e in entities if e.source == 'content'} == {
            ("bugfix", "content"), ("alice", "content"),
        }


@pytest.mark.asyncio
async def test_excluded_tag_creates_no_shares_entity_edge(temp_db_path, monkeypatch):
    """Through real sqlite-vec storage: an excluded tag links nothing (#1704)."""
    db_path = os.path.join(temp_db_path, "test.db")
    monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "", raising=False)
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "bugfix", raising=False)

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    graph = GraphStorage(db_path)

    async def _graph():
        return graph

    monkeypatch.setattr(graph_handlers, "get_graph_storage", _graph)
    try:
        svc = MemoryService(storage)
        a = await svc.store_memory("Fixed the login redirect loop", tags=["bugfix", "task:1704"])
        b = await svc.store_memory("Patched a race in the upload queue", tags=["bugfix"])
        c = await svc.store_memory("Wrote the release notes draft", tags=["task:1704"])
        assert a["success"] and b["success"] and c["success"], (a, b, c)
        hash_a, hash_b, hash_c = (r["memory"]["content_hash"] for r in (a, b, c))

        assert await graph.get_association(hash_a, hash_c) is not None
        assert await graph.get_association(hash_a, hash_b) is None
        assert "bugfix" not in await graph.get_entities_for_memory(hash_b)
    finally:
        await graph.close()
        await storage.close()

class TestGraphEntityStorage:
    @pytest.fixture
    async def graph(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        g = GraphStorage(path)
        conn = await g._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_graph (
                source_hash TEXT,
                target_hash TEXT,
                similarity REAL,
                connection_types TEXT,
                metadata TEXT,
                created_at REAL,
                relationship_type TEXT DEFAULT 'related',
                PRIMARY KEY (source_hash, target_hash)
            )
        """)
        conn.commit()
        yield g
        await g.close()
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_store_entity_link(self, graph):
        result = await graph.store_entity_link("hash123", "UserService", "service")
        assert result is True

    @pytest.mark.asyncio
    async def test_find_memories_by_entity(self, graph):
        await graph.store_entity_link("mem1", "redis", "service")
        await graph.store_entity_link("mem2", "redis", "service")
        await graph.store_entity_link("mem3", "postgres", "service")

        results = await graph.find_memories_by_entity("redis")
        assert set(results) == {"mem1", "mem2"}

    @pytest.mark.asyncio
    async def test_find_memories_by_entity_limit(self, graph):
        for i in range(5):
            await graph.store_entity_link(f"mem{i}", "entity_x", "tag")

        results = await graph.find_memories_by_entity("entity_x", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_entity_profile(self, graph):
        await graph.store_entity_link("mem1", "MyService", "service")
        await graph.store_entity_link("mem2", "MyService", "service")

        profile = await graph.get_entity_profile("MyService")
        assert profile["entity_name"] == "MyService"
        assert profile["memory_count"] == 2
        assert "service" in profile["entity_types"]
        assert profile["last_activity"] is not None

    @pytest.mark.asyncio
    async def test_get_entity_profile_empty(self, graph):
        profile = await graph.get_entity_profile("nonexistent")
        assert profile == {}
