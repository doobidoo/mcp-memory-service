"""Tests for entity extraction and graph entity storage."""

import pytest
import tempfile
import os

from mcp_memory_service.reasoning.entities import EntityExtractor, Entity
from mcp_memory_service.storage.graph import GraphStorage


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

    def test_extract_camelcase(self):
        entities = self.extractor.extract_entities("The UserService handles authentication via AuthManager")
        names = [e.name for e in entities if e.entity_type == 'service']
        assert 'UserService' in names
        assert 'AuthManager' in names

    def test_extract_allcaps(self):
        entities = self.extractor.extract_entities("Set REDIS_HOST and DATABASE_URL in config")
        names = [e.name for e in entities if e.entity_type == 'project']
        assert 'REDIS_HOST' in names
        assert 'DATABASE_URL' in names

    def test_allcaps_excludes_common(self):
        entities = self.extractor.extract_entities("Use HTTP API with JSON format")
        names = [e.name for e in entities if e.entity_type == 'project']
        assert 'HTTP' not in names
        assert 'API' not in names
        assert 'JSON' not in names

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
