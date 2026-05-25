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


# ---------------------------------------------------------------------------
# Phase 1b tests — configurable terms + frequency-based batch extraction
# ---------------------------------------------------------------------------

class TestConfiguredTerms:
    """EntityExtractor with user-defined term list."""

    def test_configured_term_found(self):
        extractor = EntityExtractor(configured_terms={"redis": "service"})
        entities = extractor.extract_entities("We use Redis for caching")
        names = [e.name for e in entities if e.entity_type == "service"]
        assert "redis" in names

    def test_configured_term_case_insensitive(self):
        extractor = EntityExtractor(configured_terms={"PostgreSQL": "service"})
        entities = extractor.extract_entities("migrated to postgresql last week")
        names = [e.name.lower() for e in entities if e.entity_type == "service"]
        assert "postgresql" in names

    def test_configured_term_source_is_configured(self):
        extractor = EntityExtractor(configured_terms={"kafka": "service"})
        entities = extractor.extract_entities("Kafka consumer group reset")
        matches = [e for e in entities if e.name.lower() == "kafka"]
        assert len(matches) == 1
        assert matches[0].source == "configured"

    def test_configured_term_not_in_content(self):
        extractor = EntityExtractor(configured_terms={"redis": "service"})
        entities = extractor.extract_entities("No database mentions here")
        types = [e.entity_type for e in entities]
        assert "service" not in types

    def test_multiple_configured_terms(self):
        extractor = EntityExtractor(configured_terms={
            "redis": "service",
            "postgres": "service",
            "hanuman": "agent",
        })
        entities = extractor.extract_entities("Redis query failed; postgres fallback; agent hanuman notified")
        names = {e.name.lower() for e in entities}
        assert "redis" in names
        assert "postgres" in names
        assert "hanuman" in names

    def test_configured_term_deduplication(self):
        """Term appearing multiple times should produce one entity."""
        extractor = EntityExtractor(configured_terms={"redis": "service"})
        entities = extractor.extract_entities("redis redis redis")
        service_entities = [e for e in entities if e.entity_type == "service"]
        assert len(service_entities) == 1

    def test_configured_term_does_not_suppress_regex(self):
        """Regex extraction still runs alongside configured terms."""
        extractor = EntityExtractor(configured_terms={"redis": "service"})
        entities = extractor.extract_entities("@alice uses redis and #caching")
        types = {e.entity_type for e in entities}
        assert "person" in types
        assert "service" in types
        assert "tag" in types

    def test_empty_configured_terms(self):
        extractor = EntityExtractor(configured_terms={})
        entities = extractor.extract_entities("some content without special terms")
        # Should still work — just no configured entities
        configured = [e for e in entities if e.source == "configured"]
        assert configured == []


class TestFrequencyExtraction:
    """EntityExtractor.extract_entities_batch with frequency_threshold."""

    def test_batch_returns_per_index_results(self):
        extractor = EntityExtractor()
        results = extractor.extract_entities_batch(["@alice here", "@bob there"])
        assert "0" in results and "1" in results

    def test_batch_empty_input(self):
        extractor = EntityExtractor(frequency_threshold=2)
        results = extractor.extract_entities_batch([])
        assert results == {}

    def test_frequency_threshold_zero_disables_promotion(self):
        """With threshold=0, no frequency entities should appear."""
        contents = ["word1 word2 word3"] * 5
        extractor = EntityExtractor(frequency_threshold=0)
        results = extractor.extract_entities_batch(contents)
        for entities in results.values():
            assert all(e.source != "frequency" for e in entities)

    def test_frequent_token_promoted(self):
        """Token in >= threshold memories is promoted."""
        contents = [
            "postgres is the primary database",
            "postgres connection pool exhausted",
            "migrated postgres to version 15",
        ]
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        freq_names = {
            e.name for entities in results.values()
            for e in entities if e.source == "frequency"
        }
        assert "postgres" in freq_names

    def test_infrequent_token_not_promoted(self):
        """Token below threshold is not promoted."""
        contents = [
            "postgres is great",
            "postgres rocks",
            "mysql is also here",  # only once
        ]
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        freq_names = {
            e.name for entities in results.values()
            for e in entities if e.source == "frequency"
        }
        assert "mysql" not in freq_names

    def test_frequent_entity_source_is_frequency(self):
        contents = ["redis cache hit"] * 4
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        redis_entities = [
            e for entities in results.values()
            for e in entities if e.name == "redis"
        ]
        assert all(e.source == "frequency" for e in redis_entities)

    def test_frequent_entity_type_is_frequent(self):
        contents = ["kafka consumer lag"] * 4
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        kafka_entities = [
            e for entities in results.values()
            for e in entities if e.name == "kafka"
        ]
        assert all(e.entity_type == "frequent" for e in kafka_entities)

    def test_min_token_length_filters_short_tokens(self):
        """Tokens shorter than min_token_length are not promoted."""
        contents = ["or is and to be"] * 5
        extractor = EntityExtractor(frequency_threshold=3, min_token_length=4)
        results = extractor.extract_entities_batch(contents)
        freq_names = {
            e.name for entities in results.values()
            for e in entities if e.source == "frequency"
        }
        # All tokens above are 2-3 chars — none should be promoted
        assert len(freq_names) == 0

    def test_frequency_counted_per_memory_not_raw_count(self):
        """A token appearing 100x in one memory counts as 1, not 100."""
        contents = [
            "redis " * 100,  # 100 occurrences in one memory
            "postgres is here",  # redis absent
            "other content here",  # redis absent
        ]
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        freq_names = {
            e.name for entities in results.values()
            for e in entities if e.source == "frequency"
        }
        # redis only appears in 1 distinct memory → below threshold of 3
        assert "redis" not in freq_names

    def test_frequency_does_not_duplicate_regex_entities(self):
        """Frequent token already extracted by regex should not be duplicated."""
        contents = ["#redis cache config"] * 4
        extractor = EntityExtractor(frequency_threshold=3)
        results = extractor.extract_entities_batch(contents)
        for entities in results.values():
            redis_entities = [e for e in entities if e.name.lower() == "redis"]
            assert len(redis_entities) <= 1

    def test_batch_with_metadata(self):
        meta_list = [{"tags": ["python"]}, {"tags": ["go"]}]
        extractor = EntityExtractor()
        results = extractor.extract_entities_batch(["content a", "content b"], meta_list)
        py_tags = [e for e in results["0"] if e.name == "python" and e.source == "metadata"]
        assert len(py_tags) == 1

    def test_configured_plus_frequency_combined(self):
        """Configured terms and frequency extraction work together."""
        contents = [
            "postgres upgrade scheduled",
            "postgres replica lag",
            "postgres autovacuum tuning",
        ]
        extractor = EntityExtractor(
            configured_terms={"postgres": "service"},
            frequency_threshold=3,
        )
        results = extractor.extract_entities_batch(contents)
        for entities in results.values():
            sources = {e.source for e in entities if e.name.lower() == "postgres"}
            # Should appear as 'configured' (from term list), not duplicated as 'frequency'
            assert "configured" in sources
            configured_count = sum(1 for e in entities if e.name.lower() == "postgres")
            assert configured_count == 1
