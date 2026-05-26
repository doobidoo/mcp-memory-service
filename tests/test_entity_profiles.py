"""Tests for RFC #732 Phase 2: Entity Profiles."""

import os
import pytest
import tempfile

from mcp_memory_service.reasoning.entities import EntityExtractor
from mcp_memory_service.storage.graph import GraphStorage


@pytest.fixture
def graph_db():
    """Create a temporary GraphStorage instance."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_graph.db")
    gs = GraphStorage(db_path)
    conn = gs._create_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL DEFAULT 0.0,
            connection_types TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at REAL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash, relationship_type)
        )
    """)
    conn.commit()
    conn.close()
    gs._connection = None
    yield gs
    # Cleanup temp directory
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


class TestCustomTermsExtraction:
    def test_custom_terms_matched(self, monkeypatch):
        """Custom terms from env var are extracted."""
        monkeypatch.setenv("MCP_ENTITY_CUSTOM_TERMS", "kiro,roma-connect")
        # Reload config value
        import mcp_memory_service.config as cfg
        monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "kiro,roma-connect")

        extractor = EntityExtractor()
        entities = extractor.extract_entities("We deployed kiro to production with roma-connect")

        names = [e.name for e in entities]
        assert "kiro" in names
        assert "roma-connect" in names

    def test_custom_terms_case_insensitive(self, monkeypatch):
        """Custom terms match case-insensitively."""
        import mcp_memory_service.config as cfg
        monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "Kiro")

        extractor = EntityExtractor()
        entities = extractor.extract_entities("KIRO is running fine")

        names = [e.name for e in entities]
        assert "Kiro" in names

    def test_no_custom_terms(self, monkeypatch):
        """Empty custom terms config produces no custom entities."""
        import mcp_memory_service.config as cfg
        monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "")

        extractor = EntityExtractor()
        entities = extractor.extract_entities("just some text")

        custom = [e for e in entities if e.source == 'config']
        assert custom == []


class TestFrequencyExtraction:
    def test_frequent_terms_above_threshold(self):
        """Terms in >= min_count memories are returned."""
        memories = [f"python is great for backend {i}" for i in range(6)]
        result = EntityExtractor.extract_frequent_terms(memories, min_count=5)
        assert "python" in result
        assert "backend" in result
        assert "great" in result

    def test_below_threshold_excluded(self):
        """Terms below min_count are not returned."""
        memories = ["python rocks"] + ["java rocks"] * 5
        result = EntityExtractor.extract_frequent_terms(memories, min_count=5)
        assert "python" not in result
        assert "rocks" in result
        assert "java" in result

    def test_stopwords_excluded(self):
        """Common stopwords are excluded."""
        memories = ["the quick fox and the lazy dog"] * 10
        result = EntityExtractor.extract_frequent_terms(memories, min_count=5)
        assert "the" not in result
        assert "and" not in result
        assert "quick" in result

    def test_short_tokens_excluded(self):
        """Tokens shorter than 3 chars are excluded."""
        memories = ["go is ok but python is better"] * 10
        result = EntityExtractor.extract_frequent_terms(memories, min_count=5)
        assert "go" not in result
        assert "is" not in result
        assert "python" in result


class TestListEntities:
    @pytest.mark.asyncio
    async def test_list_entities_sorted_by_count(self, graph_db):
        """list_entities returns entities sorted by count descending."""
        await graph_db.store_entity_link("h1", "python", "tag")
        await graph_db.store_entity_link("h2", "python", "tag")
        await graph_db.store_entity_link("h3", "python", "tag")
        await graph_db.store_entity_link("h1", "java", "tag")

        entities = await graph_db.list_entities(limit=10)

        assert len(entities) == 2
        assert entities[0]["entity_name"] == "python"
        assert entities[0]["count"] == 3
        assert entities[1]["entity_name"] == "java"
        assert entities[1]["count"] == 1

    @pytest.mark.asyncio
    async def test_list_entities_respects_limit(self, graph_db):
        """list_entities respects the limit parameter."""
        for i in range(5):
            await graph_db.store_entity_link(f"h{i}", f"entity_{i}", "tag")

        entities = await graph_db.list_entities(limit=3)
        assert len(entities) == 3

    @pytest.mark.asyncio
    async def test_list_entities_empty(self, graph_db):
        """list_entities returns empty list when no entities exist."""
        entities = await graph_db.list_entities()
        assert entities == []


class TestEntityProfile:
    @pytest.mark.asyncio
    async def test_entity_profile_with_memories(self, graph_db):
        """entity_profile returns enriched profile with memory_hashes."""
        await graph_db.store_entity_link("hash_a", "docker", "tag")
        await graph_db.store_entity_link("hash_b", "docker", "service")

        profile = await graph_db.get_entity_profile("docker")

        assert profile["entity_name"] == "docker"
        assert profile["memory_count"] == 2
        assert set(profile["entity_types"]) == {"tag", "service"}
        assert profile["last_activity"] is not None

        hashes = await graph_db.find_memories_by_entity("docker")
        assert set(hashes) == {"hash_a", "hash_b"}

    @pytest.mark.asyncio
    async def test_entity_profile_not_found(self, graph_db):
        """entity_profile returns empty dict for unknown entity."""
        profile = await graph_db.get_entity_profile("nonexistent")
        assert profile == {}
