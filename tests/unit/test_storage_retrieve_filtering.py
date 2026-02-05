"""
Unit tests for storage backend retrieve() method with filtering.

These tests verify that storage backends support database-level filtering
for tags, memory_type, and min_similarity parameters in the retrieve() method.

Current Status: EXPECTED TO FAIL
- Storage backends don't yet accept filter parameters
- Once implemented, these tests should pass
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestSqliteVecFiltering:
    """Test SQLite-vec backend filtering in retrieve() method."""

    @pytest.mark.asyncio
    async def test_retrieve_with_tag_filter(self, temp_db_path):
        """Test that retrieve() accepts and applies tag filtering at database level."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memories with different tags
        content1 = "Python programming guide"
        mem1 = Memory(
            content=content1, content_hash=generate_content_hash(content1), tags=["python", "coding"], memory_type="note"
        )
        content2 = "Python debugging tips"
        mem2 = Memory(
            content=content2, content_hash=generate_content_hash(content2), tags=["python", "debug"], memory_type="note"
        )
        content3 = "Java coding standards"
        mem3 = Memory(
            content=content3, content_hash=generate_content_hash(content3), tags=["java", "coding"], memory_type="note"
        )

        await storage.store(mem1)
        await storage.store(mem2)
        await storage.store(mem3)

        # Test: retrieve() should accept tags parameter
        # EXPECTED TO FAIL: retrieve() doesn't accept tags parameter yet
        results = await storage.retrieve("programming", n_results=10, tags=["python"])

        # Verify database-level filtering
        assert len(results) == 2, "Should return only python-tagged memories"
        for result in results:
            assert "python" in result.memory.tags, "All results should have python tag"

    @pytest.mark.asyncio
    async def test_retrieve_with_memory_type_filter(self, temp_db_path):
        """Test that retrieve() accepts and applies memory_type filtering."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memories with different types
        content_note1 = "Note 1"
        await storage.store(
            Memory(content=content_note1, content_hash=generate_content_hash(content_note1), tags=[], memory_type="note")
        )
        content_rem1 = "Reminder 1"
        await storage.store(
            Memory(content=content_rem1, content_hash=generate_content_hash(content_rem1), tags=[], memory_type="reminder")
        )
        content_note2 = "Note 2"
        await storage.store(
            Memory(content=content_note2, content_hash=generate_content_hash(content_note2), tags=[], memory_type="note")
        )

        # EXPECTED TO FAIL: retrieve() doesn't accept memory_type parameter yet
        results = await storage.retrieve("1", n_results=10, memory_type="note")

        assert len(results) == 2, "Should return only note-type memories"
        for result in results:
            assert result.memory.memory_type == "note"

    @pytest.mark.asyncio
    async def test_retrieve_with_combined_filters(self, temp_db_path):
        """Test retrieve() with both tag and memory_type filters."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store 10 memories with various combinations
        for i in range(10):
            content = f"Memory {i}"
            await storage.store(
                Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["tag1"] if i < 5 else ["tag2"],
                    memory_type="note" if i % 2 == 0 else "reminder",
                )
            )

        # Filter for tag1 + note type (should get 3 memories: 0, 2, 4)
        # EXPECTED TO FAIL: retrieve() doesn't accept filter parameters yet
        results = await storage.retrieve(query="Memory", n_results=10, tags=["tag1"], memory_type="note")

        assert len(results) == 3, "Should return only tag1 + note memories"
        for result in results:
            assert "tag1" in result.memory.tags
            assert result.memory.memory_type == "note"

    @pytest.mark.asyncio
    async def test_retrieve_with_min_similarity_filter(self, temp_db_path):
        """Test retrieve() with minimum similarity threshold."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store similar and dissimilar memories
        content_prog = "Python programming language"
        await storage.store(Memory(content=content_prog, content_hash=generate_content_hash(content_prog), tags=[]))
        content_tutorial = "Python coding tutorial"
        await storage.store(Memory(content=content_tutorial, content_hash=generate_content_hash(content_tutorial), tags=[]))
        content_unrelated = "Completely unrelated topic"
        await storage.store(Memory(content=content_unrelated, content_hash=generate_content_hash(content_unrelated), tags=[]))

        # EXPECTED TO FAIL: retrieve() doesn't accept min_similarity parameter yet
        results = await storage.retrieve(query="Python programming", n_results=10, min_similarity=0.7)

        # Should only return highly similar results
        assert len(results) >= 1, "Should return at least one high-similarity result"
        for result in results:
            assert result.similarity_score >= 0.7, f"Similarity {result.similarity_score} below threshold"

    @pytest.mark.asyncio
    async def test_retrieve_filters_are_database_level_not_python(self, temp_db_path):
        """Verify filtering happens in SQL WHERE clause, not post-retrieval in Python."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store 100 memories - only 10 match filter
        for i in range(100):
            content = f"Test memory {i}"
            await storage.store(
                Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["target"] if i < 10 else ["other"],
                    memory_type="note",
                )
            )

        # Request 100 results with filter for "target" tag
        # EXPECTED TO FAIL: retrieve() doesn't accept tags parameter yet
        results = await storage.retrieve(query="Test", n_results=100, tags=["target"])

        # Should return exactly 10 memories (all that match filter)
        # If filtering was post-hoc in Python, would return up to 100 and filter down
        assert len(results) == 10, "Should return exactly 10 matching memories"
        for result in results:
            assert "target" in result.memory.tags


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
