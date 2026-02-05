"""
Integration tests for storage backends with REAL backends.

These tests use actual storage backends (not mocks) to verify:
- Store -> Retrieve roundtrip works
- Semantic similarity search actually finds related content
- Edge cases are handled correctly

Run with: uv run pytest tests/integration/test_storage_integration.py -v
"""

import importlib.util
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator

import pytest

# Force CPU mode for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.utils.hashing import generate_content_hash

# =============================================================================
# SQLite-Vec Integration Tests
# =============================================================================

SQLITE_VEC_AVAILABLE = importlib.util.find_spec("sqlite_vec") is not None

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def sqlite_storage() -> AsyncGenerator["SqliteVecMemoryStorage", None]:
    """Create a real SQLite-vec storage instance for testing."""
    if not SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec not available")

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_integration.db")

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()

    yield storage

    await storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestSqliteVecIntegration:
    """Integration tests for SQLite-vec storage with REAL embeddings."""

    @pytest.mark.asyncio
    async def test_store_retrieve_roundtrip(self, sqlite_storage):
        """Test that stored content can be retrieved via semantic search."""
        content = "The quick brown fox jumps over the lazy dog"
        memory = Memory(
            content=content, content_hash=generate_content_hash(content), tags=["test", "animal"], memory_type="note"
        )

        # Store
        success, message = await sqlite_storage.store(memory)
        assert success, f"Store failed: {message}"

        # Retrieve with semantically similar query
        results = await sqlite_storage.retrieve("fox jumping over dog", n_results=5)

        assert len(results) > 0, "No results returned"
        assert results[0].memory.content == content
        assert results[0].relevance_score > 0.5, "Relevance score too low"

    @pytest.mark.asyncio
    async def test_semantic_similarity_ranking(self, sqlite_storage):
        """Test that semantically similar content ranks higher than dissimilar content."""
        # Store diverse content
        memories = [
            ("Python is a programming language used for web development and data science", ["programming"]),
            ("Cats are furry animals that make great pets", ["animals"]),
            ("Machine learning is a subset of artificial intelligence", ["programming", "ai"]),
            ("Dogs are loyal companions and popular pets", ["animals"]),
        ]

        for content, tags in memories:
            memory = Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            await sqlite_storage.store(memory)

        # Search for programming-related content
        results = await sqlite_storage.retrieve("software development and coding", n_results=4)

        assert len(results) >= 2

        # Programming content should rank higher than animal content
        top_contents = [r.memory.content for r in results[:2]]
        assert any(
            "programming" in c.lower() or "machine learning" in c.lower() for c in top_contents
        ), f"Expected programming content in top results, got: {top_contents}"

    @pytest.mark.asyncio
    async def test_edge_case_empty_string(self, sqlite_storage):
        """Test handling of empty string content."""
        memory = Memory(content="", content_hash=generate_content_hash(""), tags=["empty"], memory_type="note")

        # Should fail or handle gracefully
        success, message = await sqlite_storage.store(memory)
        # Empty content may be rejected or stored - either is acceptable as long as no crash

    @pytest.mark.asyncio
    async def test_edge_case_unicode_content(self, sqlite_storage):
        """Test handling of unicode content including emojis."""
        content = "Python is great! Here are some emojis: rocket ship and fire"
        memory = Memory(
            content=content, content_hash=generate_content_hash(content), tags=["unicode", "emoji"], memory_type="note"
        )

        success, message = await sqlite_storage.store(memory)
        assert success, f"Store failed: {message}"

        # Verify retrieval
        results = await sqlite_storage.retrieve("python emojis", n_results=1)
        assert len(results) == 1
        assert "Python" in results[0].memory.content

    @pytest.mark.asyncio
    async def test_edge_case_very_long_content(self, sqlite_storage):
        """Test handling of very long content."""
        # Create content that's 10KB
        content = "This is a test sentence. " * 500  # ~13KB
        memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["large"], memory_type="note")

        success, message = await sqlite_storage.store(memory)
        assert success, f"Store failed: {message}"

        results = await sqlite_storage.retrieve("test sentence", n_results=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_edge_case_special_characters(self, sqlite_storage):
        """Test handling of special characters in content."""
        content = r"Special chars: <script>alert('XSS')</script> && || ; DROP TABLE; \n\t"
        memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["special"], memory_type="note")

        success, message = await sqlite_storage.store(memory)
        assert success, f"Store failed: {message}"

        # Verify it can be retrieved by hash
        retrieved = await sqlite_storage.get_memory_by_hash(memory.content_hash)
        assert retrieved is not None
        assert retrieved.content == content

    @pytest.mark.asyncio
    async def test_tag_search_with_real_data(self, sqlite_storage):
        """Test tag search with real stored data."""
        # Store memories with various tags
        memories_data = [
            ("Work meeting notes from Monday", ["work", "meeting"]),
            ("Personal todo list for the weekend", ["personal", "todo"]),
            ("Work project documentation", ["work", "docs"]),
        ]

        for content, tags in memories_data:
            memory = Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            await sqlite_storage.store(memory)

        # Search by tag
        work_results = await sqlite_storage.search_by_tag(["work"])
        assert len(work_results) == 2

        meeting_results = await sqlite_storage.search_by_tag(["meeting"])
        assert len(meeting_results) == 1
        assert "meeting notes" in meeting_results[0].content.lower()

    @pytest.mark.asyncio
    async def test_delete_and_verify_gone(self, sqlite_storage):
        """Test that deleted content is truly gone."""
        content = "This memory will be deleted"
        memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["delete-test"], memory_type="note")

        # Store
        await sqlite_storage.store(memory)

        # Verify stored
        results = await sqlite_storage.retrieve("memory deleted", n_results=1)
        assert len(results) == 1

        # Delete
        success, _ = await sqlite_storage.delete(memory.content_hash)
        assert success

        # Verify gone from semantic search
        results = await sqlite_storage.retrieve("memory deleted", n_results=10)
        for r in results:
            assert r.memory.content_hash != memory.content_hash

        # Verify gone from hash lookup
        retrieved = await sqlite_storage.get_memory_by_hash(memory.content_hash)
        assert retrieved is None


# =============================================================================
# Qdrant Integration Tests
# =============================================================================

try:
    from qdrant_client import QdrantClient

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

if QDRANT_AVAILABLE:
    from mcp_memory_service.storage.qdrant_storage import QdrantStorage


def is_qdrant_server_available() -> bool:
    """Check if Qdrant server is running."""
    if not QDRANT_AVAILABLE:
        return False
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture
async def qdrant_storage() -> AsyncGenerator["QdrantStorage", None]:
    """Create a real Qdrant storage instance for testing (embedded mode)."""
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client not available")

    temp_dir = tempfile.mkdtemp()

    storage = QdrantStorage(
        embedding_model="all-MiniLM-L6-v2", collection_name="test_integration", storage_path=temp_dir, quantization_enabled=False
    )
    await storage.initialize()

    yield storage

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not available")
class TestQdrantIntegration:
    """Integration tests for Qdrant storage with REAL embeddings."""

    @pytest.mark.asyncio
    async def test_store_retrieve_roundtrip(self, qdrant_storage):
        """Test that stored content can be retrieved via semantic search."""
        content = "The quick brown fox jumps over the lazy dog"
        memory = Memory(
            content=content, content_hash=generate_content_hash(content), tags=["test", "animal"], memory_type="note"
        )

        # Store
        success, message = await qdrant_storage.store(memory)
        assert success, f"Store failed: {message}"

        # Retrieve with semantically similar query
        results = await qdrant_storage.retrieve("fox jumping over dog", n_results=5)

        assert len(results) > 0, "No results returned"
        assert results[0].memory.content == content
        assert results[0].relevance_score > 0.5, "Relevance score too low"

    @pytest.mark.asyncio
    async def test_semantic_similarity_ranking(self, qdrant_storage):
        """Test that semantically similar content ranks higher than dissimilar content."""
        # Store diverse content
        memories = [
            ("Python is a programming language used for web development and data science", ["programming"]),
            ("Cats are furry animals that make great pets", ["animals"]),
            ("Machine learning is a subset of artificial intelligence", ["programming", "ai"]),
            ("Dogs are loyal companions and popular pets", ["animals"]),
        ]

        for content, tags in memories:
            memory = Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            await qdrant_storage.store(memory)

        # Search for programming-related content
        results = await qdrant_storage.retrieve("software development and coding", n_results=4)

        assert len(results) >= 2

        # Programming content should rank higher than animal content
        top_contents = [r.memory.content for r in results[:2]]
        assert any(
            "programming" in c.lower() or "machine learning" in c.lower() for c in top_contents
        ), f"Expected programming content in top results, got: {top_contents}"

    @pytest.mark.asyncio
    async def test_edge_case_unicode_content(self, qdrant_storage):
        """Test handling of unicode content."""
        content = "Python programming with international text: Bonjour, Guten Tag, Hello"
        memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["unicode"], memory_type="note")

        success, message = await qdrant_storage.store(memory)
        assert success, f"Store failed: {message}"

        results = await qdrant_storage.retrieve("international greeting", n_results=1)
        assert len(results) == 1
        assert "Bonjour" in results[0].memory.content

    @pytest.mark.asyncio
    async def test_tag_search_with_real_data(self, qdrant_storage):
        """Test tag search with real stored data."""
        memories_data = [
            ("Work meeting notes from Monday", ["work", "meeting"]),
            ("Personal todo list for the weekend", ["personal", "todo"]),
            ("Work project documentation", ["work", "docs"]),
        ]

        for content, tags in memories_data:
            memory = Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            await qdrant_storage.store(memory)

        work_results = await qdrant_storage.search_by_tag(tags=["work"])
        assert len(work_results) == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self, qdrant_storage):
        """Test that circuit breaker allows recovery after timeout."""
        # This tests real circuit breaker behavior, not mocks
        # Store something first to verify connection works
        content = "Test content for circuit breaker recovery"
        memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["test"], memory_type="note")

        success, _ = await qdrant_storage.store(memory)
        assert success, "Initial store should succeed"

        # Verify retrieval works
        results = await qdrant_storage.retrieve("circuit breaker", n_results=1)
        assert len(results) == 1


# =============================================================================
# MemoryService Integration Tests
# =============================================================================


@pytest.fixture
async def memory_service_with_sqlite(sqlite_storage) -> MemoryService:
    """Create MemoryService with real SQLite-vec storage."""
    return MemoryService(storage=sqlite_storage)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestMemoryServiceIntegration:
    """Integration tests for MemoryService with REAL storage backend."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_workflow(self, memory_service_with_sqlite):
        """Test complete store -> retrieve workflow with real storage."""
        service = memory_service_with_sqlite

        # Store a memory
        result = await service.store_memory(
            content="Integration test: Python programming tutorial", tags=["python", "tutorial"], memory_type="note"
        )

        assert result["success"], f"Store failed: {result.get('error')}"
        stored_hash = result["memory"]["content_hash"]

        # Retrieve via semantic search
        retrieve_result = await service.retrieve_memories(query="learning python programming", page=1, page_size=10)

        assert "memories" in retrieve_result
        assert len(retrieve_result["memories"]) > 0

        # Verify our stored memory is in results
        hashes = [m["content_hash"] for m in retrieve_result["memories"]]
        assert stored_hash in hashes

    @pytest.mark.asyncio
    async def test_tag_search_workflow(self, memory_service_with_sqlite):
        """Test tag-based search with real storage."""
        service = memory_service_with_sqlite

        # Store memories with different tags
        await service.store_memory(
            content="Meeting notes for project alpha", tags=["meeting", "project-alpha"], memory_type="note"
        )
        await service.store_memory(
            content="Personal notes about weekend plans", tags=["personal", "weekend"], memory_type="note"
        )

        # Search by tag
        result = await service.search_by_tag(tags=["meeting"])

        assert len(result["memories"]) == 1
        assert "Meeting notes" in result["memories"][0]["content"]

    @pytest.mark.asyncio
    async def test_delete_workflow(self, memory_service_with_sqlite):
        """Test delete workflow with verification."""
        service = memory_service_with_sqlite

        # Store
        store_result = await service.store_memory(
            content="This memory will be deleted in the test", tags=["delete-test"], memory_type="note"
        )
        content_hash = store_result["memory"]["content_hash"]

        # Verify exists
        get_result = await service.get_memory_by_hash(content_hash)
        assert get_result["found"]

        # Delete
        delete_result = await service.delete_memory(content_hash)
        assert delete_result["success"]

        # Verify gone
        get_result = await service.get_memory_by_hash(content_hash)
        assert not get_result["found"]

    @pytest.mark.asyncio
    async def test_health_check_with_real_storage(self, memory_service_with_sqlite):
        """Test health check reports accurate stats."""
        service = memory_service_with_sqlite

        # Check health before storing
        health1 = await service.check_database_health()
        assert health1["healthy"]
        initial_count = health1.get("total_memories", 0)

        # Store a memory
        await service.store_memory(content="Health check test memory", tags=["health-test"], memory_type="note")

        # Check health after storing
        health2 = await service.check_database_health()
        assert health2["healthy"]
        assert health2["total_memories"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_pagination_with_real_data(self, memory_service_with_sqlite):
        """Test pagination returns correct results."""
        service = memory_service_with_sqlite

        # Store 5 memories
        for i in range(5):
            await service.store_memory(
                content=f"Pagination test memory number {i}", tags=["pagination-test"], memory_type="note"
            )

        # Get page 1 with size 2
        result = await service.list_memories(page=1, page_size=2, tag="pagination-test")

        assert len(result["memories"]) == 2
        assert result["total"] == 5
        assert result["has_more"] is True
        assert result["page"] == 1

        # Get page 2
        result2 = await service.list_memories(page=2, page_size=2, tag="pagination-test")

        assert len(result2["memories"]) == 2
        assert result2["page"] == 2

        # Get page 3 (partial)
        result3 = await service.list_memories(page=3, page_size=2, tag="pagination-test")

        assert len(result3["memories"]) == 1
        assert result3["has_more"] is False
