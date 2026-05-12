"""Tests for MemoryService.store_memory() — Task 2: conversation_id threading."""
import os
import tempfile
import pytest

# Ensure sqlite_vec backend and a fresh test DB for this module
os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
_test_db_dir = tempfile.mkdtemp(prefix='mcp-service-test-')
os.environ['MCP_MEMORY_SQLITE_PATH'] = os.path.join(_test_db_dir, 'test.db')
os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = 'false'  # off by default; tests override

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def memory_service(tmp_path):
    """Create a fresh MemoryService backed by a temp SQLite-Vec database."""
    db_path = str(tmp_path / "test_service.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    service = MemoryService(storage=storage)
    yield service
    await storage.close()


class TestStoreMemory:
    """Tests for MemoryService.store_memory()."""

    @pytest.mark.asyncio
    async def test_conversation_id_bypasses_semantic_dedup(self, memory_service):
        """Storing with conversation_id skips semantic dedup."""
        # Enable semantic dedup on the storage backend
        memory_service.storage.semantic_dedup_enabled = True

        result1 = await memory_service.store_memory(
            content="Claude Code is a powerful CLI tool for software engineering.",
            tags=["test"],
            conversation_id="conv-abc"
        )
        assert result1["success"]

        # Similar content, same conversation — should succeed (dedup skipped)
        result2 = await memory_service.store_memory(
            content="The Claude Code CLI is an excellent software development tool.",
            tags=["test"],
            conversation_id="conv-abc"
        )
        assert result2["success"], f"Expected success with conversation_id, got: {result2.get('error')}"

        # Similar content, NO conversation_id — should be rejected by semantic dedup
        result3 = await memory_service.store_memory(
            content="Claude Code CLI is a top-tier software engineering tool.",
            tags=["test"]
        )
        assert not result3["success"]
        assert "semantically similar" in result3.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_conversation_id_persisted_in_metadata(self, memory_service):
        """conversation_id is stored in memory metadata for future grouping/retrieval."""
        result = await memory_service.store_memory(
            content="A memory tagged with a conversation ID for retrieval.",
            tags=["test"],
            conversation_id="conv-persist-check"
        )
        assert result["success"]
        content_hash = result["memory"]["content_hash"]

        # Retrieve the stored memory and verify conversation_id is in metadata
        memory = await memory_service.storage.get_by_hash(content_hash)
        assert memory is not None
        assert memory.metadata.get("conversation_id") == "conv-persist-check"

    @pytest.mark.asyncio
    async def test_session_memory_type_bypasses_semantic_dedup(self, memory_service):
        """Storing with memory_type='session' skips semantic dedup.

        Session logs are verbose multi-turn transcripts. Comparing them
        against short atomic facts via cosine similarity is a category error —
        near-identical scores are expected, not a sign of actual duplication.
        """
        # Enable semantic dedup on the storage backend
        memory_service.storage.semantic_dedup_enabled = True

        # Store an atomic memory about a topic
        result1 = await memory_service.store_memory(
            content="Milvus deployment requires etcd, MinIO, and the standalone server.",
            tags=["milvus"],
        )
        assert result1["success"]

        # Store a session discussing the same topic — should succeed
        session_content = (
            "[user] How do I deploy Milvus on K8s?\n"
            "[assistant] You need etcd, MinIO, and Milvus standalone. "
            "Deploy them in order with initContainers."
        )
        result2 = await memory_service.store_memory(
            content=session_content,
            tags=["session:test-session-123"],
            memory_type="session",
        )
        assert result2["success"], (
            f"Expected session to bypass semantic dedup, got: {result2.get('error')}"
        )

    @pytest.mark.asyncio
    async def test_non_session_still_blocked_by_semantic_dedup(self, memory_service):
        """Non-session memories with similar content are still blocked."""
        memory_service.storage.semantic_dedup_enabled = True

        result1 = await memory_service.store_memory(
            content="Redis timeout should be set to 3 seconds for production.",
            tags=["config"],
        )
        assert result1["success"]

        # Similar content without session type — should be rejected
        result2 = await memory_service.store_memory(
            content="Redis timeout is configured at 3s in production environments.",
            tags=["config"],
        )
        assert not result2["success"]
        assert "semantically similar" in result2.get("error", "").lower()
