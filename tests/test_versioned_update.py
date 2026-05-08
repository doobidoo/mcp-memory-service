"""Tests for update_memory_versioned (doobidoo feedback implementation)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def mock_storage():
    """Create a mock storage with update_memory_versioned support."""
    storage = AsyncMock()
    storage.conn = True
    storage.store = AsyncMock(return_value=(True, "Stored"))
    storage.update_memory_metadata = AsyncMock(return_value=(True, "Updated"))
    storage._execute_with_retry = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_versioned_update_happy_path(tmp_path):
    """Creates memory, versions it, verifies chain (old has superseded_by in metadata, new exists)."""
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

    storage = SqliteVecMemoryStorage(str(tmp_path / "test.db"))
    await storage.initialize()

    # Store original memory
    original = Memory(
        content="original content",
        content_hash=generate_content_hash("original content"),
        tags=["test"],
        memory_type="note",
    )
    ok, msg = await storage.store(original, skip_semantic_dedup=True)
    assert ok, f"Failed to store original: {msg}"

    # Perform versioned update
    success, message, new_hash = await storage.update_memory_versioned(
        content_hash=original.content_hash,
        new_content="updated content v2",
        new_tags=["test", "v2"],
        reason="test evolution",
    )
    assert success, f"Versioned update failed: {message}"
    assert new_hash is not None
    assert new_hash == generate_content_hash("updated content v2")

    # Verify old memory has superseded_by in metadata
    import json

    def _read_old():
        cursor = storage.conn.execute(
            "SELECT metadata FROM memories WHERE content_hash = ?",
            (original.content_hash,),
        )
        return cursor.fetchone()

    row = await storage._execute_with_retry(_read_old)
    assert row is not None
    metadata = json.loads(row[0]) if row[0] else {}
    assert metadata.get("superseded_by") == new_hash
    assert metadata.get("evolution_reason") == "test evolution"

    # Verify new memory exists
    def _read_new():
        cursor = storage.conn.execute(
            "SELECT content, tags FROM memories WHERE content_hash = ?",
            (new_hash,),
        )
        return cursor.fetchone()

    new_row = await storage._execute_with_retry(_read_new)
    assert new_row is not None
    assert new_row[0] == "updated content v2"

    await storage.close()


@pytest.mark.asyncio
async def test_versioned_update_unsupported_backend():
    """Mock without update_memory_versioned method returns error in handler."""
    from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata

    # Create a mock server with storage that lacks update_memory_versioned
    server = MagicMock()
    storage = AsyncMock()
    # Remove the method to simulate unsupported backend
    del storage.update_memory_versioned
    server._ensure_storage_initialized = AsyncMock(return_value=storage)

    result = await handle_update_memory_metadata(server, {
        "content_hash": "abc123",
        "updates": {"content": "new content"},
        "versioned": True,
    })

    assert "not supported" in result[0].text.lower()


@pytest.mark.asyncio
async def test_versioned_update_nonexistent_memory(tmp_path):
    """Hash inválido returns error."""
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

    storage = SqliteVecMemoryStorage(str(tmp_path / "test.db"))
    await storage.initialize()

    success, message, new_hash = await storage.update_memory_versioned(
        content_hash="nonexistent_hash_abc123",
        new_content="some new content",
    )

    assert not success
    assert "not found" in message.lower()
    assert new_hash is None

    await storage.close()
