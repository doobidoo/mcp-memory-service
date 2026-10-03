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


def test_memory_update_schema_declares_versioned_fields():
    """updates schema must declare every field the versioned handler reads.

    handle_update_memory_metadata() hard-requires updates["content"] (and reads
    updates["reason"]) when versioned=true. If the schema omits them, clients
    that build arguments from the declared properties can never satisfy the
    requirement.
    """
    from mcp_memory_service.tools.registry import TOOL_REGISTRY

    tool = next(t for t in TOOL_REGISTRY if t.name == "memory_update")
    updates_props = tool.input_schema["properties"]["updates"]["properties"]

    assert "content" in updates_props
    assert "reason" in updates_props


@pytest.mark.asyncio
async def test_inplace_update_strips_versioned_only_fields():
    """content/reason are versioned-only and must not reach update_memory_metadata.

    Otherwise content advances updated_at via the structural-change check and
    reason falls through into custom metadata — callers following the schema
    would change timestamps or metadata they did not intend to change.
    """
    from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata

    server = MagicMock()
    storage = AsyncMock()
    storage.update_memory_metadata = AsyncMock(return_value=(True, "Updated"))
    server._ensure_storage_initialized = AsyncMock(return_value=storage)

    result = await handle_update_memory_metadata(server, {
        "content_hash": "abc123",
        "updates": {
            "content": "new content",
            "reason": "some reason",
            "tags": ["keep"],
            "priority": "urgent",
        },
    })

    assert "Successfully" in result[0].text
    received = storage.update_memory_metadata.call_args.kwargs["updates"]
    assert "content" not in received
    assert "reason" not in received
    assert received["tags"] == ["keep"]
    assert received["priority"] == "urgent"


@pytest.mark.asyncio
async def test_versioned_update_metadata_overrides_inherited(tmp_path):
    """updates["metadata"] wins over the inherited values on the versioned path.

    update_memory_versioned() copies the old row's custom metadata onto the new
    version, and the memory_update schema tells callers that fields they pass in
    metadata override those inherited values. The handler has to apply them
    after the write, otherwise the declared override silently does nothing.
    """
    from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

    storage = SqliteVecMemoryStorage(str(tmp_path / "test.db"))
    await storage.initialize()
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock(return_value=storage)

    original = Memory(
        content="The backup job runs nightly at 02:00.",
        content_hash=generate_content_hash("The backup job runs nightly at 02:00."),
        tags=["backup"],
        memory_type="observation",
        metadata={"source": "runbook", "ticket": "OPS-17"},
    )
    ok, msg = await storage.store(original, skip_semantic_dedup=True)
    assert ok, f"Failed to store original: {msg}"

    new_content = "The backup job runs nightly at 03:00."
    result = await handle_update_memory_metadata(server, {
        "content_hash": original.content_hash,
        "updates": {
            "content": new_content,
            "reason": "schedule change",
            "metadata": {"source": "harvest"},
        },
        "versioned": True,
    })

    assert "Versioned update successful" in result[0].text
    new = await storage.get_by_hash(generate_content_hash(new_content))
    assert new.metadata["source"] == "harvest"
    assert new.metadata["ticket"] == "OPS-17"

    await storage.close()
