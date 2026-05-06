"""Test versioned update via memory_update handler (issue #742)."""
import pytest
import pytest_asyncio
from mcp_memory_service.server import MemoryServer


@pytest_asyncio.fixture
async def server():
    srv = MemoryServer()
    yield srv


@pytest.mark.asyncio
async def test_versioned_update_creates_new_version():
    """Versioned=True creates a new memory version and supersedes the old one."""
    server = MemoryServer()
    result = await server.store_memory(content="original content", metadata={"tags": ["test"]})
    assert result["success"]
    original_hash = result["hash"]

    # Perform versioned update
    response = await server.handle_update_memory_metadata({
        "content_hash": original_hash,
        "updates": {"content": "updated content v2", "tags": ["test", "v2"]},
        "versioned": True,
    })

    text = response[0].text
    assert "Versioned update successful" in text
    assert "New hash:" in text
    assert f"parent hash: {original_hash}" in text

    # Extract new hash from response
    new_hash = text.split("New hash: ")[1].split(",")[0]
    assert new_hash != original_hash

    # Verify old memory is superseded
    storage = server.storage
    row = storage.conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (original_hash,)
    ).fetchone()
    assert row is not None
    assert row[0] == new_hash


@pytest.mark.asyncio
async def test_versioned_update_requires_content():
    """Versioned update without content field returns error."""
    server = MemoryServer()
    result = await server.store_memory(content="some content")
    original_hash = result["hash"]

    response = await server.handle_update_memory_metadata({
        "content_hash": original_hash,
        "updates": {"tags": ["new-tag"]},
        "versioned": True,
    })

    assert "content" in response[0].text.lower() and "require" in response[0].text.lower()


@pytest.mark.asyncio
async def test_non_versioned_update_unchanged():
    """Default (versioned=False) still does in-place metadata update."""
    server = MemoryServer()
    result = await server.store_memory(content="stable content", metadata={"tags": ["old"]})
    original_hash = result["hash"]

    response = await server.handle_update_memory_metadata({
        "content_hash": original_hash,
        "updates": {"tags": ["new"]},
    })

    assert "Successfully updated" in response[0].text

    # Verify no superseded_by set
    storage = server.storage
    row = storage.conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (original_hash,)
    ).fetchone()
    assert row is not None
    assert row[0] is None
