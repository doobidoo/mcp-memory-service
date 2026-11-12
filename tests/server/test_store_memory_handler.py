"""Tests for handle_store_memory handler.

Covers ontology-coercion surfacing introduced for issue #843. The handler
must echo a clear warning when Memory.__post_init__ silently rewrites an
unknown memory_type, so callers don't see "success" while their type
filter quietly breaks.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_server(stored_memory_type: str = "note"):
    """Build a minimal mock server. The fake storage layer records the
    `memory_type` that ended up persisted so the test can simulate
    coercion (e.g. requested='foo', stored='observation')."""
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock()
    server.memory_service = MagicMock()
    server.memory_service.store_memory = AsyncMock(return_value={
        "success": True,
        "memory": {
            "content_hash": "abc123",
            "memory_type": stored_memory_type,
        },
    })
    return server


@pytest.mark.asyncio
async def test_store_memory_explicit_valid_type_no_warning():
    from mcp_memory_service.server.handlers.memory import handle_store_memory
    server = _make_server(stored_memory_type="note")

    result = await handle_store_memory(server, {
        "content": "hello",
        "metadata": {"type": "note"},
    })

    assert len(result) == 1
    text = result[0].text
    assert "Memory stored successfully" in text
    assert "Warning" not in text


@pytest.mark.asyncio
async def test_store_memory_names_a_supersede():
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    server = _make_server()
    server.memory_service.store_memory = AsyncMock(return_value={
        "success": True,
        "memory": {"content_hash": "new-hash", "memory_type": "note"},
        "dedup_outcome": "superseded",
        "matched_hash": "old-hash",
    })

    result = await handle_store_memory(server, {"content": "updated fact"})

    text = result[0].text
    assert "Memory stored successfully (hash: new-hash)" in text
    assert "Superseded previous memory old-hash" in text
    assert "dedup_outcome: superseded" in text


@pytest.mark.asyncio
async def test_store_memory_names_a_dedup_rejection():
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    server = _make_server()
    server.memory_service.store_memory = AsyncMock(return_value={
        "success": False,
        "error": "Duplicate content detected (semantically similar to old-hash)",
        "dedup_outcome": "deduplicated",
        "matched_hash": "old-hash",
    })

    with pytest.raises(RuntimeError) as caught:
        await handle_store_memory(server, {"content": "same fact again"})

    message = str(caught.value)
    assert "semantically similar to old-hash" in message
    assert "dedup_outcome: deduplicated" in message
    assert "matched_hash: old-hash" in message


@pytest.mark.asyncio
async def test_chunked_store_names_each_superseded_memory():
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    server = _make_server()
    server.memory_service.store_memory = AsyncMock(return_value={
        "success": True,
        "memories": [{"content_hash": "chunk-hash", "memory_type": "note"}],
        "total_chunks": 2,
        "original_hash": "original",
        "failed_chunks": 0,
        "dedup_decisions": [{
            "chunk_index": 0,
            "dedup_outcome": "superseded",
            "matched_hash": "old-hash",
        }],
    })

    result = await handle_store_memory(server, {"content": "a long note"})

    text = result[0].text
    assert "Successfully stored 1 memory chunks" in text
    assert "Superseded previous memory old-hash" in text
    assert "chunk 0: dedup_outcome: superseded" in text


@pytest.mark.asyncio
async def test_fastmcp_store_memory_reports_a_supersede():
    from mcp_memory_service import mcp_server

    class _Service:
        async def store_memory(self, **_kwargs):
            return {
                "success": True,
                "memory": {"content_hash": "new-hash"},
                "dedup_outcome": "superseded",
                "matched_hash": "old-hash",
            }

    class _RequestContext:
        lifespan_context = type("Life", (), {"memory_service": _Service()})()
        request = None

    class _Ctx:
        request_context = _RequestContext()

    result = await mcp_server.store_memory("updated fact", _Ctx())

    assert result["success"] is True
    assert result["content_hash"] == "new-hash"
    assert result["dedup_outcome"] == "superseded"
    assert result["matched_hash"] == "old-hash"


@pytest.mark.asyncio
async def test_store_memory_implicit_default_type_no_warning():
    """When the user omits `type`, the default 'note' is applied silently —
    no warning should fire even though we technically substituted a value."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory
    server = _make_server(stored_memory_type="note")

    result = await handle_store_memory(server, {
        "content": "hello",
        "metadata": {},
    })

    text = result[0].text
    assert "Memory stored successfully" in text
    assert "Warning" not in text


@pytest.mark.asyncio
async def test_store_memory_unknown_type_warns_about_coercion():
    """Unknown memory_type → silent coercion to 'observation'. The handler
    must include a Warning + the MCP_CUSTOM_MEMORY_TYPES hint so the user
    can fix their setup."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory
    # Storage coerced the type from 'foo' to 'observation'
    server = _make_server(stored_memory_type="observation")

    result = await handle_store_memory(server, {
        "content": "hello",
        "metadata": {"type": "foo"},
    })

    text = result[0].text
    assert "Memory stored successfully" in text
    assert "Warning" in text
    assert "'foo'" in text
    assert "'observation'" in text
    assert "MCP_CUSTOM_MEMORY_TYPES" in text


@pytest.mark.asyncio
async def test_store_memory_explicit_empty_string_warns_on_coercion():
    """An explicit empty `type` is still a deliberate user choice. Ontology
    validation rejects it and coerces to 'observation'; the handler must
    surface the warning instead of silently treating "" as "not provided"."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory
    server = _make_server(stored_memory_type="observation")

    result = await handle_store_memory(server, {
        "content": "hello",
        "metadata": {"type": ""},
    })

    text = result[0].text
    assert "Memory stored successfully" in text
    assert "Warning" in text
    assert "'observation'" in text


@pytest.mark.asyncio
async def test_store_memory_chunked_response_warns_on_coercion():
    """Chunked path uses memories[0].memory_type — same coercion surfacing
    must apply."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock()
    server.memory_service = MagicMock()
    server.memory_service.store_memory = AsyncMock(return_value={
        "success": True,
        "memories": [
            {"content_hash": "h1", "memory_type": "observation"},
            {"content_hash": "h2", "memory_type": "observation"},
        ],
        "original_hash": "orig",
        "total_chunks": 2,
    })

    result = await handle_store_memory(server, {
        "content": "x" * 100000,  # large enough to trigger chunking in real path
        "metadata": {"type": "bogus_type"},
    })

    text = result[0].text
    assert "Successfully stored 2 memory chunks" in text
    assert "Warning" in text
    assert "'bogus_type'" in text
    assert "'observation'" in text
