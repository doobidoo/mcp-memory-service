from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers.memory import handle_memory_search

CONTENT_HASH = "aa6caea3d060e736c8b412e6b34a1354148ec04527a4bf547cfd4185140573d0"


@pytest.fixture
def search_server():
    server = MagicMock()
    storage = AsyncMock()
    storage.get_by_hash_is_store_scoped = True
    server._ensure_storage_initialized = AsyncMock(return_value=storage)
    return server, storage


@pytest.mark.asyncio
async def test_content_hash_bypasses_text_search(search_server):
    server, storage = search_server
    storage.get_by_hash.return_value = Memory(
        content="Retrieved by hash",
        content_hash=CONTENT_HASH,
        tags=["hash-test"],
    )

    response = await handle_memory_search(server, {"content_hash": CONTENT_HASH})

    assert "Found 1 memories (mode: hash)" in response[0].text
    assert "Retrieved by hash" in response[0].text
    storage.get_by_hash.assert_awaited_once_with(CONTENT_HASH, store="default")
    storage.search_memories.assert_not_awaited()


@pytest.mark.asyncio
async def test_content_hash_trims_surrounding_whitespace(search_server):
    server, storage = search_server
    storage.get_by_hash.return_value = None

    response = await handle_memory_search(
        server, {"content_hash": f"  {CONTENT_HASH}\n"}
    )

    assert f"for content hash: '{CONTENT_HASH}'" in response[0].text
    storage.get_by_hash.assert_awaited_once_with(CONTENT_HASH, store="default")


@pytest.mark.asyncio
async def test_hash_miss_does_not_fall_back_to_search(search_server):
    server, storage = search_server
    storage.get_by_hash.return_value = None

    response = await handle_memory_search(server, {"content_hash": CONTENT_HASH})

    assert "No memories found" in response[0].text
    storage.get_by_hash.assert_awaited_once_with(CONTENT_HASH, store="default")
    storage.search_memories.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_store, expected_store",
    [
        (None, "default"),
        ("docs", "docs"),
        ("all", None),
    ],
)
async def test_hash_lookup_preserves_store_scope(
    search_server,
    requested_store,
    expected_store,
):
    server, storage = search_server
    storage.get_by_hash.return_value = None
    arguments = {"content_hash": CONTENT_HASH}
    if requested_store is not None:
        arguments["store"] = requested_store

    await handle_memory_search(server, arguments)

    storage.get_by_hash.assert_awaited_once_with(CONTENT_HASH, store=expected_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_store", [None, "project-a"])
async def test_hash_lookup_rejects_scopes_backend_cannot_enforce(
    search_server, requested_store
):
    server, storage = search_server
    storage.get_by_hash_is_store_scoped = False
    arguments = {"content_hash": CONTENT_HASH}
    if requested_store is not None:
        arguments["store"] = requested_store

    response = await handle_memory_search(server, arguments)

    assert response[0].text == (
        "Error: store-scoped content_hash lookup is not supported on this "
        "storage backend"
    )
    storage.get_by_hash.assert_not_awaited()


@pytest.mark.asyncio
async def test_hash_lookup_all_stores_does_not_require_scoped_storage(search_server):
    server, storage = search_server
    storage.get_by_hash_is_store_scoped = False
    storage.get_by_hash.return_value = Memory(
        content="Retrieved across stores",
        content_hash=CONTENT_HASH,
    )

    response = await handle_memory_search(
        server, {"content_hash": CONTENT_HASH, "store": "all"}
    )

    assert "Retrieved across stores" in response[0].text
    storage.get_by_hash.assert_awaited_once_with(CONTENT_HASH, store=None)


@pytest.mark.asyncio
async def test_sha256_text_in_query_remains_text_search(search_server):
    server, storage = search_server
    storage.search_memories.return_value = {
        "memories": [],
        "total": 0,
        "query": CONTENT_HASH,
        "mode": "semantic",
    }

    await handle_memory_search(server, {"query": CONTENT_HASH})

    storage.search_memories.assert_awaited_once()
    storage.get_by_hash.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_and_content_hash_are_rejected_together(search_server):
    server, storage = search_server

    response = await handle_memory_search(
        server,
        {
            "query": "checksums in a log",
            "content_hash": CONTENT_HASH,
        },
    )

    assert response[0].text == "Error: query and content_hash are mutually exclusive"
    storage.get_by_hash.assert_not_awaited()
    storage.search_memories.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported_arguments",
    [
        {"time_expr": "last week"},
        {"after": "2024-01-01"},
        {"before": "2024-06-30"},
        {"tags": ["project"]},
        {"agent_id": "agent-a"},
        {"as_of": "2024-01-01"},
        {"include_superseded": True},
        {"mode": "semantic"},
        {"fallback": True},
        {"future_search_option": "value"},
    ],
)
async def test_hash_lookup_rejects_unsupported_arguments(
    search_server, unsupported_arguments
):
    server, storage = search_server
    argument_name = next(iter(unsupported_arguments))

    response = await handle_memory_search(
        server,
        {"content_hash": CONTENT_HASH, **unsupported_arguments},
    )

    assert response[0].text == (
        f"Error: content_hash cannot be combined with search arguments: {argument_name}"
    )
    storage.get_by_hash.assert_not_awaited()
    storage.search_memories.assert_not_awaited()


def test_content_hash_is_exposed_in_tool_schema():
    from mcp_memory_service.tools.registry import TOOL_REGISTRY

    search_tool = next(tool for tool in TOOL_REGISTRY if tool.name == "memory_search")
    content_hash = search_tool.input_schema["properties"]["content_hash"]

    assert content_hash["type"] == "string"
    assert content_hash["minLength"] == 1
    assert (
        "query and other search arguments are rejected" in content_hash["description"]
    )
    assert "store" in content_hash["description"]
    assert "max_response_chars" in content_hash["description"]
