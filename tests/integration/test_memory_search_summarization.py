"""Issue #1103: exercise the real MCP route and SQLite storage, mocking HTTP only."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.tools.routing import resolve_handler
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def llm_post(monkeypatch):
    monkeypatch.setenv("HARVEST_LLM_PROVIDERS", "test")
    monkeypatch.setenv("HARVEST_LLM_TEST_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HARVEST_LLM_TEST_MODEL", "summary-model")
    monkeypatch.delenv("HARVEST_LLM_TEST_API_KEY", raising=False)
    post = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={"choices": [{"message": {"content": "Wait for replication [1]."}}]},
        )
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return post


@pytest_asyncio.fixture
async def search_server(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/summary.db")
    await storage.initialize()

    async def ensure_storage():
        return storage

    server = SimpleNamespace(
        _ensure_storage_initialized=ensure_storage,
        memory_service=MemoryService(storage),
    )
    rows = []
    for content, tags in [
        (
            "Replication ordering: acknowledge messages only after replicated state propagates. "
            + "Additional investigation notes. " * 60,
            ["bus"],
        ),
        (
            "Replication ordering: database lag was ruled out. "
            + "Database investigation context. " * 60,
            ["database"],
        ),
    ]:
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=tags,
            memory_type="decision",
            metadata={"owner": "bus-team", "required": ["ordering"]},
        )
        success, message = await storage.store(memory)
        assert success, message
        rows.append(memory)
    yield server, storage, rows
    await storage.close()


async def search(server, **arguments):
    return (
        await resolve_handler("memory_search")(
            server,
            {"query": "replication ordering", "mode": "exact", **arguments},
        )
    )[0].text


@pytest.mark.asyncio
async def test_summary_keeps_sources_and_originals_queryable(search_server, llm_post):
    server, storage, rows = search_server
    before = [
        copy.deepcopy((await storage.get_by_hash(m.content_hash)).to_dict())
        for m in rows
    ]
    raw = await search(server)
    summarized = await search(server, summarize=True, include_debug=True)
    result = json.loads(summarized)

    assert result["summary"] == "Wait for replication [1]."
    assert result["summarized"] is True
    assert result["total"] == 2
    assert result["summarized_count"] == 2
    assert result["omitted_count"] == 0
    assert result["source_hashes"] == [result["snapshot"][0]["content_hash"]]
    assert {item["content_hash"] for item in result["snapshot"]} == {
        m.content_hash for m in rows
    }
    assert all(item["tags"] and item["created_at_iso"] for item in result["snapshot"])
    assert all(item["owner"] == "bus-team" for item in result["snapshot"])
    assert all(item["required"] == ["ordering"] for item in result["snapshot"])
    assert result["provider"] == "test"
    assert result["model"] == "summary-model"
    assert "debug" in result
    assert all(m.content not in summarized for m in rows)
    assert len(summarized) < len(raw)
    for index, memory in enumerate(rows):
        assert (await storage.get_by_hash(memory.content_hash)).to_dict() == before[
            index
        ]
    llm_post.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["semantic", "hybrid", "ranked"])
async def test_summary_works_with_other_search_modes(search_server, llm_post, mode):
    server, _, rows = search_server
    result = json.loads(await search(server, summarize=True, mode=mode))

    assert result["mode"] == mode
    assert result["summary"] == "Wait for replication [1]."
    assert set(result["source_hashes"]).issubset(
        {memory.content_hash for memory in rows}
    )
    llm_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_requested_beliefs_preserved_in_valid_json(search_server, llm_post):
    from datetime import datetime, timezone

    from mcp_memory_service.consolidation.belief_service import BeliefService

    server, storage, rows = search_server
    await BeliefService(storage)._create_belief(
        "belief-hash",
        "Ordering is significant",
        0.9,
        "active",
        [rows[0].content_hash],
        [],
        datetime.now(timezone.utc),
    )
    result = json.loads(await search(server, summarize=True, include_beliefs=True))

    assert "Ordering is significant" in result["beliefs"]
    assert "belief-hash" in result["beliefs"]
    assert result["source_hashes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", [{}, {"summarize": False}, {"summarize": "false"}]
)
async def test_default_and_false_keep_raw_results(search_server, llm_post, arguments):
    server, _, rows = search_server
    response = await search(server, **arguments)

    assert all(memory.content in response for memory in rows)
    assert response.startswith("Found 2 memories")
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["invalid-citation", "empty", "provider-error", "unconfigured"]
)
async def test_failure_is_visible_and_preserves_raw_results(
    search_server, llm_post, monkeypatch, failure
):
    server, storage, rows = search_server
    raw = await search(server)
    if failure == "unconfigured":
        monkeypatch.delenv("HARVEST_LLM_PROVIDERS")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    elif failure == "provider-error":
        llm_post.side_effect = httpx.ConnectError("unreachable")
    else:
        text = "Invented source [999]." if failure == "invalid-citation" else ""
        llm_post.return_value = httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={"choices": [{"message": {"content": text}}]},
        )

    response = await search(server, summarize=True)
    assert "Summarization unavailable" in response
    assert response.endswith(raw)
    assert "Invented source" not in response
    for memory in rows:
        stored = await storage.get_by_hash(memory.content_hash)
        assert stored.content == memory.content
        assert stored.tags == memory.tags


@pytest.mark.asyncio
async def test_summary_runs_after_tag_filter(search_server, llm_post):
    server, _, rows = search_server
    response = json.loads(await search(server, summarize=True, tags=["bus"]))
    prompt = llm_post.call_args.kwargs["json"]["messages"][0]["content"]

    assert rows[0].content in prompt
    assert rows[1].content not in prompt
    assert response["source_hashes"] == [rows[0].content_hash]
    assert len(response["snapshot"]) == 1


@pytest.mark.asyncio
async def test_summary_uses_final_plugin_results(search_server, llm_post):
    server, _, rows = search_server

    async def retrieve_plugin(query, results):
        return [
            result
            for result in results
            if result["content_hash"] == rows[0].content_hash
        ]

    server.memory_service._plugin_registry.ctx.on("on_retrieve", retrieve_plugin)
    result = json.loads(await search(server, summarize=True))
    prompt = llm_post.call_args.kwargs["json"]["messages"][0]["content"]

    assert rows[0].content in prompt
    assert rows[1].content not in prompt
    assert result["total"] == 1
    assert result["source_hashes"] == [rows[0].content_hash]


@pytest.mark.asyncio
async def test_oversized_provider_response_falls_back_to_raw(search_server, llm_post):
    server, _, _ = search_server
    raw = await search(server)
    llm_post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
        json={
            "choices": [{"message": {"content": "Unbounded answer. " * 2000 + "[1]"}}]
        },
    )

    result = await search(server, summarize=True)
    assert result.endswith(raw)
    assert "Summarization unavailable" in result
    assert "Unbounded answer" not in result


@pytest.mark.asyncio
async def test_cancellation_propagates_through_handler(search_server, llm_post):
    import asyncio

    server, _, _ = search_server
    llm_post.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await search(server, summarize=True)


@pytest.mark.asyncio
async def test_summary_honors_response_limit_without_losing_metadata(
    search_server, llm_post
):
    server, _, _ = search_server
    raw = await search(server, max_response_chars=100)
    response = await search(server, summarize=True, max_response_chars=100)

    assert "Summarization unavailable" in response
    assert response.endswith(raw)
    assert '"snapshot"' not in response


@pytest.mark.asyncio
async def test_time_only_and_empty_search_do_not_summarize(search_server, llm_post):
    server, _, _ = search_server
    time_only = await search(
        server, query=None, mode="semantic", tags=["bus"], summarize=True
    )
    empty = await search(server, query="nonexistent content", summarize=True)

    assert "Found 1 memories" in time_only
    assert "Summarization unavailable" in time_only
    assert empty.startswith("No memories found")
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_error_does_not_call_llm(search_server, llm_post):
    server, _, _ = search_server
    response = await search(server, mode="invalid", summarize=True)

    assert response.startswith("Error: Invalid mode")
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Ordering matters [1].", "Invented source [999]."])
async def test_http_mcp_round_trip_with_read_scope(search_server, monkeypatch, answer):
    from fastapi import FastAPI

    from mcp_memory_service.server import MemoryServer
    from mcp_memory_service.web.api import mcp as mcp_module
    from mcp_memory_service.web.oauth.middleware import (
        AuthenticationResult,
        require_read_access,
    )

    _, storage, rows = search_server
    monkeypatch.setenv("HARVEST_LLM_PROVIDERS", "test")
    monkeypatch.setenv("HARVEST_LLM_TEST_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HARVEST_LLM_TEST_MODEL", "summary-model")
    monkeypatch.delenv("HARVEST_LLM_TEST_API_KEY", raising=False)
    monkeypatch.setattr(mcp_module, "_memory_server", MemoryServer(storage=storage))

    async def read_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="summary-test",
            scope="read",
            auth_method="test",
        )

    app = FastAPI()
    app.include_router(mcp_module.router)
    app.dependency_overrides[require_read_access] = read_user
    provider_requests = []

    def provider(request):
        provider_requests.append(request)
        payload = json.loads(request.content)
        assert request.url == "https://llm.example/v1/chat/completions"
        assert payload["max_tokens"] == 200
        assert "replication ordering" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    # Keep real HTTP request/response serialization. Replace only the external
    # provider transport; the /mcp route and MemoryServer dispatch are real.
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    async with async_client(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        advertised = (
            await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                },
            )
        ).json()
        search_tool = next(
            tool
            for tool in advertised["result"]["tools"]
            if tool["name"] == "memory_search"
        )
        assert search_tool["inputSchema"]["properties"]["summarize"]["default"] is False

        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_search",
                    "arguments": {
                        "query": "replication ordering",
                        "mode": "exact",
                        "summarize": True,
                    },
                },
            },
        )
    assert response.status_code == 200
    wire_result = response.json()
    assert "error" not in wire_result
    text = wire_result["result"]["content"][0]["text"]
    if "999" in answer:
        assert "Summarization unavailable" in text
        assert all(memory.content in text for memory in rows)
    else:
        result = json.loads(text)
        assert result["summary"] == answer
        assert result["source_hashes"] == [result["snapshot"][0]["content_hash"]]
    assert len(provider_requests) == 1
