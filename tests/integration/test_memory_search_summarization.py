"""Issue #1103: exercise the real MCP route and SQLite storage, mocking HTTP only."""

import copy
import json
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.tools.routing import resolve_handler
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture(autouse=True)
def enable_search_summarization(monkeypatch):
    """Existing behavior tests run with explicit operator permission."""
    from mcp_memory_service.config import search as search_config

    # Seed the proposed setting when tests run against the base branch too.
    monkeypatch.setattr(
        search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", True, raising=False
    )


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
@pytest.mark.parametrize("setting", [None, "false", "invalid"])
@pytest.mark.parametrize("provider_config", ["chain", "legacy"])
async def test_disabled_policy_preserves_raw_results_without_calling_provider(
    search_server, monkeypatch, llm_post, setting, provider_config
):
    from mcp_memory_service.config import search as search_config
    from mcp_memory_service.harvest.rewriter import HarvestRewriter

    if setting is None:
        monkeypatch.delenv("MCP_SEARCH_SUMMARIZE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MCP_SEARCH_SUMMARIZE_ENABLED", setting)
    settings = runpy.run_path(
        search_config.__file__, run_name="mcp_memory_service.config._policy_test"
    )
    monkeypatch.setattr(
        search_config,
        "MCP_SEARCH_SUMMARIZE_ENABLED",
        settings["MCP_SEARCH_SUMMARIZE_ENABLED"],
    )
    if provider_config == "legacy":
        monkeypatch.delenv("HARVEST_LLM_PROVIDERS")
        monkeypatch.setenv("HARVEST_LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    assert HarvestRewriter().is_configured
    provider_call = AsyncMock()
    monkeypatch.setattr(HarvestRewriter, "_call_llm", provider_call)
    server, storage, rows = search_server
    before = [(await storage.get_by_hash(row.content_hash)).to_dict() for row in rows]
    raw = await search(server)

    response = await search(server, summarize=True)

    assert response == (
        "Summarization unavailable: disabled by MCP_SEARCH_SUMMARIZE_ENABLED. "
        "Returning raw results.\n\n" + raw
    )
    provider_call.assert_not_called()
    llm_post.assert_not_called()
    assert [
        (await storage.get_by_hash(row.content_hash)).to_dict() for row in rows
    ] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 100, 2500])
async def test_disabled_policy_respects_response_limit(
    search_server, monkeypatch, llm_post, cap
):
    from mcp_memory_service.config import search as search_config

    monkeypatch.setattr(search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", False)
    server, _, rows = search_server
    response = await search(server, summarize=True, max_response_chars=cap)

    assert len(response) <= cap
    assert response.startswith("Summarization unavailable"[:cap])
    if cap == 2500:
        assert "disabled by MCP_SEARCH_SUMMARIZE_ENABLED" in response
        assert response.count("=== Memory") == 1
        assert any(row.content in response for row in rows)
    llm_post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [0, 1, 100, 500])
@pytest.mark.parametrize("include_beliefs", [False, True])
async def test_disabled_empty_search_warns_without_initializing_provider(
    search_server, monkeypatch, llm_post, cap, include_beliefs
):
    from mcp_memory_service.config import search as search_config
    from mcp_memory_service.server.handlers import memory as memory_handler
    from mcp_memory_service.services import search_summarizer

    monkeypatch.setattr(search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", False)
    rewriter_init = Mock(side_effect=AssertionError("provider must not initialize"))
    monkeypatch.setattr(search_summarizer, "HarvestRewriter", rewriter_init)
    beliefs = "\n\nBeliefs:\n" + "Supporting context. " * 30
    monkeypatch.setattr(
        memory_handler,
        "_format_beliefs_section",
        AsyncMock(return_value=beliefs if include_beliefs else ""),
    )
    server, storage, rows = search_server
    before = [(await storage.get_by_hash(row.content_hash)).to_dict() for row in rows]
    raw = await search(
        server, query="nonexistent content", include_beliefs=include_beliefs
    )

    response = await search(
        server,
        query="nonexistent content",
        summarize=True,
        max_response_chars=cap,
        include_beliefs=include_beliefs,
    )

    warning = (
        "Summarization unavailable: disabled by MCP_SEARCH_SUMMARIZE_ENABLED. "
        "Returning raw results.\n\n"
    )
    if cap == 0:
        assert response == warning + raw
    else:
        assert len(response) <= cap
        assert response.startswith(warning[:cap])
        if cap == 500:
            assert "No memories found for query: 'nonexistent content'" in response
            assert "Supporting context" not in response
    rewriter_init.assert_not_called()
    llm_post.assert_not_called()
    assert [
        (await storage.get_by_hash(row.content_hash)).to_dict() for row in rows
    ] == before


@pytest.mark.asyncio
async def test_stored_access_queries_never_leave_the_summary_path(
    search_server, llm_post
):
    server, storage, rows = search_server
    rows[0].record_access("private prior search")
    assert await storage.update_memory(rows[0])
    before = copy.deepcopy((await storage.get_by_hash(rows[0].content_hash)).to_dict())
    assert before["access_queries"][0]["query"] == "private prior search"

    response = await search(server, summarize=True)
    result = json.loads(response)
    prompt = llm_post.call_args.kwargs["json"]["messages"][0]["content"]

    assert "private prior search" not in prompt
    assert "access_queries" not in prompt
    assert "private prior search" not in response
    assert "access_queries" not in response
    assert result["source_hashes"] == [result["snapshot"][0]["content_hash"]]
    source = next(
        item
        for item in result["snapshot"]
        if item["content_hash"] == rows[0].content_hash
    )
    assert source["owner"] == "bus-team"
    assert source["required"] == ["ordering"]
    assert (await storage.get_by_hash(rows[0].content_hash)).to_dict() == before


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
@pytest.mark.parametrize("cap", [2500, 5000])
@pytest.mark.parametrize("failure", ["provider-error", "summary-budget"])
async def test_large_beliefs_do_not_crowd_out_raw_fallback(
    search_server, llm_post, cap, failure
):
    from datetime import datetime, timezone

    from mcp_memory_service.consolidation.belief_service import BeliefService

    server, storage, rows = search_server
    belief = "Large derived belief. " * 1000
    await BeliefService(storage)._create_belief(
        "large-belief-hash",
        belief,
        0.9,
        "active",
        [rows[0].content_hash],
        [],
        datetime.now(timezone.utc),
    )
    llm_post.side_effect = httpx.ConnectError("unreachable")

    without_beliefs = await search(server, summarize=True, max_response_chars=cap)
    if failure == "summary-budget":
        llm_post.side_effect = None
    response = await search(
        server, summarize=True, include_beliefs=True, max_response_chars=cap
    )

    assert len(response) <= cap
    assert response.startswith("Summarization unavailable")
    assert "Optional section omitted" in response
    assert "large-belief-hash" not in response
    shown = [memory for memory in rows if memory.content_hash in response]
    assert len(shown) == (1 if cap == 2500 else 2)
    for memory in rows:
        assert (memory.content_hash in response) == (
            memory.content_hash in without_beliefs
        )
        if memory in shown:
            assert memory.content in response
        assert (
            await storage.get_by_hash(memory.content_hash)
        ).content == memory.content


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
    response = await search(server, summarize=True, max_response_chars=100)

    assert "Summarization unavailable" in response
    assert len(response) <= 100
    assert "=== Memory" not in response
    assert '"snapshot"' not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 25, 100, 512, 2500, 5000])
@pytest.mark.parametrize(
    "failure", ["provider-error", "unconfigured", "invalid-citation"]
)
async def test_summary_fallback_enforces_the_complete_response_budget(
    search_server, llm_post, monkeypatch, cap, failure
):
    server, _, rows = search_server
    if failure == "unconfigured":
        monkeypatch.delenv("HARVEST_LLM_PROVIDERS")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    elif failure == "provider-error":
        llm_post.side_effect = httpx.ConnectError("unreachable")
    else:
        llm_post.return_value = httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={"choices": [{"message": {"content": "Invented source [999]."}}]},
        )

    response = await search(server, summarize=True, max_response_chars=cap)

    assert len(response) <= cap
    assert response.startswith("Summarization unavailable"[:cap])
    assert "Invented source" not in response
    for row in rows:
        # A returned memory must retain its complete content and hash.
        if row.content_hash in response:
            assert row.content in response
    if cap == 2500:
        assert response.count("=== Memory") == 1
        assert "1 result(s) omitted" in response
    if cap == 5000:
        assert all(
            row.content in response and row.content_hash in response for row in rows
        )


@pytest.mark.asyncio
async def test_time_only_and_empty_search_do_not_summarize(
    search_server, llm_post, monkeypatch
):
    from mcp_memory_service.services import search_summarizer

    rewriter_init = Mock(side_effect=AssertionError("provider must not initialize"))
    monkeypatch.setattr(search_summarizer, "HarvestRewriter", rewriter_init)
    server, _, _ = search_server
    time_only = await search(
        server, query=None, mode="semantic", tags=["bus"], summarize=True
    )
    empty = await search(server, query="nonexistent content", summarize=True)

    assert "Found 1 memories" in time_only
    assert "Summarization unavailable" in time_only
    assert empty == "No memories found for query: 'nonexistent content'"
    rewriter_init.assert_not_called()
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_error_does_not_call_llm(search_server, llm_post):
    server, _, _ = search_server
    response = await search(server, mode="invalid", summarize=True)

    assert response.startswith("Error: Invalid mode")
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Ordering matters [1].", "Invented source [999]."])
@pytest.mark.parametrize("operator_enabled", [False, True])
@pytest.mark.parametrize("query", ["replication ordering", "nonexistent content"])
async def test_http_mcp_round_trip_with_read_scope(
    search_server, monkeypatch, answer, operator_enabled, query
):
    from fastapi import FastAPI

    from mcp_memory_service.config import search as search_config
    from mcp_memory_service.server import MemoryServer
    from mcp_memory_service.web.api import mcp as mcp_module
    from mcp_memory_service.web.oauth.middleware import (
        AuthenticationResult,
        require_read_access,
    )

    monkeypatch.setattr(search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", operator_enabled)
    _, storage, rows = search_server
    rows[0].record_access("HTTP private prior query")
    assert await storage.update_memory(rows[0])
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
        assert "HTTP private prior query" not in request.content.decode()
        assert "access_queries" not in request.content.decode()
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
                        "query": query,
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
    assert "HTTP private prior query" not in text
    assert "access_queries" not in text
    if not operator_enabled:
        assert "disabled by MCP_SEARCH_SUMMARIZE_ENABLED" in text
        if query == "nonexistent content":
            assert text.endswith(f"No memories found for query: '{query}'")
        else:
            assert all(memory.content in text for memory in rows)
    elif query == "nonexistent content":
        assert text == f"No memories found for query: '{query}'"
    elif "999" in answer:
        assert "Summarization unavailable" in text
        assert all(memory.content in text for memory in rows)
    else:
        result = json.loads(text)
        assert result["summary"] == answer
        assert result["source_hashes"] == [result["snapshot"][0]["content_hash"]]
    assert len(provider_requests) == int(
        operator_enabled and query == "replication ordering"
    )
    stored = await storage.get_by_hash(rows[0].content_hash)
    assert stored.metadata["access_queries"][0]["query"] == "HTTP private prior query"
