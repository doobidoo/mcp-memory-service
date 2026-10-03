"""Contract tests for dispatcher-level MCP tool rate limiting."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from mcp import types

from mcp_memory_service.server import MemoryServer
from mcp_memory_service.utils.tool_rate_limit import ToolRateLimiter


async def _echo_handler(_server, arguments):
    return [types.TextContent(type="text", text=json.dumps(arguments))]


@pytest.fixture
def dispatcher(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    monkeypatch.setattr(
        "mcp_memory_service.tools.routing.resolve_handler",
        lambda _name: _echo_handler,
    )
    return MemoryServer()


def _is_error(result):
    return isinstance(result, types.CallToolResult) and result.isError


def _error_payload(result):
    assert _is_error(result)
    return json.loads(result.content[0].text)


def test_http_wrapper_preserves_standard_mcp_error_shape():
    from mcp_memory_service.web.api.mcp import _wrap_tool_result

    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="rate limited")],
        isError=True,
    )

    assert _wrap_tool_result(result) == {
        "content": [{"type": "text", "text": "rate limited"}],
        "isError": True,
    }


@pytest.mark.asyncio
async def test_dispatcher_is_unlimited_by_default(dispatcher, monkeypatch):
    monkeypatch.delenv("MCP_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("MCP_RATE_LIMIT_memory_search", raising=False)
    monkeypatch.delenv("MCP_RATE_LIMIT_MEMORY_SEARCH", raising=False)

    for _ in range(100):
        result = await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
        assert not _is_error(result)


@pytest.mark.asyncio
async def test_dispatcher_enforces_per_agent_limit(dispatcher, monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.delenv("MCP_RATE_LIMIT_memory_search", raising=False)
    monkeypatch.delenv("MCP_RATE_LIMIT_MEMORY_SEARCH", raising=False)

    for _ in range(2):
        result = await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
        assert not _is_error(result)

    payload = _error_payload(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
    )
    assert payload["error"]["type"] == "rate_limit_exceeded"
    assert payload["error"]["tool"] == "memory_search"
    assert payload["error"]["limit_per_minute"] == 2
    assert 0 <= payload["error"]["retry_after_seconds"] <= 60
    assert "agent-a" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_per_tool_override_replaces_default_for_that_tool(
    dispatcher, monkeypatch
):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("MCP_RATE_LIMIT_memory_store", "2")

    for _ in range(2):
        result = await dispatcher.call_tool(
            "memory_store",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
        assert not _is_error(result)

    payload = _error_payload(
        await dispatcher.call_tool(
            "memory_store",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
    )
    assert payload["error"]["limit_per_minute"] == 2

    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
    )
    search_payload = _error_payload(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "agent-a"},
            fallback_key="session-a",
        )
    )
    assert search_payload["error"]["limit_per_minute"] == 1


@pytest.mark.asyncio
async def test_agent_windows_are_isolated(dispatcher, monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search", {"agent_id": "agent-a"}, fallback_key="session-a"
        )
    )
    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search", {"agent_id": "agent-b"}, fallback_key="session-a"
        )
    )
    assert _is_error(
        await dispatcher.call_tool(
            "memory_search", {"agent_id": "agent-a"}, fallback_key="session-a"
        )
    )


@pytest.mark.asyncio
async def test_transport_agent_hint_precedes_filter_argument(dispatcher, monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "filter-a"},
            fallback_key="session-a",
            agent_id_hint="agent-a",
        )
    )
    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "filter-b"},
            fallback_key="session-a",
            agent_id_hint="agent-b",
        )
    )
    assert _is_error(
        await dispatcher.call_tool(
            "memory_search",
            {"agent_id": "filter-c"},
            fallback_key="session-a",
            agent_id_hint="agent-a",
        )
    )


@pytest.mark.asyncio
async def test_missing_agent_id_falls_back_to_connection_without_crossing_agents(
    dispatcher, monkeypatch
):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not _is_error(
        await dispatcher.call_tool("memory_search", {}, fallback_key="connection-a")
    )
    assert not _is_error(
        await dispatcher.call_tool("memory_search", {}, fallback_key="connection-b")
    )
    assert _is_error(
        await dispatcher.call_tool("memory_search", {}, fallback_key="connection-a")
    )

    # A resolved agent id takes precedence over the connection fallback.
    assert not _is_error(
        await dispatcher.call_tool(
            "memory_search", {"agent_id": "agent-a"}, fallback_key="connection-a"
        )
    )


def test_window_boundary_is_sliding(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")
    limiter = ToolRateLimiter(max_keys=8)

    assert limiter.check("agent:agent-a", "memory_search", now=100.0).allowed
    assert limiter.check("agent:agent-a", "memory_search", now=159.0).allowed
    assert not limiter.check("agent:agent-a", "memory_search", now=159.9).allowed
    assert limiter.check("agent:agent-a", "memory_search", now=160.0).allowed
    assert not limiter.check("agent:agent-a", "memory_search", now=161.0).allowed


def test_concurrent_checks_do_not_over_admit(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "5")
    limiter = ToolRateLimiter(max_keys=8)

    def check(_index):
        return limiter.check("agent:agent-a", "memory_search", now=100.0).allowed

    with ThreadPoolExecutor(max_workers=32) as pool:
        allowed = list(pool.map(check, range(100)))

    assert allowed.count(True) == 5
    assert allowed.count(False) == 95


def test_key_state_is_bounded(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "10")
    limiter = ToolRateLimiter(max_keys=3)

    for index in range(20):
        limiter.check(f"agent:agent-{index}", "memory_search", now=100.0)

    assert limiter.bucket_count <= 3


@pytest.mark.asyncio
async def test_rate_limit_logging_does_not_leak_identity(
    dispatcher, monkeypatch, caplog
):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")
    caplog.set_level("WARNING")

    await dispatcher.call_tool(
        "memory_search", {"agent_id": "secret-agent-id"}, fallback_key="session-a"
    )
    await dispatcher.call_tool(
        "memory_search", {"agent_id": "secret-agent-id"}, fallback_key="session-a"
    )

    assert "secret-agent-id" not in caplog.text
