"""Dispatcher-level per-caller rate limiting (#1096).

The limiter keys callers by the client-sent ``X-Agent-ID`` request header,
read from the transport layer -- never from tool arguments (for
``memory_search``/``memory_list`` the ``agent_id`` argument is an author
filter, not a caller identity).
"""

import json
from types import SimpleNamespace
from unittest.mock import PropertyMock

import pytest
from mcp import types

from mcp_memory_service.server import MemoryServer
from mcp_memory_service.utils.tool_rate_limit import (
    ToolRateLimiter,
    resolve_caller_identity,
    resolve_limit,
)


async def _echo_handler(_server, arguments):
    return [types.TextContent(type="text", text=json.dumps(arguments))]


@pytest.fixture
def dispatcher(monkeypatch):
    monkeypatch.setattr(
        "mcp_memory_service.tools.routing.resolve_handler",
        lambda _name: _echo_handler,
    )
    return MemoryServer()


def _clear_env(monkeypatch):
    for var in (
        "MCP_RATE_LIMIT_PER_MINUTE",
        "MCP_RATE_LIMIT_memory_search",
        "MCP_RATE_LIMIT_MEMORY_SEARCH",
    ):
        monkeypatch.delenv(var, raising=False)


def _is_rate_limited(result) -> bool:
    if isinstance(result, types.CallToolResult) and result.isError:
        try:
            err = json.loads(result.content[0].text).get("error", {})
        except Exception:
            return False
        return err.get("type") == "rate_limit_exceeded" and err.get("code") == -32029
    return False


@pytest.mark.asyncio
async def test_dispatcher_is_unlimited_by_default(dispatcher, monkeypatch):
    _clear_env(monkeypatch)
    for _ in range(50):
        result = await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
        assert not _is_rate_limited(result)


@pytest.mark.asyncio
async def test_dispatcher_enforces_per_agent_limit(dispatcher, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")

    for _ in range(2):
        result = await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
        assert not _is_rate_limited(result)
    assert _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
    )


@pytest.mark.asyncio
async def test_agent_windows_are_isolated(dispatcher, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
    )
    # agent-a is now at its limit; agent-b must be unaffected.
    assert _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
    )
    assert not _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:b"
        )
    )


@pytest.mark.asyncio
async def test_limiter_never_reads_tool_arguments(dispatcher, monkeypatch):
    """``arguments["agent_id"]`` is an author filter for search/list.

    Changing it must not change the rate-limit bucket; only the explicit
    caller identity does.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search",
            {"query": "x", "agent_id": "author-x"},
            caller_identity="agent:real-caller",
        )
    )
    # Same caller, different author filter -> still the same bucket.
    assert _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search",
            {"query": "x", "agent_id": "author-y"},
            caller_identity="agent:real-caller",
        )
    )
    # Different caller, same author filter -> fresh bucket.
    assert not _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search",
            {"query": "x", "agent_id": "author-x"},
            caller_identity="agent:other-caller",
        )
    )


@pytest.mark.asyncio
async def test_per_tool_override(dispatcher, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "10")
    monkeypatch.setenv("MCP_RATE_LIMIT_memory_search", "1")

    assert not _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
    )
    assert _is_rate_limited(
        await dispatcher.call_tool(
            "memory_search", {"query": "x"}, caller_identity="agent:a"
        )
    )


def _mock_sdk_request(monkeypatch, dispatcher, headers, client_host):
    client = SimpleNamespace(host=client_host) if client_host else None
    fake_request = SimpleNamespace(headers=headers, client=client, scope={})
    monkeypatch.setattr(
        type(dispatcher.server),
        "request_context",
        PropertyMock(return_value=SimpleNamespace(request=fake_request)),
    )


@pytest.mark.asyncio
async def test_sdk_path_resolves_identity_from_header(dispatcher, monkeypatch):
    """SDK transports (SSE / Streamable HTTP) expose the per-request ASGI
    request as ``server.request_context.request``; the header is read there."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")
    _mock_sdk_request(monkeypatch, dispatcher, {"X-Agent-ID": "sdk-agent"}, "10.0.0.5")

    assert not _is_rate_limited(
        await dispatcher.call_tool("memory_search", {"query": "x"})
    )
    assert _is_rate_limited(await dispatcher.call_tool("memory_search", {"query": "x"}))


@pytest.mark.asyncio
async def test_sdk_fallback_chain_header_then_ip_then_process(dispatcher, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "100")

    # No header -> socket peer IP.
    _mock_sdk_request(monkeypatch, dispatcher, {}, "10.0.0.9")
    assert dispatcher._rate_limit_caller_identity() == "ip:10.0.0.9"
    # Header beats IP.
    _mock_sdk_request(
        monkeypatch, dispatcher, {"X-Agent-ID": "  hdr-agent "}, "10.0.0.9"
    )
    assert dispatcher._rate_limit_caller_identity() == "agent:hdr-agent"
    # No header, no client -> single per-process bucket.
    _mock_sdk_request(monkeypatch, dispatcher, {}, None)
    assert dispatcher._rate_limit_caller_identity() == "process:default"


def test_resolve_caller_identity_unit():
    assert resolve_caller_identity("  a1 ", "1.2.3.4") == "agent:a1"
    assert resolve_caller_identity("", "1.2.3.4") == "ip:1.2.3.4"
    assert resolve_caller_identity(None, "1.2.3.4") == "ip:1.2.3.4"
    assert resolve_caller_identity(None, None) == "process:default"
    assert resolve_caller_identity("   ", None) == "process:default"


def test_resolve_limit_defaults_to_unlimited(monkeypatch):
    _clear_env(monkeypatch)
    assert resolve_limit("memory_search") is None
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "0")
    assert resolve_limit("memory_search") is None
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "7")
    assert resolve_limit("memory_search") == 7


def test_eviction_prefers_windows_without_live_hits(monkeypatch):
    """An eviction must not reset an active window while a dead one exists."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")
    limiter = ToolRateLimiter(max_keys=2, window_seconds=60)

    assert limiter.check("agent:a", "tool", now=1000.0).allowed
    assert limiter.check("agent:a", "tool", now=1000.0).allowed
    assert limiter.check("agent:b", "tool", now=900.0).allowed  # dead by t=1000
    assert limiter.check("agent:c", "tool", now=1000.0).allowed  # evicts b, not a
    assert not limiter.check("agent:a", "tool", now=1000.0).allowed


def test_new_window_is_never_evicted_itself(monkeypatch):
    """A new identity past capacity must be tracked, not given a free pass."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")
    limiter = ToolRateLimiter(max_keys=1, window_seconds=60)

    assert limiter.check("agent:a", "tool", now=1000.0).allowed
    # New identity evicts a's live window but its own window is retained:
    # the second call from b must be denied.
    assert limiter.check("agent:b", "tool", now=1000.0).allowed
    assert not limiter.check("agent:b", "tool", now=1000.0).allowed


def test_invalid_env_value_does_not_inject_log_lines(monkeypatch, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "not-a-number")
    with caplog.at_level("WARNING", logger="mcp_memory_service.utils.tool_rate_limit"):
        assert resolve_limit("memory_search\nINJECTED") is None
    assert not any("INJECTED" in r.getMessage() for r in caplog.records)
