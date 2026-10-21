"""Rate limiting on the ``mcp-memory-server`` FastMCP entry point (#1606).

Its ``@mcp.tool`` functions call ``MemoryService`` directly and never go
through ``MemoryServer.call_tool``, so each one runs the limiter check
itself. These tests drive ``tools/call`` through the SDK's own request
handler with a request context holding an HTTP request, the way the
Streamable HTTP transport does. ``get_cache_stats`` reads module state only,
so no storage is opened. All but the default-unlimited test fail on ``main``.
"""

import json
from types import SimpleNamespace

import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from mcp_memory_service import mcp_server
from mcp_memory_service.utils.tool_rate_limit import ToolRateLimiter

TOOL = "get_cache_stats"


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Each test starts with empty windows and no rate-limit settings."""
    # raising=False: on main the attribute does not exist, and the tests should fail
    # on what the tools do (calls that should be denied go through), not on setup.
    monkeypatch.setattr(mcp_server, "_tool_rate_limiter", ToolRateLimiter(), raising=False)
    for var in (
        "MCP_RATE_LIMIT_PER_MINUTE",
        f"MCP_RATE_LIMIT_{TOOL}",
        f"MCP_RATE_LIMIT_{TOOL.upper()}",
    ):
        monkeypatch.delenv(var, raising=False)


def _request(agent_id=None, host="127.0.0.1"):
    headers = {} if agent_id is None else {"X-Agent-ID": agent_id}
    client = None if host is None else SimpleNamespace(host=host)
    return SimpleNamespace(headers=headers, client=client)


async def _call(agent_id=None, host="127.0.0.1", request=True, tool=TOOL, arguments=None):
    """Call a tool through the SDK's tools/call handler and return the CallToolResult."""
    handler = mcp_server.mcp._mcp_server.request_handlers[types.CallToolRequest]
    token = request_ctx.set(
        RequestContext(
            request_id=1,
            meta=None,
            session=None,
            lifespan_context=None,
            request=_request(agent_id, host) if request else None,
        )
    )
    try:
        result = await handler(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name=tool, arguments=arguments or {}),
            )
        )
    finally:
        request_ctx.reset(token)
    return result.root


def _rate_limit_error(result):
    """The rate-limit payload of a denied call, or None for any other result."""
    if not result.isError:
        return None
    text = result.content[0].text
    start = text.find("{")
    if start < 0:
        return None
    return json.loads(text[start:])["error"]


async def test_unlimited_by_default():
    for _ in range(5):
        result = await _call(agent_id="agent-a")
        assert not result.isError


async def test_third_call_from_one_agent_is_denied(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")

    assert not (await _call(agent_id="agent-a")).isError
    assert not (await _call(agent_id="agent-a")).isError
    denied = await _call(agent_id="agent-a")

    error = _rate_limit_error(denied)
    assert denied.isError
    assert error["code"] == -32029
    assert error["type"] == "rate_limit_exceeded"
    assert error["tool"] == TOOL
    assert error["limit_per_minute"] == 2
    assert error["retry_after_seconds"] >= 1
    # Another agent has its own window.
    assert not (await _call(agent_id="agent-b")).isError


async def test_per_tool_override_uses_this_servers_tool_name(monkeypatch):
    monkeypatch.setenv(f"MCP_RATE_LIMIT_{TOOL}", "1")

    assert not (await _call(agent_id="agent-a")).isError
    assert _rate_limit_error(await _call(agent_id="agent-a"))["limit_per_minute"] == 1


async def test_without_the_header_the_peer_ip_is_the_caller(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not (await _call(host="10.0.0.1")).isError
    assert _rate_limit_error(await _call(host="10.0.0.1")) is not None
    assert not (await _call(host="10.0.0.2")).isError


async def test_without_a_request_one_process_bucket(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    assert not (await _call(request=False)).isError
    assert _rate_limit_error(await _call(request=False)) is not None


def test_identity_never_comes_from_tool_arguments():
    """``agent_id`` on retrieve_memory/list_memories is an author filter, not the caller."""
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(request=_request(agent_id="real-caller"))
    )
    assert mcp_server._rate_limit_caller_identity(ctx) == "agent:real-caller"
    no_request = SimpleNamespace(request_context=SimpleNamespace(request=None))
    assert mcp_server._rate_limit_caller_identity(no_request) == "process:default"


async def test_limiter_failure_does_not_fail_the_call(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    def broken(*_args, **_kwargs):
        raise RuntimeError("limiter down")

    monkeypatch.setattr(mcp_server._tool_rate_limiter, "check", broken)
    for _ in range(3):
        assert not (await _call(agent_id="agent-a")).isError


@pytest.mark.parametrize(
    "tool",
    [
        "store_memory",
        "retrieve_memory",
        "search_by_tag",
        "delete_memory",
        "check_database_health",
        "list_memories",
        "memory_graph",
        "get_cache_stats",
    ],
)
async def test_every_tool_checks_before_it_runs(tool, monkeypatch):
    """A denied call never reaches the tool body, so no storage is needed."""
    calls = []

    def deny(_identity, tool_name, now=None):
        calls.append(tool_name)
        return _DENIED

    monkeypatch.setattr(mcp_server._tool_rate_limiter, "check", deny)
    error = _rate_limit_error(await _call(agent_id="agent-a", tool=tool, arguments=_ARGS.get(tool)))
    assert error is not None and error["tool"] == tool
    assert calls == [tool]


_DENIED = SimpleNamespace(allowed=False, limit_per_minute=1, retry_after_seconds=60, should_log=False)
_ARGS = {
    "store_memory": {"content": "x"},
    "retrieve_memory": {"query": "x"},
    "search_by_tag": {"tags": ["x"]},
    "delete_memory": {"content_hash": "x"},
    "memory_graph": {"action": "connected", "hash": "x"},
}
