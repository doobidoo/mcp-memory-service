# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write scope on the SSE and Streamable HTTP transports (GHSA-9x7v-2vjj-xghg).

GHSA-2r68-g678-7qr3 put the per-tool 'write' check into the FastAPI shim
only. SSE and Streamable HTTP dispatch through the MCP SDK straight into
`MemoryServer.call_tool`, and `prompts/get learning_session` stores without
going through `call_tool` at all, so a token granted only 'read' could
store, update and delete memories on both transports.

These tests drive the SDK's own request handlers with a request context
that carries the scope `_credentials_authenticate` recorded, which is the
path both transports take.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

# Via the package, not mcp_memory_service.server_impl directly: the direct
# import races the package __init__ and raises a circular ImportError.
from mcp_memory_service.server import MemoryServer
from mcp_memory_service.utils import startup_orchestrator
from mcp_memory_service.web.oauth.middleware import AuthenticationResult

LEARNING_ARGS = {"topic": "scope-test", "key_points": "a, b", "questions": "c"}

# Valid arguments, so the SDK's schema validation passes and the request
# reaches the scope check, as a real attack would.
TOOL_ARGS = {
    "memory_store": {"content": "x"},
    "memory_update": {"content_hash": "h", "updates": {"tags": ["t"]}},
    "memory_delete": {"content_hash": "h"},
    "memory_search": {"query": "x"},
}


@pytest.fixture
def server(monkeypatch):
    """A MemoryServer with a mocked backend and a recording tool handler."""
    srv = MemoryServer()
    srv.remote_transport = True
    srv.storage = AsyncMock()
    srv.storage.store.return_value = (True, "stored")
    monkeypatch.setattr(srv, "_ensure_storage_initialized", AsyncMock())

    srv.handled = []

    async def handler(_server, arguments):
        srv.handled.append(arguments)
        return [types.TextContent(type="text", text="handled")]

    monkeypatch.setattr(
        "mcp_memory_service.tools.routing.resolve_handler", lambda name: handler
    )
    return srv


def _as_caller(scope):
    """Put a transport request into the SDK context, as SSE/Streamable HTTP do.

    `scope=None` means an authenticated request carries no recorded scope;
    use `_no_request()` for stdio.
    """
    asgi = {} if scope is None else {"mcp_auth_scope": scope}
    ctx = RequestContext(
        request_id=1, meta=None, session=None, lifespan_context=None,
        request=SimpleNamespace(scope=asgi),
    )
    return request_ctx.set(ctx)


async def _call_tool(server, name, scope):
    token = _as_caller(scope)
    try:
        handler = server.server.request_handlers[types.CallToolRequest]
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=TOOL_ARGS[name]),
        )
        return (await handler(req)).root
    finally:
        request_ctx.reset(token)


async def _get_prompt(server, scope, arguments=LEARNING_ARGS):
    token = _as_caller(scope)
    try:
        handler = server.server.request_handlers[types.GetPromptRequest]
        req = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="learning_session", arguments=arguments),
        )
        return await handler(req)
    finally:
        request_ctx.reset(token)


class TestToolCallScope:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["memory_store", "memory_update", "memory_delete"])
    async def test_read_token_cannot_write(self, server, tool):
        result = await _call_tool(server, tool, "read")
        assert result.isError is True
        assert "insufficient_scope" in result.content[0].text
        assert server.handled == [], "the handler must not run"

    @pytest.mark.asyncio
    async def test_read_token_can_search(self, server):
        result = await _call_tool(server, "memory_search", "read")
        assert not result.isError
        assert len(server.handled) == 1

    @pytest.mark.asyncio
    async def test_write_token_can_store(self, server):
        result = await _call_tool(server, "memory_store", "read write")
        assert not result.isError
        assert len(server.handled) == 1

    @pytest.mark.asyncio
    async def test_empty_scope_fails_closed(self, server):
        result = await _call_tool(server, "memory_store", "")
        assert result.isError is True
        assert server.handled == []

    @pytest.mark.asyncio
    async def test_unauthenticated_local_server_keeps_full_access(self, server):
        """No recorded scope: a loopback server with no auth configured."""
        result = await _call_tool(server, "memory_store", None)
        assert not result.isError
        assert len(server.handled) == 1

    @pytest.mark.asyncio
    async def test_direct_call_without_request_context_is_unchanged(self, server):
        """stdio and the FastAPI shim (which gates on its own) have no SDK request."""
        await server.call_tool("memory_store", {"content": "x"})
        assert len(server.handled) == 1


class TestLearningSessionPromptScope:

    @pytest.mark.asyncio
    async def test_read_token_cannot_store(self, server):
        with pytest.raises(PermissionError, match="insufficient_scope"):
            await _get_prompt(server, "read")
        server.storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_token_stores(self, server):
        await _get_prompt(server, "read write")
        server.storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_token_keeps_unpersisted_preview(self, server):
        result = await _get_prompt(server, "read", {"topic": "$1", "key_points": "$2", "questions": "$3"})
        assert "Preview only" in result.root.messages[0].content.text
        server.storage.store.assert_not_called()


class TestTransportRecordsScope:

    @pytest.fixture
    def oauth_only(self, monkeypatch):
        from mcp_memory_service import config
        from mcp_memory_service.web.oauth import middleware
        monkeypatch.setattr(config, "OAUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", None)

        async def bearer(token):
            if token == "read-token":
                return AuthenticationResult(True, client_id="c", scope="read", auth_method="oauth")
            return AuthenticationResult(False, error="invalid_token")

        monkeypatch.setattr(middleware, "authenticate_bearer_token", bearer)

    @pytest.mark.asyncio
    async def test_bearer_scope_lands_on_the_request(self, oauth_only):
        scope = {"headers": [(b"authorization", b"Bearer read-token")]}
        assert await startup_orchestrator._credentials_authenticate(scope) is True
        assert scope["mcp_auth_scope"] == "read"

    @pytest.mark.asyncio
    async def test_rejected_bearer_records_nothing(self, oauth_only):
        scope = {"headers": [(b"authorization", b"Bearer forged")]}
        assert await startup_orchestrator._credentials_authenticate(scope) is False
        assert "mcp_auth_scope" not in scope

    @pytest.mark.asyncio
    async def test_api_key_keeps_full_scope(self, monkeypatch):
        from mcp_memory_service import config
        monkeypatch.setattr(config, "OAUTH_ENABLED", False)
        monkeypatch.setattr(config, "API_KEY", "the-key")
        monkeypatch.setattr(
            "mcp_memory_service.web.oauth.middleware.API_KEY", "the-key", raising=False
        )
        scope = {"headers": [(b"x-api-key", b"the-key")]}
        assert await startup_orchestrator._credentials_authenticate(scope) is True
        assert "write" in scope["mcp_auth_scope"].split()
