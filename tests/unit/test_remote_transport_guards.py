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

"""Guards for the two transport-level advisories fixed on 2026-09-05.

GHSA-7crr-2r7w-cpfm: `local_only_tools()` was applied only by the FastAPI
JSON-RPC shim, so the Streamable HTTP and SSE transports served the
filesystem tools (`memory_harvest`, `memory_ingest`) to remote callers.

GHSA-2hh8-qjxc-43x3: the SSE transport served `/sse` and `/messages/` with
no authentication at all, and would happily do so on a non-loopback bind.

The point of testing at the MemoryServer level rather than per transport is
that a future transport inherits the guard instead of having to remember it.
That is precisely what went wrong the first time.
"""

import json

import pytest

# Via the package, not mcp_memory_service.server_impl directly: the direct
# import races the package __init__ and raises a circular ImportError.
from mcp_memory_service.server import MemoryServer
from mcp_memory_service.utils.startup_orchestrator import _is_loopback_host


LOCAL_ONLY = ("memory_harvest", "memory_ingest")


@pytest.fixture
def server():
    """A MemoryServer that never touches storage.

    `list_tools()` reads the declarative registry and `call_tool()` refuses
    local-only names before any handler resolves, so neither path needs a
    backend for what is asserted here.
    """
    return MemoryServer()


class TestLocalOnlyToolsOverRemoteTransports:

    def test_defaults_to_local(self, server):
        """stdio is the default and keeps the filesystem tools."""
        assert server.remote_transport is False

    @pytest.mark.asyncio
    async def test_stdio_lists_local_only_tools(self, server):
        names = {t.name for t in await server.list_tools()}
        for tool in LOCAL_ONLY:
            assert tool in names, f"{tool} should be available over stdio"

    @pytest.mark.asyncio
    async def test_remote_hides_local_only_tools(self, server):
        server.remote_transport = True
        names = {t.name for t in await server.list_tools()}
        for tool in LOCAL_ONLY:
            assert tool not in names, f"{tool} must not be listed to a remote caller"
        # The filter must not empty the list; the other tools still ship.
        assert len(names) > 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", LOCAL_ONLY)
    async def test_remote_refuses_direct_call(self, server, tool):
        """Hiding a tool from tools/list is not enough on its own.

        A caller can name the tool directly, so the refusal has to sit in
        call_tool() as well, and it has to happen before the handler
        resolves.
        """
        server.remote_transport = True
        result = await server.call_tool(tool, {"project_path": "/etc"})
        payload = json.loads(result[0].text)
        assert "error" in payload
        assert "not available over this transport" in payload["error"]

    @pytest.mark.asyncio
    async def test_remote_still_serves_other_tools(self, server):
        """The guard is scoped to the local-only set, not a blanket block."""
        server.remote_transport = True
        assert server._reject_local_only("memory_search") is False
        assert server._reject_local_only("memory_harvest") is True


class TestLoopbackDetection:
    """`_is_loopback_host` decides whether SSE may start without auth, so an
    over-permissive answer here is the whole vulnerability."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.53"])
    def test_loopback(self, host):
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "10.0.0.1", "::", "example.com"])
    def test_not_loopback(self, host):
        assert _is_loopback_host(host) is False

    def test_bind_guard_refuses_open_network_bind(self, monkeypatch):
        """Binding to the network with no auth configured must not start."""
        import mcp_memory_service.config as config  # inline import: patched per test
        from mcp_memory_service.utils import startup_orchestrator as so

        monkeypatch.setattr(config, "OAUTH_ENABLED", False)
        monkeypatch.setattr(config, "API_KEY", None)
        with pytest.raises(RuntimeError, match="no authentication configured"):
            so._assert_bind_is_authenticated("0.0.0.0", 8000)

    def test_bind_guard_allows_loopback_without_auth(self, monkeypatch):
        import mcp_memory_service.config as config  # inline import: patched per test
        from mcp_memory_service.utils import startup_orchestrator as so

        monkeypatch.setattr(config, "OAUTH_ENABLED", False)
        monkeypatch.setattr(config, "API_KEY", None)
        so._assert_bind_is_authenticated("127.0.0.1", 8000)

    def test_bind_guard_allows_network_bind_with_api_key(self, monkeypatch):
        import mcp_memory_service.config as config  # inline import: patched per test
        from mcp_memory_service.utils import startup_orchestrator as so

        monkeypatch.setattr(config, "OAUTH_ENABLED", False)
        monkeypatch.setattr(config, "API_KEY", "a-key")
        so._assert_bind_is_authenticated("0.0.0.0", 8000)

    def test_empty_host_is_not_loopback(self):
        """An empty bind address means every interface, not loopback.

        Worth its own test because the obvious implementation groups "" with
        "localhost" as a falsy-ish default, which fails open.
        """
        assert _is_loopback_host("") is False


class TestFastMCPEntryPointBind:
    """`mcp-memory-server` (mcp_server.main) builds FastMCP with no auth, so it
    must never bind beyond loopback, whatever auth is configured
    (GHSA-26rx-6fvr-qjqg)."""

    @pytest.fixture
    def runs(self, monkeypatch):
        """Calls to mcp.run(), with every form of auth configured."""
        import mcp_memory_service.config as config  # inline import: patched per test
        mcp_server = pytest.importorskip("mcp_memory_service.mcp_server")
        monkeypatch.setattr(config, "API_KEY", "a-key")
        monkeypatch.setattr(config, "OAUTH_ENABLED", True)
        calls = []
        monkeypatch.setattr(mcp_server.mcp, "run", lambda *a, **k: calls.append(a))
        return calls

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", ""])
    def test_refuses_network_bind_even_with_auth(self, runs, monkeypatch, host):
        """Configured auth must not let it through: nothing here checks it."""
        from mcp_memory_service import mcp_server
        monkeypatch.setattr(mcp_server.mcp.settings, "host", host)
        with pytest.raises(RuntimeError, match="Refusing to start mcp-memory-server"):
            mcp_server.main()
        assert runs == []

    def test_starts_on_loopback(self, runs, monkeypatch):
        from mcp_memory_service import mcp_server
        monkeypatch.setattr(mcp_server.mcp.settings, "host", "127.0.0.1")
        mcp_server.main()
        assert runs == [("streamable-http",)]

    def test_guard_reads_the_address_uvicorn_binds(self, runs):
        """FastMCP's run_streamable_http_async binds settings.host, which is the
        value main() checks; it has to come from MCP_HTTP_HOST."""
        from mcp_memory_service import mcp_server
        assert mcp_server.mcp.settings.host == mcp_server.HTTP_HOST


class TestBrowserReachability:
    """GHSA-8p26-g7wp-65jw: an unauthenticated loopback server was reachable
    from any web page (wildcard CORS preflight) and through DNS rebinding (no
    Host/Origin check). These pin the allowlists both paths now share."""

    @pytest.fixture
    def so(self, monkeypatch):
        import mcp_memory_service.config as config  # inline import: patched per test
        from mcp_memory_service.utils import startup_orchestrator as so
        monkeypatch.setattr(config, "OAUTH_ENABLED", False)
        monkeypatch.setattr(config, "API_KEY", None)
        monkeypatch.delenv("MCP_CORS_ORIGINS", raising=False)
        return so

    async def _validate(self, settings, host, origin=None):
        """Run the SDK's own request check, the one the transports apply."""
        from mcp.server.transport_security import TransportSecurityMiddleware
        from starlette.requests import Request
        headers = [(b"host", host.encode()), (b"content-type", b"application/json")]
        if origin:
            headers.append((b"origin", origin.encode()))
        req = Request({"type": "http", "method": "POST", "path": "/mcp", "headers": headers})
        resp = await TransportSecurityMiddleware(settings).validate_request(req, is_post=True)
        return resp.status_code if resp else 200

    @pytest.mark.asyncio
    async def test_unauthenticated_server_rejects_rebinding_host(self, so):
        assert await self._validate(so._transport_security_settings(), "evil.example:8765") == 421

    @pytest.mark.asyncio
    async def test_unauthenticated_server_rejects_foreign_origin(self, so):
        status = await self._validate(so._transport_security_settings(), "127.0.0.1:8765", "https://evil.example")
        assert status == 403

    @pytest.mark.asyncio
    async def test_unauthenticated_server_serves_local_clients(self, so):
        settings = so._transport_security_settings()
        assert await self._validate(settings, "127.0.0.1:8765") == 200
        assert await self._validate(settings, "localhost:8765", "http://localhost:6274") == 200

    def test_authenticated_server_leaves_host_unchecked(self, so, monkeypatch):
        """Behind a reverse proxy Host is the public name; the token is the gate."""
        import mcp_memory_service.config as config  # inline import: patched per test
        monkeypatch.setattr(config, "API_KEY", "a-key")
        assert so._transport_security_settings() is None

    def test_preflight_grants_no_foreign_origin(self, so):
        assert so._mcp_preflight_headers("https://evil.example") == {}
        assert so._mcp_preflight_headers("") == {}

    def test_preflight_reflects_local_origin(self, so):
        headers = so._mcp_preflight_headers("http://localhost:6274")
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:6274"
        assert headers["Vary"] == "Origin"

    def test_configured_origin_is_allowed_on_both_paths(self, so, monkeypatch):
        monkeypatch.setenv("MCP_CORS_ORIGINS", "https://app.example")
        assert "https://app.example" in so._transport_security_settings().allowed_origins
        assert so._mcp_preflight_headers("https://app.example")["Access-Control-Allow-Origin"] == "https://app.example"
        assert so._mcp_preflight_headers("https://evil.example") == {}

    def test_authenticated_server_keeps_wildcard_preflight(self, so, monkeypatch):
        import mcp_memory_service.config as config  # inline import: patched per test
        monkeypatch.setattr(config, "API_KEY", "a-key")
        assert so._mcp_preflight_headers("https://anything.example")["Access-Control-Allow-Origin"] == "*"
        monkeypatch.setenv("MCP_CORS_ORIGINS", "*")
        assert so._mcp_preflight_headers("https://anything.example")["Access-Control-Allow-Origin"] == "*"

    def test_explicit_wildcard_does_not_open_unauthenticated_server(self, so, monkeypatch):
        monkeypatch.setenv("MCP_CORS_ORIGINS", "*")
        assert so._mcp_preflight_headers("https://evil.example") == {}
