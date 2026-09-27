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
