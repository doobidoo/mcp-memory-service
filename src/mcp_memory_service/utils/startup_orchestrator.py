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

"""
Server startup orchestration utilities.

Extracted from server_impl.py Phase 3.2 refactoring to reduce async_main complexity.
This module provides orchestrator classes for:
- Startup validation checks
- Initialization with retry logic
- Server execution mode management
"""

import os
import sys
import asyncio
import logging
import traceback
from typing import Any

# Import necessary functions and constants
from ..server.client_detection import MCP_CLIENT
from ..config import SERVER_NAME, SERVER_VERSION, MCP_SSE_HOST, MCP_SSE_PORT, MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE, MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN
from ..lm_studio_compat import patch_mcp_for_lm_studio, add_windows_timeout_handling
from ..dependency_check import run_dependency_check
from ..server.environment import check_uv_environment, check_version_consistency

# MCP imports
import mcp.server.stdio
from mcp.server import InitializationOptions, NotificationOptions

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


def _is_loopback_host(host: str) -> bool:
    """True when `host` can only be reached from the machine itself."""
    import ipaddress  # inline import: only needed by this one helper, keeps module import cheap
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an IP literal. That covers a hostname we cannot resolve here, and
        # also the empty string, which binds every interface rather than
        # loopback. The safe answer to "is this only reachable locally" is no.
        return False


def _assert_bind_is_authenticated(host: str, port: int) -> None:
    """Refuse to serve MCP to the network with no authentication configured.

    The default bind is loopback, so this only fires when someone
    deliberately widened it without setting up auth, which is exactly the
    deployment GHSA-2hh8-qjxc-43x3 describes. Documenting "do not expose SSE"
    is not a control. Refusing to start is.
    """
    if _is_loopback_host(host) or _auth_configured():
        return
    raise RuntimeError(
        f"Refusing to start an MCP transport on {host}:{port} with no authentication "
        "configured. Its endpoints expose the full MCP tool surface. Set MCP_API_KEY, "
        "or enable OAuth with MCP_OAUTH_ENABLED=true, or bind to 127.0.0.1."
    )


def _auth_configured() -> bool:
    """True when the HTTP transports check a credential on every MCP request."""
    from ..config import OAUTH_ENABLED, API_KEY
    return bool(OAUTH_ENABLED or API_KEY)


def _configured_cors_origins() -> list:
    return [o.strip() for o in os.environ.get("MCP_CORS_ORIGINS", "").split(",") if o.strip()]


_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _transport_security_settings():
    """Host/Origin validation for the SSE and Streamable HTTP transports.

    With no auth configured the server is loopback-only (see
    _assert_bind_is_authenticated), and the browser is the only way a remote
    party can reach it: a cross-origin page, or a DNS-rebinding domain that
    resolves to 127.0.0.1 (GHSA-8p26-g7wp-65jw). Both are refused here by
    pinning Host to loopback names and Origin to loopback plus
    MCP_CORS_ORIGINS. The SDK leaves this off unless it is passed in.

    With auth configured this returns None on purpose: every request needs a
    credential a browser never attaches on its own, and a server behind a
    reverse proxy sees the public hostname in Host, which a loopback
    allowlist would answer with 421. Gate on auth, not on the bind address.
    """
    if _auth_configured():
        return None
    from mcp.server.transport_security import TransportSecuritySettings  # inline import: only the HTTP transports need it
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(_LOOPBACK_HOSTS),
        allowed_origins=_LOOPBACK_ORIGINS + _configured_cors_origins(),
    )


def _origin_matches(origin: str, allowed: list) -> bool:
    """Same rule as the SDK's TransportSecurityMiddleware: exact, or `base:*`."""
    return any(
        origin == a or (a.endswith(":*") and origin.startswith(a[:-1]))
        for a in allowed
    )


def _mcp_preflight_headers(origin: str) -> dict:
    """CORS headers for an OPTIONS preflight on /mcp.

    An origin is granted only when the transport would also accept its
    request, so the preflight and the SDK's Origin check cannot disagree.
    Until GHSA-8p26-g7wp-65jw this answered every origin with `*`, which
    let any web page drive an unauthenticated local server.
    """
    configured = _configured_cors_origins()
    if _auth_configured() and (not configured or "*" in configured):
        # Every request still needs a token the page does not have. Without
        # auth an explicit `*` is not honoured: that is the advisory itself.
        allow = "*"
    else:
        settings = _transport_security_settings()
        allowed = settings.allowed_origins if settings else configured
        allow = origin if origin and _origin_matches(origin, allowed) else None
    if allow is None:
        return {}
    headers = {
        "Access-Control-Allow-Origin": allow,
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",
    }
    if allow != "*":
        headers["Vary"] = "Origin"
    return headers


async def _serve_lifespan(receive, send, context=None) -> None:
    """ASGI lifespan loop shared by the HTTP transports.

    `context` is an async context manager held open from startup to shutdown
    (the Streamable HTTP session manager); SSE passes none.
    """
    entered = False
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            if context is not None:
                await context.__aenter__()
                entered = True
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            if entered:
                await context.__aexit__(None, None, None)
            await send({"type": "lifespan.shutdown.complete"})
            return


async def _answer_health_or_404(path: str, scope, receive, send) -> None:
    from starlette.responses import Response  # inline import: only the HTTP transports need it
    if path == "/health":
        response = Response('{"status":"ok"}', media_type="application/json")
    else:
        response = Response("Not Found", status_code=404)
    await response(scope, receive, send)


async def _handle_mcp_request(session_manager, scope, receive, send) -> None:
    """Serve one request on /mcp: CORS preflight, auth, then the transport."""
    from starlette.responses import Response  # inline import: only the HTTP transports need it
    from ..utils.agent_context import set_current_agent_id, reset_current_agent_id
    headers = dict(scope.get("headers", []))
    if scope.get("method") == "OPTIONS":
        origin = headers.get(b"origin", b"").decode("latin-1")
        response = Response("", status_code=204, headers=_mcp_preflight_headers(origin))
        await response(scope, receive, send)
        return
    if _auth_configured() and not await check_transport_auth(scope, receive, send):
        return
    # X-Agent-ID is authorship ATTRIBUTION, not authentication: it names the
    # writer and grants no permission. Bind it per-request into a ContextVar so
    # the single resolution point in memory_service.store_memory can honor it
    # (precedence arg > header > MCP_AGENT_ID env). Reset in finally so
    # identity never leaks between concurrent requests on the same worker.
    agent_header = headers.get(b"x-agent-id")
    agent_id = agent_header.decode("latin-1").strip() if agent_header else None
    token = set_current_agent_id(agent_id or None)
    try:
        await session_manager.handle_request(scope, receive, send)
    finally:
        reset_current_agent_id(token)


def _build_oauth_app(logger):
    """OAuth 2.1 endpoints (discovery, DCR, authorization, token) for /mcp."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from ..web.oauth.discovery import router as discovery_router
    from ..web.oauth.registration import router as registration_router
    from ..web.oauth.authorization import router as authorization_router
    from ..config import OAUTH_ISSUER

    oauth_app = FastAPI(title="MCP Memory OAuth", docs_url=None, redoc_url=None)
    cors_origins = _configured_cors_origins()
    oauth_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        # allow_credentials requires explicit origins; wildcard + credentials
        # is rejected by browsers (CORS spec §3.2.2).
        allow_credentials=bool(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    oauth_app.include_router(discovery_router)
    oauth_app.include_router(registration_router, prefix="/oauth")
    oauth_app.include_router(authorization_router, prefix="/oauth")

    # RFC 9728: OAuth Protected Resource Metadata
    @oauth_app.get("/.well-known/oauth-protected-resource")
    @oauth_app.get("/.well-known/oauth-protected-resource/{path:path}")
    async def protected_resource_metadata(path: str = ""):
        return JSONResponse({
            "resource": OAUTH_ISSUER,
            "authorization_servers": [OAUTH_ISSUER],
            "scopes_supported": ["read", "write", "admin"],
            "bearer_methods_supported": ["header"],
        })

    logger.info("OAuth 2.1 endpoints enabled on Streamable HTTP transport")
    return oauth_app


async def _credentials_authenticate(scope) -> bool:
    """True when the request carries a credential this server accepts.

    Three ways in, tried in order: an OAuth bearer token, an X-API-Key
    header, and a bearer token that is really the API key (which some MCP
    clients send that way).

    On success the granted scope is recorded as `scope["mcp_auth_scope"]`,
    so `MemoryServer` can refuse write tools to a read-only token on this
    message (GHSA-9x7v-2vjj-xghg). Authenticating alone did not do that.
    """
    from ..config import OAUTH_ENABLED, API_KEY
    from ..web.oauth.middleware import (
        authenticate_bearer_token,
        authenticate_api_key,
    )

    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode("latin-1")
    api_key_header = headers.get(b"x-api-key", b"").decode("latin-1")

    is_bearer = auth_header.lower().startswith("bearer ")
    token = auth_header[7:] if is_bearer else ""

    result = None
    if is_bearer and OAUTH_ENABLED:
        result = await authenticate_bearer_token(token)
    if not (result and result.authenticated) and api_key_header and API_KEY:
        result = authenticate_api_key(api_key_header)
    if not (result and result.authenticated) and is_bearer and API_KEY:
        result = authenticate_api_key(token)
    if not (result and result.authenticated):
        return False
    scope["mcp_auth_scope"] = result.scope or ""
    return True


async def _send_unauthorized(scope, receive, send) -> None:
    """Answer 401 with the WWW-Authenticate header OAuth clients expect."""
    from starlette.responses import Response as StarletteResponse
    from ..config import OAUTH_ENABLED, OAUTH_ISSUER

    resource_metadata_url = f"{OAUTH_ISSUER.rstrip('/')}/.well-known/oauth-protected-resource"
    www_auth = (
        f'Bearer resource_metadata="{resource_metadata_url}"'
        if OAUTH_ENABLED and OAUTH_ISSUER else "Bearer"
    )
    response = StarletteResponse(
        '{"error":"unauthorized","error_description":"Valid Bearer token or API key required"}',
        status_code=401,
        headers={"WWW-Authenticate": www_auth, "Content-Type": "application/json"},
    )
    await response(scope, receive, send)


async def _sse_request_is_allowed(path: str, scope, receive, send) -> bool:
    """Gate the two SSE endpoints that carry the MCP surface.

    `/sse` and `/messages/` both reach the full tool list, so both are
    gated; `/health` and unknown paths are not, and are answered by the
    caller. Until 2026-09-05 none of them was gated at all
    (GHSA-2hh8-qjxc-43x3).

    Returns False only when it has already answered the request with 401.
    """
    if path != "/sse" and not path.startswith("/messages/"):
        return True
    if not _auth_configured():
        # Nothing configured to check against. _assert_bind_is_authenticated()
        # has already refused a non-loopback bind in this case, so what is left
        # is a deliberate local-only server.
        return True
    return await check_transport_auth(scope, receive, send)


async def check_transport_auth(scope, receive, send) -> bool:
    """Validate auth on an ASGI request. Returns True if authorized, else
    answers 401 itself and returns False.

    Shared by the Streamable HTTP and SSE transports. It lived inside
    run_streamable_http() until 2026-09-05, which is part of why SSE served
    /sse and /messages/ with no auth at all (GHSA-2hh8-qjxc-43x3): the check
    was not reachable from there. A second copy would have drifted, so it is
    one function now.
    """
    if await _credentials_authenticate(scope):
        return True
    await _send_unauthorized(scope, receive, send)
    return False


class StartupCheckOrchestrator:
    """Orchestrate startup validation checks."""

    @staticmethod
    def run_all_checks() -> None:
        """Run all startup checks in sequence."""
        # Apply LM Studio compatibility patch
        patch_mcp_for_lm_studio()

        # Add Windows-specific timeout handling
        add_windows_timeout_handling()

        # Run dependency check
        run_dependency_check()

        # Check if running with UV
        check_uv_environment()

        # Check for version mismatch
        check_version_consistency()


class InitializationRetryManager:
    """Manage server initialization with timeout and retry logic."""

    def __init__(self, max_retries: int = 2, timeout: float = 30.0, retry_delay: float = 2.0):
        """
        Initialize retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each initialization attempt
            retry_delay: Delay in seconds between retry attempts
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)

    async def initialize_with_retry(self, server: 'MemoryServer') -> bool:
        """
        Initialize server with timeout and retry logic.

        Args:
            server: MemoryServer instance to initialize

        Returns:
            True if initialization succeeded, False otherwise
        """
        retry_count = 0
        init_success = False

        while retry_count <= self.max_retries and not init_success:
            if retry_count > 0:
                self.logger.warning("Retrying initialization (attempt %s/%s)...", retry_count, self.max_retries)

            init_task = asyncio.create_task(server.initialize())
            try:
                # Timeout for initialization
                init_success = await asyncio.wait_for(init_task, timeout=self.timeout)
                if init_success:
                    self.logger.info("Async initialization completed successfully")
                else:
                    self.logger.warning("Initialization returned failure status")
                    retry_count += 1
            except asyncio.TimeoutError:
                self.logger.warning("Async initialization timed out. Continuing with server startup.")
                # Don't cancel the task, let it complete in the background
                break
            except Exception as init_error:
                self.logger.error("Initialization error: %s", _sanitize_log_value(init_error))
                self.logger.error(traceback.format_exc())
                retry_count += 1

                if retry_count <= self.max_retries:
                    self.logger.info("Waiting %s seconds before retry...", self.retry_delay)
                    await asyncio.sleep(self.retry_delay)

        return init_success


class ServerRunManager:
    """Manage server execution modes and lifecycle."""

    def __init__(self, server: 'MemoryServer', system_info: Any):
        """
        Initialize server run manager.

        Args:
            server: MemoryServer instance to manage
            system_info: System information object (from get_system_info)
        """
        self.server = server
        self.system_info = system_info
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def is_streamable_http_mode() -> bool:
        """Check if running in Streamable HTTP transport mode."""
        return os.environ.get('MCP_STREAMABLE_HTTP_MODE', '').lower() == '1'

    @staticmethod
    def is_sse_mode() -> bool:
        """Check if running in SSE transport mode."""
        return os.environ.get('MCP_SSE_MODE', '').lower() == '1'

    @staticmethod
    def is_standalone_mode() -> bool:
        """Check if running in standalone mode."""
        standalone_mode = os.environ.get('MCP_STANDALONE_MODE', '').lower() == '1'
        return standalone_mode

    @staticmethod
    def is_docker_environment() -> bool:
        """Check if running in Docker."""
        return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)

    async def run_standalone(self) -> None:
        """Run server in standalone mode (Docker without active client)."""
        self.logger.info("Running in standalone mode - keeping server alive without active client")
        if MCP_CLIENT == 'lm_studio':
            print("MCP Memory Service running in standalone mode", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel

        # Keep the server running indefinitely
        try:
            while True:
                await asyncio.sleep(60)  # Sleep for 60 seconds at a time
                self.logger.debug("Standalone server heartbeat")
        except asyncio.CancelledError:
            self.logger.info("Standalone server cancelled")
            raise

    async def run_stdio(self) -> None:
        """Run server with stdio communication."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            self.logger.info("Server started and ready to handle requests")

            if self.is_docker_environment():
                self.logger.info("Detected Docker environment - ensuring proper stdio handling")
                if MCP_CLIENT == 'lm_studio':
                    print("MCP Memory Service running in Docker container", file=sys.stderr, flush=True)  # debug: inside run_stdio, so stdout is the JSON-RPC channel it would corrupt

            try:
                await self.server.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=SERVER_NAME,
                        server_version=SERVER_VERSION,
                        protocol_version="2024-11-05",
                        capabilities=self.server.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={
                                "hardware_info": {
                                    "architecture": self.system_info.architecture,
                                    "accelerator": self.system_info.accelerator,
                                    "memory_gb": self.system_info.memory_gb,
                                    "cpu_count": self.system_info.cpu_count
                                }
                            },
                        ),
                    ),
                )
            except asyncio.CancelledError:
                self.logger.info("Server run cancelled")
                raise
            except Exception as e:
                self._handle_server_exception(e)
            finally:
                self.logger.info("Server run completed")

    async def run_sse(self) -> None:
        """Run server with SSE (Server-Sent Events) transport over HTTP."""
        from mcp.server.sse import SseServerTransport

        init_options = InitializationOptions(
            server_name=SERVER_NAME,
            server_version=SERVER_VERSION,
            protocol_version="2024-11-05",
            capabilities=self.server.server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={
                    "hardware_info": {
                        "architecture": self.system_info.architecture,
                        "accelerator": self.system_info.accelerator,
                        "memory_gb": self.system_info.memory_gb,
                        "cpu_count": self.system_info.cpu_count
                    }
                },
            ),
        )

        # Remote transport: filesystem tools are filtered out of tools/list and
        # refused on tools/call. See MemoryServer.local_only_tools().
        self.server.remote_transport = True

        sse = SseServerTransport("/messages/", security_settings=_transport_security_settings())
        server_instance = self.server.server

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                await _serve_lifespan(receive, send)
                return

            path = scope.get("path", "")
            if not await _sse_request_is_allowed(path, scope, receive, send):
                return

            if path == "/sse":
                async with sse.connect_sse(scope, receive, send) as streams:
                    await server_instance.run(
                        streams[0],
                        streams[1],
                        init_options,
                    )
            elif path.startswith("/messages/"):
                await sse.handle_post_message(scope, receive, send)
            else:
                await _answer_health_or_404(path, scope, receive, send)

        await self._serve_http(app, "SSE")

    async def _serve_http(self, app, label: str) -> None:
        """Bind and serve an HTTP transport app with uvicorn."""
        import uvicorn  # inline import: heavy, and only the HTTP transports need it
        # Re-read env at runtime so late-set values (e.g. from CLI flags in
        # cli/main.py after package __init__ has already frozen config.py
        # module constants) still take effect.
        from ..config import safe_get_int_env
        sse_host = os.environ.get('MCP_SSE_HOST', MCP_SSE_HOST)
        sse_port = safe_get_int_env('MCP_SSE_PORT', MCP_SSE_PORT, min_value=1024, max_value=65535)
        # Both transports serve the full tool surface, so neither may bind to
        # the network without auth (GHSA-2hh8-qjxc-43x3).
        _assert_bind_is_authenticated(sse_host, sse_port)

        self.logger.info("Starting %s transport on %s:%s", label, _sanitize_log_value(sse_host), sse_port)
        config = uvicorn.Config(
            app,
            host=sse_host,
            port=sse_port,
            log_level="info",
            timeout_keep_alive=MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE,
            timeout_graceful_shutdown=MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN,
        )
        await uvicorn.Server(config).serve()

    async def run_streamable_http(self) -> None:
        """Run server with Streamable HTTP transport.

        Uses StreamableHTTPSessionManager for the MCP protocol transport,
        which is what Claude.ai and modern MCP clients expect.

        When OAuth is enabled (MCP_OAUTH_ENABLED=true), this also serves
        OAuth 2.1 endpoints (discovery, DCR, authorization, token) alongside
        the MCP transport endpoint, and requires Bearer token auth on /mcp.
        """
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from ..config import OAUTH_ENABLED

        # Remote transport: filesystem tools are filtered out of tools/list and
        # refused on tools/call. See MemoryServer.local_only_tools().
        self.server.remote_transport = True

        server_instance = self.server.server
        session_manager = StreamableHTTPSessionManager(
            app=server_instance,
            event_store=None,
            stateless=True,
            security_settings=_transport_security_settings(),
        )

        oauth_app = _build_oauth_app(self.logger) if OAUTH_ENABLED else None

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                await _serve_lifespan(receive, send, session_manager.run())
                return

            path = scope.get("path", "")
            if path in ("/mcp", "/mcp/"):
                await _handle_mcp_request(session_manager, scope, receive, send)
            elif oauth_app and path.startswith(("/.well-known/", "/oauth/")):
                await oauth_app(scope, receive, send)
            else:
                await _answer_health_or_404(path, scope, receive, send)

        await self._serve_http(app, "Streamable HTTP")

    def _handle_server_exception(self, e: BaseException) -> None:
        """Handle exceptions during server run."""
        # Handle ExceptionGroup specially (Python 3.11+)
        if type(e).__name__ == 'ExceptionGroup' or 'ExceptionGroup' in str(type(e)):
            error_str = str(e)
            # Check if this contains the LM Studio cancelled notification error
            if 'notifications/cancelled' in error_str or 'ValidationError' in error_str:
                self.logger.info("LM Studio sent a cancelled notification - this is expected behavior")
                self.logger.debug("Full error for debugging: %s", _sanitize_log_value(error_str))
                # Don't re-raise - just continue gracefully
            else:
                self.logger.error("ExceptionGroup in server.run: %s", _sanitize_log_value(e))
                self.logger.error(traceback.format_exc())
                raise
        else:
            self.logger.error("Error in server.run: %s", _sanitize_log_value(e))
            self.logger.error(traceback.format_exc())
            raise
