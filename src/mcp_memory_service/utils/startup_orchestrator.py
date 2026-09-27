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
    from ..config import OAUTH_ENABLED, API_KEY
    if _is_loopback_host(host) or OAUTH_ENABLED or API_KEY:
        return
    raise RuntimeError(
        f"Refusing to start an MCP transport on {host}:{port} with no authentication "
        "configured. Its endpoints expose the full MCP tool surface. Set MCP_API_KEY, "
        "or enable OAuth with MCP_OAUTH_ENABLED=true, or bind to 127.0.0.1."
    )


async def _credentials_authenticate(scope) -> bool:
    """True when the request carries a credential this server accepts.

    Three ways in, tried in order: an OAuth bearer token, an X-API-Key
    header, and a bearer token that is really the API key (which some MCP
    clients send that way).
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

    if is_bearer and OAUTH_ENABLED and (await authenticate_bearer_token(token)).authenticated:
        return True
    if api_key_header and API_KEY and authenticate_api_key(api_key_header).authenticated:
        return True
    if is_bearer and API_KEY and authenticate_api_key(token).authenticated:
        return True
    return False


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
    from ..config import OAUTH_ENABLED, API_KEY
    if not (OAUTH_ENABLED or API_KEY):
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
        from starlette.responses import Response
        import uvicorn  # inline import: heavy, and only the HTTP transports need it

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

        sse = SseServerTransport("/messages/")
        server_instance = self.server.server

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
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
            elif path == "/health":
                response = Response('{"status":"ok"}', media_type="application/json")
                await response(scope, receive, send)
            else:
                response = Response("Not Found", status_code=404)
                await response(scope, receive, send)

        # Re-read env at runtime so late-set values (e.g. from CLI flags in
        # cli/main.py after package __init__ has already frozen config.py
        # module constants) still take effect.
        from ..config import safe_get_int_env
        sse_host = os.environ.get('MCP_SSE_HOST', MCP_SSE_HOST)
        sse_port = safe_get_int_env('MCP_SSE_PORT', MCP_SSE_PORT, min_value=1024, max_value=65535)

        _assert_bind_is_authenticated(sse_host, sse_port)

        self.logger.info("Starting SSE transport on %s:%s", _sanitize_log_value(sse_host), sse_port)
        config = uvicorn.Config(
            app,
            host=sse_host,
            port=sse_port,
            log_level="info",
            timeout_keep_alive=MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE,
            timeout_graceful_shutdown=MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN,
        )
        uvi_server = uvicorn.Server(config)
        await uvi_server.serve()

    async def run_streamable_http(self) -> None:
        """Run server with Streamable HTTP transport.

        Uses StreamableHTTPSessionManager for the MCP protocol transport,
        which is what Claude.ai and modern MCP clients expect.

        When OAuth is enabled (MCP_OAUTH_ENABLED=true), this also serves
        OAuth 2.1 endpoints (discovery, DCR, authorization, token) alongside
        the MCP transport endpoint, and requires Bearer token auth on /mcp.
        """
        import uvicorn  # inline import: heavy, and only the HTTP transports need it
        from starlette.responses import Response as StarletteResponse
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from ..config import OAUTH_ENABLED, API_KEY

        # Remote transport: filesystem tools are filtered out of tools/list and
        # refused on tools/call. See MemoryServer.local_only_tools().
        self.server.remote_transport = True

        server_instance = self.server.server
        session_manager = StreamableHTTPSessionManager(
            app=server_instance,
            event_store=None,
            stateless=True,
        )

        # Build OAuth sub-app if enabled
        oauth_app = None
        if OAUTH_ENABLED:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            from ..web.oauth.discovery import router as discovery_router
            from ..web.oauth.registration import router as registration_router
            from ..web.oauth.authorization import router as authorization_router
            from ..config import OAUTH_ISSUER

            from fastapi.middleware.cors import CORSMiddleware

            oauth_app = FastAPI(title="MCP Memory OAuth", docs_url=None, redoc_url=None)
            _cors_origins = [o.strip() for o in os.environ.get("MCP_CORS_ORIGINS", "").split(",") if o.strip()]
            oauth_app.add_middleware(
                CORSMiddleware,
                allow_origins=_cors_origins or ["*"],
                # allow_credentials requires explicit origins; wildcard + credentials
                # is rejected by browsers (CORS spec §3.2.2).
                allow_credentials=bool(_cors_origins),
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

            self.logger.info("OAuth 2.1 endpoints enabled on Streamable HTTP transport")

        _session_manager_ctx = None

        async def app(scope, receive, send):
            nonlocal _session_manager_ctx
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        _session_manager_ctx = session_manager.run()
                        await _session_manager_ctx.__aenter__()
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        if _session_manager_ctx:
                            await _session_manager_ctx.__aexit__(None, None, None)
                        await send({"type": "lifespan.shutdown.complete"})
                        return

            path = scope.get("path", "")
            if path == "/mcp" or path == "/mcp/":
                # Handle CORS preflight for /mcp
                if scope.get("method") == "OPTIONS":
                    req_hdrs = dict(scope.get("headers", []))
                    origin = req_hdrs.get(b"origin", b"").decode("latin-1")
                    # Reuse _cors_origins parsed at startup (set when OAUTH_ENABLED);
                    # fall back to re-parsing if OAuth is off but CORS origins are set.
                    _mcp_cors = _cors_origins if OAUTH_ENABLED else [
                        o.strip() for o in os.environ.get("MCP_CORS_ORIGINS", "").split(",") if o.strip()
                    ]
                    allow_origin = "*" if not _mcp_cors else (origin if origin in _mcp_cors else _mcp_cors[0])
                    response = StarletteResponse(
                        "",
                        status_code=204,
                        headers={
                            "Access-Control-Allow-Origin": allow_origin,
                            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                            "Access-Control-Max-Age": "86400",
                        },
                    )
                    await response(scope, receive, send)
                    return
                # Auth check on /mcp
                if OAUTH_ENABLED or API_KEY:
                    if not await _check_auth_from_scope(scope, receive, send):
                        return
                await session_manager.handle_request(scope, receive, send)
            elif oauth_app and (
                path.startswith("/.well-known/") or
                path.startswith("/oauth/")
            ):
                await oauth_app(scope, receive, send)
            elif path == "/health":
                response = StarletteResponse('{"status":"ok"}', media_type="application/json")
                await response(scope, receive, send)
            else:
                response = StarletteResponse("Not Found", status_code=404)
                await response(scope, receive, send)

        async def _check_auth_from_scope(scope, receive, send) -> bool:
            """Validate auth on /mcp requests. Returns True if authorized."""
            return await check_transport_auth(scope, receive, send)

        # Re-read env at runtime; see run_sse() for rationale.
        from ..config import safe_get_int_env
        sse_host = os.environ.get('MCP_SSE_HOST', MCP_SSE_HOST)
        sse_port = safe_get_int_env('MCP_SSE_PORT', MCP_SSE_PORT, min_value=1024, max_value=65535)
        # Same guard as SSE. /mcp gates on `OAUTH_ENABLED or API_KEY`, so with
        # neither set it serves the full tool surface unauthenticated, exactly
        # like SSE did. The advisory named SSE because that transport had no
        # gate at all; the unauthenticated-bind case is common to both.
        _assert_bind_is_authenticated(sse_host, sse_port)

        self.logger.info("Starting Streamable HTTP transport on %s:%s", _sanitize_log_value(sse_host), sse_port)
        config = uvicorn.Config(
            app,
            host=sse_host,
            port=sse_port,
            log_level="info",
            timeout_keep_alive=MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE,
            timeout_graceful_shutdown=MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN,
        )
        uvi_server = uvicorn.Server(config)
        await uvi_server.serve()

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
