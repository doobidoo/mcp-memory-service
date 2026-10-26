"""
MCP (Model Context Protocol) endpoints for HTTP transport.

The tool surface and dispatch logic come from the same `MemoryServer`
instance the stdio transport uses (`server_impl.MemoryServer.list_tools`
and `.call_tool`). This file is just the HTTP framing on top of that
shared core — adding a tool requires no changes here.
"""

import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Response, Request
from fastapi.responses import JSONResponse
import mcp.types as types
from pydantic import BaseModel, ConfigDict

from ..._version import __version__
from ...compat import _sanitize_log_value
from ...utils.tool_rate_limit import resolve_caller_identity
from ..oauth.middleware import require_read_access, get_current_user, AuthenticationResult

logger = logging.getLogger(__name__)

def _is_local_only(server, tool_name: Optional[str]) -> bool:
    """Return True if this tool name must not be exposed over HTTP.

    Some handlers (currently `memory_harvest` and `memory_ingest`) read
    arbitrary filesystem paths from caller-controlled arguments. They were
    designed for a local user who already has filesystem access; exposing
    them over an authenticated remote transport would turn the auth
    boundary into a confused-deputy primitive that can read any file the
    server process can read. The canonical list lives on `MemoryServer`
    so any future local transport (e.g. a unix socket) inherits the same
    restriction by importing the same source of truth.
    """
    if not tool_name:
        return False
    return tool_name in server.local_only_tools()


async def _requires_write(server, tool_name: Optional[str], arguments: Optional[dict] = None) -> bool:
    """Return True if calling this tool requires the 'write' scope.

    The classification lives on `MemoryServer.tool_requires_write`, which
    the SSE and Streamable HTTP transports use too (GHSA-9x7v-2vjj-xghg).
    """
    return await server.tool_requires_write(tool_name, arguments)

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    """MCP protocol request structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP protocol response structure.

    JSON-RPC 2.0 requires successful responses to EXCLUDE the 'error' field
    entirely (not include it as null) and error responses to EXCLUDE
    'result'. `exclude_none` enforces this on serialization.
    """
    model_config = ConfigDict(exclude_none=True)

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# Singleton MemoryServer shared across HTTP requests. Constructed lazily on
# the first /mcp request so import-time cost is avoided. Storage init is
# itself lazy inside MemoryServer, deferred to the first tool call that
# needs it.
_memory_server: Optional[Any] = None


def _get_memory_server():
    """Return the process-wide MemoryServer used by /mcp.

    The HTTP REST layer already initializes a `MemoryStorage` (and
    optionally a consolidator + scheduler) in its FastAPI lifespan, so
    use those.
    """
    global _memory_server
    if _memory_server is None:
        # `server_impl` and `server/__init__.py` have a top-level circular
        # dependency. Force-load the `server` subpackage first so that
        # `server_impl`'s top-level imports are satisfied before we touch
        # `MemoryServer`. Without this, the lazy import here may fire while
        # `server` is mid-load and raise "cannot import name 'main' from
        # partially initialized module".
        import mcp_memory_service.server  # noqa: F401  # inline import: breaks the server_impl circular dependency described above
        from ...server_impl import MemoryServer
        from ..dependencies import get_storage

        cons, sched = None, None
        try:
            from ...api.client import get_consolidator, get_scheduler
            cons = get_consolidator()
            sched = get_scheduler()
        except (ImportError, AttributeError):
            # The HTTP lifespan may not have set these — consolidation is
            # feature-gated by CONSOLIDATION_ENABLED; absence is fine.
            pass

        _memory_server = MemoryServer(
            storage=get_storage(),
            consolidator=cons,
            consolidation_scheduler=sched,
        )
    return _memory_server


def _tool_to_dict(tool) -> Dict[str, Any]:
    """Convert an `mcp.types.Tool` to the wire shape MCP-over-HTTP returns."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
    }


def _wrap_tool_result(result, is_error: bool = False) -> Dict[str, Any]:
    """Pack the shared dispatcher result into an MCP `tools/call` result."""
    if isinstance(result, types.CallToolResult):
        payload = {
            "content": [
                {"type": tc.type, "text": tc.text}
                for tc in result.content
            ],
        }
        if result.isError:
            payload["isError"] = True
        if result.structuredContent is not None:
            payload["structuredContent"] = result.structuredContent
        return payload

    payload: Dict[str, Any] = {
        "content": [
            {"type": "text", "text": tc.text}
            for tc in result
        ]
    }
    if is_error:
        payload["isError"] = True
    return payload


@router.post("/")
@router.post("")
async def mcp_endpoint(
    request: MCPRequest,
    http_request: Request,
    user: AuthenticationResult = Depends(get_current_user)
):
    """Main MCP protocol endpoint. Delegates to the shared MemoryServer.

    Scope is enforced per tool call (see tools/call below), not at the route,
    so a store-restricted token (e.g. ``read:projectA``) can still initialize
    and list tools but may only invoke tools within its stores.
    """
    try:
        # JSON-RPC 2.0: a message without `id` is a Notification; the server
        # MUST NOT reply. MCP Streamable HTTP requires HTTP 202 Accepted with
        # no body in that case. Returning an error here breaks strict clients
        # (e.g. Codex's rmcp) during the `notifications/initialized` handshake.
        if request.id is None:
            return Response(status_code=202)

        server = _get_memory_server()

        if request.method == "initialize":
            response = MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "mcp-memory-service",
                        "version": __version__,
                    },
                },
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        elif request.method == "tools/list":
            tools = await server.list_tools()
            local_only = server.local_only_tools()
            response = MCPResponse(
                id=request.id,
                result={"tools": [_tool_to_dict(t) for t in tools if t.name not in local_only]},
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        elif request.method == "tools/call":
            tool_name = request.params.get("name") if request.params else None
            arguments = request.params.get("arguments", {}) if request.params else {}

            # A `tools/call` without a `name` is a malformed request — reject
            # it before any other dispatch. Without this guard a missing
            # name reaches `call_tool(None, {})` which raises ValueError; the
            # outer catch returns HTTP 200 with an internal-error string,
            # and (worse) the write-scope check would have already
            # short-circuited on a falsy name.
            if not tool_name:
                response = MCPResponse(
                    id=request.id,
                    error={
                        "code": -32602,
                        "message": "Invalid params: 'name' is required for tools/call",
                    },
                )
                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    status_code=400,
                )

            # Local-only tools (e.g. memory_harvest, memory_ingest) read
            # caller-supplied filesystem paths and must not reach a remote
            # caller. Return method-not-found so the tool looks nonexistent
            # to the HTTP client — matches what `tools/list` reports.
            if _is_local_only(server, tool_name):
                response = MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {tool_name}",
                    },
                )
                return JSONResponse(content=response.model_dump(exclude_none=True))

            # Scope enforcement, per tool call. Mutating tools require 'write',
            # read-only tools require 'read' (GHSA-2r68-g678-7qr3). When the
            # call names a `store`, the check is per-store: a global
            # 'read'/'write' grants every store, else 'read:<store>'/
            # 'write:<store>' grants exactly that store ('all' -> global). A
            # call with no `store` targets the implicit default and requires
            # the global operation scope.
            operation = "write" if await _requires_write(server, tool_name, arguments) else "read"
            store_arg = arguments.get("store")
            if store_arg is not None:
                resolved_store = None if store_arg == "all" else store_arg
                authorized = user.has_store_scope(operation, resolved_store)
                required = (
                    f"'{operation}' (global)" if resolved_store is None
                    else f"'{operation}' or '{operation}:{store_arg}'"
                )
            else:
                authorized = user.has_scope(operation)
                required = f"'{operation}'"
            if not authorized:
                response = MCPResponse(
                    id=request.id,
                    error={
                        "code": -32003,
                        "message": f"Insufficient scope: tool requires {required} access",
                        "data": {"required_scope": operation, "tool": tool_name, "store": store_arg},
                    },
                )
                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    status_code=403,
                )

            # Resolve the rate-limit identity from the transport BEFORE the
            # arguments rewrite below: the shim copies X-Agent-ID into
            # arguments["agent_id"] when the call has none, and the limiter
            # must never read tool arguments (#1096).
            caller_identity = resolve_caller_identity(
                http_request.headers.get("X-Agent-ID"),
                http_request.client.host if http_request.client else None,
            )

            # Inject X-Agent-ID header if agent_id argument not explicitly provided
            if not arguments.get("agent_id"):
                header_agent_id = http_request.headers.get("X-Agent-ID")
                if header_agent_id:
                    arguments["agent_id"] = header_agent_id.strip()

            # This path bypasses the SDK wrapper, so a tool exception would surface
            # as JSON-RPC -32603 (a request error) instead of the tool result with
            # isError=true that MCP promises. Convert it here (#1601).
            try:
                text_contents = await server.call_tool(
                    tool_name, arguments, caller_identity=caller_identity
                )
            except Exception as tool_error:
                logger.warning(
                    "Tool %s failed: %s",
                    _sanitize_log_value(tool_name),
                    _sanitize_log_value(tool_error),
                )
                text_contents = [types.TextContent(type="text", text=str(tool_error))]
                response = MCPResponse(
                    id=request.id,
                    result=_wrap_tool_result(text_contents, is_error=True),
                )
                return JSONResponse(content=response.model_dump(exclude_none=True))

            response = MCPResponse(
                id=request.id,
                result=_wrap_tool_result(text_contents),
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        else:
            response = MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}",
                },
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

    except Exception as e:
        logger.error("MCP endpoint error: %s", _sanitize_log_value(e))
        response = MCPResponse(
            id=request.id,
            error={"code": -32603, "message": f"Internal error: {str(e)}"},
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))


@router.get("/tools")
async def list_mcp_tools(
    user: AuthenticationResult = Depends(require_read_access)
):
    """List available MCP tools for discovery."""
    server = _get_memory_server()
    tools = await server.list_tools()
    local_only = server.local_only_tools()
    return {
        "tools": [_tool_to_dict(t) for t in tools if t.name not in local_only],
        "protocol": "mcp",
        "version": "1.0",
    }


@router.get("/health")
async def mcp_health():
    """MCP transport liveness check.

    Deliberately unauthenticated and deliberately contentless. It used to
    return storage statistics, the backend class name and the tool count to
    any anonymous caller (GHSA-7w86-2vmv-fqwm). GHSA-73hc-m4hx-79pj had
    already established that this class of data requires a proven identity —
    it stripped /api/health down to a status field and moved the statistics
    behind Depends(require_read_access) on /api/health/detailed — but that
    change only touched health.py, so this parallel route stayed open.

    The route stays open so that liveness probes keep working without
    credentials; the statistics are simply gone. Authenticated callers get
    them from /api/health/detailed, which is where they already live.
    """
    return {"status": "healthy", "protocol": "mcp"}
