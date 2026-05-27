"""
MCP (Model Context Protocol) endpoints for HTTP transport.

The tool surface and dispatch logic come from the same `MemoryServer`
instance the stdio transport uses (`server_impl.MemoryServer.list_tools`
and `.call_tool`). This file is just the HTTP framing on top of that
shared core — adding a tool requires no changes here.
"""

import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ..._version import __version__
from ...compat import DEPRECATED_TOOLS
from ..oauth.middleware import require_read_access, AuthenticationResult

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

    Deprecated aliases (e.g. `ingest_document` → `memory_ingest`) are
    resolved to their v10 target before classification — otherwise a
    remote client could call the local-only tool through its old name.
    """
    if not tool_name:
        return False
    target = tool_name
    alias = DEPRECATED_TOOLS.get(tool_name)
    if alias is not None:
        target = alias[0]
    return target in server.local_only_tools()


async def _requires_write(server, tool_name: Optional[str]) -> bool:
    """Return True if calling this tool requires the 'write' scope.

    Derived per-call from `server.list_tools()`: a tool counts as write
    unless its annotations explicitly set `readOnlyHint=True`. This matches
    the MCP spec's default ("destructive=True, readOnly=False" unless
    declared otherwise) and keeps the set in lockstep with the canonical
    tool list — adding a new tool to `MemoryServer.list_tools` automatically
    applies the right scope check.

    Conditional tools like `memory_consolidate` only appear in the list once
    their gating state is satisfied, so we derive on each call rather than
    caching (the alternative would risk a stale cache locked in before the
    consolidator finished initializing).

    Deprecated aliases (e.g. `store_memory` → `memory_store`) are resolved
    to their v10 target before classification — without this a read-only
    token could call `store_memory` and have it silently dispatched past
    the scope gate (GHSA-2r68-g678-7qr3 regression risk).

    Conservative default: a tool name we don't recognise is treated as
    requiring write scope.
    """
    if not tool_name:
        return False
    target = tool_name
    alias = DEPRECATED_TOOLS.get(tool_name)
    if alias is not None:
        target = alias[0]
    # Fail closed if list_tools() raises during classification: the request
    # MUST NOT be allowed through the scope gate just because we couldn't
    # introspect the surface. A read-only token gets a clean -32003/403;
    # call_tool would also fail, but the wrong response code there masks
    # the scope decision.
    try:
        tools = await server.list_tools()
    except Exception:
        logger.exception("list_tools() raised during write-scope classification — failing closed")
        return True
    for tool in tools:
        if tool.name == target:
            return not (tool.annotations and getattr(tool.annotations, "readOnlyHint", False))
    return True

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
        import mcp_memory_service.server  # noqa: F401
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


def _wrap_tool_result(text_contents) -> Dict[str, Any]:
    """Pack a `list[mcp.types.TextContent]` into an MCP `tools/call` result."""
    return {
        "content": [
            {"type": "text", "text": tc.text}
            for tc in text_contents
        ]
    }


@router.post("/")
@router.post("")
async def mcp_endpoint(
    request: MCPRequest,
    user: AuthenticationResult = Depends(require_read_access)
):
    """Main MCP protocol endpoint. Delegates to the shared MemoryServer."""
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

            # Mutating tools require 'write' scope even through MCP. Read-only
            # tools remain accessible with 'read' scope (GHSA-2r68-g678-7qr3).
            if await _requires_write(server, tool_name) and not user.has_scope("write"):
                response = MCPResponse(
                    id=request.id,
                    error={
                        "code": -32003,
                        "message": "Insufficient scope: tool requires 'write' access",
                        "data": {"required_scope": "write", "tool": tool_name},
                    },
                )
                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    status_code=403,
                )

            text_contents = await server.call_tool(tool_name, arguments)
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
        logger.error(f"MCP endpoint error: {e}")
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
    """MCP-specific health check."""
    server = _get_memory_server()
    await server._ensure_storage_initialized()
    stats = await server.storage.get_stats() if server.storage else {}
    tools = await server.list_tools()
    return {
        "status": "healthy",
        "protocol": "mcp",
        "tools_available": len(tools),
        "storage_backend": (
            server.storage.__class__.__name__ if server.storage else "uninitialized"
        ),
        "statistics": stats,
    }
