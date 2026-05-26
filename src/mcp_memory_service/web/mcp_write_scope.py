"""OAuth write-scope enforcement for MCP mutating tools (GHSA-2r68-g678-7qr3)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from .oauth.middleware import AuthenticationResult

# Legacy HTTP /mcp names and v10 unified tool names that mutate storage.
MCP_WRITE_TOOLS: frozenset[str] = frozenset({
    "store_memory",
    "delete_memory",
    "memory_store",
    "memory_delete",
    "memory_observe",
})

_mcp_auth_context: ContextVar[Optional[AuthenticationResult]] = ContextVar(
    "mcp_auth_context",
    default=None,
)


def set_mcp_auth_context(auth: Optional[AuthenticationResult]) -> None:
    """Bind OAuth/API-key auth for the current MCP request (streamable HTTP)."""
    _mcp_auth_context.set(auth)


def get_mcp_auth_context() -> Optional[AuthenticationResult]:
    return _mcp_auth_context.get()


def check_write_scope_for_tool(tool_name: str) -> Optional[str]:
    """Return an error message when write scope is required but missing."""
    if tool_name not in MCP_WRITE_TOOLS:
        return None

    auth = get_mcp_auth_context()
    if auth is None:
        return None

    if auth.has_scope("write"):
        return None

    return f"Insufficient scope: tool '{tool_name}' requires 'write' access"
