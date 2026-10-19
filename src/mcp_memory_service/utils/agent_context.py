"""Per-request agent identity propagation for the streamable-http transport.

The ``X-Agent-ID`` header is an **authorship attribution** signal, not an
authentication or authorization credential: it only names who wrote a memory
and grants no permission whatsoever. Auth/authz live on a different axis
(OAuth / API key on ``/mcp``); see ``spec-agent-id-streamable.md`` §3.4.

The streamable-http transport (``StreamableHTTPSessionManager``) has no hook to
pass per-request HTTP headers down to the tool-call path. This module bridges
that gap with a :class:`contextvars.ContextVar`: an ASGI middleware reads the
header once per request, stores it here, and resets it in a ``finally`` block so
identity never leaks between concurrent requests sharing the same worker.

The single resolution point (``services/memory_service.py``) consults this
ContextVar between the explicit argument and the ``MCP_AGENT_ID`` env var.
"""

from contextvars import ContextVar
from typing import Optional

# Default None => "no header for this request"; resolution falls back to env.
_current_agent_id: ContextVar[Optional[str]] = ContextVar(
    "mcp_memory_current_agent_id", default=None
)


def set_current_agent_id(agent_id: Optional[str]) -> object:
    """Bind the request-scoped agent id. Returns a reset token for ``finally``."""
    return _current_agent_id.set(agent_id)


def reset_current_agent_id(token: object) -> None:
    """Restore the previous value using the token from :func:`set_current_agent_id`."""
    _current_agent_id.reset(token)


def get_current_agent_id() -> Optional[str]:
    """Return the agent id bound to the current request, or ``None``."""
    return _current_agent_id.get()
