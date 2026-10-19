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

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class AgentIdResolution:
    """Outcome of :func:`resolve_agent_id`: the chosen value and where it came from.

    ``origin`` is one of ``"argument"``, ``"header"``, ``"env"``, ``"metadata"``
    or ``"none"`` (R5: the resolution source, logged at DEBUG once per call).
    """

    agent_id: Optional[str]
    origin: str


def resolve_agent_id(
    explicit: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    default: Optional[str] = None,
) -> AgentIdResolution:
    """Resolve the authoring agent id from a single, shared precedence chain.

    This is the ONE place the precedence lives (RFC #1100 +
    spec-agent-id-streamable R2): every write path (``store_memory``,
    ``evolve_memory``, harvest ``auto_commit``) calls this helper instead of
    re-deriving the order itself. The header is attribution, not authentication
    (see module docstring).

    Precedence:
        explicit argument > request header (ContextVar) > ``MCP_AGENT_ID`` env >
        ``metadata["agent_id"]`` > ``default``.

    Returns:
        :class:`AgentIdResolution` carrying the resolved value (may be ``None``
        when nothing matched and ``default`` is ``None``) and the ``origin`` the
        value came from, for the R5 DEBUG log.
    """
    if explicit:
        return AgentIdResolution(explicit, "argument")

    header = get_current_agent_id()
    if header:
        return AgentIdResolution(header, "header")

    env = os.environ.get("MCP_AGENT_ID")
    if env:
        return AgentIdResolution(env, "env")

    meta_value = (metadata or {}).get("agent_id")
    if meta_value:
        return AgentIdResolution(meta_value, "metadata")

    if default:
        # default is a fixed, caller-chosen fallback (e.g. auto_commit's
        # "unknown") — its origin is still "none" because nothing resolved it.
        return AgentIdResolution(default, "none")

    return AgentIdResolution(None, "none")


def _mask_agent_id(value: Optional[str]) -> str:
    """Masked, production-safe rendering of an agent id for logs (R5).

    Never emit the full value: keep at most a 2-char prefix plus ``***`` so the
    DEBUG line stays useful for debugging without leaking the header value.
    """
    if not value:
        return "<none>"
    if len(value) <= 2:
        return "***"
    return f"{value[:2]}***"


def log_agent_id_resolution(resolution: AgentIdResolution, *, context: str) -> None:
    """Emit the R5 DEBUG line: resolution origin, once per call, value masked.

    Logs the ``origin`` (``argument``/``header``/``env``/``metadata``/``none``)
    and only a masked prefix of the value — never the raw header in production.
    """
    logger.debug(
        "agent_id resolved (%s): origin=%s value=%s",
        context,
        resolution.origin,
        _mask_agent_id(resolution.agent_id),
    )
