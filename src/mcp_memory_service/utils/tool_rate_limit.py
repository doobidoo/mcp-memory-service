"""Per-agent sliding-window rate limiting for MCP tool calls."""

import logging
import math
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_MAX_KEYS = 4096
_WINDOW_SECONDS = 60.0
_UNSET = object()


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of one rate-limit check.

    ``limit_per_minute`` is ``None`` when the tool is unlimited. When a call is
    denied, ``retry_after_seconds`` is the minimum whole-second delay before
    the oldest hit leaves the sliding window.
    """

    allowed: bool
    limit_per_minute: int | None = None
    retry_after_seconds: int = 0
    should_log: bool = False


@dataclass
class _Window:
    hits: deque[float] = field(default_factory=deque)
    denied_at: float | None = None


def _read_limit(env_name: str):
    """Read a non-negative integer setting.

    ``0`` explicitly means unlimited. Invalid or negative values are ignored
    and ``_UNSET`` is returned so callers can fall back to the next setting.
    """

    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return _UNSET
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected a non-negative integer",
            env_name,
            raw,
        )
        return _UNSET
    if value < 0:
        logger.warning(
            "Ignoring negative %s=%r; expected a non-negative integer",
            env_name,
            raw,
        )
        return _UNSET
    return value


def _env_names(tool_name: str) -> tuple[str, ...]:
    """Return per-tool env names in precedence order.

    The issue documents lower-case tool names (for example
    ``MCP_RATE_LIMIT_memory_store``). Accept the upper-case form too because
    most deployment tooling normalizes environment variable names.
    """

    return (
        f"MCP_RATE_LIMIT_{tool_name}",
        f"MCP_RATE_LIMIT_{tool_name.upper()}",
    )


def resolve_limit(tool_name: str) -> int | None:
    """Resolve the per-minute limit for a tool.

    A tool-specific setting overrides ``MCP_RATE_LIMIT_PER_MINUTE``. The
    default is unlimited (``None``) for backward compatibility.
    """

    for env_name in _env_names(tool_name):
        value = _read_limit(env_name)
        if value is not _UNSET:
            return None if value == 0 else value

    value = _read_limit("MCP_RATE_LIMIT_PER_MINUTE")
    if value is _UNSET or value == 0:
        return None
    return value


def _bucket_key(identity: str, tool_name: str) -> str:
    """Hash the identity and tool so arbitrary ids do not grow the key space."""

    return sha256(f"{identity}\0{tool_name}".encode()).hexdigest()


class ToolRateLimiter:
    """Thread-safe per-identity/per-tool sliding-window limiter.

    State is intentionally process-local. ``max_keys`` caps the number of
    retained windows; the least-recently-used window is evicted when a new
    identity arrives. Identities are hashed before storage and are never
    logged.
    """

    def __init__(
        self,
        max_keys: int = DEFAULT_MAX_KEYS,
        window_seconds: float = _WINDOW_SECONDS,
    ):
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_keys = max_keys
        self.window_seconds = window_seconds
        self._hits: OrderedDict[str, _Window] = OrderedDict()
        self._lock = Lock()

    @property
    def bucket_count(self) -> int:
        """Return the number of retained identity/tool windows."""

        with self._lock:
            return len(self._hits)

    def check(
        self,
        identity: str,
        tool_name: str,
        now: float | None = None,
    ) -> RateLimitDecision:
        """Record one call and return whether it is allowed."""

        limit = resolve_limit(tool_name)
        if limit is None:
            return RateLimitDecision(allowed=True)

        current = time.monotonic() if now is None else now
        key = _bucket_key(identity, tool_name)

        with self._lock:
            window = self._hits.get(key)
            if window is None:
                window = _Window()
                self._hits[key] = window
                while len(self._hits) > self.max_keys:
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(key)

            cutoff = current - self.window_seconds
            while window.hits and window.hits[0] <= cutoff:
                window.hits.popleft()

            if len(window.hits) >= limit:
                should_log = (
                    window.denied_at is None
                    or current - window.denied_at >= self.window_seconds
                )
                window.denied_at = current
                retry_after = max(
                    1,
                    math.ceil(window.hits[0] + self.window_seconds - current),
                )
                return RateLimitDecision(
                    allowed=False,
                    limit_per_minute=limit,
                    retry_after_seconds=retry_after,
                    should_log=should_log,
                )

            window.hits.append(current)
            window.denied_at = None
            return RateLimitDecision(allowed=True, limit_per_minute=limit)
