"""Short-lived, single-use tickets for EventSource connections.

Why this exists (issue doobidoo/mcp-memory-service#1117):

`EventSource` cannot set an ``Authorization`` header, so the dashboard used to
carry the long-lived API key in the URL query instead::

    GET /api/events?api_key=<full key> HTTP/1.1

uvicorn's access log records the whole request line, so every SSE connection
wrote that key in cleartext into the log file. TLS does not help — it protects
the wire, not the log. The file then outlives the session and gets pasted into
bug reports.

Access-log redaction already landed (``web/log_redaction.py``), but that treats
the symptom. This module removes the cause: the browser exchanges its real
credentials **once, in a POST body**, for a ticket that is worthless a minute
later. A leaked log line then contains nothing of value.

Design constraints, in order of how badly each one bites:

* **Single-use.** A ticket that survives replay is just a shorter-lived key.
* **Short TTL.** Long enough for the browser to open the connection and for
  EventSource's first reconnect to succeed if the network hiccups; short enough
  that a captured URL is useless almost immediately. 60 seconds is the balance
  — reconnects beyond that re-fetch a ticket, which is cheap.
* **In-memory only.** These are session-scoped credentials. Persisting them
  would recreate the exact store of secrets this change is meant to delete.
* **Constant-time comparison.** The ticket is a bearer credential; compare it
  the way one.
* **Bounded.** The table must not grow without limit if a client loops on
  ``POST /ticket`` — see ``_MAX_PENDING``.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass

# 60s: survives EventSource's first automatic reconnect, expires before a
# pasted log line has any useful life.
DEFAULT_TTL_SECONDS = 60.0

# A client that abandons tickets faster than they expire must not pin memory.
# 4096 is far above any real dashboard's concurrent-connection count.
_MAX_PENDING = 4096


@dataclass(frozen=True)
class Ticket:
    """A single issued ticket and its expiry."""

    token: str
    expires_at: float
    scope: str


class TicketStore:
    """In-memory store of issued SSE tickets.

    Thread-safe: the web app may issue from an async context while a request
    handler redeems on a worker thread.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, Ticket] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def issue(self, scope: str = "read") -> Ticket:
        """Mint a ticket. Raises if the table is saturated."""
        now = time.monotonic()
        self._purge_locked(now)
        if len(self._pending) >= _MAX_PENDING:
            # Refuse rather than grow unbounded: a client hammering this
            # endpoint must not become a memory leak.
            raise RuntimeError("too many pending SSE tickets; retry shortly")
        # 32 bytes = 256 bits. This is a bearer credential — do not skimp.
        token = secrets.token_urlsafe(32)
        ticket = Ticket(token=token, expires_at=now + self._ttl, scope=scope)
        with self._lock:
            self._pending[token] = ticket
        return ticket

    def redeem(self, token: str, scope: str = "read") -> bool:
        """Consume a ticket. True only for an unexpired ticket of the right scope.

        Always consumes on a miss too, so a brute-force loop cannot distinguish
        "wrong scope" from "already used" by whether the ticket still exists.
        """
        if not token or not isinstance(token, str):
            return False
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            # hmac.compare_digest wants bytes; the keys are urlsafe-base64 text.
            match = None
            for candidate in list(self._pending):
                if hmac.compare_digest(candidate, token):
                    match = candidate
                    break
            if match is None:
                return False
            ticket = self._pending.pop(match)
            return ticket.expires_at > now and ticket.scope == scope

    def _purge_locked(self, now: float) -> None:
        """Drop expired tickets. Caller must hold the lock, or own the table."""
        stale = [k for k, t in self._pending.items() if t.expires_at <= now]
        for k in stale:
            del self._pending[k]

    def __len__(self) -> int:
        """Live ticket count. Purges first, so an expired ticket does not count
        as pending — `len()` reporting dead entries would be a lie about how
        many credentials are actually out there."""
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            return len(self._pending)


# Process-wide store. The lifetime of these tickets is shorter than any request
# that would race on a second instance.
sse_tickets = TicketStore()
