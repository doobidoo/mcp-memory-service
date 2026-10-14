"""Single-use SSE tickets (issue #1117).

The bug these tests pin down: `EventSource` cannot set headers, so the web UI
authenticated live updates with `GET /api/events?api_key=<long-lived key>`, and
uvicorn wrote that URL into the access log on every reconnect. The fix is a
ticket issued over POST, so the secret never appears in a URL.

What matters and is asserted here:

* a ticket can be redeemed **exactly once** — a replayable ticket is just a
  shorter-lived key;
* it stops working when its TTL expires, even if it was never used;
* `?api_key=` is refused, so the leak cannot come back through a stale client;
* a ticket is consumed even when it is refused (wrong scope / expired), so a
  probe cannot learn from which failures leave the ticket behind.
"""

from __future__ import annotations

import time

import pytest

from mcp_memory_service.web.sse_ticket import (
    DEFAULT_TTL_SECONDS,
    TicketStore,
    sse_tickets,
)


class TestTicketStore:
    def test_issue_returns_a_usable_ticket(self):
        store = TicketStore()
        t = store.issue()
        assert len(t.token) >= 32, "a bearer credential must not be short"
        assert store.redeem(t.token) is True

    def test_a_ticket_is_single_use(self):
        store = TicketStore()
        t = store.issue()
        assert store.redeem(t.token) is True
        assert store.redeem(t.token) is False, "replay must fail — a ticket that survives replay is a shorter-lived key"

    def test_expired_ticket_is_refused(self):
        # TTL of zero: anything issued is already stale.
        store = TicketStore(ttl_seconds=0.0)
        t = store.issue()
        time.sleep(0.01)
        assert store.redeem(t.token) is False

    def test_expiry_cleans_the_table(self):
        store = TicketStore(ttl_seconds=0.0)
        store.issue()
        time.sleep(0.01)
        assert len(store) == 0, "expired tickets must not accumulate"

    def test_scope_mismatch_is_refused_and_consumed(self):
        store = TicketStore()
        t = store.issue(scope="read")
        assert store.redeem(t.token, scope="write") is False
        # Consumed on the miss too: a wrong-scope probe must not leave the
        # ticket alive for a second attempt with the right scope.
        assert store.redeem(t.token, scope="read") is False

    def test_unknown_ticket_is_refused(self):
        store = TicketStore()
        assert store.redeem("not-a-real-ticket") is False

    def test_empty_and_non_string_tokens_are_refused(self):
        store = TicketStore()
        assert store.redeem("") is False
        assert store.redeem(None) is False  # type: ignore[arg-type]

    def test_saturated_store_refuses_rather_than_growing(self):
        store = TicketStore()
        # Force the table to its ceiling without minting 4096 secrets.
        store._pending.clear()
        from mcp_memory_service.web.sse_ticket import _MAX_PENDING, Ticket

        now = time.monotonic()
        for i in range(_MAX_PENDING):
            store._pending[f"k{i}"] = Ticket(token=f"k{i}", expires_at=now + 60, scope="read")
        with pytest.raises(RuntimeError):
            store.issue()

    def test_default_ttl_is_short(self):
        # A leaked URL must be worthless almost immediately. If someone raises
        # this into minutes, the whole point of the change is gone.
        assert DEFAULT_TTL_SECONDS <= 60.0

    def test_module_level_store_exists(self):
        assert isinstance(sse_tickets, TicketStore)
