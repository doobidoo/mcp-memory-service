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

import inspect
import threading
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


class TestTicketHardening:
    """Regressions Greptile found in review. Each one names the failure."""

    def test_non_ascii_token_is_refused_not_raised(self):
        # hmac.compare_digest raises TypeError on non-ASCII, which escaped
        # authentication as a 500 instead of a 401.
        store = TicketStore()
        store.issue()  # a pending ticket must be present for this to matter
        assert store.redeem("é") is False
        assert store.redeem("\u00e9\u00e9") is False

    def test_issue_holds_the_lock_across_check_and_insert(self):
        # The purge, size check and insert must be one critical section.
        # Before, two concurrent callers could both pass the limit.
        src = inspect.getsource(TicketStore.issue)
        lock_idx = src.find("self._lock")
        purge_idx = src.find("_purge_locked")
        insert_idx = src.find("self._pending[token] = ticket")
        assert lock_idx != -1 and purge_idx != -1 and insert_idx != -1
        assert lock_idx < purge_idx < insert_idx, (
            "issue() must acquire the lock before purging and hold it through the insert"
        )

    def test_capacity_is_enforced_under_concurrency(self):
        # A direct check of the property the lock is there for: never exceed
        # the ceiling, no matter how many callers arrive at once.
        from mcp_memory_service.web.sse_ticket import _MAX_PENDING

        store = TicketStore()
        now = time.monotonic()
        from mcp_memory_service.web.sse_ticket import Ticket

        for i in range(_MAX_PENDING - 1):
            store._pending[f"k{i}"] = Ticket(token=f"k{i}", expires_at=now + 60, scope="read")

        results = []
        errors = []

        def _try():
            try:
                store.issue()
                results.append("ok")
            except RuntimeError:
                results.append("refused")

        threads = [threading.Thread(target=_try) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5)

        assert not errors
        assert len(store) <= _MAX_PENDING, f"ceiling exceeded: {len(store)}"
        assert results.count("ok") == 1, f"exactly one caller should succeed, got {results}"
