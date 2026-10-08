"""
RED tests for delta-sync resolver functionality (Phase 2).

These tests are designed to FAIL until the resolver implementation is complete.
They validate the CA1-CA7 requirements from the Phase 2 spec.

DO NOT add try/except blocks that make tests PASS when code is missing.
RED done right: import failures at collection time or real assertion failures.
"""

import pytest
from dataclasses import dataclass
from typing import List

# These imports WILL fail until implemented - that's the RED we expect
from mcp_memory_service.storage.sync.resolver import resolve, reduce_events, EventView


class TestDeltaSyncResolver:
    """Test suite for delta-sync resolver pure functions (no DB)."""

    def test_event_view_dataclass_exists(self):
        """
        EventView dataclass should exist with required fields.
        Expected to FAIL: EventView not implemented yet.
        """
        # This will fail with ImportError at collection time if EventView doesn't exist
        # or with TypeError at construction if fields are missing
        event = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="event-123",
            op="create",
            content_hash="hash-abc",
        )

        assert event.hlc_physical == 100
        assert event.hlc_logical == 0
        assert event.agent_id == "agent-001"
        assert event.event_id == "event-123"
        assert event.op == "create"
        assert event.content_hash == "hash-abc"

    def test_ca1_convergence_order_independence(self):
        """
        CA1 (F4 convergência): resolver.reduce_events([a,b]) == reduce_events([b,a])
        for 2 concurrent events with same content_hash.
        Expected to FAIL: resolve and reduce_events functions not implemented.
        """
        # Two concurrent events for same content_hash with different agents
        event_a = EventView(
            hlc_physical=100,
            hlc_logical=5,
            agent_id="agent-001",
            event_id="event-a",
            op="update",
            content_hash="same-hash",
        )

        event_b = EventView(
            hlc_physical=100,
            hlc_logical=8,  # Later logical time, should win
            agent_id="agent-002",
            event_id="event-b",
            op="update",
            content_hash="same-hash",
        )

        # Order [a,b] vs [b,a] should give same result
        winner_ab = reduce_events([event_a, event_b])
        winner_ba = reduce_events([event_b, event_a])

        assert winner_ab == winner_ba, "reduce_events must be order-independent"
        assert winner_ab == event_b, "event_b should win (higher logical clock)"

    def test_ca3_importance_quality_tiebreak(self):
        """
        ADR-0013: on an exact HLC tie with compatible ops, there is NO quality step —
        the winner is decided by the stable tie-break (agent_id → event_id). Here agent_id
        is equal, so the SMALLER event_id wins, regardless of any former "importance".
        This replaces the old quality-tie-break test and locks the ADR-0013 decision.
        """
        event_zzz = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="event-zzz",   # lexicographically LATER → must LOSE
            op="update",
            content_hash="same-hash",
        )

        event_aaa = EventView(
            hlc_physical=100,
            hlc_logical=0,          # Same HLC
            agent_id="agent-001",   # SAME agent → decided by event_id
            event_id="event-aaa",   # lexicographically EARLIER → must WIN
            op="update",
            content_hash="same-hash",
        )

        winner = resolve(event_zzz, event_aaa)
        # Quality is gone from the resolver (ADR-0013): smaller event_id wins the tie.
        assert winner == event_aaa, "smaller event_id must win on HLC tie (no quality step)"
        assert resolve(event_aaa, event_zzz) == event_aaa, "commutative"

    def test_ca3_prime_total_tie_event_id_tiebreak(self):
        """
        CA3' (empate total): same hlc + same agent → smaller event_id wins.
        """
        event_a = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="event-aaa",  # Lexicographically smaller
            op="update",
            content_hash="same-hash",
        )

        event_z = EventView(
            hlc_physical=100,
            hlc_logical=0,  # Same HLC
            agent_id="agent-001",  # Same agent
            event_id="event-zzz",  # Lexicographically larger
            op="update",
            content_hash="same-hash",
        )

        winner = resolve(event_a, event_z)
        assert winner == event_a, "Smaller event_id should win on total tie"

    def test_ca4_delete_vs_update_same_hlc(self):
        """
        CA4 (delete-vs-update): delete (100,0) vs update (100,0) → delete wins.
        Expected to FAIL: resolve function not implementing delete-vs-update rule.
        """
        delete_event = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="delete-event",
            op="delete",
            content_hash="same-hash",
        )

        update_event = EventView(
            hlc_physical=100,
            hlc_logical=0,  # Same HLC
            agent_id="agent-002",
            event_id="update-event",
            op="update",
            content_hash="same-hash",
        )

        winner = resolve(delete_event, update_event)
        assert winner == delete_event, "Delete should win over update with same HLC"

    def test_ca4_update_vs_delete_later_hlc(self):
        """
        CA4 (delete-vs-update): update (101,0) vs delete (100,0) → update wins.
        Expected to FAIL: resolve function not handling logical ordering properly.
        """
        delete_event = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="delete-event",
            op="delete",
            content_hash="same-hash",
        )

        update_event = EventView(
            hlc_physical=101,  # Later physical time
            hlc_logical=0,
            agent_id="agent-002",
            event_id="update-event",
            op="update",
            content_hash="same-hash",
        )

        winner = resolve(update_event, delete_event)
        assert winner == update_event, "Later HLC should win over delete-vs-update rule"

    def test_ca2_late_event_rule_in_resolver(self):
        """
        CA2 (F5 late event): the resolver should identify that an event with
        lower HLC never wins against a higher HLC event.
        Expected to FAIL: resolve function not implementing logical ordering.
        """
        early_event = EventView(
            hlc_physical=50,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="early-event",
            op="update",
            content_hash="same-hash",
        )

        later_event = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-002",
            event_id="later-event",
            op="update",
            content_hash="same-hash",
        )

        winner = resolve(early_event, later_event)
        assert winner == later_event, "Later HLC should always win (late event rule)"

    def test_resolve_function_signature(self):
        """
        Test that resolve function exists and has correct signature.
        Expected to FAIL: resolve function not implemented.
        """
        event_a = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="event-a",
            op="create",
            content_hash="hash-1",
        )

        event_b = EventView(
            hlc_physical=101,
            hlc_logical=0,
            agent_id="agent-002",
            event_id="event-b",
            op="update",
            content_hash="hash-1",
        )

        # This should not raise TypeError for wrong number of args
        result = resolve(event_a, event_b)

        # Result should be one of the input events
        assert result in [event_a, event_b], "resolve should return one of the input events"

    def test_reduce_events_function_signature(self):
        """
        Test that reduce_events function exists and handles empty/single event cases.
        Expected to FAIL: reduce_events function not implemented.
        """
        # Empty list should handle gracefully (or raise appropriate error)
        try:
            result = reduce_events([])
            # If it returns something, it should be None or raise an exception
            assert result is None, "reduce_events([]) should return None or raise"
        except (ValueError, TypeError):
            # Acceptable to raise an error for empty list
            pass

        # Single event should return itself
        single_event = EventView(
            hlc_physical=100,
            hlc_logical=0,
            agent_id="agent-001",
            event_id="single",
            op="create",
            content_hash="hash-single",
        )

        result = reduce_events([single_event])
        assert result == single_event, "reduce_events([event]) should return the event"

    def test_multiple_events_reduce(self):
        """
        Test reduce_events with multiple events of different HLCs.
        Expected to FAIL: reduce_events not implementing proper ordering.
        """
        events = [
            EventView(
                hlc_physical=100, hlc_logical=0,
                agent_id="agent-001", event_id="event-1",
                op="create", content_hash="hash-multi"
            ),
            EventView(
                hlc_physical=105, hlc_logical=0,  # Latest
                agent_id="agent-002", event_id="event-2",
                op="update", content_hash="hash-multi"
            ),
            EventView(
                hlc_physical=102, hlc_logical=0,
                agent_id="agent-003", event_id="event-3",
                op="update", content_hash="hash-multi"
            )
        ]

        winner = reduce_events(events)
        # Event-2 should win (hlc_physical=105 is highest)
        assert winner.event_id == "event-2", "Event with highest HLC should win"
        assert winner.hlc_physical == 105

    def test_logical_clock_ordering_precedence(self):
        """
        Test that logical clock (hlc_physical, hlc_logical) takes precedence over quality.
        Expected to FAIL: resolve function not implementing proper HLC ordering.
        """
        # Higher physical time wins over higher quality
        high_quality_old = EventView(
            hlc_physical=100, hlc_logical=0,
            agent_id="agent-001", event_id="old-high-quality",
            op="update", content_hash="same-hash"
        )

        low_quality_new = EventView(
            hlc_physical=101, hlc_logical=0,  # Later physical time
            agent_id="agent-002", event_id="new-low-quality",
            op="update", content_hash="same-hash"
        )

        winner = resolve(high_quality_old, low_quality_new)
        assert winner == low_quality_new, "HLC ordering must take precedence over quality"

        # Higher logical time within same physical time wins
        same_physical_low_logical = EventView(
            hlc_physical=200, hlc_logical=5,
            agent_id="agent-001", event_id="low-logical",
            op="update", content_hash="same-hash"
        )

        same_physical_high_logical = EventView(
            hlc_physical=200, hlc_logical=7,  # Higher logical within same physical
            agent_id="agent-002", event_id="high-logical",
            op="update", content_hash="same-hash"
        )

        winner = resolve(same_physical_low_logical, same_physical_high_logical)
        assert winner == same_physical_high_logical, "Higher logical clock should win within same physical time"

class TestResolverCommutativity:
    """resolve(a,b) == resolve(b,a) must hold for ALL inputs (tuvok P2 regression lock)."""

    def test_resolve_is_commutative_exhaustive(self):
        import itertools
        from mcp_memory_service.storage.sync.resolver import resolve, EventView
        ops = ['create', 'delete', 'update_metadata']
        hlcs = [(100, 0), (100, 1), (101, 0)]
        agents = [None, 'A', 'B']
        eids = ['e1', 'e2']
        evs = [
            EventView(hp, hl, ag, ei, op, 'h')
            for (hp, hl) in hlcs for ag in agents for ei in eids
            for op in ops
        ]
        for a, b in itertools.product(evs, evs):
            assert resolve(a, b) == resolve(b, a), (
                f"resolve not commutative for {a} vs {b}"
            )


class TestResolverAssociativity:
    """resolve must be associative, and reduce_events must be permutation-invariant.

    reduce_events() folds with functools.reduce, which only yields a single, order-
    independent winner if resolve is BOTH commutative (locked above) AND associative.
    Without this, two hosts folding the same events in different orders could pick
    different winners — a convergence break in a CRDT-like design (seven, Gate 1 #6).

    The domain is deliberately small but collision-rich: repeated HLCs, agents, eids,
    agents, eids and ops force the deep tie-break steps (agent_id → event_id → op),
    which is exactly where a non-associative mutant would surface.
    """

    def _domain(self):
        from mcp_memory_service.storage.sync.resolver import EventView
        ops = ['create', 'delete', 'update_metadata']
        hlcs = [(100, 0), (100, 1)]      # includes an exact-HLC tie
        agents = [None, 'A', 'B']         # includes None (sorts last)
        eids = ['e1', 'e2']               # exercises the UUID-stand-in total order
        return [
            EventView(hp, hl, ag, ei, op, 'h')
            for (hp, hl) in hlcs for ag in agents for ei in eids
            for op in ops
        ]  # 2*3*2*3*2 = 72 events

    def test_resolve_is_associative_exhaustive(self):
        import itertools
        from mcp_memory_service.storage.sync.resolver import resolve
        evs = self._domain()
        # Full 72^3 would be ~373k triples — fine for a pure function, but we sample the
        # cartesian product deterministically (every 7th triple) to keep CI fast while
        # still covering all tie classes. seven proved 0 failures over 4.25M triples;
        # this is the committed regression that locks it.
        for i, (a, b, c) in enumerate(itertools.product(evs, evs, evs)):
            if i % 7:
                continue
            left = resolve(resolve(a, b), c)
            right = resolve(a, resolve(b, c))
            assert left == right, (
                f"resolve not associative:\n  (a·b)·c = {left}\n  a·(b·c) = {right}\n"
                f"  a={a}\n  b={b}\n  c={c}"
            )

    def test_reduce_events_is_permutation_invariant(self):
        import itertools
        from mcp_memory_service.storage.sync.resolver import reduce_events
        evs = self._domain()
        # For a representative set of triples (every 11th), every permutation must reduce
        # to the SAME winner — the property hosts rely on to converge regardless of the
        # order events arrive.
        for i, triple in enumerate(itertools.product(evs, evs, evs)):
            if i % 11:
                continue
            resolved = {reduce_events(list(p)) for p in itertools.permutations(triple)}
            assert len(resolved) == 1, (
                f"reduce_events not permutation-invariant for triple:\n"
                f"  {triple}\n  winners={resolved}"
            )
