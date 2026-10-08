"""Delta-sync resolver: deterministic conflict resolution for concurrent events.

This module provides pure functions (no I/O, no network) for resolving conflicts
between concurrent events for the same content_hash. Implements ADR-0012/0013.

The resolver uses a strict ordering hierarchy:
1. HLC (Hybrid Logical Clock) ordering - later wins
2. Delete-vs-update semantics - delete wins on equal HLC
3. Agent ID then event ID - deterministic final tie-break (UUID → total order)

Quality/importance is intentionally NOT part of the hierarchy: it is a local,
mutable retrieval-ranking signal and must never influence conflict resolution,
or hosts would diverge (ADR-0013).

All functions are deterministic and commutative: resolve(a,b) == resolve(b,a),
and associative, so reduce_events converges regardless of fold order.
"""

from dataclasses import dataclass
from typing import Optional, Iterable
import functools


@dataclass(frozen=True)
class EventView:
    """Immutable view of a sync event for conflict resolution.

    This represents the minimal data needed for the resolver to make decisions.
    The resolver operates on EventView instances, not database rows.
    """
    hlc_physical: int       # Physical time component of HLC (epoch milliseconds)
    hlc_logical: int        # Logical counter component of HLC
    agent_id: Optional[str] # Agent that created this event (None for legacy/unattributed)
    event_id: str          # Unique event identifier (UUID)
    op: str                # Operation type ('create', 'delete', 'update_metadata')
    content_hash: str      # Hash of the content being modified
    # NOTE: quality_score is deliberately NOT a field here. It is a local, mutable
    # retrieval-ranking signal and must never influence conflict resolution, or hosts
    # would diverge (ADR-0013). The resolver's total order is HLC → agent_id → event_id.


def _winner_key(e: EventView) -> tuple:
    """Total-order sort key for conflict resolution; the SMALLEST key wins.

    Encodes the full ADR-0012/0013 hierarchy as one comparable tuple, so `resolve` is a
    single comparison (keeps cyclomatic complexity low) and is trivially commutative:
    `resolve(a, b)` returns whichever of `a`/`b` has the smaller key; equal keys mean the
    events are equivalent.

    Each component is oriented so that SMALLER = winner:
      1. ``-hlc_physical`` / ``-hlc_logical`` — later HLC wins, so negate (ADR-0010).
      2. ``0 if delete else 1`` — on equal HLC, delete beats update/create (ADR-0012).
      3. ``0 if agent_id else 1`` — a non-null agent beats a null one.
      4. ``agent_id`` then ``event_id`` then ``op`` then ``content_hash`` — lexicographically
         smaller wins; UUID ``event_id`` guarantees a stable, host-agnostic total order.

    Quality/importance is deliberately NOT a component: it is a local, mutable retrieval
    signal and would break cross-host convergence (ADR-0013).
    """
    return (
        -e.hlc_physical,
        -e.hlc_logical,
        0 if e.op == 'delete' else 1,
        0 if e.agent_id is not None else 1,
        e.agent_id or "",
        e.event_id,
        e.op,
        e.content_hash,
    )


def resolve(a: EventView, b: EventView) -> EventView:
    """Resolve a conflict between two concurrent events for the same content_hash.

    Pure, deterministic and commutative (`resolve(a, b) == resolve(b, a)`): returns the event
    with the smaller `_winner_key`. The whole ordering hierarchy (HLC → delete-vs-update →
    agent_id → event_id → op → content_hash) lives in `_winner_key`; quality/importance is
    intentionally excluded (ADR-0013). See `_winner_key` for the rationale.

    Args:
        a: First event
        b: Second event

    Returns:
        The winning event (either a or b). On a full key tie the events are equivalent and
        `a` is returned.
    """
    return a if _winner_key(a) <= _winner_key(b) else b


def reduce_events(events: Iterable[EventView]) -> Optional[EventView]:
    """Reduce a collection of events to the single winner using resolve().

    This applies the resolve() function across all events to find the ultimate
    winner. The order of iteration doesn't matter due to resolve's commutativity.

    Args:
        events: Iterable of EventView instances

    Returns:
        The winning event, or None if the input is empty

    Raises:
        ValueError: If the input iterable is empty
    """
    events_list = list(events)

    if not events_list:
        return None

    if len(events_list) == 1:
        return events_list[0]

    # Use functools.reduce to apply resolve() across all events
    # The commutativity of resolve() ensures consistent results regardless of order
    return functools.reduce(resolve, events_list)