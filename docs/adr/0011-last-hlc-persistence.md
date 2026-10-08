# ADR-0011: Persist last_hlc in metadata(key,value), committed with the event

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 2. Supports RFC §8.2 HLC generation.

## Context

An HLC advances monotonically per instance: each new event's clock depends on the last
HLC this instance produced/saw. Phase 2 must persist that `last_hlc` so the clock keeps
advancing across restarts and never regresses, and so the Phase 4 receive path can merge
a remote HLC against the last local one.

## Decision

Store `last_hlc` as two singleton rows in the **existing `metadata(key, value)` table**:
`sync_hlc_physical` and `sync_hlc_logical`. The update is written in the **same
transaction** as the event (ADR-0008): the event row and the clock advance commit
together or roll back together. On the first boot after migration 016, seed `last_hlc`
from `MAX(hlc_physical, hlc_logical)` over `sync_events`.

## Alternatives Considered

### New dedicated table `sync_state(k, v)` — deferred to Phase 4
- Pros: explicit home for sync state. Cons: new table for a single per-instance singleton
  now; the per-peer cursor that would justify it is a Phase 4 concern. Adopt then, not now.

### Derive from `MAX(sync_events)` on every append, don't persist — rejected
- Pros: no extra write. Cons: a `MAX()` scan per append; and it does not support the
  Phase 4 receive-merge, which needs the last *seen* HLC (including remote values merged
  in), not just the max locally written.

### metadata(key,value) singleton, committed with the event — CHOSEN
- Pros: reuses existing infra; atomic with the event (no clock/event divergence); trivial
  read. Cons: metadata table is now also sync-state — acceptable, documented here.

## Consequences

- **Positive:** clock never regresses across restarts; atomic with the event; cheap.
- **Negative:** `metadata` carries sync state alongside config keys (`schema_version`,
  `distance_metric`) — a minor overload, revisited if Phase 4 needs per-peer cursors.
- **Follow-up:** Phase 4 per-peer cursor likely moves to a dedicated table (ADR then).

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.2, §8.5 (cursor as source of truth — Phase 4)
- ADR-0008 (same-transaction atomicity), ADR-0010 (HLC format)
