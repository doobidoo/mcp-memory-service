# ADR-0009: Durable tombstones reuse existing soft-delete and survive memory purge

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 1. Satisfies RFC §8.1 (deletes = durable tombstones).

## Context

§8.1 requires deletes to be **durable tombstones**, not physical removal, so a replayed old
`create` of the same hash cannot resurrect a deleted memory. The codebase **already** has
soft-delete: `mixins/delete.py` sets `memories.deleted_at`, and `is_deleted()`/`purge_deleted()`
/`_purge_tombstone()` exist. We must decide whether to build a new tombstone store or reuse
this, and how the tombstone outlives a hard purge of the `memories` row.

## Decision

**Reuse the existing `memories.deleted_at`** as the authoritative *local* soft-delete; it is
unchanged. The durable tombstone *for sync* is the **`sync_events` row with `op='delete'`**
(ADR-0007). Because `sync_events` is a separate table, the delete event **survives**
`_purge_tombstone()`/`purge_deleted()` hard-deleting the `memories` row. Conflict resolution
("tombstone wins over a replayed old create") consults `sync_events` by `content_hash` via
the `(op, content_hash)` index. In Phase 1 "wins" is decided by local `seq`; logical/HLC
ordering is Phase 2 (§8.2).

## Alternatives Considered

### New dedicated `tombstones` table — rejected
- Pros: explicit. Cons: duplicates state already in `sync_events(op='delete')` and in
  `deleted_at`; three sources of delete truth to keep consistent. The delete event already
  is the durable tombstone.

### Rely only on `memories.deleted_at` — rejected
- Pros: nothing new. Cons: `purge_deleted()` hard-deletes the row, destroying the tombstone;
  a later replayed create would resurrect it. Not durable across purge.

### Reuse `deleted_at` (local authority) + delete event (durable sync tombstone) — CHOSEN
- Pros: no duplicated delete state; durability comes free from the separate event table;
  integrates with existing soft-delete. Cons: two concepts (local soft-delete vs sync
  tombstone) that must be understood together — documented here.

## Consequences

- **Positive:** §8.1 tombstone durability satisfied; no new table; existing soft-delete
  behavior preserved; resurrection prevented.
- **Negative:** retention/compaction of delete events (only safe to purge after all known
  peers passed the cursor) is **out of scope for Phase 1** (no peers yet) and MUST be built
  before any peer-facing purge in Phase 2+. Until then, delete events accumulate — acceptable
  at current volume (~70KB/day of deltas, ADR-0002).

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.1, §9.3 (bootstrap: deleted_at → baseline tombstone)
- `src/mcp_memory_service/storage/mixins/delete.py` (existing soft-delete reused)
- ADR-0007 (table shape), ADR-0008 (atomicity)
