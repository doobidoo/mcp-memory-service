# ADR-0007: Event-log table shape — separate append-only table keyed by (agent_id, event_id)

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 1 — local event-log, sqlite_vec only.
- **Refines:** ADR-0002 (adopt delta event-log). Satisfies RFC R2 + §8.1.

## Context

Phase 1 needs a durable record of every supported mutation (create/delete/update) so a
future sync can ship deltas and reconcile idempotently. The RFC §8.1 requires a stable
origin identity `(agent_id, event_id)` with a uniqueness constraint, so re-applying a
batch or replaying after a crash cannot duplicate a mutation. We must decide the storage
shape and how event identity is generated.

## Decision

A **separate append-only table `sync_events`** (not an extension of `memories`), with:
- `seq INTEGER PRIMARY KEY AUTOINCREMENT` — local monotonic cursor (append order on this instance).
- `event_id TEXT NOT NULL` — **UUIDv4 per event** (origin identity).
- `agent_id TEXT` — author from `metadata.agent_id`/`MCP_AGENT_ID`; NULL = legacy/unattributed.
- `op`, `content_hash`, `payload` (JSON, minimal to reconstruct), `schema_version INTEGER DEFAULT 1`, `created_at`.
- **`UNIQUE (agent_id, event_id)`** — the §8.1 idempotency key; apply uses `INSERT OR IGNORE`/`ON CONFLICT DO NOTHING`.

## Alternatives Considered

### Extend the `memories` table with event columns — rejected
- Pros: one table. Cons: mixes current-state with history; a tombstone must outlive the
  memory row (purge hard-deletes it), which a column on the same row cannot do. An
  append-only log needs its own lifecycle.

### `event_id = content_hash` or a content-derived hash — rejected
- Two updates of the same memory would collide on the same id; the hash identifies the
  *target*, not the *event*. UUIDv4 gives per-event identity; `content_hash` stays as a
  separate column linking to the memory.

### `event_id = global monotonic counter — rejected for identity
- `seq` already provides local monotonic order; a global counter needs coordination across
  machines (the very thing the event-log avoids until HLC in Phase 2). UUID is coordination-free.

### Separate append-only table with (agent_id, event_id) uniqueness — CHOSEN
- Pros: clean lifecycle, idempotent apply, tombstone survives memory purge, UUID is
  coordination-free. Cons: a second table to migrate/maintain (accepted; it is the subsystem).

## Consequences

- **Positive:** idempotency guaranteed by the DB; append-only audit; tombstone durability.
- **Negative:** `UNIQUE(agent_id, event_id)` with NULL agent_id treats NULLs as distinct in
  SQLite, so uniqueness effectively rests on the UUID `event_id` in Phase 1 (single local
  writer). Revisit when multiple agents share one DB (multi-writer apply, Phase 4).
- **Follow-up:** retention/compaction (purge an event only after all known peers passed its
  cursor) is Phase 2+ (no peers yet).

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.1, §9.4, §9.5
- ADR-0002 (transport decision); migration `015_add_sync_events.sql` (Phase 1)
