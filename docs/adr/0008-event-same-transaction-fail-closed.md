# ADR-0008: Event is written in the same transaction as the mutation (fail-closed, not best-effort)

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 1. Satisfies RFC R4 + §8.1 (b).

## Context

The event-log is only trustworthy if the invariant **"never a memory without its event, nor
an event without its memory"** holds. The existing `usage_telemetry.py` writes events
best-effort with its own `conn.commit()` and swallows failures — correct for telemetry
(must never break the read path), but **wrong** for the sync event-log: a swallowed append
would leave a committed memory with no event, silently breaking delta sync.

## Decision

The event INSERT is written **inside the same SQLite transaction/SAVEPOINT as the hosting
mutation**, with **no commit of its own**. A synchronous helper `_append_sync_event(conn, op,
content_hash, payload)` is called between the mutation's write and its commit:
- `store()` / `store_batch()`: inside the item SAVEPOINT, before `RELEASE`.
- `delete()`: between the `UPDATE ... deleted_at` and `conn.commit()`.
- `update_memory_metadata()`: inside `_do_update()`, before its commit.

If the embedding/mutation fails → `ROLLBACK TO SAVEPOINT` discards memory **and** event
together. A kill-switch `MCP_SYNC_EVENTLOG` gates the whole feature, **but when enabled, a
failure to append the event ABORTS the mutation** — the deliberate inverse of the telemetry
best-effort pattern.

## Alternatives Considered

### Best-effort append like telemetry (own commit, swallow errors) — rejected
- Pros: never blocks a write; reuses the telemetry pattern. Cons: breaks the core invariant
  — a committed memory can lack its event. Defeats the purpose of the log.

### Post-commit append (outbox polled asynchronously) — rejected for Phase 1
- Pros: decouples write latency from the log. Cons: introduces a window where memory exists
  without event (crash between commit and outbox write); needs its own durability/retry
  machinery. Reconsider only if write latency becomes a measured problem.

### Same-transaction, fail-closed — CHOSEN
- Pros: atomicity by construction; crash+replay converges (§8.1 b) because each event is
  0% or 100% applied, never partial. Cons: an event-log bug can now block a memory write —
  mitigated by the kill-switch (disable the feature entirely) and by keeping the helper tiny.

## Consequences

- **Positive:** the invariant holds by construction; §8.1 (b) crash+replay fixture is
  satisfiable; no partial state possible.
- **Negative:** the sync event-log is now on the write path's critical section — a defect
  can fail a store. Accepted because correctness of the log is the whole point; the
  kill-switch is the escape hatch, and the hot-backup remains the safety net (ADR-0002).
- **Documented inversion:** reviewers must NOT "fix" this to match telemetry's best-effort
  style — it is intentional.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.1, §9.5
- `src/mcp_memory_service/storage/usage_telemetry.py` (the best-effort pattern we invert)
- ADR-0007 (table shape)
