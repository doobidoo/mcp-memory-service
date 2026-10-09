# ADR-0010: HLC format — two INTEGER columns, agent_id kept out of the clock

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 2 — deterministic ordering. Satisfies RFC §8.2.

## Context

§8.2 requires ordering by a logical/HLC clock with a stable tie-breaker
`(hlc, agent_id, event_id)`, because wall-clock timestamps don't suffice across hosts.
Phase 1 stored events with a local `seq` and a wall-clock `created_at` only. Phase 2
must add a Hybrid Logical Clock that is persistable, SQL-sortable, and mergeable for the
Phase 4 receive path.

## Decision

Persist the HLC as **two INTEGER columns** on `sync_events`:
`hlc_physical INTEGER` (epoch millis) + `hlc_logical INTEGER` (tie-break counter).
The `agent_id` is **NOT** part of the HLC — it is a separate tie-breaker field, so the
canonical sort is `ORDER BY hlc_physical, hlc_logical, agent_id, event_id`.
Generation follows the canonical HLC algorithm (send: `pt=now_ms`; if `pt>last.physical`
→ `(pt,0)` else `(last.physical, last.logical+1)`; receive merges `max(local, remote, pt)`
— receive is Phase 4 but the structure supports it now).

## Alternatives Considered

### Ordered string `'physical.logical.agent'` — rejected
- Pros: one column. Cons: lexicographic fragility (`'100' < '99'`) needs zero-padding;
  parse/format on every event; merge arithmetic in Phase 4 awkward.

### Single packed 64-bit INTEGER (`physical<<16 | logical`) — rejected
- Pros: one column, still sortable. Cons: 16 bits of logical can overflow under
  sustained clock skew; couples the two components; harder to merge.

### Two INTEGER columns, agent_id separate — CHOSEN
- Pros: native SQL ordering without padding; integer merge in Phase 4; agent_id as a
  distinct tie-breaker exactly as §8.2 specifies. Cons: two columns + an index (cheap,
  additive).

## Consequences

- **Positive:** deterministic total order via `(hlc_physical, hlc_logical, agent_id, event_id)`;
  Phase 4 receive-merge needs no schema change.
- **Negative:** a migration (016) adds the columns and backfills Phase 1 rows.
- **Follow-up:** backfill of Phase 1 events (ADR note in migration 016): `hlc_physical =
  created_at*1000`, `hlc_logical = seq` — reproducible and idempotent.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.2
- ADR-0007 (event-log table shape); migration `016_add_hlc_to_sync_events.sql`
- HLC: Kulkarni et al., "Logical Physical Clocks" (2014)
