# ADR-0012: Resolver — delete-vs-update semantics (tombstone precedence by logical order)

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 2 resolver. Satisfies RFC §8.2 + consistency with §8.1c.

## Context

The deterministic resolver must define, reproducibly, who wins when a delete and an update
for the same `content_hash` are concurrent. §8.2 says ordering is by logical/HLC clock;
§8.1c says a tombstone must not be resurrected by a replayed old create. These two must
not contradict.

## Decision

The resolver orders candidates by `(hlc_physical, hlc_logical, agent_id, event_id)` first.
For delete-vs-update of the same hash:
- **delete wins when `hlc(delete) >= hlc(update)`** (it is logically at-or-after the update);
- an update that is logically **later** than a delete wins by normal HLC order (a legitimate
  re-creation/edit after a delete) — **except** the §8.1c invariant, enforced at APPLY (not
  in the pure resolver): a `create` whose hash has a durable tombstone does not re-materialize
  the memory. The resolver orders events; the apply step honors the tombstone.

## Alternatives Considered

### delete always wins (last-write-ignores-order) — rejected
- Violates HLC order: a legitimately later update of a re-created memory would be dropped.

### update always wins (soft-delete freely reversible) — rejected
- Would resurrect deleted memories; contradicts §8.1c tombstone durability.

### Logical order first, delete wins on tie, tombstone enforced at apply — CHOSEN
- Consistent with both §8.2 (order) and §8.1c (durability); deterministic; separates
  "order events" (pure resolver) from "materialize state" (apply, honors tombstone).

## Consequences

- **Positive:** no contradiction between ordering and tombstone durability; reproducible
  across hosts.
- **Negative:** the apply step must consult the tombstone independently of the resolver —
  two cooperating pieces to understand together (documented).
- **Note:** "later update wins over older delete" is only reachable when the content is
  legitimately re-created; a stale replay cannot win because its HLC is older.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.1c, §8.2
- ADR-0009 (durable tombstones), ADR-0010 (HLC), ADR-0013 (importance role)
