# ADR-0015: embedding_pending as a real BOOLEAN column on memories

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 3. Satisfies RFC §8.3.

## Context

§8.3 requires a memory whose vector may be stale to be explicitly marked
`embedding_pending` and kept out of search until its embedding matches the content
version. We must decide how that state is represented.

## Decision

A **real column `embedding_pending INTEGER NOT NULL DEFAULT 0`** on `memories` (0 =
consistent, 1 = pending). The write/apply path sets it; the default 0 means every
existing row and every atomic local `store()` is consistent. An index
`idx_memories_embedding_pending` supports the retrieve-exclusion predicate.

## Alternatives Considered

### Derive at query time (compare a per-row model tag to the active model) — rejected
- More schema (per-row model tag) + a comparison on every query. The flag is simpler and
  is exactly what the fixtures/apply need to set.

### Separate `embedding_pending` table — rejected
- 1:1 with the memory row; a join buys nothing. A column is the natural home.

### Real BOOLEAN column, default 0 — CHOSEN
- Simple, indexed, set by the write/apply path; default makes all current data correct.

## Consequences

- **Positive:** cheap to set and to filter; migration-safe default.
- **Negative:** a column whose only live producers (model change / Phase 4 apply) do not
  exist yet in Phase 3 — intentional (ADR-0017); do not "simplify it away".

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.3
- migration `017_add_embedding_consistency.sql`; ADR-0016 (retrieve exclusion)
