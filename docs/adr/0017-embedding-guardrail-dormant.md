# ADR-0017: Embedding-consistency guardrail is built and tested in Phase 3, but dormant

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 3. Context for RFC §8.3.

## Context

Gate 1 verified (grep + source) that **local operation has no stale-vector path**:
`store()` writes content + embedding atomically in one transaction; `content_hash` is
content-only so a content change is a new row (new create), never an in-place edit;
`update_memory_metadata` never touches content; re-embed is an offline full rebuild. So
in Phase 3 LOCAL there is **no live producer** of `embedding_pending`.

## Decision

Phase 3 ships the embedding-consistency **guardrail** (ADR-0014 version stamping,
ADR-0015 pending flag, ADR-0016 retrieve exclusion) **built and test-proven**, but
**dormant**: its real producers are (a) an embedding-model change (offline re-embed window)
and (b) the Phase 4 apply of a peer event that writes a row before its local embedding is
regenerated. Fixtures drive the flag directly to prove the mechanism; they do NOT
manufacture a fake local content-update path to create a producer.

## Alternatives Considered

### Defer the whole guardrail to Phase 4 — rejected
- Phase 4 apply would then have to build storage schema + retrieve exclusion AND the
  transport logic at once. Building the guardrail now keeps Phase 4 to "flip the bit".

### Invent an in-place content-update op to justify a producer — rejected
- Dishonest; contradicts the content-immutable-per-row model (ADR via §8.3 analysis).

### Build + test now, dormant until Phase 4/model-change — CHOSEN
- Honest; the mechanism is proven in isolation; Phase 4 only sets the flag.

## Consequences

- **Positive:** Phase 4 apply is simpler and safer; the guardrail is validated before it
  is relied upon.
- **Negative:** a reviewer may see "unused" code — this ADR documents why it is
  intentional, so it is not "simplified away".
- **Honesty:** the SPEC's acceptance tests drive the pending state directly and assert the
  normal store path leaves `embedding_pending=0`.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.3, §9.4
- ADR-0014/0015/0016; `src/mcp_memory_service/storage/mixins/store.py` (atomic store)
