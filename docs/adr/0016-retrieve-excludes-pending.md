# ADR-0016: retrieve/search hard-exclude embedding_pending memories

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 3. Satisfies RFC §8.3.

## Context

Once a memory is `embedding_pending`, its vector may not match its content. §8.3 says it
must never be served as consistent. We must decide how read paths treat it.

## Decision

**Hard-exclude** pending memories from `retrieve()` and `search_by_tag()` (and every other
read that joins `memories`): add `AND (m.embedding_pending IS NULL OR m.embedding_pending = 0)`
next to the existing `deleted_at IS NULL`/superseded filters. A pending memory is invisible
to search until its embedding is regenerated and the flag cleared.

## Alternatives Considered

### Include-but-flag — rejected
- Leaks possibly-stale recall — the exact production pain the RFC names (~10 days of
  degraded recall from a model mismatch). Defeats §8.3.

### Re-embed synchronously on read — rejected
- Puts model inference on the read path (latency, failure mode). Kept off the read path
  for the same reason telemetry is best-effort and off it.

### Hard-exclude from reads — CHOSEN
- Directly satisfies "never serve (new content, old vector) as consistent"; cheap
  additive predicate on an indexed column.

## Consequences

- **Positive:** no stale vector is ever served; negligible query cost.
- **Negative:** a pending memory is temporarily invisible until re-embedded — correct
  tradeoff (better absent than wrong). All read sites joining `memories` must get the
  predicate (enumerate via `deleted_at IS NULL`); missing one would leak a pending row.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.3
- `src/mcp_memory_service/storage/mixins/retrieve.py`; ADR-0015 (the flag)
