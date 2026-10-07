# ADR-0014: Embedding version = (embedding_model_name, embedding_dim)

- **Date:** 2026-10-07
- **Status:** Accepted
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1).
- **Scope:** delta-sync (#1345) Phase 3 — embedding consistency. Satisfies RFC §8.3.

## Context

§8.3 requires that an applied content event and its embedding be observable as a single
versioned state, so a sync never serves "new content, old vector". The event already
carries `content_hash` (the content version — `generate_content_hash` is content-only,
Issue #522). We must define what identifies the *embedding* version, so a future apply
(Phase 4) can detect a vector produced by a different model.

## Decision

The embedding version is **`(embedding_model_name, embedding_dim)`**. The create event
records both. A vector matches the current content iff it was produced for that
`content_hash` **and** by the active model. `content_hash` already pins the content;
the model identity is the net-new provenance Phase 3 stamps.

## Alternatives Considered

### `content_hash` alone — rejected
- Cannot detect a model change (the §8.3 case that matters for Phase 4): same content,
  different model → incomparable vectors, but same hash.

### A monotonic `embedding_schema` integer — rejected
- Needs a registry to map int→model; the model name is already the natural key
  (`self.embedding_model_name`, base.py:95) and aligns with the #1304 model-match startup
  check. No registry needed.

### (model_name, dim) on the event — CHOSEN
- Minimal, natural key; aligns with #1304; detects model change; local cost ~zero
  (just stamping provenance already known to the process).

## Consequences

- **Positive:** events carry enough to let Phase 4 apply reject/flag a cross-model vector.
- **Negative:** two nullable columns on `sync_events`; Phase 1/2 legacy events stay NULL
  (unknown/legacy — honest, not fabricated).
- **Alignment:** consistent with #1304 Phase 4 model-match (same model-name key).

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.3
- `src/mcp_memory_service/utils/hashing.py` (content-only hash), base.py:95 (model name)
- ADR-0007 (event shape), ADR-0015/0016/0017 (pending flag, exclusion, dormant-in-P3)
