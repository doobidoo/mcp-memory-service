# SPEC — delta-sync audit fixes (#1522): tombstone, bootstrap caller, embedding_pending

**Status:** planned · **Created:** 2026-10-14 · **Fork-only · delta-sync (#1345)**
**Issue:** #1522 (doobidoo audit). **G0:** seven, evidence-based (code read post-v12 merge 346c6921).
**Scope of this SPEC:** findings #2, #3, #4 (the three that need code). #1 is already fixed by `da8772c9` (#1523) — confirm only. #5 (sync_events unbounded) is doc-only per ADR-0009.
**ADR:** #2 implements the existing ADR-0012 (no ADR change — the code never implemented the "apply honors the tombstone" clause the ADR already mandates). #3/#4 need no new ADR.

## Problem (per finding, confirmed in code)

- **#2 tombstone resurrection (corruption).** `resolver._winner_key` (resolver.py:71-80) orders by HLC first (ADR-0010); "delete beats update" only breaks an HLC tie. So a `create` with HLC strictly greater than a `delete` wins the resolver. `apply._materialize_event` (apply.py:638-659) then runs `INSERT OR REPLACE` with `deleted_at` **hard-coded NULL** and no tombstone check → the deleted memory is resurrected. ADR-0012 already requires the opposite: *"a create whose hash has a durable tombstone does not re-materialize the memory. The resolver orders events; the apply step honors the tombstone."* The resolver is correct; **apply never implemented the honor-the-tombstone clause.**
- **#3 bootstrap unreachable (completeness).** `bootstrap_from_peer` (bootstrap.py:210) has no caller in `src/` or `scripts/`. `sync_from_peer` (orchestrator.py:69-75) reads the cursor with default 0 and enters the pull loop directly. A new spoke never installs the peer's baseline → it only pulls events since seq 0 of the peer's log, missing the pre-existing corpus. Phase 5 is dead code in practice.
- **#4 embedding_pending never cleared (silent invisibility).** `apply._materialize_event` sets `embedding_pending=1` on model mismatch (apply.py:632) or embed failure (apply.py:685). Search/tag/time filter out pending rows (retrieve.py:150,269,355,431,588; hybrid.py:72,143,225). No code ever sets it back to 0 (grep: zero matches); `repair_missing_embeddings_onnx.py` inserts a vector but never clears the flag. A memory pulled under a different embedding model stays permanently invisible to search even after models align.

## Requirements (EARS)

### #2 — apply honors the durable tombstone

- REQ-2.1: WHEN `_materialize_event` processes a `create` or `update` whose `content_hash` has a durable tombstone in local `memories` (a row with `deleted_at IS NOT NULL`), THE apply step SHALL NOT re-materialize the row with `deleted_at = NULL`; it SHALL preserve the tombstone (the memory stays deleted).
- REQ-2.2: THE tombstone check SHALL be independent of the resolver (ADR-0012: resolver orders, apply honors), i.e. it SHALL hold even when the create's HLC is strictly greater than the delete's.
- REQ-2.3: WHEN the incoming event is itself a `delete`, THE apply step SHALL continue to materialize/keep the tombstone (deletes are terminal, not blocked by this rule).
- REQ-2.4: THE fix SHALL NOT change the resolver (`resolver.py`) nor ADR-0012; it is confined to the apply step (`apply.py`).
- INV-2a (anti-resurrection): a `create` event with HLC > a prior `delete` for the same hash, applied to a host holding the tombstone, SHALL leave the memory deleted (`deleted_at` preserved, row excluded from search).
- INV-2b (no false block): a legitimate first `create` of a hash with NO prior tombstone SHALL materialize normally (the rule only triggers when a durable tombstone exists).

### #3 — orchestrator calls bootstrap on first sync

- REQ-3.1: WHEN `sync_from_peer` finds no `sync_cursor` row for `peer_id` (first sync with that peer) AND the peer adapter exposes `get_baseline`, THE orchestrator SHALL call `bootstrap_from_peer` before the pull loop, then continue the pull loop from the parked watermark.
- REQ-3.2: WHEN a `sync_cursor` row already exists, THE orchestrator SHALL NOT bootstrap (normal incremental pull only).
- REQ-3.3: IF the peer adapter does not expose `get_baseline`, THE orchestrator SHALL fall back to the current behavior (pull from cursor 0) without error (back-compat with peers that cannot serve a baseline).
- INV-3a: bootstrap SHALL run at most once per peer (guarded by cursor existence); a second `sync_from_peer` SHALL NOT re-bootstrap.

### #4 — clear embedding_pending when the model matches

- REQ-4.1: WHEN a memory with `embedding_pending=1` is (re-)embedded under the local model — whether by a maintenance re-embed or a sync-time hook — THE system SHALL set `embedding_pending=0` in the same transaction that writes the vector.
- REQ-4.2: THE maintenance script `repair_missing_embeddings_onnx.py` SHALL clear `embedding_pending=0` for each row it successfully embeds.
- REQ-4.3: WHEN `_materialize_event` stores a pulled create whose `embedding_model` now MATCHES the local `embedding_model_name`, THE apply step SHALL embed and store the vector with `embedding_pending=0` (this path already exists at apply.py:632 via `can_embed`; the gap is only the backfill of rows stored pending earlier under a mismatch).
- INV-4a: after a re-embed of a pending row under the matching model, the row SHALL be returned by a store-scoped semantic search (previously invisible → now visible).
- INV-4b: a row still genuinely pending (model still mismatched, no vector) SHALL remain excluded from search (the fix clears only rows that got their vector).

## Out of scope
- #1 (push allow-list): already fixed by `da8772c9`/#1523 — confirm in the reply, no code.
- #5 (sync_events compaction): ADR-0009 leaves it out; doc-only.
- Resolver changes: none (ADR-0012 unchanged).
- Re-create semantics (same hash intentionally brought back): out — a durable tombstone is terminal; a genuine new memory has different content → different hash.

## Acceptance criteria
- INV-2a passes: a RED test resurrects a delete today; after the fix the memory stays deleted. prove-fix: without the apply change, the test fails.
- INV-3a passes: a fresh spoke (no cursor) installs the baseline; a second sync does not re-bootstrap.
- INV-4a passes: a pending row becomes searchable after re-embed under the matching model; `repair_missing_embeddings_onnx.py` clears the flag.
- Neighbor suite green: `tests/` for sync (apply/resolver/orchestrator/bootstrap) + retrieve/hybrid store-scoped, no regression.
