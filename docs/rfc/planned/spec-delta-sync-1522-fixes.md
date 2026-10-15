# SPEC — delta-sync audit fixes (#1522): tombstone, bootstrap caller, embedding_pending

**Status:** #3 and #4 MERGED upstream (`#1530`); #2 BLOCKED on an architecture decision (ADR-0012 v2 + Henry sign-off) · **Updated:** 2026-10-15 · **Fork-only · delta-sync (#1345)**
**Issue:** #1522 (doobidoo audit). **G0:** seven, evidence-based (code read post-v12 merge; re-confirmed 2026-10-15 against upstream/main f6bec0a6 / v12.1.0).
**Scope of this SPEC:** finding #2 only (the one still open). #1 fixed by `da8772c9` (#1523). #3 (bootstrap caller) and #4 (embedding_pending) fixed by `#1530`. #5 (sync_events unbounded) is doc-only per ADR-0009.
**ADR:** #2 needs **ADR-0012 v2** — the first version said "apply honors the tombstone" assuming that clause was unambiguous, but it is NOT: a legitimate same-hash re-create and a clock-ahead resurrection are the same physical event sequence, so an unconditional tombstone-preserve breaks legitimate re-create. ADR-0012 v2 must add an **intentionality signal on the event**. Posted to Henry on #1522 (2026-10-15) for the field format + back-compat call before coding.

> ⚠️ CORRECTION (2026-10-15): an earlier revision of this SPEC claimed REQ-2.1 (preserve the
> tombstone whenever one exists) was confined to `apply.py` and that "a genuine re-create has
> different content → different hash, so there is no conflict". **Both claims were wrong.** The
> repo supports re-creating the SAME hash after a delete (`test_old_edit_does_not_restore_recreated_memory_field`
> passes today and requires the memory to come back ALIVE). That test and the anti-resurrection
> RED test (`test_create_does_not_resurrect_tombstone`, xfail) are the same `delete@hlc20 →
> create same-hash@hlc30` sequence with opposite expected outcomes. HLC cannot tell them apart;
> the fix therefore needs a producer-side signal and is NOT confined to apply. See the revised
> finding #2 below.

## Problem (per finding, confirmed in code)

- **#2 tombstone resurrection (corruption) — STILL OPEN.** `resolver._winner_key` orders by HLC first (ADR-0010); "delete beats update" only breaks an HLC tie. So a `create` with HLC strictly greater than a `delete` wins the resolver. `apply._materialize_event` (ramo `op in (create, update)`) then runs `INSERT OR REPLACE` with `deleted_at` **hard-coded NULL** and no tombstone check → the deleted memory is resurrected.
  - **Why it is not a confined apply fix (the correction):** the same `delete@hlc20 → create same-hash@hlc30` sequence is BOTH a malicious clock-ahead resurrection (must stay dead — `test_create_does_not_resurrect_tombstone`, xfail) AND a legitimate re-create (must come alive — `test_old_edit_does_not_restore_recreated_memory_field`, passes). No field already on the event (`op`, `event_id`, `agent_id`, HLC, `created_at`) distinguishes them. An unconditional "preserve tombstone" closes the hole but breaks legitimate re-create; doing nothing keeps the bug. It is an impossibility result without extra information.
  - **The missing bit exists at the producer:** `StoreMixin.store()` calls `_purge_tombstone(content_hash)` immediately before writing a re-create, so the producer KNOWS the hash was deleted — but the emitted `create` payload carries nothing of it. The fix must carry an intentionality signal from the producer (`store.py`) into apply. **Not confined to apply.py.** Pending ADR-0012 v2 + Henry's call on field format and back-compat.
- **#3 bootstrap unreachable — FIXED upstream (`#1530`).** `sync_from_peer` now calls `bootstrap_from_peer` on first sync. Kept here for history; no code owed.
- **#4 embedding_pending never cleared — FIXED upstream (`#1530`, also #1521).** Re-embed under the matching model now clears the flag with its store partition. Kept here for history; no code owed.

## Requirements (EARS)

### #2 — apply honors the durable tombstone (REQs PENDING ADR-0012 v2 / Henry's call)

> These REQs are frozen until Henry picks the field format and the absence policy
> (posted on #1522, 2026-10-15). The shape below is the *recommended* Option A
> (intentionality field on the create payload); the exact field name/location and the
> absence semantics are his decision and will finalize these REQs.

- REQ-2.1 (revised): WHEN `_materialize_event` processes a `create` whose `content_hash` has a durable tombstone locally (`deleted_at IS NOT NULL`), THE apply step SHALL preserve the tombstone UNLESS the event carries an intentionality signal that legitimately supersedes that tombstone (a genuine re-create); with the signal it SHALL revive, without it it SHALL stay deleted.
- REQ-2.2: THE decision SHALL be independent of the resolver (resolver orders events; apply decides materialization), i.e. it SHALL hold even when the create's HLC is strictly greater than the delete's.
- REQ-2.3: WHEN the incoming event is itself a `delete`, THE apply step SHALL keep the tombstone (deletes are terminal, not blocked by this rule).
- REQ-2.4 (revised): THE fix SHALL add the intentionality signal at the producer (`store.py`) and consume it at apply (`apply.py`); it SHALL NOT change the resolver. It is NOT confined to apply.
- REQ-2.5 (back-compat, Henry's call): WHEN a `create` carries no intentionality signal (older client), THE apply step SHALL [revive | preserve] — default proposed: revive, with a `schema_version` bump so the protection is negotiated and nothing regresses silently.
- INV-2a (anti-resurrection): a bare `create` (no signal) with HLC > a prior `delete` for the same hash, applied to a host holding the tombstone, SHALL leave the memory deleted.
- INV-2b (legitimate re-create): a `create` carrying the intentionality signal SHALL revive the memory (the passing `test_old_edit_does_not_restore_recreated_memory_field` must keep working — it will populate the signal).
- INV-2c (no false block): a first `create` of a hash with NO prior tombstone SHALL materialize normally.

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
- #1 (push allow-list): fixed by `da8772c9`/#1523.
- #3 (bootstrap caller), #4 (embedding_pending): fixed by `#1530` — REQs below kept for history.
- #5 (sync_events compaction): ADR-0009 leaves it out; doc-only.
- Resolver changes: none (resolver stays correct; the fix is at producer + apply).
- ~~Re-create semantics (same hash intentionally brought back): out — a durable tombstone is terminal; a genuine new memory has different content → different hash.~~ **STRUCK (wrong):** the repo supports same-hash re-create after delete; that is exactly the case that makes #2 need an intentionality signal. In scope for ADR-0012 v2.

## Acceptance criteria
- INV-2a passes: a RED test resurrects a delete today; after the fix the memory stays deleted. prove-fix: without the apply change, the test fails.
- INV-3a passes: a fresh spoke (no cursor) installs the baseline; a second sync does not re-bootstrap.
- INV-4a passes: a pending row becomes searchable after re-embed under the matching model; `repair_missing_embeddings_onnx.py` clears the flag.
- Neighbor suite green: `tests/` for sync (apply/resolver/orchestrator/bootstrap) + retrieve/hybrid store-scoped, no regression.
