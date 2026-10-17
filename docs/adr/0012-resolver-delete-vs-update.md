# ADR-0012: Resolver — delete-vs-update semantics (tombstone precedence by logical order)

- **Date:** 2026-10-07 (**v2: 2026-10-16, rev 2026-10-17** — `recreate_of` 3-state signal, event-set liveness (#1564), anchor vs dtop (r4302312738))
- **Status:** Accepted (v2 supersedes the v1 apply clause "the apply step honors the tombstone")
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1). v2 sign-off: Henry (#1522, issuecomment-6204519097, 2026-10-16).
- **Scope:** delta-sync (#1345) Phase 2 resolver + apply. Satisfies RFC §8.2 + consistency with §8.1c. **v2 covers the SYNC path only; the REST/force store path is tracked separately as #1560.**

## Context

The deterministic resolver must define, reproducibly, who wins when a delete and an update
for the same `content_hash` are concurrent. §8.2 says ordering is by logical/HLC clock;
§8.1c says a tombstone must not be resurrected by a replayed old create. These two must
not contradict.

### v2 context — why the v1 apply clause was insufficient (#1522 finding #2)

v1 said "the apply step honors the tombstone", assuming that clause was unambiguous. It
is **not**. A legitimate same-hash re-create (user deletes a memory, then stores the exact
same content again — the repo supports this) and a malicious clock-ahead resurrection are
the **same physical event sequence**: `delete@hlc20 → create same-hash, new event_id,
@hlc30`. No field already on the event (`op`, `event_id`, `agent_id`, HLC, `created_at`)
distinguishes them; the resolver orders by HLC only. An **unconditional** tombstone-preserve
closes the resurrection hole but breaks legitimate re-create (regressions
`test_old_edit_does_not_restore_recreated_memory_field`,
`test_old_metadata_key_does_not_return_after_recreation`); doing nothing keeps the bug.
It is an impossibility result without extra information — the decision needs an
**intentionality signal carried from the producer**.

## Decision

The resolver orders candidates by `(hlc_physical, hlc_logical, agent_id, event_id)` first.
For delete-vs-update of the same hash:
- **delete wins when `hlc(delete) >= hlc(update)`** (it is logically at-or-after the update);
- an update that is logically **later** than a delete wins by normal HLC order (a legitimate
  re-creation/edit after a delete) — **except** the §8.1c invariant, enforced at APPLY (not
  in the pure resolver). The resolver orders events; the apply step decides materialization
  using the `recreate_of` signal defined below. **The resolver itself is unchanged in v2.**

### v2 — the `recreate_of` intentionality signal

- **Producer (`storage/mixins/store.py`):** both `store()` paths (single and batch) stamp
  `recreate_of` on **every** create. When a re-create follows a recorded delete for the
  `content_hash`, it stamps `recreate_of = <event_id of the dominant local delete>` plus the
  stable `recreate_of_hlc = [hlc_physical, hlc_logical, agent_id]` anchor (`_recreate_of_for`,
  which reads the dominant delete from `sync_events` — INDEPENDENTLY of whether the tombstone
  row still exists, since `purge_deleted`/`_purge_tombstone` hard-delete the row but never
  touch `sync_events`, #1564). A create of a fresh hash (no recorded delete) stamps the
  **empty sentinel** `recreate_of = ""` (EMPTY), NOT the absence of the key — only a v1 peer
  omits the key entirely (ABSENT). The three states are distinguished by PRESENCE (`in`), not
  `.get()`.
- **Signal shape:** `recreate_of` references the **delete's `event_id`** — an `event_id`
  names exactly one delete, whereas `deleted_at` is wall-clock and carries the very skew
  this fix fights (Henry). It lives on the **payload**, is **optional**, and there is **NO
  `schema_version` bump** (a bump would fail-stop v1 peers at `apply_remote_event` where
  `event_schema_version > KNOWN_SCHEMA_VERSION`).
- **Consumer (`storage/sync/apply.recompute_liveness`), deterministic rule evaluated over
  the WHOLE recorded event SET for the hash (NOT the current row state, #1564) — `dtop` =
  the dominant delete elected by the resolver's `_winner_key`, `cwin` = the winning create:**

  ```
  if no delete in the set:            materialize LIVE                         # fresh hash / no tombstone
  classify recreate_of (3-state by PRESENCE):
    ABSENT (v1 peer omitted key):     REVIVE iff no delete dominates cwin       # back-compat (INV-2b), via _liveness_bare
    EMPTY  (""):                      DEAD — the delete wins whenever one exists # v2 never saw the delete (Henry Ponto 2)
    NAMED  (event_id [+ recreate_of_hlc]):
        REVIVE iff names_delete(dtop) AND cwin.key() < dtop.key()               # supersedes the dominant delete (INV-2b + P1-A)
        else                          DEAD (tombstone preserved)                # stale/unrelated signal (INV-2a)
  ```

  `names_delete` matches the dominant delete by raw `event_id` OR, when the id was rewritten
  to `bl-...` by the bootstrap baseline (which preserves the HLC), by the stable
  `recreate_of_hlc` anchor compared **directly against `dtop`** (order-independent even when
  a peer holds both the original delete and its `bl-...` copy — Greptile r4302312738). A NAMED
  signal that does not name the dominant delete is stale and stays DEAD. Preserving the
  tombstone is **not** a materialization failure: apply returns processed so the cursor
  advances and the sender does not retry forever.
- **ABSENT = revive** keeps v1 peers (which omit the key) working; the first (wrong)
  implementation preserved on absence and broke the two legitimate-re-create regressions. The
  anti-resurrection guarantee is honored in the case Henry actually named — the **REST/force
  path (#1560)**, carved out of this change — and, on the sync path, by a **stale** `recreate_of`.
- **Decision is independent of HLC** (REQ-2.2): the resolver already used HLC to pick this
  create as winner; apply does not re-derive intent from the clock.

## Alternatives Considered

### delete always wins (last-write-ignores-order) — rejected
- Violates HLC order: a legitimately later update of a re-created memory would be dropped.

### update always wins (soft-delete freely reversible) — rejected
- Would resurrect deleted memories; contradicts §8.1c tombstone durability.

### Logical order first, delete wins on tie, tombstone enforced at apply — CHOSEN (v1)
- Consistent with both §8.2 (order) and §8.1c (durability); deterministic; separates
  "order events" (pure resolver) from "materialize state" (apply, honors tombstone).

### v2 alternatives for the intentionality signal

- **Unconditional tombstone-preserve at apply (v1 reading)** — rejected: breaks legitimate
  same-hash re-create (impossibility result; HLC cannot separate the two cases).
- **`absence = stay dead` (strict INV-2a)** — rejected by Henry: an older v1 peer and a new
  peer that *should* have sent `recreate_of` both emit a bare create; apply cannot tell them
  apart, so guessing "stay dead" strands legitimate re-creates from older clients.
- **Signal on `deleted_at` / HLC** — rejected: wall-clock/skew-prone; the whole bug is that
  HLC cannot discriminate. An `event_id` reference is exact.
- **`schema_version` bump to advertise the key** — rejected: fail-stops every un-updated v1
  peer at `apply_remote_event`; an additive optional payload key is back-compatible instead.
- **`recreate_of` on payload referencing the delete `event_id`, absence = revive** — CHOSEN
  (Henry, 2026-10-16).

## Consequences

- **Positive:** no contradiction between ordering and tombstone durability; reproducible
  across hosts. v2: legitimate same-hash re-create works; a stale/clock-ahead sync create
  that names an unrelated delete cannot resurrect; v1 peers interoperate unchanged.
- **Negative:** the apply step must consult the tombstone and the `recreate_of` signal
  independently of the resolver — two cooperating pieces to understand together (documented).
  The producer must look up the delete `event_id` before purging (one extra `sync_events`
  read per re-create over a tombstone).
- **Note:** "later update wins over older delete" is only reachable when the content is
  legitimately re-created; on the sync path, intent is now explicit via `recreate_of`.
- **Scope caveat:** the REST/force store path (a direct clock-ahead resurrection outside
  sync) is **not** covered here — tracked as **#1560**.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.1c, §8.2
- `docs/rfc/planned/spec-delta-sync-1522-fixes.md` (finding #2, reconciled rule + EARS REQ-2.x/INV-2x)
- #1522 (doobidoo audit, finding #2), Henry sign-off issuecomment-6204519097; #1560 (REST/force path)
- ADR-0009 (durable tombstones), ADR-0010 (HLC), ADR-0013 (importance role)
