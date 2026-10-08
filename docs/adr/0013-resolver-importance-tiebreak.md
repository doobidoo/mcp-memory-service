# ADR-0013: Resolver — quality/importance is NOT a conflict tie-breaker (determinism over "importance")

- **Date:** 2026-10-07
- **Status:** Accepted (supersedes the 2026-10-07 earlier draft of this ADR that made
  `quality_score` step 3 of the resolver)
- **Deciders:** Claudio (owner/client), Zero (agent). Arch input: seven (Gate 1, finding #1).
- **Scope:** delta-sync (#1345) Phase 2 resolver. Satisfies RFC §8.2; refines/narrows R3.

## Context

R3 originally framed reconciliation as "timeline + importance". The first draft of this
ADR implemented that literally: on an exact HLC tie, the resolver preferred the event with
the higher `quality_score`. During Gate-1 review (seven, finding #1) a convergence hazard
surfaced, and the owner then questioned whether importance belongs in the resolver at all.

**The hazard.** `quality_score` is a *local, mutable* signal, not an immutable property of
an event. It is derived per-host and rewritten over time by: user rating
(`web/api/quality.py`), computed quality + user rating composition
(`quality/config.py::effective_quality`), and time decay (`consolidation/decay.py`). Two
hosts routinely hold *different* `quality_score` for the same memory.

The resolver, however, must be a **pure convergence function**: given the same two events,
every host must pick the **same** winner, or the hosts silently diverge (same memory, two
different surviving contents, while sync reports success). If the resolver reads a
host-local quality, that guarantee breaks.

**The owner's insight (the deciding argument).** Quality-as-a-signal is legitimately
*local*: a memory that helped *me here* earns a thumbs-up and ranks higher *in my search* —
and it is fine that the same memory ranks differently for another agent (e.g. T'Pol on the
VPS). That locality is correct **for retrieval ranking**. But "which edit of a memory
survives a write race" has no causal relationship to "how useful that memory was in a
search". Coupling the two is both surprising and, because quality is local, a divergence
source. The clean resolution is not to pick a source for quality in the resolver — it is to
**remove quality from the resolver entirely**. HLC order plus a UUID `event_id` already give
a complete, deterministic total order without it.

## Decision

The resolver decides in strict order, **with no quality/importance step**:
1. **Logical order** — `(hlc_physical, hlc_logical)`, higher wins.
2. **delete-vs-update** on exactly equal HLC — delete wins (ADR-0012).
3. **Stable final tie-breaker** — `agent_id` (non-null before null), then `event_id`
   (UUIDv4 → total order, never ties).

`quality_score` is **not** an input to the resolver and is removed from `EventView`. It
remains a purely local retrieval-ranking signal (unchanged), computed and mutated per host,
and is never synced for the purpose of conflict resolution.

This narrows R3: the "importance" half of "timeline + importance" is dropped from
*reconciliation*. Importance keeps its real job — ranking search results locally — which is
where the owner actually wants it.

## Alternatives Considered

### Quality as step-3 tie-break, read from host-local state — rejected (the original draft)
- Non-deterministic across hosts: local quality diverges → different winners → silent
  divergence. This is the hazard that triggered the rewrite.

### Quality as step-3 tie-break, read from the synced event payload — rejected
- Fixes determinism (same number on every host) but keeps a conceptually wrong coupling
  (search-usefulness deciding write-race winners) and forces `delete`/`update_metadata`
  payloads to start carrying a quality snapshot they don't carry today, enlarging the event
  schema for a tie-breaker that fires only on an exact HLC collision. Cost without benefit.

### No quality in the resolver; HLC → agent_id → event_id — CHOSEN
- Fully deterministic and convergent by construction (UUID event_id guarantees a total
  order). Eliminates the divergence class instead of managing it. Keeps quality 100% local
  for its real purpose (retrieval ranking). Simplest resolver, smallest event payload.

## Consequences

- **Positive:** the resolver is a pure, host-agnostic convergence function; no local signal
  can leak into conflict resolution; `delete`/`update_metadata` payloads need no quality
  field; one fewer thing to keep in sync.
- **Negative:** on the (rare) exact-HLC tie, the surviving edit is chosen by `agent_id`/
  `event_id`, not by "importance". This is a deterministic, arbitrary-but-stable choice, not
  a semantic one — acceptable, because an exact HLC collision between two concurrent edits of
  the same hash is already a near-degenerate case, and correctness (both hosts agree) beats
  picking the "nicer" edit.
- **Retrieval unchanged:** `quality_score` in `metadata` continues to drive local search
  ranking exactly as before; nothing in the ranking path changes.

## References
- `docs/rfc/planned/rfc-delta-sync.md` §8.2 (R3')
- ADR-0010 (HLC), ADR-0012 (delete-vs-update)
- Gate-1 review (seven) finding #1: quality_score cross-host divergence hazard
