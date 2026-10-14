# RFC: Belief Synthesis — derive_beliefs must distill, not copy

- **Status:** Planned (fork-only). Supersedes the unbuilt parts of
  `docs/rfc/implemented/rfc-s2-belief-store.md` (see "Relationship" below).
- **Date:** 2026-10-14
- **Deciders:** Claudio (owner/client), Zero (agent). Problem definition: seven (Gate 0).
- **Arc:** Belief Synthesis (the learning-loop root cause, measured 14/out).

## Problem (measured, not assumed)

The learning loop's job is to turn use into **reusable, abstract knowledge** (beliefs).
It does not. Three measurements on the live DB (314 active beliefs):

- **Recall@pool = 0.30** (baseline). For 10 queries on themes we actually worked on,
  only 3 have any belief in the pool that answers — all three are stable infra/convention
  facts repeated many times (RFC-fork-only, EIP, geoserver). The 7 missing are
  process/decision knowledge (utility_prior, worktree-per-PR, manager-verifies, …) that we
  *decided/learned* but which never became beliefs. **No ranking recovers what is not in
  the pool** — this is why 5 ranking mechanisms (re-rank A, util, graph, NER, cosine) did
  not help: they rank a corpus missing 70% of the answers.
- **Verbatim ratio = 314/314.** Every belief's content is byte-identical (120-char prefix)
  to one of its source memories. `derive_beliefs` clusters observations and copies the
  cluster representative; there is **no synthesis step**. A "belief" is a memory with a
  badge, not a distilled claim.
- **Source bias = 15/314.** Only 15 beliefs have any `decision/learning/mistake` source;
  578 sources are `observation`. The most actionable knowledge is structurally excluded.

Root cause: **`derive_beliefs` moves text; it does not create knowledge.** It also needs
critical mass (≥N observations of the same thing), so a decision made once never distills.

## Relationship to rfc-s2-belief-store (the divergence)

rfc-s2 (01/jun) specified beliefs as `memory_type=belief` memories promoted by
multi-session confirmation (`_promote_to_belief`) + a scope classifier in the rewriter.
**That was never built as written** (`_promote_to_belief` does not exist in the code). What
shipped (#33, 03/jun) is a different design: a dedicated `beliefs` table populated by
`BeliefService.derive_beliefs` (cluster + copy). RFC and code diverged from day one; the
RFC was marked "implemented" for a system that does not match it. This RFC documents the
REAL `derive_beliefs` and specifies the missing synthesis — it does not amend the fantasy.

## Objective

`derive_beliefs` produces an abstract, reusable CLAIM per cluster (not a verbatim copy),
and the actionable types (`decision/learning/mistake`) reach belief status even from a
single occurrence. Measured success: Recall@pool rises from 0.30 toward ≥0.8 and the
verbatim ratio drops.

## Requirements (EARS)

- **R1** — WHEN deriving a belief from a cluster of observations, THE system SHALL produce
  a synthesized claim (an abstract statement of the shared idea), NOT a verbatim copy of
  any single source memory.
- **R2** — THE synthesis SHALL have a non-LLM deterministic fallback (so derivation never
  hard-depends on an LLM provider), and MAY use the existing rewriter LLM when available.
- **R3** — THE representative/anchor of a cluster SHALL be chosen by informativeness
  (e.g. centrality / coverage of shared terms), NOT by arbitrary loop order
  (`_group_observations` currently takes the first).
- **R4** — THE derivation SHALL include `decision`, `learning`, and `mistake` memory types
  as belief sources (today only `observation` dominates), reusing the ontology filter.
- **R5** — A single actionable memory (`decision/learning/mistake`) that will never reach
  the multi-observation threshold SHALL still be eligible as a singular belief
  (the `MCP_BELIEF_SINGULAR` path already exists — wire it to the actionable types).
- **R6** — THE synthesis SHALL be opt-in and reversible (flag; default off reproduces the
  current verbatim behavior byte-for-byte), so it can be A/B measured before activation.
- **R7** — THE belief content SHALL NOT retain structural noise markers (`TYPE:`,
  `[CHECKPOINT]`, `Session:`); synthesis replaces them with the claim.

## Design (slices, to detail per gate)

1. Representative selection by informativeness (R3) — pure, no dependency.
2. Synthesis step (R1/R2) — deterministic template first (extract the shared claim from
   the cluster's common terms + the most informative source), LLM rewriter as enhancement.
3. Source expansion (R4/R5) — include decision/learning/mistake; wire singular path.
4. Flag + A/B (R6) — measure Recall@pool and verbatim ratio on/off.

## Acceptance criteria (measurable)

- **CA1** — Flag off → derived belief content byte-identical to current (zero regression).
- **CA2** — On a fixed cluster, flag on → belief content differs from every source memory
  (synthesis happened) AND still asserts the cluster's shared claim (sense preserved).
- **CA3** — Recall@pool on the 10-query baseline rises from 0.30 (document the new number);
  verbatim ratio drops from 314/314.
- **CA4** — A single `decision` memory becomes a belief (singular path), proving actionable
  knowledge no longer requires repetition.

## Open questions (for seven rounds before G3)

- What IS synthesis here without an LLM — template over shared terms, or extractive
  (pick the most informative sentence)? How to keep it honest (not hallucinate a claim)?
- Is "useful" definable without `injected_then_used` (which is structurally 0)?
- Should beliefs stay a separate table, or converge to the rfc-s2 `memory_type=belief`
  model? (architectural — decide with Claudio.)

## Out of scope

- Ranking changes (re-rank A, util, graph, cosine) — all downstream of a synthesized
  corpus; do NOT touch here.
- Bootstrap profile / multi-agent isolation (rfc-s2 §5-§7) — separate concern.
