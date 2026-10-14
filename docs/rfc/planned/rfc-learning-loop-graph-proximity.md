# RFC: Learning-Loop — Graph-Proximity signal in context injection

- **Status:** Planned (fork-only — never an upstream PR; see fork methodology)
- **Date:** 2026-10-14
- **Deciders:** Claudio (owner), Zero (agent). Arch input: seven (Gate 0).
- **Arc:** 40e6ecfb (ex-INC4 repositioned). Depends on: qualidade-de-belief (PR #1533) and
  utility_prior wire (F10) — both done.

## Problem

The proactive belief injection (`memory_context` + re-rank A, `_apply_rerank_a` in
`storage/context_injection.py:301`) ranks beliefs by
`relevance * (1 + λ·util_surplus) + advantage + confidence + freshness − generality`.
For queries about ABSTRACT or NEW themes, relevance-to-theme is weak (lexical+cosine
fusion on short abstract terms) and the historical `utility_prior` then amplifies
GENERIC, highly-reused beliefs. Measured A/B (14/out, live DB): with util converged,
the top-1 was an off-topic infra belief ("EIP cannot be in two places", util 0.98,
util_mult 1.20) while the belief most relevant to the asked theme (relevance 0.46, the
highest in the pool) sat at rank 6 because its util was 0.5 (a new belief with no usage
history). This is the rich-get-richer failure: adding any merit signal (util/confidence)
on top of weak theme-relevance amplifies the generic.

The memory graph (`memory_graph` table: `has_entity` memory→entity edges, `shares_entity`,
SPO entity↔entity) already groups memories by theme, but NO code consumes it in the
search/injection path today (confirmed G0: the only consumers of
`get_entities_for_memory`/`find_memories_by_entity` are the `memory_explore`/`graph`
handlers and the store-time `EntityLinker`, never `context_injection.py`). The graph is a
theme signal with no consumer where it would most help.

## Objective

Add an ADDITIVE graph-proximity signal to the injection re-rank so that beliefs
thematically linked (via shared entities) to the query are surfaced on their theme
relevance, BEFORE the utility term can amplify generic beliefs — attacking the root
(theme relevance) not the symptom. The existing R4 ranking must be preserved exactly when
the signal is off (byte-identical score at weight 0).

## Requirements (EARS)

- **R1** — THE system SHALL compute, once per `memory_context` call, a
  `graph_proximity` value in [0,1] per candidate belief, derived from the overlap between
  the query's entities and the entities of the belief's source memories (`derived_from`).
- **R2** — THE re-rank SHALL add `w_graph · graph_proximity` as an independent additive
  term in `_apply_rerank_a`'s weighted sum, without modifying the relevance, util_mult,
  advantage, confidence, freshness, or generality terms.
- **R3** — WHERE `MCP_INJECT_GRAPH_PROXIMITY` is unset or false, THE system SHALL set
  `w_graph = 0`, producing a `_rank_a_score` byte-identical to the current R4 (zero
  regression, opt-in).
- **R4** — THE re-rank breakdown dict `item['rerank_a']` SHALL gain a `graph_proximity`
  key WITHOUT removing or renaming any existing key (telemetry/test compatibility).
- **R5** — WHERE no query entities can be resolved (no extractor hit, no linked
  entities), THE system SHALL set `graph_proximity = 0` for all beliefs (neutral, never
  negative — absence of a graph link is not evidence against a belief).
- **R6** — THE graph-proximity computation SHALL reuse the existing read primitives
  (`GraphStorage.find_memories_by_entity`, `get_entities_for_memory`,
  `EntityExtractor.extract_entities`) and SHALL NOT introduce new DDL or write to the
  graph.
- **R7** — THE signal SHALL be computed only inside the existing re-rank-A guard
  (`_get_rerank_a_enabled()` AND `embedding_available`), leaving the freshness-sort and
  confidence-sort fallbacks unchanged.

## Design (slices)

1. **Entity resolution for the query:** reuse `EntityExtractor.extract_entities(task)` to
   get the query's entities. (Fallback path, if extractor is empty: the explore pattern
   retrieval→`get_entities_for_memory` — deferred unless R1 proves insufficient.)
2. **Proximity map:** for each query entity, `find_memories_by_entity(entity)` →
   set of source memory hashes; `graph_proximity[belief] = |query-entities covered by the
   belief's derived_from memories| / |query entities|`. Precomputed once in `memory_context`
   next to `_generality_by_query_hash`, passed into `_apply_rerank_a` as a dict (same shape
   as `generality`).
3. **Additive term:** `+ w_graph * graph_proximity` on the `score = (...)` sum
   (context_injection.py:374-380); `w_graph = _rerank_a_weight("graph")`, default 0.0 in
   `_RERANK_A_DEFAULTS`.

## Acceptance criteria (measurable, offline, no LLM-judge)

- **CA1** — With `MCP_INJECT_GRAPH_PROXIMITY` off, the `_rank_a_score` of every belief for
  a fixed pool is byte-identical to pre-change (regression test).
- **CA2** — A/B on the live abstract-theme query (the 14/out case): with the signal on,
  the belief whose source memories share the query's entities rises in rank vs off; the
  generic off-topic belief (no shared entity) does not rise. Reported as rank delta.
- **CA3** — Offline MRR on the adversarial eval set (eval/rerank_bench) does NOT regress
  with the signal at its default-tuned weight.

## Risks (from code, G0 seven)

- The re-rank feeds BOTH the `memory_context` tool and the auto-push on search
  (`MCP_SEARCH_INJECT_CONTEXT`), via the same function — the flag gates both, good for
  consistency, but rollout must be measured on both paths.
- Query entity extraction has no current call site on queries; running
  `extract_entities` per call adds cost — keep it O(entities) and behind the flag.
- Proximity must be neutral (0), never negative, to avoid a second rich-get-richer in
  reverse (penalizing beliefs with no graph link).

## Out of scope

- INC5 (Hebbian edge reweighting via usage_events): a refinement of what the graph
  consumer reads; only valuable AFTER this consumer proves out. Follow-up, not a goal.
- INC7 (memory_explore + LLM-gated reflection): LOW, re-evaluate or cancel after this.
- Changing the R4 fusion (lex+cosine), util model, or any existing term.

## Before enabling (G5 caveats — merge approved ≠ activation approved)

The code is merged opt-in default-off (byte-identical R4 when off, verified). Resolve
these BEFORE turning `MCP_INJECT_GRAPH_PROXIMITY` on in any environment:

- **Entity recall on prose queries:** `EntityExtractor` is term/pattern based; a prose
  task without @mention/#hashtag/URL/path may extract no entities, so `graph_proximity`
  degrades to {} (neutral R4) — the feature is then inert, not harmful. Before activation,
  measure entity recall on real abstract-theme queries (the 14/out case) and, if low,
  decide whether to seed `MCP_ENTITY_CUSTOM_TERMS` or add a retrieval→has_entity fallback
  (the explore pattern). CA2 only bites if the theme's entities are actually extractable.
- **CA3 (offline MRR) not yet exercised:** run `eval/rerank_bench` with the signal on at
  the tuned weight and confirm no MRR regression before activation.
- **Weight tuning:** `MCP_INJECT_RERANK_A_W_GRAPH` default 0.0; pick the activation weight
  from the A/B (CA2) + MRR (CA3), not by guess.
