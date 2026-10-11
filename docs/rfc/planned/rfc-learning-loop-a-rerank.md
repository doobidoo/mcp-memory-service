# RFC: Learning-Loop Frente A — Re-rank for Context Precision ("tiro certeiro")

**Status:** planned · **Created:** 2026-10-10 · **Fork-only (maturing) · learning-loop frente A (RFC §14)**
**Author:** Claudio + Zero · **ARC:** Learning Agêntico (07bcf182) · **Depends on:** G0 (seven, 10/out) + pesquisa multi-IA (8 respostas)
**Research:** CdIA `pesquisa/learning-loop-rerank-attribution-2026-10/CONSOLIDADO-A.md`

## Problem

Injection ranks beliefs by `relevance * confidence` as the primary term
(`context_injection.py:312-325`). This structurally favors GENERIC, heavily-reinforced,
high-confidence beliefs over the SPECIFIC rare ones that actually prevent the concrete
error. Confidence is a **proxy for frequency**: `belief.py:75`
`conf = sigmoid(support_score − λ·contradict_score)` grows with the number of supporting
observations, so a belief that "matches everything" accumulates support → high confidence
→ wins. This is the classic **popularity/frequency bias** (Steck, RecSys 2020).

Reproduced live (10/out): `memory_context("re-rank context precision belief específico
raro vence genérico")` returned at top a GENERIC belief ("RFC/spec primeiro sempre",
conf 0.98, relevance 0.28) plus cross-domain NOISE ("EIP preso no NAT", relevance 0.23).
In the fisioterapia analogy (learning-loop §0): on the exam, the agent is handed the
platitude instead of the specific, decisive note. This is Dimension 1 (Recall precision)
+ the "tiro certeiro" the whole learning-loop is aiming at.

## Objective

Make the injection reranker value SPECIFICITY/RARITY + theme-adherence, so the specific
decisive belief beats the generic one — local-first (sqlite-vec, O(1) hot path), with
cross-encoder/LLM only opt-in and gated. Reuse the existing `reasoning/ranked_search.py`
(`RankedSearchWeights` + `compute_ranked_score`) weighted-sum template.

## Requirements (EARS)

- **R1 (weighted sum):** THE injection reranker SHALL compute a per-belief score as a
  weighted sum of relevance, specificity, generality-penalty, saturated-confidence,
  historical-utility and freshness terms (replacing the primary `relevance*confidence`
  product at `context_injection.py:312-325`).
- **R2 (preserve freshness):** THE reranker SHALL integrate the existing freshness term
  (frente C, already merged) as a weighted component, NOT compete with it; WHEN
  `MCP_INJECT_FRESHNESS` is off, the freshness weight is 0.
- **R3 (specificity):** THE system SHALL compute per-belief specificity from IDF
  propagated via `derived_from` over `memory_content_fts`, and SHALL persist it in
  `beliefs.metadata` (JSON, no DDL — precedent: frente B counters).
- **R4 (generality penalty):** THE reranker SHALL penalize beliefs by a generality score
  computed as the fraction of distinct `query_hash` values in whose injection the belief
  appeared, over total injections observed (from `usage_events`); query text is never
  stored, so coocurrence by `query_hash` is the only signal.
- **R5 (advantage fallback):** WHERE beliefs lack their own embedding, THE reranker SHALL
  use advantage-over-the-field (`sim − mean_pool_sim`, O(n), no precompute) in place of
  embedding-based specificity.
- **R6 (saturated confidence):** THE reranker SHALL derive the confidence term as
  `log1p(support)/log1p(CAP)` so reinforcement beyond CAP observations SHALL NOT increase
  the confidence term (decouples confidence from frequency).
- **R7 (historical utility):** THE reranker SHALL compute historical-utility as a
  Laplace-smoothed useful-injection rate `(useful+0.5)/(injected+1)` from `usage_events` (L4).
- **R8 (stopwords PT-BR):** THE term-overlap relevance SHALL use an expanded PT-BR
  stoplist (today `context_injection.py:72` is EN-only; "sempre/primeiro/importante" leak).
- **R9 (fallback no-embedding):** WHEN no embedding model is available THE reranker SHALL
  fall back to confidence-only ordering (preserve current REQ-7 behavior).
- **R10 (weights via env):** THE reranker SHALL expose per-term weights via environment
  variables with normalized defaults, mirroring `RankedSearchWeights`, and SHALL emit a
  per-belief score breakdown into `debug_info`.
- **R11 (MMR, opt-in):** WHERE `MCP_INJECT_RERANK_MMR` is enabled THE reranker SHALL apply
  MMR (λ default 0.7) to the final top-k selection. (Requires belief embeddings — see Risks.)
- **R12 (cross-encoder, opt-in):** WHERE a cross-encoder is enabled THE reranker SHALL
  rerank only the top-20 candidates and SHALL fall back to the heuristic ordering if the
  latency budget is exceeded.
- **R13 (no double-count A×B×C):** THE reranker SHALL define an explicit order of
  application with frente B (`_apply_unused_penalty` on confidence) and frente C
  (freshness) so a belief is NOT penalized/boosted twice for the same signal.

## Design (slices)

- **Inc 0 (precompute, no behavior):** offline job writes `spec_idf` + `gen_score` into
  `beliefs.metadata`. No DDL. Unit-tested on the computation.
- **Inc 1 (CORE, O(1) hot path, no new dep — DELIVERS THE FIX):** replace the primary
  ordering term with the R1 weighted sum, reusing `RankedSearchWeights`. Fixed weights,
  env-override. Expand PT-BR stopwords (R8). Resolve R13 ordering first.
- **Inc 2 (opt-in):** MMR (R11) — needs belief embeddings; verify before promising.
- **Inc 3 (opt-in):** FTS5/BM25 over beliefs (new virtual table) + RRF fusion.
- **Inc 4 (opt-in, fallback):** cross-encoder ONNX bge-reranker-v2-m3 (PT-BR) over top-20.
- **Inc 5 (later, ≥500 labels):** calibrated logistic LTR over the features.

## Acceptance criteria (measurable, offline, no LLM-judge)

- [ ] GIVEN the gold set of specific queries, THE reranker places the specific-relevant
  belief above the generic one in ≥X% of cases — **generic-wins rate drops** vs baseline.
- [ ] MRR@5 is maintained or improved vs baseline **0.4140** (no recall regression).
- [ ] Measured via `IDBasedContextPrecision@5` + `MRR@5`, deterministic, no LLM.
- [ ] Ablation rel-only → +spec → −gen → +util → +MMR, delta reported per query category.
- [ ] CLIENT veredito: live `memory_context("...")` returns a SPECIFIC top, no EIP/NAT
  noise, no "RFC/spec primeiro" drowning the specific.

## Risks (from code, G0 seven)

- R1 beliefs may lack embeddings → MMR/embedding-specificity become extra deps; use
  advantage-over-the-field (R5) in the MVP.
- R2 `relevance` is term-overlap, not cosine → inherits PT-BR stopword weakness (R8 fixes).
- R3 raw query never stored → generality only by `query_hash` coocurrence (R4).
- R4 **A×B×C collision** on confidence/freshness — define order (R13), a design decision.
- R5 CONFIDENCE_FLOOR 0.35 vs candidate floor 0.0 — reranker sees sub-floor beliefs (good
  for the rare specific, but confirm no junk enters).

## Out of scope

- Chunk attribution / feedback loop that feeds utility — that is **frente D** (separate RFC).
- Dreaming-loop navigation / associations in injection — separate arc (dreaming-loop RFC).
