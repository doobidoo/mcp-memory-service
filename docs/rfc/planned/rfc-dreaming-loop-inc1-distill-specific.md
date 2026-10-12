# RFC: Dreaming-Loop Inc 1 — Distill the recent SPECIFIC (make it injectable same-day)

**Status:** planned · **Created:** 2026-10-10 · **Fork-only (maturing) · dreaming-loop (ARC 07bcf182 Fase 3)**
**Author:** Claudio + Zero · **Depends on:** G0 (seven) + pesquisa multi-IA (9 respostas)
**Research:** CdIA `pesquisa/dreaming-loop-navegacao-2026-10/CONSOLIDADO.md`

## Problem

A specific, decisive piece of knowledge produced TODAY (e.g. the delta-sync empty-content
deadlock fix) does NOT become an injectable belief. Verified live (CLIENT, 10/out): zero of
235 active beliefs mention delta-sync/deadlock/skippable, the most-worked theme of the day.
Four cascading barriers in `belief_service.py` kill it before injection:

1. `derive_beliefs()` (:131) reads ONLY `memory_type="observation"` — a fix stored as
   `decision`/`learning` never enters derivation.
2. `_NOISE_PREFIXES` (:38-47) drops `[CHECKPOINT]`/`[SESSÃO]`.
3. `_is_noise` (:55-74): >500 chars = "dump", bullet-ratio>60% = "list" → a detailed fix is dropped.
4. `PROVENANCE_FLOOR=2` (belief.py:27; should_promote :92-94): a fix that happened ONCE never
   reaches 2 supporting observations → never promotes to `active` → never injectable.

So the whole re-rank (frente A) and freshness (frente C) work on a corpus that is missing the
very item the agent needs. No ranking fixes an absent target. This is the real "tiro certeiro"
gap: distillation, not ranking.

## Objective

Give a recent SPECIFIC fix/decision a path to become an injectable belief the SAME DAY, WITHOUT
weakening the confidence calibration of the ~235 existing derived beliefs. Keep pure
checkpoints/session dumps out.

## Requirements (EARS)

- **R1 (configurable intake types):** THE belief derivation SHALL consider memories of
  configurable types (default: observation + decision + learning + error) as derivation input,
  not observation-only (today belief_service.py:131).
- **R2 (don't drop actionable detail):** WHEN a memory carries an actionable marker
  (type ∈ {decision, learning, error}) THE noise filter SHALL NOT discard it via the
  length>500 nor the bullet-ratio>0.6 rules (today :64-74 would).
- **R3 (singular belief):** WHILE a candidate has exactly 1 supporting observation of an
  actionable type, THE system SHALL create a distinct "singular" belief with an EXPLICIT
  (fixed) confidence, WITHOUT requiring PROVENANCE_FLOOR=2 and WITHOUT feeding the
  Σ-observation confidence calibration used by derived beliefs.
- **R4 (preserve checkpoint filter):** THE `[CHECKPOINT]`/`Session:` prefix filter SHALL keep
  dropping pure checkpoints/session dumps, so R2 does not re-admit volatile state.
- **R5 (same-day injectable):** WHEN memory_context(task) runs, THE system SHALL surface a
  relevant singular belief within the token budget, ranked with the others.
- **R6 (opt-in, reversible):** THE new intake behavior SHALL be gated by an env flag (default
  preserves current behavior), byte-reversible (repo pattern, cf. MCP_INJECT_RERANK_A).

## Design (slice Inc 1 only)

- The "singular" belief is a SEPARATE construct, not a lowered floor: a belief row with an
  explicit confidence (e.g. a fixed `MCP_SINGULAR_BELIEF_CONFIDENCE`, bounded) and a marker in
  metadata (`origin: "singular"`), so `derive_confidence` (Σ support − λ·contradiction) is never
  applied to it and the derived-belief calibration is untouched (RISK #1).
- Intake type set via env (`MCP_BELIEF_INTAKE_TYPES`), default adds decision/learning/error.
- `_is_noise` gains an `is_actionable` bypass for the length/bullet rules ONLY (prefix/pattern
  checkpoint filters stay — R4).

## Acceptance criteria (measurable)

- [ ] GIVEN one specific fix stored as type=learning (~300 chars, not a checkpoint), WHEN
  derive_beliefs runs, THEN a singular active belief is created and appears in memory_context
  for its theme the same day.
- [ ] The confidence distribution / promotion of the existing derived beliefs is UNCHANGED when
  the flag is off (regression), and derived beliefs still require PROVENANCE_FLOOR=2.
- [ ] Pure `[CHECKPOINT]` entries are still dropped (R4 test).

## Out of scope (later increments)

- Unified heterogeneous graph (node_type/edge_type) — Inc 3.
- Associations → injection via spreading activation — Inc 4.
- Hebbian edge reweighting (sono→acordado loop) — Inc 5.
- memory_consolidate(theme=...) — Inc 2.

## Open design (needs Claudio — touches the live 235-belief core)

- Exact representation of "singular" confidence: fixed value vs a separate column vs metadata
  flag. Must NOT contaminate `derive_confidence`. This is the RISK #1 the G0 flagged; it changes
  the core of belief derivation, so it gets explicit sign-off before implementation.

---

## Addendum (12/out) — R1 refinado: usar a ONTOLOGIA + filtrar STORE (medido)

Diagnóstico empírico (veredito CLIENT + medição) refinou o R1 original:

**Achado 1 — a ontologia já resolve o intake.** `derive_beliefs` (belief_service.py:425) filtra
`memory_type="observation"` LITERAL, ignorando a hierarquia de `models/ontology.py`. Como `note`,
`insight`, `bug`, `mistake` são SUBTYPES (de observation/learning/error), eles nunca entram — mesmo
corretamente classificados. FIX: usar `get_parent_type()` / `get_base_types()` da ontologia para
aceitar base types `{decision, learning, error, observation}` resolvendo subtypes. NÃO hardcodar lista
(usa a taxonomia já definida, não cria paralela).

**Achado 2 — filtrar por STORE (multi-store = RAG).** `derive_beliefs` não filtra `store`. O multi-store
existe para o serviço servir de RAG (material HCSO/MIR ingerido em stores `docs`/`docs-shared`/`hackathon`/`mir`).
Beliefs devem destilar SÓ do `default` (operacional), não do RAG. FIX: `get_all_memories(..., store="default")`.
Corrige de brinde o mesmo defeito de store-não-propagado dos #1465/#1466.

**Medição (cópia, store=default):** HOJE 5750 (observation literal) → R1 18147 (3.2x, +12397), composto por
observation 8083 + learning 4386 + error 3721 + decision 1957 — quase tudo acionável (o ruído `document`
de ~8k ficou isolado nos stores de RAG, fora do default). Resíduo: 665 `document` vazaram para o default
(disciplina de store, não bloqueia).

**R6 reforçado:** opt-in + cap de batch obrigatório (3.2x de intake sem cap floodaria num ciclo).
Medir QUANTOS beliefs SAEM (agrupamento condensa) antes de ligar no vivo.
