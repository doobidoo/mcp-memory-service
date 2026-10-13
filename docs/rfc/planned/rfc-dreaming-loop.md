# RFC: Dreaming-Loop — consolidação, grafo heterogêneo e navegação por tema (INC 2-7)

**Status:** planned · **Created:** 2026-10-13 · **Fork-only (maturing) · dreaming-loop (ARC 07bcf182, item FASE 3 = 03c1d773)**
**Base:** CONSOLIDADO G0 arch + 9 IAs (CdIA pesquisa/dreaming-loop-navegacao-2026-10). INC 1 já implementado = R1/F9 (rfc-dreaming-loop-inc1-distill-specific.md).

## Problem
O learning tem o lado "acordado" (destilar/re-rank/feedback — F1-F9 feitas) mas o lado "sono"
(dreaming offline) está incompleto. Dores vivas do CLIENT (veredito 13/out, mem 00f02b0c):
1. **Loop lento** — derive só roda 03:00/domingo; o que aprendo hoje não vira belief hoje.
2. **Vem por palavra, não por tema** — a injeção ranqueia por lexical+embedding do belief e
   IGNORA as associações temáticas. PRODUTOR de associações vivo (memory_graph 528k edges,
   atualizado a cada consolidação) mas CONSUMIDOR não existe (achado 1d84a7c4).
3. **Fragmentos-lixo** destilados ('TYPE:', 'convention', 'Causa Raiz Identificada:').

## Objective
Fechar o dreaming-loop: consolidar/associar/agrupar por TEMA (não só janela temporal) e fazer
esse conhecimento ALIMENTAR a injeção, de modo que a busca traga o tema certo e o aprendizado
recente fique disponível rápido — sem LLM no hot-path, byte-reversível por flag.

## Requirements (EARS)

### INC 2 — consolidação por tema (`memory_consolidate(theme=...)`)
- REQ-2.1: WHEN memory_consolidate is called with a `theme`, THE system SHALL run the existing
  6 consolidation phases restricted to the subset of memories matching the theme.
- REQ-2.2: THE theme selection SHALL reuse semantic retrieval to pick candidates (gap = seleção,
  não pipeline); SHALL NOT alter `_get_memories_for_horizon` (derivação de belief já é global).
- REQ-2.3: opt-in / backward-compatible (sem theme = comportamento atual).

### INC 3 — grafo heterogêneo tipado (pré-req de 4/5/7)
- REQ-3.1: THE memory_graph SHALL carry `node_type` ∈ {memory,entity,cluster,belief} and
  `edge_type` ∈ {associated_with,member_of,derived_from,mentions,supports,contradicts}.
- REQ-3.2: THE migration SHALL be additive (ALTER add column, default backfill) and idempotent;
  SHALL NOT rewrite the graph nor break existing 528k edges.
- REQ-3.3: Invariante — episódio=ground-truth, belief=única unidade de injeção, cluster=view
  derivada, associação=aresta de hipótese. (semântica dos tipos, não só rótulo)

### INC 4 — associações→injeção (a peça do "por tema")
- REQ-4.1: WHEN memory_context(task) runs, THE injection SHALL expand the vector-seeded result
  via spreading activation over memory_graph (≤2 hops, budget-bounded).
- REQ-4.2: THE expansion SHALL apply a CONVERGENCE RULE — an association only enters if BOTH
  endpoints are activated by INDEPENDENT paths (anti-ruído); plus similarity sweet-spot +
  centrality cap.
- REQ-4.3: THE hot-path SHALL NOT call any LLM.
- REQ-4.4: opt-in flag; OFF = comportamento atual (byte-reversível).

### INC 5 — loop sono→acordado (reweighting Hebbiano)
- REQ-5.1: THE edge weights SHALL update from usage_events: `edge.weight += η·util`;
  presented-and-ignored decays; half-life applied.
- REQ-5.2: Updates SHALL be EMA (never single-jump) to avoid crystallizing noise.

### INC 6 — gatilho por importância por tema (ataca loop lento)
- REQ-6.1: THE system SHALL trigger a theme-scoped consolidation when accumulated importance for
  a theme crosses a threshold, AS A COMPLEMENT to the cron (não substitui).
- REQ-6.2: Importance SHALL derive from EXISTING signals (usage_events/Frente D), NOT an LLM;
  novelty counted PER THEME.
- REQ-6.3: Guards against Generative-Agents "confident hallucinations".

### INC 7 (opcional) — memory_explore unificado + reflection LLM-gated
- REQ-7.1: memory_explore SHALL traverse the heterogeneous graph.
- REQ-7.2: reflection SHALL be opt-in, sampled offline, OUT of the hot-path.

## Design (per-INC, do CONSOLIDADO)
- INC2: filtro de candidatos por tema antes das 6 fases; sem DDL.
- INC3: ALTER memory_graph + backfill; estende o que existe (dual_write/graph_only já gravam).
- INC4: SA/PPR semeado pela busca vetorial; regra de convergência é o anti-ruído central.
- INC5: EMA + half-life sobre arestas, alimentado por usage_events (reusa Frente D).
- INC6: reflection-trigger por importância acumulada; reusa sinais existentes.

## Out of scope
- Reescrever o grafo (só estender). LLM no hot-path. Cripto. Backends não-sqlite.

## Acceptance criteria (measurable, por INC)
- INC2: consolidar com theme='X' processa só memórias de X; sem theme = idêntico ao atual.
- INC3: migração aplica em cópia, 528k edges intactos, node_type/edge_type populados por backfill.
- INC4: memory_context de um tema traz, no top-k, um item alcançado SÓ via associação (não pelo
  lexical/embedding direto) — e a regra de convergência barra um associado espúrio (teste).
- INC5: aresta usada sobe weight; aresta ignorada decai; sem salto único (EMA).
- INC6: acúmulo de importância num tema dispara consolidação fora do cron.
- VEREDITO CLIENT final: memory_context do tema trabalhado traz o conhecimento certo no topo,
  mesmo-dia. Se feliz → F4 upstream (OK Claudio + Henry).

## Riscos (do CONSOLIDADO)
3 (ruído sem convergência→INC4), 5 (cristalizar ruído→EMA+half-life), 6 (O(n) retrieve por grupo
→ custo), 7 (ALTER+backfill idempotente). Todos endereçados nos REQs.

## Ordem (dependências)
INC2 ∥ INC3 (0 dep) → INC4 (dep 3) → INC6 (dep 2) → INC5 (dep 3+4) → INC7 (dep 3, opc).
Gate completo por INC (rok RED → reg GREEN → tuvok PASS) + E2E + veredito CLIENT. Fork-only.
