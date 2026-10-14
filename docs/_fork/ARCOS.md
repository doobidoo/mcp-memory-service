# ARCOS — mapa de trabalho (agente)

> Doc DENSO para carga rápida do agente. Onde estamos, por arco, cruzando RFC ↔ estado ↔ próximo passo.
> Regra: mudou → atualiza AQUI (e no ESTADO.md a árvore). Atualizado: 2026-10-09.
> Fonte de verdade dos arcos. RFCs em `docs/rfc/{planned,implemented}/`. Estado operacional em `pilha-prs-runbook.md`.

## Snapshot
- **main fork:** 0 atrás do upstream, 246 à frente (09/out). Reconciliada pós-#1489 + L4 quality-recompute preservado.
- **ARCO DELTA-SYNC (#1345): 4/5 fases MERGED** — #1478 (F1 event-log), #1485 (F2 HLC+resolver), #1487 (F3 embedding consistency), #1489 (F4 pull+push+scheduler). **Fase 5 (per-spoke identity + bootstrap) é o próximo e FECHA o arco.**
- **ARCO #1304 (hybrid HTTP secondary): FECHADO** — 4 fases MERGED (#1470/#1471/#1474/#1476). Raiz que destravou o delta-sync. PR #1480 (CF-guard) pode estar pendente.
- **ARCO rating/quality: FECHADO** (#1349/#1368/#1391/#1404).
- **ARCO learning-loop:** L1/L2/L3/L4 fork-only. L3 push CONFIRMADO disparando (10/out). L4 DESTRAVADO: fix proveniência (source_hashes, 63ddfaaa) tirou injection_coverage de 0.0 estrutural → 0.013 vivo. Frentes B (sinal-negativo abbea491) ✅ C (freshness 5b8c6d1f) ✅ E (eval adversarial 708dcd38) ✅ TODAS FEITAS via gate (G0 seven→G3 rok→G4 reg→G5 tuvok). O loop agora é MENSURÁVEL E2E: eval offline emite MRR/taxa_freshness/taxa_forgetting e DISCRIMINA feature-on/off. ARC e3dc542d. PRÓXIMO: empacotar TODO learning-loop em PRs atômicos upstream (85febdca). **A/D RESOLVIDOS (11/out, ARC 07bcf182 fechado no valor):** A re-rank (727d9757, MCP_INJECT_RERANK_A) + dreaming singular belief (0ed9576e/fe735300, MCP_BELIEF_SINGULAR) + D1 utility_prior passivo (b9a1d43e, MCP_BELIEF_UTILITY_PRIOR, separado de confidence) — TODOS no vivo. Roadmap D2/D3/D4 anotado (9 IAs).
- **ARCO harvest-de-sessões (NOVO, 11/out, ARC 323e1654):** learning de COLHEITAS (par do learning de memórias). Sessões .jsonl sub-colhidas (61% nunca). FASE 1 fix do tracker (667→19 regressão) ✅ c5f3a3a4 no vivo + tracker reconstruído 19→1299. FASE 1b fix do triage cego ao formato legado (causa-raiz real do furo — descartava 100%) ⚠️ workaround ativo, fix pendente. FASE 2 backfill 788 pendentes 🔨. FASE 3 scheduler / FASE 4 resumo-por-sessão 🔴. Incidente embedding ausente resolvido no caminho.
- **Foco atual (09/out):** Fase 5 do delta-sync — fechar o arco completo.
- **⭐ ARCO SÍNTESE DE BELIEF (NOVO, 14/out — a DOENÇA do learning-loop, não o sintoma):** medido no vivo que derive_beliefs NÃO destila, COPIA (314/314 verbatim); Recall@pool=0.30 (7/10 temas trabalhados não têm belief que responda); 15/314 beliefs têm fonte decision/learning (observation domina). TODO o maquinário de ranking (re-rank A, util, grafo, NER, cosine) afinava a recuperação de um corpus sem síntese — "leme perfeito, barco sem motor". Autoria NOSSA (#33). RFC rfc-s2-belief-store nunca foi implementada como escrita (código divergiu dia 1) → RFC nova docs/rfc/planned/rfc-belief-synthesis.md (G1): sintetizar claim abstrato (não copiar) + ingerir decision/learning + singular path. Resgatado da rfc-s2: subtipos de belief, scope global vs project, promoção emerging→active. Baseline de aceite: Recall@pool 0.30→≥0.8. INC5/INC7/ligar-grafo ficam ABAIXO (ranking sobre corpus-sem-síntese reencena rich-get-richer). mem 0ab1aff0/bcbc468b/25a0766.

## Arcos

| Arco | Estado | RFCs (docs/rfc/) | Próximo passo |
|------|--------|------------------|---------------|
| **Delta-sync (#1345)** | ✅ 4/5 fases MERGED | planned/rfc-delta-sync (v0.4) + spec-fase1..4d | **Fase 5: per-spoke identity (§8.4) + bootstrap (§9.3)** — fecha o arco; depois op (schedule regime, 3 máquinas, aposentar Insync) |
| **Ingestão multi-agente** (NOVO guarda-chuva) | 🟡 design + Henry respondeu via #1346 | planned/rfc-ingestao-multi-agente v0.3 — discussion #1393 (só nosso comentário) | PR ponto-1 de #1346: coverage_report() visível no resultado+log do harvest; depois update RFC (compat #2, kill-switch #3, legacy ToolResult #4) e comentar #1393 linkando |
| ├ Camada 1: registro/descoberta de fontes N | 🔴 design | implemented/rfc-harvest-source-identity (ABSORVIDA) | declarativo + auto-descoberta assistida |
| ├ Camada 2: perfil de parsing por agente (YAML) | 🔴 design | implemented/rfc-harvest-kiro-sessions (ABSORVIDA) | spec-fase0 escrita; triage.py consolidado (193 testes); aguarda aval #1393 |
| ├ Camada 3: extração/qualidade sinal-ruído+LLM | 🟡 parcial | implemented/rfc-harvest-design-extraction (ABSORVIDA) | heurísticas declarativas 1º, LLM depois |
| │  ├ I0 coverage instrument | ✅ #1350 merged | — | — |
| │  ├ I0-lang idioma | ✅ #1379 merged | — | — |
| │  ├ IA discovery / IB SQLite | ✅ #1378/#1379 merged | — | — |
| │  └ I1 ToolResults CLI {kind} | 🟢 fork (78e29b05) | — | consolidar em PR upstream quando arco amadurecer |
| ** Rating / quality-model** | ✅ FECHADO | implemented/rfc-quality-model | — #1349+#1368+#1391(decay, mergeado 1/out) fecharam. #1352 (supersession orphan) → PR #1404 nosso, área storage |
| **Portabilidade** (bring your memory ⊕ use in any agent) | 🟡 design aceito | planned/rfc-memory-portability, planned/rfc-importers | wiki Memory-Portability-Map viva; conversor mem0 (#1390); parte absorvida pela ingestão multi-agente |
| **Hub memória multi-agente** | 🟡 parcial | planned/rfc-hub-memoria-centralizada, planned/rfc-delta-sync, planned/rfc-sync-multi-agente, implemented/rfc-agent-id-multi-agent | agent_id F1/F2 mergeados; falta F3-F8 (estrela, consolidação nas pontas, NLI cross-agent) |

## Sub-arcos ativos / decisões abertas
- **Redação de segredos** (todo #5): NÃO no store local (mesmo disco). No PONTO DE SAÍDA (sync hub / LLM). Ação futura.
- **RFC guarda-chuva ingestão multi-agente:** a redigir → levar à discussion (Henry decide design antes do código) → materializar specs por camada.
- **1º passo executável (aceito pelo Henry):** extrair regras hardcoded de UM agente (Kiro) para `harvest/agents/kiro.yaml` (refactor dado→config, golden test de cobertura byte-idêntica). Não misturar com heurísticas/LLM.

## Issues/PRs upstream vivas
- #1390 (tracking conversor mem0) — aberta. #1345 (delta-sync, colab ducanhnguyen223) / #1346 (design-extraction) — RFCs abertas.

## Índice reverso RFC → arco → status
Ver `docs/rfc/_index.md` para a lista completa. Absorvidas pela ingestão multi-agente: harvest-source-identity (C1), harvest-kiro-sessions (C2), harvest-design-extraction (C3).

## Docs irmãos (navegação)
- **ESTADO.md** — versão legível deste mapa (para o Claudio).
- **roadmap.md** — épicos com árvore/esforço. **pilha-prs-runbook.md** — estado operacional dos PRs.
- **reference-pipeline.md** — arquitetura do pipeline de memória.
- **../rfc/_index.md** — índice de todas as RFCs (planned/implemented).
