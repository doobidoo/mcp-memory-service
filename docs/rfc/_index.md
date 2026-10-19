# Índice de RFCs — mcp-memory-service (fork)

> Specs de trabalho do fork. Classificação VERIFICADA contra o código (atualizado 4/out): a feature está em `src/`? → `implemented/`; senão → `planned/`. NÃO pelo label do doc.
> Navegação por arcos: `docs/rfc/README.md`. Decisões de arquitetura: `docs/adr/`. Acompanhamento: `docs/_fork/ESTADO.md`.
> Regra: mudou → atualiza. Ao implementar: `git mv planned/ implemented/`.

## implemented/ (24 — feature presente no código, com evidência)

| RFC | Arco | Evidência / nota |
|-----|------|------------------|
| rfc-quality-model | Rating | #1349 mergeado — arco FECHADO |
| rfc-nli-cascade | Reasoning | #1215 mergeado (reasoning/nli.py) |
| rfc-harvest-provenance | Ingestão (transv.) | #1243 mergeado |
| rfc-agent-id-multi-agent | Hub | #1278/#1297 (F1/F2) mergeados |
| rfc-harvest-design-extraction | Ingestão C3 | ABSORVIDA; I0/I0-lang/I1 no código |
| rfc-harvest-source-identity | Ingestão C1 | ABSORVIDA (camada 1) |
| rfc-harvest-kiro-sessions | Ingestão C2 | ABSORVIDA (camada 2); #1378/#1379 mergeados |
| rfc-g0-design-extractor-i2 / rfc-g0-i1-toolresults-cli-kind / rfc-g0-store-ner | Ingestão/NER | gates G0 executados |
| rfc-inventario-sessoes-kiro / rfc-recuperacao-sessoes-kiro | Ingestão | levantamento feito |
| rfc-s2-belief-store | Infra | consolidation/belief_service.py + migration 012 |
| rfc-s6-anti-hallucination | Infra | consolidation/quarantine.py + contradictions.py + nli.py |
| rfc-s8-handler-extraction | Infra | server/handlers/*.py |
| rfc-s8-tool-registry | Infra | tools/registry.py |
| rfc-s9-split-config | Infra | config/ modularizado (15+ módulos) |
| rfc-s13-tool-registry-wiring | Infra | tools/routing.py + server_impl list_tools |
| rfc-i11-schema-versioning | Infra | storage/migration_runner.py + migrations/*.sql |
| rfc-multi-store-domain-ner | NER | reasoning/store_terms.py + ner_patterns/ |
| rfc-sync-multi-agente | Hub | scripts/sync/*.py + web/api/sync.py — OPERACIONAL |
| rfc-pipeline-harvest-quality | Ingestão | harvest/extractor.py (sentence + confidence gate) |
| rfc-config-audit-2026-07-10 | Infra | fixes aplicados (graph_only, store_associations, schema_version) |
| rfc-fact-extraction | Aprendizado | SUPERSEDED by rfc-mm-02 (geração anterior, histórico) |

## planned/ (24 — design/draft)

> Materializados no código (4/out, telemetria+injeção): `usage_events` + `usage_telemetry.py` (feedback passivo L4, mm-01) e `memory_context` (injeção proativa L3). Ver docs/adr/ e docs/rfc/README.md.

| RFC | Arco | Nota |
|-----|------|------|
| **rfc-learning-loop** | **Aprendizado (guarda-chuva do PORQUÊ)** 🟡 parcial | **L3 injeção (memory_context) + L4 telemetria (usage_events) NO CÓDIGO; L2 belief store destila. Linhagem: self-service→autolearn→mm-01/02/03→learning-loop. ADR-0003/0004/0005.** |
| rfc-ingestao-multi-agente v0.3 | Ingestão multi-agente | guarda-chuva das 3 camadas |
| **rfc-hermes-first-class v0.1** | **Harness (Use in any agent)** 🟡 draft | **Filha da rfc-memory-portability, que deixa o Hermes fora de escopo ("já usa"). Cobre os 8 pontos que separam "conectado" de "de primeira classe": modo híbrido (D1), provider opcional (D2), doctor (D3), autoria #1598 (D4), importador do state.db (D5), hooks TUI/CLI (D6), instalador (D7), porta (D8).** |
| spec-fase0-kiro-yaml-triagem | Ingestão multi-agente | filha executável (Kiro→YAML + triagem) |
| rfc-hub-memoria-centralizada | Hub/Sync | transporte HISTÓRICO — ADR-0002 escolheu delta-sync |
| rfc-delta-sync | Hub/Sync | **transporte ESCOLHIDO (ADR-0002)** · §8 invariantes #1345 portados |
| rfc-memory-portability | Portabilidade | #1364 wiki; 5 camadas |
| rfc-importers | Portabilidade | mem0/letta/zep (#1390) |
| rfc-mm-02-fact-extraction | Aprendizado/Extração | L2 destilação (absorve fact-extraction, agora superseded) |
| rfc-mm-01-feedback-loop | Aprendizado/L4 | 🟡 telemetria no código; falta job recálculo (ADR-0003) |
| rfc-mm-03-gap-detection | Aprendizado | filha indep.; 0 EARS, completar |
| rfc-autolearn-autodream | Aprendizado | 🟡 memory_distill feito; diagnóstico §1.3 → ADR-0003 |
| rfc-self-service-memory-intelligence | Aprendizado | 🟡 geração anterior absorvida (P1/P3/P4/P5/P8 no código) |
| rfc-server-side-lifecycle | Aprendizado/infra | 🟡 parcial (consolidação server-side) |
| rfc-embedding-quantization / rfc-ranking-upgrades / rfc-query-intent / rfc-working-memory (reusa memory_context, ADR-0004) / rfc-persona-tier / rfc-multimodal-memory / rfc-memory-hygiene / rfc-skill-auto-generation | Diversos/backlog | peças ortogonais, ver README árvore |
| rfc-onboarding-discoverable / rfc-structural-improvements (🟡 §8/§13 feitos) / rfc-kiro-headless-llm-provider | Diversos | design |

## superseded/ (em implemented/, mantidos como histórico)
| rfc-fact-extraction | → rfc-mm-02-fact-extraction | duplicata, geração anterior (ADR via README) |

## Convenções
- Nome: `rfc-<feat>.md` minúsculo. Implementou → `git mv planned/ implemented/`.
- Absorvida por guarda-chuva: manter arquivo, marcar "ABSORVIDA → <arco>" no topo + aqui.
- Classificação por EVIDÊNCIA de código, não por label do doc.
