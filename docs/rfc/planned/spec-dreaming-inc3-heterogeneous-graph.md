# SPEC — Dreaming INC 3: grafo heterogêneo tipado

**Status:** planned (reduzido pelo G0) · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 3 · **Pré-req de INC 4/5/7.**

## Achado do G0 (muda o escopo — mem 0b7e1de1)
O `memory_graph` JÁ é tipado: coluna `relationship_type` populada (shares_entity 376k, has_entity 83k, related 52k, derived_from 10k, contradicts, causes, supports, fixes) + `connection_types` JSON (entity/temporal/thematic/shared_concepts). É o vocabulário de edge_type do CONSOLIDADO, VIVO. É tabela só de arestas; "nós" são memórias/beliefs por hash (node_type derivável do memory_type). **NÃO precisa ALTER nem backfill de 528k.**

## Objetivo
Confirmar e DOCUMENTAR que o grafo satisfaz os requisitos de tipagem heterogênea para o INC 4 consumir, sem migração de schema.

## Requisitos (EARS)
- REQ-3.1: THE edge-type vocabulary SHALL cobrir {associated_with/related, member_of, derived_from, mentions/has_entity, supports, contradicts} — mapear os `relationship_type` existentes a esse vocabulário canônico. (confirmação, não ALTER)
- REQ-3.2: WHERE um edge-type canônico não tenha equivalente no `relationship_type` atual, THE sistema SHALL documentar o gap (não criar coluna sem necessidade comprovada pelo INC 4).
- REQ-3.3: THE node-type SHALL ser derivável sem nova coluna (memory→memory_type; belief→tabela beliefs). Documentar a derivação.
- REQ-3.4: SE o INC 4 provar que uma coluna de tipo é necessária, THE ALTER SHALL ser aditivo + idempotente + testado em cópia (backup antes).

## Critérios de aceite
- Documento de mapeamento `relationship_type` → vocabulário canônico, com cobertura e gaps.
- INC 4 consegue consumir o grafo usando só o schema atual (prova de que REQ-3.4 não dispara).

## Invariantes / anti-casos
- NÃO rodar ALTER/backfill de 528k edges "preventivo" — só se o INC 4 comprovar necessidade (risco a dados reais).
- store=COLUNA, não metadata (lição do bug 8017d35c).

## Fora de escopo
Reescrever o grafo. Grafo de entidades separado (entity graph é outro silo — unificação é design futuro, não bloqueia INC 4).
