# SPEC — Dreaming INC 4: associações→injeção (navegação por tema)

**Status:** planned · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 4 (HIGH) · **Dep:** INC 3 (satisfeito pelo schema atual).
**Histórico:** 1ª tentativa revertida (tag backup/inc4-subtractive-20261013) por sair SUBTRATIVA + convergência bypassada. Esta spec fixa os invariantes que faltaram. Mistake 436d3a7b (pular spec).

## Problema
A injeção (`memory_context`) ranqueia por lexical+embedding do belief e IGNORA o grafo de associações (528k arestas, produtor vivo). Dor do CLIENT: "memória vem por palavra, não por tema". Falta o CONSUMIDOR (achado 1d84a7c4).

## Objetivo
Expandir o resultado da injeção com memórias/beliefs alcançados via associação temática no grafo, SEM remover nada do resultado atual e SEM introduzir ruído.

## Requisitos (EARS)
- REQ-4.1: WHEN `memory_context(task)` roda com a flag ON, THE injeção SHALL EXPANDIR o conjunto semeado pela busca vetorial com candidatos alcançados via `find_connected` (grafo), ≤2 hops.
- REQ-4.2 (CONVERGÊNCIA — anti-ruído, SEMPRE): THE candidato por associação SHALL entrar SOMENTE SE alcançado por ≥2 caminhos INDEPENDENTES (de seeds distintos). Esta regra SHALL valer inclusive no caso de 1 seed — se há só 1 seed, NÃO há convergência possível, logo NENHUM candidato por associação entra (nunca bypassar a regra por falta de seeds).
- REQ-4.3 (sweet-spot + centralidade): THE candidato SHALL ter similaridade de aresta dentro de um sweet-spot (nem trivial nem idêntico) E o endpoint NÃO SHALL ser um hub de centralidade acima de um cap (ex. `shares_entity` com 376k arestas — hub trivial é barrado).
- REQ-4.4 (ADITIVO — invariante central): THE expansão SHALL APENAS ADICIONAR candidatos convergentes ao resultado; SHALL NUNCA REMOVER, filtrar ou reordenar-para-baixo itens que o pipeline atual já traria. Ligar a flag NÃO PODE fazer o count de resultados relevantes DIMINUIR.
- REQ-4.5 (budget REAL): THE número de candidatos adicionados SHALL ser limitado por um budget configurável; o budget SHALL efetivamente truncar (teste prova que N+1 candidatos → N no resultado).
- REQ-4.6: THE hot-path SHALL NÃO chamar LLM nem rede (só SQL + set ops).
- REQ-4.7: opt-in flag `MCP_INJECT_GRAPH_NAV`; OFF (default) = byte-idêntico ao atual.
- REQ-4.8: `find_connected` SHALL rodar via asyncio.to_thread (não bloquear event loop sobre 528k arestas) + usar _conn_lock conforme convenção do repo.

## Critérios de aceite (medidos, não-inertes)
- CA1 (aditivo): flag OFF→count=N; flag ON com beliefs SEM aresta→count=N (não cai). Sabotar (tornar subtrativo)→RED.
- CA2 (convergência barra ruído): belief alcançável por 1 caminho só → AUSENTE; por ≥2 caminhos independentes → PRESENTE. Sabotar (aceitar 1 caminho)→RED.
- CA3 (1 seed): com 1 seed, nenhum candidato por associação entra (regra não bypassada). Sabotar→RED.
- CA4 (centralidade): hub shares_entity não é injetado por convergência trivial.
- CA5 (budget): N+1 candidatos convergentes → só N no resultado.
- CA6 (byte-idêntico OFF): comparar OFF vs baseline (não OFF vs OFF); excluir freshness_score volátil da comparação.
- CA7: complexidade de TODA função tocada ≤8 (radon colado); não aumentar a dívida de memory_context (F47) — a integração entra por 1 chamada guardada pela flag + helpers ≤8.
- VEREDITO CLIENT: `memory_context` de um tema traz, no top-k, um item alcançado SÓ via associação (não pelo lexical/embedding direto).

## Invariantes / anti-casos (os que quebraram na 1ª tentativa)
1. ADITIVO, nunca subtrativo (REQ-4.4) — o erro que reverteu.
2. Convergência SEMPRE, nunca bypassada com <2 seeds (REQ-4.2/CA3).
3. Sweet-spot + centralidade são OBRIGATÓRIOS, não "fase 2" (REQ-4.3).
4. Usar o `find_connected` existente (graph.py), não reimplementar 1-hop.
5. Teste do no-LLM deve ASSERIR ausência de chamada (mock), não só "injected is True".

## Fora de escopo
Reweighting das arestas (= INC 5). Gatilho automático (= INC 6).
