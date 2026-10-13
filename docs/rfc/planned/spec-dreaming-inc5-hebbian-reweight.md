# SPEC — Dreaming INC 5: loop sono→acordado (reweighting Hebbiano)

**Status:** planned (SPEC v1 — revisar após INC 4, pois depende de como a navegação consome os pesos) · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 5 · **Dep:** INC 3 + INC 4.

## Objetivo
Fazer o USO (acordado) recalibrar os PESOS das arestas (sono): aresta cuja associação ajudou sobe; aresta apresentada-e-ignorada decai. Fecha o laço dreaming↔uso.

## Requisitos (EARS) — v1
- REQ-5.1: WHEN uma aresta participa de uma injeção que foi USADA (sinal positivo de usage_events/Frente D), THE peso da aresta SHALL aumentar via EMA: `w += η·util` (η pequeno).
- REQ-5.2: WHEN uma aresta é apresentada e IGNORADA (sinal negativo), THE peso SHALL decair; com half-life temporal para esquecer ruído antigo.
- REQ-5.3: THE atualização SHALL ser EMA (média móvel), NUNCA salto único — para não cristalizar ruído de um evento.
- REQ-5.4: THE peso SHALL alimentar o ranking da navegação do INC 4 (aresta de peso alto preferida na expansão).
- REQ-5.5: opt-in flag; OFF = pesos estáticos (comportamento INC 4 puro).

## Critérios de aceite — v1
- Aresta usada N vezes tem peso > aresta nunca usada. Aresta ignorada decai ao longo de ciclos.
- Sem salto único (um evento não dobra o peso). Half-life aplicado (peso antigo esquece).
- 0 regressão no INC 4 com flag OFF.

## Invariantes / anti-casos
- EMA + half-life OBRIGATÓRIOS (risco 5 do CONSOLIDADO: cristalizar ruído).
- Onde persistir o peso: provável coluna/campo em memory_graph.metadata — DECIDIR no G0 do INC 5 (pode exigir o único ALTER do arco; aditivo+idempotente+cópia).

## ⚠️ Revisar após INC 4
Como o INC 4 consome o peso (REQ-4.3 sweet-spot) define o shape deste INC. Reavaliar REQ-5.4 quando o INC 4 fechar.

## Fora de escopo
Gatilho (INC 6). Reflection (INC 7).
