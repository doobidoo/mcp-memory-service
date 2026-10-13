# SPEC — Dreaming INC 7 (opcional): memory_explore unificado + reflection LLM-gated

**Status:** planned (SPEC v1 — OPCIONAL, só se INC 2-6 provarem valor) · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 7 · **Dep:** INC 3.

## Objetivo
Camada de exploração sobre o grafo heterogêneo (navegar memory/belief/cluster/associação) + reflection (Generative Agents) opt-in e offline para gerar insights de 2ª ordem.

## Requisitos (EARS) — v1
- REQ-7.1: THE `memory_explore` SHALL percorrer o grafo heterogêneo (nós tipados + arestas tipadas) e retornar uma visão navegável por tema.
- REQ-7.2: THE reflection SHALL ser opt-in, amostrada OFFLINE (fora do hot-path de injeção), gated por flag + budget de LLM.
- REQ-7.3: THE reflection SHALL ter guardas contra "confident hallucinations" (marcar origem, não promover a belief sem corroboração).

## Critérios de aceite — v1
- `memory_explore(tema)` retorna subgrafo navegável coerente.
- reflection roda só offline, nunca no hot-path; flag OFF = sem reflection.

## Invariantes / anti-casos
- NUNCA LLM no hot-path (reflection é offline).
- Opcional: só priorizar se o CLIENT, após INC 4+6, ainda sentir falta de insight de 2ª ordem.

## ⚠️ Revisar após INC 3-6
Depende do grafo heterogêneo final (INC 3) e de o CLIENT confirmar necessidade. Pode ser descartado se INC 4+6 já satisfizerem.

## Fora de escopo
Tudo que INC 2-6 cobrem.
