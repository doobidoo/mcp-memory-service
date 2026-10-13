# SPEC — Dreaming INC 6: gatilho por importância por tema (loop rápido)

**Status:** planned (SPEC v1 — revisar após INC 2 em uso) · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 6 · **Dep:** INC 2 (consolidate theme, DONE — desbloqueado).

## Problema (dor direta do CLIENT)
Derive/consolidação só rodam no cron (03:00/domingo). O que aprendo HOJE não vira belief injetável HOJE — "loop lento". É a dor mais sentida no uso real.

## Objetivo
Disparar consolidação temática (INC 2) quando a importância acumulada de um tema cruzar um limiar — COMPLEMENTO ao cron, para o aprendizado recente ficar disponível no mesmo dia.

## Requisitos (EARS) — v1
- REQ-6.1: WHEN a importância acumulada de um TEMA cruza um limiar configurável, THE sistema SHALL disparar `memory_consolidate(theme=...)` (INC 2) para aquele tema, COMO COMPLEMENTO ao cron (não o substitui).
- REQ-6.2: THE importância SHALL derivar de sinais JÁ EXISTENTES (usage_events / Frente D: frequência, novidade por tema), SHALL NÃO usar LLM.
- REQ-6.3: THE novidade SHALL ser contada POR TEMA (não global) — um tema quente dispara independente dos outros.
- REQ-6.4: THE gatilho SHALL ter guardas contra disparo excessivo (debounce/cooldown por tema) e contra "confident hallucinations" do padrão Generative-Agents (não gerar belief especulativo, só consolidar o que existe).
- REQ-6.5: opt-in flag; OFF = só cron (comportamento atual).

## Critérios de aceite — v1
- Acumular importância num tema (simular N usos/novidades) → dispara consolidação daquele tema fora do cron. Medir: belief do tema recente fica injetável no mesmo dia (ataca a dor).
- Cooldown impede disparo repetido em janela curta.
- OFF = nenhum disparo extra.

## Invariantes / anti-casos
- SEM LLM no gatilho (REQ-6.2).
- Importância por sinais existentes, não inventar métrica nova sem medir.
- Não floodar: debounce obrigatório.

## ⚠️ Revisar após INC 2 em uso real
Medir primeiro os sinais de importância disponíveis (usage_events por tema) antes de fixar o limiar.

## Fora de escopo
Reweighting (INC 5). Reflection LLM (INC 7).
