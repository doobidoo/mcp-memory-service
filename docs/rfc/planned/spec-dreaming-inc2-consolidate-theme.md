# SPEC — Dreaming INC 2: consolidação por tema (`memory_consolidate(theme=...)`)

**Status:** implemented (13/out) · **RFC:** rfc-dreaming-loop.md · **Fase:** INC 2 · **Spec retroativa** (feature já no fork, commits 522d9cf6..13374daa).

## Objetivo
Permitir consolidar um SUBCONJUNTO de memórias por tema, reusando as 6 fases existentes, sem alterar a derivação global de belief.

## Requisitos (EARS)
- REQ-2.1: WHEN `memory_consolidate` é chamado com `theme` (string não-vazia), THE sistema SHALL restringir as 6 fases ao subconjunto de memórias selecionadas por retrieval semântico do tema.
- REQ-2.2: THE seleção por tema SHALL ocorrer na seleção de candidatos da consolidação; SHALL NOT alterar `_get_memories_for_horizon` (derivação de belief permanece global).
- REQ-2.3: WHEN `theme` ausente, THE comportamento SHALL ser byte-idêntico ao atual (opt-in).
- REQ-2.4: THE valor de `theme` SHALL ser declarado no inputSchema da tool e sanitizado em logs (_sanitize_log_value).

## Critérios de aceite (medidos)
- `consolidate("weekly", theme="python")` processa só memórias do tema; `consolidate("weekly")` processa todas (byte-idêntico). ✅ 7/7 testes, 422 suite, 0 regressão.
- Teste não-inerte: neutralizar o filtro → RED (provado por sabotagem do tuvok). ✅

## Invariantes / anti-casos
- Mock de `search_memories` DEVE honrar o contrato real (retorna dicts via to_dict(), não objetos Memory) — senão o teste mascara bug de acesso.
- Handler `handle_consolidate_memories` complexidade ≤8 (extraídos helpers _run_incremental_with_timeout/_validate_time_horizon/_format_report).

## Fora de escopo
Gatilho automático por tema (= INC 6). Navegação de associações (= INC 4).
