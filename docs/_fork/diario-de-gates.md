# Diário de gates — harness de Recall@k e A/B de modelos

> Registro por incremento, com números. Regra: gate só fecha com execução real,
> não com auto-relato.

## G3 — testes RED (spec-recall-harness.md §7)

- 7 testes escritos antes da implementação (`tests/test_recall_harness.py`).
- RED confirmado: cada teste falha na própria linha de import
  (`ModuleNotFoundError: scripts.benchmarks.recall_harness`), não na coleta.
- Ruff limpo.

## G4 — implementação (harness)

- `scripts/benchmarks/recall_harness.py` (23 KB).
- 7/7 verdes; ruff limpo.

## E1 — divergência 0% × 20%: explicada

Duas causas, ambas confirmadas por execução:

1. **O harness usava o banco vivo por default.** O relatório do executor trazia
   `meta.db = /home/claudio/local-data/mcp/sqlite_vec.db`, não o snapshot. Rodando
   com `--db` explícito contra o snapshot, o número aparece.
2. **Caso inválido por vocabulário do gabarito.** `builder-one-mapelites` tinha
   `gold_terms: ["MAP-Elites", "MAP-Elites + UCB1"]`; a segunda string não existe
   literalmente no acervo, e o validador invalidou o caso inteiro (9 válidos, 1
   inválido). Corrigido para `["MAP-Elites"]`.

Lição (reforço de B10): número que não reproduz não é medida, é ruído. Pedir o
`--db` explícito e validar o gabarito contra o acervo são pré-requisitos.

## E2 — baseline oficial (snapshot, 2 execuções)

| modo | recall@10 | mrr@10 |
|---|---|---|
| semantic | 0,200 | 0,125 |
| hybrid | 0,200 | 0,125 |
| ranked | 0,200 | 0,114 |
| exact | 0,000 | 0,000 |

- 10 casos válidos, 0 inválidos.
- Determinismo: JSON idêntico nas duas execuções.
- Data: snapshot `~/local-data/mcp/lab/snapshot-20261018.db` (1,1 GB, integrity ok).

**Baseline de trabalho: Recall@10 = 20% (2 de 10).** É o número contra o qual todo
ganho passa a ser medido.

## Ressalva R6 — escrita em banco read-only

O storage tenta persistir metadados de acesso (`last_accessed`) mesmo com o
SQLite aberto em `mode=ro`, produzindo `Failed to persist access metadata:
attempt to write a readonly database` no log. Não quebra a busca e não altera
dados (a escrita falha), mas:

- viola o espírito de R6 ("nenhuma escrita");
- polui a saída e pode mascarar erro real.

Pendência: suprimir telemetria quando o backend é read-only (flag de sessão ou
`MCP_MEMORY_DISABLE_ACCESS_TELEMETRY=1`). Não bloqueia o harness.

## E5 — gabarito de 120 casos e baseline definitivo (18/out)

O gabarito anterior tinha 10 casos: um acerto valia 10 pontos percentuais. Substituído
por 120 casos verificados (`eval/recall_cases_v2.json`), gerados por 4 subagentes em
faixas disjuntas de memória e validados por código (gold_term presente no conteúdo,
ausente na pergunta, hash único, sem perguntas repetidas). 80 em PT, 40 em EN.

| modo | recall@10 | mrr@10 |
|---|---|---|
| semantic | 0,292 | 0,163 |
| hybrid | 0,292 | 0,163 |
| ranked | 0,275 | 0,169 |
| exact | 0,000 | 0,000 |

**Baseline definitivo: Recall@10 = 29,2% (35/120).** O valor anterior (20%, com 2/10)
era ruído amostral — confirma que a régua curta enganava.

O `exact` em zero é resultado, não falha: as perguntas foram escritas sem conter o
termo-chave, então busca léxica pura não as resolve. Serve como controle negativo.

