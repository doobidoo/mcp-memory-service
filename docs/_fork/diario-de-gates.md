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

## E6 — baseline v2 reproduzido e o teto do reranking (18/out)

Repetição independente do E5 sobre o mesmo snapshot, com os relatórios gravados no
repo do lab (`eval/lab_recall_v2_k10_20261018.json`, `eval/lab_recall_v2_k100_20261018.json`,
commit `695db04e`, repo local sem remote):

| modo | recall@10 | mrr@10 |
|---|---|---|
| semantic | **0,2917** | 0,1626 |
| hybrid | **0,2917** | 0,1626 |
| ranked | 0,2750 | 0,1693 |
| exact | 0,0000 | 0,0000 |

O 29,2% do E5 se confirma. O E5 não deixou o JSON da execução versionado; esta
repetição fecha essa lacuna.

Medição nova, `--k 100` (mesmo dataset, mesmo snapshot):

| modo | recall@100 | mrr@100 |
|---|---|---|
| semantic | 0,4750 | 0,1703 |
| hybrid | 0,4750 | 0,1702 |
| ranked | 0,4833 | 0,1775 |
| exact | 0,0000 | 0,0000 |

Distribuição dos 120 casos em `semantic`: 35 no top-10, 22 entre 11 e 100,
**63 fora do top-100**.

Consequência para o plano, e ela corrige a leitura do §8 do relatório da noite:

1. **Teto de qualquer reranker sobre o índice atual é 47,5%.** Nenhum ganho de
   ordenação pode passar disso — 63 casos não são candidatos.
2. **Reranking só pode resolver 22 casos** (11–100). Ganho máximo teórico: +18,3 p.p.
3. **A maior parte da falha é representação, não ordenação** — 52,5% dos casos
   nunca entram no top-100. Isso reabilita o truncamento (A2) e o modelo do índice
   (A1) como gargalo principal, e rebaixa "a qualidade do ranking" a segundo
   problema. A conclusão do §8 de que o chunking "não é a intervenção principal"
   foi medida antes de existir régua com poder; agora ela tem um critério de
   decisão: só vale intervenção que mova o recall@100.
4. Idioma não é fator: recall@100 é 48% em PT e 48% em EN.
5. Ordem revisada: (a) medir recall@100 como métrica-alvo do próximo experimento;
   (b) atacar representação — chunking/truncamento com o índice completo, agora
   medido por recall@100 e não por recall@10; (c) reranker só depois, para colher
   os 22 casos resgatáveis.


