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

## E7 — o índice está partido em três; a sujeira do corpus é irrelevante (18/out)

### E7.1 A régua do harness é a mais frouxa das três

O harness credita acerto quando **qualquer** `gold_term` aparece em **qualquer**
resultado (`posicao_do_acerto`). Medidas as três definições na mesma execução,
120 casos, modelo `all-MiniLM-L6-v2`:

| definição | recall@10 | recall@100 |
|---|---|---|
| termo-any (harness) | 0,500 | 0,758 |
| termo-all (todos os termos no mesmo resultado) | 0,375 | 0,592 |
| documento (`gold_hashes`) | **0,333** | **0,500** |

11 casos recebem crédito por termo sem o documento gold no top-100; o inverso é zero.
**Decisão: documento é a métrica de decisão.** Artefatos: `eval/lab_metric_definitions_*.json`.

### E7.2 O corpus vive em três espaços vetoriais

Amostra aleatória de 200 memórias do store default, comparando o vetor gravado com a
re-encodificação por cada candidato (cosine ~1,0 = mesmo modelo):

| coorte | amostra | janela temporal |
|---|---|---|
| all-MiniLM-L6-v2 | 75 (37,5%) | 10/fev → 13/set |
| paraphrase-multilingual | 104 (52,0%) | 20/jul → 17/out |
| nenhum modelo reproduz (órfãos) | 21 (10,5%) | 09/jul, 08h21–09h23 |

Fronteira da troca principal: rowid 16.179 → 16.180, em 2026-07-09 08:54:18.
Contagem por coorte: 7.268 em all-MiniLM (35,8%) e 13.042 em paraphrase (64,2%),
dos quais ~2.100 são os órfãos. Há ainda três stores com nome binário corrompido
(`\x0b\x12\x0b\t\r\r\n` e similares).

Consequência metodológica: **a detecção automática do harness estava amostrando as 16
memórias mais recentes** (cauda = coorte paraphrase) e elegendo o modelo da cauda para
o acervo inteiro. Corrigido: amostra estratificada por rowid, decisão por maioria de
coorte e aviso `corpus misto` no relatório. 7/7 testes verdes.

### E7.3 O modelo certo é o configurado; o índice é que está errado

Índice paralelo completo (20.310 memórias), força bruta, cosine exato, régua por documento:

| cenário | @10 | @30 | @100 | custo |
|---|---|---|---|---|
| configuração viva (consulta paraphrase × índice misto) | 0,008 | 0,017 | 0,017 | — |
| consulta all-MiniLM pelo caminho do serviço | 0,333 | — | 0,500 | — |
| índice uniforme all-MiniLM | 0,433 | 0,508 | 0,600 | 214 s |
| **índice uniforme paraphrase** | **0,592** | **0,708** | **0,767** | 367 s |

O paraphrase-multilingual (o modelo configurado) vence por 15,9 p.p. em @10 e 16,7 em
@100 no acervo em português. **Não se troca o modelo: re-embeda-se o acervo.**
Artefato: `eval/lab_reembed_models.json`.

Diferença entre caminho do serviço (0,333) e cosine puro (0,433) com o mesmo modelo:
~10 p.p. de pedágio dentro do storage (fusão híbrida / limiares / pós-processamento).
Frente própria, não é o gargalo principal.

### E7.4 Chunking: resultado negativo, hipótese da noite encerrada

Blocos de 126 tokens (stride 96), 36.830 blocos, agregação max-pool por documento:

| cenário | @10 | @30 | @100 |
|---|---|---|---|
| índice simples | **0,592** | **0,708** | 0,767 |
| chunking + max-pool | 0,567 | 0,683 | 0,775 |

O truncamento em 128 tokens não é o gargalo quando consulta e índice são coerentes.
Artefato: `eval/lab_chunking_maxpool.json`.

### E7.5 Higienizar o corpus não muda o recall

Corpus higienizado (sem duplicatas normalizadas, sem traceback/git/binário/fragmento
<40 chars, golds protegidos) e variante com dedup semântico (cosine ≥ 0,97):

| corpus | memórias | @10 | @30 | @100 |
|---|---|---|---|---|
| completo | 20.310 | 0,592 | 0,708 | 0,767 |
| higienizado | 17.656 (−186 lixo, −2.468 duplicata) | 0,592 | 0,708 | 0,767 |
| higienizado + dedup semântico | 15.198 | 0,600 | 0,708 | 0,767 |

Higienização compra 1 caso em 120. A sujeira existe (12,1% de duplicatas, 42% de JSON
bruto de ferramenta), mas **não governa o recall**. Artefato: `eval/lab_clean_corpus.json`.

### E7.6 O que ficou visível e não é ranking

- **26,9% do acervo é invisível por padrão**: 20.310 no store `default` contra 27.776
  vivos no total (`mir` 4.509, `docs-shared` 2.154, `hackathon` 735, `docs` 65). A busca
  sem `store` explícito nunca vê esses 7.466 registros.
- Latência não é problema: p95 137 ms, p50 97 ms.
- Ruído no top-10 não é problema: 0,08 slots redundantes por consulta.

### E7.7 Estratégia: dois planos

- **Plano A — bancada de referência**: corpus higienizado, um modelo, `embedding_version`
  por linha. Mede o sistema e decide intervenção (0,592 @10 é o número atual).
- **Plano B — banco vivo**: migração com checkpoints (dedup, stores corrompidos,
  re-embed uniforme), não pesquisa.

Regra: dado do plano A não julga o banco vivo; mudança no banco vivo não se valida
só nele.

### E7.8 Ordem de execução

1. Migração do acervo: re-embed uniforme em paraphrase + `embedding_version` por linha,
   dedup, stores corrompidos. Devolve os +32,5 p.p. já medidos.
2. Re-medir no banco migrado e comparar com a bancada (0,592 @10). Divergência = defeito
   de migração.
3. Reranker: teto 0,767 @100, atual 0,592 @10 → 17,5 p.p. disponíveis.
4. Sinal de ausência (nunca medido).
5. Visibilidade de store (E7.6).



