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
a hipótese inicial foi "~10 p.p. de pedágio dentro do storage". **Refutada no E8**: com
índice uniforme, o caminho do serviço entrega exatamente o mesmo que a força bruta
(0,592). A diferença anterior era artefato do índice misto (KNN particionado sobre
coortes incompatíveis), não um custo fixo do storage.

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

## E8 — migração executada em cópia e validada (18/out)

Cópia do snapshot (`~/local-data/mcp/lab/migracao/migrado2.db`), migrador oficial do
produto (`scripts/maintenance/migrate_embeddings.py`) apontado para uma ponte local
OpenAI-compatible sobre o ONNX do serviço (`scripts/lab/onnx_embeddings_api.py`).

### E8.1 O migrador oficial quebrava a busca (bug corrigido)

Primeira execução: 27.777 memórias re-embedadas, integridade e KNN "ok" no relatório
do script — e **semantic/ranked/exact retornando zero resultado**, com
`sqlite3.OperationalError: no such column: store`.

Causa: o script recria `memory_embeddings` com DDL fixa
(`vec0(content_embedding FLOAT[384] distance_metric=cosine)`) e descarta a coluna de
partição que a versão atual do serviço usa:

```sql
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine,
    store TEXT partition key
)
```

Toda busca que filtra por store passa a falhar. Corrigido em `13c0b967`: a DDL
original é lida antes do drop, recriada com apenas a dimensão substituída, e o insert
inclui a coluna de partição. Correção no repo principal, ainda não enviada ao upstream.

### E8.2 Resultado da migração validada (caminho do serviço, régua por documento)

| métrica | antes (config viva) | índice uniforme | migrado pelo serviço |
|---|---|---|---|
| documento @10 | 0,008 | 0,592 | **0,592** |
| documento @30 | 0,017 | 0,708 | **0,708** |
| documento @100 | 0,017 | 0,767 | **0,775** |
| termo-any @10 (harness) | 0,292 | 0,500 | 0,825 |
| termo-any @100 | 0,475 | 0,758 | 0,942 |

- 120 casos válidos, 0 inválidos, sem warning de corpus misto.
- Tamanho: 1.110 MB → 825 MB.
- 27.777 vetores uniformes; 573.559 arestas de grafo preservadas (`--keep-graph`).
- Critério de aceitação do plano ("divergência entre bancada e serviço = defeito de
  migração"): **não houve divergência** — 0,592 nas duas medições.
- Artefatos: `eval/lab_migrado2_k10.json`, `eval/lab_migrado2_k100.json`,
  `eval/lab_metric_definitions_migrado.json`.

### E8.3 O que a migração não resolve

- 40,8% dos casos ainda não têm o documento no top-10; 22,5% continuam fora do top-100.
  Teto de reranking passa a ser 0,775.
- Duplicatas (2.468) e os três stores com nome corrompido continuam intactos: o
  migrador re-embeda, não higieniza. Higienização vale 1 caso em 120 (E7.5).
- 26,9% do acervo segue invisível por padrão (E7.6).
- Sinal de ausência (F5) segue sem medição.

## E9 — F5 refutada e o modo `ranked` é pior do que o `semantic` (18/out)

Medições sobre o índice migrado (`migrado2.db`), régua por documento.

### E9.1 O score de similaridade não denuncia ausência

71 acertos (documento no top-10) contra 49 erros, score do top-1:

| grupo | n | média | cauda |
|---|---|---|---|
| acertos | 71 | 0,841 | p10 = 0,791 |
| erros | 49 | 0,822 | p90 = 0,867 |

As distribuições se sobrepõem quase por inteiro. Varredura de limiares:

| limiar | erros detectados | acertos perdidos | precisão da abstenção |
|---|---|---|---|
| 0,75 | 1/49 | 0/71 | 1,00 (cobre 2% dos erros) |
| 0,80 | 17/49 | 13/71 | 0,57 |
| 0,85 | 34/49 | 42/71 | **0,45** |
| 0,90 | 49/49 | 65/71 | 0,43 |

Em 0,85 detecta 69% dos erros e descarta 59% dos acertos: pior que decidir por sorteio.
**Abstenção por limiar de cosseno está refutada** — a ideia que estava no plano (F5)
não se sustenta. Se o sinal for perseguido, tem de vir de outro mecanismo (score de
cross-encoder, entailment, cobertura lexical), não da similaridade.
Artefato: `eval/lab_abstain.json`.

### E9.2 `ranked` perde 26,7 p.p. — e a régua frouxa escondia isso

Mesmo índice, mesmos 120 casos:

| modo | documento @10 | documento @100 | termo-any @10 |
|---|---|---|---|
| semantic | **0,592** | **0,775** | 0,825 |
| ranked | 0,325 | 0,475 | 0,833 |

Na régua antiga (termo-any) o `ranked` parecia **melhor** (+0,8 p.p.). Na régua por
documento ele perde 26,7 p.p.: o `relevance_score` (similaridade + decaimento +
boost de acesso) promove memórias muito acessadas — uma memória com `access_count`
406 apareceu no topo de uma consulta sem relação. É a demonstração mais clara de
por que a métrica de decisão passou a ser a de documento.

O serviço usa `semantic` por padrão (`arguments.get("mode", "semantic")`), então o
dano está limitado a quem pede `ranked` explicitamente. Ainda assim: não usar `ranked`
para busca de conhecimento, e não vendê-lo como melhoria.
Artefato: `eval/lab_metric_definitions_migrado_ranked.json`.

### E9.3 Reranker: bloqueado por dependência, decisão adiada

O cross-encoder do produto (`quality/onnx_ranker.py`, `ms-marco-MiniLM-L-6-v2`) não
tem grafo ONNX exportado nesta máquina (`ImportError: ONNX model not found ... Either
download pre-exported model or install transformers+torch`), e não há `torch` no venv.
Custo para destravar: instalar `transformers`+`torch` (~2 GB) e exportar, ou baixar um
reranker multilíngue (~2,3 GB) — o `ms-marco` é treinado em inglês e o acervo é PT.

Decisão de gestão: **não gastar 2 GB agora**. O teto do reranking é 0,775 e o ganho
realista num corpus PT com modelo inglês é incerto. Antes disso, há itens de valor
maior e custo zero (E10): visibilidade de store e legibilidade do que volta.

### E9.4 Estado do lote após E9

| frente | estado |
|---|---|
| F1 régua por documento | feito |
| F2 índice uniforme + migração validada | feito (aguarda autorização para produção) |
| F3 chunking | refutado |
| F4 reranking | bloqueado por dependência, adiado |
| F5 sinal de ausência | refutado por limiar de similaridade |
| F6 ruído/latência | medido, não é problema |

## E10 — régua cross-store: a busca padrão acha 0% dos outros stores (18/out)

A régua de 120 casos só tem ouro no store `default`, então não podia julgar a busca
restrita. Construída uma segunda régua com 60 casos cujo documento está em `mir`,
`docs-shared` e `hackathon` (20 cada), gerada por 3 subagentes em faixas disjuntas e
**revalidada por código** (hash existe no store certo, `gold_terms` literais no
conteúdo, nenhum termo na pergunta, hashes e perguntas únicos): 60/60 válidos.
Arquivo: `eval/recall_cases_xstore.json`.

Mesma pergunta, dois escopos de busca (índice migrado, régua por documento):

| escopo | documento @10 | @30 | @100 |
|---|---|---|---|
| só `default` | **0,000** | 0,000 | 0,000 |
| todos os stores | **0,350** | 0,433 | 0,600 |

Zero. Não é degradação: é ausência total. As 60 respostas existem no acervo e nenhuma
delas é alcançável numa busca normal, porque o handler usa
`arguments.get("store", "default")` e o acervo fora do `default` são 7.466 memórias
(26,9% do total).

Custo no lado oposto, medido na régua de 120 casos: abrir os stores derruba o @10 de
0,592 para 0,567 (−2,5 p.p.), e 21,2% dos slots do top-10 passam a vir de fora.
**O trade é claro: −2,5 p.p. em precisão compram 35% de cobertura sobre um quarto do
acervo.** Para um sistema de memória, esquecer em silêncio é o pior modo de falha.

Decisão proposta (Plano B): busca padrão passa a varrer todos os stores, com o store
de origem devolvido em cada resultado para o cliente filtrar; `store` explícito
continua disponível para isolamento por projeto.
Artefatos: `eval/lab_store_visibility.json`, `eval/lab_store_visibility_xstore.json`.







## E11 — migração aplicada em PRODUÇÃO (19/out)

Execução autorizada pelo Claudio. Serviço parado às 13:51, religado às 14:09.
Backup verificado antes: `~/local-data/mcp/lab/pre-migracao-posstop.db`
(`sqlite3 .backup`, 1,2 GB, `integrity_check ok`, 29.423 memórias).

### E11.1 Incidente: rodei o migrador errado (detectado antes de subir o serviço)

Primeira execução usou `scripts/maintenance/migrate_embeddings.py` da **branch de bancada
`lab/precisao-recall`**, que não tem a correção `13c0b967` (ela vive na `main`). Resultado:
`memory_embeddings` recriada **sem** `store TEXT partition key`, 28.738 vetores inseridos.

Detecção: o harness rodou com o serviço ainda parado e devolveu `semantic = 0,000` e
`ranked = 0,000` com `hybrid = 0,733`. O zero no ramo semântico com híbrido funcionando é a
assinatura de partição perdida (o filtro `store` estoura em `no such column: store`; o
híbrido chega ao resultado pela via FTS).

Contenção: banco defeituoso preservado como `sqlite_vec.db.migrado-sem-particao-20261019`;
restauração a partir do backup `.backup` (não do `copy2` do migrador); nova migração com o
script da **main**.

**Lição (vira regra):** antes de rodar script de manutenção, verificar que a correção
conhecida está no arquivo que se vai executar — `grep -c "partition key" <script>` tem de
ser ≥1 neste caso. Branch de bancada não herda correção da `main` sozinha; nesta data a
branch recebeu o cherry-pick `90cf1e8f` e foi pushada.

Segunda lição, de método: **medir antes de religar o serviço** foi o que conteve o dano.
Se o serviço tivesse subido antes da medição, o defeito apareceria como "busca quebrada"
sem causa identificada.

### E11.2 Resultado (banco vivo, régua por documento)

| métrica | antes (config viva, índice misto) | depois (uniforme paraphrase) |
|---|---|---|
| documento @10 | 0,008 | **0,567** |
| documento @30 | 0,017 | **0,675** |
| documento @100 | 0,017 | **0,750** |
| termo-any @10 (harness) | 0,292 | **0,775** |

- 120 casos válidos, 0 inválidos, sem warning de corpus misto.
- 28.738 vetores uniformes · 623.322 arestas preservadas · 476,8 s de geração.
- Banco: 1.195,1 MB → **910,5 MB** · `PRAGMA integrity_check` ok · KNN ok.
- Serviço: `/health` ok; busca real por MCP devolveu como top-1 a decisão da Fase 5 do
  delta-sync, **criada em 10/out** (coorte que antes era inalcançável).
- Latência medida (5 resultados): 0,19 / 0,28 / 0,21 s · payload 7,7–11,4 KB.

**Comparação com o E8 (0,592 em cópia):** a régua agora dá 0,567 (−2,5 p.p.). Não é
regressão da migração: o acervo cresceu ~961 memórias entre 18 e 19/out, o que aumenta o
número de distratores para o mesmo gabarito. O número correto a citar é este, medido no
acervo vivo; o 0,592 foi medido em cópia do snapshot de 18/out.

## E12 — importador do `state.db`: bug de schema e orçamento medido (19/out)

### E12.1 O importador consultava uma coluna que nunca existiu

`scripts/harvest/hermes_source.py` consultava `profile_name` em `sessions`. O schema real
do `state.db` do Hermes nao tem essa coluna (tem `source`, `user_id`, `display_name`,
`title`, …). Resultado, no caminho real:

```
sqlite3.OperationalError: no such column: s.profile_name
```

Os 22 testes passavam porque a **fixture inventava a coluna**. E' a reincidencia exata do
pitfall "fixture tem que reproduzir o formato REAL da fonte" — o teste protegia o
codigo e escondia o produto.

Correcao (`c9721ab4`, branch do lab): fixture reconstruida a partir de
`sqlite3 ~/.hermes/state.db '.schema sessions'`; identidade do agente passa a vir de
`source` (fallback `hermes`); teste novo reproduz o erro de coluna e falha antes do fix.
Depois: **29 testes verdes** e dry-run real funcionando.

### E12.2 O acervo de sessoes do Hermes neste host e pequeno

`select count(*) from sessions` = **7** (3 `cli`, 2 `tui` na primeira medicao; cresce a cada
sessao nova). O dry-run de 18/out que reportava 107 sessoes **nao e reproduzivel neste
estado** — o numero precisa ser re-medido sempre, nunca citado de relatorio antigo.

### E12.3 Orcamento: o criterio de 10% nao se aplica a este acervo, e o desperdicio e real

`MAX_SESSION_CHARS = 15_000` por sessao (head 70% / tail 30%). Medido no acervo atual
(7 sessoes, 6 elegiveis), com o script em copia temporaria (constante e default de
argumento — monkeypatch depois do import NAO tem efeito):

| teto por sessao | chars enviados | sessoes truncadas | turnos perdidos |
|---|---|---|---|
| 15k (atual) | 71.396 | 4 de 6 | **711** |
| 30k | 124.736 | 3 de 6 | 375 |
| 60k | 186.059 | 1 de 6 | 77 |

Leitura correta: o criterio CA3 (≤10% do lido) foi calibrado para um acervo de 107
sessoes; com 6 elegiveis ele mede proporcao de conjunto pequeno, nao desperdicio. O numero
que importa e **turno perdido** — 711 turnos e perda de conhecimento, nao economia.

**`--apply` NAO foi executado.** A escrita e irreversivel (o importador marca a sessao como
importada) e a escolha do teto e decisao de produto — pendente com o Claudio.

## E13 — autoria por `X-Agent-ID` no streamable-http: G5 e G6 (19/out)

Classe: **BUG** (corrupção silenciosa de autoria). Spec: `rfc/planned/spec-agent-id-streamable.md`.

### E13.1 Gate

| gate | resultado |
|---|---|
| G3 RED | `b666de14` — CA1/CA2 falham nomeando o defeito (`agent_id` = `zero`); CA3/CA4 verdes como guardas |
| G4 GREEN | `66c39700` — ContextVar + middleware ASGI + precedência em `store_memory` |
| G5 1a | **NEEDS_CHANGES**: B1 (falta log de origem, R5); P2-1 (comentário mentiroso em evolve) |
| G5 2a | **APPROVED** (B1 sanado; B2 sanado; nenhum defeito novo) |
| G6 vivo | merge `ef7df187` na main, restart, 3 gravações reais |

**Achado da manager no G5:** o revisor classificou como P2 o comentário de `evolve_memory`, mas o
defeito real era maior — a precedência **não existia** em `evolve_memory` (L642) nem no
`auto_commit` do harvest (`server_impl.py:1870`). R2 exige todos os caminhos de escrita. Virou
bloqueador B2 e a correção consolidou os três num helper único `resolve_agent_id()`.

Lição para o perfil do revisor: **quebra de requisito explícito da spec é bloqueador**, nunca P2 —
mesmo quando o sintoma parece cosmético (um comentário desatualizado era o sintoma de código que
não fazia o que o comentário dizia).

### E13.2 Prova no serviço vivo (três gravações, conferidas no banco)

| caso | header | argumento | `metadata.agent_id` |
|---|---|---|---|
| R1 | `X-Agent-ID: g6-header` | — | **`g6-header`** |
| R4 | — | — | **`zero`** (fallback `MCP_AGENT_ID`) |
| R2 | `X-Agent-ID: g6-header-deve-perder` | `g6-argument` | **`g6-argument`** |

Sondas removidas do acervo ao final (soft-delete confirmado por `deleted_at`).

### E13.3 Achado lateral (item separado, fora deste PR)

O `memory_store` devolve **erro de negócio com `isError: false`**:

```
"text": "Error storing memory: Duplicate content detected (semantically similar to a5d7…)"
```

Cliente que confia em `isError` reporta gravação bem-sucedida que não aconteceu. É contrato de
API, não autoria — precisa de issue própria. Descoberto porque a primeira sonda de `CA3` foi
absorvida por dedup e o endpoint disse "ok".

### E13.4 Publicacao upstream (19/out)

- Issue **#1597** (bug, com evidencia de dois registros e enquadramento correto: o header e
  documentado no upstream desde #1100 Fase 2 / PR #1297, mas **inerte** no transporte
  streamable — nao e "feature inexistente", e "feature sem ponte no caminho que o servico usa").
- Claim comentado na issue antes do PR (o `Claim Check` passou).
- PR **#1598** — recorte limpo de `upstream/main` em worktree proprio (`~/git/mms/pr-agent-id`):
  6 arquivos, nada fork-only, `changelog.d/1597.fixed.md` presente.
- CI parcial no momento do registro: `Changed tests fail without the change` **pass** (56 s),
  CodeQL pass, Analyze Python pass, changelog pass, claim pass. Pendentes: suites completas e
  Greptile.
