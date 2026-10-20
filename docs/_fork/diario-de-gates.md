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

### E13.5 Cerco com codebase-memory (Greptile P1-A e generalizacao)

O indice estava defasado (16/out), entao foi reindexado (`home-claudio-git-mms-mms`, 15.423 nos,
68.321 arestas). `trace_path(inbound)` em `MemoryService.store_memory` devolveu **11 call sites**:
`handle_commit_session_legacy`, `handle_memory_harvest`, `handle_memory_distill`,
`handle_memory_distill`/_scheduled_distill_check, `MemoryServer.store_memory`,
`_post_commit_learning`, `_increment_session_counter`, `mistake_note_add`,
`mistake_note_update`, `web.api.memories.store_memory` e `store_session`.

Cruzando com `grep '"agent_id"'`, **quatro** chamadas de `store_memory` declaram o autor por
`metadata` — e portanto ficam subordinadas ao header na nova precedencia:

| linha | funcao | observacao |
|---|---|---|
| 2258 | `handle_commit_session_legacy` (observacao de sessao) | apontado pelo Greptile |
| 2285 | `handle_commit_session_legacy` (user_corrections) | nao apontado |
| 2301 | `handle_commit_session_legacy` (erros/correcoes) | nao apontado |
| 2450 | `_increment_session_counter` | nao apontado; contador interno de sessao |

A linha 1991 e `commit_args` (dicionario para o proprio `commit_session_legacy`), nao chamada
direta de `store_memory` — fica correta quando o handler passar o argumento.

**Licao de metodo:** o achado do revisor automatico nomeia UM sintoma; o grafo mostra a classe.
Corrigir so o ponto apontado deixaria tres caminhos com o mesmo defeito — e o teste novo cobre
apenas o caminho que o revisor viu.

### E13.6 Delegacao: briefing grande estoura o contexto do subagente

O primeiro subagente que atacou os 3 achados do Greptile terminou com
`Context length exceeded: max compression attempts (3) reached` — 13 chamadas de API, nenhum
commit, nenhum relatorio. Causa: um unico briefing com os TRES achados + o teste de integracao
de ~400 linhas como referencia + contratos de prova.

**Regra adotada:** fatiar por ARQUIVO, nao por achado. Achados no mesmo arquivo vao juntos
(compartilham contexto); achados em arquivos diferentes vao em worktrees diferentes, para nao
colidir na escrita. Briefing minimo: caminho, sintoma em 2 linhas, correcao exigida, testes,
provas. Nada de colar o teste existente inteiro — basta mandar nao altera-lo.

Resultado desta rodada: `pr-fix-a` (server_impl: 4 chamadas de `store_memory`) e `pr-fix-b`
(agent_context: normalizacao de tipo + sanitizacao de log), dois worktrees, dois subagentes.

## E14 — F2 store visibility, fatia 1 (19/out)

Worktree `~/git/mms/fix-store-visibility`, branch `fix/store-visibility`, commit `1e15dbe9`.

- `store` omitido em `memory_search` consulta todos os stores; `store` explicito continua
  isolando; `Memory` ganhou campo de primeira classe `store` (origem do hit, lido de
  `memories.store`); a resposta declara `stores_searched` (R5).
- Testes: `tests/test_store_visibility.py` com 5 casos (4 do escopo + 1 guarda de contrato).
  `tests/ -k store` = **418 passed, 22 skipped, 1 xfailed**. Load-bearing provado: sem o codigo,
  os 4 casos de escopo falham.

### E14.1 Risco achado na revisao da manager (nao apanhado pelo subagente)

O subagente mudou o default de `RetrieveMixin.retrieve()` de `store='default'` para `None`.
Isso fazia DUAS chamadas internas de dedup, que omitem o argumento, passarem a comparar entre
stores: `server_impl.exact_match_retrieve` (L2979) e `belief_service._group_observations`
(L1367). Efeito: gravar em `default` poderia "colidir" com conteudo identico em `mir` — mudanca
silenciosa no caminho de ESCRITA (territorio do #1517).

Correcao: default do storage volta a `'default'`; escopo amplo passa a ser **opt-in explicito**
(`store=None`), que e o que o service faz quando o usuario omitiu. Guarda de regressao adicionada
(`test_storage_retrieve_omitted_store_keeps_default_scope`); com o default em `None` ela falha.

**Licao:** mudar default de funcao compartilhada e mudanca de contrato; o blast radius tem de ser
enumerado (call sites que OMITEM o argumento) antes de aceitar o diff — CI verde nao cobre isso.

## E15 — Reranker ONNX medido com a regua de decisao (19/out)

`bge-reranker-base` int8 (Xenova, 278.825.308 bytes), 120 casos x 100 candidatos, CPU 16 threads:

| regua por documento | base | com reranker | delta |
|---|---|---|---|
| @5 | 0,4917 | 0,7083 | +21,7 p.p. |
| @10 | 0,5667 | **0,7333** | **+16,66 p.p.** |
| @30 | 0,675 | 0,7417 | +6,7 p.p. |

Latencia: **p50 = 5.314 ms** por consulta (53,1 ms por par). Com 40 candidatos, ~2,1 s estimado
(declarado como estimativa, nao medido).

Leitura: o ganho e ~2x a estimativa anterior (+7,5 p.p.) e ataca justamente os 22 casos entre 11
e 100. Mas 5,3 s contra os 0,19-0,28 s atuais e troca inaceitavel no caminho padrao.

Recomendacao: reranker apenas onde latencia nao importa — injecao proativa em background e busca
profunda sob demanda. NAO ligar em `memory_search` comum sem decisao explicita.

### E15.1 Instrumento: hf_hub_download travou; curl funcionou

`hf_hub_download` ficou 10 min com **0% de CPU** e 44 KB baixados; o mesmo arquivo baixou por
`curl` a ~4,7 MB/s. Causa do primeiro resultado invalido: `curl -m 45` **truncou** o download em
202 MB (o arquivo tem 266 MB) e o ONNX falhou com `INVALID_PROTOBUF`.
Regra: download de modelo se valida por `content-length` vs tamanho no disco antes de culpar o
runtime. Wrapper local: `eval/run_f5_local.py` (monkeypatch de `hf_hub_download`, sem tocar a
logica de medicao).

### E15.2 Curva de candidatos (medida, nao estimada)

Mesmo dump/regua, variando quantos candidatos o reranker reprocessa:

| candidatos | @5 | @10 | @30 | ganho @10 | p50 |
|---|---|---|---|---|---|
| base (sem reranker) | 0,4917 | 0,5667 | 0,675 | — | 0,19-0,28 s |
| **40** | 0,6417 | **0,6667** | 0,675 | **+10,0 p.p.** | **2.098 ms** |
| 100 | 0,7083 | **0,7333** | 0,7417 | **+16,66 p.p.** | 5.314 ms |

Leitura: 40 candidatos entregam 60% do ganho com 40% da latencia. O @30 nao muda com 40
candidatos (o ouro fora dos 40 continua fora) — o reranker reordena, nao inventa candidato.
Duas configuracoes defensaveis: 40 cands/2,1 s para busca profunda sob demanda; 100 cands/5,3 s
para batch e avaliacao offline. Nenhuma delas justifica ligar o reranker no `memory_search`
padrao, que responde em 0,2 s.

## E16 — F2 fatia 2: o gap nao estava onde eu supus (19/out)

A fatia 1 (commit `c08c39ef`, `MemoryService.retrieve_memories`) foi implementada no **caminho
errado**: `handle_memory_search` do MCP nao passa pelo `MemoryService` — chama
`_retrieve_search_results(storage, arguments, normalize_tags)` direto. Naquele helper,
`store = arguments.get("store", "default")` mantinha a busca presa no default mesmo com o
argumento omitido, e os fallbacks (`_search_exact_fallback`, `_search_tag_fallback`) nem sequer
recebiam o escopo.

Consequencia: a fatia 1 melhora a camada de servico (usada pela Web API), mas nao alterava o
comportamento do MCP. Descoberto ao medir o caminho real, nao ao ler o diff.

Correcao (commit `4bbb5eb3`):
- `_resolve_store_scope(arguments)`: omitido/vazio/`"all"` -> None (todos); qualquer outro ->
  particao exata.
- fallbacks exact e tag propagam o escopo resolvido.
- 11 testes novos (`tests/test_store_handler_scope.py`): 4 modos x escopo, fallback e regra do
  helper. `tests/ -k store` = **429 passed, 22 skipped, 1 xfailed**.
- Load-bearing provado revertendo SO o comportamento (`store = arguments.get("store","default")`)
  com o helper mantido: 5 falhas nomeadas, sem erro de colecao. Reverter o arquivo inteiro dava
  `ImportError` — RED invalido por preparacao, nao por defeito.

**Licao (a terceira desta sessao):** antes de implementar, identificar QUAL funcao o transporte
real chama. `grep` do nome da tool no `server_impl` -> `handlers` -> funcao concreta. Implementar
na camada "certa" por arquitetura (servico) sem confirmar o call graph gasta uma fatia inteira.

## E17 — F2 aplicado no vivo (19/out)

Merge de `fix/store-visibility` na `main` (`394b4a49`), restart do servico, R2/R5 acrescentados ao
caminho MCP (`018a7265`).

**Distribuicao real deste banco** (corrige o pressuposto que veio de outro host): `default` 28.964,
`mms` 7, `inbox` 7. Nao ha `mir`/`docs-shared`/`hackathon` aqui — os 7.466 (26,9%) eram de outra
maquina. A prova foi feita com `mms`.

**E2E no servico vivo** (hash alvo `76e55dfb…`, store `mms`):

| request | resultado |
|---|---|
| `store` omitido | achou=True · `scope: default, inbox, mms` |
| `store="default"` | achou=False (isolamento preservado) |
| `store="mms"` | achou=True · `scope: mms` |

Item formatado no vivo: `Created: 2026-10-18T13:48:30Z [source:sirdata] [store: mms]`.

Testes: `tests/ -k store` = **429 passed, 22 skipped, 1 xfailed**.

**Armadilha registrada:** o primeiro teste de E2E deu falso positivo porque procurava a string
"Inbox mms" no texto — palavras comuns no corpus. A prova valida e por HASH da memoria alvo.

## E18 — #1601: causa raiz era o dispatcher, nao o handler (19/out)

O defeito `isError:false` nao estava so em `handle_store_memory`: o dispatcher
(`server_impl.call_tool`, ~L1633) capturava **qualquer** excecao de tool e devolvia
`{"error": ...}` como `TextContent` dentro de um resultado bem-sucedido. Por isso nenhum
handler podia sinalizar falha — o envelope sempre dizia sucesso.

Correcao (`866bbff4`):
- dispatcher: loga e **re-levanta** (o SDK mapeia excecao para `isError=True`);
- `handle_store_memory` e `handle_store_session`: recusa de escrita levanta, nao retorna;
- `except` de ambos re-levanta (antes reembalava o erro em texto).
- Teste de integracao no transporte real (`tests/integration/test_store_error_envelope.py`):
  recusa -> `isError=True`; aceite -> `False`. Load-bearing provado revertendo o dispatcher.

**Pre-existentes do merge upstream** (nao regressao nossa, comprovado por `git stash`):
9 falhas em `tests/unit/test_entity_filter_log_forgery.py` e
`tests/unit/test_vector_search_eligible_rows.py`.

**Armadilha de E2E:** `curl` no streamable-http recebe apenas o evento `: ping` em
`memory_store` — o resultado vem por outro evento e o curl nao o entrega. Prova confiavel no
vivo: consulta ao banco + cliente MCP real (o mesmo dos testes). Busca via curl funciona.

## E19 — #1603 mergeado: o que o revisor viu e o CI nao (19/out)

PR #1603 mergeado por doobidoo (20:21), issue #1601 fechada. Greptile: APPROVED. CI 17/17 verde.

### Os 3 P1 do Greptile (todos procedentes)

1. **Payload vazio ainda "sucedia"**: `memory_store` com `content:""` retornava antes do caminho de erro;
   `memory_store_session` idem quando todo turno era espaco. Corrigido: ambos levantam.
2. **Chunks parciais**: `store_memory` devolve `success:True` com `failed_chunks>0`; o handler
   reportava sucesso. Corrigido: `failed_chunks` conta como falha.
3. **Endpoint HTTP**: o FastAPI `/mcp` chama `call_tool` sem o wrapper do SDK, entao a excecao
   virava JSON-RPC `-32603` em vez de tool result com `isError`. Corrigido com try/except local e
   `_wrap_tool_result(..., is_error=True)`.

O achado 3 e uma consequencia DIRETA do fix original: eu tratei o caminho MCP (SDK) e esqueci o
caminho HTTP, que e outro dispatcher. **Licao de blast radius:** "propagar excecao" tem um
consumidor por transporte; enumere-os antes.

### O que o CI pegou e o Greptile nao

`tests/web/api/test_jsonrpc_endpoint.py` falhou com `-32603` — o mesmo achado 3, visto pelo teste
antes do revisor. Confirmou o fix.

### Testes que congelavam o contrato antigo

6 testes esperavam `TextContent` de erro em vez de excecao:
- `test_server_handlers.py` (3): atualizados para `pytest.raises`.
- `test_unified_handlers.py` (3): o **setup** gravava fixture e morria na dedup semantica, porque
  textos com mesmo prefixo e UUID diferente sao considerados similares. Criado `_store_fixture()`
  que tolera recusa por dedup e propaga qualquer outro erro — o teste e sobre delete/list.

**Licao:** quando o contrato de erro muda, o que quebra nao sao so as assercoes — e o **setup**
que dependia do caminho silencioso.

### Numeros

- Suite integracao+web: **507 passed** (base limpa `476db1c8`: 501, +6 testes novos).
- CI do PR: 17 pass, 1 skipping, 0 fail.
- Fork sincronizado: `be9a1636`.

## E20 — Reranker com intervalo de confianca (20/out)

Matriz fechada com **bootstrap pareado** (10.000 reamostragens sobre os 120 casos, IC 95%),
script proprio `eval/rerank_ci.py` (o `onnx_rerank.py` publica so agregados).

| candidatos | recall@10 | ganho | IC95 | melhoraram | pioraram | p50 |
|---|---|---|---|---|---|---|
| base | 0,5667 | — | — | — | — | 0,19-0,28 s |
| **40** | **0,6667** | **+10,0 p.p.** | **[+5,0 · +15,8]** | 12 | **0** | 2,10 s |
| **100** | **0,7333** | **+16,7 p.p.** | **[+10,0 · +23,3]** | 20 | **0** | 5,31 s |

**Multilingue REJEITADO com numero:** `bge-reranker-v2-m3` int8 (545 MB) da 0,6750 @10 com 40
candidatos = **+0,83 p.p. sobre o bge-base por 6,5x a latencia** (13,7 s vs 2,1 s) e o dobro do
disco. Ganho concentrado em @5 (0,675 vs 0,642) — so importa se o cliente mostra 5 e nunca 10.

**Propriedade mais forte que a media:** zero casos pioraram nos dois cortes, em 120 casos. O
reranker promove, nao troca acerto por erro.

**Validacao do instrumento:** o script reproduziu exatamente os valores ja conhecidos
(0,6667 com 40; 0,7333 com 100) — nao foi ajustado para a conclusao.

**Licao de metodo:** ganho de recall reportado sem intervalo e anedota. Com n=120, +10 p.p. tem
piso de +5 p.p.; o numero so vira decisao quando o piso tambem vale a pena.

Publicado na #1602 (com a pergunta ao Henry sobre onde ligar) e na #1095 (D5 do ramo UIAA/hermes).
