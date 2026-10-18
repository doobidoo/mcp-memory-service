# Relatório da noite — 17/18 out 2026 (execução autônoma)

Contrato: `docs/_fork/plano-autonomo-20261018.md`. Nada de produção foi tocado:
serviço, acervo vivo, main e upstream intactos. Todo o trabalho em
`~/git/mms-lab-precisao` e no snapshot descartável.

## 1. Entregas

| # | entrega | estado |
|---|---|---|
| E1 | divergência do harness (0% × 20%) | explicada em parte (ver §4) |
| E2 | baseline oficial | **Recall@10 = 20%** (2/10), determinístico |
| E3 | A/B de modelos | **interrompido por falta de poder discriminante** (§4) |
| E4 | relatório | este arquivo |

Commits no repo principal: `e96444f3` (contrato), `3a4c401b` (diário de gates),
`6456b1ae` (truncamento), `4bb50e61` (modelo do índice), `8f5625f6` (custo de
contexto).
Commit no lab: `9f2744e9` (harness + A/B, 13 testes verdes, ruff limpo).

## 2. Achado A1 — o índice foi construído com outro modelo

O serviço está configurado com `MCP_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`,
mas os vetores gravados são de **all-MiniLM-L6-v2**:

| modelo comparado com o vetor gravado | cosseno médio | > 0,99 |
|---|---|---|
| all-MiniLM-L6-v2 | 0,9997 | 100% |
| paraphrase-multilingual (configurado) | 0,3160 | 0% |

A consulta é embedada com um modelo e comparada com documentos de outro espaço
vetorial. **É a causa raiz mais grave** e precede qualquer discussão de ranking.

## 3. Achado A2 — truncamento silencioso em 128 tokens

O `tokenizer.json` do modelo carrega `truncation max_length=128`. Sobre 27.777
memórias (4.473.710 tokens): 28,2% passam de 128 tokens e a cobertura é **46,4%**.
Mais da metade do acervo nunca entra no vetor. Memórias longas — inclusive as
importadas do `state.db` — são vetorizadas só pelo começo.

## 4. Achado A3 — o A/B não discrimina (e por quê)

Resultados medidos (corpus = store default, 20.310 memórias):

| cenário | recall@10 | mrr@10 | ms/embedding |
|---|---|---|---|
| harness (caminho do serviço) | **0,200** | 0,125 | — |
| paraphrase @128 (cosseno puro) | 0,000 | 0,000 | 17,4 |
| all-MiniLM @256 (cosseno puro) | 0,000 | 0,000 | 60,2 |

Dois modelos, dois resultados idênticos no zero. O teste semântico mostra que os
modelos funcionam (paráfrases com cosseno 0,958), e o mesmo documento gold cai na
posição 79 de 301 num corpus pequeno. Interpretação: **o fato relevante está no
trecho truncado**, então o embedding do gold não representa o que a pergunta pede.
Sem resolver A2 (chunking), nenhum A/B de modelo pode discriminar — todos medem a
mesma cegueira.

Fica aberto, e registrado como aberto: por que o harness (que usa os vetores
heterogêneos) acha 2 em 10 enquanto a comparação consistente acha 0. Hipótese de
trabalho: vantagem acidental da mistura de espaços (hubness). Não foi provada.

## 5. Erros e correções desta noite

1. O harness rodava por default no banco **vivo**, não no snapshot → passou a exigir
   `--db` explícito.
2. O gabarito exigia a string "MAP-Elites + UCB1", inexistente no acervo → 1 caso
   foi invalidado; corrigido para "MAP-Elites".
3. O A/B comparava corpus de 27.777 com o de 20.310 que o serviço realmente varre
   (`retrieve(store='default')`) → `--store` adicionado.
4. O executor delegado estourou o tempo fazendo sweep de threads em vez de entregar
   o número → escopo cortado, execução assumida com script de bancada.
5. `pkill -f` casou com o próprio shell de depuração e o matou → usar PIDs
   específicos.

## 6. O que eu NÃO fiz

- re-embedar o acervo (exige decisão sobre qual modelo — e o A/B não deu base);
- concluir os cenários granite/e5 (interrompidos: não discriminaram nos dois
  primeiros, e cada um custa 20–40 min de CPU);
- criar o repo `tripulacao` (segurado por decisão do Claudio);
- qualquer ação em produção, main ou upstream.

## 7. Próximo passo, em ordem

1. **Experimento de chunking** no lab: dividir memórias longas em blocos de ≤128
   tokens, embedar cada bloco e considerar acerto se qualquer bloco do gold entrar
   no top-k. É o único caminho para saber se A2 é o teto real.
2. Só depois: re-embedar com **um** modelo + `embedding_version` por linha (ADR-0014).
3. Só depois: A/B de modelos, agora com poder de discriminação.

Ordem justificada: medir modelo antes de resolver o truncamento é medir a mesma
cegueira quatro vezes — foi o que esta noite demonstrou.
