# Diagnóstico — por que a memória "não acha" o que sabemos que ela tem

> Medido em 2026-10-17/18 nesta máquina (acervo 27.989 memórias, 93 importadas do Hermes
> em outubro). Pergunta que originou: "trabalho gigante no ESTADO.md, mas ele não acha o
> problema de precisão da memória".
> Método: gabarito conhecido (memórias cuja existência foi verificada por SQL) + consulta
> natural em quatro modos de busca (`semantic`, `exact`, `hybrid`, `ranked`).

## 1. O que NÃO é o problema (hipóteses refutadas)

| Hipótese | Teste | Resultado |
|---|---|---|
| Hubness (poucos vetores dominam) | 20 queries diversas, contagem de hashes no top-10 | **Refutada**: 194 hashes distintos em 200 aparições; nenhum aparece mais que 2× |
| Cobertura/ingestão (o conteúdo não entrou) | SQL no acervo vivo | **Refutada**: 10 memórias com `Qwen3.8-27B`, 12 com `15.000`, 1 com `MAX_SESSION_CHARS`, 3 com `delta vazio` |
| Orçamento head/tail descartou o miolo | dry-run da sessão do reviewer | **Refutada**: sessão não truncada; o fato estava na última mensagem e entrou |
| Falta de fusão léxica | mesma query nos 4 modos | **Parcialmente refutada**: quando há sobreposição de termos, `hybrid` vence com folga (posição 2 vs 15 do `semantic`) |

## 2. O que É o problema (medido)

Busca por **termo exato**: todos os modos acham. Busca por **pergunta natural**: falha.

| Pergunta natural | Gabarito | semantic | exact | hybrid | ranked |
|---|---|---|---|---|---|
| "qual o orçamento por sessão do importador" | `MAX_SESSION_CHARS` | ausente | ausente | ausente | ausente |
| "qual modelo escolhemos para subagentes depois do benchmark" | `Qwen3.8-27B` | ausente | ausente | ausente | ausente |
| "o que o reviewer bloqueou no importador" | `delta vazio` | 15º | ausente | **2º** | ausente |

Leitura: quando a pergunta compartilha vocabulário com o documento ("reviewer",
"importador"), o `hybrid` encontra em 2º. Quando a pergunta é **paráfrase** ("orçamento
por sessão" ↔ `MAX_SESSION_CHARS`; "modelo escolhido" ↔ `Qwen3.8-27B`), **nenhum modo
encontra**, embora a memória exista.

## 3. Causas, em ordem de impacto

1. **Representação — chunk heterogêneo.** O serviço fatia sessões em blocos de ~1.500
   caracteres por TAMANHO. Um bloco típico mistura pergunta do usuário, saída de
   ferramenta, raciocínio e resposta. O embedding resultante é a média de vários temas e
   não representa nenhum. Tema específico ("orçamento") fica diluído em ~1/5 do vetor.
   Afeta toda memória de sessão/harvest (~9 mil de 27,9 mil).
2. **Expansão de query — ausente.** Não há reescrita/sinônimos no caminho de recuperação.
   O acervo é rico em tokens técnicos exatos (`MAX_SESSION_CHARS`, `Qwen3.8-27B`), e a
   pergunta humana usa linguagem funcional ("orçamento", "modelo escolhido").
3. **Modo padrão — semantic puro.** `hybrid` existe e ganhou por 13 posições no único caso
   com sobreposição lexical, mas não é o default.

## 4. A lacuna que explica por que isso não foi visto antes

O projeto mede **MRR em benchmark sintético** (`eval/rerank_bench.py`) — ótimo para
comparar variantes de re-rank, cego para a pergunta real do usuário. **Não existe
avaliação de Recall@k com perguntas naturais e gabarito verificado.** Sem ela, todo ajuste
de peso é otimização sem termômetro: explica por que o arco de ranking avançou
(MRR 0,658 → 0,683 → 0,750) enquanto a sensação de "não acha" permaneceu.

## 5. Proposta (ordem por custo/benefício)

| # | Item | Classe | Custo | Ganho esperado |
|---|---|---|---|---|
| I1 | Default de busca = `hybrid` (cliente e/ou servidor) | config | trivial | imediato no único caso medido (15º → 2º) |
| I2 | Harness de Recall@k com perguntas naturais + gabarito | FEAT | baixo | cria o termômetro que falta; mede I1/I3 |
| I3 | Chunking semântico (por turno/tema em vez de 1.500 chars) | FEAT | médio | ataca a causa nº 1, a de maior alcance |
| I4 | Expansão de query (sinônimos + termos técnicos, LLM barato) | FEAT | médio | ataca paráfrase, que hoje é 0% |

Recomendação: **I2 primeiro** (mede), **I1 na mesma janela** (config, reversível), depois I3.

## 6. Reprodução

```bash
# gabarito existe?
sqlite3 ~/local-data/mcp/sqlite_vec.db \
  "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL AND content LIKE '%MAX_SESSION_CHARS%';"
# mesma pergunta nos 4 modos (comparar posição do gabarito)
# semantic / exact / hybrid / ranked — ver tabela da §2
```
