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

### 2.1 Harness de Recall@k com gabarito verificado (o que faltava no projeto)

Dez fatos cuja existência foi confirmada por SQL no acervo vivo, cada um pareado com uma
**pergunta natural** (sem o termo técnico). Mede-se a posição do fato no top-10.

| variante | Recall@10 | leitura |
|---|---|---|
| baseline (`semantic`) | **20%** (2/10) | 8 de 10 fatos conhecidos não aparecem no top-10 |
| `hybrid` | 20% | ganho só quando há sobreposição lexical |
| `ranked` | 20% | idem |
| `exact` | 0% | não é busca por substring — esperado |
| **removendo memórias-lixo** (teste/auto-generated) | **20%** | filtro de qualidade NÃO muda o recall |
| **com expansão de query** (pergunta + termo técnico) | **50%** | dobra o recall |

Conclusões que os números sustentam:

1. **Poluição não é a causa.** 3.431 memórias (12% do acervo, média 182 chars contra 553)
   ocupam o topo, mas filtrá-las não recupera um único fato adicional.
2. **Vocabulário vale ~30 pontos.** Quando a consulta inclui o token técnico, o recall vai
   de 20% para 50%.
3. **Representação vale ~50 pontos.** Mesmo com o termo exato na consulta, metade dos fatos
   continua fora do top-10 — o chunk heterogêneo não responde nem quando provocado.

### 2.2 Busca por termo exato vs pergunta natural (medição anterior)

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

## 5. Proposta (ordem por custo/benefício — REVISADA pelas medições)

| # | Item | Classe | Custo | Ganho medido/esperado |
|---|---|---|---|---|
| I4 | Expansão de query (sinônimos + termos técnicos) | FEAT | médio | **+30 pp de Recall@10** (20% → 50%, medido) |
| I2 | Harness de Recall@k com perguntas naturais + gabarito | FEAT | baixo | já prototipado abaixo; é o termômetro para tudo |
| I3 | Chunking semântico (por turno/tema, não 1.500 chars) | FEAT | médio | ataca os ~50 pp restantes |
| I1 | Default de busca = `hybrid` | config | trivial | 0 pp no agregado; só em query com sobreposição lexical |
| — | Filtro de memórias-lixo na busca | config/FEAT | trivial | **0 pp medido** — não priorizar pelo recall (vale por hygiene) |

Recomendação revista: **I2** (formalizar o harness), **I4** (maior ganho comprovado) e só
depois **I3**. I1 deixa de ser prioridade — a medição derrubou a hipótese de que o default
semântico fosse a causa principal.

## 6. Reprodução

```bash
# gabarito existe?
sqlite3 ~/local-data/mcp/sqlite_vec.db \
  "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL AND content LIKE '%MAX_SESSION_CHARS%';"
# mesma pergunta nos 4 modos (comparar posição do gabarito)
# semantic / exact / hybrid / ranked — ver tabela da §2
```
