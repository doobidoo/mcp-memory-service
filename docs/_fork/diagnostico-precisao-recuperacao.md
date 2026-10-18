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

### 3.1 Achado físico (medido em 18/out): 53,6% do texto nunca vira vetor

O `tokenizer.json` do modelo em uso (`paraphrase-multilingual-MiniLM-L12-v2`)
traz a truncagem **gravada dentro do arquivo**:

```
enable_truncation -> {'max_length': 128, 'stride': 0,
                      'strategy': 'longest_first', 'direction': 'right'}
```

Qualquer texto acima de 128 tokens é cortado antes de virar embedding — e o corte
é silencioso. Medido sobre o snapshot (27.777 memórias vivas, 4.473.710 tokens
reais):

| métrica | valor |
|---|---|
| p50 / p75 / p90 / p95 / p99 (tokens) | 58 / 150 / 374 / 577 / 970 |
| média | 161,1 tokens |
| memórias > 128 tokens | 7.834 (28,2%) |
| memórias > 256 tokens | 4.419 (15,9%) |
| **cobertura do embedding atual** | **46,4% dos tokens** |

Consequência direta: **mais da metade do acervo é invisível para a busca**. Uma
memória de sessão importada (chunk de 1.500 caracteres ≈ 400 tokens) tem ~70% do
conteúdo descartado no momento do embedding.

Isso reordena as causas: o item 1 (chunk heterogêneo) e este achado são o mesmo
problema visto por dois ângulos — a representação perde conteúdo por **tamanho**
antes de perder por **heterogeneidade**.

Correções candidatas, em ordem de custo:

1. **Aumentar o limite** (`--max-len` 256/512). Barato, mas o MiniLM foi treinado
   com 128: acima disso a qualidade pode cair. Precisa de medição — é o que o A/B
   em curso faz.
2. **Chunking por sentença/tema** com limite de ~128 tokens e ponteiro para o pai.
   Resolve por completo, ao custo de multiplicar linhas de embedding (×2 a ×4).
3. **Modelo com contexto nativo maior** (granite/e5/BGE-M3: 512 tokens), que resolve
   o limite sem inventar chunking — mas não dispensa o chunking para as memórias
   acima de 512.

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

## 6. Multilíngue: o drift EN/PT-BR também é um viés de recuperação

Contexto: o upstream (Henry) escreve em inglês; nós escrevemos em português. O acervo é
misto — heurística no banco vivo: ~7.900 memórias em PT, ~3.200 em EN, ~16.700
indeterminadas; 14.071 contêm acentuação (ç/ã), portanto o armazenamento é UTF-8 e não há
problema de codificação. Modelo em uso: `paraphrase-multilingual-MiniLM-L12-v2` (384 dim,
ONNX), com `MCP_LOCALE=en,pt_BR` e `HARVEST_LOCALE=en,pt_BR`.

Medição 1 — mesmas perguntas, mesmos fatos, dois idiomas:

| métrica | pergunta em PT | pergunta em EN |
|---|---|---|
| Recall@10 | 16% (1/6) | **50% (3/6)** |
| MRR@10 | 0,042 | 0,111 |

Medição 2 — controle com memórias escritas em **português puro** (para excluir a hipótese
"o acervo é que fala inglês"): mesmo assim o inglês ranqueia melhor.

| fato (memória em PT) | posição com query PT | posição com query EN |
|---|---|---|
| Botão "Baixar CCIR" | 4º | **1º** |
| Autenticação Gov.br | 5º | **1º** |
| Consulta de Imóvel Rural | 10º | 10º |
| Compartilhamento de Imóveis | ausente | ausente |

Conclusão: **não é o acervo, é o modelo.** O MiniLM multilíngue é assimétrico — foi treinado
predominantemente em inglês e recupera melhor quando a consulta está em inglês, mesmo que o
documento esteja em português. Consequência prática grave para nós: escrever em PT-BR (a
preferência declarada do Claudio) piora a recuperação em relação a escrever em inglês.

Implicações para a proposta:

1. **A expansão de query (I4) deve ser bilíngue.** Traduzir/expandir a consulta para inglês
   ataca duas causas de uma vez: o gap de vocabulário técnico *e* o viés do modelo. É a
   intervenção mais barata — não exige re-embedar 27 mil memórias.
2. **Bancos por idioma (ideia do Zero) são laboratório, não correção.** Servem para isolar
   variáveis (medir recall monolíngue, testar modelo novo sem tocar produção, validar
   re-embed). Não corrigem o viés, porque o viés está na consulta, não na partição.
3. **Trocar o modelo é o caminho caro** e deve ser medido antes: `multilingual-e5`,
   `BGE-M3` e `jina-embeddings-v3` são candidatos com melhor cross-lingual, mas exigem
   re-embed de todo o acervo (mecanismo `embedding_version`/`pending` já existe — ADR-0014/0015).
   O harness I2 é o que permite decidir com número.
4. **UTF-8 já está resolvido.** O risco real nunca foi a codificação, e sim o tokenizer de
   relevância (corrigido para Unicode `\w+` nos commits 7c3307be/f0d8d90a/41224fcd) e o
   FTS5 trigram, que não indexa termos com menos de 3 caracteres.

### 6.1 O viés também é de tamanho, não só de modelo (medido em 18/out)

O achado da truncagem em 128 tokens (§3.1) tem efeito assimétrico por idioma. Medido no
mesmo snapshot, classificando por heurística de stopwords:

| idioma | n | tokens médios | p90 | >128 tokens | cobertura do embedding |
|---|---|---|---|---|---|
| PT | 11.612 | 231,1 | 574 | 43,6% | **38,4%** |
| EN | 4.435 | 181,6 | 400 | 39,6% | **48,6%** |
| indeterminado | 11.729 | 84,0 | 118 | 8,7% | 66,5% |

Memórias em português são 27% mais longas em tokens e perdem 10 pontos percentuais a mais
de conteúdo no corte. Ou seja: **parte do viés EN>PT é artefato de configuração**, não
apenas limitação do modelo — escrever em português produz textos mais longos, que o
truncamento de 128 tokens pune mais.

Isso reforça a ordem das intervenções: aumentar o limite ou fragmentar (§3.1) corrige
simultaneamente a cobertura geral e o viés de idioma, sem tocar no modelo. E dá uma
previsão testável: se o ganho em PT for maior que o ganho em EN após elevar o limite, a
hipótese está confirmada.


## 7. Candidatos a modelo de embedding (verificado no MTEB Multilingual, captura 2026-10-17)

Todos são modelos de embedding para retrieval — mesma função do
`paraphrase-multilingual-MiniLM-L12-v2` que usamos hoje (384d, ~118M parâmetros, arquitetura
de 2021). O ranking abaixo é MTEB Multilingual v2; **retrieval** é a coluna que importa para
nós (nDCG@10), e MIRACL seria a mais próxima do nosso caso cross-lingual PT↔EN.

| modelo | params | dim | retrieval (nDCG@10) | observação |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B | 8B | 4096 (MRL) | **70,9** | topo absoluto; inviável em CPU |
| Qwen/Qwen3-Embedding-4B | 4B | 2560 (MRL) | 69,6 | precisa GPU |
| google/gemini-embedding-001 | — | 3072 | 67,7 | API (não local) |
| jina-embeddings-v4 | 3,8B | 2048 | 66,4 | GPU |
| **microsoft/harrier-oss-v1-270m** | 270M | 640 | **66,4** | melhor relação qualidade/tamanho |
| granite-embedding-311m-multilingual-r2 | 311M | 768 | 65,2 | IBM, forte em multilíngue |
| **Qwen/Qwen3-Embedding-0.6B** | 0,6B | 1024 (MRL) | **64,65** | melhor custo/benefício geral |
| **granite-embedding-97m-multilingual-r2** | 97M | **384** | **60,3** | **mesma dimensão do nosso** → storage idêntico |
| multilingual-e5-large-instruct | 0,6B | 1024 | 57,1 | exige prefixos `query:`/`passage:` |
| BGE-M3 | 0,6B | 1024 | ~62 (mean) | dense+sparse+ColBERT; 8192 tokens |
| jina-embeddings-v3 | 0,57B | 1024 (MRL) | 55,8 | LoRA por tarefa; 8192 tokens |
| multilingual-e5-small | 96M | 384 | 50,9 | comparação direta com o nosso |
| **paraphrase-multilingual-MiniLM-L12-v2 (atual)** | 118M | 384 | não listado (~50) | arquitetura de 2021 |

Leitura:

1. **Trocar mantendo 384 dimensões custa zero de storage**: `granite-embedding-97m-r2`
   entrega ~60 de retrieval contra ~50 do atual, com o mesmo tamanho de vetor.
2. **O sweet spot local é 0,6B**: Qwen3-0.6B (64,65) e harrier-270m (66,4) triplicam ou
   quadruplicam a qualidade do atual, com storage de 640-1024d.
3. **BGE-M3 e jina-v3 não são os melhores hoje** para retrieval puro, apesar da fama —
   BGE-M3 compensa pelo modo sparse (que o nosso FTS já cobre) e jina-v3 pela LoRA por tarefa.
4. **Custo do re-embed (medido em 18/out, nesta máquina, CPU):** `paraphrase-multilingual`
   ONNX processou **156,8 textos/s** (batch 16, seq ≤256, textos de ~336 chars). Para as
   27.777 memórias do acervo: **~3 minutos**. Modelos de 0,6B devem ficar na casa de 10-30
   min — a medir. **Não há custo em tokens**: embedding local consome CPU e tempo, não API.
   Tokens só se aplica a modelo em nuvem (OpenAI/Cohere/Voyage/Gemini).
5. **Ressalva metodológica**: MTEB mean não é o nosso caso. Só o harness I2, medindo PT e EN
   no nosso acervo, responde. MIRACL PT↔EN é o número que mais deveria pesar.

## 8. Reprodução

```bash
# gabarito existe?
sqlite3 ~/local-data/mcp/sqlite_vec.db \
  "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL AND content LIKE '%MAX_SESSION_CHARS%';"
# mesma pergunta nos 4 modos (comparar posição do gabarito)
# semantic / exact / hybrid / ranked — ver tabela da §2
```
