# RFC — Arco Harvest de Sessões (learning de colheitas)

> Status: planned · Autor: Zero (MANAGER) · G0 por seven · 11/out/2026 · fork mcp-memory-service
> Par simétrico do learning de memórias. As sessões .jsonl são a fonte mais rica e estão sub-colhidas.

## Problema (diagnóstico, mem 00cdc317 + G0 seven)
Dois furos: **cobertura** (61% das 1314 sessões / ~550MB nunca colhidas; tracker regrediu de 667→19 IDs em 11/set)
e **profundidade** (harvest é regex + reescrita LLM de trechos; não indexa o transcript — raciocínio que não casa regex se perde).

## ENTREGA 1 — Causa raiz da regressão do tracker (confirmada no código)
Dois caminhos duplicados atualizam o tracker (`scheduler.py:605-619` e `server_impl.py:1842-1860`), com 3 mecanismos de perda:
1. **Leitura que falha vira vazio, não aborta** (`scheduler.py:601-602`, `server_impl.py:1799`: `except: pass` → `set()`).
   Se `list_memories` falha (lock SQLite / VACUUM do backup-and-sync) → `already=∅` → grava só o lote (~19).
2. **Delete-antes-de-store sem transação** (`scheduler.py:610-611`): deleta o tracker de 667 ANTES de confirmar o store do novo.
3. **Corrida scheduler×handler** (lock só na leitura do handler, não na escrita).
Confirmação empírica: 216 versões do tracker; MAX 667 @ 11/set 12:00; serrilhado 667→400→…→19. Não é truncamento de string (conjunto regride).

### Fix desenhado (para o reg na FASE 1)
- Helper único `persist_tracker` (eliminar duplicação) com: (a) `read_tracker_strict` que distingue "não existe" (∅, 1ª run)
  de "falha de leitura" (None → ABORTA, não grava); (b) merge monotônico `novo = atual ∪ recém`; (c) guarda anti-shrink
  (`len(novo) < len(atual)` só com `force_shrink=True`); (d) **store ANTES de delete**; (e) lock compartilhado na escrita.
- Reconstrução one-shot: unir os IDs das **216 versões históricas** (contém a de 667) + os 228 com `metadata.harvest_session_id`.
  Nota: só 228/4890 memórias têm `harvest_session_id` → não dá para depender dele; passar a gravá-lo em TODA memória de harvest.

## ENTREGA 2 — Decisão de arquitetura: FASEADO A→B (C descartado)
- **A (consertar cobertura): fazer primeiro.** Tracker + backfill + scheduler. Pré-req barato; sem ele B/C herdam cobertura furada.
- **B (resumo-por-sessão via LLM): fazer em seguida.** 1 resumo denso por sessão como `memory_type=session-summary` com embedding.
  Ganho: a sessão vira UNIDADE RECUPERÁVEL (hoje só fragmentos); captura o raciocínio que o regex perde (alvo do arco).
  Custo controlado: O(nº sessões) não O(nº chunks); infra LLM já existe (`rewriter.py` BATCH_PROMPT, Groq+fallback); aditivo ao heurístico.
- **C (RAG completo chunk+embed de tudo): DESCARTADO.** Custo O(nº chunks) explode; ruído alto (dumps/logs); contra local-first.
  Reavaliar só se B provar insuficiente.

## ENTREGA 3 — Qualidade do acervo atual
Amostra de 25 de 4890 session-harvest: **~84% ouro** (insight autocontido, timeless, com tool/comando/regra), ~16% ruído
(fragmentos de checkpoint temporal + wrappers `TYPE:` vazados). Veredito: NÃO reprocessar; B é aditivo; higiene barata via regex opcional.

## Fases do arco (ARC 323e1654) — estado em 11/out (pós-execução)
- FASE 0 planejamento (este RFC) — ✅ DONE.
- FASE 1 fix do tracker (merge monotônico + anti-shrink + store-antes-de-delete + lock de módulo) — ✅ DONE,
  commit c5f3a3a4, NO VIVO. Gate rok→reg→tuvok em 4 rodadas (tuvok reprovou 3x: delete morto por page_size=1,
  persist_tracker código morto, guard anti-shrink INVERTIDO que sabotaria o backfill — todos corrigidos). 6/6 testes.
  Tracker reconstruído 19→1299 IDs gravado no vivo (une as 216 versões históricas + harvest_session_id).
- FASE 1b (NOVA, descoberta na execução) fix do triage cego ao formato — ⚠️ WORKAROUND feito, FIX DE CÓDIGO PENDENTE.
  CAUSA-RAIZ REAL do furo de 61%: harvest/triage.py score_session (L112-114) só lê o formato v4 'payload', mas as
  sessões são formato LEGADO 'kind' (version/kind/data). Com MCP_HARVEST_TRIAGE=1 (estava ligado no serviço),
  score_session dava 0.0/empty para TODAS → _apply_triage descartava 100% ANTES do parser. O parser entende os dois;
  o triage não. Workaround: MCP_HARVEST_TRIAGE=0 (comprovado restaura sessions:50|candidates:11). Fix: score_session
  ler o formato 'kind' OU reusar TranscriptParser.parse_file(); _apply_triage fail-open em 0-msgs-reconhecidas (hoje
  só em exceção). Gate rok→reg→tuvok, depois religar triage.
- FASE 2 backfill das 788 pendentes (546 de outubro) — EM WORK. Tracker reconstruído gravado. Dry-run (triage off)
  comprova candidatos. FALTA: harvest real com LLM (operação longa, cap anti-flood), medir qualidade.
- FASE 3 religar scheduler (MCP_HARVEST_SCHEDULE opt-in, idempotente) — depende FASE 2.
- FASE 4 (decisão B) resumo-por-sessão session-summary — depende FASE 3.

## Incidente resolvido na execução (11/out)
Serviço memory-service estava SEM backend de embedding (onnxruntime/tokenizers ausentes no venv apesar de
MCP_MEMORY_USE_ONNX=1) → health 200 mas RECUSAVA toda escrita ("embedding backend unavailable; refusing to write
hash pseudo-vectors"). A trava do produto é correta (evita poluir 26598 embeddings), mas o serviço parecia saudável.
Fix: `VIRTUAL_ENV=.venv uv pip install onnxruntime tokenizers huggingface-hub` + restart. Backend ONNX precisa do
extra sqlite-ml (onnxruntime>=1.17 + tokenizers). Lição: health 200 ≠ escrita funcional — testar escrita real no boot.

## Riscos (G0)
Duplicação da lógica do tracker (tocar os 2 ou unificar) · harvest_session_id cobre só 228/4890 · delete sem transação +
lock só na leitura · sessions_to_track com stored==0 reseleciona vazias (infla custo de B). Verificar logs do scheduler 09-11/set (hipótese VACUUM).

---

# v0.3 (11/out) — Evolução informada pela PoC e pela descoberta de idioma

> Status: planned v0.3 · Enriquecido após a PoC de trituração (sessão cb14f957) + análise arch (seven) +
> pesquisa web (Weaviate, arXiv). FATO = ancorado em código/dado. PROPOSTA = recomendação acionável.

## v0.3-§1 PoC de trituração manual (ideia Claudio, insight em 2º plano)
Experimento: Zero tritura UMA sessão (a própria, cb14f957, formato Kiro CLI atual) SEM o memory-service,
com julgamento de prosa. Resultados (doc: pesquisa/frente-d-attribution-sem-disciplina-2026-10/POC-...md):
- **FATO: 94% do .jsonl é ToolResults** (I/O: SQL/pytest/git). A prosa (prompt+raciocínio) é 6% (345KB de 5.5MB).
  → RAG do bruto afogaria o conhecimento em ruído. **Colher a PROSA, não o arquivo.** (RAG-cru DESCARTADO por dado.)
- **FATO: estrutura multi-tema** — principal (fechar learning-loop) / secundário (harvest, pesquisa) /
  paralelo (evolução de agentes, doc-sync) / ortogonal (fato>memória, não depender da disciplina do agente).
  Conhecimento de sessão é MULTI-CAMADA (fato atômico / resumo-de-sessão / grafo de temas), não uniforme.
- **FATO (PoC de recall): o learning afogou o tema.** memory_context("feedback passivo…") trouxe no topo
  "EIP não pode estar em 2 lugares" (rede, utility 0.95) — nenhum belief do tema. utility_prior ADITIVO alto
  afoga relevância baixa. → fix na fila (FASE 7, util MODULAR não aditivo).

## v0.3-§2 DESCOBERTA DE IDIOMA (pergunta do Claudio: pt-BR confunde o learning que roda p/ EN?)
**SIM — confirmado empiricamente. Mas não onde se esperava.** O harvest (colheita) JÁ é bilíngue; a
degradação pt-BR está no CONSUMO (re-rank/busca). Mapa (arch, ancorado):
- **CRÍTICO — re-rank A tokeniza ASCII** (`context_injection.py:62` `_WORD_RE = re.compile(r"[a-z0-9]+")`).
  Quebra acento: `injeção`→`inje`, `decisão`→`decis`, `função`→`fun`, `saída`→`sa`. Metade dos termos
  técnicos pt-BR é MUTILADA → overlap lexical falha → relevância baixa (0.13-0.30 na PoC). ESTA é a causa-raiz
  da PoC afogar o tema (soma-se ao util aditivo). Teste empírico confirmou (5/5 termos mutilados).
- **SINTOMA — `_STOPWORDS`** (:70) tem `"n"`,`"o"` (fragmentos pós-quebra) como band-aid do bug acima.
- **PARCIAL — FTS5 `tokenize='trigram'`** (`migrations.py:52`) é accent-SENSITIVE sem fold: `injeção`≠`injecao`.
- **OK — parser/extractor/rewriter**: já multi-locale (`patterns/pt_BR.yaml`, `_detect_language`, rewriter
  "same language"). triage e derivação de beliefs são idioma-neutros. (o furo do triage é FORMATO, não idioma.)
- **IRONIA ESTRUTURAL**: o embedding é `paraphrase-multilingual-MiniLM` (entende pt-BR), MAS a tabela
  `beliefs` NÃO tem coluna de embedding (`012_add_belief_store.sql`) → o re-rank é FORÇADO a usar overlap
  ASCII em vez do embedding multilíngue que já existe. Temos a ferramenta certa e usamos a errada.

**Respaldo (pesquisa web):** Weaviate — "BM25 e a tokenização que o alimenta falham silenciosamente quando
o analisador está errado; nenhum ajuste de embedding salva. Divida uma palavra que não devia e o BM25 não
tem nada útil para casar." Técnica nomeada: **accent folding**. arXiv 2210.05481 (tokenização p/ línguas sem
tokenizer custom); arXiv 2510.20609 (BM25 word-level é forte e 10× mais rápido p/ PL→PL — lexical importa).

## v0.3-§3 PROPOSTA DE EVOLUÇÃO DO HARVEST CLI + RECALL idioma-consciente (incrementos)
Separar COLHEITA (quase ok) de RECALL (quebrado em pt-BR). Ordem por ganho/custo:
- **I5 [CRÍTICO, barato — 1 linha + testes]**: `_WORD_RE` → `re.compile(r"\w+", re.UNICODE)` (ou `[^\W\d_]+`)
  + `casefold()`. Destrava o learning em pt-BR HOJE. Sem ele, qualquer resumo-por-sessão pt-BR fica invisível
  ao re-rank. Band-aid stopwords de fragmento tornam-se inócuos. (Opcional: normalização NFD p/ accent-fold.)
- **I9 [cobertura]**: `harvest_session_id` em TODA memória de harvest (hoje só 228/4890) — pré-req de ligar sessão↔beliefs.
- **I2 [colheita]**: rebaixar ToolResults a contexto (não candidato 1ª classe); extrair só prosa user/assistant → menos ruído ao LLM.
- **I3/I4 [Opção B, = FASE 4]**: `memory_type=session-summary` — 1 resumo denso por sessão via LLM (reusa rewriter BATCH),
  COM embedding, prompt multi-tema (principal/secundário/paralelo/ortogonal), preservando idioma. Sessão vira unidade recuperável.
- **I6 [estrutural]**: dar embedding ao belief (coluna em `beliefs` ou cache on-the-fly) → re-rank faz cosine
  multilíngue real (não overlap ASCII). Custo O(nº beliefs), baixo. Caminho híbrido: lexical unicode-aware + cosine, fusão RRF (`_fuse_rrf` já existe).
- **I7 [FTS5]**: avaliar `unicode61 remove_diacritics 2` (fold de acento) vs trigram — decidir por benchmark pt-BR.
- **I8**: `_consolidate_similar._jaccard` → tokenização unicode-aware (consistência com I5).
- **I10 [prova]**: benchmark de recall pt-BR (query acentuada → espera belief/memória pt-BR) como regressão, ANTES de religar o scheduler (FASE 3).

**Ordem:** I5 → I9 → I2 → I3/I4 → I6 → I7/I8/I10. I5 é o maior ganho/custo (destrava já).
**Conexão com o learning:** colher não basta — os fatos têm que VIRAR BELIEFS (derivação/dreaming) para serem
injetáveis, E o re-rank tem que enxergar pt-BR (I5/I6). harvest → dreaming → learning: o ciclo só fecha com idioma-consciência.

## v0.3-§4 ARQUEOLOGIA: a decisão ASCII do Henry (escopada) vs os 3 ASCII indevidos (arch + git)
Claudio lembrou que o ASCII foi DISCUTIDO com o Henry sob condições. CONFIRMADO no repo (arch, ancorado):

**LEGÍTIMO (decisão do Henry, escopada a IDENTIFICADORES):** Codeberg #56/#61 (11/jun), doc
`docs/handoffs/two-phase-query-skeletons.md:19` ("entity_id normalization — agreed: NFKD + hash fallback").
Implementado em `storage/graph.py:43-60` `normalize_entity_id()`: NFKD → strip combining → slug `[a-z0-9-]` →
fallback `e-<sha1[:8]>` para scripts não-latinos. CORRETO para chaves/ids (determinístico, dedup estável). NÃO MEXER.

**INDEVIDO (3 pontos, introduzidos por nós, NÃO acordados com o Henry, aplicam ASCII onde fere relevância):**
1. `storage/context_injection.py:62` `_WORD_RE=[a-z0-9]+` (commit f752dd13, 03/out) — tokenizer do re-rank.
2. `reasoning/entities.py:175` `[a-zA-Z]{3,}` — tokenizer de entidade NLI.
3. `storage/mixins/migrations.py:56` FTS5 `tokenize='trigram'` — accent-sensitive (query "decisao"≠índice "decisão").

**PROVA EMPÍRICA (teste 11/out, tokenizer atual vs fix):** em 8 termos técnicos pt-BR, ASCII mutila **11 palavras**
("consolidação"→`consolida`+`o`, "código"→`c`+`digo`, "injeção"→`inje`+`o`); o padrão `[a-zà-ÿ0-9]+` mutila **ZERO**.

**A solução JÁ EXISTE no próprio repo:** `harvest/parser.py:127` usa `[a-zà-ÿ]+` (Latin-1 completo, accent-aware).
O write-side (harvest) respeita acento; o recall-side (re-rank) não. **Não se perdeu em migração — nunca foi
propagado.** `MCP_LOCALE` governa o que é EXTRAÍDO, não o que é RANKEADO/BUSCADO.

**Mapa multilíngue VIVO vs NÃO-PROPAGADO (arch):**
- VIVO (write/extraction): config/locale.py (MCP_LOCALE), patterns/{en,de,pt_BR}.yaml, nli_patterns, ner_patterns,
  _detect_language (parser.py:120), rewriter "same language". Consumido por nli/rewriter/harvester/extractor/multilingual.
- NÃO-PROPAGADO (recall/learning): context_injection.py (ASCII, zero locale), belief.py/belief_service.py
  (grep locale/accent = VAZIO), entities.py:175, FTS5 trigram. O pt-BR MORRE na fronteira harvest→recall.

## v0.3-§5 PLANO DAS DUAS FRENTES — multilíngue (pt-BR garantido)
Princípio: NFKD-fold CONSISTENTE nos DOIS lados (query E índice) OU tokenizer Latin-aware `[a-zà-ÿ0-9]+` nos dois.
Preservar a decisão do Henry (entity_id NFKD+hash intacto). Gate completo rok→reg→tuvok, pt-BR como teste de regressão.

LEARNING (re-rank/recall):
- L-I5 [crítico, barato]: `context_injection._WORD_RE` → `[a-zà-ÿ0-9]+` (padrão do parser.py:127) + casefold.
  Remove a gambiarra de stopwords de fragmento (`"n"`,`"o"`). Benchmark pt-BR: query acentuada acha belief pt-BR.
- L-I6 [estrutural]: embedding no belief (tabela beliefs SEM coluna embedding hoje, 012_add_belief_store.sql) →
  re-rank cosine multilíngue real. Fecha também o FASE-7 (util modular) de forma coerente.
- L-entities: entities.py:175 → Latin-aware consistente.
HARVEST (busca/consolidação):
- H-FTS5 [I7]: `unicode61 remove_diacritics 2` (fold query+índice igual) OU documentar trigram accent-sensitive — decidir por benchmark.
- H-jaccard [I8]: `_consolidate_similar._jaccard` → tokenização Latin-aware (consistência).
- belief_service locale-awareness: propagar MCP_LOCALE à derivação (hoje VAZIO).

INVARIANTE: NÃO tocar `normalize_entity_id` (graph.py:43 — decisão Henry, ids). O fold de acento é para
RELEVÂNCIA/BUSCA, não para chaves. Dois usos distintos de ASCII, dois tratamentos.
