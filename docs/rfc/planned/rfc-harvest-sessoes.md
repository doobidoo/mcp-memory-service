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
