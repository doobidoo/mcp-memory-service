# SPEC — Delta-Sync Fase 3: consistência de embedding (§8.3)

- **Data:** 2026-10-07
- **Arco:** delta-sync #1345 · **Item:** 41c7c8e3 · **Branch:** `feat/delta-sync`
- **Cadeia:** RFC `rfc-delta-sync.md` §8.3 → ADR-0014/0015/0016/0017 → **este SPEC** → gate G0-G5 + E2E
- **Co-autoria RFC:** @ducanhnguyen223 (§8.3)
- **Escopo:** APENAS Fase 3 (LOCAL, sem transporte). Backend sqlite_vec (§9.4).

## Honestidade de escopo (ADR-0017 — ler antes)
A operação LOCAL não tem furo de embedding stale: `store()` grava conteúdo+embedding na
mesma transação; `content_hash` é content-only (conteúdo imutável por linha → mudar
conteúdo = nova linha = novo create); `update_memory_metadata` nunca toca conteúdo;
re-embed é rebuild offline. Logo o `embedding_pending` **não tem produtor vivo na Fase 3**.
A Fase 3 CONSTRÓI e PROVA o guardrail (dormente), para a Fase 4 apenas acionar. Os testes
dirigem o flag diretamente; NÃO inventam um update-de-conteúdo local falso.

## Requisitos cobertos
- **§8.3** (estado versionado único OU embedding_pending; nunca servir conteúdo-novo+vetor-velho) — núcleo.
- **R3'-adjacente**: provê o hook que a Fase 4 (apply) e uma troca de modelo usarão.
- Depende de: Fase 1 (sync_events), Fase 2 (HLC). NÃO cobre transporte (Fase 4).

## Requisitos funcionais (EARS)

**F1 — versão do embedding no evento**
> WHEN a create event is appended, THE storage SHALL record the embedding version `(embedding_model, embedding_dim)` on the event; legacy Phase 1/2 events remain NULL (unknown), not fabricated (ADR-0014).

**F2 — flag de pendência**
> THE `memories` table SHALL have `embedding_pending` (default 0 = consistent); the write/apply path MAY set it to 1 when a vector may not match the content version (ADR-0015).

**F3 — exclusão na leitura (hard exclude)**
> WHILE a memory has `embedding_pending = 1`, THE retrieve/search paths SHALL NOT return it, so a possibly-stale vector is never served as consistent (ADR-0016).

**F4 — store atômico permanece consistente**
> WHEN a memory is stored via the normal atomic path, THE storage SHALL leave `embedding_pending = 0` (content+embedding written together — no pending ever produced locally) (ADR-0017).

**F5 — clear re-habilita**
> WHEN a pending memory's embedding is regenerated and the flag cleared, THE memory SHALL become visible to search again.

## Requisitos não-funcionais
- **NF1** migração `017_add_embedding_consistency.sql`: ADD COLUMN `memories.embedding_pending INTEGER NOT NULL DEFAULT 0` + índice; ADD COLUMN `sync_events.embedding_model TEXT`, `embedding_dim INTEGER` (nullable). Aditiva, idempotente, `.down.sql` documental.
- **NF2** todos os read-sites que fazem join em `memories` ganham o predicado de exclusão (enumerar via `deleted_at IS NULL`). Perder um vazaria pending.
- **NF3** sem custo de inferência no read path; sem re-embed síncrono na leitura (ADR-0016).
- **NF4** retrocompat: Fase 1/2 eventos ficam com embedding_model NULL; kill-switch inalterado; store atômico intacto.

## Critérios de aceite (= fixtures §8.3, viram testes RED no Gate 2)
- [ ] **CA1 (F3/F5)** marcar uma memória `embedding_pending=1` → NÃO aparece em retrieve nem search_by_tag; limpar o flag → reaparece.
- [ ] **CA2 (F1)** create event carrega `embedding_model`/`embedding_dim` corretos (do modelo ativo); eventos legacy (sem) ficam NULL, não inventados.
- [ ] **CA3 (F4 honestidade)** store() normal deixa `embedding_pending=0` (caminho atômico nunca produz pending).
- [ ] **CA4 (F3 enumeração)** uma memória pending é invisível em TODOS os read-sites que juntam memories (retrieve, search_by_tag, e os demais que usam deleted_at IS NULL).
- [ ] **CA5 (NF1)** migração 017: colunas + índice criados; rows existentes `embedding_pending=0`; idempotente.

## E2E a quente (fecha a fase)
1. `MCP_SYNC_EVENTLOG=on`: store N memórias → eventos create carregam embedding_model/dim; todas `embedding_pending=0`; todas pesquisáveis.
2. Marcar 1 memória pending manualmente (simula o estado que a Fase 4/troca-de-modelo criará) → confirmar invisível em retrieve/search; limpar → reaparece.
3. Confirmar que nenhum store normal produz pending (guardrail dormente, correto).

## Gate (G0-G5) — delegação
- G0/G1 feito (Gate 1 seven APROVADO + ADRs + SPEC).
- G2 RED: rok (CA1-CA5, dirige o flag direto, sem content-update falso, anti-skip).
- G3 impl: reg — migração 017 + stamp no `_append_sync_event` + coluna/flag + exclusão em retrieve/search (TODOS os sites).
- G4 verify: suite + reindexar. G5 review: tuvok (enumerar read-sites, honestidade do guardrail dormente, retrocompat, mutation).
- E2E a quente após G5.
