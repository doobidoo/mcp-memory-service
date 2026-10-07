# SPEC — Delta-Sync Fase 1: event-log local (sqlite_vec, sem transporte)

- **Data:** 2026-10-07
- **Arco:** delta-sync #1345 · **Item:** fc7a578a · **Branch:** `feat/delta-sync`
- **Cadeia:** RFC `rfc-delta-sync.md` v0.4 → ADR-0002/0007/0008/0009 → **este SPEC** → gate G0-G5 + E2E
- **Co-autoria RFC:** @ducanhnguyen223 (§8 invariantes, §9 escopo)
- **Escopo:** APENAS Fase 1 (§9.5). Sem rede, sem HLC, sem cripto, sem hybrid/CF/Milvus.

## Requisitos cobertos (rastreabilidade)
- **R2** (event-log append-only auditável) — núcleo desta fase.
- **R4** (não corromper o banco) — via atomicidade (ADR-0008).
- **§8.1** (identidade de evento / idempotência / tombstone durável) — critérios de aceite.
- **§9.3.6** (envelope versionado) — coluna `schema_version`.
- **§9.4** (escopo de backend: sqlite_vec; inventário explícito do que é "sincronizado").
- NÃO cobre: R1/R3/R5/R6/R7/R8, §8.2/§8.3/§8.4/§8.5 (fases posteriores).

## Requisitos funcionais (EARS)

**F1 — append atômico por mutação suportada**
> WHEN a supported mutation (`store`, `store_batch` item, `delete`/soft-delete, `update_memory_metadata`) commits successfully, THE sqlite_vec storage SHALL append exactly one `sync_events` row within the SAME transaction as the mutation.

**F2 — nunca memória sem evento nem evento sem memória**
> WHILE a supported mutation is being written, IF the event append fails (and `MCP_SYNC_EVENTLOG` is enabled), THEN THE storage SHALL roll back the mutation together with the event (no partial state).

**F3 — identidade de origem única**
> WHEN an event is appended, THE storage SHALL assign a UUIDv4 `event_id` and record `(agent_id, event_id)` under a UNIQUE constraint, so re-applying the same event is a no-op.

**F4 — tombstone durável**
> WHEN a memory is deleted, THE storage SHALL append a `op='delete'` event that PERSISTS even after the `memories` row is hard-purged (`purge_deleted`/`_purge_tombstone`).

**F5 — envelope versionado**
> WHEN an event is appended, THE storage SHALL stamp `schema_version` on the row, so a future reader can reject/quarantine an unknown version without advancing a cursor.

**F6 — kill-switch**
> WHERE `MCP_SYNC_EVENTLOG` is disabled, THE storage SHALL NOT append events and SHALL behave exactly as before (back-compat).

**F7 — inventário explícito (escopo honesto §9.4)**
> THE event-log SHALL register ONLY `store`, `store_batch`, `delete` (+ `delete_by_timeframe`/`delete_before_date` that delegate to `delete`), and `update_memory_metadata`. All other mutations (`delete_by_tag`/`tags`, `update_memory_versioned` link, batch ops, `resolve_conflict`, `cleanup_duplicates`) SHALL NOT be presented as synced and SHALL be documented as not-yet-registered.

## Requisitos não-funcionais
- **NF1** migração `015_add_sync_events.sql` aditiva e isolada (nova tabela, zero ALTER em `memories`, zero trigger). Rollback documental `.down.sql` (runner não executa down — ADR via RFC §9.5.1).
- **NF2** caminho Cloudflare/Milvus intocado; hybrid dispara o evento 1× no primary sqlite (sem duplicação — blast radius do Gate 1).
- **NF3** overhead do append desprezível no store (1 INSERT na mesma tx).

## Critérios de aceite (= fixtures §8.1, viram testes RED no Gate 2)
- [ ] **CA1 (§8.1a)** aplicar o mesmo batch 2× com `event_id` fixos → estado idêntico, zero linha nova em `sync_events` e em `memories` (ON CONFLICT DO NOTHING + dedup de conteúdo).
- [ ] **CA2 (§8.1b)** crash simulado entre a mutação e o commit → replay converge; nenhum estado meio-aplicado (memória sem evento OU evento sem memória).
- [ ] **CA3 (§8.1c)** delete seguido de replay de um create antigo do mesmo `content_hash` → tombstone vence (não ressuscita); o delete event persiste após purge da linha `memories`.
- [ ] **CA4 (F1/F6)** com `MCP_SYNC_EVENTLOG` on: cada store/delete/update suportado gera exatamente 1 evento na mesma tx; com off: zero eventos, comportamento idêntico ao atual.
- [ ] **CA5 (F7)** `delete_by_tag` e `update_memory_versioned` NÃO geram evento (documentado); `delete_by_timeframe` gera N eventos (um por hash, herdado de `delete`).
- [ ] **CA6 (NF2)** hybrid com primary sqlite → store gera 1 evento (não 2); backends CF/Milvus não tocam `sync_events`.
- [ ] **CA7 (F8 — retry-safety sob concorrência)** com um escritor concorrente REAL segurando o lock do SQLite (`BEGIN IMMEDIATE` noutra conexão WAL), um `store_batch` que bate em `database is locked` no meio da transação DEVE, após o escritor liberar durante o backoff, concluir no retry com TODOS os itens persistidos + seus eventos (nada meio-aplicado) e deixar a conexão USÁVEL (sem transação pendurada, sem `cannot start a transaction within a transaction`).

## Requisito funcional adicionado (pós-review ducanhnguyen223)

**F8 — batch retry-safe sob lock concorrente**
> WHILE a competing writer holds the SQLite write lock, IF a `store_batch` attempt fails mid-transaction with `database is locked`, THEN after the lock is released THE storage SHALL retry the batch to successful completion (all items + their events persisted, no partial state) AND leave the connection usable (no lingering transaction).

> Motivação (cliente): este host roda múltiplos agentes concorrentes contra o MESMO `sqlite_vec.db` (single-writer). `database is locked` transitório é operação normal, não exceção. Uma falha mal tratada aqui ou perde a memória do agente, ou grava memória sem evento (quebra o delta-sync), ou deixa o agente com a conexão envenenada pela sessão inteira. Furo reportado e reproduzido por @ducanhnguyen223 (`store.py:232`, dois conns WAL reais). Implementação já no commit `7650a06` (rollback antes do `BEGIN` no início do callable retentável); este CA é a regressão que o trava.

## E2E a quente (fecha a fase, no ambiente real)
1. Habilitar `MCP_SYNC_EVENTLOG` no serviço local; store de N memórias reais via MCP → confirmar N linhas em `sync_events` com `event_id`/`agent_id`/`op=create` corretos, na mesma tx (sem órfãos).
2. Delete de 1 memória → `op=delete`; rodar `purge_deleted` → a linha `memories` some, o evento delete PERMANECE.
3. Desabilitar a flag → novo store NÃO gera evento (back-compat).
4. Coletar números (overhead de latência do store com/sem flag) como subsídio.

## Gate (G0-G5) — delegação
- G0 contexto (feito: Gate 1 seven + este SPEC + ADRs). G1 arch (feito: seven APROVADO).
- G2 RED: rok escreve CA1-CA6 como testes falhos (regra anti-padrão RED aplicada).
- G3 impl: reg — migração 015 + helper `_append_sync_event` + ganchos nos 4 call-sites.
- G4 verify: suite + reindexar -dev no grafo. G5 review: tuvok (atomicidade, R13-equivalente: CF intocado, fail-closed, testes honestos).
- E2E a quente no fork main após G5.
