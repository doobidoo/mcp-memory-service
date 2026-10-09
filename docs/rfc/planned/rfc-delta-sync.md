# RFC: Delta Event-Log Sync (malha de memória multi-agente)

**Data:** 2026-09-13 (rev. 2026-09-27)
**Autor:** Claudio + Zero (Kiro CLI)
**Co-autor:** @ducanhnguyen223 (§8 invariantes de correção + §9 alternativas/bootstrap/escopo)
**Branch de código:** `feat/delta-sync` (a partir de `upstream/main`)
**Base:** `upstream/main` v11.14.0+ (agent_id #1100 fases 1+2 JÁ MERGEADAS)
**Versão:** 0.4 (draft — §9 do @ducanhnguyen223 reconciliada; uso confirmado MULTI-WRITER)
**Inspiração:** Mnemosyne `sync.py` / `sync_server.py` (delta event-log + cripto client-side)
**Reintegra:** dor de sync via Insync (tasks internas 5d41dda2 stale-reads, corrupção SQLite+WAL)
**Relacionado:** #1304 (hybrid remote secondary — o caminho *hybrid-nativo* do mesmo destino; ver §7), #1100 (agent_id, base de R6), #57 upstream (federated retrieval, fora de escopo)
**Status:** DRAFT v0.4 — issue guarda-chuva aberta; implementação FORK-FIRST (o #1304 já está integrado e estável no fork; não espera merge upstream). Multi-writer confirmado (Zero×3 + T'Pol + Scotty escrevem com autoria).

---

## 1. Problema

O sync de memória entre máquinas e agentes (Zero em 3 máquinas; T'Pol/Scotty na VPS) é feito hoje por **file-sync do `.db` inteiro** via Insync/OneDrive + hot-backup com score composto. Isso transfere o arquivo todo, corre risco de corrupção e não modela "memória por agente".

### Evidências

- Banco de 457 MB sincronizado inteiro a cada operação (Insync) — custo escala com o tamanho.
- Corrupção documentada: "Insync + SQLite = corrupção — WAL ativo durante sync" (lessons learned).
- Stale reads: task interna 5d41dda2 — o processo mantém conexão aberta e não vê o `.db` atualizado por outra máquina.
- Modelo atual não é "memória por agente": é um banco copiado, não uma troca deliberada entre agentes (Zero não "sabe o que T'Pol fez" de forma seletiva — ou copia tudo, ou nada).

### 🔬 Experimento exploratório (19/set/2026, backups reais dos 3 hosts)

Comparados os `content_hash` dos bancos mais recentes de cada host:

| host | memórias | tamanho |
|------|---------:|--------:|
| DNBSCDC289 | 21.757 | 449 MB |
| sirdata | 22.258 | 474 MB |
| socrates | 21.454 | 371 MB |

- Comum aos 3: **20.865** · União: **22.446** · **Divergência: 1.581 (7,0%)**.
- Exclusivas: DNBSCDC289 **0**, sirdata 100, socrates 188.

**Achados que dimensionam a RFC:**
1. O full-file sync transfere **~432 MB por operação por host**, mas o delta real é
   **7%** (1.581 memórias) — ~93% de cada sync é redundante. Um delta-sync por
   content_hash transferiria só o que difere → economia de ordem de magnitude.
2. O sync converge razoavelmente (93% comum) mas DERIVA: socrates tem 188 memórias
   exclusivas (locais que não subiram), sirdata 100. Deriva é justamente o que um
   event-log delta corrige — hoje o full-file não reconcilia seletivamente.
3. DNBSCDC289 com 0 exclusivas sugere que ele só recebe (ou o backup é mais antigo,
   17/set vs 19/set dos outros) — confirmar antes de fixar o modelo de reconciliação.
4. Os 4 registros de `store` corrompido (bytes de controle) estão nos 3 hosts →
   propagaram via full-file sync; o delta por hash não os re-propagaria se limpos.

### Causas

1. **Sync de arquivo, não de deltas.** Transfere o `.db` inteiro; não há protocolo de mudanças incrementais.
2. **Sem event-log.** Não há registro append-only de operações que permita reconciliação e auditoria.
3. **Sem fronteira por agente.** Não há noção de "o que este agente compartilha" vs "o que é privado".

### Risco / motivação estratégica

Corrupção e stale reads são problemas ativos. E o modelo de arquivo impede a visão de **malha de memória da tripulação**: cada agente com sua memória, compartilhando seletivamente o que faz sentido (Zero vê a pesquisa da T'Pol e a implementação do Scotty).

---

## 2. Objetivo

Substituir o file-sync por um **sync delta baseado em event-log**, bidirecional, com reconciliação por timeline+importância, opcionalmente com criptografia client-side, modelando compartilhamento **por agente**.

**Não-objetivos:** forçar criptografia (opcional); remover o hot-backup (permanece como rede de segurança); federar retrieval entre stores (RFC #57 upstream separada).

---

## 3. Requisitos (Prosa + EARS)

> Convenção EARS (DEVELOPMENT-STANDARDS §8.4.1).

### Funcional

**R1**: O sync transfere apenas mudanças desde o último sync.

> EARS: WHEN a sync runs, THE sync client SHALL transfer only the events changed since the last successful sync.

**R2**: As operações são registradas em log append-only auditável.

> EARS: WHEN a memory is created, updated or deleted, THE system SHALL append an event to an immutable event log with id, timestamp and operation type.

**R3**: O sync é bidirecional com reconciliação determinística.

> EARS: WHEN two instances have divergent events, THE sync SHALL reconcile them by timeline and importance, deterministically.

**R4**: O sync não corrompe o banco durante operação.

> EARS: WHILE syncing, THE sync SHALL operate over the event log and never copy the raw `.db` file with an active WAL.

**R5**: O conteúdo pode ser criptografado no cliente (opcional).

> EARS: WHERE client-side encryption is enabled, THE sync SHALL encrypt payload before transmission so the server sees only metadata.

**R6**: O compartilhamento é escopado por agente, usando o `agent_id` que **já existe** no upstream (v11.14.0: metadata.agent_id + header X-Agent-ID via #1278/#1297, tag `agent:<id>` no web layer).

> EARS: WHEN an agent marks memories as shareable, THE sync SHALL propagate only those to peers, keeping the rest private. THE agent identity SHALL be read from `metadata.agent_id` (the merged #1100 Phase 1/2 field), not from a new mechanism.

### Não-Funcional

**R7**: O sync é resiliente a stale reads.

> EARS: WHEN the local `.db` changes on disk, THE service SHALL detect the change (e.g. `PRAGMA data_version`) and avoid serving stale data.

**R8**: O hot-backup permanece como rede de segurança.

> EARS: THE delta sync SHALL coexist with the existing hot-backup rather than replace it during transition.

---

## 4. Design

- `sync.py`: event-log append-only (id, ts, op, payload, agent_id) + cliente delta bidirecional.
- `sync_server.py`: servidor HTTP (na VPS) com auth (API key/JWT); vê só metadata quando cripto ativa.
- Cripto opcional: Fernet/PyNaCl, chave client-side (nunca sai da máquina).
- Escopo por agente: flag `shareable` + `agent_id` na memória; reconciliação timeline+importância.
- Detecção de stale: `PRAGMA data_version` no serviço para reconectar/recarregar.
- Coexistência: manter Insync/hot-backup durante transição; migrar quando o delta sync provar estabilidade.

---

## 5. Fora de Escopo

- Federated retrieval cross-store (RFC #57 upstream).
- Remoção imediata do Insync (coexistência na transição).
- Criptografia obrigatória (opcional; rede LAN + OneDrive já é privada para nós).
- **Identidade criptográfica forte por spoke (agent_id derivado de transporte autenticado) — FUTURO.** Decisão 09/out (Fase 5): a allow-list de autoria da Fase 4c (`MCP_SYNC_PUSH_ALLOWED_AGENTS`, §8.4/R4) cobre o cenário real (hub confiável + spokes conhecidos em rede confiável). Identidade criptográfica por spoke só se justifica com um spoke NÃO-confiável na malha — reavaliar quando/se isso ocorrer. A §8.4 "autoria derivada do transporte" fica como requisito de desenho, não implementada nesta fase.
- **Privacidade do baseline (gating por `shareable`) — FUTURO.** Decisão 09/out (Fase 5): a regra 3 de §9.3 ("itens unattributed/legacy ficam privados e não entram no baseline compartilhado sem a política `shareable` vigente autorizar") NÃO é implementada agora — o serviço não tem o mecanismo `shareable` em lugar nenhum (é design, não código). Gatear o baseline numa política inexistente seria meia-feature. O bootstrap da Fase 5 inclui todas as memórias; revisar junto quando a política `shareable` for construída.

---

## 6. Critérios de Aceite

- [ ] Sync transfere só deltas desde o último sync (não o `.db` inteiro).
- [ ] Event-log append-only auditável (id/ts/op/agent).
- [ ] Reconciliação bidirecional determinística por timeline+importância.
- [ ] Sync não copia `.db` com WAL ativo (sem corrupção).
- [ ] Compartilhamento escopado por agente (shareable vs privado).
- [ ] Serviço detecta mudança em disco e não serve dado stale.
- [ ] Coexiste com hot-backup na transição.

---

## 7. Relação com #1304 (hybrid remote secondary) — não bifurcar a superfície de sync

O Henry (no #1304) pediu que este framing fique **explícito**, para os dois esforços não divergirem:

- **#1304 = o caminho *hybrid-nativo*.** Torna o `secondary` do backend `hybrid` plugável (`cloudflare|http`), para que uma instância self-hosted seja o hub e cada cliente mantenha cache local. É **single-writer por cliente** contra um hub, drift detectado por `list_content_hashes()`. Resolve o hub-and-spoke de forma nativa no storage layer.
- **Esta RFC = o caminho *multi-writer*.** Event-log append-only com `agent_id`, reconciliação determinística, compartilhamento escopado por agente. Modela **N agentes escrevendo com autoria** (Zero/T'Pol/Scotty), não só N caches de um dono.

**Regra de convivência (acordada com o Henry):** o que **entrar primeiro** define a superfície; o que entrar **em segundo se dobra ao primeiro**. Como o #1304 é menor e já está sendo desenhado, é provável que ele entre antes — então esta RFC deve, quando implementada, **reusar** o que o #1304 estabelecer (o `remote_http` storage, o endpoint de bulk-hash, o model-match startup-check) em vez de criar um canal paralelo. O event-log é a camada de *autoria + reconciliação seletiva* por cima do transporte que o #1304 cria, não um transporte concorrente.

O ângulo de autoria se enuncia contra o `agent_id` **já mergeado** (#1100 fases 1+2), não contra a proposta original.

## 8. Invariantes de correção (fixtures de aceite)

Revisão do `@ducanhnguyen223` (2026-09-27): antes de implementar, quatro invariantes
ficam **explícitos como fixtures de aceite**, não apenas como prosa. Cada um vira
teste RED antes do código. Adotados na íntegra — refinam R2/R3 e adicionam garantias
que a v0.2 tratava informalmente ("timeline + importância" precisa de resolver
reprodutível).

### 8.1 Identidade de evento / idempotência (refina R2)

- Todo evento carrega uma origem estável `(agent_id, event_id)` com **constraint de
  unicidade**. Re-tentar um batch ou re-aplicar após crash **não pode** duplicar uma
  mutação de memória (aplicação idempotente por chave de origem).
- Deletes são **tombstones duráveis**, não remoção física, com regra de
  retenção/compactação definida (um tombstone só é coletável após todos os peers
  conhecidos terem passado do seu cursor).
- **Fixtures:** (a) aplicar o mesmo batch 2× → estado idêntico, zero duplicatas;
  (b) crash no meio do apply + replay → convergência sem duplicar; (c) delete
  seguido de replay de um create antigo do mesmo hash → o tombstone vence (não
  ressuscita).

### 8.2 Ordenação determinística (refina R3)

- Timestamps de wall-clock **não bastam** entre hosts. A ordenação usa um relógio
  **lógico/HLC** + tie-breaker estável (ex. `(hlc, agent_id, event_id)`).
- O resolver define explicitamente: update-vs-update de mesmo conteúdo,
  delete-vs-update, e chegada de evento atrasado (late event). "Timeline +
  importância" (R3) passa a ser um **resolver reproduzível** documentado, não uma
  prioridade informal — importância só desempata quando a ordem lógica empata.
- **Fixtures:** dois hosts com eventos concorrentes sobre o mesmo hash convergem ao
  **mesmo** estado independente da ordem de chegada; um late event que perde a
  ordem lógica não sobrescreve o vencedor.

### 8.3 Consistência de embedding (novo — fecha um furo da v0.2)

- Um evento de conteúdo aplicado e seu estado de embedding/índice devem ser
  observáveis como **um estado versionado único**, ou explicitamente marcados
  `embedding_pending`. Sem isso, um sync bem-sucedido pode tornar uma memória
  visível com um vetor **antigo**.
- O evento carrega um **hash de conteúdo/versão**; o apply só marca a memória como
  pesquisável quando o embedding corresponde à versão do conteúdo.
- **Fixtures:** replay de um update de conteúdo detecta embedding stale (vetor da
  versão anterior) e ou re-embedda ou expõe `embedding_pending` — nunca serve o par
  (conteúdo novo, vetor velho) como consistente.

### 8.4 Identidade e escopo de compartilhamento (refina R6)

- `agent_id` é derivado do **transporte/sessão autenticada**, e o `metadata.agent_id`
  é o **valor sincronizado**, não a autoridade. Um cliente não pode publicar em nome
  de outro agente reescrevendo metadata.
- **Fixtures:** (a) evento replayado mantém a autoria original; (b) agente A tentando
  publicar memória privada do agente B é rejeitado; (c) mudança de política de
  compartilhamento **depois** de um evento já enfileirado — definir se aplica ao
  backlog ou só a eventos novos (decisão: a política vigente no momento do **apply**
  no peer, não no enqueue, para não vazar retroativamente).

### 8.5 Cursor de sync como fonte de verdade (corrige R7)

Observação do revisor, adotada: `PRAGMA data_version` detecta staleness da **conexão
local**, mas **não prova** que o event-log de outro host foi totalmente observado. O
**cursor/ack de sync permanece a fonte de verdade** do "até onde já vi de cada peer".
R7 fica restrito ao seu escopo real (staleness de conexão local); a completude de
observação entre hosts é responsabilidade do cursor por-peer, não do `data_version`.

> EARS (R3'): WHEN divergent events are reconciled, THE resolver SHALL order them by
> a logical/HLC clock with a stable tie-breaker, using importance only to break a
> logical-order tie, deterministically and reproducibly.

---

## 9. Alternativas, bootstrap e limite do primeiro PR (v0.4)

### 9.1 Quando o event-log é necessário

O event-log não é requisito para todo cenário de sincronização. Para um cliente local com um único escritor e um hub terminal, #1304 (`remote_http` + reconciliação por `content_hash`) é a opção preferida: tem menos estados e não introduz resolução de conflitos entre escritores. Esta RFC só se justifica quando há múltiplos agentes que escrevem, é preciso preservar autoria/escopo e os peers precisam reconciliar mutações concorrentes e exclusões de modo repetível.

### 9.2 Alternativas consideradas

| Alternativa | Onde é suficiente | Por que não cobre, sozinha, o escopo multi-writer desta RFC |
| --- | --- | --- |
| #1304: `remote_http` + `list_content_hashes()` | Hub-and-spoke com um escritor por cliente; detectar drift e transferir memórias ausentes | Hashes mostram diferença de estado, mas não preservam a sequência/autoria das operações nem definem conflitos concorrentes, tombstones duráveis ou cursor de eventos. Se esse escopo bastar, usar #1304 e não implementar esta RFC. |
| CRDT SQLite existente, por exemplo [cr-sqlite](https://github.com/vlcn-io/cr-sqlite) | Quando a semântica de merge por coluna/tabela é aceitável e a extensão pode ser carregada/testada em todos os ambientes-alvo | CRDT resolve merges de estado segundo tipos de coluna; isso não equivale automaticamente às regras de autoria, política `shareable` e aceitação de eventos de §8. O README do projeto descreve a abordagem atual como history-free e marca o causal event log como trabalho futuro; não tratá-lo como uma implementação pronta para este contrato. |
| [Litestream](https://litestream.io/how-it-works/) | Backup contínuo e recuperação/replicação de SQLite, incluindo réplicas de leitura | É uma estratégia de réplica/recuperação do banco, não um protocolo de eventos multi-writer com identidade de agente, autorização de compartilhamento e resolução determinística entre peers. |
| Continuar sincronizando o arquivo completo, usando snapshot/backup SQLite consistente | Migração temporária ou recuperação de um único estado | Snapshot correto reduz risco de copiar arquivo ativo/WAL incorretamente, mas ainda transfere estado completo e não fornece autoria, tombstones/eventos nem confirmação por-peer. |

**Decisão:** para a topologia hub-and-spoke, implementar apenas #1304. Manter esta RFC em fase de desenho enquanto #1304 define a interface híbrida. Se a necessidade real não exigir escritores concorrentes e autoria, fechar ou reduzir esta RFC em vez de manter dois mecanismos.

### 9.3 Bootstrap, migração e compatibilidade de eventos

1. **Snapshot inicial:** criar um snapshot SQLite consistente e identificá-lo com um `snapshot_id`/watermark durável. O corte entre snapshot e log precisa ser atômico ou coberto por replay idempotente, para nenhuma mutação concorrente ficar fora de ambos.
2. **Estado legado:** inicializar a partir do estado atual, não fingir que existe histórico anterior ao bootstrap. Gerar eventos de baseline com IDs determinísticos a partir do snapshot e do ID/versão da memória; repetir o mesmo bootstrap deve produzir o mesmo conjunto lógico.
3. **Autoria e privacidade:** preservar `metadata.agent_id` quando presente, mas marcar como `legacy/unattributed` qualquer registro sem autoria verificável — nunca inventar um agente. Itens legados permanecem locais/privados por padrão; só entram no baseline compartilhado quando a política `shareable` vigente os autoriza.
4. **Deletes existentes:** converter `deleted_at` em tombstone de baseline preservando o momento de deleção disponível. Não remover fisicamente o tombstone até que a regra de retenção/compactação de §8.1 prove que todos os peers conhecidos o observaram.
5. **Peer novo:** instalar snapshot e seu watermark como uma unidade; depois consumir eventos a partir desse watermark. A instalação repetida deve ser idempotente. Não assumir que `content_hash` sozinho é suficiente para representar delete ou mudança de política.
6. **Versão incompatível:** cada envelope de evento deve declarar sua versão. Se um peer não entender um tipo/versão, deve rejeitar ou colocar o evento em quarentena, informar a incompatibilidade e **não avançar a confirmação/cursor além dele**. Não ignorar silenciosamente um evento desconhecido. Compatibilidade e negociação de versão devem ser definidas antes de existir transporte.
7. **Migração reversa:** restaurar hot-backup permanece o caminho de rollback. Depois que novos eventos forem emitidos, downgrade que não entende o log não pode continuar escrevendo como se o cursor estivesse sincronizado; exigir modo de manutenção/exportação ou migração explícita.

Esses passos são requisitos de desenho; volume, duração e limites práticos de bootstrap precisam de medição com a base real antes de alegar escala para 20.000+ memórias.

### 9.4 Escopo de backend

- **Fase inicial: `sqlite_vec` somente.** A primeira implementação deve gravar a mutação de memória e o evento na mesma transação SQLite, condição necessária para não criar memória sem evento nem evento sem memória. O inventário de mutações define explicitamente quais operações entram no primeiro slice; as demais não podem ser apresentadas aos peers como estado sincronizado.
- **`hybrid`: fora da fase inicial.** Esperar #1304. Antes de habilitá-lo, declarar qual storage é dono do log canônico e provar que sync local/secondary não gera evento duplicado nem reatribui a autoria. Até lá, uma instância `hybrid` não participa do event-log.
- **Cloudflare e Milvus: fora da fase inicial.** Não presumir atomicidade entre seus writes e uma tabela SQLite local; cada backend exige contrato de outbox/transaction, retries e tombstone próprio mais os fixtures de §8. Não alegar suporte por compartilhar a interface `MemoryStorage`.
- Expandir o escopo só com um backend nomeado, um caminho de mutação end-to-end e testes de falha/rollback para o backend correspondente. Backends não listados permanecem sem suporte explícito.

### 9.5 Primeiro PR proposto (depois da estabilização de #1304)

Um PR pequeno, sem transporte:

1. adicionar schema versionado do event-log no SQLite-vec e migração reversível;
2. após inventariar os pontos de escrita, anexar evento na mesma transação para cada mutação anunciada como suportada; deixar fora da primeira etapa e documentar qualquer operação que ainda não possa ser registrada atomicamente;
3. implementar os casos de teste de §8.1 para unicidade, repetição/replay e tombstone, incluindo rollback/crash na fronteira da transação;
4. não incluir endpoint de rede, sync bidirecional, resolução HLC, criptografia, backend hybrid/Cloudflare/Milvus ou remoção do hot-backup.

**Gate:** só abrir esse PR depois que #1304 tiver sido integrado e os contratos de storage/secondary estiverem estáveis; confirmar primeiro que o event-log local ainda agrega valor acima de #1304. Cada etapa posterior terá PR e casos de teste próprios. Este RFC não autoriza iniciar implementação antes desse gate.

> **Nota de reconciliação (mantenedor do fork, 2026-10-07):** o gate acima é TÉCNICO — exige que o #1304 esteja integrado e estável. **No fork, está:** Fases 1-4 no `main`, suíte verde, E2E provado em produção, contratos (`remote_http`, `list_content_hashes`, model-match) estáveis. Portanto a implementação segue FORK-FIRST, no compasso do mantenedor; o merge upstream do #1304 é o tempo do Henry e **não** bloqueia estas fases. Uso confirmado **multi-writer** (Zero×3 + T'Pol + Scotty com autoria), então o event-log agrega valor acima do #1304 — a confirmação que o gate pedia está dada.
