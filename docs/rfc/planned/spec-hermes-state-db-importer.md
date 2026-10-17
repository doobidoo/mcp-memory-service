# SPEC — Importador do state.db do Hermes Agent (bring your memory)

**Deriva de:** issue #1095 (Hermes state.db data source adapter) + `rfc-memory-portability` §4.2 ("bring")
**Data:** 2026-10-17 · **Status:** DRAFT G2 (aprovada a direção no G1; esta spec aguarda G3 RED)
**Autor:** T'Pol (Hermes) · **Classe:** FEATURE (fila serial)
**Escopo:** 1 incremento. Leitor read-only do `state.db` do Hermes + envio ao acervo via tool MCP. Sem mudança de contrato no servidor.

---

## 1. Problema (prosa)

O Hermes Agent guarda todo o histórico de sessões em `~/.hermes/state.db` (SQLite, FTS5).
Esse material é hoje sub-colhido: o acervo do memory-service não enxerga nada dele, e o
workaround existente é um script manual frágil, dependente de detalhes internos.

O objetivo é tornar a ingestão repetível, incremental e auditável: cada sessão do Hermes
vira memória no acervo, com proveniência explícita (`source:hermes`), identidade do agente
(`agent_id` vindo do profile) e sem arrastar o lixo que domina o volume — sobretudo saída
bruta de ferramentas.

### Evidências medidas (17/out, banco real desta máquina)

| Métrica | Valor |
|---|---|
| Sessões | 98 |
| Mensagens ativas | 13.515 (0 compactadas) |
| Volume de `tool` | 17.978.934 chars (89% do total) |
| Volume de `assistant` | 1.558.679 chars |
| Volume de `user` | 397.770 chars |
| `git_repo_root` preenchido | 0 de 98 |
| `cwd` preenchido | 32 de 98 |

Conclusão que orienta os requisitos: ingerir o payload de `tool` produziria ~18 MB de ruído.
A ingestão útil é `user` + `assistant` + metadados de `tool`.

### Achado de runtime (contradiz o plano inicial)

O G0 mapeou `POST /api/sessions` como caminho de menor risco. A verificação de runtime
mostrou que a Web API **não está servida** nesta instalação:

```
/api/health   -> 404
/api/sessions -> 404
/api/memories -> 404
única porta em escuta: 127.0.0.1:3202 (memory-server, MCP_STREAMABLE_HTTP_MODE=1)
```

Portanto o transporte desta spec é a **tool MCP `memory_store_session` via JSON-RPC**
(`POST http://127.0.0.1:3202/mcp`, header `Accept: application/json, text/event-stream`).
A Web API fica como caminho alternativo caso seja habilitada no futuro (não é pré-requisito).

---

## 2. Objetivo e não-objetivos

**Objetivo:** um comando no repo do memory-service que importe sessões do Hermes de forma
incremental, idempotente e com relatório verificável, sem alterar o servidor.

**Não-objetivos:**
- destilar/sintetizar as sessões importadas (isso é o arco de síntese de belief);
- importar memória embutida do Hermes (`MEMORY.md`/`USER.md`) — incremento separado;
- criar endpoint novo (`/api/harvest/hermes`);
- alterar `memory_store_session` ou o schema.

---

## 3. Requisitos (prosa + EARS)

**R1 — Leitura estritamente read-only.**
O importador abre o `state.db` em modo somente-leitura e nunca escreve no arquivo do Hermes.

> EARS: The importer SHALL open the Hermes state database with `mode=ro` URI and SHALL NOT issue any statement that mutates it.

**R2 — Seleção de conteúdo útil.**
Somente mensagens `active=1` e `compacted=0`; de mensagens `role='tool'` entram apenas
metadados (nome da ferramenta, indicativo de sucesso/erro e primeira linha truncada),
nunca o payload completo. `role='session_meta'` é ignorado.

> EARS: WHEN a message row has role='tool' THE importer SHALL emit metadata-only (tool name, ok/error signal, first line up to N chars) and SHALL NOT emit the full content.
> EARS: WHEN a message row has compacted=1 or active=0 THE importer SHALL exclude it.

**R3 — Identidade e proveniência.**
Cada memória gravada leva `agent_id` (de `sessions.profile_name`, com fallback `hermes`) e
tag `source:hermes`; a tag `session:<id>` é adicionada pela própria tool.

> EARS: WHEN storing any imported memory THE importer SHALL set metadata.agent_id (profile_name when present, else 'hermes') and SHALL include the tag source:hermes.

**R4 — Partição de destino.**
As memórias vão para `store='default'`. Projetos são sinalizados por tag derivada de `cwd`
(slug do último componente), não por store — porque busca sem `store` só alcança `default`
(verificado: busca sem filtro retorna apenas o default).

> EARS: The importer SHALL store into 'default' and SHALL tag the project as project:<slug> when cwd is present.

**R5 — Filtro de qualidade na entrada.**
Sessão com menos de 2 mensagens úteis (user+assistant) é descartada e reportada, não
convertida em memória.

> EARS: WHEN a session has fewer than 2 useful messages THE importer SHALL skip it and SHALL report it as discarded with reason 'insufficient_content'.

**R6 — Idempotência e incremento.**
O importador mantém estado local por sessão (`session_id`, contagem de mensagens úteis, hash
do último conteúdo útil). Reexecutar sem novidade não grava nada; sessão que cresceu importa
apenas o delta, ou é reenviada integralmente com `version` incrementada (decisão aberta D2).

> EARS: WHEN the importer runs twice with no new messages THE second run SHALL store zero new memories.

**R7 — Modo dry-run como padrão.**
Sem `--apply`, o importador não grava: produz relatório com sessões elegíveis, memórias a
gerar, caracteres a enviar, descartes por motivo e store/tags por sessão.

> EARS: WHEN invoked without --apply THE importer SHALL emit the report and SHALL issue zero store calls.

**R8 — Relatório auditável.**
O relatório é JSON estável (campos fixos) e um resumo legível, permitindo comparar execuções.

> EARS: The importer SHALL emit a JSON report containing totals per role, per discard reason, and per session destination.

---

## 4. Design

### 4.1 Fluxo

```
state.db (ro) ──► seleção ──► normalização ──► transporte ──► acervo
                   R2/R5       R3/R4          MCP tool        tags+metadata
                                      │
                                      └──► relatório (R7/R8)
                                      └──► estado local (R6)
```

### 4.2 Fonte

- `sessions`: `id, profile_name, cwd, git_repo_root, git_branch, title, started_at,
  message_count, model, source`
- `messages`: `session_id, role, content, tool_name, active, compacted, timestamp`
- Conexão: `sqlite3.connect("file:<path>?mode=ro", uri=True)`; tolerar `-wal`/`-shm`.

### 4.3 Normalização de turnos

- `user`/`assistant` → `{role, content}` com content truncado em `MAX_TURN_CHARS` (default 4000).
- `tool` → `{role: 'tool', content: "[tool:<nome>] <ok|erro> — <primeira linha truncada>"}`.
- Limite de turnos por sessão via head+tail (padrão já usado em harvest): mantém início e fim.

### 4.4 Transporte

`POST http://127.0.0.1:3202/mcp`, JSON-RPC 2.0, `tools/call` de `memory_store_session` com
`{turns, session_id, tags, metadata, store}`. Header `Accept: application/json, text/event-stream`
obrigatório (resposta vem em SSE). Token opcional via `MCP_API_KEY` (ausente nesta instalação).

### 4.5 Estado local

`~/.hermes/cache/mms-hermes-import.json`: `{session_id: {useful_msgs, last_hash, imported_at}}`.

---

## 5. Casos de erro e bordas

| Caso | Comportamento |
|---|---|
| state.db ausente ou lock | falha explícita, exit ≠ 0, sem escrita |
| schema inesperado (coluna faltando) | erro claro citando a coluna; não assumir |
| sessão vazia / só `session_meta` | descartada (`insufficient_content`) |
| conteúdo enorme em uma mensagem | truncar em `MAX_TURN_CHARS` e registrar |
| serviço fora do ar | nenhuma gravação parcial do lote; reportar e parar |
| sessão em execução (crescendo) | importar snapshot e registrar hash; próximo run pega delta |
| `-wal` com dados recentes | abrir em ro inclui WAL; documentar no relatório |

---

## 6. Critérios de aceitação (G4 check)

| # | Critério | Verificação |
|---|---|---|
| CA1 | `--dry-run` não grava | spy/mock: 0 chamadas a store |
| CA2 | Duas execuções seguidas ⇒ 0 novas na segunda | contagem de memórias antes/depois |
| CA3 | Volume enviado ≤ 10% do lido | soma de chars no relatório (17,9M → ≤1,8M) |
| CA4 | 0 memórias de mensagem compactada | auditoria por amostragem + teste unitário |
| CA5 | 100% das gravadas têm `agent_id` e tag `source:hermes` | consulta ao acervo |
| CA6 | Relatório discrimina descartes por motivo | estrutura do JSON |
| CA7 | state.db inalterado após execução | hash sha256 antes/depois |

---

## 7. Testes RED previstos (G3, delegado a rok)

1. `test_tool_payload_excluded` — fixture com tool content gigante; saída não contém payload.
2. `test_compactada_excluida` — linha `compacted=1` não aparece.
3. `test_agent_id_from_profile` — `profile_name='zero'` ⇒ `metadata.agent_id='zero'`.
4. `test_idempotencia` — segunda execução com mesmo estado grava 0.
5. `test_dry_run_nao_grava` — transporte injetado como espião; 0 chamadas.
6. `test_readonly_db` — hash do arquivo idêntico antes/depois.
7. `test_sessao_insuficiente_descartada` — motivo correto no relatório.
8. `test_store_default_e_tag_project` — tags/marcação conferidas.

Red-on-main: os testes 1, 2, 4, 5 e 6 falham sem a implementação, e falham por comportamento
(ausência de filtro/idempotência), não por ImportError.

---

## 8. Riscos e decisões abertas

- **D1 — Turnos de tool:** resumir (proposto) vs omitir completamente. Resumo preserva a
  narrativa de execução; omitir reduz ainda mais o volume. Recomendação: resumir.
- **D2 — Sessão que cresce:** reenviar integral com `version++` vs enviar só o delta em nova
  memória. Recomendação: delta em memória nova com tag `session-delta`, para não perder histórico.
- **D3 — Importar `MEMORY.md`/`USER.md`:** incremento separado; são 3.632 chars de altíssima
  densidade e provavelmente merecem tratamento próprio (não como sessão).
- **D4 — Web API:** se for habilitada, `POST /api/sessions` passa a ser o transporte preferido
  (menos overhead que JSON-RPC/SSE). Não bloqueia este incremento.
- **Risco R-1:** o schema do state.db é interno do Hermes e pode mudar entre versões.
  Mitigação: detectar colunas no início e falhar explicitamente, nunca adivinhar.

---

## 9. Definição de pronto (DoD)

1. `python scripts/harvest/hermes_source.py --dry-run` produz relatório estável.
2. `--apply` grava com `agent_id` e `source:hermes`, respeitando CA1-CA7.
3. Testes RED→GREEN no CI (pytest, cobertura ≥70% no arquivo novo).
4. Linha no `LEDGER-feats.md` (estado IMPL-FORK) e esta spec marcada IMPLEMENTED ao final.
5. Nenhuma alteração em `src/` nesta primeira entrega.
