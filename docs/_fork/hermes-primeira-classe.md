# Hermes como cliente de 1ª classe do mms

> **Doc de trabalho (fork-only).** Não é RFC de feat nova: é o mapa do que falta para o
> Hermes deixar de ser cliente de 2ª classe. A fonte viva por harness é a wiki
> `Memory-Portability-Map`; as decisões públicas acontecem na discussion
> [#1364](https://github.com/doobidoo/mcp-memory-service/discussions/1364).
> Inventário medido em 19/out/2026 na DNBSCDC289. Nada aqui é estimativa de cabeça:
> cada linha tem evidência.

## 1. Por que este doc existe

O mms nasceu e amadureceu dentro do Kiro. O Kiro é o espelho de referência: o serviço foi
desenhado contra ele e validado nele. O Hermes é cliente novo, e a diferença aparece em
furos que só o uso real revela — autoria de memória, ciclo de vida, preferência de store.

Consequência prática: "conecta por MCP e funciona" é o piso, não o teto. Um harness pode
estar verde nas cinco camadas da wiki e ainda gravar memórias anônimas.

## 2. Duas numerações, uma só verdade

A wiki usa **5 camadas** (aceitas pelo mantenedor no #1364). A numeração C1–C7 é interna
nossa e sobrevive só como apelido — quando divergirem, a wiki manda.

| Camada da wiki | Cobre | Nossa numeração |
|---|---|---|
| 1. Connectivity | alcançar o store (MCP/REST) | C1 |
| 2. Connector / plugin | instalável que pluga o store no harness | C7 |
| 3. Lifecycle hooks | session start / capture / injection / end | C4, C6 |
| 4. Preference override | o harness PREFERE este store ao nativo | C5 |
| 5. Usage toolkit | skill/manual de uso correto | C2 |
| — (transversal) | **autoria**: quem escreveu a memória | C3 |

Identidade não cabe em nenhuma das cinco: atravessa 1, 3 e 4.

## 3. Estado medido do Hermes (19/out/2026)

| Camada | Estado | Evidência | Falta |
|---|---|---|---|
| 1. Connectivity | ✅ | `http://127.0.0.1:8000/mcp`, `MCP_STREAMABLE_HTTP_MODE=1`, `/health` ok | — (porta unificada em 8000 em 21/out, nas 3 máquinas) |
| 2. Connector | 🔴 | nada no repo; o kit de viagem é manual e vive fora do mms | instalador com `uninstall` idempotente |
| 3. Lifecycle hooks | 🟡 | 5 hooks shell: contexto, inbox, replay de erro, carimbo, **checkpoint periódico** | capture/injection de memória por turno |
| 4. Preference override | 🔴 | Hermes segue gravando no `state.db` nativo em paralelo | provider nativo (`plugins/memory/` tem 8, nenhum é o mms) |
| 5. Usage toolkit | 🟡 | 3 skills locais (`memoria-mcp-primario`, `mms-client-manager`, `memoria-atomica-checkpoint`) | `agents/hermes/README.md` mergeado no upstream (#1618, 21/out); skills seguem fora do repo |
| transversal: autoria | 🔴 | ver §5 | injeção de `agent_id` no transporte |

### C6 — colheita do `state.db`: implementada, não aplicada

Correção de inventário: em 19/out eu havia registrado C6 como inexistente e **estava
errado** — medi antes do merge de 12:31, que trouxe a feat do sirdata.

- `scripts/harvest/hermes_source.py` (642 linhas) · spec `docs/rfc/planned/spec-hermes-state-db-importer.md` · `tests/test_hermes_state_db_importer.py` (1.323 linhas)
- Status real: **IMPL-FORK (G6)**, 22 testes verdes, G5 aprovado com ressalvas
- Dry-run medido no acervo real: 107 sessões, 97 elegíveis, 1.009.927 chars = **4,86% do lido** (critério CA3 ≤10%)
- **`--apply` nunca foi executado no acervo vivo.** É o próximo passo óbvio da camada 3.
- Ressalvas abertas: DB inexistente gera traceback; `to_store` no relatório de falha é plano, não realizado.

## 4. Entregue em 19/out/2026

1. **Checkpoint periódico mecânico** — `~/.hermes/agent-hooks/checkpoint_periodic.py`
   em `pre_llm_call`. A cada 15 turnos **ou** 45 min: grava snapshot volátil no mms em
   processo destacado (~24 s, não bloqueia o turno) e injeta aviso pedindo registros
   atômicos. Limiares por env (`HERMES_CKPT_TURNS`, `HERMES_CKPT_MINUTES`). Fail-open.
   Preenche a lacuna de capture periódico da camada 3.
2. **Skill `memoria-atomica-checkpoint`** — doutrina de particionamento (1 assunto =
   1 registro, checkpoint ≤500 chars só com estado volátil) + pitfalls medidos.
   Camada 5.
3. **Skin por sessão** — `~/.local/bin/hermes-sess <1-5>` e widget de TUI
   (`tui-widgets/sessao.mjs`). Auxiliar de operação, **não** é peça de UIAA.

## 5. O furo mais grave: autoria

`X-Agent-ID` (#1100 Fase 2 / PR #1297) é lido **só** em `web/api/mcp.py`. O serviço roda
em streamable-http e `startup_orchestrator.run_streamable_http` (L416) instancia o
`StreamableHTTPSessionManager` sem inspecionar headers.

Prova medida (dois registros consecutivos, mesmo servidor):

| Registro | Como | `agent_id` gravado |
|---|---|---|
| id 82179 | header `X-Agent-ID: tpol`, sem argumento | `zero` |
| id 82180 | argumento `agent_id` explícito | `tpol` |

Divida no acervo: 21.168 memórias sem autor, 7.476 carimbadas `zero`, 51 `tpol`.

**Patch proposto:** injetar `agent_id` antes do `StreamableHTTPSessionManager`, espelhando
as quatro linhas de `web/api/mcp.py` L259–263. Não implementado.

**Ponte em vigor:** `agent_id="tpol"` explícito em cada `memory_store`.

## 6. Critérios de aceitação

| Camada | Critério verificável |
|---|---|
| 1 | `/health` ok em 8000 (migrada de 3202 em 21/out, nas 3 máquinas) |
| 2 | instalador escreve config + skill e `uninstall` remove o que criou, verificado por execução |
| 3 | `--apply` do importador roda com `4,86%` ou menos do lido, sem duplicata em re-run; capture por turno grava ≤500 chars |
| 4 | provider nativo listado em `plugins/memory/`; Hermes deixa de escrever no `state.db` como fonte primária |
| 5 | ✅ `agents/hermes/README.md` publicado no upstream (#1618, mergeado 21/out) · corrigido no #1619 |
| autoria | gravação com header, sem argumento, resulta em `agent_id` correto em 100% de 3 tentativas |

## 7. Próximos passos, na ordem

1. Rodar `--apply` do importador do `state.db` no acervo vivo (fecha a camada 3 com número).
2. Patch de autoria no transporte streamable (fecha corrupção silenciosa).
3. Atualizar a wiki: porta 3202, camada 5 com as skills existentes, eixo de identidade.
4. Provider nativo (camada 4) — o que de fato muda o patamar.
5. Instalador (camada 2), absorvendo o kit de viagem.

## 8. Fora de escopo

**BYM (bring your memory)** — importar de sistemas concorrentes (mem0, letta, zep, honcho…) —
é outra frente, com RFC própria (`rfc-importers`) e histórico upstream próprio (#1390, PR #1401).
Não se mistura com UIAA.
