# RFC: Hermes como cliente de primeira classe do memory-service

**Data:** 2026-10-19
**Autor:** Claudio + T'Pol
**Status:** DRAFT v0.1 — proposta, nenhuma decisão fechada além do modo híbrido
**RFC-mãe:** `rfc-memory-portability` (arco "Use in any agent") — esta é a filha que a mãe
deixou de fora deliberadamente: a mãe afirma "Hermes já usa o serviço (não é par a construir)".
O que se descobriu em 19/out é que **usar ≠ usar corretamente**, e a distância entre as duas
coisas é exatamente esta RFC.
**Código:** nenhum nesta RFC. As fatias de código vivem nas filhas (`rfc-agent-id-multi-agent`
para autoria; esta para o acoplamento de harness).

---

## 1. Problema

O Hermes já consome o memory-service por MCP (camada 1) e já tem skill de doutrina (camada 3).
Isso entrega *acesso*, não *integração*. As consequências medidas em 19/out:

- **Autoria corrompida**: `X-Agent-ID` era lido só na Web API; o serviço roda em streamable-http.
  Todo cliente que grava sem passar `agent_id` explícito cai no `MCP_AGENT_ID` do processo.
  No acervo: 21.168 memórias sem autor, 7.476 carimbadas como `zero`, 51 como `tpol`.
- **Histórico do próprio agente fora do acervo**: o importador do `state.db` está construído
  (22 testes verdes, dry-run em 4,86% do lido) e **nunca foi aplicado** ao acervo vivo.
  Medição de 19/out nesta máquina: 113 sessões, 105 elegíveis, 77 a gravar, 812.094 chars.
- **Deploy sem prova**: a propagação do fix de autoria (upstream #1598, mergeado 19/out 19:15Z)
  não é verificada em cada host — uma máquina pode rodar meses um código sem o fix e ninguém vê.
- **Protocolo sem instalador**: cada máquina foi montada à mão (kit + correções ad hoc). Não há
  caminho idempotente, nem `--uninstall`, nem verificação pós-escrita.

O objetivo desta RFC é fechar a distância entre "conectado" e "de primeira classe" para o
harness Hermes, com decisões explícitas e critérios verificáveis.

---

## 2. Modelo de convivência (D1) — DECIDIDO

**Híbrido**, por decisão do Claudio em 19/out:

| Horizonte | Fonte | Justificativa |
|---|---|---|
| Imediato (hoje/ontem) | MEMORY.md / USER.md do Hermes (2200 + 1375 chars, injetados todo turno) | custo zero de latência; o agente já nasce sabendo |
| Histórico (semana/mês, por tema) | memory-service via MCP | busca semântica, ranking, janela temporal, grafo |

Consequência direta: **provider (camada 2) fica desligado**. O modo híbrido é MCP + skill +
hooks seletivos. Ligar provider sobre MEMORY.md intacta produz injeção duplicada e gravação dupla
(`on_memory_write` espelha o que o tool `memory` já gravou).

Critério de aceitação: `hermes memory status` reporta `Provider: (none — built-in only)` e
`memory.provider: None` — e isso é o estado **correto**, não uma pendência.

---

## 3. Plugin provider (D2) — OPCIONAL, ADIADO

O plugin existe como possibilidade, não como necessidade. Se um dia for ligado (modo `tudo`),
ele mora em `$HERMES_HOME/plugins/mms/` com `plugin.yaml` + `__init__.py` + `README.md`,
e o contrato mínimo é `name`, `is_available`, `prefetch`, `sync_turn`, `get_tool_schemas`,
`handle_tool_call`.

Três formas de distribuir, em ordem de alcance: diretório local; pacote Python com entry point
`hermes_agent.memory_providers` (PyPI); catálogo curado (`hermes plugins pack` + `validate` +
`install`). O modelo "instalador que posiciona arquivos" (gabarito `codebase-memory-mcp`) é o
quarto caminho e o mais adequado a produto externo ao ecossistema.

**Decisão:** não implementar enquanto o modo for híbrido. Um provider ativo sem decisão sobre a
memória embutida é regressão, não avanço.

---

## 4. Diagnóstico e degradação (D3) — A IMPLEMENTAR

Regras obrigatórias para qualquer peça que fale com o serviço:

1. `is_available()` faz `GET /health` com timeout curto (≤2 s) e devolve `unavailable_reason`
   legível — nunca engole a causa.
2. Falha degrada, não quebra: o Hermes segue com memória embutida; o operador é informado.
3. **Nunca silêncio**: toda falha de integração sai no log com causa e caminho de correção.
   O delta-sync do próprio serviço já viola isso — falha a cada minuto no `journal` sem que
   nada no `memory status` ou no skill revele o problema.
4. Ferramenta de saúde dedicada (`mms-doctor`) com checks e reparo seguro:

| Check | Reparo automático |
|---|---|
| `/health` responde | reiniciar `memory-service` (systemd --user), uma vez |
| porta/endpoint efetivo igual ao configurado | reportar (não adivinhar) |
| versão instalada == versão do código em uso | reportar (reinstalar é decisão humana) |
| delta-sync com o hub | reportar (TLS/vhost é infra, não cliente) |
| autoria: última gravação tem `agent_id` esperado | reportar |
| embeddings órfãos / faltantes | disparar reparo (script existente) |

Critério: o doctor responde em ≤10 s, com exit code 0/1, e cada check cita a evidência lida
(arquivo:linha, HTTP status, versão).

---

## 5. Autoria (D4) — FECHADO NO UPSTREAM, PENDENTE DE PROPAGAÇÃO

Estado verificado em 19/out:

- PR **#1598** MERGED às 19:15Z; commit `d6d3fd47` "honour X-Agent-ID on the streamable-http transport"
  em `upstream/main`. Issue **#1100** CLOSED.
- `pr-fix-a` (`server_impl.py`, 4 chamadas) e `pr-fix-b` (`agent_context.py`, normalização e
  sanitização do log) integrados na mesma branch.
- **Nesta máquina (sirdata)** o serviço roda o fork local, 47 commits atrás de `upstream/main`,
  com o fix ausente; o venv reporta 11.15.0 enquanto o `pyproject` diz 12.0.0.

Regra que esta RFC fixa: **"mergeado no upstream" não é "em produção"**. Cada host declara a
versão que roda e a evidência (git log + versão do venv + teste de carimbo), e essa declaração
entra no `mms-doctor`.

Critério de aceitação: gravar com `X-Agent-ID: tpol` e **sem** argumento `agent_id` resulta em
`agent_id=tpol` em todos os hosts; gravar sem header e sem argumento cai no env de forma
documentada, não por acidente.

---

## 6. Colheita do histórico do agente (D5) — A EXECUTAR

O importador existe em `scripts/harvest/hermes_source.py` (23 KB), com `tests/test_hermes_state_db_importer.py`
(58 KB). CLI: `--db` (default `~/.hermes/state.db`), `--state` (cursor local), `--endpoint`
(default `http://127.0.0.1:3202/mcp`), `--apply` (sem ele é dry-run). Idempotente por cursor.

Dry-run medido em 19/out nesta máquina: **113 sessões · 105 elegíveis · 77 a gravar · 28 skip ·
8 descartadas · 812.094 chars**, com truncamento por orçamento declarado (`turns_dropped`).

Plano: rodar `--apply`, conferir a contagem no acervo (soma de `chars_to_send` não é critério —
o critério é o número de sessões com `action=new` presentes), e agendar em cron diário no host,
já que o `state.db` é local a cada máquina.

Critério: `.hermes/cache/mms-hermes-import.json` avança e uma segunda execução grava **zero**
(delta vazio), provando idempotência.

---

## 7. Hooks e superfície de interface (D6) — MEDIDO, CORRIGIDO

O achado "hooks de shell não rodam na TUI" estava **errado** para o Hermes 0.22.2.
`tui_gateway/server.py` chama `register_from_config(cfg)` imediatamente antes de instanciar
`AIAgent` (commit `3094cde513`, 07/set/2026). Prova: 4 linhas `shell hook registered` no
`agent.log` às 20:49:51 de uma sessão TUI.

O que a TUI **não** faz é auto-aprovar hook novo (`accept_hooks` fica no default `False`): hook
fora da allowlist é ignorado até aprovação no TTY, `HERMES_ACCEPT_HOOKS=1` ou
`hooks_auto_accept: true`.

Decisão: hooks de injeção valem para CLI e TUI igualmente; o instalador registra a aprovação de
forma explícita (nunca depende de prompt interativo em máquina de produção).

Critério: `hermes hooks doctor` com todos os hooks `allowlisted`, e uma prova de injeção real
(contexto anexado ao turno) em cada interface.

---

## 8. Instalador de primeira classe (D7) — A ENTREGAR

Artefato único, idempotente, com `--uninstall` cirúrgico. Ordem de execução (checklist do CDIA
`padroes/hermes-extension-surfaces.md`):

1. Resolver `$HERMES_HOME` (perfis vivem em `~/.hermes/profiles/<nome>/`).
2. Detectar Hermes (binário, versão) e serviço (`/health`) antes de configurar.
3. Escrever pela CLI (`hermes config set`, `hermes mcp add`), nunca YAML à mão.
4. Instalar a skill de doutrina.
5. Gravar o modo de convivência (`híbrido`) no config como decisão legível.
6. Verificar pós-escrita com busca real de token conhecido.
7. `/reload-mcp`; só pedir reinício se as tools não aparecerem.
8. Marcar tudo o que criou, para remoção cirúrgica.

Local e publicação: `scripts/hermes/` no repositório do serviço, distribuído com ele; opcionalmente
empacotado como plugin do catálogo. Não vai dentro de pacote de terceiro.

Critério: em máquina limpa, uma execução deixa o agente respondendo busca real; a segunda
execução não altera nada (diff zero); `--uninstall` remove exatamente o que foi criado.

---

## 9. Porta e transporte (D8) — DECISÃO ABERTA

Fato: o default do projeto é **8000** (`MCP_HTTP_PORT`, `config/transport.py:30`). A nossa é
**3202**, que veio do transporte SSE legado (`MCP_SSE_PORT=3202` no env por-host), quando
conviviam vários MCPs na mesma máquina — origem da "salada de portas".

Opções:

- **A. Manter 3202** (recomendada): já está em 3 hosts, no `kit-dtp`, nos scripts e nos configs
  de cliente. Migrar não entrega função nova e amplia o blast radius.
- **B. Migrar para 8000**: alinha com o default e reduz surpresa para quem lê a doc do projeto;
  custa atualizar env, units, kit, skills e configs de todos os clientes.

Em qualquer caso: a porta efetiva passa a ser declarada em UM lugar (env por-host), lida pelo
instalador e pelo `mms-doctor`, e a doc do harness deve dizer explicitamente "3202 por escolha,
8000 é o default do projeto".

---

## 10. Critérios de aceitação do arco

1. `hermes memory status` = built-in only, e o skill explica por que isso é o estado correto.
2. Busca por tema de semana passada devolve a decisão com data, em PT e em EN.
3. Última gravação de cada host carrega o `agent_id` do agente que gravou, provado com sonda.
4. Segunda execução do importador grava zero; primeira grava o planejado.
5. `hermes hooks doctor` verde nas duas interfaces, com prova de injeção.
6. Instalador idempotente e reversível, validado em máquina limpa.
7. `mms-doctor` em ≤10 s, com exit code e evidência por check.

## 11. Fora de escopo

- Provider ativo (modo `tudo`) e redução da MEMORY.md.
- Migração de portas em produção.
- Autoria por `_meta` por chamada (evolução posterior à #1100).
- Adapters de outros harnesses (pertencem à RFC-mãe de portabilidade).
