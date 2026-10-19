# spec-agent-id-streamable — autoria por header no transporte streamable-http

> Status: **PLANNED** (G2). Fork-only. Origem: `docs/_fork/hermes-primeira-classe.md` §5.
> Classe: **BUG** (corrupção silenciosa de autoria — sem fila de feature).
> Autor da spec: T'Pol (manager). Implementação: delegada. Review: independente, obrigatório.

## 1. Defeito, com evidência

O produto lê `X-Agent-ID` **apenas** em `src/mcp_memory_service/web/api/mcp.py:259-263`.
O serviço roda em streamable-http (`MCP_STREAMABLE_HTTP_MODE=1`) e
`src/mcp_memory_service/utils/startup_orchestrator.py:436` instancia
`StreamableHTTPSessionManager(app=server_instance, event_store=None, stateless=True)`
sem nenhum ponto que inspecione headers.

Prova medida (19/out/2026, dois registros consecutivos no mesmo servidor):

| registro | como foi gravado | `metadata.agent_id` |
|---|---|---|
| id 82179 | header `X-Agent-ID: tpol`, sem argumento | `zero` (env `MCP_AGENT_ID`) |
| id 82180 | argumento `agent_id="tpol"` | `tpol` |

Dívida no acervo no momento da medição: 21.168 memórias sem autor, 7.476 carimbadas `zero`,
51 `tpol`. A ponte em vigor (argumento explícito em cada chamada) depende da disciplina do
cliente — o que a própria RFC inter-agent-inbox já demonstrou ser não-confiável.

## 2. Requisitos (EARS)

- **R1** — WHEN uma requisição `tools/call` chega ao transporte streamable-http com o header
  `X-Agent-ID` e o argumento `agent_id` ausente, THE memory service SHALL atribuir a autoria
  ao valor do header.
- **R2** — THE memory service SHALL preservar a precedência **argumento explícito > header >
  `MCP_AGENT_ID` > ausente**, em todos os caminhos de escrita (`memory_store`,
  `memory_store_session`, `memory_consolidate`/harvest e auto-commit).
- **R3** — WHEN duas sessões concorrentes gravam com headers distintos, THE memory service
  SHALL NOT propagar a identidade de uma para a outra.
- **R4** — WHEN o header está ausente, THE comportamento SHALL ser byte-idêntico ao atual
  (guarda de regressão: `MCP_AGENT_ID` continua valendo).
- **R5** — THE memory service SHALL registrar em log nível DEBUG a origem da resolução
  (`argument`, `header`, `env`, `none`) uma vez por chamada, sem logar o valor do header
  em produção.

## 3. Desenho obrigatório (não improvisar)

1. **Middleware ASGI** envolvendo o `session_manager` em `run_streamable_http`: lê
   `X-Agent-ID` da requisição e grava em `ContextVar` (escopo por task/requisição).
2. **Ponto único de resolução**: `services/memory_service.py:438-443` passa a consultar
   `argumento → ContextVar → env`. Nenhuma tool duplica essa lógica.
3. **Reset explícito** do `ContextVar` no `finally` do middleware — sem isso, R3 falha por
   vazamento entre requisições no mesmo worker.
4. **Auth**: o header não concede permissão alguma; é atribuição, não autenticação. Declarar
   isso em docstring para o review não confundir os dois eixos.

Proibido: alterar o schema de `memories`, mexer no caminho `web/api/mcp.py` (já correto),
adicionar dependência nova.

## 4. Critérios de aceitação

| # | critério | verificação |
|---|---|---|
| CA1 | header sem argumento → `agent_id` correto em 3/3 tentativas | integração real |
| CA2 | 2 clientes concorrentes, 20 chamadas intercaladas → 0 cruzamento | integração real |
| CA3 | sem header → mesmo resultado do código atual | teste de regressão |
| CA4 | precedência argumento > header comprovada | integração |
| CA5 | suíte existente de `agent_id` verde, sem teste alterado | `pytest -k agent` |

## 5. Plano de testes (G3 RED antes da correção)

O teste tem de **falhar pelo motivo certo** em `main` antes da mudança:

- subir o serviço em porta efêmera com `MCP_AGENT_ID=zero`;
- dois clientes MCP streamable, headers `X-Agent-ID: alfa` e `beta`;
- `memory_store` sem argumento `agent_id`;
- ler de volta por `memory_search` e comparar `metadata.agent_id`.

Teste unitário com mock **não serve** — o defeito está no transporte, exatamente a camada que
o mock substitui.

## 6. Diário de gate

| gate | estado | evidência |
|---|---|---|
| G0 Think | feito (T'Pol) | assimetria provada por leitura + 2 registros reais |
| G1 Plan review | **pendente — Claudio** | esta spec |
| G2 Spec | este documento | EARS R1-R5, CA1-CA5 |
| G3 RED | a fazer | teste de integração falhando em `main` |
| G4 GREEN | a fazer | mínimo para os 5 CAs |
| G5 Review | a fazer | revisor independente (não quem implementou) |
| G6 Validação | a fazer | serviço vivo: 3 gravações + 2 clientes concorrentes |

## 7. Fora de escopo

- Migrar as 21.168 memórias sem autor (decisão separada: não há como inferir autoria).
- Autenticação/autorização por agente (eixo distinto, RFC própria).
- `X-Agent-ID` em outros transportes (stdio não tem header; SSE é outro caminho).
