# SPEC — D3: `mms-doctor` — diagnóstico, degradação e prova por host

**Status:** PLANNED (G1) · **Arco:** UIAA/hermes (D3) · **Autor:** T'Pol
**RFC-mãe:** `rfc-hermes-first-class.md` §4 · **Pesquisa:** `hermes-primeira-classe.md` §1
**Classe:** instrumento de verificação (pré-requisito do instalador)

## Problema

Hoje "funcional" não é verificável. Três fatos medidos:

1. **Mergeado no upstream ≠ em produção.** O fix de autoria (#1598, merge `d6d3fd47`) está no
   upstream desde 19/out; o sirdata roda o fork **47 commits atrás**, sem ele. Nada denuncia isso.
2. **Falha silenciosa.** O delta-sync falha a cada minuto no `journal`, e nem `memory status` nem a
   skill revelam o problema.
3. **Ausência de instrumento.** A verificação de hoje é artesanal: `curl /health`, `git log`,
   comparar versão do venv com `pyproject`. Feito à mão, não é feito.

## Requisitos (EARS)

- **R1** — THE `mms-doctor` SHALL executar os six checks abaixo em **≤10 s** e sair com `0` (tudo ok)
  ou `1` (ao menos uma falha).
- **R2** — WHEN um check falhar, THE ferramenta SHALL imprimir a **evidência lida** (HTTP status,
  `arquivo:linha`, versão lida, valor esperado), nunca só "falhou".
- **R3** — IF o check de `/health` falhar, THEN SHALL tentar **um** restart do serviço
  (`systemctl --user restart memory-service`) e re-testar uma vez; SHALL NOT reiniciar mais de uma vez.
- **R3-bis (achado 20/out — incidente real)** — THE reparo SHALL NOT reiniciar um serviço que esteja
  em `activating` (start em andamento): o serviço leva ~30 s para ficar pronto ("Dependencies missing,
  extending base timeout to 30.0s"), e reiniciar durante o start impede que ele complete. Cada
  execução do doctor somava um restart, o systemd entrou em `start-limit-hit` e **o serviço ficou
  fora do ar**. Condições obrigatórias antes de reiniciar: (a) estado da unit é `failed`, não
  `activating`; (b) após o restart, aguardar readiness com backoff (até o timeout declarado) antes
  de declarar falha; (c) nunca disparar restart se o motivo for porta errada — nesse caso o
  diagnóstico é de configuração, não de processo.
- **R3-ter** — THE ferramenta SHALL registrar no relatório que executou um restart, com o estado da
  unit antes e depois, para que o operador saiba que houve intervenção.
- **R4** — THE ferramenta SHALL NOT alterar versão, porta, configuração de sync ou dados do acervo.
  Correção automática existe **apenas** para o check de `/health`.

### Os seis checks

| # | Check | Como mede | Reparo |
|---|---|---|---|
| 1 | `/health` responde | `GET /health` com timeout ≤2 s | restart único (R3) |
| 2 | porta efetiva == declarada | compara a porta da fonte única (D8) com a que responde | reportar |
| 3 | versão instalada == código em uso | `import mcp_memory_service; __version__` vs `pyproject.toml` do repo em execução | reportar |
| 4 | delta-sync com o hub | última linha de `sync_cursor`/`sync_events` + erro recente no `journal` da unit | reportar |
| 5 | autoria da última gravação | última memória: `metadata.agent_id` preenchido e ≠ valor do env quando houve header | reportar |
| 6 | embeddings órfãos/faltantes | `memory_embeddings_rowids` (tabela real) contra `memories` vivas | disparar reparo existente |

> **Achado do G0 (20/out):** `memory_embeddings` é `vec0` virtual e o `sqlite3` do sistema **não
> carrega o módulo** — `select count(*) from memory_embeddings` falha com "no such module: vec0".
> O check precisa ler `memory_embeddings_rowids` (tabela normal, 29.145 linhas neste host contra
> 29.139 memórias vivas), senão o próprio doctor quebra em máquina limpa.

## Critérios de aceitação

- **CA1** — Em host saudável, sai `0` e imprime os seis checks com evidência, em ≤10 s (medido).
- **CA2** — Com o serviço parado, o check 1 falha, **um** restart é tentado, o resultado é re-testado
  e a saída cita o status HTTP obtido antes e depois.
- **CA3** — Com versão divergente (venv ≠ `pyproject`), o check 3 falha citando as duas versões.
- **CA4** — Com `sync_cursor` parado e erro no journal, o check 4 falha citando a data do último
  cursor e a linha do erro.
- **CA5** — Nenhum check altera config, dados ou versão (verificável: `git status` do repo limpo e
  hash do banco inalterado antes/depois, exceto o reparo do check 6 quando invocado explicitamente).
- **CA6** — Saída legível em terminal de 80 colunas, sem stack trace no caminho de falha esperada.

## Gate

G0 (seven: enumerar as fontes de cada evidência) → G1 (esta spec) → G3 RED (fixture por falha
simulada: health off, versão divergente, cursor parado, embedding faltante) → G4 GREEN →
G5 (tuvok: confirmar que nenhum check "passa por construção") → G6 (rodar no host real e colar saída).

## Fora de escopo

- Reparar TLS/vhost do hub (é infraestrutura).
- Reinstalar versão (decisão humana).
- Substituir o instalador (D7).
