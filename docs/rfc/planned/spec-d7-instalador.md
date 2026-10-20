# SPEC — D7: instalador de primeira classe (um comando, idempotente, reversível)

**Status:** PLANNED (G1) · **Arco:** UIAA/hermes (D7) · **Autor:** T'Pol
**RFC-mãe:** `rfc-hermes-first-class.md` §8 · **Pesquisa:** `hermes-primeira-classe.md` §1 camada 2
**Classe:** entregável de produto — é a meta do arco

## Meta

**Um comando** instala o memory-service, integra-o a um Hermes existente e deixa o agente pronto para
uso — sem edição manual de YAML, sem passo esquecido, sem estado pela metade. Aceitação final: em um
**Hermes limpo com banco zerado**, uma execução deixa a busca funcionando de ponta a ponta.

Público: colegas que já usam Hermes e não conhecem o mms.

## Problema

A montagem atual é artesanal e não reprodutível: cada máquina recebeu o "kit de viagem" manualmente,
mais correções ad hoc. Não existe caminho idempotente, nem `--uninstall`, nem verificação pós-escrita.
Consequência medida: três hosts com estados diferentes — um deles 47 commits atrás, sem o fix de
autoria, e ninguém sabia.

## Requisitos (EARS)

- **R1** — WHEN executado, THE instalador SHALL resolver `$HERMES_HOME` (respeitando perfis em
  `~/.hermes/profiles/<nome>/`) antes de escrever qualquer coisa, e SHALL recusar-se a agir se não
  encontrar um Hermes instalado.
- **R2** — THE instalador SHALL verificar o serviço (`GET /health`) **antes** de configurar o cliente;
  IF o serviço não estiver no ar, THEN SHALL orientar como subir e encerrar sem deixar config parcial.
- **R3** — THE instalador SHALL escrever pela CLI oficial (`hermes mcp add --url …`, `hermes config
  set …`), SHALL NOT editar `config.yaml` à mão.
- **R4** — THE instalador SHALL instalar a skill de doutrina e SHALL registrar a decisão de
  convivência (`híbrido`) no config, de forma legível para o operador.
- **R5** — WHEN a instalação terminar, THE instalador SHALL verificar **pós-escrita** com uma busca
  real de um token conhecido, e SHALL falhar se a busca não devolver resultado.
- **R6** — WHEN executado uma segunda vez em ambiente já instalado, THE instalador SHALL produzir
  **diff zero** (idempotência provada por hash da configuração antes/depois).
- **R7** — THE instalador SHALL suportar `--uninstall` que remove **exatamente** o que ele criou,
  sem tocar em configuração pré-existente do usuário nem no banco de dados.
- **R8** — IF o banco estiver zerado, THEN o serviço SHALL inicializar o schema sozinho e a busca
  pós-instalação SHALL funcionar após a primeira gravação de teste.
- **R9** — THE instalador SHALL marcar todo artefato criado (arquivo de manifesto), para que a
  remoção seja cirúrgica e auditável.

## Critérios de aceitação

- **CA1** — Em `HERMES_HOME` temporário e **banco novo**: uma execução → `hermes mcp test` OK → busca
  real devolve a memória de teste. Sem intervenção manual.
- **CA2** — Segunda execução: hash de `config.yaml` e da lista de MCPs idênticos ao estado anterior;
  saída declara "nada a fazer".
- **CA3** — `--uninstall` remove MCP, skill e hooks criados; `config.yaml` volta ao estado anterior,
  exceto chaves que o usuário já tinha; banco intacto (contagem de memórias inalterada).
- **CA4** — Com o serviço parado, a instalação aborta antes de escrever config (nenhum artefato
  criado; verificado por manifesto ausente).
- **CA5** — Nenhum YAML editado à mão: a prova é que todo `config.yaml` foi alterado por `hermes
  config set` (auditável pelo log de comandos do próprio instalador).
- **CA6** — Relatório final cita: porta usada, versão instalada, caminho da skill, resultado da busca
  de verificação.

## Gate

G0 (seven: mapear superfícies que o instalador toca e o que já existe em `scripts/hermes/`) →
G1 (esta spec) → G3 RED (testes de idempotência, aborto sem serviço, uninstall cirúrgico, banco
zerado) → G4 GREEN → G5 (tuvok) → **G6 = E2E em Hermes limpo com banco zerado**, mediado.

## Fora de escopo

- **Atualização de versão ("upgrade").** O projeto não tem esse verbo e esta spec não cria um. O que
  existe é a operação documentada `pip install --upgrade mcp-memory-service==X` + restart do serviço.
  Reexecutar o instalador em ambiente já instalado apenas confirma o estado (R6); não troca versão.
- Provider de memória nativo (D2, adiado por decisão).
- Migração de portas (D8).
- Instalação do próprio Hermes (o alvo já tem Hermes).
- Pacote PyPI nesta entrega; `scripts/hermes/` no repo do serviço primeiro.

## Nota de escopo (20/out)

`scripts/hermes/` **não existe hoje** — verificado. Este item nasce do zero, e é por isso que o D7 é
o de maior esforço dos três: não é ajuste de algo existente, é o primeiro artefato de instalação
reprodutível do projeto.

### Achado do G0 (20/out): existe um protótipo, e ele cobre só metade

`~/dtp/ai-configs/hermes/kit-dtp/` já traz `instalar.sh` (72 linhas) e `verificar.sh` (113 linhas).
O `instalar.sh` é idempotente e faz backup, mas **instala apenas o lado cliente**: copia payload
(config, skills, hooks) para `$HERMES_HOME`, corrige caminhos absolutos e marca hooks executáveis.
Ele **não** instala o serviço, não cria unit do systemd, não registra MCP por CLI e não verifica
busca real.

Consequência para o desenho — o D7 tem duas metades:

| metade | o que faz | origem |
|---|---|---|
| **serviço** | venv/pacote, env de porta, unit systemd, subir e checar `/health` | inexistente hoje |
| **cliente** | `hermes mcp add`, skill, hooks, verificação pós-escrita | absorve o `kit-dtp/instalar.sh` |

Requisito novo derivado: **R10** — THE instalador SHALL absorver o comportamento do
`kit-dtp/instalar.sh` (backup antes de sobrescrever, merge que não apaga o que não veio no payload,
correção de caminho absoluto), e SHALL NOT depender do kit estar presente na máquina.

### Achado do G0 (20/out): o público é Windows

Levantamento do usuário: **95% dos colegas usam Windows**; destes, cerca de **1/3** tem WSL
habilitado. Isso inverte a prioridade do desenho — o alvo primário não é o Linux desta máquina.

O que o projeto já oferece (verificado):

| plataforma | mecanismo existente | arquivo |
|---|---|---|
| Linux | `systemd --user` (recomendado na doc) | `docs/deployment/systemd-service.md` |
| Windows | **Task Scheduler no logon** | `scripts/server/start_http_server.bat` (existe) |
| ambos | health check | `scripts/server/check_server_health.py` |

Requisitos derivados:

- **R11** — THE instalador SHALL funcionar em **Windows nativo**, sem exigir WSL.
- **R12** — THE instalador SHALL escolher o backend de serviço pela plataforma: `systemd --user`
  no Linux, **Task Scheduler** no Windows (`schtasks /create /sc onlogon`), e SHALL declarar no
  relatório final qual backend usou.
- **R13** — WHEN a plataforma não for suportada, THE instalador SHALL abortar **antes** de escrever
  qualquer artefato, dizendo qual plataforma detectou.
- **R14** — THE instalador SHALL ser escrito em Python (stdlib + o próprio pacote), e SHALL NOT
  depender de `bash`, `systemctl` ou ferramentas POSIX no caminho Windows.

## Critérios de aceitação adicionais (Windows)

- **CA7** — Em Windows nativo: instalação deixa o serviço respondendo em `/health` após logon, com a
  tarefa criada e nomeada de forma identificável; `--uninstall` remove a tarefa.
- **CA8** — O caminho Windows não usa `bash`, `systemctl` nem `chmod`.

## Bloqueio declarado (honestidade de escopo)

Não há máquina Windows nesta infraestrutura para o G6. O E2E Windows será validado por:
(a) teste automatizado do **gerador de comando** `schtasks` (unidade, roda em qualquer SO, verifica a
string gerada); (b) execução real em uma máquina de colega, com o relatório do instalador colado na
issue. Declarar isso explicitamente — não fingir paridade de teste.
