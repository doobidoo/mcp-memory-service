# SPEC — D8: porta e transporte declarados num único lugar

**Status:** PLANNED (G1) · **Arco:** UIAA/hermes (D8) · **Autor:** T'Pol
**RFC-mãe:** `rfc-hermes-first-class.md` §9 · **Pesquisa:** `hermes-primeira-classe.md` §1 (camadas)
**Classe:** decisão + normalização (não feature)

## Problema

O default do projeto é **8000** (`MCP_HTTP_PORT`, `config/transport.py:30`). A nossa instalação usa
**3202**, herdado do transporte SSE legado (`MCP_SSE_PORT`) da época em que vários MCPs conviviam na
mesma máquina. Hoje a porta está espalhada por: env por-host, `systemd` unit, `kit-dtp`, scripts de
sonda, configs de cliente MCP e skills.

A pesquisa `hermes-primeira-classe.md` §1 já registra o sintoma na camada 1 (Connectivity): o serviço
responde em `127.0.0.1:3202` com `/health` ok, **mas a wiki registra 8000**. Ou seja, o erro de
documentação não é hipótese — está mapeado, e é o que confunde quem chega de fora.

## Decisão

**Manter 3202.** Migrar não entrega função nova, e o blast radius alcança 3 hosts, kit, units,
scripts e configs de cliente. O que falta não é mudar a porta: é **declará-la em um lugar só**.

## Requisitos (EARS)

- **R1** — WHEN o instalador configurar um cliente MCP, THE sistema SHALL obter a porta efetiva de
  uma única fonte declarada (env por-host), nunca de valor embutido no código do instalador.
- **R2** — WHEN a porta declarada divergir da porta em que o serviço responde, THE `mms-doctor`
  SHALL reportar a divergência com os dois valores como evidência, e SHALL NOT tentar corrigir.
- **R3** — THE documentação do harness SHALL declarar explicitamente: "3202 por escolha; 8000 é o
  default do projeto".
- **R4** — IF a variável de porta não estiver definida, THEN o sistema SHALL assumir 3202 e registrar
  de onde veio o valor (env, arquivo de config ou default).

## Critérios de aceitação

- **CA1** — Existe exatamente **um** artefato que define a porta; instalador, doctor e scripts leem
  dele. Teste: alterar o valor nesse artefato muda o endpoint usado por instalador e doctor, sem
  editar código.
- **CA2** — Doctor detecta mismatch (porta declarada ≠ porta que responde) e sai com código 1,
  citando `declarado=3202 ouvido=3203`.
- **CA3** — A documentação do harness (wiki + `docs/agents/hermes.md` quando existir) contém a frase
  de R3 com o valor real, verificável por grep. A wiki deixa de registrar 8000 como porta do serviço.
- **CA4** — Nenhum host é migrado nesta entrega; 3202 continua sendo o valor em produção.

## Gate

G0 (seven: mapear onde a porta aparece hoje) → G1 (esta spec) → G3 RED (testes dos CAs 1/2) →
G4 GREEN → G5 (tuvok) → G6 (prova no host).

## Fora de escopo

- Migração para 8000.
- Mudança de transporte (SSE ↔ streamable-http).
