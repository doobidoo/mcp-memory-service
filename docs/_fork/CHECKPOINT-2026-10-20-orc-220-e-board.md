# Checkpoint — 20/out/2026 ~13:10 — Orc 2.2.0 + reorganização do board

## Publicado e em produção

- **Orc 2.2.0 no PyPI** (`orc-hestrator-server`, wheel + sdist), tag `v2.2.0`, CI run
  `39423879210`. VPS `cfnarede.dev` roda 2.2.0; backup pre-deploy
  `tasks.pre-2.2.0-20261020-0912.db`; `get_context` 117.252 chars no board vivo.
- AC-D1 final: **−13,86%** (206.419 → 177.800 chars, board real, 183 itens).
  O fix do `blocked_by` aninhado (achado do G5) devolveu 2,6 p.p. Depois disso o
  critério voltou de 11% para os **12% originais**, atendido.
- Commits-chave: `47147e2` (fixes do G5), `1cbce5f` (AC-D1), `a5de451` (changelog
  EN + CONVENTIONS), `3a8b8e5` (hierarquia do board).

## Decisões escritas hoje (não repetir em conversa)

- `docs/CONVENTIONS.md` no orc: idioma por artefato (EN publica / PT-BR interna),
  formato do changelog, medição com ambiente, hierarquia do board.
- DEVELOPMENT-STANDARDS (CdIA): §8.0, §8.4.2, §8.4.3, §8.8, §8.9, §9.0.2.
- Skills do harness: `orc-work-item-management` (compact-first com tabela de
  tokens), `worker-dev-python` (sem `git stash` em árvore compartilhada),
  `worker-architect` (guarda contra `tool_calls` como texto).
- Steering do Zero: `~/.kiro/steering/dynamic/startup-hook.md` usa
  `verbosity="compact"` e `compact=True` — abertura de sessão caiu de ~34.725
  para ~380 tokens.

## Estado do board (produção)

- `ARC: Orc` (`344b09cd`) reorganizado: `RELEASE 2.2.0` (`484d3c7f`), `RELEASE 2.1.0`
  (`c79cc99c`) com as FEATs dentro, `ARCO DOC-ORC` (`823c63c8`) e o achado do
  `import_graph` agora são filhos. 10 reparents feitos por `parent_id`.
- Convenção: `ARC: <produto> > RELEASE <versão> > fatia`, profundidade máxima 4.
- Aberto: **`45e4ebe5`** — lateral de Arcos da UI mostra as 29 raízes de todos os
  projetos; `/api/arcs` não escopa por workspace nem por raiz.
- Arco DOC-ORC em `work`: DOC-1 (`280c15ae`, onboarding) e DOC-3 (`de468809`, docs
  humanas) em `work` com nota `gate-red`; DOC-2 (`278a2b18`), DOC-4 (`ead509d4`),
  DOC-5 (`d59f7865`) em `queue`.

## Pendente imediato (pedido do Claudio, 20/out)

1. Reorganizar o mesmo jeito no **QueryFlow** (sub-arcos de versão).
2. Criar **`ARC: mms`** e colocar os sub-arcos do mms dentro (hoje são raízes:
   `PROJETO mcp-memory-service › Harvest & Ingestão`,
   `PROJETO mcp-memory-service › Delta-sync & Infra Hub`).
3. `ESTADO.md` do mms tem a árvore ASCII planejada — usar como referência.

## Onde retomar

Board: `get_context(item_id="344b09cd…")` ou `query_items(parent_id=…)`.
Instrumentos: `~/orc-hot/` (`measure.py`, `prod_check.py`, `reparent.py`,
`snap-orig.db`). Plano de documentação: `docs/_fork/PLANO-documentacao-2026-10-20.md`
no repo do orc.
