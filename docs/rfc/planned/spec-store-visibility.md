# spec-store-visibility — busca padrão enxerga todos os stores

> Status: **PLANNED** (G2). Fork-only. Origem: `docs/_fork/diario-de-gates.md` §E10.
> Classe: **BUG de design** (perda silenciosa de memória por escopo implícito).
> Autor da spec: T'Pol (manager). Implementação: delegada. Review: independente.

## 1. Defeito, com evidência

`memory_search` sem `store` explícito usa `"default"` (assinaturas em
`services/memory_service.py:298` e `:386`). As 7.466 memórias dos stores `mir`,
`docs-shared` e `hackathon` (**26,9% do acervo**) são inalcançáveis nessa busca.

Régua cross-store de 60 casos (20 por store, validada por código), índice uniforme,
régua por documento:

| escopo | @10 | @30 | @100 |
|---|---|---|---|
| só `default` | **0,000** | 0,000 | 0,000 |
| todos os stores | **0,350** | 0,433 | 0,600 |

Custo no lado oposto, régua de 120 casos (ouro no `default`): abrir os stores derruba
o @10 de 0,592 para 0,567 (**−2,5 p.p.**) e 21,2% dos slots do top-10 passam a vir de
fora. O trade é: −2,5 p.p. de precisão compram 35% de cobertura sobre um quarto do acervo.

## 2. Requisitos (EARS)

- **R1** — WHEN `memory_search` é chamado sem o argumento `store`, THE memory service
  SHALL buscar em todos os stores existentes.
- **R2** — THE memory service SHALL devolver, em cada resultado, o store de origem como
  campo de primeira classe (não escondido em metadata opcional).
- **R3** — WHEN o argumento `store` é fornecido explicitamente, THE memory service SHALL
  restringir a busca a esse store, preservando o comportamento atual.
- **R4** — THE filtro `store` SHALL ser aplicado em todos os modos (`semantic`, `hybrid`,
  `ranked`, `exact`) e em todas as combinações de filtro (tag, tempo, tipo) — não apenas
  no ramo semântico. Corrige a raiz compartilhada com #1466.
- **R5** — THE resposta SHALL declarar o escopo efetivo da busca (ex.: `stores_searched`),
  para que o cliente saiba o que foi e o que não foi varrido.
- **R6** — WHEN o acervo tem stores com nome corrompido (3 conhecidos), THE busca SHALL
  não falhar; o store é devolvido como está e o nome aparece em log nível WARNING.

## 3. Desenho

1. `store=None` significa "todos"; `store="default"` continua significando só o default.
   **Nunca** trocar o default do parâmetro por `"default"` no meio do caminho.
2. No `storage`, o filtro vira lista: `store IN (...)` ou ausência de cláusula. A consulta
   ao `vec0` particionado precisa incluir todos os stores quando não há restrição —
   validar com `EXPLAIN QUERY PLAN` que não vira varredura completa quando há store explícito.
3. O campo `store` no resultado vem de `memories.store`, não inferido.
4. Não mover dados entre stores. Não renomear os stores corrompidos nesta entrega.

Proibido: alterar schema, mudar o default de escrita (`memory_store` continua gravando em
`default` quando não especificado), tocar o caminho de injeção proativa nesta entrega.

## 4. Critérios de aceitação

| # | critério | verificação |
|---|---|---|
| CA1 | régua cross-store @10 sai de 0,000 para ≥0,300 | harness, 60 casos |
| CA2 | régua de 120 casos @10 cai no máximo 3 p.p. (≥0,562) | harness, 120 casos |
| CA3 | `store` explícito isola: 0 resultado de outro store | teste de integração |
| CA4 | todos os modos honram o filtro (regressão de #1465/#1466) | teste por modo × filtro |
| CA5 | cada resultado traz `store` correto (100% conferido por hash→store) | verificação por código |
| CA6 | escrita sem store continua em `default` | teste de regressão |

## 5. Plano de testes (G3 RED antes da correção)

- Teste que falha em `main`: 60 casos cross-store com ouro em `mir`/`docs-shared`/`hackathon`
  → `recall@10 == 0` hoje.
- Teste por modo × filtro (4 modos × 4 filtros) provando isolamento com `store` explícito.
- Teste de campo: `result["store"]` presente e igual ao store do hash.
- Guarda de regressão da escrita.

## 6. Diário de gate

| gate | estado | evidência |
|---|---|---|
| G0 Think | feito (medição E10) | 0,000 × 0,350 na régua cross-store |
| G1 Plan review | **pendente — Claudio** | esta spec |
| G2 Spec | este documento | EARS R1-R6, CA1-CA6 |
| G3 RED | a fazer | 60 casos cross-store falhando em `main` |
| G4 GREEN | a fazer | mínimo para os 6 CAs |
| G5 Review | a fazer | revisor independente |
| G6 Validação | a fazer | serviço vivo: as duas réguas medidas de uma vez |

## 7. Fora de escopo

- Higienização dos stores com nome corrompido (`\x0b\x12...`) — item separado.
- Deduplicação dos 2.468 duplicados (vale 1 caso em 120, medido em E7.5).
- Mudar o store default de escrita.
