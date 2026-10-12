# Estratégia Fork ↔ Upstream — manter learning-loop vivo enquanto o upstream anda

> Criado 2026-10-12. Gatilho: inbox cheio de atividade upstream desde sex 09/out + citações de partes nossas.
> Regra-mestra (Claudio): **trazer o que der sem quebrar o learning-loop; integrar incremental; medir antes de mexer.**

## Estado da divergência (verificado via git, 12/out 10:3x)

- **fork `main` = 309 commits à frente / 11 atrás** de `upstream/main`.
- Os 309 à frente = nosso learning-loop (A/B/C/E, D1/D2/D3, dreaming, util-modular) + multilíngue + docs fork — TUDO fork-only, não endossado (F4 travada).
- Os 11 atrás = atividade upstream desde 09/out. **Nenhum colide com arquivos do learning-loop.**

## Os 11 commits upstream — classificação por risco ao learning

| Commit | PR | Assunto | Arquivos | Toca learning? | Ação |
|---|---|---|---|---|---|
| `2a182d02` | #1515 | fix hybrid: diff hashes no pull | `storage/hybrid.py`, teste | ❌ (delta-sync, arco fechado) | TRAZER |
| `840b467a` | #1500/#1492 | consolidation NÃO comprime o que o serviço escreveu | `consolidation/compression.py` +4 testes | ⚠️ mesmo módulo, MAS não tocamos compression.py → **a FAVOR** do learning (protege o que ele gera) | TRAZER (prioridade — alinha com nossa tese) |
| `73f78978` | #1499 | delta-sync deadlock em poison events | `storage/sync/*` | ❌ (follow-up do NOSSO arco delta-sync) | TRAZER |
| `e1e7bc0a` | #1501 | remember-session skill | `claude-hooks/*` | ❌ (plugin) | TRAZER (neutro) |
| 7× | #1498/1508/1509/1510/1511/1512/1513 | redesign site/OG/fonts/thumbnails | `site/`, `docs/`, `scripts/maintenance/site_thumbnails.mjs` | ❌ zero | TRAZER (cosmético) |

**Conclusão:** dos 11, só 1 (`compression.py`) está no mesmo módulo que o learning, e NÃO há conflito (nunca editamos esse arquivo). Os outros 10 são sync (fechado), plugin e site. **Risco de quebrar o learning ao trazer os 11 = baixíssimo.**

## "Citação de partes nossas" (o que o inbox mostrou)

- `#1499` (delta-sync deadlock) é o **Henry evoluindo em cima do nosso arco delta-sync** (5 fases que mergeamos). Edge-case "poison event" (create com content vazio) — correção legítima, não regressão nossa. É sinal de que o upstream está construindo sobre o que entregamos.
- `#1500/#1492` (não comprimir o que o serviço escreveu) **valida nossa tese do learning**: memória que o serviço gera (ex. beliefs, feedback) não deve ser comprimida como ruído. Alinhado.

## Estratégia de integração (incremental, sem quebrar)

### Princípio
`upstream/main` é a base canônica. O learning-loop é fork-only EMPILHADO sobre ela. A cada janela de atividade upstream: **rebase/merge do upstream PRIMEIRO, re-rodar o gate do learning (testes + benchmark), SÓ então seguir com código novo (R4...).**

### Passo a passo desta rodada
1. **Merge `upstream/main` no fork `main`** (os 11 commits). Esperado: ZERO conflito (verificado — nenhum arquivo em comum com o learning).
2. **Re-rodar a suíte do learning** (test_fechar_loop_a_d, test_frente_d_*, test_util_modular, test_multilingual_recall) + o **benchmark** (baseline MRR 0.683). Se algum número mudar → investigar ANTES de prosseguir. É o "medir antes de mexer".
3. **Validar o fix do `compression.py` (#1500)** contra o learning: confirmar que beliefs/feedback que o serviço escreve agora ficam protegidos da compressão (pode até resolver um risco latente nosso).
4. Só então **retomar o R4** sobre a base atualizada.

### Daqui pra frente (trabalho fork↔upstream intenso)
- **Sincronizar cedo e com frequência** — não deixar divergência acumular (309 à frente já é muito; cada PR nosso que mergear reduz).
- **Ordem de PRs nossos para upstream** (quando F4 abrir): seguir a tabela de 10 PRs já validada (item 85febdca). PR1 (telemetria) desbloqueado por #1499.
- **O learning novo (R4/multilíngue) é "muda ranking de todos"** → opt-in env-gated (recomendação seven), mesmo gate da F4, corroboração Henry.
- A cada sync: re-rodar gate + benchmark. O benchmark é o guarda-contra-regressão do learning.

## Invariante
Nenhum merge do upstream pode baixar o MRR do benchmark nem quebrar as flags do learning. Se baixar → o upstream mudou algo que nos afeta → investigar e adaptar o fork, não desligar o learning.
