# Plano de trabalho autônomo — noite de 17-18/out/2026

> Autorizado por Claudio em 17/out ("consegue seguir autônomo no mms até amanhã,
> testando e verificando os caminhos?"). Este arquivo é o contrato da autonomia:
> o que faço, o que NÃO faço, e como você audita.

## Escopo PERMITIDO (sem nova autorização)

| # | Ação | Onde |
|---|---|---|
| A1 | Rodar testes, harness e medições | worktree `~/git/mms-lab-precisao` |
| A2 | Re-embedar **o snapshot** (`~/local-data/mcp/lab/snapshot-20261018.db`) | lab |
| A3 | Baixar modelos ONNX/HF para teste (1-2 GB cada) | cache local |
| A4 | Commits no branch `lab/precisao-recall` | lab |
| A5 | Corrigir bugs encontrados no próprio lab, com RED→GREEN | lab |
| A6 | Documentar resultados, inclusive negativos | repo (docs/_fork) |
| A7 | Rodar A/B dos 4 modelos e produzir tabela comparativa | lab |

## Escopo VEDADO (só com o Claudio presente)

| # | Ação | Motivo |
|---|---|---|
| V1 | Tocar a `main` do repo mms (merge/cherry-pick) | linha viva; serviço roda dela |
| V2 | Alterar o serviço em produção (config, restart, env) | pode derrubar o acervo |
| V3 | Deletar/alterar memórias do acervo vivo | dados do usuário |
| V4 | Push para upstream (Henry/doobidoo) | comunicação externa exige draft + OK |
| V5 | Criar o repo `tripulacao` no GitHub | ele mandou segurar a execução |
| V6 | Qualquer gasto de API paga | A/B é local; custo zero por decisão |

## Plano da noite (ordem e critério de parada)

| etapa | entrega | critério de sucesso | reversão |
|---|---|---|---|
| E1 | explicar a divergência harness×tool (0% vs 20%) | causa identificada com evidência e harness corrigido; 7 testes verdes | branch lab |
| E2 | baseline oficial | `semantic` entre 15% e 25%, estável em 2 execuções, contra snapshot | nada a reverter (leitura) |
| E3 | A/B de modelos (atual, granite-97m, Qwen3-0.6B, harrier-270m) | tabela com recall PT, recall EN, ms/embedding, tamanho do banco | snapshot descartável |
| E4 | relatório final | `docs/_fork/relatorio-noturno-20261018.md` com números e recomendação | — |

**Regra de parada:** se E1 não fechar, não avanço para E2-E3. Número não confiável
não vira base de decisão — foi o erro que já custou uma rodada hoje.

## Como o Claudio audita

1. `docs/_fork/relatorio-noturno-20261018.md` — resultado único, com números.
2. `git log --oneline lab/precisao-recall` — o que foi commitado.
3. `LEDGER-feats.md` — estado de cada item.
4. O snapshot é descartável: nenhum dado real foi tocado.

## Riscos declarados

- CPU ocupada por horas (re-embed de 4 modelos sobre 27,7 mil memórias).
- Download de 1-2 GB por modelo candidato.
- Se a sessão cair, o trabalho está no repo e no snapshot — nada se perde.
