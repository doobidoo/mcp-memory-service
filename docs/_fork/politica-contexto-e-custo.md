# Política de contexto e custo — Hermes/DeepSeek

> Motivo: o histórico da sessão é reenviado a cada turno. O custo cresce de forma
> quase quadrática com o número de turnos e, em sessões longas, a maior parte do
> que se paga é **gestão**, não trabalho. Claudio pediu explicitamente (17/out):
> "monitore isso e documente; precisamos ser econômicos, senão tu pode ficar sem
> token para operar."

## Instrumento

```
python scripts/ops/session_cost.py            # top 10 sessões
python scripts/ops/session_cost.py --session <id>
```

Estimativa: `chars / 3.5` (misto PT/EN). Serve para tendência e gatilho de
política — não é contabilidade de fatura.

## Limiares

| nível | tokens estimados | ação obrigatória |
|---|---|---|
| ok | < 150 mil | seguir |
| ATENÇÃO | 150–300 mil | checkpoint no repo + no mms antes de continuar |
| CRÍTICO | > 300 mil | encerrar a sessão e abrir nova a partir do checkpoint |

## Regras de trabalho (o que reduz o custo na origem)

1. **Delegar o trabalho pesado.** Subagente tem contexto isolado: as idas e voltas
   dele não inflam o meu histórico. O que volta é o resumo.
2. **Achado vai para o repo, não para o contexto.** Commit imediato + uma linha de
   resumo. Histórico não é armazenamento.
3. **Não despejar saída de ferramenta.** Ler o necessário, filtrar no comando
   (`--json`, `grep`, `head`), nunca o arquivo inteiro quando 20 linhas bastam.
4. **Resposta curta por padrão.** Tabela e número em vez de narrativa.
5. **Sem releitura.** Se já li o arquivo nesta sessão, uso o que li; se preciso
   conferir, uso `grep` dirigido.
6. **Checkpoint antes de sessão nova**: `docs/_fork/` + mms + LEDGER.

## Registro — 17/out/2026

Sessão `20261017_154550_b60f7c` (a que trabalhou o mms):

| métrica | valor |
|---|---|
| mensagens | 810 |
| caracteres | 1.006.501 (~1 MB) |
| tokens estimados | ~288 mil |
| nível | ATENÇÃO próxima de CRÍTICO |

Diagnóstico: o volume não veio do trabalho, veio da **gestão** — dezenas de
comandos de diagnóstico encadeados, com saída completa no histórico. A partir
de agora, cada bloco de investigação começa com medição e termina com commit.

## Relação com a compactação automática

O Hermes compacta em `threshold: 0.50`, mas com **piso de 0,75** para modelos
abaixo de 512K — ou seja, a compressão só dispara perto do limite da janela.
Ela evita o estouro, não evita o custo: comprimir tarde significa pagar caro
antes de comprimir. Por isso a política acima é mais rigorosa que o default.
