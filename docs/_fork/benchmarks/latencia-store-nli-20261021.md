# Latência de `memory_store` com e sem o NLI no caminho de escrita — 21/out/2026

Instrumento: `scripts/bancada/medir_store.py` sobre um snapshot consistente do acervo real
(`scripts/bancada/snapshot_db.py`). Issue correspondente: **#1630**.

## Método

1. `snapshot_db.py` copia o acervo com a API de backup do sqlite (respeita WAL) e confere
   `PRAGMA integrity_check`, contagem de memórias, crenças ativas e eventos de sync.
2. `medir_store.py` sobe um `memory-server` **próprio**, em porta livre, apontando para uma
   **cópia de trabalho** do snapshot (o snapshot em si nunca é escrito), mede N gravações e
   encerra o processo — a produção não é tocada em momento algum.
3. Cada gravação usa texto com vocabulário sorteado: só um uuid diferente não basta, porque a
   dedup semântica recusa o texto e a medição viraria retry.
4. O `--env-file` aceita **vários** arquivos, aplicados em ordem (base do serviço + override do
   host). A bancada força porta, banco e `MCP_NLI_ON_STORE`, e registra no JSON a flag
   **efetiva** (`nli_effective`) e o backend (`nli_backend`).

Snapshot medido: **1.141 MB**, integridade `ok`, **29.357 memórias vivas**, **625 crenças ativas**
(`confidence >= 0.35` — o filtro que o loop usa), 585.067 eventos de sync.

## Resultado

Mesma máquina, mesmo snapshot, mesmo env completo — muda só a flag:

| configuração | n | p50 | p95 | máx |
|---|---|---|---|---|
| `MCP_NLI_ON_STORE=false` | 5 | **0,527 s** | 0,607 s | 0,607 s |
| `MCP_NLI_ON_STORE=true` (`backend=cascade`) | 2 | **27,26 s** | 27,76 s | 27,76 s |

**Fator medido: ~52× no p50.** Confirma em bancada o que a instância de produção mostrava
(25–27 s por gravação) e isola a causa no NLI on-store: nem banco, nem embeddings, nem host.

## Achado metodológico — o instrumento também precisa de prova

A primeira execução com `--nli on` deu **0,46 s** e parecia refutar a hipótese. Não era o serviço:
passando só o env do host, faltava `MCP_NLI_BACKEND=cascade` do env base, então o classificador
caía no default `heuristic` (rápido, sem LLM). Duas consequências adotadas:

- `--env-file` passou a aceitar vários arquivos (base + override), como o systemd faz;
- o relatório grava `nli_effective` e `nli_backend`, para que uma flag silenciosamente
  sobrescrita nunca mais passe por resultado.

Regra que fica: **medição sem registro da configuração efetiva não é medição** — é impressão com
casas decimais.

## Mecanismo (para quem for corrigir)

`handle_store_memory` → `check_beliefs_on_store` → `for belief in beliefs[:20]: await
classifier.classify(...)` → `_llm_classify` → `HarvestRewriter._call_llm`, que resolve
`HARVEST_LLM_PROVIDERS`. Com 625 crenças elegíveis, o loop roda 20 vezes; a 1,1–1,4 s por chamada
LLM medida nesta máquina, 20 × ~1,2 s ≈ 25 s.

## Reproduzir

```bash
python3 scripts/bancada/snapshot_db.py ~/local-data/mcp/sqlite_vec.db /tmp/mms-snap.db

E=~/dtp/ai-configs/services/env
python3 scripts/bancada/medir_store.py --db /tmp/mms-snap.db --nli off \
  --env-file $E/memory-service.env $E/memory-service.<host>.env --reps 5

python3 scripts/bancada/medir_store.py --db /tmp/mms-snap.db --nli on \
  --env-file $E/memory-service.env $E/memory-service.<host>.env --reps 2
```

Custo de atenção: com `--nli on`, cada gravação pode consumir até 20 chamadas de LLM.

## Regra de publicação

Divulgue **agregados** (n, p50, p95, máx), o comando e a configuração efetiva. Nunca o conteúdo
das memórias do corpus — o acervo é dado interno; o que viaja para o upstream é número.
