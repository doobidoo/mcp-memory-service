# Bancada — instrumentos de medição do fork

Ferramentas para medir o serviço **antes de afirmar qualquer coisa sobre ele**, com o acervo real
e sem tocar produção. O resultado de cada medição vira documento em `docs/_fork/benchmarks/`.

| script | para quê |
|---|---|
| `snapshot_db.py` | cópia consistente de um banco sqlite do mms (respeita WAL) e conferência de integridade/contagens |
| `medir_store.py` | sobe uma instância efêmera em porta livre, mede N `memory_store` e reporta p50/p95/máx |

## Regras da bancada

1. **Produção nunca é tocada.** A instância da bancada usa porta própria e banco próprio.
2. **O snapshot não é escrito.** Cada execução grava numa cópia de trabalho, descartada no fim.
3. **Registre a configuração efetiva.** `medir_store.py` grava `nli_effective` e `nli_backend` no
   JSON; uma flag sobrescrita em silêncio já produziu resultado falso uma vez (ver o documento de
   21/out).
4. **Publique agregado, nunca conteúdo.** Números sim; texto de memória do corpus, jamais.
5. **Antes/depois vem da bancada; o CI do upstream recebe fixture pequeno e determinístico.**
   São papéis diferentes: evidência não é teste de regressão.

## Resultados publicados

- `docs/_fork/benchmarks/latencia-store-nli-20261021.md` — 0,53 s vs 27,3 s por gravação (issue #1630).
- `docs/_fork/benchmarks/baseline-locomo-20261009.md` — baseline de retrieval.
