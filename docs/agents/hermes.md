# Hermes como cliente do memory-service

Como ligar o `mcp-memory-service` a um Hermes existente e usá-lo como memória primária.
Este documento descreve o que foi **medido**, não o que se pretende.

> Escopo: Hermes ≥ 0.22.2. Linux validado; Windows tem caminho preparado mas **não validado**
> (ver §7).

## 1. Instalação em um comando

O instalador vive em `scripts/hermes/instalar.py` e é stdlib-only:

```bash
python scripts/hermes/instalar.py --port 3202
```

Com `HERMES_HOME` definido, instala naquele perfil; sem ele, em `~/.hermes`.
Flags: `--dry-run` (mostra o plano), `--hermes-home`, `--port`, `--env-file`, `--uninstall`.

O que ele faz, nesta ordem:

1. resolve o `HERMES_HOME` (env > `~/.hermes`) e **aborta antes de escrever** se o serviço não
   responder em `/health`;
2. detecta o que já existe (`hermes mcp list`, `config.yaml`, health) e planeja só o que falta;
3. registra o servidor via `hermes mcp add mms --url http://localhost:<porta>/mcp`;
4. habilita a skill de doutrina e os hooks de convivência;
5. grava uma **sonda** no acervo e busca de volta — a instalação só é declarada bem-sucedida se a
   busca devolver resultado (inclusive em banco zerado);
6. grava um manifesto com os comandos de undo.

**Idempotência medida**: segunda execução imprime `nada a fazer: instalacao ja completa` e não roda
nenhum comando. `--uninstall` remove o MCP e desabilita skill/hooks, sem tocar no que não criou.

## 2. Porta: 3202 por escolha

O default do projeto é **8000** (`MCP_HTTP_PORT`); o transporte SSE legado usa 8765. A nossa
instalação usa **3202**, definida em **um único lugar** — o arquivo de env do serviço lido pela unit:

```
MCP_SSE_PORT=3202
```

Instalador e `mms-doctor` leem a porta dessa fonte (e de `--port` explícito), nunca de valor
embutido no código. Se a porta declarada divergir da que responde, o doctor reporta `declarado=X
ouvido=Y` — não adivinha.

## 3. O regime de memória (o que muda na prática)

Dois canais, com papéis distintos:

| canal | conteúdo | quando usar |
|---|---|---|
| `MEMORY.md` / `USER.md` do Hermes | identidade, preferências duráveis, regras curtas — injetados todo turno (~1,7k chars) | sempre presente; **não** é fonte de estado |
| `mcp-memory-service` | histórico, decisões datadas, procedimentos, o que não cabe no arquivo | busca semântica sob demanda |

Regra que evita o erro clássico: **pergunta sobre estado, pendência ou "o que fizemos" → mms
primeiro.** `MEMORY.md` responde "quem é o usuário e quais são as regras", não "onde paramos".

Na prática isso significa migrar procedimento para o acervo com tags (ex.: `procedimento`) e deixar
no arquivo só o que precisa estar no contexto sem busca. Foi o que reduziu `MEMORY.md` de 2.169 para
1.761 chars sem perder informação: o conteúdo operacional passou a ser recuperável por busca.

Cada canal (CLI, TUI, Telegram) é uma sessão separada: o transcript não atravessa, mas o acervo sim.

## 4. Verificação: `mms-doctor`

```bash
mms-doctor --db-path ~/local-data/mcp/sqlite_vec.db --repo-path ~/git/mms/mms
```

Seis checks com evidência, `≤10 s`, exit `0`/`1`:

| check | o que confirma |
|---|---|
| `health` | `/health` responde; em falha, **um** restart e espera de readiness com backoff |
| `port` | porta declarada (env-file) == porta que responde |
| `version` | `__version__` do pacote == `pyproject.toml` do código em execução |
| `sync` | cursor do delta-sync dentro da janela (`MCP_DOCTOR_SYNC_MAX_AGE`, default 1 h) |
| `author` | a última memória carrega `agent_id` |
| `embeddings` | memórias vivas têm embedding (tombstones não contam) |

**Guarda importante**: o reparo nunca reinicia uma unit em `activating`/`reloading`/`active`. O
serviço leva ~30 s para subir; reiniciar durante o start impede que ele complete (observado: systemd
`start-limit-hit`, serviço fora do ar). Porta errada não é motivo de restart — é configuração.

## 5. Pegadinhas do CLI do Hermes (medidas em 20/out/2026)

Três comportamentos que custaram um E2E inteiro para descobrir:

1. **`hermes mcp add` é interativo.** Ele lista as tools e pergunta `Enable all N tools?`. Sem stdin
   respondido, imprime o prompt, imprime `Cancelled.` e **retorna 0** — parece sucesso e não
   registra nada. `echo "Y"` sozinho não basta (mais de um prompt); o instalador envia várias linhas.
2. **`hermes config get` não existe.** O subcomando aceita `show/edit/set/path/env-path/check/migrate`.
   Para observar `skills.mms.enabled` a única via stdlib é ler `config.yaml` (leitura, nunca escrita).
3. **`hermes config unset` não existe.** O undo de `config set X true` é `config set X false`.

## 6. Autoria

O serviço em streamable-http honra `X-Agent-ID` (upstream #1598). Quem grava sem `agent_id`
explícito cai no `MCP_AGENT_ID` do processo — por isso o hook `mms_agent_stamp.py` carimba as
chamadas de escrita, e o doctor verifica a última gravação.

## 7. Windows (não validado)

O público-alvo é majoritariamente Windows nativo; o instalador já escolhe o backend por plataforma:

| plataforma | backend |
|---|---|
| Linux | `systemd --user` |
| Windows | `schtasks /create /sc onlogon` |
| outro | aborta **antes** de escrever, declarando a plataforma |

O caminho Windows não usa `bash`, `systemctl` nem `chmod` (verificado por teste). Falta a execução
real em máquina Windows — prevista via VM VirtualBox, **não** concluída.

## 8. O que ainda não existe

- **Comando de upgrade.** Não há verbo de atualização; o que existe é
  `pip install --upgrade mcp-memory-service==X` + restart. Há uma proposta em discussão
  (Discussion #1609) para um `mms upgrade` que detecte o modo de instalação (empacotado × editable)
  e faça backup + verificação + rollback.
- **Provider nativo de memória.** Decidido não ligar no modo híbrido: sobre uma `MEMORY.md` intacta
  ele produziria injeção e gravação duplicadas.
