# The installer method — learned from `codebase-memory-mcp`

Anotação do método do instalador do `codebase-memory-mcp` 0.11.0, observado em 20/out/2026
neste host (`install --dry-run` e `install --clients`). É o modelo que queremos para o
memory-service: **um instalador, N harnesses, superfícies explícitas**.

## 1. O princípio

Instalar um MCP num agente **não é** escrever uma linha no config. É depositar um payload em
**todas as superfícies** que o harness usa, uma por camada de uso:

| superfície | o que é | exemplo (Hermes) |
|---|---|---|
| `mcp` | registro do servidor | `~/.hermes/config.yaml` |
| `instructions` / `steering` | texto que o agente lê sempre | `~/.kiro/steering/<nome>.md` |
| `skill` | doutrina de uso (quando recordar, como etiquetar) | `~/.hermes/skills/<nome>/SKILL.md` |
| `hook` | ciclo de vida (captura, injeção, autoria) | `~/.hermes/config.yaml` (`pre_llm_call`, `pre_tool_call`) |
| `agent` | subagentes/perfis auxiliares | `~/.kiro/agents/<nome>-scout.json` |
| `extension` | plugin do harness | `~/.config/opencode/plugins/<nome>.ts` |

Só o `mcp` deixa o agente *alcançar* a memória. As outras quatro fazem ele **usar bem** — que é
a diferença entre conectividade e suporte de primeira classe.

## 2. Detecção, não configuração manual

```
$ codebase-memory-mcp install --dry-run
Detected agents: OpenCode Antigravity VS-Code Cursor Kiro Hermes Warp Amazon Q Developer IDE
```

O instalador **descobre** os harnesses presentes e lista, para cada um, o caminho exato de cada
superfície que tocaria — antes de tocar. O operador vê o plano inteiro.

**Alvos condicionais**: harness cujo marcador/config documentado não existe **não é criado**.
Isso evita poluir a máquina com configuração de ferramenta que a pessoa não usa.

**Tokens explícitos** para quando a detecção não basta:

```
$ codebase-memory-mcp install --clients
  claude · codex · gemini · zed · opencode · aider · vscode · cursor · windsurf
  kiro · hermes · openhands · cline · warp · junie · goose · amp · …
```
45 superfícies hoje, com uma lista de "manual/UI MCP boundaries" (Qodo, JetBrains AI, Replit…)
que não são automatizáveis e são declaradas como tal.

## 3. Supervisor de ciclo de vida

O binário instala a si mesmo — não existe "copie este script":

```
install   [-y|-n] [--force] [--dry-run] [--dir=<path>] [--skip-config] [--clients=<tokens>]
uninstall [-y|-n] [--dry-run]
update    [-y|-n]
config    <list|get|set|reset>
```

- `--dry-run` mostra tudo e não escreve;
- `--skip-config` instala só o artefato, sem tocar em config (pacote ≠ configuração);
- `-y/-n` tornam a confirmação explícita, então automação é possível sem prompt;
- `update` é cidadão de primeira classe — não é "reinstale por cima".

## 4. Preservar estado do usuário

Antes de qualquer escrita, o instalador **reconhece o que já existe** e diz que vai manter:

```
Found 19 existing index(es). Keeping them.
  /home/claudio/.cache/codebase-memory-mcp/<projeto>.db   (×19)
```

Instalar não é resetar. Índices, bancos e configuração prévia sobrevivem — e a mensagem prova
isso ao operador, em vez de pedir confiança.

## 5. Caminhos por harness (observados neste host)

| harness | mcp | instructions | skill | outros |
|---|---|---|---|---|
| **Hermes** | `~/.hermes/config.yaml` | — | `~/.hermes/skills/<n>/SKILL.md` | hook em `config.yaml` (`pre_llm_call`); também toca `~/.agents/skills/` e `~/.copilot/skills/` |
| **Kiro** | `~/.kiro/settings/mcp.json` | `~/.kiro/steering/<n>.md` | `~/.kiro/skills/<n>/SKILL.md` | `~/.kiro/agents/<n>[-scout|-auditor].json` |
| **OpenCode** | `~/.config/opencode/opencode.jsonc` | `~/.config/opencode/AGENTS.md` | `~/.config/opencode/skills/<n>/SKILL.md` | `~/.config/opencode/agents/<n>.md`, plugin `plugins/<n>.ts` |
| **VS Code** | config do cliente | — | — | — |
| **Cursor** | config do cliente | — | — | — |

O padrão que se repete: **config do MCP** + **um arquivo de instrução** + **um diretório de
skill** + **subagentes**, variando só o caminho raiz por harness.

## 6. O que isso nos obriga a mudar no instalador do mms

Estado atual: `scripts/hermes/instalar.py` serve **um** harness e **uma** superfície (mcp),
habilitando skill/hooks por `config set` — sem depositar os arquivos.

Alvo:

1. **payload versionado no repo** — `scripts/hermes/payload/{agent-hooks,skills}/…`, não em
   `~/.hermes` do autor;
2. **detecção de harness** — Hermes e Kiro no mínimo (os dois que rodamos), com tabela de
   caminhos explícita e alvos condicionais;
3. **depositar, não só habilitar** — copiar hook e skill para o diretório do harness;
4. **registrar hooks por merge de config** — `hermes config set` não cria listas
   (`hooks.pre_tool_call[0].command` vira chave literal `pre_tool_call[0]`), e não existe
   `hermes hooks add`; então merge com backup, usando `PyYAML` (já é dependência do projeto);
5. **identidade por harness, não por pessoa** — o hook lê `MCP_AGENT_ID` do ambiente; o
   instalador grava essa variável no `.env` do perfil. Nada de `AGENTE = "tpol"` no código;
6. **supervisor completo** — `install`, `uninstall`, `update`, `config`, `--dry-run`,
   `--skip-config`, `--clients`.

## 7. O que não copiar

- **45 clientes** é ambição de projeto maduro; começamos com os que rodamos (Hermes, Kiro) e
  adicionamos com evidência de teste real, um por PR.
- **Subagentes especializados** (`-scout`, `-auditor`) são valor do codebase-memory, não do
  nosso caso: memória não precisa de auditor de grafo.
