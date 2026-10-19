# Worktrees do fork mms (DNBSCDC289)

> Registro canônico. Worktree é área de trabalho **de uma sessão por vez** — trabalhar no
> worktree de outra sessão causa colisão de branch e perda de estado. Antes de criar um novo,
> verifique esta lista; ao criar, **adicione a linha aqui no mesmo commit**.
> Atualizado: 19/out/2026.

## Lista medida

| caminho | branch | para quê | estado |
|---|---|---|---|
| `~/git/mms/mms` | `main` | **LINHA VIVA** — o serviço systemd roda daqui (venv editable ONNX) | ativo, serviço |
| `~/git/mms/mms-dev` | `pr/delta-sync-phase3-upstream` | pilha de PRs para o upstream (sai de `upstream/main` limpo) | ativo |
| `~/git/mms/mms-pr-recreate-of` | `fix/1522-recreate-of` | PR pontual de apoio ao Henry (#1522) | ativo |
| `~/git/mms/_wt-fsspec-214` | `fix/fsspec-214-ssti` | correção fsspec/SSTI | ativo |
| `~/git/mms/lab-recall` | `lab/precisao-recall-tpol` | **bancada de recall** (criado 19/out): harness, gabaritos, migração, ponte ONNX, deep_search | ativo — sessão T'Pol |
| `~/lab-learning-loop/code/fork` | detached HEAD | laboratório de learning loop — cópia do fork | ativo |
| `~/lab-learning-loop/code/upstream` | detached HEAD | laboratório de learning loop — cópia do upstream | ativo |

Rastreio: `git -C ~/git/mms/mms worktree list --porcelain` é a fonte de verdade; esta tabela
explica **para quê** cada um existe, o que o comando não diz.

## Regras

1. **Um worktree, uma sessão.** Se o worktree estiver em branch diferente da esperada ou com
   arquivos modificados que você não fez, não trabalhe nele.
2. **Bancada é nova, não a de outra sessão:** `git worktree add ~/git/mms/lab-<tema> -b lab/<tema> main`.
3. **Remoção exige commit ou descarte explícito:** `git worktree remove <path>` recusa com
   trabalho sujo — isso é proteção, não erro.
4. **Documentação anda junto:** worktree criado, linha nesta tabela. Worktree removido, linha
   removida. A regra do LEDGER vale aqui também — rastro ou não existiu.
5. Worktrees não compartilham `venv`: para rodar testes num worktree sem criar venv,
   `PYTHONPATH=$PWD/src <venv-do-repo-principal>/bin/python -m pytest ...` e confirme com
   `python -c "import mcp_memory_service; print(m.__file__)"` que importa do worktree.

## Histórico

- **19/out/2026** — criado `~/git/mms/lab-recall` a partir de `github/lab/precisao-recall`
  (branch local `lab/precisao-recall-tpol`), porque o laboratório de recall vivia apenas no
  remoto e o diretório de bancada anterior (`~/local-data/mcp/lab/`) não existe mais neste host.
  Os artefatos estão versionados na branch: `eval/`, `scripts/lab/onnx_embeddings_api.py`,
  `scripts/maintenance/migrate_embeddings.py`, `scripts/deep_search.py`.
