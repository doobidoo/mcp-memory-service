# PyPI Defensive Name Placeholders

Tracking issue: [#809](https://github.com/doobidoo/mcp-memory-service/issues/809)

These packages are minimal **redirect placeholders** registered on PyPI to prevent name squatting on candidate names that would be obvious choices if `mcp-memory-service` ever needs to rebrand.

| Name | Import | Purpose | Status |
| --- | --- | --- | --- |
| `agent-memory-service` | `agent_memory_service` | Closest drop-in alternative | ✅ Reserved |
| `ai-memory-service` | `ai_memory_service` | Broadest framing | ✅ Reserved |
| `memory-for-agents` | `memory_for_agents` | Descriptive | ✅ Reserved |
| ~~`agent-memory`~~ | ~~`agent_memory`~~ | ~~Short, brandable~~ | ❌ Blocked — see note |

**Why `agent-memory` is missing**: PyPI rejected it with HTTP 400 because [`agentmemory`](https://pypi.org/project/agentmemory/) (without separator) already exists, and PyPI's name-confusability check (PEP 541-adjacent) treats `agent-memory` and `agentmemory` as too similar. We did not pursue a workaround — the strategic intent of three reserved names plus the existing project is sufficient brand insurance, and the few remaining short alternatives (`agent-mem`, `mcp-agent-memory`) are either similarly blocked or redundant with the actual project name. If a "short brandable" slot becomes critical later, revisit and pick a name that is not a substring/superset of an existing project.

## What each placeholder does

- Imports cleanly on `pip install <name>` so users who guess the wrong name get a clear pointer to the real package.
- Emits a `DeprecationWarning` on import telling them to install `mcp-memory-service` instead.
- Reports the same homepage and "real package" URL via PyPI metadata so the PyPI project page is self-explanatory.
- Stays at version `0.0.1` indefinitely. Do **not** bump the version unless redirecting somewhere else — placeholder packages with no real release schedule are fine under [PEP 541](https://peps.python.org/pep-0541/) as long as they redirect to a real project.

## Building & uploading

The canonical upload path is the manual GitHub Actions workflow [`.github/workflows/publish-placeholders.yml`](../../.github/workflows/publish-placeholders.yml). It uses the existing `PYPI_TOKEN` secret, so no PyPI token ever needs to live on a maintainer's local machine.

**To run:**

1. GitHub → **Actions** → **Publish PyPI Placeholders** → **Run workflow**.
2. Inputs: `target=pypi`, `skip_existing=true`. Click **Run workflow**.
3. Each package runs as its own matrix job (`fail-fast: false`), so a single name failing does not block the others.
4. After the run, verify the project pages render:
   - <https://pypi.org/project/agent-memory-service/>
   - <https://pypi.org/project/ai-memory-service/>
   - <https://pypi.org/project/memory-for-agents/>

**Local fallback** (only if you need to rebuild artifacts or test offline):

```bash
cd tools/pypi-placeholders

# Bash + zsh array (zsh does NOT word-split a plain "x y z" string by default,
# so `for d in $PKGS` would treat the whole list as a single path. Arrays
# work in both shells.)
PKGS=(agent-memory-service ai-memory-service memory-for-agents)

# Build + check. Use `break` instead of `exit 1` so a failure in this
# copy-pasted snippet does not terminate the user's interactive shell.
for d in "${PKGS[@]}"; do
  ( cd "$d" && python -m build && twine check dist/* ) || break
done
```

If you also want to upload locally (requires a PyPI API token in `~/.pypirc`):

```bash
# The glob must be OUTSIDE the quotes so the shell expands it; "$d/dist/*"
# as one quoted string is a literal path and twine fails with
# InvalidDistribution: Cannot find file (or expand pattern).
for d in "${PKGS[@]}"; do
  twine upload "$d"/dist/*
done
```

`~/.pypirc` template (account-scoped API token from <https://pypi.org/manage/account/token/>):

```ini
[pypi]
username = __token__
password = pypi-AgEIcHl... # token, NOT account password
```

## When to delete a placeholder

If MCP-as-brand fades and we **do** rebrand to one of these names, we'd promote that placeholder to be the real package (and possibly delete the others). Until then: leave them alone.
