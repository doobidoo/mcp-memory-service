# PyPI Defensive Name Placeholders

Tracking issue: [#809](https://github.com/doobidoo/mcp-memory-service/issues/809)

These four packages are minimal **redirect placeholders** registered on PyPI to prevent name squatting on candidate names that would be obvious choices if `mcp-memory-service` ever needs to rebrand.

| Name | Import | Purpose |
| --- | --- | --- |
| `agent-memory-service` | `agent_memory_service` | Closest drop-in alternative |
| `ai-memory-service` | `ai_memory_service` | Broadest framing |
| `agent-memory` | `agent_memory` | Short, brandable |
| `memory-for-agents` | `memory_for_agents` | Descriptive |

## What each placeholder does

- Imports cleanly on `pip install <name>` so users who guess the wrong name get a clear pointer to the real package.
- Emits a `DeprecationWarning` on import telling them to install `mcp-memory-service` instead.
- Reports the same homepage and "real package" URL via PyPI metadata so the PyPI project page is self-explanatory.
- Stays at version `0.0.1` indefinitely. Do **not** bump the version unless redirecting somewhere else — placeholder packages with no real release schedule are fine under [PEP 541](https://peps.python.org/pep-0541/) as long as they redirect to a real project.

## Building & uploading

Each placeholder is a self-contained `pyproject.toml` project. Build all four and upload to PyPI:

```bash
cd tools/pypi-placeholders

PKGS="agent-memory-service ai-memory-service agent-memory memory-for-agents"

# Build + check all four. Use `break` instead of `exit 1` so a failure in
# this copy-pasted snippet does not terminate the user's interactive shell.
for d in $PKGS; do
  ( cd "$d" && python -m build && twine check dist/* ) || break
done

# Upload all four to PyPI (interactive — twine will prompt for token / use ~/.pypirc)
for d in $PKGS; do
  twine upload "$d/dist/*"
done
```

Recommended `~/.pypirc` (use a project-scoped API token from <https://pypi.org/manage/account/token/>):

```ini
[pypi]
username = __token__
password = pypi-AgEIcHl... # token, NOT account password
```

After upload, verify the PyPI project pages render correctly:

- <https://pypi.org/project/agent-memory-service/>
- <https://pypi.org/project/ai-memory-service/>
- <https://pypi.org/project/agent-memory/>
- <https://pypi.org/project/memory-for-agents/>

## Why not use a CI workflow

Defensive placeholders are a one-time upload. A CI workflow adds publish credentials and complexity for an action that won't repeat. Manual upload via the snippet above is the right scope.

## When to delete a placeholder

If MCP-as-brand fades and we **do** rebrand to one of these names, we'd promote that placeholder to be the real package (and possibly delete the others). Until then: leave them alone.
