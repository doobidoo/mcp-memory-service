# Repository Guidelines

## Project Structure & Module Organization

MCP Memory Service is a Python 3.11+ package. Source lives in
`src/mcp_memory_service/`, including storage backends, embeddings, MCP handlers,
and the HTTP API. Dashboard assets live in `src/mcp_memory_service/web/static/`.
Tests live in `tests/`, including `unit/`, `integration/`, and `consolidation/`.
Use `docs/` for documentation, `scripts/` for tooling, and `changelog.d/` for
per-change release fragments.

## Build, Test, and Development Commands

Run from the repository root in a virtual environment:

- `python install.py`: install with platform-specific dependency selection.
- `python -m pip install -e '.[dev]'`: install editable code and development tools.
- `python scripts/validation/check_dev_setup.py`: verify the editable installation.
- `memory launch`: start the HTTP service and dashboard; `memory server` runs MCP stdio.
- `pytest tests/`: run tests; `pytest tests/unit/<test_file>.py -q` targets a file.
- `pytest --cov=mcp_memory_service tests/`: collect coverage.
- `black --check src/ tests/` and `ruff check src/ tests/`: check formatting and lint.
- `python -m build`: build distribution artifacts.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, Black's 88-character lines, and double
quotes. Use `snake_case` for functions/modules and `PascalCase` for classes.
Add type hints to function signatures and Google-style docstrings to public APIs.
Follow neighboring code and use specific exceptions.

## Testing Guidelines

Use pytest and pytest-asyncio; name files and functions `test_*`. For `src/`
changes, provide a regression test that fails against the base code and passes
with the change; pure removals are exempt. Exercise real storage and handlers;
mock external services. Cover changed behavior, edge cases, and error paths.
See [CONTRIBUTING.md](CONTRIBUTING.md) for CI proof requirements.

## Commit & Pull Request Guidelines

Branch before committing, using names such as `fix/issue-description`. Follow
semantic commits, e.g. `fix(storage): handle literal tag wildcards`. Open PRs on
GitHub with the problem, solution, linked issue, test commands/results, and
screenshots for UI changes. Changes under `src/` need
`changelog.d/<number>.<category>.md`; categories are `added`, `fixed`, `removed`,
or `internal`. Codeberg is an archive.

## Issue Claim Automation

Issues labelled `auto-assign` accept a standalone `/claim` comment. Wait for the actual assignment and confirmation before working;
the confirmation approves only the issue's prepared scope. Other issues retain
manual maintainer confirmation. See CONTRIBUTING.md for the full rules.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **mcp-memory-service** (30132 symbols, 60251 relationships, 1069 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/mcp-memory-service/context` | Codebase overview, check index freshness |
| `gitnexus://repo/mcp-memory-service/clusters` | All functional areas |
| `gitnexus://repo/mcp-memory-service/processes` | All execution flows |
| `gitnexus://repo/mcp-memory-service/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
