# PR Workflow - Mandatory Quality Checks

> Issue and PR numbers in the incident notes below are from the project's GitHub era
> (before June 2026). They are kept because the lessons are still the point; do not try
> to look them up against current numbering.

## Before Creating a PR

**Mandatory**: run the quality checks BEFORE opening the PR. Every multi-iteration
review cycle in this project's history started with skipping this.

```bash
# Step 1: stage your changes — the check reads the index, not the working tree
git add .

# Step 2: run the pre-PR gate (mandatory)
bash scripts/pr/pre_pr_check.sh

# Step 3: only open the PR if it passes
```

Open the PR through the forge web UI or its REST API. `pre_pr_check.sh` prints a
ready-made `gh pr create` line at the end if the GitHub CLI is configured.

### Two traps in pre_pr_check.sh that cost real time

- **It reads the index only.** Run it *staged* and *before* committing. Running it after
  a commit checks an empty index and passes vacuously.
- **Checks 5 and 6.5 scan the whole staged file, not your diff.** Touching one line in an
  old file inherits every pre-existing finding in that file, and both checks block.
  Budget for it. Check 5 wants `# inline import` as a *trailing* comment on the import
  line — a block comment above it does not satisfy the check.
- A `PASS` summary can hide `SKIP`ped checks that were never evaluated (missing optional
  tooling). Read the per-check lines, not just the total.

### What pre_pr_check.sh covers

1. `quality_gate.sh --staged --with-pyscn` — complexity ≤8, security scan, PEP 8
2. Test suite
3. Import ordering
4. Debug-code detection (stray prints, breakpoints)
5. Inline-import markers
6. Docstring coverage
7. Log-injection check — user-provided values in f-string `logger.*` calls must be
   wrapped with `_sanitize_log_value()`

Blocking conditions: any security finding, or a health score below 50.

Checks 1 and 2 of the quality gate (complexity, security) need a text model. They call a
local OpenAI-compatible endpoint through `scripts/pr/lib/llm_prompt.py` — default
`http://127.0.0.1:11437/v1`, override with `MCP_QUALITY_LLM_URL`,
`MCP_QUALITY_LLM_MODEL`, `MCP_QUALITY_LLM_API_KEY`. With no endpoint reachable the gate
exits 3 and `pre_pr_check.sh` reports SKIP, so those two checks are simply not evaluated —
that is the `PASS`-hides-`SKIP` case above, and it was the standing state on this machine
until the local backend was wired up. `MCP_QUALITY_LLM=gemini` uses the Gemini CLI instead.

### Manual fallback (if the script is unavailable)

```bash
bash scripts/pr/quality_gate.sh --staged --with-pyscn
.venv/bin/pytest tests/
```

## Merging Contributor PRs — Pre-Merge Checklist (mandatory)

Before approving and merging any contributor PR, verify all of the following:

1. **CI green** — every required check passes on the PR head, checked on the forge, not
   assumed from a local run.
2. **Automated review read.** A reviewer bot comments on PRs. Read its findings before
   approving, including non-blocking comments — they surface real bugs. Retrieve them
   from the forge API:
   ```
   GET /api/v1/repos/{owner}/{repo}/issues/{n}/comments
   GET /api/v1/repos/{owner}/{repo}/pulls/{n}/reviews
   ```
   Verify each claim against the actual diff before acting on it; review bots do report
   findings that do not hold.
3. **Own review complete** — diff read, security paths checked, findings addressed.
4. **No unresolved change requests.**

> **Why**: PR #1025 (2026-05-27) was approved and merged before the bot review was read.
> It had flagged a real bug (`_Path(env_override)` never calling `.expanduser()`), which
> then needed a follow-up commit. "CI green plus small diff" is not sufficient.

### Fork PRs get no CI on this instance

A PR from a fork does not run CI here, so "CI green" is unavailable for it and the review
has to carry the whole load: read the full diff, and test the branch locally before
merging. Regular collaborators push in-repo branches instead, precisely so their PRs get
CI.

## Community PR Review Policy (mandatory)

A submitted PR is a commitment to a complete, reviewable piece of work. Incomplete PRs
slow the project and have caused real incidents (v10.59.0 merged a partial OAuth fix
that needed two follow-up patch releases).

| Situation | Action |
|-----------|--------|
| PR description is empty or placeholder | Request changes — ask the author to fill in Description and Motivation before any review |
| PR is in Draft status | Do not review or merge. Ask them to mark it ready when complete. |
| PR has TODOs, "coming in a follow-up", or half-wired code | Request changes — no dead code, no deferred wiring |
| We decide to implement it ourselves | Trace the **full call path end-to-end**, not just the diff. Every validation layer must be covered. |

### When to redirect to an Issue

If the author cannot complete the implementation, ask them to open an Issue instead:

> "This looks like it needs more work before it's ready to merge. Would you mind opening
> an Issue describing the problem and your proposed approach? That way the community can
> help shape the solution."

### Why this exists

- **v10.59.0 incident**: PR #942 (cursor/vscode OAuth schemes) was merged with an empty
  description. The `ALLOWED_SCHEMES` whitelist change was correct, but
  `AuthorizationRequest` and `TokenRequest` still used Pydantic `HttpUrl`, silently
  rejecting `cursor://` before the whitelist was ever reached. Two follow-up patch
  releases (v10.59.1, v10.59.2) were needed to actually deliver working Cursor OAuth.
- Root cause: "CI green and it looks small" is not sufficient. Full call-path analysis is.

## Merging Multiple PRs That Touch the Same Files

When batch-merging several PRs, conflicts arise if they modify the same file.

1. **Order first**: identify which PRs share files, merge the base or largest change
   first, dependents after.
2. **Verify each merge before proceeding.** Do not trust the PR's reported state —
   confirm the commit actually landed on `main`:
   ```bash
   git fetch origin main
   git merge-base --is-ancestor <pr-head-sha> FETCH_HEAD && echo landed
   ```
3. **A PR whose base is not `main` can report "merged" while nothing reaches `main`.**
   Check `base.ref` before believing a merged status.
4. **Stacked PRs look like conflicts.** The forge shows both as mergeable against `main`;
   use the ancestor check above rather than the UI's verdict, and re-list open PRs before
   filing anything as a duplicate.
5. **If a PR conflicts after earlier merges**: fetch the head, rebase onto current main,
   push to a new branch, open a substitute PR, merge that, then close the original with
   an explanatory comment.

**Incident (v10.25.0)**: PRs #557, #558, #560 all touched `sqlite_vec.py`. #557 was
reported merged but stayed open; #558 and #562 merged next, so #557 conflicted. Required
a manual rebase and two substitute PRs (#562, #563).

**Incident**: a merge conflict prediction was run against a nearby branch instead of the
current tip, and reported "no conflict" for a change that did conflict. Predict conflicts
against what will actually land — every release commit rewrites the same lines in
`CLAUDE.md`.

### Why This Matters

- **PR #280 lesson**: 7 review iterations, 20 issues found across 7 cycles
- **Root cause**: quality checks not run before PR creation
- **Prevention**: the mandatory pre-PR gate catches these early
- **Time saved**: roughly 30-60 min per PR against multi-day review cycles

### PR Checklist

- [ ] Pre-PR gate passed (complexity ≤8, no security findings, health ≥50)
- [ ] Tests pass locally, with the command output to show for it
- [ ] `Memory` fields accessed by attribute, never via `metadata.get(...)`
- [ ] User-provided values in `logger.*` wrapped with `_sanitize_log_value()`
- [ ] Removed a feature, port, or command? References in `docs/` and `README.md` cleaned
      up in the same change
- [ ] Dashboard change? Verified in a browser, with a note or screenshot of what was
      exercised
- [ ] Diff self-reviewed
