# /merge-sweep — Merge Open PRs With Minimal Effort

Role: merge coordinator for open PRs (GitHub, `gh`). Goal: merge as many PRs as
possible with minimal effort.

## 1. Inventory

```bash
gh pr list --state open --json number,title,author,headRefOid,isDraft,mergeable,mergeStateStatus,labels
```

## 2. Check each PR — read only, change nothing

- CI per head SHA (`gh api repos/{owner}/{repo}/commits/<sha>/check-runs`), not `gh pr checks`.
- Unresolved review threads (GraphQL `reviewThreads`, `isResolved`). ResolveThreads has no bypass.
- Conflicts or drift against main. A fork PR showing `action_required` means the run needs approval.
- Dependabot: did the bump widen a load-bearing constraint?

Only when a question needs a local answer: worktree under `../worktrees/pr-<N>`, run
`.venv/bin/pytest` scoped there. Never touch the main tree — several sessions share it.

## 3. Classify

- **MERGE**: green, no open threads, no conflicts
- **QUICK-FIX**: under 10 minutes (rebase, answer a thread, approve a run) — name the fix
- **DECISION**: needs the maintainer's judgment
- **SKIP**: draft, red for a real reason, large

## 4. Report

One table: PR | title | category | reason (one line) | evidence command.
Then STOP. Merge nothing.

## 5. After approval

- Merge one at a time: `gh pr merge <N> --squash --admin`. Re-fetch head SHA and
  threads immediately before each merge.
- Strict policy: after each merge, update the next PR and wait for green CI.
- Run `collect_changelog.py` only after the last merge (it deletes the fragments).
- Finally remove the worktrees and sync the GitLab mirror with a proven fast-forward
  (see CLAUDE.md "Source Control & Hosting").
