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
- Claim state for every PR labelled `unclaimed` or `no-issue`: match it to one row of
  the table in CONTRIBUTING.md "How unclaimed PRs and stale claims are decided" and
  name the row. Evidence: the issue's assign/unassign events
  (`gh api repos/{owner}/{repo}/issues/<N>/events`), claim comments by others before the
  PR's `createdAt` (a `/claim` mangled by Git Bash into a path still counts), and other
  open PRs for the same issue (`gh pr list --search "<N>" --state open`).
- Stale claims: open issues claimed more than 3 days ago with no PR, draft or update
  since. A claim is an assignment, a `claim-pending` label, or a claim comment that was
  never confirmed; all three block later PRs, so list all three with the row they fall
  under (ask, or expire).

Only when a question needs a local answer: worktree under `../worktrees/pr-<N>`, run
`.venv/bin/pytest` scoped there. Never touch the main tree — several sessions share it.

## 3. Classify

- **MERGE**: green, no open threads, no conflicts, automated-review findings (Greptile
  summary and confidence) read, and for contributor PRs the diff reviewed (`/pr-review`)
- **QUICK-FIX**: under 10 minutes (rebase, answer a thread, approve a run) — name the fix
- **CLAIM**: unclaimed or no-issue; state the table row and the action it prescribes
  (assign author, close, ask, expire, `skip-issue`). Never merge a PR in this category
  before the claim is settled.
- **DECISION**: needs the maintainer's judgment
- **SKIP**: draft, red for a real reason, large

## 4. Report

One table: PR | title | category | reason (one line) | evidence command.
Then STOP. Merge nothing.

## 5. After approval

- Queue each approved PR: `gh pr merge <N> --auto --squash --match-head-commit <sha>`.
  GitHub merges it once checks, review and threads are all green. The Merge Train
  workflow (`.github/workflows/merge-train.yml`) updates the oldest queued PR that
  is behind main, one at a time, and turns auto-merge off with a comment on a
  conflict or a failed required check. Do not update queued branches by hand.
- Without the Merge Train App configured (the workflow run is skipped), merge one
  at a time: `gh pr merge <N> --squash --admin --match-head-commit <sha>`, re-fetch
  head SHA, threads and check-runs right before, then update the next PR and wait for
  green CI.
- Do not run `collect_changelog.py` — collecting fragments belongs to `/release`.
- Finally remove the worktrees and sync the GitLab mirror with a proven fast-forward
  (see CLAUDE.md "Source Control & Hosting").
