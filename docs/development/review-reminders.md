# Review reminders

The daily `Triage Digest` workflow also checks unresolved PR review threads.
It runs at 04:00 UTC. It reads GitHub metadata only and never executes PR code.
The existing `ResolveThreads` ruleset already blocks merging with open threads.

- Draft PRs, bot PRs, and the maintainer's own PRs are excluded.
- A thread whose last comment comes from the PR author appears in the maintainer
  digest under **Review threads awaiting reviewer reply**. This is a routing
  heuristic: a reply does not prove that the feedback has been addressed.
- A thread whose last comment comes from someone else becomes eligible for a
  reminder after 48 hours without a reply in that thread. Both human and bot
  reviews count. Commits and unrelated PR comments do not count as thread replies.
- One reminder mentions the contributor and links up to 20 eligible threads.
  A count points to any remaining threads in Files changed.
- If eligible threads remain, one final reminder is allowed at least seven days
  after the first. The allowance is per PR, not per thread: new commits, new
  reviews, and resolved/reopened threads never restart it.
- Hidden markers on comments by `github-actions[bot]` or repository collaborators
  retain this history across workflow runs. Deleting those comments removes that
  history. The script never resolves threads, closes PRs, or applies code fixes.
- Outdated threads still count when unresolved.

The workflow serializes manual and scheduled runs, checks all API pages, and
rechecks PR state, review threads, and reminder history immediately before posting.
It uses `pull-requests: write` for reminders and `issues: write` for the digest.
It needs no additional secret, service, model, or dependency.

## Preview

Manual workflow runs default to **dry run**, covering both reminders and the
digest. Scheduled runs post reminders. Digest comments additionally require the
existing `TRIAGE_DIGEST_ISSUE` repository variable; without it the digest still
appears in the Actions run summary. A failed review scan fails the run but leaves
the existing digest step enabled, with an explicit warning in the summary.
The scan has a two-minute step deadline; the digest retains a separate
five-minute step budget inside a ten-minute job, so a slow scan cannot consume
the digest's time allowance.

Locally, with `gh` authenticated:

```bash
.venv/bin/python scripts/maintenance/github_review_reminders.py --report /tmp/review-report.json
.venv/bin/python scripts/maintenance/github_triage_digest.py --review-report /tmp/review-report.json
```

Only an explicit `--post` sends reminders. Avoid concurrent local posting while
the workflow runs; local processes do not share GitHub Actions concurrency locks.
