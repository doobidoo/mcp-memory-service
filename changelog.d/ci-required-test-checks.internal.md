- **CI starts on every pull request, so the test jobs can be required checks.** `ci.yml`
  no longer carries `paths-ignore` on `pull_request`; a new `changes` job runs
  `is_docs_only()` from `scripts/pr/pre_pr_check.sh` over the PR diff, and the test
  jobs skip when it reports docs-only. A skipped job reports success to a required
  check, which a workflow that never starts cannot. Until now the only required check
  was CodeQL's `Analyze Python Code`, so auto-merge did not wait for the tests: #1416
  merged while `Tests with ML Extras` was still running. If the classifier itself fails,
  the tests run rather than skip. Push to main keeps `paths-ignore`.
