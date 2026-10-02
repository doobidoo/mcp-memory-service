- **CI starts on every pull request, so the test jobs can be required checks.** `ci.yml`
  no longer carries `paths-ignore` on `pull_request`; a new `changes` job runs
  `is_docs_only()` from the base commit's `scripts/pr/pre_pr_check.sh` over the PR diff (a PR cannot redefine what skips its own tests), and the test
  jobs skip when it reports docs-only. A skipped job reports success to a required
  check, which a workflow that never starts cannot. Until now the only required check
  was CodeQL's `Analyze Python Code`, so auto-merge did not wait for the tests: #1416
  merged while `Tests with ML Extras` was still running. If the classifier itself fails,
  the tests run rather than skip. Push to main keeps `paths-ignore`.
- **The pre-PR gate runs only the tests a change can reach.** `scripts/pr/lib/select_tests.py`
  maps the PR diff to test targets: changed test files, the test files that import a
  changed `src/` module, and its `tests/<subpackage>/`. Dependency, pytest-config and
  conftest changes still run the full suite, as does a `src/` module nothing names. A
  change no Python test can reach runs none. CI's `Tests + Coverage` still runs the full
  suite as a required check, and `PRE_PR_FULL_SUITE=1` runs it locally. On this branch
  the gate went from more than seven minutes to 41 seconds.
