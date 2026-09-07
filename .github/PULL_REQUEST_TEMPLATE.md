<!--
  Thanks for sending a patch. A few things that make review fast, none of them
  mandatory. Delete whatever does not apply.
-->

## What this changes

<!-- One or two sentences. What behaviour is different after this lands? -->

## Why

<!-- The problem, not the patch. If there is an issue, link it: Fixes #123 -->

## How it was verified

<!--
  What you actually ran, and what it printed. "Should work" is not a result.
  If you skipped something, say so; that is useful information, not a failing.
-->

A change under `src/` needs a test, and CI checks that the test is red without
the change (`tests-prove-fix`). Paste both runs:

```
# on main (or with your src/ change stashed): must fail
.venv/bin/pytest tests/<your test file> -q

# on this branch: must pass
.venv/bin/pytest tests/<your test file> -q
```

Drive the real code path: the sqlite-vec storage through the `temp_db_path`
fixture, the real registry, the real handler. A mock of the function under test
proves nothing about it, and a mock that mirrors the bug passes on both sides.
Mock only network and external services. See CONTRIBUTING.md, "Testing
Requirements".

<!--
  The pre-PR gate is: bash scripts/pr/pre_pr_check.sh
  It is not required from outside contributors, and a red gate on code you did
  not touch is not your problem. Say what you saw and we will sort it out.
-->

## Notes for the reviewer

<!--
  Anything that would take a reviewer a while to work out on their own. A design
  choice you were unsure about, a case you deliberately did not handle, a
  follow-up you would rather do separately.
-->

---

Security vulnerabilities do not belong in a pull request. Report them privately
through [Security Advisories](https://github.com/doobidoo/mcp-memory-service/security/advisories/new).
See [SECURITY.md](../SECURITY.md).

Contribution guide: [CONTRIBUTING.md](../CONTRIBUTING.md)
