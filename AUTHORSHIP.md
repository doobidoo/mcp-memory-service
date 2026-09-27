# Authorship, Copyright, and Project Governance

This document states who authors this project, who holds copyright, and how
changes are reviewed before they reach `main`.

## Authorship and ownership

- **Author and maintainer:** Heinrich Krupp (`@doobidoo`), sole project owner
  and lead maintainer.
- Architecture, design decisions, review, and release management are performed
  by the maintainer.
- **Every change is human-reviewed before it lands.** No code reaches `main`
  without maintainer review and passing the automated quality and security
  gates described below.
- The project has **external contributors** and an active review process;
  contributions are evaluated on their merits and reviewed by the maintainer.

## Copyright and licensing

- The project is licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
- Copyright is held by the author, Heinrich Krupp, and by contributors for their
  respective contributions.
- Contributions are accepted under the terms of the project license
  (see [`CONTRIBUTING.md`](CONTRIBUTING.md)). Contributors are expected to have
  the right to license the code they submit.
- The copyright status of the codebase is clear and defined: original work by
  identifiable authors, released under an OSI-approved license.

## Safeguards against harmful code

Every change passes layered, automated safeguards before it can be merged,
specifically to catch insecure or harmful code:

- **Pre-commit gate** - complexity and security screening on every commit.
- **Mandatory pre-PR gate** - [`scripts/pr/pre_pr_check.sh`](scripts/pr/pre_pr_check.sh)
  blocks on security findings and on a health score below threshold, and
  includes a log-injection check.
- **Log-injection / ANSI-injection guard** - user-provided values are sanitized
  via `_sanitize_log_value()` before logging; path inputs are validated and
  confined to expected base directories.
- **Continuous integration** - GitHub Actions (`.github/workflows/`) run the
  test suite and checks on every pull request.
- **Test coverage** - approximately 2,400 automated tests across the codebase.
- **Supply-chain hygiene** - CI actions are referenced by fully qualified URL
  and pinned to a specific commit SHA.

## Summary

An original, human-authored and human-reviewed open-source project with clear
Apache-2.0 licensing and layered automated safeguards against harmful code.
