# Version Management - Release Workflow

## Always use the documented release workflow

**Never do manual releases** (major, minor, patch, or hotfixes). Manual workflows miss steps and are error-prone.

## Release Branch Workflow (Adopted 2026-01-27)

```
feature-branches → main (development)
                      ↓
              release/vX.Y.Z (preparation)
                      ↓
                  tag vX.Y.Z
                      ↓
              merge back to main
```

### Workflow Steps:

1. **Development**: All feature/fix branches merge to `main`
2. **Release Preparation**: Create `release/vX.Y.Z` branch from `main`
3. **Version Bump**: Update version files on release branch
4. **PR & Merge**: Create PR, wait for CI, merge it
5. **Tag**: Create the annotated tag locally and push it **with git**:
   ```bash
   git tag -a vX.Y.Z -m "<summary>" <merge-sha>
   git push origin refs/tags/vX.Y.Z     # explicit refspec, never --tags
   ```
   The **push** is what triggers `.github/workflows/release.yml` (PyPI main + lite,
   Docker Hub). Never create the tag through the forge API or the web UI — see the
   rule below.
6. **Verify the artifacts**, not just the run (see below)
7. **Release notes**: Publish a release object from the tag
8. **Sync**: Release branch deleted, main stays current

### The tag must be pushed with git

`release.yml` triggers on `push: tags: 'v*.*.*'`. A tag created inside the forge — via
the release API, or by filling in the tag field on the release form — is **not a push
event**, so nothing fires and the release publishes nothing at all.

This is not hypothetical. **v11.8.1 was tagged that way on 2026-08-22 and never
published.** Paging back through the workflow run history shows the full
Test → PyPI → Docker chain for v11.8.0 and v11.7.0 and no run whatsoever for v11.8.1.
PyPI stayed on 11.8.0 and `docker 11.8.1` returned 404 for a full day, with eight
fixes in it. It hid because a release object with notes looks exactly like a finished
release, and nothing anywhere says "no artifacts were built" — the green CI you
remember is the release PR's `ci.yml`, not `release.yml`.

### Verify the artifact, not the run

A release is done when it is installable, not when the tag exists. After the tag push,
check the publish endpoints directly:

```bash
bash scripts/release/verify_artifacts.sh X.Y.Z
```

It verifies both PyPI distributions by `.info.version`, all four image tags
(`X.Y.Z`, `X.Y.Z-slim`, `X.Y`, `X.Y-slim`) by tag name, and that `X.Y`, `X.Y-slim` and
`latest` resolve to the same **digests** as the exact version tags. Nothing in it treats
an HTTP status code as evidence. It is read-only, so re-running it is free and is the
intended way to confirm a `gh run rerun --failed` actually recovered the release.

Pending checks are polled until a deadline (`--timeout`, default 600s) because PyPI's
JSON endpoint lags the upload by a minute or two — a stale version there right after a
green publish job is cache, not failure. Docker Hub also rate-limits anonymous callers,
and a throttled response looks exactly like a missing tag, which is the other reason the
checks retry instead of concluding on a single probe.

Exit 0 means every artifact is present and consistent. Exit 1 prints each unmet check
with what was actually observed — do not create the release object until it is 0.

### A release can publish half of itself

v11.11.0 on 2026-09-05: `Test` and `Publish to PyPI (main + lite)` green, `Publish
Docker images` red at `docker login` with `unauthorized: incorrect username or
password`. Both secrets were set (masked as `***` in the log, so neither was empty);
the values were wrong. `DOCKER_PASSWORD` has to be a Docker Hub **access token**, not
the account password.

This looks nothing like the v11.8.1 failure above — there is a run, and two of three
jobs are green — but for Docker users the effect is the same, and worse in one respect:
`latest` kept pointing at the previous build, so the most-used tag served a version with
three open critical advisories and nothing about the tag said so.

Recover with a job re-run, never a dispatch:

```bash
gh run rerun <run-id> --failed
```

That repeats only the failed job and keeps `github.ref_name` at the tag, so the image
tags still derive correctly. A `workflow_dispatch` from `main` would push junk `main`
tags and clobber `latest`.

**Create the release object last**, after the artifacts are verified. A release with
notes looks finished, which is exactly what hid v11.8.1 for a day.

`release.yml` has a `workflow_dispatch` fallback, but it is **PyPI catch-up only**: the
Docker job derives its image tags from `github.ref_name`, so a manual dispatch from
`main` pushes junk `main` tags and clobbers `latest`.

### Benefits:
- `main` = active development (current work)
- Release branches only when needed
- No permanent `develop` branch to maintain
- Clear separation of release preparation

## Version Bump Procedure

Always bumped together, in one commit:

1. `src/mcp_memory_service/_version.py` (`__version__ = "X.Y.Z"`) — this is the canonical source
2. `pyproject.toml` (line ~7: `version = "X.Y.Z"`)
3. `README.md` (Latest Release section)
4. `CHANGELOG.md` — retitle `[Unreleased]` to `[X.Y.Z]` with the date, and leave a new
   empty `## [Unreleased]` heading above it. The entries move under the version; the
   heading stays, because the next PR adds its entry there.
5. `uv lock` to update the dependency lock file

Of those five, **only `_version.py` and `pyproject.toml` are covered by a CI gate.**

`CLAUDE.md` used to carry a "Current Version" line and was the one that actually got
forgotten: v11.8.2 shipped without it, nothing failed, and main announced the previous
version until someone noticed by eye. That line was removed on 2026-09-05 in favour of a
pointer to CHANGELOG.md, which is the real fix. Do not reintroduce it.

Conditional, and each one is enforced by a CI gate:

- `site/index.html` version strings — required whenever MAJOR.MINOR changes, exempt for
  PATCH (`version-drift-check`, `scripts/ci/check_versions.sh`)
- `claude-hooks/.claude-plugin/plugin.json` — required when anything under
  `claude-hooks/` changed since the last commit that moved the manifest version
  (`plugin-version-check`, `scripts/ci/check_plugin_version.sh`)

Do **not** hand-bump `pyproject-lite.toml`: the publish workflow force-syncs it from
`_version.py`, and a manual value there is what left the lite distribution stuck on an
old version once.

## Release Commands

```bash
# Last release, and what has accumulated since
git tag --list 'v*' --sort=-version:refname | head -1
git log <last-tag>..HEAD --oneline
```

`git describe --tags --abbrev=0` is the wrong tool here — it picks up the non-version
tag `archive/github-workflows-pre-codeberg`.

PR creation, review-comment retrieval, squash-merge, and the release object all run
against the GitHub REST API via `gh`.

## Merge Discipline

`main` carries a ruleset named `ProtectMain` (id 5097493) that requires changes to arrive
through a pull request. Read its live state rather than trusting this paragraph:

```bash
gh api repos/doobidoo/mcp-memory-service/rulesets/5097493 --jq '.rules[]|"\(.type): \(.parameters|tostring)"'
```

As of 2026-09-15 it carries two rules:

- `pull_request` with `required_approving_review_count: 1`. This paragraph previously
  said the count had been dropped to zero on 2026-09-05; it never was. The author
  cannot approve their own pull request — but Greptile can, and does: it posts an
  approving review when it finds nothing, which satisfies the requirement and puts the
  PR at `CLEAN`, mergeable without `--admin`. When it finds something it comments
  instead of approving, and the PR stays `BLOCKED` until a human approves or an admin
  bypasses. So the review gate is in practice "Greptile is happy, or someone looked",
  and reaching for `--admin` is how a PR with unread findings gets merged. That is
  exactly what happened to the five findings on the v11.12.0 release PRs. A
  contributor PR that Greptile has commented on, like filhocf's #1243, sits at
  `BLOCKED` for the same reason and wants a real review, not a bypass.
- `required_status_checks` with `strict_required_status_checks_policy: true` and one
  required context, `Analyze Python Code`. Strict means a branch has to be up to date
  with `main` before it can merge. Added on 2026-09-15 after three regressions in one
  week reached `main` through merges whose result CI had never run on: #1184 reverted
  the ownership guard from #1224 while refactoring on an older base, #1224's own
  commits were cherry-picked onto `main` carrying two already-red tests because
  `_run_background()` never passed the port to `_write_pid()`, and #1232 restored the
  `src.` imports #1238 had just removed. Until then the ruleset required no status
  check at all.

The required context is `Analyze Python Code` (`codeql.yml`) specifically because it is
the only job that runs on every pull request. Everything in `ci.yml` sits behind
`paths-ignore` for `docs/**`, root `*.md`, `.github/**/*.md`, `LICENSE`, `NOTICE` and
`.gitignore`, so on a documentation-only PR that workflow never starts — and a required
check that never reports blocks the PR permanently. Before requiring any `ci.yml` job,
that has to be solved.

Strict also has a documented precondition: it takes effect only while at least one
status check is required. Setting the flag with an empty check list changes nothing.

The rest of the discipline:

- Never commit straight to `main`; branch first.
- Merge through a PR, squash.
- Verify CI is green on the PR before merging, and check `gh run list --branch main`
  after it lands — a PR that was green on its own base can still break `main`.
- Read the review comments before merging, not just the check buckets. `Greptile
  Review: pass` in `gh pr checks` means the reviewer ran, not that it found nothing;
  its findings arrive as inline review comments and are invisible to that status.
  `gh api repos/doobidoo/mcp-memory-service/pulls/<N>/comments` lists them. All five
  findings it left on the v11.12.0 release PRs were valid and were merged over,
  including a stale `og:description` and a dropped `[Unreleased]` heading. A PR sitting
  at `BLOCKED` while its checks are green is the signal: Greptile declined to approve,
  which means it wrote something. Read that before reaching for `--admin`.
- Keep an empty `## [Unreleased]` heading above the new version when cutting a release
  (see the version-bump procedure above). v11.11.0 and every release before it kept
  one; v11.12.0 dropped it, which Greptile caught and the merge ignored.
- If several sessions share one checkout, isolate into a worktree first.

## Hotfix Workflow (Critical Bugs)

- **Speed target**: 8-10 minutes from bug report to release
- **Process**: Fix, test, version bump, commit, then the documented release workflow
- **Branch**: Can go directly to release branch if urgent
- **Issue management**: Post detailed root cause analysis

## Why Not By Hand

What manual releases have actually cost:

- Forgotten `README.md` update
- Incomplete release notes
- Publish pipeline never verified after the tag push
- Version mismatch between files
- **Real incident (v10.8.0, Feb 8, 2026)**: `_version.py` not updated, so the dashboard
  reported the wrong version

The workflow keeps every version file, the CHANGELOG, and the release notes in step,
which is exactly what goes wrong when the bump is done by hand. That holds for "simple"
hotfixes too.
