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
# both distributions, not just the main one
curl -s https://pypi.org/pypi/mcp-memory-service/json      | jq -r .info.version
curl -s https://pypi.org/pypi/mcp-memory-service-lite/json | jq -r .info.version

# all four image tags: X.Y.Z, X.Y.Z-slim, X.Y, X.Y-slim
curl -s -o /dev/null -w '%{http_code}\n' \
  https://hub.docker.com/v2/repositories/doobidoo/mcp-memory-service/tags/X.Y.Z
```

PyPI's JSON endpoint lags the upload by a minute or two, so a stale version there right
after a green publish job is cache, not failure — re-check before concluding anything.

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
4. `CHANGELOG.md` (convert [Unreleased] to [X.Y.Z] with date)
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

`main` carries a ruleset named `ProtectMain` that requires changes to arrive through a
pull request. It required one approving review until 2026-09-05; that was dropped to
zero, because the author cannot approve their own PR and a solo maintainer is always the
author, so every PR needed an admin bypass. The PR requirement itself stays:

- Never commit straight to `main`; branch first.
- Merge through a PR, squash.
- Verify CI is green on the PR before merging.
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
