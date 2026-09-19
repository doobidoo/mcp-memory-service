# /release — Full Release Cycle

Execute the complete release workflow for mcp-memory-service by delegating to the `github-release-manager` agent.

## What This Command Does

Spawn the `github-release-manager` agent to handle the entire release lifecycle:

1. **Pre-Release**: List open PRs, merge approved ones, verify CI green on main
2. **Version Bump**: Determine bump type (MAJOR/MINOR/PATCH) from PR labels, update `_version.py`, `pyproject.toml`, `README.md`, `uv.lock`
3. **Documentation**: Run `python3 scripts/release/collect_changelog.py` first — it merges the `changelog.d/` fragments every PR left behind into `[Unreleased]` and deletes them. Then retitle `[Unreleased]` in CHANGELOG.md to the new version and leave an empty `## [Unreleased]` heading above it; update README.md ("Latest Release"). Do **not** add a version line to CLAUDE.md — it was removed on 2026-09-05 in favour of a pointer to CHANGELOG.md, because it was the file that actually got forgotten.
4. **Landing Page**: Update `site/index.html` for MINOR/MAJOR releases only — version badge, What's New cards, test count. Deploys automatically to mcpmemory.services on merge (`.github/workflows/deploy-site.yml`)
5. **Tag**: Annotated tag on the merge commit, pushed **with git** and an explicit refspec:
   ```bash
   git tag -a vX.Y.Z -m "<summary>" <merge-sha>
   git push origin refs/tags/vX.Y.Z     # never --tags, never the forge API
   ```
   The push is what triggers `release.yml`. A tag created through the release API or the web form is not a push event, so nothing fires — that is how v11.8.1 was tagged and never published.
6. **Verify the artifacts, not the run**: `bash scripts/release/verify_artifacts.sh X.Y.Z` must exit 0 before the release object exists. It checks both PyPI distributions by version and all four Docker tags plus `latest` by digest. Never accept an HTTP 200, or a green run, as proof: v11.11.0 had two of three jobs green while `latest` still served the previous build.
7. **Release Creation**: Publish the release object from the tag, with notes and contributor recognition — last, after step 6 is green. A release with notes looks finished, which is what hid v11.8.1 for a day. Re-fetch the published release afterwards and quote the live notes back as evidence.
8. **Post-Release**: Fast-forward the GitLab mirror (prove the ancestor check first, never tags), clean up branches, close resolved issues.

## Rules

- **NEVER manually edit version files** — the agent synchronizes all files atomically
- **CI must be green** before any merge or release — stop if red
- **Read the review comments before merging**, not just the check buckets. `Greptile Review: pass` means the reviewer ran, not that it found nothing; its findings arrive as inline comments (`gh api repos/doobidoo/mcp-memory-service/pulls/<N>/comments`). No `--admin` bypass over unresolved feedback.
- **Landing page**: MINOR/MAJOR only, skip for PATCH
- **Save release summary** to MCP Memory with tags: `mcp-memory-service`, `release`, `v<VERSION>`

The full procedure, with the incidents behind each rule, is
[`.claude/directives/version-management.md`](../directives/version-management.md). This
file is the entry point, not a second copy — duplicating the procedure is what let both
release agents rot until 2026-09-06.

## Invocation

Spawn `github-release-manager` agent with context about what triggered the release (merged PRs, completed features, fixed issues).

If `/release` does not reach this file, a same-named skill is shadowing it — a user-level
skill wins over a repo command. On 2026-09-19 `~/.claude/skills/release/SKILL.md` did
exactly that, so v11.13.0 was prepared by hand instead of by the agent. Start the agent
explicitly in that case (`subagent_type: github-release-manager`) and remove the
shadowing skill.
