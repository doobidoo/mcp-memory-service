# Version Management - Release Workflow

## ⚠️ CRITICAL: Always Use github-release-manager Agent

**NEVER do manual releases** (major, minor, patch, or hotfixes). Manual workflows miss steps and are error-prone.

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
4. **PR & Merge**: Create PR, merge with `--admin --squash`
5. **Tag**: Create annotated tag `vX.Y.Z`
6. **GitHub Release**: Create release from tag
7. **Sync**: Release branch auto-deleted, main stays current

### Benefits:
- `main` = active development (current work)
- Release branches only when needed
- No permanent `develop` branch to maintain
- Clear separation of release preparation

## Five-File Version Bump Procedure

1. Update `src/mcp_memory_service/_version.py` (`__version__ = "X.Y.Z"`)
2. Update `pyproject.toml` (line ~7: `version = "X.Y.Z"`)
3. Update `README.md` (Latest Release section)
4. Update `CHANGELOG.md` (convert [Unreleased] to [X.Y.Z] with date)
5. Run `uv lock` to update dependency lock file
6. Commit all files together

## Release Commands

```bash
# Check release status
gh release list --limit 5
git log <last-tag>..HEAD --oneline

# Use the agent for releases
# The agent handles: version bump, CHANGELOG, PR, merge, tag, release

# Admin merge (branch protection bypass)
gh pr merge <PR-NUMBER> --admin --squash --delete-branch
```

## Branch Protection Setup

Branch protection configured to allow admin bypass:
- `enforce_admins: false` - Admin can use `--admin` flag
- Required reviews: 1 (for non-admins)
- CodeQL scanning enabled

## Hotfix Workflow (Critical Bugs)

- **Speed target**: 8-10 minutes from bug report to release
- **Process**: Fix → Test → Five-file bump → Commit → github-release-manager agent
- **Branch**: Can go directly to release branch if urgent
- **Issue management**: Post detailed root cause analysis

## Why Agent-First?

**Manual releases** (❌):
- Forgot README.md update
- Incomplete GitHub Release
- Missed workflow verification
- Version mismatch between files

**With agent** (✅):
- All files updated consistently
- Proper release created
- Complete documentation
- CHANGELOG properly formatted

**Lesson**: Always use agents, even for "simple" hotfixes
