---
name: Publishing Protocol *(CRITICAL)*
description: Never publish directly; always delegate to the release agent
---

# Publishing Protocol *(CRITICAL)*

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**NEVER** execute `npm publish` or `gh release create` directly. **ALWAYS** delegate to release agent.

## Forbidden Actions

- ❌ `npm publish` (bypasses validation, GitHub release, audit trail)
- ❌ `gh release create` (direct command - let agent orchestrate)
- ❌ Manual version tagging without release agent
- ❌ Using `/release` slash command with arguments (incorrect invocation)

## Required Workflow

**If you ARE the release agent:**
- ✅ Execute workflow directly: run pre-flight checks, create GitHub release via `gh release create`, monitor Actions
- ❌ NEVER delegate to yourself or invoke `mcp__genie__run` with agent="release"

**If you are NOT the release agent (genie/planner/main):**
1. Commit code + version bump to main
2. Delegate to release agent: `mcp__genie__run with agent="release" and prompt="Create release for vX.Y.Z"`
3. Release agent validates, creates GitHub release, monitors npm publish
4. Provide release URL to user

## Why This Matters

- **Safety**: Pre-flight checks (clean git, tests pass, version valid)
- **Consistency**: Follows project workflow (GitHub Actions)
- **Audit trail**: All releases documented in GitHub
- **Rollback**: Structured process easier to revert

## Recent Violations

**2025-10-14:**
- Attempted `gh release create` manually (bypassed validation)
- Attempted `npm publish` directly (timed out, triggered background agent)
- Attempted `/release` with arguments instead of proper MCP invocation
- **Result**: Inconsistent state, manual cleanup required
- **Evidence**: Commits 0c6ef02, 30dce09, GitHub Actions runs 18506885592

**2025-10-17:**
- Session ~00:50Z: Recognized RC5 release work but attempted direct handling
- Failed to check routing matrix before acting on release request
- Acknowledged "I'm learning" but did NOT invoke learn agent for documentation
- **Result**: Routing pattern not propagated to framework
- **Evidence**: User teaching 2025-10-17

## Validation

When user says "publish" or "release", immediately check routing matrix and delegate to release agent via MCP. When user identifies routing failures, invoke learn agent immediately to document correction.
