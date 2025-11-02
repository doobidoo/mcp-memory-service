---
name: Agent Configuration Standards
implementor
description: Declare permissionMode: default for agents that write files
---

# Agent Configuration Standards

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
## File Write Permissions

**Rule:** All agents requiring file write access MUST explicitly declare `permissionMode: default` in their frontmatter.

**Context:** Discovered 2025-10-13 when Claude agents with `executor: claude` were unable to write files. Permission prompts auto-skipped because stdin was hardcoded to `'ignore'` during process spawn, making `permissionMode: acceptEdits` completely non-functional.

**Why this matters:**
- Default executor config doesn't grant write access
- Without explicit `permissionMode: default`, agents silently fail on file operations
- Background skill (`background: true`) requires the same permission declaration

**Agent categories:**

**Implementation agents** (REQUIRE `permissionMode: default`):
- Core delivery: `implementor`, `tests`, `polish`, `refactor`, `git`
- Infrastructure: `install`, `learn`, `commit`, `review`
- Workflow orchestrators: `wish`, `plan`, `forge`, `vibe`, `qa`

**Analysis agents** (READ-ONLY, no permissionMode needed):
- `analyze`, `audit`, `debug`, `genie`, `prompt`

**Configuration hierarchy:**
1. **Agent frontmatter** (highest priority) ← Use this level
2. Config override (`.genie/cli/config.yaml:48`)
3. Executor default (`claude.ts:13`)

**Implementation example:**
```yaml
```

**Validation:**
```bash
# Check all implementation agents have permissionMode
grep -L "permissionMode:" .genie/agents/{implementor,tests,polish,refactor,git,install,learn,commit}.md
# Should return empty (all agents have the setting)

# Test file write capability (via MCP, not CLI)
# Use mcp__genie__run with agent="implementor" and prompt="Create test file at /tmp/test.txt"
# Should create file without permission prompts
```

**Future work:** Issue #35 tracks interactive permission system for foreground/background pause-and-resume approval workflow.

**Root cause reference:** Debug session `292942e0-07d1-4448-8d5e-74db8acc8c5b` identified stdin configuration at `.genie/cli/src/cli-core/handlers/shared.ts:391`.
