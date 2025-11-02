# QA - Genie Project Context
**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
## Validation Commands

### CLI Operations
```bash
# Help system
npx automagik-genie --help

# Version display
npx automagik-genie --version

# Init command
npx automagik-genie init code
npx automagik-genie init create
npx automagik-genie init  # Test error handling
```

### MCP Operations
```bash
# Agent catalog
mcp__genie__list_agents

# Agent execution
mcp__genie__run with agent="plan" and prompt="test prompt"
mcp__genie__run with agent="orchestrator" and prompt="Mode: analyze. Test."

# Session management
mcp__genie__list_sessions
mcp__genie__resume with sessionId="<id>" and prompt="follow-up"
mcp__genie__view with sessionId="<id>" and full=false
mcp__genie__view with sessionId="<id>" and full=true
mcp__genie__stop with sessionId="<id>"

# Error scenarios
mcp__genie__run with agent="nonexistent" and prompt="test"
mcp__genie__resume with sessionId="invalid" and prompt="test"
mcp__genie__view with sessionId="nonexistent"
mcp__genie__stop with sessionId="invalid"
```

### Build & Test
```bash
# Full test suite
pnpm test

# Specific suites
pnpm test:unit
pnpm test:integration

# Type checking
pnpm run check

# Build
pnpm run build

# Lint
pnpm run lint
```

---

## Evidence Paths

**Base:** `.genie/qa/evidence/`

**Naming:**
- Commands: `cmd-<name>-<YYYYMMDDHHmm>.txt`
- Errors: `error-<scenario>-<YYYYMMDDHHmm>.txt`
- Screenshots: `screenshot-<view>-<YYYYMMDDHHmm>.png`
- Logs: `<scenario>.log`

---

## Performance Baselines

| Operation | Target | Baseline | Last Measured |
|-----------|--------|----------|---------------|
| mcp__genie__list_agents | <100ms | 85ms | 2025-10-16 |
| mcp__genie__list_sessions | <100ms | TBD | Never |
| pnpm test | <60s | 45s | 2025-10-15 |
| pnpm run build | <120s | 95s | 2025-10-15 |

---

## Environment Setup

**Required:**
- Node.js 18+
- pnpm 8+
- Git configured

**For MCP Testing:**
- Claude Code with MCP enabled
- .genie/ directory initialized

**For CLI Testing:**
- Package installed globally or via npx
- Write permissions in test directory

---

## Domain-Specific Scenarios

### Agent System Validation
- Core agents loaded from `.genie/agents/`
- Project-specific notes embedded in agents/skills (no `custom/` directory)
- Mode selection via orchestrator (18 modes)
- Agent sessions persistence

### Template System
- Code template initialization
- Create template initialization
- Template files deployed correctly
- Custom directories excluded from templates

### MCP Server
- Server starts without errors
- Tools registered correctly
- Session state persists across calls
- Background execution with bypassPermissions

### Workflow Integration
- Plan → Wish → Forge → Review flow
- Learning agent updates documentation
- Git agent handles commits/PRs/issues
- Release agent orchestrates publishing

---

## Critical Edge Cases

### Permission Handling
- Background agents bypass prompts (permissionMode: bypassPermissions)
- Foreground agents respect approval gates (permissionMode: default)
- File write operations with proper permissions

### Session State
- Empty sessions.json ([] array)
- Corrupted sessions.json (invalid JSON)
- Missing .genie/state/ directory
- Concurrent session creation

### Error Messages
- Clear agent not found errors (list alternatives)
- Missing template errors (show available)
- Invalid session ID (suggest list_sessions)
- File system errors (permissions, missing dirs)

---

## Project-Specific Success Criteria

**From wishes:**
- Check @.genie/wishes/<slug>/<slug>-wish.md for validation commands
- Extract success criteria from spec contracts
- Map to checklist items or create new ones via learn agent

**From forge plans:**
- Validation hooks per execution group
- Evidence paths specified in plan
- Approval checkpoints to validate

**From custom workflows:**
- Natural routing triggers (routing.md patterns)
- Agent session collaboration flows
- Learning integration (auto-updates)
