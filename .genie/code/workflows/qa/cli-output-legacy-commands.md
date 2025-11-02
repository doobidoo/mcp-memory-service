**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: cli-output-legacy-commands
description: Ensure session output references MCP tools, not ./genie (Bug #89)

---

# QA Workflow — CLI Output Legacy Commands

## Steps
1) Start any agent
```
mcp__genie__run agent="plan" prompt="noop"
```
2) Inspect output for guidance lines

## Success Criteria
- No occurrences of `./genie <cmd>` in output
- Preferred references:
  - `mcp__genie__resume with sessionId="<id>"`
  - `mcp__genie__view with sessionId="<id>"`
  - `mcp__genie__list_sessions`
  - Or `npx automagik-genie` for CLI

## Evidence
- Save output to `.genie/qa/evidence/cli-output-legacy-<timestamp>.txt`

