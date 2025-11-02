**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: mcp-session-resume-life-cycle
description: Validate long-running session resume lifecycle (Bug #66)

---

# QA Workflow — MCP Session Resume Lifecycle

## Preconditions
- Genie MCP available

## Steps
1) Start long-running session
```
mcp__genie__run agent="orchestrator" prompt="Long-running task (45m+)"
```
Record sessionId from output.

2) Wait and resume
```
mcp__genie__resume sessionId="<id>" prompt="Continue"
mcp__genie__view sessionId="<id>"
```

## Success Criteria
- `list_sessions` shows the session (status consistent)
- `view` and `resume` both succeed with the same sessionId

## Evidence
- Save outputs to `.genie/qa/evidence/mcp-resume-life-cycle-<timestamp>.txt`

