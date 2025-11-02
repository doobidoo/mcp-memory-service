**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: session-running-stuck
description: Verify running sessions transition to completed/abandoned (Bug #92)

---

# QA Workflow — Session Running Stuck

## Steps
1) List sessions and find old running entries: `mcp__genie__list_sessions`
2) Inspect logs under `.genie/state/agents/logs/` for those sessions
3) Confirm actual status (completed/abandoned)

## Success Criteria
- Stale "running" sessions reconcile to completed/abandoned
- If not, file bug and attach evidence

## Evidence
- Save findings to `.genie/qa/evidence/session-running-stuck-<timestamp>.txt`

