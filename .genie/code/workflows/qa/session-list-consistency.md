**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: session-list-consistency
description: Ensure sessions in STATE.md match list+store (Bug #91)

---

# QA Workflow — Session List Consistency

## Steps
1) Read documented sessions: `@.genie/SESSION-STATE.md`
2) List sessions: `mcp__genie__list_sessions`
3) Inspect store: `.genie/state/agents/sessions.json`

## Success Criteria
- Sessions present in STATE.md exist in list_sessions and sessions.json
- Completed sessions remain queryable

## Evidence
- Save diffs/notes to `.genie/qa/evidence/session-list-consistency-<timestamp>.txt`

