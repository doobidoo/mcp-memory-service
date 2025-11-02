**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: session-id-collision
description: Ensure unique, immutable session IDs (Bug #102)

---

# QA Workflow — Session ID Collision

## Steps
1) List sessions: `mcp__genie__list_sessions`
2) Check for duplicate IDs across agents
3) For any duplicate, view both: `mcp__genie__view sessionId="<id>"`

## Success Criteria
- No duplicate session IDs across agents
- Each ID maps to exactly one agent/session

## Evidence
- Save findings to `.genie/qa/evidence/session-id-collision-<timestamp>.txt`

