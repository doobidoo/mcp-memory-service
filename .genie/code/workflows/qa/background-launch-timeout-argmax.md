**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: background-launch-timeout-argmax
description: Validate background launch (no timeout) and large prompt handling (ARG_MAX) (Bug #104)

---

# QA Workflow — Background Launch Timeout / ARG_MAX

## Steps
1) Start background session with long prompt
```
mcp__genie__run agent="git" prompt="<long content>"
```
2) List sessions and view recent logs
```
mcp__genie__list_sessions
mcp__genie__view sessionId="<id>"
```

## Success Criteria
- No background launch timeout
- Long prompt handled (no ARG_MAX failure in executor command construction)

## Evidence
- Save output/log refs to `.genie/qa/evidence/background-argmax-<timestamp>.txt`

