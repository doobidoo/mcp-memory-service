**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: mcp-agent-start-failure
description: Validate agent start for large prompts (Bug #93)

---

# QA Workflow — MCP Agent Start Failure

## Steps
1) Start agent with long prompt (~500-700 lines)
```
mcp__genie__run agent="learn" prompt="<long discovery/implementation/verification content>"
```
2) Observe start outcome and session ID

## Success Criteria
- Agent starts; session ID captured; no "Command failed" errors

## Evidence
- Save output to `.genie/qa/evidence/mcp-agent-start-<timestamp>.txt`

