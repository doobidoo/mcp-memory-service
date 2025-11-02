**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: agent-listing-latency
description: Validate `mcp__genie__list_agents` returns in under 100ms

---

# QA Workflow — Agent Listing Latency

## Preconditions
- MCP server available (stdio or HTTP)

## Command
```
time mcp__genie__list_agents
```

## Success Criteria
- Wall time < 100ms

## Evidence
- Save output to: `.genie/qa/evidence/cmd-list-agents-<timestamp>.txt`

## Notes
- See checklist: @.genie/product/docs/qa-checklist.md

