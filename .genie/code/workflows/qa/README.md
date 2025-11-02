**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

# QA Workflows (Vertical Growth)

Each QA item is defined as a small workflow. This lets us grow vertically by adding independent, focused scenarios that agents can run.

## Structure
- One file per QA scenario under `code/workflows/qa/`
- Keep scenarios atomic; define command(s), expected outputs, and evidence paths

## Example fields
```
name: agent-listing-latency
description: Validate mcp__genie__list_agents under <100ms
evidence: .genie/qa/evidence/cmd-list-agents-<timestamp>.txt
command: time mcp__genie__list_agents
success: '<100ms wall time'
```

