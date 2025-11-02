**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: forge-global-health-check
description: Sanity check Forge orchestration and MCP access (domain-agnostic)

---

# Forge Global Health Check

## Goals
- Verify Genie MCP tools are available
- Verify core agents (forge, wish, review) are resolvable
- Optional: verify a no-op wish→review roundtrip without domain specifics

## Steps
1) List agents
```
mcp__genie__list_agents
```
Expect `forge`, `wish`, `review` to be present.

2) Start noop wish (optional)
```
mcp__genie__run agent="wish" prompt="Create a placeholder wish (no code), context: @.genie/product/mission.md"
```
Record wish path if created.

3) Dry review (optional)
```
mcp__genie__run agent="review" prompt="Review placeholder wish (no code)"
```

## Evidence
- Save outputs to `.genie/qa/evidence/forge-health-<timestamp>.txt`

