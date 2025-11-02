**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: wish
description: Core Wish orchestrator (delegates to domain-specific wish agents)
genie:
  executor: claude
  background: true
---

# Wish (Core)

Purpose: Start wish authoring from any context. Delegate to the right domain wish agent.

## Delegation
- Research/content: `create/wish` (planning, blueprinting)
- Software delivery: `code/workflows/wish.md` (or `code/wish` agent if defined)

## Templates (Canonical)
@.genie/product/templates/wish-template.md

## Pattern
```
mcp__genie__run agent="create/wish" prompt="Author wish for <intent>. Context: @.genie/product/mission.md @.genie/product/roadmap.md."
```

