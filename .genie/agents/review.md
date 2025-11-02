**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: review
description: Core Review orchestrator (delegates to domain-specific review agents)
genie:
  executor: claude
  background: true
---

# Review (Core)

Purpose: Validate outcomes against acceptance criteria/evaluation matrix. Delegate to domain reviews.

## Delegation
- Software delivery: `code/review` (loads report template, scores matrix)

## Templates (Canonical)
@.genie/product/templates/review-report-template.md

## Pattern
```
mcp__genie__run agent="code/review" prompt="Review @.genie/wishes/<slug>/<slug>-wish.md with matrix scoring."
```

