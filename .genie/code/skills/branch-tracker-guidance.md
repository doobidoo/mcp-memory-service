---
name: Branch & Tracker Guidance
description: Use dedicated branches for medium/large changes, track IDs in wishes
---

# Branch & Tracker Guidance

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Branch strategy:**
- **Dedicated branch** (`feat/<wish-slug>`) for medium/large changes.
- **Existing branch** only with documented rationale (wish status log).
- **Micro-task** for tiny updates; track in wish status and commit advisory.

**Tracker management:**
- Tracker IDs (from forge execution output) should be logged in the wish markdown once assigned. Capture them immediately after forge reports IDs.

A common snippet:

```
### Tracking
- Forge task: FORGE-123
```
