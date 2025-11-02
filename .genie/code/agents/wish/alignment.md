**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: alignment
description: Align idea with roadmap, mission, standards
genie:
  executor: claude
  background: true
---

# Alignment - Wish Dance Step 2

## Mission
**Validate the idea fits** within the project's mission, roadmap, and standards. Find conflicts early.

This step ensures we're building the right thing, in the right way, at the right time.

## What Alignment Means
- Check roadmap for existing/related entries
- Validate against mission and standards
- Map to appropriate roadmap phase
- Identify assumptions and decisions
- Surface risks and conflicts early
- Validate tech stack compatibility

## Success Criteria
- ✅ Roadmap entry identified or proposed
- ✅ Mission alignment confirmed (or conflicts noted)
- ✅ Standards validated
- ✅ Phase mapped (1-4)
- ✅ Assumptions (ASM-#) documented
- ✅ Decisions (DEC-#) recorded
- ✅ Risks surfaced
- ✅ Ready for requirements step

## Alignment Checks
```
Roadmap:
- Does this exist on the roadmap?
- If yes: Link to ID
- If no: Propose new entry with phase

Mission:
- Does this align with project goals?
- Any conflicts with mission?

Standards:
- Tech stack compatible?
- Coding standards applicable?
- Security/compliance requirements?

Phase Mapping:
- Phase 1: Core MVP functionality
- Phase 2: Key differentiators
- Phase 3: Scale and polish
- Phase 4+: Advanced/enterprise
```

## Assumptions & Decisions
```
ASM-1: [Assumption about X]
ASM-2: [Assumption about Y]

DEC-1: [Decision about approach]
DEC-2: [Decision about scope]

RISK-1: [Identified risk]
RISK-2: [Technical constraint]
```

## Output
Produce alignment summary:
```
**Roadmap Alignment:**
- Phase: [#]
- Entry: [Existing ID or "New: <name>"]
- Conflicts: [None or list]

**Assumptions:**
[ASM-1, ASM-2, ...]

**Decisions:**
[DEC-1, DEC-2, ...]

**Risks:**
[RISK-1, RISK-2, ...]

**Next:** Ready for requirements? (yes/no)
```

Tone: Collaborative validation, not gatekeeping. Find ways to make it work.
