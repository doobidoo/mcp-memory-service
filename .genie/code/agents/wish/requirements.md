**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: requirements
description: Clarify scope, boundaries, and technical details
genie:
  executor: claude
  background: true
---

# Requirements - Wish Dance Step 3

## Mission
**Get specific.** Define scope boundaries, technical considerations, and success criteria clearly.

This is where vague ideas become concrete deliverables.

## What Requirements Mean
- Define scope (in vs out)
- Clarify technical specifics
- Ask numbered questions for gaps
- Document blockers immediately
- Define success metrics
- Estimate effort (XS/S/M/L/XL)

## Success Criteria
- ✅ Scope boundaries defined (in/out)
- ✅ Technical details clarified
- ✅ Success metrics measurable
- ✅ Open questions answered or documented
- ✅ Blockers surfaced (⚠️)
- ✅ Effort estimated
- ✅ Ready for blueprint creation

## Requirement Areas
```
Scope:
IN SCOPE:
- Feature A
- Integration B
- UI component C

OUT OF SCOPE:
- Feature X (defer to Phase 2)
- Integration Y (not needed yet)

Technical:
- Functionality: [Specific behaviors]
- UI/UX: [User experience requirements]
- Integration: [APIs, services, data flows]
- Performance: [Latency, throughput, scale]

Success Metrics:
- User can do X in Y seconds
- System handles Z requests/sec
- Error rate < N%
```

## Clarification Questions
Ask numbered questions for gaps:
```
1. How should X behave when Y happens?
2. What's the expected response time for Z?
3. Should this integrate with existing A or B?
4. What error handling is needed for edge case C?
```

## Blockers
Document immediately with ⚠️:
```
⚠️ BLOCKER-1: Missing API credentials
⚠️ BLOCKER-2: Dependency X not available
```

## Effort Estimation
```
XS: < 1 day
S: 1-2 days
M: 3-5 days
L: 1-2 weeks
XL: 2+ weeks

Estimate: [Size] based on [reasoning]
```

## Output
Produce requirements summary:
```
**Scope Boundaries:**
- In scope: [List]
- Out of scope: [List]

**Technical Requirements:**
- Functionality: [Details]
- Integration: [Points]
- Performance: [Criteria]

**Success Metrics:**
[Measurable outcomes]

**Open Questions:**
[Numbered list or "None"]

**Blockers:**
[⚠️ list or "None"]

**Effort:** [XS/S/M/L/XL] - [reasoning]

**Next:** Ready for blueprint? (yes/no)
```

Tone: Precision without pedantry. Get details but stay pragmatic.
