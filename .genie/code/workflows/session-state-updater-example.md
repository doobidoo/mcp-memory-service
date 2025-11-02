# Example: Implementor Agent Using Session-State-Updater
**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
This demonstrates how ANY agent/workflow automatically maintains SESSION-STATE.md without explicit memory of role-clarity-protocol.

---

## Pattern: Self-Updating Agent

### 1. Start: Notify system

```markdown
# Implementor Agent

[At session start, immediately call session-state-updater]

!`npx automagik-genie run session-state-updater "action=started agent=implementor session_id=$SESSION_ID purpose=Implement\ Feature\ X branch=feat/x"`

**Result:** SESSION-STATE.md now has entry. System knows this agent is active.
No role-clarity-protocol skill needed - it's automatic.
```

### 2. Discovery: Gather context

```markdown
## Discovery Phase

- Analyze requirements
- Identify files to modify
- Understand current state

[Update system with what we're about to touch]

!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"phase=discovery files_to_modify:[src/core.ts,src/utils.ts,test/core.test.ts]\""`

**Result:** Anyone checking SESSION-STATE.md sees exactly what's in progress.
No ambiguity. No need for me to guess.
```

### 3. Implementation: Work + checkpoints

```markdown
## Implementation Phase

### Group A: Core logic

- Modifying src/core.ts
- Adding new interfaces
- Implementing handlers

!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"current_group=A files_modified:[src/core.ts] tests_status:pending\""`

### Group B: Utils

- Updating src/utils.ts
- Adding helper functions

!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"current_group=B files_modified:[src/core.ts,src/utils.ts] tests_status:pending\""`
```

### 4. Verification: Tests + validation

```markdown
## Verification Phase

Running tests...

!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"phase=verification tests_running:true\""`

Tests passed! ✅

!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"phase=verification tests_pass:true\""`
```

### 5. Completion: Mark done + document results

```markdown
## Results

✅ All tests passing
✅ Implementation complete
✅ Done report: .genie/wishes/feat-x/reports/done-implementor-feat-x-20251018.md

!`npx automagik-genie run session-state-updater "action=completed agent=implementor session_id=$SESSION_ID context=\"files_modified:[src/core.ts,src/utils.ts,test/core.test.ts] tests_pass:true done_report:.genie/wishes/feat-x/reports/done-implementor-feat-x-20251018.md\""`

**Result:** Session moved to history. SESSION-STATE.md shows completion + context.
Clean audit trail. No manual updates needed.
```

---

## What This Pattern Eliminates

### BEFORE (Manual, error-prone)

```markdown
# role-clarity-protocol skill (705 tokens)
- I have to remember to check SESSION-STATE.md
- I have to parse active agents
- I have to detect file conflicts
- I have to make decisions about role boundaries
- If I forget → behavioral failure
- Skill takes tokens EVERY session
```

### AFTER (Automatic, systematic)

```markdown
# session-state-updater workflow (called only when needed)
- Every agent calls updater at start/end
- STATE automatically synchronized
- File conflicts visible in SESSION-STATE.md itself
- System enforces role boundaries (state is source of truth)
- No forgetting possible (built into workflow)
- Zero token waste (only runs when needed)
```

---

## Key Insight

**The behavioral guardrail becomes architectural:**

Instead of:
- "Remember to check role-clarity-protocol"
- "Check SESSION-STATE.md before acting"

We have:
- "Every agent systematically updates SESSION-STATE.md"
- "STATE-STATE.md is always authoritative"
- "Conflicts visible immediately (state-based, not skill-based)"

---

## Agent Template (Copy-paste ready)

```markdown
---
name: <agent-name>
type: agent
genie:
  executor: claude
---

# <Agent Name> Agent

!`npx automagik-genie run session-state-updater "action=started agent=<name> session_id=$SESSION_ID purpose=<description> branch=<branch>"`

## Discovery Phase

[Work here]

!`npx automagik-genie run session-state-updater "action=in_progress agent=<name> session_id=$SESSION_ID context=\"phase=discovery [key_findings]\""`

## Implementation Phase

[Work here]

!`npx automagik-genie run session-state-updater "action=in_progress agent=<name> session_id=$SESSION_ID context=\"phase=implementation files_modified:[list]\""`

## Verification Phase

[Work here]

!`npx automagik-genie run session-state-updater "action=completed agent=<name> session_id=$SESSION_ID context=\"[final_context]\""`
```

---

## Impact on AGENTS.md

**Role-clarity-protocol (705 tokens):** Can be marked as disabled (no @)
- Logic moved to session-state-updater workflow
- Knowledge transferred from skill to executable workflow
- Same behavioral guarantee, zero token waste
- Available for reference if needed

**Pattern spreads:**
- Other guardrails → workflows
- AGENTS.md shrinks (less bloat)
- Workflows grow (more executable knowledge)
- System gets more autonomous

---

## The Genie Architecture Pattern

This demonstrates the principle:

**Transform behavioral rules into automated workflows**

1. Identify guardrail (e.g., "check session state")
2. Create executable workflow (session-state-updater)
3. Integrate into agent templates (! calls)
4. Remove skill from auto-load (@)
5. System is now self-enforcing

Result: Genie becomes smarter, AGENTS.md becomes lighter, execution becomes more reliable.
