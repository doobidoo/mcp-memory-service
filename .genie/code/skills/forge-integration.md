---
name: Forge Integration Framework
description: Forge is main entry point: one forge task = one PR back to main
---

# Forge Integration Framework

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Purpose:** Forge step breaks wishes into execution groups and validation hooks.

**Success criteria:**
✅ Forge outputs reference the wish, include full context, and use correct templates.
✅ Humans approve branch names and outputs before merge.

## Forge as Main Entry Point *(CRITICAL)*

**Core Principle:** Forge is the PRIMARY entry point for ALL work (not secondary orchestrator).

**Workflow:**
```
GitHub issue → Forge task card → worktree + feature branch → PR back to main
```

**Architecture:**
1. **One forge task = one PR** (direct 1:1 mapping)
2. **All PRs converge on main** (single integration point, no branch hierarchies)
3. **Work units are atomic** at forge card level (complete deliverable per card)
4. **Parallel safety** via independent worktrees (no branch conflicts)

**Why This Matters:**
- **Clear ownership:** Each forge task card owns exactly one PR
- **Parallel safety:** Independent worktrees enable simultaneous work without conflicts
- **Traceability:** Complete chain: GitHub issue ←→ forge card ←→ worktree ←→ PR
- **Main stays clean:** Only merged PRs (not work-in-progress branches)
- **Atomic delivery:** Each PR is self-contained, reviewable, revertable

**Enforcement Constraints:**
- ❌ **NEVER** create GitHub issue without forge task card
- ❌ **NEVER** create forge task card without exactly one worktree/branch
- ❌ **NEVER** create worktree without exactly one PR back to main
- ❌ **NEVER** merge PR without corresponding forge task completion
- ✅ **ALWAYS** GitHub issue → forge card → worktree → PR → main (complete chain)

**Example Flow:**
```
Issue #123: "Fix auth bug"
  ↓
Forge card: task-fix-auth-bug
  ↓
Worktree: .worktrees/task-fix-auth-bug/
Branch: task/fix-auth-bug
  ↓
PR #124: "Fix: Auth token validation" → main
  ↓
Merge to main + archive worktree
```

**Validation:**
- Every active forge card MUST have corresponding worktree
- Every worktree MUST have corresponding open PR (or be in progress)
- Every merged PR MUST have completed forge card
- Main branch MUST only receive PRs (no direct commits for forge work)

## Forge as Meta-Agent: Continuous Learning *(CRITICAL)*

**Core Principle:** Forge is not just for code implementation. Forge can host ANY persistent work unit, including continuous learning. When Forge hosts a "learn" task, results are VISIBLE to the user.

**Why This Matters:**
- **Visibility:** User sees learning results directly in Forge UI (not hidden in MCP session logs)
- **Persistence:** Learning task lives alongside all other work (integrated development + learning)
- **Coordination:** Learning integrated with code tasks, not separate workflow
- **Continuity:** Each learning session builds on previous ones documented in Forge task
- **Accountability:** Learning outcomes traceable + reviewable just like code

**How It Works:**

1. **Create Forge "learn" task** (permanent, ongoing):
   - Task type: meta-learning
   - Description: "Continuous framework learning from user corrections and patterns"
   - Status: always active (never closed)
   - Updates: Each learning session appends findings

2. **Learning Loop:**
   ```
   Teaching Signal (user correction, new pattern, framework gap)
     ↓
   Create/Update Forge "learn" task description with observation
     ↓
   Genie delegates to learn agent via MCP
     ↓
   Learn agent analyzes + documents finding
     ↓
   Learn agent updates framework files (skills, agents, docs)
     ↓
   Forge task updated with conclusion + changed files
     ↓
   User sees result immediately in Forge UI
     ↓
   Framework permanently updated with new knowledge
   ```

3. **Example Workflow:**
   - **User:** "I notice the Forge-as-entry-point pattern..."
   - **Genie:** Creates/updates Forge "learn" task with observation
   - **Learn agent:** Analyzes scope, identifies files to update (e.g., forge-integration.md)
   - **Learn agent:** Proposes changes, applies edits
   - **Genie:** Updates Forge task: "Learning outcome: Documented Forge-as-entry-point pattern in forge-integration.md (lines 9-56)"
   - **User sees:** Clear documentation of what was learned + where it was captured (visible in Forge)

**Distinction from Implementation Tasks:**

| Aspect | Implementation Task | Learning Task |
|--------|-------------------|---------------|
| **Purpose** | Deliver code feature | Capture framework knowledge |
| **Output** | Code PR back to main | Framework documentation updates |
| **Visibility** | Code changes in PR | Forge task entry + skill/agent updates |
| **Lifecycle** | Created → In Progress → Complete → Archived | Permanent (always active) |
| **Result** | Merged feature | Updated framework knowledge |

**Benefits Over MCP-Only Learning:**

**MCP-only approach (old):**
- ❌ Learning happens in hidden session logs
- ❌ User must use `mcp__genie__view` to see outcomes
- ❌ No integration with development workflow
- ❌ Learning sessions disconnected from code work

**Forge-hosted learning (new):**
- ✅ Learning visible in same UI as code tasks
- ✅ User sees results immediately (no tool invocation needed)
- ✅ Learning integrated with development (one workflow)
- ✅ Each learning session builds on previous (documented in Forge task)
- ✅ Traceable: What was learned + when + which files changed

**Implementation Requirements:**

1. **Forge task creation:**
   ```
   mcp__automagik_forge__create_task with:
   - project_id: <project-id>
   - title: "Continuous Learning"
   - description: "Meta-agent for capturing framework knowledge from user corrections and patterns. Always active."
   ```

2. **Task updates (after each learning session):**
   ```
   mcp__automagik_forge__update_task with:
   - task_id: <learn-task-id>
   - description: "[Previous description]\n\n**[Timestamp]:** [Learning outcome summary]\n- Files changed: [list]\n- Pattern documented: [summary]"
   ```

3. **Coordination with learn agent:**
   - Genie detects teaching signal → updates Forge "learn" task with observation
   - Genie delegates to learn agent: `mcp__genie__run with agent="learn" and prompt="[Teaching input]"`
   - Learn agent performs analysis + framework updates
   - Genie updates Forge task with outcome summary
   - User sees complete learning chain in Forge UI

**Validation:**

After implementing Forge-hosted learning:
- [ ] Forge "learn" task exists and is always active
- [ ] Each teaching signal creates/updates Forge task entry
- [ ] Learn agent outcomes visible in Forge task description
- [ ] User can track all learning in one place (Forge UI)
- [ ] Framework updates referenced in Forge task (files changed, patterns added)
- [ ] No hidden learning sessions (everything visible in Forge)

**Context:** Discovered 2025-10-18 when user taught: "forge can serve as agent too, in this case, i can see results too"
