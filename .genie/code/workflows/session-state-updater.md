**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: session-state-updater
type: workflow
genie:
  executor: claude
  background: false
---

# Session State Updater Workflow

**Purpose:** Systematically update SESSION-STATE.md, MASTER-PLAN.md, and USERCONTEXT.md from any agent or workflow.

**Called by:** ANY agent/workflow via `!`npx automagik-genie run session-state-updater "..."`

**Foreground mode:** No `--background` flag = blocks execution until complete

---

## Discovery Phase

### Input Parameters

This workflow accepts a prompt string with the following parameters (space-separated or key=value):

```
action=<started|in_progress|completed|waiting|paused>
agent=<agent-name>
session_id=<uuid>
purpose="<description>"
context="<optional-rich-data>"
parent_session="<optional-parent-uuid>"
branch="<optional-branch-name>"
```

**Examples:**
```
action=started agent=implementor session_id=abc123 purpose="Implement feature X" branch=feat/x

action=in_progress agent=implementor session_id=abc123 context="files_modified:[src/x.ts,src/y.ts]"

action=completed agent=implementor session_id=abc123 context="files_modified:[src/x.ts] tests_pass:true"
```

### Parse & Validate

- Extract all key=value pairs from prompt
- Validate `action` is one of: started, in_progress, completed, waiting, paused
- Validate `session_id` is non-empty UUID format
- Validate `agent` is recognized (implementor, tests, git, genie, learn, release, roadmap)
- Parse `context` as structured data if provided

---

## Implementation Phase

### Step 1: Read Current State

Read `.genie/SESSION-STATE.md` fully.

Parse:
- Active sessions section
- Session history section
- Any existing entry for this `session_id`

### Step 2: Update or Create Entry

**If session already exists:**
- Update status field
- Update timestamp (Last Updated)
- Append context data
- Preserve created timestamp

**If session is new:**
- Create entry in "Active Sessions" section
- Set:
  ```
  ### <Agent Name> - <Purpose>
  **Session ID:** `<session_id>`
  **Started:** <ISO timestamp>
  **Status:** <action>
  **Agent:** <agent-name>
  **Purpose:** <purpose>
  **Context:** <context>
  **Next:** [to be filled]
  ```

### Step 3: Handle State Transitions

**started → active:**
- Add to "Active Sessions"
- Record start timestamp

**in_progress → active:**
- Update status
- Keep in "Active Sessions"

**completed → history:**
- Move from "Active Sessions" to "Session History (Recent)"
- Add completion timestamp
- Preserve all context

**waiting → paused:**
- Keep in "Active Sessions"
- Mark waiting status
- Document reason in context

### Step 4: Update Parent/Child Relationships

If `parent_session` provided:
- Find parent entry
- Add this session to parent's "Children:" list (if not already present)
- Mark this entry with "Parent:" reference

### Step 5: Write Back

1. Format updated SESSION-STATE.md
2. Write to `.genie/SESSION-STATE.md`
3. Git add the file
4. Return success + summary

---

## Verification Phase

### Validation

- [ ] SESSION-STATE.md is valid markdown
- [ ] Session entry created/updated correctly
- [ ] Parent-child relationships are bidirectional
- [ ] Timestamps are ISO format (YYYY-MM-DD HH:MM UTC)
- [ ] No duplicate session IDs
- [ ] Entry matches requested action

### Output

Return JSON response:
```json
{
  "status": "success",
  "action": "started|updated|completed",
  "session_id": "<uuid>",
  "agent": "<agent-name>",
  "updated_files": ["SESSION-STATE.md"],
  "timestamp": "<ISO timestamp>",
  "message": "Session state updated successfully"
}
```

On failure:
```json
{
  "status": "failed",
  "error": "<error message>",
  "details": "<debug info>"
}
```

---

## Usage Pattern in Agents

### At Agent Start

```markdown
# Implementor Agent

!`npx automagik-genie run session-state-updater "action=started agent=implementor session_id=$SESSION_ID purpose=Implement\ Feature\ X branch=feat/x"`

## Work happens here...
```

### Mid-work Update

```markdown
!`npx automagik-genie run session-state-updater "action=in_progress agent=implementor session_id=$SESSION_ID context=\"files_modified:[src/core.ts,src/utils.ts] tests_pass:false\""`
```

### On Completion

```markdown
## Results

!`npx automagik-genie run session-state-updater "action=completed agent=implementor session_id=$SESSION_ID context=\"files_modified:[src/core.ts,src/utils.ts,test/core.test.ts] tests_pass:true done_report:.genie/wishes/feat-x/reports/done-implementor.md\""`
```

---

## Benefits

**Eliminates:**
- Manual SESSION-STATE.md updates
- Stale session tracking
- role-clarity-protocol guardrails (state always current)
- Documentation bloat (knowledge transferred here)

**Enables:**
- Automatic state synchronization
- Rich context capture at runtime
- Parent-child workflow tracking
- Complete session history
- Foreground blocking (forces discipline)

---

## Technical Notes

- **Foreground mode:** No `--background` flag ensures execution waits
- **Markdown integration:** Called via `!` syntax from any agent/workflow
- **Idempotent:** Safe to call multiple times (updates same entry)
- **Transactional:** Entire state file updated atomically
- **Git-aware:** Auto-stages updated file

---

## Future Extensions

- Auto-cleanup old completed sessions (archive to history)
- Webhook notifications on state changes
- Analytics on session duration/completion rates
- Integration with GitHub via issues/PRs
