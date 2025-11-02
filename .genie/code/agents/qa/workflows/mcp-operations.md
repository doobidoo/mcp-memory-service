**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/mcp-operations
description: MCP tool operations testing (list, run, view, resume, stop)
parent: qa
---

# QA Workflow • MCP Operations

## Test Scenario
Validate core MCP tool functionality: agent discovery, session management, error handling.

## Test Suite

### List Agents
**Command:**
```
mcp__genie__list_agents
```

**Expected Evidence:**
- Table with agent names, descriptions, categories
- All core agents displayed (genie, implementor, tests, polish, git, release, etc.)
- Custom agents listed if present

**Verification:**
- [ ] Agent catalog displays
- [ ] No errors or crashes
- [ ] Output formatted correctly

---

### Run Agent (Valid)
**Command:**
```
mcp__genie__run with agent="genie" and prompt="Test prompt"
```

**Expected Evidence:**
- Session ID returned
- Agent starts successfully
- Session appears in sessions.json

**Verification:**
- [ ] Session ID generated
- [ ] Agent launches
- [ ] Status transitions to "running"

---

### Run Agent (Invalid)
**Command:**
```
mcp__genie__run with agent="nonexistent" and prompt="test"
```

**Expected Evidence:**
- Clear error message
- List of available agents shown
- No crash or hang

**Verification:**
- [ ] Error message clear and helpful
- [ ] Suggests valid agent names
- [ ] Fails gracefully

---

### List Sessions
**Command:**
```
mcp__genie__list_sessions
```

**Expected Evidence:**
- Table with session IDs, agents, status, timing
- Active sessions shown
- Empty table if no sessions

**Verification:**
- [ ] All active sessions listed
- [ ] Status accurate (running/completed/stopped)
- [ ] Timestamps correct

---

### Resume Session (Valid)
**Command:**
```
mcp__genie__resume with sessionId="<valid-id>" and prompt="Follow-up"
```

**Expected Evidence:**
- Session continues with context preserved
- Previous conversation referenced
- New message appended to transcript

**Verification:**
- [ ] Session resumes successfully
- [ ] Context preserved
- [ ] Multi-turn conversation works

---

### View Session Summary
**Command:**
```
mcp__genie__view with sessionId="<id>" and full=false
```

**Expected Evidence:**
- Recent messages from session
- Summary format (not full transcript)
- Session metadata included

**Verification:**
- [ ] Summary displays
- [ ] Recent messages shown
- [ ] Concise output

---

### View Full Session
**Command:**
```
mcp__genie__view with sessionId="<id>" and full=true
```

**Expected Evidence:**
- Complete conversation transcript
- All messages from session start
- Full context available

**Verification:**
- [ ] Full transcript returned
- [ ] All messages present
- [ ] No truncation (Bug #90 fix)

---

### Stop Session
**Command:**
```
mcp__genie__stop with sessionId="<id>"
```

**Expected Evidence:**
- Session terminated gracefully
- State preserved for viewing
- Status updated to "completed"

**Verification:**
- [ ] Session stops
- [ ] Status transitions correctly
- [ ] Transcript still accessible

---

## Error Scenarios

### Invalid Session Resume
**Command:**
```
mcp__genie__resume with sessionId="invalid-id" and prompt="test"
```

**Expected Evidence:**
- Clear error message
- No crash
- Helpful guidance

**Verification:**
- [ ] Error message clear
- [ ] System remains stable
- [ ] User knows what to fix

---

### Nonexistent Session View
**Command:**
```
mcp__genie__view with sessionId="nonexistent"
```

**Expected Evidence:**
- "Session not found" error
- No crash
- Clear message

**Verification:**
- [ ] Error handled gracefully
- [ ] Message helpful
- [ ] No system instability

---

### Invalid Session Stop
**Command:**
```
mcp__genie__stop with sessionId="invalid"
```

**Expected Evidence:**
- Error message returned
- No crash
- System stable

**Verification:**
- [ ] Graceful error handling
- [ ] Clear feedback
- [ ] System remains stable

---

## Execution Notes

**Run this workflow via:**
```
mcp__genie__run with agent="qa/mcp-operations" and prompt="Execute all MCP operation tests"
```

**Evidence capture:**
- Save outputs to `.genie/qa/evidence/mcp-ops-<timestamp>.txt`
- Document any failures with reproduction steps
- Update checklist with results
