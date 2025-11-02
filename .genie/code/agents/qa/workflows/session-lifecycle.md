**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/session-lifecycle
description: Session state management and lifecycle testing
parent: qa
---

# QA Workflow • Session Lifecycle

## Test Scenario
Validate session creation, persistence, resumption, and termination throughout complete lifecycle.

## Test Suite

### Session Creation
**Command:**
```
mcp__genie__run with agent="genie" and prompt="Test session creation"
```

**Expected Evidence:**
- Unique session ID generated (UUID format)
- Session entry created in sessions.json
- Status starts as "starting" then transitions to "running"
- Log file created

**Verification:**
- [ ] Session ID unique (no duplicates)
- [ ] sessions.json entry exists
- [ ] Status transitions correctly
- [ ] Log file has content

---

### Session Tracking
**Command:**
```
# After creating session above
mcp__genie__list_sessions
```

**Expected Evidence:**
- New session appears in list
- Metadata correct (agent, timestamp, status)
- No duplicate session IDs

**Verification:**
- [ ] Session listed immediately after creation
- [ ] All metadata accurate
- [ ] No session ID collision (Bug #102 fix)

---

### Context Preservation (Single Resume)
**Command:**
```
# Create session
mcp__genie__run with agent="genie" and prompt="Remember this: TEST_VALUE_12345"

# Resume session
mcp__genie__resume with sessionId="<id>" and prompt="What value should you remember?"
```

**Expected Evidence:**
- Agent recalls TEST_VALUE_12345
- Previous conversation referenced
- Context carried forward

**Verification:**
- [ ] Agent remembers previous message
- [ ] Context not lost
- [ ] Conversation flows naturally

---

### Multi-Turn Conversation
**Command:**
```
# Create session
mcp__genie__run with agent="genie" and prompt="Turn 1: Set counter to 1"

# Resume 3+ times
mcp__genie__resume with sessionId="<id>" and prompt="Turn 2: Increment counter"
mcp__genie__resume with sessionId="<id>" and prompt="Turn 3: Increment counter"
mcp__genie__resume with sessionId="<id>" and prompt="Turn 4: What is counter value?"
```

**Expected Evidence:**
- Counter value tracks correctly across all turns
- Context builds incrementally
- No memory loss between resumes

**Verification:**
- [ ] Multi-turn context preserved
- [ ] All interactions tracked
- [ ] Context accumulates properly

---

### Session Persistence After Resume (Bug #66 Fix)
**Command:**
```
# Create session
mcp__genie__run with agent="implementor" and prompt="Persistence test"

# Resume session
mcp__genie__resume with sessionId="<id>" and prompt="Continue work"

# View session
mcp__genie__view with sessionId="<id>" and full=false
```

**Expected Evidence:**
- Session exists after resume
- Both original and resume messages visible
- No "Session not found" error

**Verification:**
- [ ] Session persists through resume
- [ ] All messages accessible
- [ ] No disappearance (Bug #66 regression test)

---

### Graceful Session Stop
**Command:**
```
# Create active session
mcp__genie__run with agent="genie" and prompt="Test stop"

# Stop session
mcp__genie__stop with sessionId="<id>"

# Verify status
mcp__genie__list_sessions
```

**Expected Evidence:**
- Session terminated cleanly
- Status updated to "completed" or "stopped"
- State preserved in sessions.json

**Verification:**
- [ ] Session stops gracefully
- [ ] Status transitions from "running" to "completed/stopped"
- [ ] No zombie sessions (Bug #92 fix)

---

### Post-Stop Transcript Access
**Command:**
```
# After stopping session above
mcp__genie__view with sessionId="<id>" and full=true
```

**Expected Evidence:**
- Full transcript still accessible
- All messages preserved
- Complete conversation history available

**Verification:**
- [ ] Transcript accessible after stop
- [ ] All messages intact
- [ ] History preserved

---

### Long-Running Session State
**Command:**
```
# Create session, let it run for extended period
mcp__genie__run with agent="genie" and prompt="Long-running test"

# Wait 5+ minutes

# Check status
mcp__genie__list_sessions
mcp__genie__view with sessionId="<id>"
```

**Expected Evidence:**
- Session remains stable over time
- Status doesn't drift to incorrect state
- No spontaneous termination

**Verification:**
- [ ] Session stable over time
- [ ] Status accurate
- [ ] No state corruption

---

### Concurrent Sessions
**Command:**
```
# Create 3 sessions simultaneously
mcp__genie__run with agent="genie" and prompt="Concurrent test 1"
mcp__genie__run with agent="implementor" and prompt="Concurrent test 2"
mcp__genie__run with agent="tests" and prompt="Concurrent test 3"

# Verify all tracked
mcp__genie__list_sessions
```

**Expected Evidence:**
- All 3 sessions created successfully
- Each has unique session ID
- No interference between sessions

**Verification:**
- [ ] All sessions created
- [ ] No session ID collision
- [ ] Independent execution

---

## Execution Notes

**Run this workflow via:**
```
mcp__genie__run with agent="qa/session-lifecycle" and prompt="Execute all session lifecycle tests"
```

**Evidence capture:**
- Save outputs to `.genie/qa/evidence/session-lifecycle-<timestamp>.txt`
- Track session IDs for all tests
- Document timing and state transitions
