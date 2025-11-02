**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/bug-92
description: Regression test for Bug #92 - Zombie sessions stuck in running status
parent: qa
---

# Bug #92 Regression Test • Zombie Sessions

## Bug Description
**Fixed in:** RC9
**Issue:** https://github.com/namastexlabs/automagik-genie/issues/92

Some sessions remained in "running" status for days despite being completed or abandoned, polluting the session list and making it difficult to identify actually active sessions.

**Root Cause:** No cleanup or abandonment marking. Sessions never transitioned from "running" to "completed" or "abandoned" status.

**Fix:** Added session lifecycle management with status transitions and cleanup marking.

## Test Scenario
Verify that sessions transition from "running" to "completed" or "stopped" status appropriately, and old/abandoned sessions don't remain stuck in "running" status indefinitely.

## Test Steps

### Step 1: Create Short-Lived Session
**Command:**
```
mcp__genie__run with agent="genie" and prompt="Quick test - respond with one sentence"
```

**Capture:**
- Session ID
- Start timestamp

**Expected:**
- Session starts with status "running"
- Agent responds quickly

### Step 2: Stop Session Immediately
**Command:**
```
mcp__genie__stop with sessionId="<id>"
```

**Expected Evidence:**
- Session stops gracefully
- Status transitions to "completed" or "stopped"
- NOT stuck in "running"

**Verification Checks:**
- [ ] Stop command succeeds
- [ ] Status changes from "running" to "completed/stopped"
- [ ] Transition happens immediately

### Step 3: Verify Status in List
**Command:**
```
mcp__genie__list_sessions
```

**Expected Evidence:**
- Session shows "completed" or "stopped" status
- NOT "running"
- Status field accurate

**Verification Checks:**
- [ ] Session listed with correct status
- [ ] NOT showing as "running" after stop
- [ ] Status reflects actual state

### Step 4: Check sessions.json Directly
**Command:**
```bash
cat .genie/state/agents/sessions.json | jq '.sessions["<session-id>"].status'
```

**Expected Evidence:**
- Status field shows "completed" or "stopped"
- NOT "running"
- Matches list_sessions output

**Verification Checks:**
- [ ] sessions.json status correct
- [ ] Matches MCP output
- [ ] State persisted correctly

### Step 5: Create Session and Let Complete Naturally
**Command:**
```
# Create session
mcp__genie__run with agent="genie" and prompt="Test natural completion - respond with one sentence"

# DON'T call stop - let it complete naturally

# Wait 30 seconds for response

# Check status
mcp__genie__list_sessions
```

**Expected Evidence:**
- Session completes naturally after response
- Status transitions to "completed" (not stuck in "running")
- No manual stop required

**Verification Checks:**
- [ ] Session completes on its own
- [ ] Status transitions to "completed"
- [ ] NOT stuck in "running" after completion

### Step 6: Test Long-Running Session Status
**Command:**
```
# Create session
mcp__genie__run with agent="genie" and prompt="Long task simulation - please list 20 items with descriptions"

# Check status immediately
mcp__genie__list_sessions

# Wait for completion

# Check status after completion
mcp__genie__list_sessions
```

**Expected Evidence:**
- Status "running" while actively responding
- Status transitions to "completed" after response finishes
- Clear before/after status change

**Verification Checks:**
- [ ] Status "running" during execution
- [ ] Status "completed" after finish
- [ ] Transition happens automatically

### Step 7: Test Multiple Sessions Cleanup
**Command:**
```
# Create 5 sessions
mcp__genie__run with agent="genie" and prompt="Test 1"
mcp__genie__run with agent="genie" and prompt="Test 2"
mcp__genie__run with agent="genie" and prompt="Test 3"
mcp__genie__run with agent="genie" and prompt="Test 4"
mcp__genie__run with agent="genie" and prompt="Test 5"

# Stop all sessions
mcp__genie__stop with sessionId="<id-1>"
mcp__genie__stop with sessionId="<id-2>"
mcp__genie__stop with sessionId="<id-3>"
mcp__genie__stop with sessionId="<id-4>"
mcp__genie__stop with sessionId="<id-5>"

# Verify all marked stopped
mcp__genie__list_sessions
```

**Expected Evidence:**
- All 5 sessions show "completed" or "stopped"
- NONE show "running" after stop
- Bulk cleanup works correctly

**Verification Checks:**
- [ ] All sessions stopped
- [ ] No zombie "running" sessions
- [ ] Bulk status transition works

### Step 8: Check for Stale Running Sessions
**Command:**
```bash
# Query sessions.json for old "running" sessions
cat .genie/state/agents/sessions.json | jq '.sessions | to_entries[] | select(.value.status == "running") | {id: .key, lastUsed: .value.lastUsed, status: .value.status}'
```

**Expected Evidence:**
- No sessions with status="running" and old lastUsed timestamps (>1 hour ago)
- All old sessions marked completed/abandoned
- Session list clean

**Verification Checks:**
- [ ] No zombie sessions found
- [ ] Old sessions properly marked
- [ ] Status reflects reality

## Pass Criteria
✅ Sessions transition from "running" to "completed/stopped" correctly
✅ Stop command updates status immediately
✅ Natural completion updates status automatically
✅ No sessions stuck in "running" for extended periods
✅ Bulk cleanup works for multiple sessions
✅ sessions.json status matches list_sessions output

## Fail Criteria
❌ Sessions stuck in "running" after stop
❌ Natural completion doesn't update status
❌ Old sessions remain "running" indefinitely
❌ Status mismatch between sessions.json and list_sessions

## Evidence Capture
```bash
# Save session list before and after stops
mcp__genie__list_sessions > .genie/qa/evidence/bug-92-before-stop-<timestamp>.txt

# [Perform stops]

mcp__genie__list_sessions > .genie/qa/evidence/bug-92-after-stop-<timestamp>.txt

# Compare
diff .genie/qa/evidence/bug-92-before-stop-<timestamp>.txt .genie/qa/evidence/bug-92-after-stop-<timestamp>.txt
```

## Related Tests
## Maintenance
**Test Frequency:** Every RC release
**Last Tested:** [Date]
**Result:** [✅ Pass | ❌ Fail]
**Notes:** [Any observations]
