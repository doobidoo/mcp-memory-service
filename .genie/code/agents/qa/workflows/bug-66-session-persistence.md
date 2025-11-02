**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/bug-66
description: Regression test for Bug #66 - Session disappears after resume
parent: qa
---

# Bug #66 Regression Test • Session Persistence After Resume

## Bug Description
**Fixed in:** RC9 (likely via Bug #102 fix)
**Issue:** https://github.com/namastexlabs/automagik-genie/issues/66

MCP session disappeared after resume, returning "No run found" error when attempting to view or resume session that previously existed.

**Root Cause:** Related to Bug #102 (session collision). When sessions.json used agent name as key, resuming would overwrite previous session entry, making it appear to disappear.

**Fix:** Session collision fix (using sessionId as key) resolved this issue as side effect.

## Test Scenario
Verify that sessions persist correctly after resume operations and don't disappear from session tracking.

## Test Steps

### Step 1: Create Initial Session
**Command:**
```
mcp__genie__run with agent="implementor" and prompt="Persistence test - Phase 1"
```

**Capture:**
- Session ID
- Initial message content

**Expected:**
- Session created successfully
- Session ID returned

### Step 2: Verify Session Exists
**Command:**
```
# Check in session list
mcp__genie__list_sessions

# View session
mcp__genie__view with sessionId="<id>" and full=false
```

**Expected Evidence:**
- Session appears in list
- View returns session content
- Initial message visible

**Verification Checks:**
- [ ] Session in list_sessions output
- [ ] View succeeds
- [ ] Initial message present

### Step 3: Resume Session (First Resume)
**Command:**
```
mcp__genie__resume with sessionId="<id>" and prompt="Persistence test - Phase 2 (first resume)"
```

**Expected Evidence:**
- Resume succeeds
- Session continues with context
- New message appended

**Verification Checks:**
- [ ] Resume command succeeds
- [ ] No "No run found" error
- [ ] Context preserved

### Step 4: Verify Session Still Exists (After First Resume)
**Command:**
```
# Check list again
mcp__genie__list_sessions

# View again
mcp__genie__view with sessionId="<id>" and full=false
```

**Expected Evidence:**
- Session still in list (not disappeared)
- View still works
- Both messages visible (Phase 1 + Phase 2)

**Verification Checks:**
- [ ] Session still exists in list
- [ ] View command works
- [ ] Both messages present
- [ ] Session ID unchanged

### Step 5: Resume Session Again (Second Resume)
**Command:**
```
mcp__genie__resume with sessionId="<id>" and prompt="Persistence test - Phase 3 (second resume)"
```

**Expected Evidence:**
- Second resume succeeds
- Session context includes all previous messages
- No session loss

**Verification Checks:**
- [ ] Second resume works
- [ ] No errors
- [ ] Context complete

### Step 6: Verify Session Still Exists (After Second Resume)
**Command:**
```
mcp__genie__list_sessions
mcp__genie__view with sessionId="<id>" and full=true
```

**Expected Evidence:**
- Session still exists
- Full transcript shows all 3 phases
- No disappearance after multiple resumes

**Verification Checks:**
- [ ] Session persists through multiple resumes
- [ ] All 3 messages visible
- [ ] Session ID consistent
- [ ] No "No run found" error

### Step 7: Create Second Session of Same Agent Type
**Command:**
```
mcp__genie__run with agent="implementor" and prompt="Second implementor session"
```

**Capture:**
- New session ID

**Expected:**
- New unique session ID (different from Step 1)

### Step 8: Verify Both Sessions Exist
**Command:**
```
mcp__genie__list_sessions
```

**Expected Evidence:**
- BOTH implementor sessions visible
- Different session IDs
- First session not overwritten by second

**Verification Checks:**
- [ ] Both sessions in list
- [ ] First session ID unchanged
- [ ] Second session has different ID
- [ ] No overwrite/collision

### Step 9: Resume First Session While Second Exists
**Command:**
```
mcp__genie__resume with sessionId="<first-session-id>" and prompt="Persistence test - Phase 4"
```

**Expected Evidence:**
- First session resumes successfully
- Second session unaffected
- No disappearance despite multiple sessions of same agent

**Verification Checks:**
- [ ] First session resumes
- [ ] No collision with second session
- [ ] Both sessions intact

### Step 10: Verify sessions.json Integrity
**Command:**
```bash
cat .genie/state/agents/sessions.json | jq '.sessions | keys | length'
# Expected: 2 (or more if other tests running)

cat .genie/state/agents/sessions.json | jq '.sessions | to_entries[] | select(.value.agent == "genie/agents/implementor/implementor") | .key'
# Should show both session IDs
```

**Expected Evidence:**
- Both session IDs present as keys
- No overwriting
- Each session has complete data

**Verification Checks:**
- [ ] Both sessions in sessions.json
- [ ] sessionId used as key (not agent name)
- [ ] No data loss or overwrite

## Pass Criteria
✅ Sessions persist through single resume
✅ Sessions persist through multiple resumes
✅ No "No run found" error after resume
✅ Session ID remains constant across resumes
✅ Multiple sessions of same agent type don't overwrite each other
✅ sessions.json maintains both sessions correctly

## Fail Criteria
❌ Session disappears after resume
❌ "No run found" error occurs
❌ Session ID changes after resume
❌ Second session of same type overwrites first
❌ Session data lost in sessions.json

## Evidence Capture
```bash
# Track session through lifecycle
mcp__genie__list_sessions > .genie/qa/evidence/bug-66-before-resume-<timestamp>.txt

# [Perform resumes]

mcp__genie__list_sessions > .genie/qa/evidence/bug-66-after-resumes-<timestamp>.txt

# Verify session persisted
grep "<session-id>" .genie/qa/evidence/bug-66-before-resume-<timestamp>.txt
grep "<session-id>" .genie/qa/evidence/bug-66-after-resumes-<timestamp>.txt
# Should appear in both files
```

## Related Tests
## Maintenance
**Test Frequency:** Every RC release
**Last Tested:** [Date]
**Result:** [✅ Pass | ❌ Fail]
**Notes:** [Any observations]
