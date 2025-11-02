**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/bug-102
description: Regression test for Bug #102 - Session ID collision
parent: qa
---

# Bug #102 Regression Test • Session ID Collision

## Bug Description
**Fixed in:** RC9
**Issue:** https://github.com/namastexlabs/automagik-genie/issues/102

Session ID `4946bad6-98f4-4822-b90b-6abc09d21fc7` appeared **THREE times** in `mcp__genie__list_sessions()` output with different agents and creation times.

**Root Cause:** sessions.json used agent name as key instead of sessionId, causing collisions when multiple sessions of same agent type were created.

**Fix:** Changed sessions.json to use sessionId as key.

## Test Scenario
Verify that each session gets a unique session ID, even when multiple sessions of the same agent type are created.

## Test Steps

### Step 1: Create Multiple Sessions with Same Agent Type
**Command:**
```
# Create 3 genie sessions
mcp__genie__run with agent="genie" and prompt="Test 1"
mcp__genie__run with agent="genie" and prompt="Test 2"
mcp__genie__run with agent="genie" and prompt="Test 3"
```

**Capture:**
- Record all 3 session IDs returned

### Step 2: Create Sessions with Different Agent Types
**Command:**
```
# Create sessions with different agents
mcp__genie__run with agent="implementor" and prompt="Test 4"
mcp__genie__run with agent="tests" and prompt="Test 5"
mcp__genie__run with agent="git" and prompt="Test 6"
```

**Capture:**
- Record all 3 session IDs returned

### Step 3: List All Sessions
**Command:**
```
mcp__genie__list_sessions
```

**Expected Evidence:**
- 6 unique session IDs displayed
- No duplicate session IDs in output
- Each session maps to exactly one agent invocation

**Verification Checks:**
- [ ] All 6 sessions listed
- [ ] All session IDs unique (no duplicates)
- [ ] Agent types correctly associated with each session
- [ ] Timestamps different for each session

### Step 4: Verify sessions.json Structure
**Command:**
```bash
cat .genie/state/agents/sessions.json | jq '.sessions | keys'
```

**Expected Evidence:**
- 6 session IDs as keys
- No agent names as keys
- Each key is a unique session ID

**Verification Checks:**
- [ ] sessions.json uses sessionId as keys (not agent names)
- [ ] 6 unique keys present
- [ ] No key duplication

### Step 5: View Each Session Individually
**Command:**
```
# For each session ID captured above
mcp__genie__view with sessionId="<id-1>" and full=false
mcp__genie__view with sessionId="<id-2>" and full=false
# ... etc for all 6
```

**Expected Evidence:**
- Each view returns correct session
- No cross-contamination between sessions
- Prompts match what was submitted

**Verification Checks:**
- [ ] Each view returns correct session data
- [ ] No ambiguity about which session is returned
- [ ] Session content matches creation prompt

## Pass Criteria
✅ All 6 sessions have unique session IDs
✅ No duplicate session IDs in list_sessions output
✅ sessions.json uses sessionId as keys (not agent names)
✅ Each view command returns correct session
✅ No session collision or cross-contamination

## Fail Criteria
❌ Any duplicate session IDs found
❌ sessions.json uses agent names as keys
❌ View command returns wrong session
❌ Session data cross-contaminated

## Evidence Capture
```bash
# Save list_sessions output
mcp__genie__list_sessions > .genie/qa/evidence/bug-102-session-list-<timestamp>.txt

# Save sessions.json structure
cat .genie/state/agents/sessions.json | jq '.' > .genie/qa/evidence/bug-102-sessions-json-<timestamp>.json

# Extract and count unique session IDs
cat .genie/state/agents/sessions.json | jq '.sessions | keys | length'
# Expected: 6
```

## Related Tests
## Maintenance
**Test Frequency:** Every RC release
**Last Tested:** [Date]
**Result:** [✅ Pass | ❌ Fail]
**Notes:** [Any observations]
