**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/bug-xxx-background-launch
description: NEW BUG - Background agent launch failure (executor never spawns)
parent: qa
---

# NEW BUG • Background Agent Launch Failure

## Bug Description
**Discovered:** 2025-10-17 (during RC9 testing)
**Status:** UNRESOLVED
**Issue:** TBD (needs GitHub issue creation)

Background agents (with `background: true`) fail to launch silently. Session entry created in sessions.json but executor process never spawns.

**Symptoms:**
- Session status stuck in "starting"
- executorPid remains null
- Log file created but 0 bytes (no content)
- runnerPid process doesn't exist
- `list_sessions` doesn't show session (filters out "starting" status?)
- `view` returns "No messages yet"

**Evidence from Discovery:**
- qa agent invocation (session: temp-genie/agents/qa/qa-1760728615256)
- implementor agent invocation (session: temp-genie/agents/implementor/implementor-1760728741265)
- Both stuck in identical failure state

## Test Scenario
Attempt to launch background agents and verify they spawn executor processes correctly.

## Test Steps

### Step 1: Launch Background Agent (qa)
**Command:**
```
mcp__genie__run with agent="qa" and prompt="Background launch test"
```

**Expected Evidence:**
- Session ID returned
- Status transitions "starting" → "running"
- executorPid populated (not null)
- Log file has content
- Process visible in system

**Actual Behavior (Bug):**
- Timeout waiting for session ID
- Status stuck at "starting"
- executorPid null
- Log file 0 bytes
- runnerPid process doesn't exist

**Verification Checks:**
- [ ] Session ID returned (not timeout)
- [ ] Status transitions to "running"
- [ ] executorPid not null
- [ ] Log file has content

### Step 2: Check sessions.json Entry
**Command:**
```bash
cat .genie/state/agents/sessions.json | jq '.sessions | to_entries[] | select(.value.status == "starting")'
```

**Expected (Bug Present):**
```json
{
  "status": "starting",
  "executorPid": null,
  "runnerPid": <number>,
  "background": true
}
```

**Expected (Bug Fixed):**
- No sessions stuck in "starting" status
- executorPid populated
- Status "running"

**Verification Checks:**
- [ ] No sessions with status="starting" and executorPid=null
- [ ] All background sessions have executorPid

### Step 3: Check Process Existence
**Command:**
```bash
# Extract runnerPid from sessions.json
RUNNER_PID=$(cat .genie/state/agents/sessions.json | jq -r '.sessions["<session-id>"].runnerPid')

# Check if process exists
ps -p $RUNNER_PID -o pid,comm,args
```

**Expected (Bug Present):**
- Process not found

**Expected (Bug Fixed):**
- Process exists and running

**Verification Checks:**
- [ ] runnerPid process exists
- [ ] Process is active

### Step 4: Check Log File
**Command:**
```bash
# Extract log file path
LOG_FILE=$(cat .genie/state/agents/sessions.json | jq -r '.sessions["<session-id>"].logFile')

# Check log size and content
ls -lh "$LOG_FILE"
head -20 "$LOG_FILE"
```

**Expected (Bug Present):**
- File size: 0 bytes
- No content

**Expected (Bug Fixed):**
- File has content
- Session output visible

**Verification Checks:**
- [ ] Log file not empty
- [ ] Session output written

### Step 5: Test list_sessions Visibility
**Command:**
```
mcp__genie__list_sessions
```

**Expected (Bug Present):**
- Session NOT visible in list (filtered out due to "starting" status)

**Expected (Bug Fixed):**
- Session visible with status="running"

**Verification Checks:**
- [ ] Session appears in list_sessions
- [ ] Status shown as "running"

### Step 6: Test Other Background Agents
**Command:**
```
# Test multiple background agents
mcp__genie__run with agent="implementor" and prompt="Background test"
mcp__genie__run with agent="release" and prompt="Background test"
mcp__genie__run with agent="tests" and prompt="Background test"
```

**Expected:**
- All background agents should launch successfully
- No stuck "starting" status

**Verification Checks:**
- [ ] All background agents launch
- [ ] No executorPid=null failures
- [ ] All statuses transition to "running"

### Step 7: Compare with Foreground Agent
**Command:**
```
# Launch agent with background: false (foreground)
# (Need to identify which agents are foreground)
mcp__genie__run with agent="<foreground-agent>" and prompt="Foreground test"
```

**Expected:**
- Foreground agents work correctly
- Bug specific to background agents?

**Verification Checks:**
- [ ] Foreground agents launch normally
- [ ] Issue isolated to background agents

### Step 8: Check Stderr/Stdout
**Command:**
```bash
# Check for error messages in MCP stderr output
# (Captured during mcp__genie__run invocation)
```

**Evidence from Discovery:**
```
Stderr:
🧞 Starting agent: genie/agents/qa/qa
   Executor: claude (from agent)
   Mode: default
   Model: sonnet
   Permissions: default
```

**Expected (Bug Fixed):**
- Additional output showing executor spawn
- Session ID confirmation

**Verification Checks:**
- [ ] Stderr shows executor spawn confirmation
- [ ] No silent failure

## Root Cause Investigation

**Hypothesis 1: Permission Issue**
- Check if bypassPermissions actually works
- Verify executor can access required files

**Hypothesis 2: Path Issue**
- Check if executor binary path correct
- Verify Claude CLI available in PATH

**Hypothesis 3: Spawn Configuration**
- Check process spawn options (stdin, stdout, stderr)
- Verify background spawn logic

**Hypothesis 4: Timeout Issue**
- Check if executor spawning but taking too long
- Verify timeout thresholds

## Pass Criteria
✅ Background agents launch successfully
✅ executorPid populated (not null)
✅ Status transitions "starting" → "running"
✅ Log files have content
✅ Processes visible in system
✅ Sessions appear in list_sessions

## Fail Criteria (Current State)
❌ executorPid remains null
❌ Status stuck at "starting"
❌ Log files empty (0 bytes)
❌ runnerPid process doesn't exist
❌ Sessions invisible in list_sessions

## Evidence Capture
```bash
# Capture sessions.json state
cat .genie/state/agents/sessions.json > .genie/qa/evidence/bug-xxx-sessions-json-<timestamp>.json

# Capture process list
ps aux > .genie/qa/evidence/bug-xxx-processes-<timestamp>.txt

# Capture log file status
ls -lhR .genie/state/agents/logs/ > .genie/qa/evidence/bug-xxx-log-files-<timestamp>.txt
```

## Related Tests
## Maintenance
**Test Frequency:** Every RC release (until fixed)
**Last Tested:** 2025-10-17 (RC9)
**Result:** ❌ FAIL
**Notes:** Critical bug blocking background agent functionality
