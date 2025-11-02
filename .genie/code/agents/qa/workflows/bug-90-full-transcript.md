**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/bug-90
description: Regression test for Bug #90 - full=true returns truncated checkpoints
parent: qa
---

# Bug #90 Regression Test • Full Transcript Retrieval

## Bug Description
**Fixed in:** RC9
**Issue:** https://github.com/namastexlabs/automagik-genie/issues/90

When calling `mcp__genie__view` with `full=true`, the MCP response returned truncated "Key Checkpoints" (~12 snippets) instead of the full session transcript.

**Root Cause:** View command returned checkpoints summary instead of complete log when `full=true` parameter was set.

**Fix:** Modified view logic to return complete transcript when `full=true`, checkpoints summary when `full=false`.

## Test Scenario
Verify that `mcp__genie__view` with `full=true` returns complete conversation transcript, not abbreviated checkpoints.

## Test Steps

### Step 1: Create Session with Substantial Content
**Command:**
```
mcp__genie__run with agent="genie" and prompt="
Please provide a detailed response covering the following topics:
1. Software testing methodologies (5 paragraphs)
2. Validation strategies (3 paragraphs)
3. Evidence capture techniques (3 paragraphs)
4. Quality assurance principles (4 paragraphs)

Make each paragraph substantial (50+ words) to create a long transcript.
"
```

**Capture:**
- Session ID

**Expected:**
- Agent responds with 15+ paragraphs
- Substantial first message content

### Step 2: Resume Session Multiple Times
**Command:**
```
# Resume #1
mcp__genie__resume with sessionId="<id>" and prompt="Continue with 3 more paragraphs about regression testing"

# Resume #2
mcp__genie__resume with sessionId="<id>" and prompt="Add 2 paragraphs about automation frameworks"

# Resume #3
mcp__genie__resume with sessionId="<id>" and prompt="Conclude with 2 paragraphs about continuous integration"
```

**Expected:**
- 4 total messages in conversation (1 original + 3 resumes)
- Each message has substantial content
- Total transcript length: 20+ paragraphs

### Step 3: View with full=false (Summary)
**Command:**
```
mcp__genie__view with sessionId="<id>" and full=false
```

**Expected Evidence:**
- Recent summary displayed
- Checkpoint format (abbreviated)
- NOT complete transcript
- Concise output

**Verification Checks:**
- [ ] Summary format returned
- [ ] Output abbreviated (not full conversation)
- [ ] Key points highlighted

### Step 4: View with full=true (Complete Transcript)
**Command:**
```
mcp__genie__view with sessionId="<id>" and full=true
```

**Expected Evidence:**
- **COMPLETE** conversation transcript
- All 4 messages present (original + 3 resumes)
- Full content from each message
- All 20+ paragraphs visible
- No truncation or abbreviation

**Verification Checks:**
- [ ] Full transcript returned (not checkpoints)
- [ ] All 4 messages present
- [ ] All paragraphs visible (20+ total)
- [ ] No content truncation
- [ ] Complete conversation history

### Step 5: Compare Output Lengths
**Command:**
```bash
# Capture full=false output
mcp__genie__view with sessionId="<id>" and full=false > /tmp/summary.txt

# Capture full=true output
mcp__genie__view with sessionId="<id>" and full=true > /tmp/full.txt

# Compare sizes
wc -l /tmp/summary.txt /tmp/full.txt
wc -w /tmp/summary.txt /tmp/full.txt
```

**Expected Evidence:**
- full.txt significantly larger than summary.txt
- full.txt contains all conversation content
- Line count difference substantial (10x+ ratio expected)
- Word count difference substantial

**Verification Checks:**
- [ ] full=true output > full=false output (by large margin)
- [ ] full=true contains all messages
- [ ] No truncation in full=true output

### Step 6: Verify Against Raw Log File
**Command:**
```bash
# Find log file for session
grep -l "<session-id>" .genie/state/agents/logs/*.log

# Compare log file to full=true output
cat <log-file> | wc -l
# vs
mcp__genie__view with sessionId="<id>" and full=true | wc -l
```

**Expected Evidence:**
- full=true output matches raw log file content
- Line counts similar (within reasonable margin)
- All interactions present in both

**Verification Checks:**
- [ ] full=true matches log file
- [ ] All log content reflected in full=true output
- [ ] No arbitrary truncation

## Pass Criteria
✅ full=true returns complete transcript (all messages)
✅ full=false returns abbreviated summary
✅ Clear difference in output length between full=true and full=false
✅ full=true matches raw log file content
✅ No truncation or checkpoint abbreviation with full=true

## Fail Criteria
❌ full=true returns checkpoints instead of full transcript
❌ full=true output similar length to full=false
❌ Missing messages or content in full=true output
❌ Arbitrary truncation in full=true output

## Evidence Capture
```bash
# Save both outputs
mcp__genie__view with sessionId="<id>" and full=false > .genie/qa/evidence/bug-90-summary-<timestamp>.txt
mcp__genie__view with sessionId="<id>" and full=true > .genie/qa/evidence/bug-90-full-<timestamp>.txt

# Document size comparison
echo "Summary: $(wc -l < .genie/qa/evidence/bug-90-summary-<timestamp>.txt) lines"
echo "Full: $(wc -l < .genie/qa/evidence/bug-90-full-<timestamp>.txt) lines"
```

## Related Tests
## Maintenance
**Test Frequency:** Every RC release
**Last Tested:** [Date]
**Result:** [✅ Pass | ❌ Fail]
**Notes:** [Any observations]
