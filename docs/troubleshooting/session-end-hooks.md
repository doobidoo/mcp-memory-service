# SessionEnd Hook Troubleshooting Guide

## Overview

SessionEnd hooks automatically consolidate conversation outcomes when you exit Claude Code. However, many users are confused about **when these hooks actually fire** and why memories might not be created as expected.

This guide clarifies the session lifecycle and common troubleshooting scenarios.

---

## Critical Concept: Session Lifecycle

Claude Code distinguishes between **session pause/suspend** and **session termination**:

| User Action | Session State | Hook Triggered | Memory Created? |
|-------------|---------------|----------------|-----------------|
| **Ctrl+C (once)** | Interrupt input | None | ❌ No |
| **Ctrl+C (twice)** | Suspend session | None | ❌ No |
| **Resume session** | Continue existing | `SessionStart:resume` | ❌ No (loads existing) |
| **`/exit` command** | Terminate | `SessionEnd` | ✅ **Yes** |
| **Close terminal** | Terminate | `SessionEnd` | ✅ **Yes** |
| **Kill process** | May terminate | `SessionEnd` (if graceful) | ⚠️ Maybe |

### Key Takeaway

**Ctrl+C does NOT trigger SessionEnd hooks.** It suspends the session, which you can later resume. Only actual session termination (e.g., `/exit`) triggers SessionEnd.

---

## Common Issue: "My Session Didn't Create a Memory"

### Symptom

You exited Claude Code with Ctrl+C (twice), resumed later, and noticed no `session-consolidation` memory was created for your previous session.

### Root Cause

**Ctrl+C suspends the session rather than ending it.** When you resume with `SessionStart:resume`, the session continues from where you left off - no SessionEnd hook fires.

### Evidence

When you resume a session, you'll see:
```
SessionStart:resume hook success
```

This confirms you **resumed** an existing session, not started a new one.

### Solution

**Always use `/exit` to properly terminate sessions** if you want SessionEnd memories created:

```bash
# In Claude Code prompt:
/exit
```

This triggers graceful shutdown and SessionEnd hook execution.

---

## Common Issue: Connection Failures (SessionEnd & SessionStart)

> **Note**: This issue affects both SessionEnd and SessionStart hooks, but with different symptoms:
> - **SessionEnd**: Hard failure - cannot store session memory
> - **SessionStart**: Soft failure - falls back to MCP tools, shows "No relevant memories found"
>
> See [hooks-quick-reference.md](hooks-quick-reference.md#sessionstart-hook-issues) for detailed SessionStart troubleshooting.

### Symptom (SessionEnd)

During SessionStart, you see:
```
⚠️ Memory Connection → Failed to connect using any available protocol
💾 Storage → 💾 Unknown Storage (http://127.0.0.1:8000)
```

### Symptom (SessionStart)

Multiple "MCP Fallback" messages and no memories loaded:
```
↩️  MCP Fallback → Using standard MCP tools
↩️  MCP Fallback → Using standard MCP tools
↩️  MCP Fallback → Using standard MCP tools
📭 Memory Search → No relevant memories found
```

### Root Cause

**HTTP/HTTPS protocol mismatch** between hook configuration and memory service.

**Example**:
- **Server running**: `https://localhost:8000` (HTTPS)
- **Hook configured**: `http://127.0.0.1:8000` (HTTP)

### Diagnosis

Check your server protocol:
```bash
# Check server status
systemctl --user status mcp-memory-http.service
# Look for: "Uvicorn running on https://0.0.0.0:8000" or "http://..."

# Or test connection
curl -sk "https://localhost:8000/api/health"  # HTTPS
curl -s "http://127.0.0.1:8000/api/health"    # HTTP
```

Check your hook configuration:
```bash
grep endpoint ~/.claude/hooks/config.json
# Should show: "endpoint": "https://localhost:8000"
```

### Solution

Update `~/.claude/hooks/config.json` to match server protocol:

```json
{
  "memoryService": {
    "http": {
      "endpoint": "https://localhost:8000",  // Match your server
      "apiKey": "your-api-key-here"
    }
  }
}
```

**No restart required** - hooks reload config on next execution.

---

## SessionEnd Requirements

Even if SessionEnd fires correctly, memory creation requires:

### 1. Minimum Session Length
- Default: **100+ characters** total conversation
- Configurable: `sessionAnalysis.minSessionLength` in `config.json`
- Reason: Prevents noise from trivial sessions

### 2. Minimum Confidence Score
- Default: **> 0.1** (10% confidence)
- Based on conversation analysis quality
- Low confidence = session too generic to extract insights

### 3. Session Consolidation Enabled
```json
{
  "memoryService": {
    "enableSessionConsolidation": true  // Must be true
  }
}
```

### 4. One Summary per Session
- The summary is tagged `sys:claude-session:<session_id>` and stores `session_id` in its metadata. `claude --resume` keeps the session id unless you pass `--fork-session`.
- Ending again without new messages stores nothing ("Session summary unchanged since it was last stored, skipping"); a grown transcript updates the stored summary instead of adding one.
- A failed or slow lookup, or an MCP-only connection (no tag lookup), stores a new summary as before.
- SessionEnd hooks get 1.5 s in total unless `CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS` raises it (a `timeout` on a plugin hook does not). Before Claude Code v2.1.268 that variable raised only the overall budget, and each hook was still cut off after 1.5 s.

### What Gets Extracted

SessionEnd analyzes your conversation to extract:

- **Topics**: Keywords like "implementation", "debugging", "architecture", "performance"
- **Decisions**: Phrases like "decided to", "will use", "chose to", "going with"
- **Insights**: Phrases like "learned that", "discovered", "realized"
- **Code Changes**: Phrases like "implemented", "created", "refactored"
- **Next Steps**: Phrases like "next we need", "TODO", "remaining"

If conversation lacks these patterns, confidence will be low and memory won't be created.

---

## Verification & Debugging

### 1. Check Recent Session Memories

```bash
# Search for recent session consolidation memories
curl -sk "https://localhost:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["session-consolidation"], "limit": 5}' | \
  python -m json.tool | grep created_at_iso
```

Look for recent timestamps (today/yesterday).

### 2. Test SessionEnd Hook Manually

```bash
# Run hook with test conversation
node ~/.claude/hooks/core/session-end.js
```

Check output for:
- `[Memory Hook] Session ending - consolidating outcomes...`
- `[Memory Hook] Session analysis: X topics, Y decisions, confidence: Z%`
- `[Memory Hook] Session consolidation stored successfully`

### 3. Verify Connection

```bash
# Test server health
curl -sk "https://localhost:8000/api/health"

# Check config matches
grep endpoint ~/.claude/hooks/config.json
```

### 4. Check SessionEnd Configuration

```bash
# Verify SessionEnd hook is configured
grep -A 10 "SessionEnd" ~/.claude/settings.json

# Should show:
# "SessionEnd": [
#   {
#     "hooks": [
#       {
#         "type": "command",
#         "command": "node \"/home/user/.claude/hooks/core/session-end.js\"",
#         "timeout": 15
#       }
#     ]
#   }
# ]
```

---

## Quick Diagnosis Checklist

Use this checklist when SessionEnd memories aren't being created:

- [ ] **Did I use `/exit`** or just Ctrl+C?
  - **Fix**: Use `/exit` command for proper termination

- [ ] **Does `config.json` endpoint match server protocol?**
  - **Check**: HTTP vs HTTPS in both config and server
  - **Fix**: Update endpoint in `~/.claude/hooks/config.json`

- [ ] **Is the memory service running?**
  - **Check**: `curl https://localhost:8000/api/health`
  - **Fix**: Start server with `systemctl --user start mcp-memory-http.service`

- [ ] **Was conversation meaningful?**
  - **Check**: Total length > 100 characters
  - **Fix**: Have longer conversations with decisions/insights

- [ ] **Is session consolidation enabled?**
  - **Check**: `enableSessionConsolidation: true` in config
  - **Fix**: Update `~/.claude/hooks/config.json`

- [ ] **Is SessionEnd hook installed?**
  - **Check**: `grep SessionEnd ~/.claude/settings.json`
  - **Fix**: Run `cd claude-hooks && python install_hooks.py --all`

---

## Best Practices

### For Reliable Memory Consolidation

1. **Always use `/exit`** when you want session memories created
2. **Avoid Ctrl+C for final exit** - Use it only for interrupts/corrections
3. **Have meaningful conversations** - Include decisions, insights, plans
4. **Verify endpoint configuration** - HTTP vs HTTPS must match
5. **Check session memories periodically** - Confirm system is working

### For Debugging

1. **Check recent memories** - Look for session-consolidation tag
2. **Test hook manually** - Run `session-end.js` directly
3. **Verify connection** - Test health endpoint
4. **Read hook logs** - Look for error messages in terminal
5. **Consult session requirements** - Length, confidence, enabled settings

---

## Technical Details

### SessionEnd Hook Implementation

**File**: `~/.claude/hooks/core/session-end.js`

**Key Code Sections**:
- **Lines 298-365**: Main `onSessionEnd()` function
- **Line 316**: Minimum session length check (100 chars)
- **Line 329**: Minimum confidence check (0.1)
- **Line 305**: Session consolidation enabled check
- **Lines 213-293**: `storeSessionMemory()` - HTTP API call

### Configuration Structure

**File**: `~/.claude/hooks/config.json`

```json
{
  "memoryService": {
    "protocol": "auto",
    "preferredProtocol": "http",
    "http": {
      "endpoint": "https://localhost:8000",  // Must match server
      "apiKey": "your-api-key",
      "healthCheckTimeout": 3000
    },
    "enableSessionConsolidation": true
  },
  "sessionAnalysis": {
    "extractTopics": true,
    "extractDecisions": true,
    "extractInsights": true,
    "extractCodeChanges": true,
    "extractNextSteps": true,
    "minSessionLength": 100,
    "minConfidence": 0.1
  }
}
```

### Hook Settings

**File**: `~/.claude/settings.json`

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node \"/home/user/.claude/hooks/core/session-end.js\"",
            "timeout": 15  // 15 seconds (vs 10s for SessionStart)
          }
        ]
      }
    ]
  }
}
```

---

## Related Documentation

- **Hook Installation**: `claude-hooks/README.md`
- **Configuration Guide**: `claude-hooks/CONFIGURATION.md`
- **HTTP Server Management**: `docs/http-server-management.md`
- **General Troubleshooting**: `docs/troubleshooting/general.md`
- **SessionStart Windows Bug**: `claude-hooks/WINDOWS-SESSIONSTART-BUG.md`

---

## Common Questions

### Q: Why didn't my session create a memory even though I used `/exit`?

**A**: Check these conditions:
1. Conversation was too short (< 100 chars)
2. Conversation lacked decision/insight patterns (low confidence)
3. Connection to memory service failed (check endpoint)
4. Session consolidation disabled in config

### Q: Does Ctrl+C ever trigger SessionEnd?

**A**: No. Ctrl+C sends SIGINT which interrupts/suspends but doesn't terminate the session. Use `/exit` for proper termination.

### Q: Can I test if SessionEnd will work before exiting?

**A**: Yes:
```bash
node ~/.claude/hooks/core/session-end.js
```

This runs the hook with a test conversation and shows what would happen.

### Q: How do I see all my session consolidation memories?

**A**:
```bash
curl -sk "https://localhost:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["session-consolidation"]}' | \
  python -m json.tool
```

### Q: What's the difference between SessionStart and SessionEnd hooks?

**A**:
- **SessionStart**: Loads and injects memory context at session start
- **SessionEnd**: Analyzes and stores session outcomes at session end
- Both can have connection issues (check endpoint configuration)
- SessionStart has timeout issues on Windows (Ctrl+C hang bug)

---

**Last Updated**: 2025-11-01
**Applies to**: v8.15.1+
**Author**: Community Documentation
