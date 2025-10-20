# Windows SessionStart Hook Bug - Complete Analysis

## Executive Summary

**Issue**: SessionStart hooks cause Claude Code to become completely unresponsive on Windows, requiring terminal closure to recover.

**Root Cause**: Windows-specific subprocess lifecycle management bug in Claude Code's initialization phase. The parent process doesn't properly detect subprocess completion during synchronous initialization.

**Status**: **UNFIXABLE** at hook level - requires Claude Code core fix

**GitHub Issue**: [#160](https://github.com/doobidoo/mcp-memory-service/issues/160)

**Workaround**: Disable SessionStart hooks on Windows; use UserPromptSubmit hook instead

---

## Technical Analysis

### The Deadlock Mechanism

```
┌─ Claude Code Initialization (Windows Only) ─────┐
│                                                  │
│ 1. Start initialization (synchronous)            │
│ 2. Spawn SessionStart hook subprocess           │
│ 3. **BLOCK** waiting for subprocess completion  │  ← Deadlock starts here
│ 4. Initialize event loop (NEVER REACHED)        │
│ 5. Accept user input (NEVER REACHED)            │
│                                                  │
└──────────────────────────────────────────────────┘
         ↓ subprocess spawned

┌─ SessionStart Hook Subprocess ──────────────────┐
│                                                  │
│ 1. Execute hook code                             │
│ 2. Complete and call process.exit(0)            │
│ 3. Send exit signal to parent                   │
│ 4. Subprocess terminates successfully            │
│                                                  │
└──────────────────────────────────────────────────┘
         ↓ exit signal sent

┌─ Claude Code Parent Process ────────────────────┐
│                                                  │
│ ❌ Still in synchronous initialization           │
│ ❌ NOT polling subprocess status                 │
│ ❌ Exit signal sits in queue unprocessed         │
│ ❌ Infinite wait for subprocess                  │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Why Other Hooks Work

**UserPromptSubmit** and **SessionEnd** hooks execute AFTER Claude Code has completed initialization and entered its event loop. At that point:
- Parent process is actively polling subprocess status
- Exit signals are properly detected and processed
- Subprocess completion unblocks parent immediately

### Platform Differences

| Platform | Subprocess Handling | SessionStart Works? |
|----------|---------------------|---------------------|
| macOS    | Async signals via kqueue | ✅ YES |
| Linux    | Async signals via epoll  | ✅ YES |
| Windows  | Synchronous polling required | ❌ NO |

**Windows Specific Issue**:
- Windows doesn't have Unix-style async subprocess signals
- Requires active polling of subprocess handles
- Claude Code's sync initialization doesn't poll
- Exit signals queue up but aren't processed

---

## Failed Solution Attempts

### ❌ Attempt 1: Force Process Exit
**Code**: Added `process.exit(0)` with `setTimeout` fallback
```javascript
.finally(() => {
    setTimeout(() => process.exit(0), 100);
});
```
**Result**: Hook exits successfully, but parent still hangs
**Reason**: Exit signal not being detected by blocked parent

---

### ❌ Attempt 2: Wrapper Handler
**Code**: Wrapped handler function with forced exit
```javascript
async function wrappedHandler(context) {
    try {
        await onSessionStart(context);
    } finally {
        setTimeout(() => process.exit(0), 100);
    }
}
```
**Result**: Same hang - process exits but parent doesn't detect it
**Reason**: Still a subprocess that parent must wait for

---

### ❌ Attempt 3: Batch Wrapper
**Code**: Created `.bat` wrapper
```batch
@echo off
node "%~dp0session-start.js" %*
exit /b 0
```
**Result**: Batch exits, but Node subprocess still causes hang
**Reason**: Parent waits for batch AND its children on Windows

---

### ❌ Attempt 4: Emergency Debug (Minimal Hook)
**Code**: Hook that ONLY calls `process.exit(0)` without doing ANY work
```javascript
if (require.main === module) {
    console.log('Hook disabled');
    process.exit(0); // Immediate exit, no work
}
```
**Result**: **STILL HANGS** despite doing nothing
**Critical Discovery**: Proves issue is NOT the hook logic, but the subprocess invocation itself

---

### ❌ Attempt 5: PowerShell Detachment
**Code**: PowerShell's `Start-Process` with detachment
```powershell
Start-Process -FilePath "node" `
              -ArgumentList $sessionStartScript `
              -NoNewWindow `
              -PassThru `
              -WindowStyle Hidden
exit 0
```
**Result**: PowerShell exits immediately, but Node process still blocks parent
**Reason**: Windows job objects - spawned processes inherit parent's job, creating implicit wait

---

## Why No Solution Works at Hook Level

**The Fundamental Problem**:
1. Claude Code spawns hook subprocess during synchronous initialization
2. On Windows, subprocess completion requires active polling
3. Synchronous code path doesn't poll - it blocks waiting
4. No amount of hook-side manipulation can make parent poll
5. Even zero-duration hooks cause infinite wait

**This requires a fix in Claude Code itself:**
- Move SessionStart to async initialization phase
- OR add timeout + polling during sync init
- OR skip SessionStart on Windows platform

---

## Recommended Workaround

### Solution: Enhanced UserPromptSubmit Hook

Instead of SessionStart, enhance the UserPromptSubmit hook to inject context on first prompt:

```javascript
class MidConversationHook {
    constructor(config = {}) {
        this.isFirstPrompt = true; // Track first prompt of session
        // ... rest of initialization
    }

    async analyzeMessage(userMessage, context = {}) {
        // On first prompt, always inject context (bypass pattern detection)
        if (this.isFirstPrompt) {
            this.isFirstPrompt = false;
            return {
                shouldTrigger: true,
                confidence: 1.0,
                reasoning: 'First prompt - session initialization',
                // ... rest of result
            };
        }

        // Normal pattern detection for subsequent prompts
        // ... existing logic
    }
}
```

**Benefits**:
- ✅ Zero Claude Code changes required
- ✅ Works on all platforms
- ✅ Memory context injected on first user interaction
- ✅ Only ~100ms delay (user types prompt anyway)
- ✅ Falls back to natural triggers for rest of session

**Tradeoff**:
- ⚠️ No automatic context before user types first prompt
- ⚠️ User must initiate interaction (but they would anyway)

---

## Reporting to Claude Code Team

**GitHub Issue**: [#160](https://github.com/doobidoo/mcp-memory-service/issues/160)

**Minimal Reproduction**:
```javascript
// session-start-minimal.js
console.log('Starting');
process.exit(0);
```

```json
// settings.json
{
  "hooks": {
    "SessionStart": [{
      "matchers": ["*"],
      "hooks": [{
        "type": "command",
        "command": "node session-start-minimal.js",
        "timeout": 5000
      }]
    }]
  }
}
```

**Expected**: Claude Code becomes responsive after hook exits
**Actual**: Claude Code hangs indefinitely, requires terminal closure

**Platform**: Windows 10/11, any Node.js version

---

## References

- **Main Documentation**: `CLAUDE.md` lines 587-650
- **GitHub Issue**: [#160](https://github.com/doobidoo/mcp-memory-service/issues/160)
- **Memory Tags**: `windows-subprocess-bug`, `sessionstart-hook`, `claude-code-bug`

**Date**: 2025-10-14
**Status**: Documented, workaround implemented, awaiting Claude Code fix
