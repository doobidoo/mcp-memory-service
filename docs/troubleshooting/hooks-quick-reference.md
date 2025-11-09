# Hooks Troubleshooting Quick Reference

## SessionEnd Hook Issues

### When SessionEnd Actually Triggers

**Triggers on**:
- `/exit` command
- Terminal/window close
- Normal Claude Code exit

**Does NOT trigger on**:
- Ctrl+C (once or twice) - This suspends the session
- Session resume

### Common Issues

| Symptom | Root Cause | Solution |
|---------|-----------|----------|
| No memory after Ctrl+C | Ctrl+C suspends, doesn't end session | Use `/exit` to properly terminate |
| Connection failures during store | HTTP/HTTPS protocol mismatch | Match endpoint in config.json to server protocol (see SessionStart section) |
| No memory created despite /exit | Insufficient session content | Ensure 100+ characters and confidence > 0.1 |

### Memory Creation Requirements

1. **Minimum session length**: 100+ characters
2. **Minimum confidence**: > 0.1 from conversation analysis
3. **Session consolidation enabled**: `enableSessionConsolidation: true` in config

### Quick Verification

```bash
# Check recent session memories
curl -sk "https://localhost:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["session-consolidation"], "limit": 5}' | \
  python -m json.tool | grep created_at_iso

# Test SessionEnd hook manually
node ~/.claude/hooks/core/session-end.js

# Verify connection
curl -sk "https://localhost:8000/api/health"
```

## SessionStart Hook Issues

### No Relevant Memories Found / MCP Fallback

**Symptoms**:
- Session starts with multiple "MCP Fallback" messages (typically 3x)
- Message: "📭 Memory Search → No relevant memories found"
- Git analysis works but no memories are injected
- Hook appears to work but provides no memory context

**Example Output**:
```
 ▐▛███▜▌   Claude Code v2.0.36
▝▜█████▛▘  Sonnet 4.5 · Claude Max
  ▘▘ ▝▝    /home/hkr/Repositories/mcp-memory-service
  ⎿  SessionStart:startup hook succeeded: 🧠 Memory Hook → Initializing session awareness...
     📂 Project Detector → Analyzing mcp-memory-service
     📊 Detection Result → mcp-memory-service (Documentation) • 100%
     📂 Project Detector → Analyzing mcp-memory-service (Documentation)
     🔗 Connection → Using HTTP protocol
     💾 Storage → 🪶 sqlite-vec (Connected) • 2529 memories • 8.51MB
     📍 Path → Unknown location
     📊 Git Analysis → Analyzing repository context...
     [Git Analyzer] Analyzed 20 commits, 3 changelog entries
     [Git Analyzer] Keywords: docs, memory, http, chore, version, v8.22.0, fix, session, v8.21.0, feat, release, v8.20.1, v8.20.0, v8.19.1, mcp, v8.19.0, api, fixed, added, bug
     📊 Git Context → 10 commits, 3 changelog entries
     🔑 Keywords → docs, memory, http, chore, version
     ↩️  MCP Fallback → Using standard MCP tools
     ↩️  MCP Fallback → Using standard MCP tools
     📚 Memory Search → Found 8 relevant memories (4 recent)
     [Memory Age Analyzer] {
       avgAge: 5,
       medianAge: 9,
       p75Age: 12,
       recentPercent: 100,
       isStale: false,
       adjustments: 'Recent memory set (avg: 5d old)'
     }
     [Memory Scorer] Scoring 8 memories for project: mcp-memory-service
     [Memory Scorer] Top scored memories:
       1. Score: 0.996 - # Session Summary - mcp-memory-service
     **Project**: mcp-memo...
       2. Score: 0.996 - # Session Summary - mcp-memory-service
     **Project**: mcp-memo...
       3. Score: 0.887 - MCP Memory Service Repository Statistics - 10 Month Analysis...
     [Memory Scorer] Top memory breakdown: TimeDecay=100%, TagMatch=100%, Content=86%, Quality=84%, RecencyBonus=0%
     🎯 Scoring → Top relevance: 100%🕒 (today), 100%🕒 (today), 89% (8d ago)
     [Context Formatter] Selected 8 from 516 collected memories
     🔄 Processing → 8 memories selected
     [Context Formatter] Deduplicated 8 → 7 memories

     ┌─ 🧠 Injected Memory Context → mcp-memory-service, FastAPI, Environment Config, Python
     │
     ├─ 🪶 sqlite-vec (Connected) • 2529 memories
     ├─ 📍 Unknown location
     ├─ 📚 8 memories loaded
     │
     ├─ 🔥 Recent Work:
     │  ├─ Session Summary - mcp-memory-service Project: mcp-memory-service
     │  │        (Documentation) ## 🎯 Topics Discussed implementation
     │  │        architecture performance testing ## 🏛️ Decisions Made We
     │  │        decided to use hooks for session management and implement
     │  │        automatic context injection ## 💡 Key Insights 🕒 today
     │  ├─ bug causing incorrect memory retrieval Solution: Enhanced
     │  │        searchbytag() with optional timestart parameter across all
     │  │        backends Replaced limited parsetimequery with robust
     │  │        parsetimeexpression from utils. 🕒 today
     │  └─ Time parser tests: 14/14 PASS (100%) ## Impact Amp PR Automator:
     │        Provides lightweight, credit-efficient alternative to Gemini
     │        automation Eliminates OAuth friction in automation workflows
     │        Enables parallel quality checks for faster PR processing Bug
     │        Fix**: Restores correct tag+time filtering behavior Ensures
     │        memory hooks retrieve relevant context 🕒 today
     │
     ├─ ⚠️ Current Problems:
     │  └─ day).getDay() (local timezone throughout) Documentation: Issue
     │        #173: MCP schema caching is protocol limitation, not bug
     │        Comprehensive troubleshooting in CLAUDE. 📅 Oct 29
     │
     └─ 📋 Additional Context:
        ├─ MCP Memory Service Repository Statistics - 10 Month Analysis
        │        (Dec 2024 - Oct 2025) ## Executive Summary Total Activity: 1,536
        │        commits across 10 months (~5 commits/day average) 173 releases
        │        (~17 releases/month) 1,040,315 lines added, 729,217 removed (net
        │        +311,098) 10,583 file changes 98 GitHub issues (96% closure
        │        rate) 📅 Oct 31
        ├─ MCP and API still had different code paths ### v8.12. 📅 Oct 28
        └─ CORRECTED: DRY Refactoring Timeline v8.8. 📅 Oct 28


     ┌─ 🧠 Injected Memory Context → mcp-memory-service, FastAPI, Environment Config, Python
     │
     ├─ 🪶 sqlite-vec (Connected) • 2529 memories
     ├─ 📍 Unknown location
     ├─ 📚 8 memories loaded
     │
     ├─ 🔥 Recent Work:
     │  ├─ Session Summary - mcp-memory-service Project: mcp-memory-service
     │  │        (Documentation) ## 🎯 Topics Discussed implementation
     │  │        architecture performance testing ## 🏛️ Decisions Made We
     │  │        decided to use hooks for session management and implement
     │  │        automatic context injection ## 💡 Key Insights 🕒 today
     │  ├─ bug causing incorrect memory retrieval Solution: Enhanced
     │  │        searchbytag() with optional timestart parameter across all
     │  │        backends Replaced limited parsetimequery with robust
     │  │        parsetimeexpression from utils. 🕒 today
     │  └─ Time parser tests: 14/14 PASS (100%) ## Impact Amp PR Automator:
     │        Provides lightweight, credit-efficient alternative to Gemini
     │        automation Eliminates OAuth friction in automation workflows
     │        Enables parallel quality checks for faster PR processing Bug
     │        Fix**: Restores correct tag+time filtering behavior Ensures
     │        memory hooks retrieve relevant context 🕒 today
     │
     ├─ ⚠️ Current Problems:
     │  └─ day).getDay() (local timezone throughout) Documentation: Issue
     │        #173: MCP schema caching is protocol limitation, not bug
     │        Comprehensive troubleshooting in CLAUDE. 📅 Oct 29
     │
     └─ 📋 Additional Context:
        ├─ MCP Memory Service Repository Statistics - 10 Month Analysis
        │        (Dec 2024 - Oct 2025) ## Executive Summary Total Activity: 1,536
        │        commits across 10 months (~5 commits/day average) 173 releases
        │        (~17 releases/month) 1,040,315 lines added, 729,217 removed (net
        │        +311,098) 10,583 file changes 98 GitHub issues (96% closure
        │        rate) 📅 Oct 31
        ├─ MCP and API still had different code paths ### v8.12. 📅 Oct 28
        └─ CORRECTED: DRY Refactoring Timeline v8.8. 📅 Oct 28
```

**Root Cause**: HTTP/HTTPS protocol mismatch between hook configuration and server

**Diagnosis**:
```bash
# Check what protocol your server is using
grep HTTPS_ENABLED /path/to/mcp-memory-service/.env
# If MCP_HTTPS_ENABLED=true, server uses HTTPS

# Test HTTP connection (will fail if server uses HTTPS)
curl -s http://127.0.0.1:8000/api/health
# Empty reply = protocol mismatch

# Test HTTPS connection (will work if server uses HTTPS)
curl -sk https://127.0.0.1:8000/api/health
# {"status":"healthy",...} = server is on HTTPS

# Check hook configuration
grep endpoint ~/.claude/hooks/config.json
# Should match server protocol
```

**Solution**:

Update `~/.claude/hooks/config.json` to match your server protocol:

```json
{
  "memoryService": {
    "http": {
      "endpoint": "https://127.0.0.1:8000",  // Change http → https if server uses HTTPS
      "apiKey": "your-api-key"
    }
  }
}
```

Then restart your Claude Code session to pick up the configuration change.

**Why This Happens**:
- The `.env` file has `MCP_HTTPS_ENABLED=true`, making the server use HTTPS
- Hook config was set up for HTTP from earlier installation
- HTTP health checks fail silently, causing fallback to MCP tools
- MCP fallback path has different behavior, returning no results

### Common Issues

| Symptom | Root Cause | Solution |
|---------|-----------|----------|
| "MCP Fallback" messages (3x) | HTTP/HTTPS protocol mismatch | Update endpoint to match server protocol |
| "No relevant memories found" despite healthy DB | Connection timeout or protocol mismatch | Verify endpoint protocol and increase timeout if needed |
| Hook completes but no memory context | Code execution disabled or failed | Check `codeExecution.enabled: true` in config |
| Slow session starts (>10s) | Cold start + network delays | Normal for first start, use balanced performance profile |

### Quick Verification

```bash
# Verify server is responding on correct protocol
curl -sk "https://localhost:8000/api/health"  # For HTTPS
curl -s "http://127.0.0.1:8000/api/health"    # For HTTP

# Check database has memories
curl -sk "https://localhost:8000/api/health" | python -m json.tool
# Look for: "total_memories": 2514 (or similar non-zero value)

# Test semantic search works
curl -sk "https://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "recent development", "limit": 5}' | \
  python -m json.tool | grep -E "content|relevance"
```

## Windows SessionStart Hook Issue

**CRITICAL BUG**: SessionStart hooks cause Claude Code to hang indefinitely on Windows ([#160](https://github.com/doobidoo/mcp-memory-service/issues/160))

### Symptoms
- Claude Code unresponsive on startup
- Cannot enter prompts or cancel with Ctrl+C
- Must force-close terminal

### Workarounds

1. **Use `/session-start` slash command** (recommended)
2. **Disable SessionStart hooks** in configuration
3. **Use UserPromptSubmit hooks instead**

## Hook Configuration Synchronization

### Port Mismatch Detection

```bash
# Windows
netstat -ano | findstr "8000"

# Linux/macOS
lsof -i :8000

# Check hooks config
grep endpoint ~/.claude/hooks/config.json
```

### Common Port Mistakes

- Config.json shows 8889 but server runs on 8000
- Using dashboard port instead of API server port
- Different ports in settings.json vs hooks config

### Symptoms of Port Mismatch

- SessionStart hook hangs/times out
- Hooks show "connection timeout" in logs
- No memories injected despite hook firing

## Schema Validation Errors After PR Merges

### Quick Fix

```bash
# In Claude Code, reconnect MCP
/mcp

# For HTTP server (separate)
systemctl --user restart mcp-memory-http.service
```

### Root Cause

MCP clients cache tool schemas. After merging PRs that change schemas, you must restart the MCP server process to load the new schema.

### Verification

```bash
# Check when PR was merged
gh pr view <PR_NUMBER> --json mergedAt,title

# Check when MCP server started
ps aux | grep "memory.*server" | grep -v grep

# If server started BEFORE merge, it's running old code
```

## Emergency Debugging

```bash
# Check active MCP servers
/mcp

# Validate configuration
python scripts/validation/diagnose_backend_config.py

# Remove conflicting config
rm -f .mcp.json

# View enhanced logs (macOS)
tail -50 ~/Library/Logs/Claude/mcp-server-memory.log | grep -E "(🚀|☁️|✅|❌)"
```

## Detailed Documentation

For comprehensive troubleshooting with diagnosis checklists and technical details, see:
- `docs/troubleshooting/session-end-hooks.md`
- `docs/troubleshooting/pr162-schema-caching-issue.md`
- `docs/http-server-management.md`
