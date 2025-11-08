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
| Connection failures | HTTP/HTTPS protocol mismatch | Match endpoint in config.json to server protocol |
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
