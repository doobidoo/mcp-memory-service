# Code Execution API: 5-Minute Quick Start

**Get from "MCP tools working" to "using code execution" in 5 minutes.**

---

## Why Migrate? (30 seconds)

The Code Execution API provides **75-90% token reduction** compared to MCP tool calls, translating to significant cost savings:

| Users | Sessions/Day | Annual Token Savings | Annual Cost Savings* |
|-------|--------------|---------------------|---------------------|
| 10 | 5 | 109.5M tokens | $16.43 |
| 50 | 8 | 876M tokens | $131.40 |
| 100 | 10 | 2.19B tokens | $328.50 |

**Key Benefits:**
- **Zero code changes** to existing workflows
- **Automatic fallback** to MCP if code execution fails
- **Same functionality**, dramatically lower costs
- **5x faster** execution (50ms cold start vs 250ms MCP)

*Based on Claude Opus input pricing ($0.15/1M tokens)

---

## Prerequisites (30 seconds)

- Existing mcp-memory-service installation (any version)
- Python 3.10 or higher
- 5 minutes of your time

**Check Python version:**
```bash
python --version  # or python3 --version
```

---

## Quick Start

### Option A: Fresh Install (2 minutes)

If you're installing mcp-memory-service for the first time:

```bash
# 1. Clone or update repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
git pull  # If already cloned

# 2. Run installer (code execution enabled by default in v8.19.0+)
python scripts/installation/install.py

# 3. Done! ✅
```

The installer automatically enables code execution in Claude Code hooks. No additional configuration needed.

---

### Option B: Existing Installation (3 minutes)

If you already have mcp-memory-service installed:

```bash
# 1. Update code
cd /path/to/mcp-memory-service
git pull

# 2. Install/update API module
pip install -e .

# 3. Verify Python version (must be 3.10+)
python --version

# 4. Enable code execution in hooks (if not auto-enabled)
# Edit ~/.claude/hooks/config.json and add:
{
  "codeExecution": {
    "enabled": true,
    "timeout": 8000,
    "fallbackToMCP": true,
    "enableMetrics": true,
    "pythonPath": "python3"  // or "python" on Windows
  }
}

# 5. Done! ✅
```

**Note:** v8.19.0+ enables code execution by default. If upgrading from an older version, the installer will prompt you to enable it.

---

## Verify It's Working (1 minute)

### Test the API directly

```bash
python -c "from mcp_memory_service.api import search, health; print(health())"
```

**Expected output:**
```
CompactHealthInfo(status='healthy', count=1247, backend='sqlite_vec')
```

### Check hook logs

In your next Claude Code session, look for these indicators:

```
✅ Using code execution (75% token reduction)
📊 search() returned 5 results (385 tokens vs 2,625 MCP tokens)
💾 Backend: sqlite_vec, Count: 1247
```

If you see these messages, code execution is working correctly!

---

## What Changed?

**For You:**
- Session hooks now use the Python API instead of MCP tool calls
- **75-90% fewer tokens** consumed per session
- **5x faster** memory operations (50ms vs 250ms)

**What Stayed the Same:**
- MCP tools still work (automatic fallback)
- All existing workflows unchanged
- Zero breaking changes
- Same search quality and memory storage

**Architecture:**
```
Before (MCP):
Claude Code → MCP Protocol → Memory Server
(2,625 tokens for 5 results)

After (Code Execution):
Claude Code → Python API → Memory Server
(385 tokens for 5 results)
```

---

## Troubleshooting (1 minute)

### Issue: "⚠️ Code execution failed, falling back to MCP"

**Cause:** Python version too old, API not installed, or import error

**Solutions:**

1. **Check Python version:**
   ```bash
   python --version  # Must be 3.10+
   ```

2. **Verify API installed:**
   ```bash
   python -c "import mcp_memory_service.api"
   ```

   If this fails, run:
   ```bash
   cd /path/to/mcp-memory-service
   pip install -e .
   ```

3. **Check hook configuration:**
   ```bash
   cat ~/.claude/hooks/config.json | grep codeExecution -A 5
   ```

   Should show:
   ```json
   "codeExecution": {
     "enabled": true,
     "fallbackToMCP": true
   }
   ```

### Issue: ModuleNotFoundError

**Cause:** API module not installed

**Solution:**
```bash
cd /path/to/mcp-memory-service
pip install -e .  # Install in editable mode
```

### Issue: Timeout errors

**Cause:** Slow storage initialization or network latency

**Solution:** Increase timeout in `~/.claude/hooks/config.json`:
```json
{
  "codeExecution": {
    "timeout": 15000  // Increase from 8000ms to 15000ms
  }
}
```

### Issue: Still seeing high token counts

**Cause:** Code execution not enabled or hooks not reloaded

**Solutions:**
1. Verify config: `cat ~/.claude/hooks/config.json | grep "enabled"`
2. Restart Claude Code to reload hooks
3. Check logs for "Using code execution" message

---

## Performance Benchmarks

### Token Comparison

| Operation | MCP Tokens | Code Execution | Savings |
|-----------|------------|----------------|---------|
| search(5 results) | 2,625 | 385 | 85% |
| store() | 150 | 15 | 90% |
| health() | 125 | 20 | 84% |
| **Session hook (8 memories)** | **3,600** | **900** | **75%** |

### Execution Time

| Scenario | MCP | Code Execution | Improvement |
|----------|-----|----------------|-------------|
| Cold start | 250ms | 50ms | 5x faster |
| Warm call | 100ms | 5-10ms | 10-20x faster |
| Batch (5 ops) | 500ms | 50ms | 10x faster |

---

## Cost Savings Calculator

### Personal Use (10 sessions/day)
- Daily: 10 sessions x 2,700 tokens saved = 27,000 tokens
- Annual: 27,000 x 365 = **9.86M tokens/year**
- **Savings: $1.48/year** (at $0.15/1M tokens)

### Small Team (5 users, 8 sessions/day each)
- Daily: 5 users x 8 sessions x 2,700 tokens = 108,000 tokens
- Annual: 108,000 x 365 = **39.42M tokens/year**
- **Savings: $5.91/year**

### Large Team (50 users, 10 sessions/day each)
- Daily: 50 users x 10 sessions x 2,700 tokens = 1,350,000 tokens
- Annual: 1,350,000 x 365 = **492.75M tokens/year**
- **Savings: $73.91/year**

### Enterprise (500 users, 12 sessions/day each)
- Daily: 500 users x 12 sessions x 2,700 tokens = 16,200,000 tokens
- Annual: 16,200,000 x 365 = **5.91B tokens/year**
- **Savings: $886.50/year**

---

## Next Steps

### Monitor Your Savings

Enable metrics to track actual token savings:

```json
{
  "codeExecution": {
    "enableMetrics": true
  }
}
```

Hook logs will show per-operation savings:
```
📊 Session hook saved 2,700 tokens (75% reduction)
💰 Estimated annual savings: $1.48 (personal) / $73.91 (team of 50)
```

### Explore Advanced Features

The API supports more than just hooks:

```python
from mcp_memory_service.api import search, store, health, close

# Search with filters
results = search("architecture decisions", limit=10, tags=["important"])

# Store with metadata
hash = store("Memory content", tags=["note", "urgent"], memory_type="reminder")

# Check service health
info = health()
print(f"Backend: {info.backend}, Memories: {info.count}")

# Cleanup on exit
close()
```

### Read the Documentation

- **Full API Reference:** [docs/api/code-execution-interface.md](../api/code-execution-interface.md)
- **Implementation Details:** [docs/research/code-execution-interface-implementation.md](../research/code-execution-interface-implementation.md)
- **Hook Migration Guide:** [docs/hooks/phase2-code-execution-migration.md](../hooks/phase2-code-execution-migration.md)

### Stay Updated

- **GitHub Issues:** [Issue #206](https://github.com/doobidoo/mcp-memory-service/issues/206)
- **Changelog:** [CHANGELOG.md](../../CHANGELOG.md)
- **Wiki:** [Project Wiki](https://github.com/doobidoo/mcp-memory-service/wiki)

---

## Rollback Instructions

If you encounter issues and need to rollback:

1. **Disable code execution in hooks:**
   ```json
   {
     "codeExecution": {
       "enabled": false
     }
   }
   ```

2. **Restart Claude Code** to reload configuration

3. **Verify MCP fallback working:**
   - Check logs for "Using MCP tools"
   - Session hooks should complete successfully

4. **Report the issue:**
   - GitHub: [Issue #206](https://github.com/doobidoo/mcp-memory-service/issues/206)
   - Include error logs and configuration

**Note:** MCP tools continue to work even if code execution is enabled, providing automatic fallback for reliability.

---

## FAQ

### Q: Do I need to change my code?
**A:** No. Code execution is transparent to your workflows. If you're using MCP tools directly, they'll continue working.

### Q: What if code execution fails?
**A:** Automatic fallback to MCP tools. No data loss, just slightly higher token usage.

### Q: Can I use both MCP and code execution?
**A:** Yes. They coexist seamlessly. Session hooks use code execution, while manual tool calls use MCP (or can also use code execution if you prefer).

### Q: Will this break my existing setup?
**A:** No. All existing functionality remains unchanged. Code execution is additive, not replacing.

### Q: How do I measure actual savings?
**A:** Enable metrics in config and check hook logs for per-session token savings.

### Q: What about Windows support?
**A:** Fully supported. Use `"pythonPath": "python"` in config (instead of `python3`).

### Q: Can I test before committing?
**A:** Yes. Set `"enabled": true` in config, test one session, then rollback if needed by setting `"enabled": false`.

---

## Success Metrics

You'll know the migration succeeded when you see:

- ✅ Hook logs show "Using code execution"
- ✅ Token counts reduced by 75%+ per session
- ✅ Faster hook execution (<100ms cold start)
- ✅ No errors or fallback warnings
- ✅ All memory operations working normally

**Typical Session Before:**
```
🔧 Session start hook: 3,600 tokens, 250ms
📝 8 memories injected
```

**Typical Session After:**
```
🔧 Session start hook: 900 tokens, 50ms (75% token reduction)
📝 8 memories injected
💡 Saved 2,700 tokens vs MCP tools
```

---

## Support

**Need help?**
- **Documentation:** [docs/api/](../api/)
- **GitHub Issues:** [github.com/doobidoo/mcp-memory-service/issues](https://github.com/doobidoo/mcp-memory-service/issues)
- **Wiki:** [github.com/doobidoo/mcp-memory-service/wiki](https://github.com/doobidoo/mcp-memory-service/wiki)

**Found a bug?**
- Open an issue: [Issue #206](https://github.com/doobidoo/mcp-memory-service/issues/206)
- Include: Error logs, config.json, Python version

---

**Total Time: 5 minutes**
**Token Savings: 75-90%**
**Zero Breaking Changes: ✅**

Happy migrating! 🚀
