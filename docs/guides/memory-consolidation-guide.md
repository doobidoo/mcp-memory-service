# Memory Consolidation System - Operational Guide

**Version**: 8.23.0+ | **Last Updated**: 2025-11-11 | **Status**: Production Ready

## Quick Reference

### System Status Check
```bash
# Check scheduler status
curl http://127.0.0.1:8000/api/consolidation/status

# Verify HTTP server running
systemctl --user status mcp-memory-http.service
```

### Manual Trigger (HTTP API)
```bash
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" \
  -d '{"time_horizon": "weekly", "immediate": true}'
```

## Real-World Performance (v8.23.1 Test Results)

**Test Environment**: 2,495 memories, Hybrid backend (SQLite-vec + Cloudflare)

| Backend | First Run | Why |
|---------|-----------|-----|
| **SQLite-Vec** | 5-25s | Local-only, fast |
| **Cloudflare** | 2-4min | Network-dependent |
| **Hybrid** | **4-6min** | Local (~5ms) + Cloud sync (~150ms/update) |

**Key Finding**: Hybrid backend takes longer but provides multi-device data persistence - recommended for production.

## Report Generation Behavior ⚠️

**IMPORTANT**: Reports are only generated when consolidation **COMPLETES**.

### Report Location
```bash
~/.local/share/mcp-memory-service/consolidation/reports/
```

### When Reports Are Created
✅ **After successful consolidation completion**
- Directory is created automatically on first completed consolidation
- Report naming: `consolidation_{horizon}_{timestamp}.json`

❌ **NOT created when**:
- Consolidation is interrupted (killed curl, server restart)
- Consolidation fails
- Consolidation is still running
- No consolidations have run yet

### Verify Reports
```bash
# Check if any consolidations have completed
curl http://127.0.0.1:8000/api/consolidation/status | jq '.jobs_executed'

# If jobs_executed > 0, reports should exist:
ls -lh ~/.local/share/mcp-memory-service/consolidation/reports/
```

**Example**:
- `jobs_executed: 0` = No reports yet (waiting for first scheduled run)
- `jobs_executed: 3` = 3 reports should exist in directory

## Automatic Scheduling

| When | Next Run |
|------|----------|
| **Daily** 02:00 | Processes recent memories |
| **Weekly** Sun 03:00 | Pattern discovery, associations |
| **Monthly** 1st 04:00 | Long-term consolidation, archival |

### Monitor First Scheduled Run
```bash
# Watch logs after first scheduled consolidation (2025-11-12 02:00)
journalctl --user -u mcp-memory-http.service --since "2025-11-12 01:55:00" | grep consolidation

# Then check for reports
ls -lh ~/.local/share/mcp-memory-service/consolidation/reports/
```

## Three Manual Trigger Methods

### 1. HTTP API (Fastest)
```bash
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" \
  -d '{"time_horizon": "daily", "immediate": true}'
```

### 2. MCP Tools
```python
mcp__memory__memory_consolidate(action="run", time_horizon="daily", immediate=true)
```

### 3. Code Execution API (Most Token-Efficient)
```python
from mcp_memory_service.api import consolidate
consolidate('daily')  # 90% token reduction vs MCP tools
```

**Tip**: Use `daily` for faster test runs (fewer memories to process).

## Monitoring Consolidation

### Real-Time Progress
```bash
journalctl --user -u mcp-memory-http.service -f | grep consolidation
```

### Expected Log Patterns (Hybrid Backend)
```
INFO - Starting weekly consolidation...
INFO - Processing 2495 memories...
INFO - Successfully updated memory metadata: 735d2920...
INFO - HTTP Request: POST https://api.cloudflare.com/.../query "200 OK"
... (repeats for each memory)
INFO - Weekly consolidation completed in 245.3 seconds
INFO - Report saved: consolidation_weekly_2025-11-12_02-00-00.json
```

### Check Completion
```bash
# Method 1: Check jobs_executed counter
curl http://127.0.0.1:8000/api/consolidation/status | jq '.jobs_executed'

# Method 2: Check for report files
ls -lt ~/.local/share/mcp-memory-service/consolidation/reports/ | head -5
```

## Troubleshooting

### No Reports Generated

**Check 1**: Has any consolidation completed?
```bash
curl http://127.0.0.1:8000/api/consolidation/status | jq '.jobs_executed, .jobs_failed'
```

**If `jobs_executed: 0`**:
- No consolidations have completed yet
- Directory won't exist until first completion
- Wait for scheduled run or manually trigger shorter test

**If `jobs_failed > 0`**:
- Check server logs for errors:
```bash
journalctl --user -u mcp-memory-http.service | grep -i "consolidation.*error\|consolidation.*fail"
```

### Consolidation Takes Too Long

**Expected behavior with Hybrid backend**:
- First run: 4-6 minutes (2,495 memories)
- Cloudflare sync adds ~150ms per memory update
- This is normal - provides multi-device persistence

**To speed up**:
- Switch to SQLite-only backend (loses cloud sync)
- Use `daily` time horizon for testing (fewer memories)

### Test Consolidation Completion

**Quick test** (processes fewer memories):
```bash
# Trigger daily consolidation (faster)
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" \
  -d '{"time_horizon": "daily", "immediate": true}'

# Wait for completion (watch logs)
journalctl --user -u mcp-memory-http.service -f | grep "consolidation completed"

# Verify report created
ls -lh ~/.local/share/mcp-memory-service/consolidation/reports/
```

## Configuration

### Enable/Disable (.env)
```bash
MCP_CONSOLIDATION_ENABLED=true
MCP_HTTP_ENABLED=true
```

### Schedule (config.py)
```python
CONSOLIDATION_SCHEDULE = {
    'daily': '02:00',
    'weekly': 'SUN 03:00',
    'monthly': '01 04:00',
    'quarterly': 'disabled',
    'yearly': 'disabled'
}
```

## Summary

- ✅ Consolidation runs automatically (no manual intervention needed)
- ✅ Reports generated only after SUCCESSFUL completion
- ✅ Hybrid backend: 4-6 min first run (normal, provides multi-device sync)
- ✅ `jobs_executed: 0` until first consolidation completes
- ✅ Directory created automatically on first report
- ✅ Monitor scheduled runs via logs and status endpoint

---

**Related**: [Code Execution API](../api/code-execution-interface.md) | [Memory Maintenance](../maintenance/memory-maintenance.md) | [HTTP Server Management](../http-server-management.md)
