# TODO Tracker

**Last Updated:** 2025-11-08 08:43:00
**Scan Directory:** src/mcp_memory_service/storage
**Total TODOs:** 1

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL (P0) | 0
0 | Security, data corruption, blocking bugs |
| HIGH (P1) | 3 | Performance, user-facing, incomplete features |
| MEDIUM (P2) | 1 | Code quality, optimizations, technical debt |
| LOW (P3) | 1 | Documentation, cosmetic, nice-to-haves |

---

Loaded cached credentials.
[ERROR] [IDEClient] Failed to connect to IDE companion extension in IDE. Please ensure the extension is running. To install the extension, run /ide install.
[ERROR] Error during discovery for server 'memory-service': Connection failed for 'memory-service': fetch failed
[ERROR] Error during discovery for server 'context7': Connection failed for 'context7': SSE error: Non-200 status code (404)
## HIGH (P1)
- `src/mcp_memory_service/web/api/analytics.py:625` - Period filtering is not implemented in the analytics API, which could lead to incorrect data being returned.
- `src/mcp_memory_service/web/api/manage.py:231` - Inefficient queries for `last_used` and `memory_types` may cause performance bottlenecks.
- `src/mcp_memory_service/storage/cloudflare.py:185` - Lack of a fallback for embedding generation makes the feature unavailable if the primary service fails.

## MEDIUM (P2)
- `scripts/development/fix_sitecustomize.py:136` - A workaround for a `setuptools` issue, representing technical debt in a development script.

## LOW (P3)
- `src/mcp_memory_service/web/api/documents.py:592` - A suggestion to migrate to a newer FastAPI feature, which is a code quality improvement.
- `src/mcp_memory_service/web/api/analytics.py:213` - A minor enhancement to improve API consistency.

---

## How to Address

1. **CRITICAL**: Address immediately, block releases if necessary
2. **HIGH**: Schedule for current/next sprint
3. **MEDIUM**: Add to backlog, address in refactoring sprints
4. **LOW**: Address opportunistically or when touching related code

## Updating This Tracker

Run: `bash scripts/maintenance/scan_todos.sh`
