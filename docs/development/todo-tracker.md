# TODO Tracker

**Last Updated:** 2025-11-08 10:25:25
**Scan Directory:** src
**Total TODOs:** 5

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL (P0) | 1 | Security, data corruption, blocking bugs |
| HIGH (P1) | 2 | Performance, user-facing, incomplete features |
| MEDIUM (P2) | 2 | Code quality, optimizations, technical debt |
| LOW (P3) | 0
0 | Documentation, cosmetic, nice-to-haves |

---

## CRITICAL (P0)
- `src/mcp_memory_service/web/api/analytics.py:625` - Period filtering is not implemented, leading to incorrect analytics data.

## HIGH (P1)
- `src/mcp_memory_service/storage/cloudflare.py:185` - Lack of a fallback for embedding generation makes the service vulnerable to external API failures.
- `src/mcp_memory_service/web/api/manage.py:231` - Inefficient queries can cause significant performance bottlenecks, especially with large datasets.

## MEDIUM (P2)
- `src/mcp_memory_service/web/api/documents.py:592` - Using a deprecated FastAPI event handler; should be migrated to the modern `lifespan` context manager to reduce technical debt.
- `src/mcp_memory_service/web/api/analytics.py:213` - The `storage.get_stats()` method is missing a data point, leading to API inconsistency.

## LOW (P3)
*(None in this list)*

---

## How to Address

1. **CRITICAL**: Address immediately, block releases if necessary
2. **HIGH**: Schedule for current/next sprint
3. **MEDIUM**: Add to backlog, address in refactoring sprints
4. **LOW**: Address opportunistically or when touching related code

## Updating This Tracker

Run: `bash scripts/maintenance/scan_todos.sh`
