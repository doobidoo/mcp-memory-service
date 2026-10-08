-- Rollback Delta-Sync Phase 4c push_cursor (ADR-0027)
DROP INDEX IF EXISTS idx_push_cursor_updated;
DROP TABLE IF EXISTS push_cursor;
