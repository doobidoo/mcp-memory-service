-- Rollback 024: drop the forward window columns (#1506 part 2). Needs SQLite 3.35+.
ALTER TABLE secondary_forwarded DROP COLUMN acked;
ALTER TABLE secondary_forwarded DROP COLUMN started_at;
