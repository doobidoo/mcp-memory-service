-- Rollback for 016_add_hlc_to_sync_events.sql (documentary — the MigrationRunner is forward-only).
-- The migration is additive and isolated (two new columns on sync_events + one index, zero ALTER
-- on `memories`, zero triggers), so rolling back is safe and loses no memory data: only the HLC
-- ordering metadata is discarded. Apply manually, or restore the hot-backup, to downgrade.
--
-- IMPORTANT: SQLite before 3.35 cannot DROP COLUMN. On older engines, leave the columns in place
-- (they are nullable and ignored by Phase 1 code) and only deregister the migration below. The
-- deregistration statements are REQUIRED, not optional — run the whole file.
DROP INDEX IF EXISTS idx_sync_events_hlc;
-- Column drops require SQLite >= 3.35; harmless to skip on older engines.
ALTER TABLE sync_events DROP COLUMN hlc_logical;
ALTER TABLE sync_events DROP COLUMN hlc_physical;
-- last_hlc lives in the metadata key-value table; discard the clock state.
DELETE FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical');
DELETE FROM migration_registry WHERE version = 16;
UPDATE metadata SET value = '15' WHERE key = 'schema_version';
