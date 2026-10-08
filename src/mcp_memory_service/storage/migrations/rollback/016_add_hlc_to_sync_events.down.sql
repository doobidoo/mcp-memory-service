-- Rollback for 016_add_hlc_to_sync_events.sql (documentary — the MigrationRunner is forward-only).
-- The migration is additive and isolated (two new columns on sync_events + one index, zero ALTER
-- on `memories`, zero triggers), so rolling back is safe and loses no memory data: only the HLC
-- ordering metadata is discarded. Apply manually, or restore the hot-backup, to downgrade.
--
-- ORDERING MATTERS. The statements are split into two parts:
--   PART A (required, portable) runs first and completes the downgrade on EVERY SQLite version.
--   PART B (optional, SQLite >= 3.35 only) drops the now-unused columns.
-- Part B is last on purpose: `DROP COLUMN` was added in SQLite 3.35.0, so on older engines it
-- errors out. By then Part A has already deregistered the migration and rolled back the schema
-- version, so the downgrade is correct even if Part B aborts. The leftover columns are nullable
-- and ignored by Phase 1 code, so leaving them in place is harmless.

-- ===== PART A — required, portable (run on every SQLite version) =====
DROP INDEX IF EXISTS idx_sync_events_hlc;
-- last_hlc lives in the metadata key-value table; discard the clock state.
DELETE FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical');
DELETE FROM migration_registry WHERE version = 16;
UPDATE metadata SET value = '15' WHERE key = 'schema_version';

-- ===== PART B — optional, SQLite >= 3.35 only (skip on older engines) =====
-- If your SQLite is < 3.35, STOP here: the downgrade is already complete above.
ALTER TABLE sync_events DROP COLUMN hlc_logical;
ALTER TABLE sync_events DROP COLUMN hlc_physical;
