-- Rollback for 015_add_sync_events.sql (documentary — the MigrationRunner is forward-only).
-- The migration is additive and isolated (a new table + indexes, zero ALTER on `memories`,
-- zero triggers), so rolling back is safe and loses no memory data: only the sync event-log
-- is discarded. Apply manually, or restore the hot-backup, to downgrade.
--
-- IMPORTANT: dropping the table alone is NOT enough. The runner tracks applied migrations in
-- migration_registry; if 015 stays registered, a later startup skips it and never recreates
-- sync_events, so the event-log hooks would write to a missing table. The deregistration
-- statements below are REQUIRED, not optional — run the whole file.
DROP INDEX IF EXISTS idx_sync_events_op;
DROP INDEX IF EXISTS idx_sync_events_seq;
DROP INDEX IF EXISTS idx_sync_events_hash;
DROP TABLE IF EXISTS sync_events;
DELETE FROM migration_registry WHERE version = 15;
UPDATE metadata SET value = '14' WHERE key = 'schema_version';
