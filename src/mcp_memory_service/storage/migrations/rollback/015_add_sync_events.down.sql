-- Rollback for 015_add_sync_events.sql (documentary — the MigrationRunner is forward-only).
-- The migration is additive and isolated (a new table + indexes, zero ALTER on `memories`,
-- zero triggers), so rolling back is safe and loses no memory data: only the sync event-log
-- is discarded. Apply manually, or restore the hot-backup, to downgrade.
DROP INDEX IF EXISTS idx_sync_events_op;
DROP INDEX IF EXISTS idx_sync_events_seq;
DROP INDEX IF EXISTS idx_sync_events_hash;
DROP TABLE IF EXISTS sync_events;
-- Also remove the registry row and reset the schema version if your runner tracks them:
--   DELETE FROM migration_registry WHERE version = 15;
--   UPDATE metadata SET value = '14' WHERE key = 'schema_version';
