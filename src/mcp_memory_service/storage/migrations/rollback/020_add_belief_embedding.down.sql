-- Rollback for 020_add_belief_embedding.sql (documentary — the MigrationRunner is forward-only).
-- The migration is additive and isolated (one nullable column on beliefs + one partial
-- index, zero ALTER on other tables, zero triggers), so rolling back is safe and loses no
-- belief data: only the stored embedding vectors are discarded.
--
-- ORDERING MATTERS. The statements are split into two parts:
--   PART A (required, portable) runs first and completes the downgrade on EVERY SQLite version.
--   PART B (optional, SQLite >= 3.35 only) drops the now-unused column.
-- Part B is last on purpose: `DROP COLUMN` was added in SQLite 3.35.0, so on older engines it
-- errors out. By then Part A has already deregistered the migration and rolled back the schema
-- version, so the downgrade is correct even if Part B aborts. The leftover column is nullable
-- and ignored by code that does not read it, so leaving it in place is harmless.

-- ===== PART A — required, portable (run on every SQLite version) =====
DROP INDEX IF EXISTS idx_beliefs_embedding_pending;
DELETE FROM migration_registry WHERE version = 20;
UPDATE metadata SET value = '19' WHERE key = 'schema_version';

-- ===== PART B — optional, SQLite >= 3.35 only (skip on older engines) =====
-- If your SQLite is < 3.35, STOP here: the downgrade is already complete above.
-- Re-upgrade safety: if you skip Part B and later re-apply migration 020, the runner's
-- baseline probe detects the retained embedding column and stamps v20 as applied instead
-- of re-running ADD COLUMN (which would fail duplicate-column). See migration_runner probes.
ALTER TABLE beliefs DROP COLUMN embedding;
