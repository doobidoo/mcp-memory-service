-- Rollback for 022_add_ontology_entries.sql (documentary, the MigrationRunner is forward-only).
-- The migration is additive and isolated (one new table, zero ALTER on `memories`, zero
-- triggers), so rolling back loses no memory data. Only the ontology overlay is discarded.
-- Memories that use an overlay type keep it verbatim (#1898). New writes with it fall back
-- to `observation` until the type is registered again.
--
-- IMPORTANT: dropping the table alone is NOT enough. The runner tracks applied migrations in
-- migration_registry. If 022 stays registered, a later startup skips it and never recreates
-- ontology_entries, so add_memory_type() fails with "no such table". The deregistration
-- statements below are REQUIRED, not optional. Run the whole file.
DROP TABLE IF EXISTS ontology_entries;
DELETE FROM metadata WHERE key = 'ontology_version';
DELETE FROM migration_registry WHERE version = 22;
UPDATE metadata SET value = '21' WHERE key = 'schema_version';
