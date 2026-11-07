-- Rollback ontology overlay (#1897)
DROP TABLE IF EXISTS ontology_entries;
DELETE FROM metadata WHERE key = 'ontology_version';
